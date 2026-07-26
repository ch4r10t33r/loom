//! OpenAI-compatible HTTP API (SPEC.md "Client API", north-facing surface).
//!
//! A second request-serving surface alongside the native line-JSON RPC
//! (rpc.zig). It speaks the OpenAI schema so off-the-shelf clients (OpenWebUI,
//! Continue, aider, the OpenAI SDKs, curl) reach a Loom node with no adapter.
//! It is a thin translation over the SAME `engine.generate` path and the SAME
//! metering ledger, orthogonal to the p2p wire protocol.
//!
//! Routes: GET /v1/models, POST /v1/chat/completions, POST /v1/completions,
//! GET /health.
//!
//! Identity/metering: the client id + credit key rides `Authorization: Bearer
//! <token>` (out-of-band from the prompt, unlike the native RPC's self-asserted
//! `client` field). Usage maps onto the OpenAI `usage` object, which is already
//! the ledger's cost unit.
//!
//! STATUS: generation wired (issue #15) with SSE streaming (issue #18).
//! `POST /v1/chat/completions` and `POST /v1/completions` run the loaded model
//! over the shared `engine_lock` and return real completions with a real `usage`
//! object, metered by bearer id. `stream:true` writes an OpenAI `text/event-stream`
//! (one `data:` chunk per token, then `data: [DONE]`) over both the loom and
//! distributed-GGUF engines. Chat `messages[]` are rendered with the model's
//! detected chat template (chat_template.zig).

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const generator = @import("generator.zig");
const sockopt = @import("../core/sockopt.zig");
const chat_template = @import("../gguf/chat_template.zig");
const meter_mod = @import("meter.zig");

pub const Ctx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    gen: *generator.Generator,
    addr: []const u8,
    port: u16,
    seed: u64,
    /// Advertised model id: echoed in `/v1/models` and each response `model`
    /// field. v1 uses the CLI model spec; a later cut maps this to the manifest
    /// version so a wrong-model request is refused (SPEC.md version pinning).
    model_id: []const u8,
    /// Shared with rpc.zig: the engine holds mutable per-request state, so real
    /// generation across BOTH surfaces must serialize on this one mutex. The
    /// skeleton does not generate, so it does not lock yet.
    engine_lock: *Io.Mutex,
    /// Per-client ledger. When set, the bearer token is the client id; usage is
    /// charged here (TODO in the skeleton).
    meter: ?*meter_mod.Meter = null,
};

const Conn = struct { ctx: *Ctx, stream: net.Stream };

/// Bound concurrent HTTP handlers (mirrors rpc.zig's thread-per-accept cap).
const MAX_CONNS: u32 = 128;
var live_conns: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);

/// Monotonic completion-id counter (no time/rand dependency).
var id_counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);

pub fn serve(ctx: *Ctx) !void {
    var address = try net.IpAddress.parse(ctx.addr, ctx.port);
    var server = try address.listen(ctx.io, .{ .reuse_address = true });
    sockopt.ensureReaper(ctx.io);
    defer server.deinit(ctx.io);

    while (true) {
        const stream = server.accept(ctx.io) catch continue;
        if (live_conns.fetchAdd(1, .monotonic) >= MAX_CONNS) {
            _ = live_conns.fetchSub(1, .monotonic);
            stream.close(ctx.io);
            continue;
        }
        const conn = ctx.gpa.create(Conn) catch {
            _ = live_conns.fetchSub(1, .monotonic);
            stream.close(ctx.io);
            continue;
        };
        conn.* = .{ .ctx = ctx, .stream = stream };
        const t = std.Thread.spawn(.{}, connThread, .{conn}) catch {
            _ = live_conns.fetchSub(1, .monotonic);
            stream.close(ctx.io);
            ctx.gpa.destroy(conn);
            continue;
        };
        t.detach();
    }
}

fn connThread(conn: *Conn) void {
    handleConn(conn.ctx, conn.stream) catch {};
    conn.ctx.gpa.destroy(conn);
    _ = live_conns.fetchSub(1, .monotonic);
}

/// One HTTP/1.1 request per connection (Connection: close). Minimal hand-rolled
/// parse in the house style (rpc.zig): request line, headers, Content-Length
/// body. Sufficient for the OpenAI request shape; keep-alive is a later concern.
fn handleConn(ctx: *Ctx, stream: net.Stream) !void {
    defer stream.close(ctx.io);
    const dl = sockopt.trackServe(ctx.io, stream);
    defer sockopt.untrack(ctx.io, dl);
    var rbuf: [1 << 16]u8 = undefined;
    var wbuf: [1 << 16]u8 = undefined;
    var r = stream.reader(ctx.io, &rbuf);
    var w = stream.writer(ctx.io, &wbuf);
    const ri = &r.interface;
    const wi = &w.interface;

    const req = parseRequest(ctx.gpa, ri) catch {
        try writeHttp(wi, 400, errorJson(ctx.gpa, "malformed_request", "invalid_request_error", "not_found") catch return);
        return;
    };
    defer req.deinit(ctx.gpa);

    // A streaming handler writes the SSE response to `wi` itself and returns
    // null; otherwise we get a buffered Response to write here.
    if (try route(ctx, req, wi)) |resp| {
        defer if (resp.owned) ctx.gpa.free(resp.body);
        try writeHttp(wi, resp.status, resp.body);
    }
}

// ---- request parsing -------------------------------------------------------

const Request = struct {
    method: []u8,
    path: []u8,
    /// Bearer token from Authorization, or "" if absent.
    bearer: []u8,
    body: []u8,

    fn deinit(self: Request, gpa: std.mem.Allocator) void {
        gpa.free(self.method);
        gpa.free(self.path);
        gpa.free(self.bearer);
        gpa.free(self.body);
    }
};

fn parseRequest(gpa: std.mem.Allocator, ri: *Io.Reader) !Request {
    // request line: "METHOD PATH HTTP/1.1"
    const line0 = trimCrlf(try ri.takeDelimiterInclusive('\n'));
    var it = std.mem.splitScalar(u8, line0, ' ');
    const method = it.next() orelse return error.BadRequest;
    const path = it.next() orelse return error.BadRequest;
    const method_owned = try gpa.dupe(u8, method);
    errdefer gpa.free(method_owned);
    const path_owned = try gpa.dupe(u8, path);
    errdefer gpa.free(path_owned);

    var content_length: usize = 0;
    var bearer: []u8 = try gpa.dupe(u8, "");
    errdefer gpa.free(bearer);

    // headers until a blank line
    while (true) {
        const h = trimCrlf(try ri.takeDelimiterInclusive('\n'));
        if (h.len == 0) break;
        if (asciiHeaderIs(h, "content-length")) {
            content_length = std.fmt.parseInt(usize, std.mem.trim(u8, headerValue(h), " "), 10) catch 0;
        } else if (asciiHeaderIs(h, "authorization")) {
            const v = std.mem.trim(u8, headerValue(h), " ");
            const prefix = "Bearer ";
            if (v.len > prefix.len and std.ascii.eqlIgnoreCase(v[0..prefix.len], prefix)) {
                gpa.free(bearer);
                bearer = try gpa.dupe(u8, std.mem.trim(u8, v[prefix.len..], " "));
            }
        }
    }

    // cap body to guard against oversized posts (skeleton bound)
    if (content_length > (8 << 20)) return error.BodyTooLarge;
    const body = try gpa.alloc(u8, content_length);
    errdefer gpa.free(body);
    if (content_length > 0) try ri.readSliceAll(body);

    return .{ .method = method_owned, .path = path_owned, .bearer = bearer, .body = body };
}

fn trimCrlf(s: []const u8) []const u8 {
    return std.mem.trimEnd(u8, s, "\r\n");
}

/// Case-insensitive check that header line `h` has field name `name`.
fn asciiHeaderIs(h: []const u8, name: []const u8) bool {
    const colon = std.mem.indexOfScalar(u8, h, ':') orelse return false;
    return std.ascii.eqlIgnoreCase(std.mem.trim(u8, h[0..colon], " "), name);
}

fn headerValue(h: []const u8) []const u8 {
    const colon = std.mem.indexOfScalar(u8, h, ':') orelse return "";
    return h[colon + 1 ..];
}

// ---- routing ---------------------------------------------------------------

/// `owned` distinguishes an allocated body from a static one (security issue
/// #23): `fallback()` returns a string literal, and freeing that is an invalid
/// free whose effect depends on the allocator.
const Response = struct { status: u16, body: []u8, owned: bool = true };

/// Returns a buffered Response to write, or null if the handler already wrote
/// the response to `wi` (streaming).
fn route(ctx: *Ctx, req: Request, wi: *Io.Writer) !?Response {
    const gpa = ctx.gpa;
    const path = stripQuery(req.path);

    if (eql(req.method, "GET") and eql(path, "/health")) {
        return ok(gpa, "{\"status\":\"ok\"}") catch fallback();
    }
    if (eql(req.method, "GET") and eql(path, "/v1/models")) {
        return handleModels(ctx) catch fallback();
    }
    if (eql(req.method, "POST") and (eql(path, "/v1/chat/completions") or eql(path, "/v1/completions"))) {
        return handleCompletions(ctx, req, eql(path, "/v1/chat/completions"), wi) catch fallback();
    }

    const b = errorJson(gpa, "unknown route", "invalid_request_error", "not_found") catch return fallback();
    return .{ .status = 404, .body = b };
}

fn handleModels(ctx: *Ctx) !Response {
    const body = try std.fmt.allocPrint(ctx.gpa,
        \\{{"object":"list","data":[{{"id":"{s}","object":"model","created":0,"owned_by":"loom"}}]}}
    , .{ctx.model_id});
    return .{ .status = 200, .body = body };
}

fn handleCompletions(ctx: *Ctx, req: Request, is_chat: bool, wi: *Io.Writer) !?Response {
    const gpa = ctx.gpa;

    // Identity from the bearer token (SPEC.md: out-of-band, not prompt-forgeable).
    const client: []const u8 = if (req.bearer.len > 0) req.bearer else "anon";

    // Metering gate: refuse an exhausted client before any work.
    if (ctx.meter) |m| {
        if (m.remaining(client) == 0) {
            const b = try errorJson(gpa, "insufficient balance", "insufficient_quota", "payment_required");
            return .{ .status = 402, .body = b };
        }
    }

    // Parse the body once for model / stream / params / prompt.
    var model_id: []const u8 = ctx.model_id;
    var stream = false;
    var max_tokens: usize = 128;
    var temp: f32 = 0.0;
    var seed: u64 = ctx.seed;
    var parsed: ?std.json.Parsed(std.json.Value) = std.json.parseFromSlice(std.json.Value, gpa, req.body, .{}) catch null;
    defer if (parsed) |*p| p.deinit();
    if (parsed) |p| if (p.value == .object) {
        const o = p.value.object;
        if (o.get("model")) |v| if (v == .string) {
            model_id = v.string;
        };
        if (o.get("stream")) |v| if (v == .bool) {
            stream = v.bool;
        };
        if (o.get("max_tokens")) |v| if (v == .integer and v.integer > 0) {
            max_tokens = @intCast(v.integer);
        };
        if (o.get("temperature")) |v| switch (v) {
            .float => temp = @floatCast(v.float),
            .integer => temp = @floatFromInt(v.integer),
            else => {},
        };
        if (o.get("seed")) |v| if (v == .integer and v.integer >= 0) {
            seed = @intCast(v.integer);
        };
    };

    // Assemble the prompt: chat messages -> role-labeled text, or the raw
    // `prompt` field for the completions endpoint. Shared by both paths.
    const prompt_text = if (is_chat)
        try assembleChatPrompt(ctx, gpa, parsed)
    else
        try gpa.dupe(u8, promptField(parsed));
    defer gpa.free(prompt_text);

    // Metering budget clamps completion length to the remaining allowance.
    const budget: ?u64 = if (ctx.meter) |m| m.remaining(client) else null;

    // Streaming: write the SSE response to the connection and return null.
    if (stream) {
        streamCompletions(ctx, wi, is_chat, model_id, client, prompt_text, max_tokens, temp, seed, budget);
        return null;
    }

    // Parse special tokens only for chat formats whose scaffold uses them
    // (chatml/llama3/gemma); never for a raw completion prompt or a text-marker
    // chat format, so user content cannot inject control tokens.
    const parse_special = is_chat and chat_template.usesSpecialMarkers(ctx.gen.chatFormat());

    // Generate under the shared engine mutex (rpc + openai serialize here).
    ctx.engine_lock.lockUncancelable(ctx.io);
    var res = ctx.gen.generate(gpa, ctx.io, prompt_text, max_tokens, temp, seed, budget, null, parse_special) catch |e| {
        ctx.engine_lock.unlock(ctx.io);
        return e;
    };
    ctx.engine_lock.unlock(ctx.io);
    defer res.deinit(gpa);

    const content = try jsonEscapeAlloc(gpa, res.text);
    defer gpa.free(content);

    // Charge the ledger: prompt tokens processed + tokens generated.
    if (ctx.meter) |m| {
        _ = try m.charge(client, @intCast(res.prompt_tokens + res.completion_tokens));
    }

    const finish: []const u8 = if (res.stop) "stop" else "length";
    const id = id_counter.fetchAdd(1, .monotonic) + 1;
    const pt = res.prompt_tokens;
    const n = res.completion_tokens;

    const body = if (is_chat)
        try std.fmt.allocPrint(gpa,
            \\{{"id":"chatcmpl-loom-{d}","object":"chat.completion","created":0,"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
        , .{ id, model_id, content, finish, pt, n, pt + n })
    else
        try std.fmt.allocPrint(gpa,
            \\{{"id":"cmpl-loom-{d}","object":"text_completion","created":0,"model":"{s}","choices":[{{"index":0,"text":"{s}","finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
        , .{ id, model_id, content, finish, pt, n, pt + n });

    return .{ .status = 200, .body = body };
}

/// SSE streaming sink: emits one `chat.completion.chunk` / `text_completion`
/// event per token as it is produced. A write error (client disconnected)
/// propagates out and aborts generation.
const StreamCtx = struct {
    wi: *Io.Writer,
    gpa: std.mem.Allocator,
    id: u64,
    model: []const u8,
    is_chat: bool,

    fn emit(ptr: *anyopaque, bytes: []const u8) anyerror!void {
        const self: *StreamCtx = @ptrCast(@alignCast(ptr));
        const esc = try jsonEscapeAlloc(self.gpa, bytes);
        defer self.gpa.free(esc);
        if (self.is_chat) {
            try self.wi.print("data: {{\"id\":\"chatcmpl-loom-{d}\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{s}\"}},\"finish_reason\":null}}]}}\n\n", .{ self.id, self.model, esc });
        } else {
            try self.wi.print("data: {{\"id\":\"cmpl-loom-{d}\",\"object\":\"text_completion\",\"created\":0,\"model\":\"{s}\",\"choices\":[{{\"index\":0,\"text\":\"{s}\",\"finish_reason\":null}}]}}\n\n", .{ self.id, self.model, esc });
        }
        try self.wi.flush();
    }
};

/// Write an OpenAI SSE stream: `text/event-stream` head, one delta event per
/// token, a final finish_reason event, then `[DONE]`. All connection writes are
/// best-effort (a dead client just ends the stream); metering is still charged.
fn streamCompletions(
    ctx: *Ctx,
    wi: *Io.Writer,
    is_chat: bool,
    model_id: []const u8,
    client: []const u8,
    prompt_text: []const u8,
    max_tokens: usize,
    temp: f32,
    seed: u64,
    budget: ?u64,
) void {
    const gpa = ctx.gpa;
    const id = id_counter.fetchAdd(1, .monotonic) + 1;

    wi.print("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", .{}) catch return;
    if (is_chat) {
        wi.print("data: {{\"id\":\"chatcmpl-loom-{d}\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}\n\n", .{ id, model_id }) catch return;
    }
    wi.flush() catch return;

    var sctx = StreamCtx{ .wi = wi, .gpa = gpa, .id = id, .model = model_id, .is_chat = is_chat };
    const sink = generator.TokenSink{ .ctx = &sctx, .emit = StreamCtx.emit };

    const parse_special = is_chat and chat_template.usesSpecialMarkers(ctx.gen.chatFormat());
    ctx.engine_lock.lockUncancelable(ctx.io);
    var res = ctx.gen.generate(gpa, ctx.io, prompt_text, max_tokens, temp, seed, budget, sink, parse_special) catch |e| {
        ctx.engine_lock.unlock(ctx.io);
        wi.print("data: {{\"error\":{{\"message\":\"{s}\",\"type\":\"server_error\"}}}}\n\ndata: [DONE]\n\n", .{@errorName(e)}) catch {};
        wi.flush() catch {};
        return;
    };
    ctx.engine_lock.unlock(ctx.io);
    defer res.deinit(gpa);

    if (ctx.meter) |m| {
        _ = m.charge(client, @intCast(res.prompt_tokens + res.completion_tokens)) catch {};
    }

    const finish: []const u8 = if (res.stop) "stop" else "length";
    if (is_chat) {
        wi.print("data: {{\"id\":\"chatcmpl-loom-{d}\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{s}\"}}]}}\n\n", .{ id, model_id, finish }) catch return;
    } else {
        wi.print("data: {{\"id\":\"cmpl-loom-{d}\",\"object\":\"text_completion\",\"created\":0,\"model\":\"{s}\",\"choices\":[{{\"index\":0,\"text\":\"\",\"finish_reason\":\"{s}\"}}]}}\n\n", .{ id, model_id, finish }) catch return;
    }
    wi.print("data: [DONE]\n\n", .{}) catch return;
    wi.flush() catch return;
}

/// Render OpenAI chat `messages[]` into the model's expected prompt, using the
/// engine's detected chat format (chat_template.zig).
fn assembleChatPrompt(ctx: *Ctx, gpa: std.mem.Allocator, parsed: ?std.json.Parsed(std.json.Value)) ![]u8 {
    var msgs = std.ArrayList(chat_template.Message).empty;
    defer msgs.deinit(gpa);
    if (parsed) |p| if (p.value == .object) {
        if (p.value.object.get("messages")) |mv| if (mv == .array) {
            for (mv.array.items) |el| {
                if (el != .object) continue;
                const role = switch (el.object.get("role") orelse .null) {
                    .string => |s| s,
                    else => "user",
                };
                const content = switch (el.object.get("content") orelse .null) {
                    .string => |s| s,
                    else => continue, // array/structured content: unsupported in v1
                };
                try msgs.append(gpa, .{ .role = role, .content = content });
            }
        };
    };
    return chat_template.render(gpa, ctx.gen.chatFormat(), msgs.items, true);
}

fn promptField(parsed: ?std.json.Parsed(std.json.Value)) []const u8 {
    if (parsed) |p| if (p.value == .object) {
        if (p.value.object.get("prompt")) |v| if (v == .string) return v.string;
    };
    return "";
}

/// JSON-escape a raw byte slice into an owned string (no surrounding quotes).
fn jsonEscapeAlloc(gpa: std.mem.Allocator, bytes: []const u8) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(gpa);
    for (bytes) |b| switch (b) {
        '"' => try buf.appendSlice(gpa, "\\\""),
        '\\' => try buf.appendSlice(gpa, "\\\\"),
        '\n' => try buf.appendSlice(gpa, "\\n"),
        '\r' => try buf.appendSlice(gpa, "\\r"),
        '\t' => try buf.appendSlice(gpa, "\\t"),
        0x20...0x21, 0x23...0x5b, 0x5d...0x7e => try buf.append(gpa, b),
        else => {
            var tmp: [6]u8 = undefined;
            try buf.appendSlice(gpa, try std.fmt.bufPrint(&tmp, "\\u{x:0>4}", .{b}));
        },
    };
    return buf.toOwnedSlice(gpa);
}

// ---- helpers ---------------------------------------------------------------

fn stripQuery(path: []const u8) []const u8 {
    const q = std.mem.indexOfScalar(u8, path, '?') orelse return path;
    return path[0..q];
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn ok(gpa: std.mem.Allocator, comptime json: []const u8) !Response {
    return .{ .status = 200, .body = try gpa.dupe(u8, json) };
}

/// Last-resort body when even allocation fails; static, never freed via gpa.
fn fallback() Response {
    return .{
        .status = 500,
        .body = @constCast("{\"error\":{\"message\":\"internal\",\"type\":\"server_error\"}}"),
        .owned = false, // static storage: never freed
    };
}

fn errorJson(gpa: std.mem.Allocator, msg: []const u8, ty: []const u8, code: []const u8) ![]u8 {
    return std.fmt.allocPrint(gpa,
        \\{{"error":{{"message":"{s}","type":"{s}","code":"{s}"}}}}
    , .{ msg, ty, code });
}

fn writeHttp(wi: *Io.Writer, status: u16, body: []const u8) !void {
    const reason = switch (status) {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        402 => "Payment Required",
        404 => "Not Found",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        else => "OK",
    };
    try wi.print("HTTP/1.1 {d} {s}\r\n", .{ status, reason });
    try wi.print("Content-Type: application/json\r\n", .{});
    try wi.print("Content-Length: {d}\r\n", .{body.len});
    try wi.print("Access-Control-Allow-Origin: *\r\n", .{});
    try wi.print("Connection: close\r\n\r\n", .{});
    try wi.print("{s}", .{body});
    try wi.flush();
}
