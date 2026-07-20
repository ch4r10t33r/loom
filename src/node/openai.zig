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
//! v1 STATUS: SKELETON. Transport, routing, request/response shapes, and bearer
//! identity are implemented; the responses are well-formed OpenAI envelopes with
//! a placeholder completion (`loom_status:"skeleton"`). The following are TODO,
//! called out at their call sites:
//!   - render `messages[]` -> prompt via the model chat template, then
//!     `engine.generate` (serialized on the shared `engine_lock`);
//!   - charge the meter from real prompt+completion token counts;
//!   - SSE streaming for `stream:true` (`data:` chunks + `data: [DONE]`).

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const engine_mod = @import("../engine/engine.zig");
const Engine = engine_mod.Engine;
const meter_mod = @import("meter.zig");

pub const Ctx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    engine: *Engine,
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

    const resp = route(ctx, req);
    defer ctx.gpa.free(resp.body);
    try writeHttp(wi, resp.status, resp.body);
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

const Response = struct { status: u16, body: []u8 };

fn route(ctx: *Ctx, req: Request) Response {
    const gpa = ctx.gpa;
    const path = stripQuery(req.path);

    if (eql(req.method, "GET") and eql(path, "/health")) {
        return ok(gpa, "{\"status\":\"ok\"}") catch fallback();
    }
    if (eql(req.method, "GET") and eql(path, "/v1/models")) {
        return handleModels(ctx) catch fallback();
    }
    if (eql(req.method, "POST") and (eql(path, "/v1/chat/completions") or eql(path, "/v1/completions"))) {
        return handleCompletions(ctx, req, eql(path, "/v1/chat/completions")) catch fallback();
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

fn handleCompletions(ctx: *Ctx, req: Request, is_chat: bool) !Response {
    const gpa = ctx.gpa;

    // Identity from the bearer token (SPEC.md: out-of-band, not prompt-forgeable).
    const client: []const u8 = if (req.bearer.len > 0) req.bearer else "anon";

    // Metering gate: refuse an exhausted client before any work. In the
    // skeleton nothing is charged, so this only trips a pre-credited-to-zero
    // client; it wires the integration point.
    if (ctx.meter) |m| {
        if (m.remaining(client) == 0) {
            const b = try errorJson(gpa, "insufficient balance", "insufficient_quota", "payment_required");
            return .{ .status = 402, .body = b };
        }
    }

    // Model override from the request body (best-effort; defaults to ours).
    var model_id: []const u8 = ctx.model_id;
    var stream = false;
    var parsed: ?std.json.Parsed(std.json.Value) = std.json.parseFromSlice(std.json.Value, gpa, req.body, .{}) catch null;
    defer if (parsed) |*p| p.deinit();
    if (parsed) |p| if (p.value == .object) {
        if (p.value.object.get("model")) |mv| if (mv == .string) {
            model_id = mv.string;
        };
        if (p.value.object.get("stream")) |sv| if (sv == .bool) {
            stream = sv.bool;
        };
    };

    // TODO(stream): SSE streaming will emit `data:` chunks + `data: [DONE]`.
    // Until then a streaming client is told plainly rather than left hanging.
    if (stream) {
        const b = try errorJson(gpa, "streaming not yet implemented (skeleton)", "invalid_request_error", "not_implemented");
        return .{ .status = 501, .body = b };
    }

    // TODO(generate): render messages[]/prompt via the model chat template,
    // then serialize on ctx.engine_lock and call ctx.engine.generate(...); map
    // token counts onto the usage object and charge ctx.meter. Until then, a
    // well-formed OpenAI envelope with a placeholder completion.
    const id = id_counter.fetchAdd(1, .monotonic) + 1;
    const placeholder = "loom OpenAI-compatible endpoint is a skeleton: generation is not yet wired to the engine.";

    const body = if (is_chat)
        try std.fmt.allocPrint(gpa,
            \\{{"id":"chatcmpl-loom-{d}","object":"chat.completion","created":0,"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}},"loom_status":"skeleton"}}
        , .{ id, model_id, placeholder })
    else
        try std.fmt.allocPrint(gpa,
            \\{{"id":"cmpl-loom-{d}","object":"text_completion","created":0,"model":"{s}","choices":[{{"index":0,"text":"{s}","finish_reason":"stop"}}],"usage":{{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}},"loom_status":"skeleton"}}
        , .{ id, model_id, placeholder });

    return .{ .status = 200, .body = body };
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
    return .{ .status = 500, .body = @constCast("{\"error\":{\"message\":\"internal\",\"type\":\"server_error\"}}") };
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
