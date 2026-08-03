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
const rag_store = @import("../rag/store.zig");
const sockopt = @import("../core/sockopt.zig");
const stats = @import("../core/stats.zig");
const weights = @import("../p2p/weights.zig");
const sync = @import("../p2p/sync.zig");
const chat_template = @import("../gguf/chat_template.zig");
const meter_mod = @import("meter.zig");
const peers_mod = @import("../p2p/peers.zig");

/// The bundled chat UI, compiled into the binary.
const chat_html = @embedFile("ui.html");

pub const Ctx = struct {
    /// Opt-in alpha telemetry aggregate (numeric only; docs/ALPHA.md).
    alpha_metrics: ?*@import("alpha.zig").Metrics = null,
    /// Node console, shared with the status thread; arrival + completion
    /// lines per request (see rpc.zig).
    console: ?*Io.Writer = null,
    console_lock: ?*Io.Mutex = null,
    /// Delegate-while-cold: when this node's holdings fraction is below the
    /// threshold and a peer at >=0.9 exists, forward the generation over the
    /// p2p GEN command and relay the answer (badged in the response). 0
    /// disables. Local generation is always the fallback.
    store: ?*weights.Store = null,
    delegate_below: f64 = 0.5,
    /// Gossiped RAG chunks; null when --rag is off. Search runs before the
    /// prompt reaches the engine.
    rag: ?*rag_store.Store = null,
    rag_k: usize = 3,
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
    /// Serve the bundled chat UI at `GET /`. Set on the second listener the
    /// node runs for `--ui-port`; the API listener leaves it off so the API
    /// surface stays pure JSON.
    serve_ui: bool = false,
    /// The node's live peer table, read for `/health` so the UI header can show
    /// the peer count. Null when this process has no p2p layer (a light node,
    /// or `--p2p-port 0`).
    peers: ?*peers_mod.Table = null,
    /// True when the served weights are one of loom's random-weight fixtures.
    /// Reported by `/health` so the UI can say so: a fixture answers with
    /// meaningless text by construction, and without a warning that reads as a
    /// broken model rather than as a model that was never trained.
    synthetic: bool = false,
};

const Conn = struct { ctx: *Ctx, stream: net.Stream };

/// Bound concurrent HTTP handlers (mirrors rpc.zig's thread-per-accept cap).
const MAX_CONNS: u32 = 128;
var live_conns: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);

/// Server-side ceiling on what one request may reserve/generate, independent of
/// the client's balance (security issue #30 / #33): a self-asserted bearer id
/// can be rotated freely, so the meter alone is not a bound on how long one
/// request may hold the engine mutex.
const MAX_RESERVE: u64 = 4096;

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
        // static body: the allocated variant leaked on every malformed request
        try writeHttp(wi, 400, "{\"error\":{\"message\":\"malformed_request\",\"type\":\"invalid_request_error\"}}", .json);
        return;
    };
    defer req.deinit(ctx.gpa);

    // A streaming handler writes the SSE response to `wi` itself and returns
    // null; otherwise we get a buffered Response to write here.
    if (try route(ctx, req, wi, dl)) |resp| {
        defer if (resp.owned) ctx.gpa.free(resp.body);
        try writeHttp(wi, resp.status, resp.body, resp.content_type);
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
const ContentType = enum {
    json,
    html,

    fn mime(self: ContentType) []const u8 {
        return switch (self) {
            .json => "application/json",
            .html => "text/html; charset=utf-8",
        };
    }
};

const Response = struct { status: u16, body: []u8, owned: bool = true, content_type: ContentType = .json };

/// Returns a buffered Response to write, or null if the handler already wrote
/// the response to `wi` (streaming).
fn route(ctx: *Ctx, req: Request, wi: *Io.Writer, dl: ?usize) !?Response {
    const gpa = ctx.gpa;
    const path = stripQuery(req.path);

    if (ctx.serve_ui and eql(req.method, "GET") and (eql(path, "/") or eql(path, "/index.html"))) {
        // Static, embedded at compile time: the binary stays self-contained,
        // which is the point of shipping one executable.
        return .{ .status = 200, .body = @constCast(chat_html), .owned = false, .content_type = .html };
    }
    if (eql(req.method, "GET") and eql(path, "/health")) {
        return handleHealth(ctx) catch fallback();
    }
    if (eql(req.method, "GET") and eql(path, "/v1/models")) {
        return handleModels(ctx) catch fallback();
    }
    if (eql(req.method, "POST") and eql(path, "/v1/rag/chunks")) return handleRagIngest(ctx, req);
    if (eql(req.method, "POST") and (eql(path, "/v1/chat/completions") or eql(path, "/v1/completions"))) {
        return handleCompletions(ctx, req, eql(path, "/v1/chat/completions"), wi, dl) catch fallback();
    }

    const b = errorJson(gpa, "unknown route", "invalid_request_error", "not_found") catch return fallback();
    return .{ .status = 404, .body = b };
}

/// Liveness plus the two numbers worth watching from the UI header: how many
/// peers are reachable, and what fraction of expert reads were served locally.
fn handleHealth(ctx: *Ctx) !Response {
    const n_peers: usize = if (ctx.peers) |p| p.liveCount() else 0;
    const model_esc = try jsonEscapeAlloc(ctx.gpa, ctx.model_id);
    defer ctx.gpa.free(model_esc);
    var generating_s: i64 = -1;
    if (ctx.alpha_metrics) |am| {
        if (am.inflightSecs(stats.nowNs(ctx.io))) |secs| generating_s = @intCast(secs);
    }
    const body = try std.fmt.allocPrint(
        ctx.gpa,
        "{{\"status\":\"ok\",\"model\":\"{s}\",\"peers\":{d},\"hit_rate\":{d:.4},\"synthetic\":{},\"generating_s\":{d}}}",
        .{ model_esc, n_peers, ctx.gen.hitRate(), ctx.synthetic, generating_s },
    );
    return .{ .status = 200, .body = body };
}

/// POST /v1/rag/chunks {"chunks":["...","..."]} -- ingest text into the
/// local store. Accepted chunks reach every peer on this network through
/// the next gossip round; the response reports what was new here.
fn handleRagIngest(ctx: *Ctx, req: Request) !?Response {
    const gpa = ctx.gpa;
    const st = ctx.rag orelse return .{ .status = 503, .body = try gpa.dupe(u8, "{\"error\":\"rag disabled\"}") };
    var parsed = std.json.parseFromSlice(std.json.Value, gpa, req.body, .{}) catch
        return .{ .status = 400, .body = try gpa.dupe(u8, "{\"error\":\"bad json\"}") };
    defer parsed.deinit();
    var added: usize = 0;
    var seen: usize = 0;
    if (parsed.value == .object) {
        if (parsed.value.object.get("chunks")) |v| if (v == .array) {
            for (v.array.items) |item| {
                if (item != .string) continue;
                seen += 1;
                if (st.add(item.string)) added += 1;
            }
        };
    }
    const body = try std.fmt.allocPrint(gpa, "{{\"added\":{d},\"received\":{d},\"total\":{d}}}", .{ added, seen, st.count() });
    return .{ .status = 200, .body = body };
}

fn handleModels(ctx: *Ctx) !Response {
    const body = try std.fmt.allocPrint(ctx.gpa,
        \\{{"object":"list","data":[{{"id":"{s}","object":"model","created":0,"owned_by":"loom"}}]}}
    , .{ctx.model_id});
    return .{ .status = 200, .body = body };
}

/// Prepend the top-k retrieved chunks, or return `prompt` unchanged when
/// RAG is off, the store is empty, or the query will not embed.
fn ragAugment(ctx: *Ctx, gpa: std.mem.Allocator, prompt: []const u8) ![]const u8 {
    const st = ctx.rag orelse return prompt;
    if (st.count() == 0 or ctx.rag_k == 0) return prompt;
    const d = st.dim();
    if (d == 0) return prompt;
    const q = gpa.alloc(f32, d) catch return prompt;
    defer gpa.free(q);
    if (!st.embedQuery(gpa, prompt, q)) return prompt;
    var hits: [8]rag_store.Hit = undefined;
    const k = @min(ctx.rag_k, hits.len);
    const n = st.search(q, k, hits[0..k]);
    if (n == 0) return prompt;
    var b = std.ArrayList(u8).empty;
    errdefer b.deinit(gpa);
    try b.appendSlice(gpa, "Context:\n");
    for (hits[0..n]) |h| {
        const t = st.textByIndexAlloc(gpa, h.idx) orelse continue;
        defer gpa.free(t);
        try b.appendSlice(gpa, "- ");
        try b.appendSlice(gpa, t);
        try b.append(gpa, '\n');
    }
    try b.appendSlice(gpa, "\n");
    try b.appendSlice(gpa, prompt);
    return b.toOwnedSlice(gpa);
}

fn handleCompletions(ctx: *Ctx, req: Request, is_chat: bool, wi: *Io.Writer, dl: ?usize) !?Response {
    // The request is parsed, so the slowloris window is closed and the
    // connection is about to do real work. Generation on a partial node can
    // take minutes before the first token — every expert it does not hold is a
    // round trip — and the read-phase deadline would shut the socket down
    // mid-prefill.
    sockopt.refreshServe(ctx.io, dl);
    const gpa = ctx.gpa;

    // Identity from the bearer token (SPEC.md: out-of-band, not prompt-forgeable).
    const client: []const u8 = if (req.bearer.len > 0) req.bearer else "anon";

    // Metering gate is folded into the reservation below (security issue #30):
    // checking and charging separately let N concurrent requests each see the
    // full balance.

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
    const base_prompt = if (is_chat)
        try assembleChatPrompt(ctx, gpa, parsed)
    else
        try gpa.dupe(u8, promptField(parsed));
    defer gpa.free(base_prompt);

    // Retrieval: the closest gossiped chunks are prepended as context. The
    // model sees them; the metering charge counts them, because they are
    // real prompt tokens.
    const prompt_text = try ragAugment(ctx, gpa, base_prompt);
    defer if (prompt_text.ptr != base_prompt.ptr) gpa.free(prompt_text);

    // Reserve the client's allowance up front: this both gates an exhausted
    // client and debits atomically, and it means an aborted request still pays
    // for the work already done (settled from actuals below).
    var reserved: u64 = 0;
    var budget: ?u64 = null;
    if (ctx.meter) |m| {
        reserved = m.reserve(client, MAX_RESERVE) catch 0;
        if (reserved == 0) {
            const b = try errorJson(gpa, "insufficient balance", "insufficient_quota", "payment_required");
            return .{ .status = 402, .body = b };
        }
        budget = reserved;
    }

    consoleLine(ctx, "request  {s}: {d} prompt byte(s), max_tokens {d}{s}", .{
        if (is_chat) @as([]const u8, "chat") else "completion", prompt_text.len, max_tokens,
        if (stream) @as([]const u8, ", streaming") else "",
    });

    // Delegate-while-cold: a mostly-empty store means most expert reads
    // would stall on the network anyway; a warm peer computes the whole
    // answer faster than we can fetch our way through it. Any failure falls
    // through to local generation, so this can only help.
    if (delegateTarget(ctx)) |target| {
        defer gpa.free(target.addr);
        const parse_special_d = is_chat and chat_template.usesSpecialMarkers(ctx.gen.chatFormat());
        // DSD draft-local (whitepaper roadmap 6): greedy requests first try
        // drafting with the partial local store and having the warm peer
        // verify windows -- traffic becomes tokens instead of experts, and
        // every emitted token is exactly what the peer would have produced.
        // A collapsed acceptance rate bails to wholesale GEN below; any error
        // does the same. LOOM_NO_DRAFT=1 disables for A/B.
        if (temp <= 0 and std.c.getenv("LOOM_NO_DRAFT") == null) {
            if (draftDelegate(ctx, target.addr, prompt_text, max_tokens, parse_special_d)) |maybe| {
                if (maybe) |dres| {
                    defer gpa.free(dres.text);
                    if (ctx.meter) |m| _ = m.settle(client, reserved, @intCast(dres.prompt_tokens + dres.completion_tokens));
                    return try delegatedResponse(ctx, is_chat, stream, model_id, dres, wi);
                }
                // null = controller bailed early; fall through to GEN
            } else |e| {
                consoleLine(ctx, "draft-verify with {s} failed ({s}); trying wholesale delegation", .{ target.addr, @errorName(e) });
            }
        }
        if (delegateGenerate(ctx, target.addr, prompt_text, max_tokens, temp, seed, parse_special_d)) |dres| {
            defer gpa.free(dres.text);
            if (ctx.meter) |m| _ = m.settle(client, reserved, @intCast(dres.prompt_tokens + dres.completion_tokens));
            consoleLine(ctx, "served   {s}: {d} token(s) delegated to {s} ({d:.2} tok/s there)", .{
                if (is_chat) @as([]const u8, "chat") else "completion", dres.completion_tokens, target.addr, dres.tok_per_s,
            });
            return try delegatedResponse(ctx, is_chat, stream, model_id, dres, wi);
        } else |e| {
            consoleLine(ctx, "delegate to {s} failed ({s}); generating locally", .{ target.addr, @errorName(e) });
        }
    }

    // Streaming: write the SSE response to the connection and return null.
    if (stream) {
        if (ctx.alpha_metrics) |am| am.beginGen(stats.nowNs(ctx.io));
        defer if (ctx.alpha_metrics) |am| am.endGen();
        streamCompletions(ctx, dl, wi, is_chat, model_id, client, prompt_text, max_tokens, temp, seed, budget, reserved);
        return null;
    }

    // Parse special tokens only for chat formats whose scaffold uses them
    // (chatml/llama3/gemma); never for a raw completion prompt or a text-marker
    // chat format, so user content cannot inject control tokens.
    const parse_special = is_chat and chat_template.usesSpecialMarkers(ctx.gen.chatFormat());

    // Generate under the shared engine mutex (rpc + openai serialize here).
    const gen_t0 = stats.nowNs(ctx.io);
    if (ctx.alpha_metrics) |am| am.beginGen(gen_t0);
    ctx.engine_lock.lockUncancelable(ctx.io);
    var res = ctx.gen.generate(gpa, ctx.io, prompt_text, max_tokens, temp, seed, budget, null, parse_special) catch |e| {
        ctx.engine_lock.unlock(ctx.io);
        if (ctx.alpha_metrics) |am| am.endGen();
        consoleLine(ctx, "failed   {s}: {s}", .{ if (is_chat) @as([]const u8, "chat") else "completion", @errorName(e) });
        // failed generation still releases the reservation
        if (ctx.meter) |m| _ = m.settle(client, reserved, 0);
        return e;
    };
    const gen_hit = ctx.gen.hitRate();
    ctx.engine_lock.unlock(ctx.io);
    defer res.deinit(gpa);
    {
        const secs = @as(f64, @floatFromInt(stats.nowNs(ctx.io) - gen_t0)) / 1e9;
        const tok_s = if (secs > 0) @as(f64, @floatFromInt(res.completion_tokens)) / secs else 0;
        if (ctx.alpha_metrics) |am| {
            am.recordGen(tok_s, gen_hit);
            am.endGen();
        }
        consoleLine(ctx, "served   {s}: {d} token(s) in {d:.1}s ({d:.2} tok/s, hit {d:.3})", .{
            if (is_chat) @as([]const u8, "chat") else "completion", res.completion_tokens, secs, tok_s, gen_hit,
        });
    }

    const content = try jsonEscapeAlloc(gpa, res.text);
    defer gpa.free(content);

    // Settle: keep prompt + completion, refund the rest of the reservation.
    if (ctx.meter) |m| {
        _ = m.settle(client, reserved, @intCast(res.prompt_tokens + res.completion_tokens));
    }

    const finish: []const u8 = if (res.stop) "stop" else "length";
    const id = id_counter.fetchAdd(1, .monotonic) + 1;
    // model_id comes from the request body; unescaped it could inject JSON keys
    // (a forged usage/total_tokens ahead of the real one desynchronised
    // light-node accounting) or terminate an SSE event (security issue #30)
    const model_esc = try jsonEscapeAlloc(gpa, model_id);
    defer gpa.free(model_esc);
    const pt = res.prompt_tokens;
    const n = res.completion_tokens;

    const body = if (is_chat)
        try std.fmt.allocPrint(gpa,
            \\{{"id":"chatcmpl-loom-{d}","object":"chat.completion","created":0,"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
        , .{ id, model_esc, content, finish, pt, n, pt + n })
    else
        try std.fmt.allocPrint(gpa,
            \\{{"id":"cmpl-loom-{d}","object":"text_completion","created":0,"model":"{s}","choices":[{{"index":0,"text":"{s}","finish_reason":"{s}"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
        , .{ id, model_esc, content, finish, pt, n, pt + n });

    return .{ .status = 200, .body = body };
}

/// SSE streaming sink: emits one `chat.completion.chunk` / `text_completion`
/// event per token as it is produced. A write error (client disconnected)
/// propagates out and aborts generation.
const StreamCtx = struct {
    wi: *Io.Writer,
    gpa: std.mem.Allocator,
    io: Io,
    /// Connection deadline slot, refreshed on every token. Without this a
    /// generation slower than `SERVE_TIMEOUT_S` has its socket shut down
    /// mid-answer by the idle reaper — measured across two machines, every
    /// distributed request died at exactly 30 s no matter how much progress it
    /// had made. A stream that is emitting tokens is not idle.
    dl: ?usize,
    id: u64,
    model: []const u8,
    is_chat: bool,
    /// Tokens actually emitted. A client that disconnects mid-stream makes the
    /// write fail, which aborts generation — the compute is already spent, so
    /// this is what gets billed (security issue #30).
    emitted: u64 = 0,

    fn emit(ptr: *anyopaque, bytes: []const u8) anyerror!void {
        const self: *StreamCtx = @ptrCast(@alignCast(ptr));
        self.emitted += 1;
        sockopt.refreshServe(self.io, self.dl);
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
    dl_slot: ?usize,
    wi: *Io.Writer,
    is_chat: bool,
    model_id: []const u8,
    client: []const u8,
    prompt_text: []const u8,
    max_tokens: usize,
    temp: f32,
    seed: u64,
    budget: ?u64,
    reserved: u64,
) void {
    const gpa = ctx.gpa;
    const id = id_counter.fetchAdd(1, .monotonic) + 1;
    // client-controlled: unescaped it can terminate an SSE event and forge
    // `data:` frames to the client (security issue #30)
    const model_esc = jsonEscapeAlloc(gpa, model_id) catch return;
    defer gpa.free(model_esc);

    wi.print("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", .{}) catch return;
    if (is_chat) {
        wi.print("data: {{\"id\":\"chatcmpl-loom-{d}\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}\n\n", .{ id, model_esc }) catch return;
    }
    wi.flush() catch return;

    var sctx = StreamCtx{ .wi = wi, .gpa = gpa, .io = ctx.io, .dl = dl_slot, .id = id, .model = model_esc, .is_chat = is_chat };
    const sink = generator.TokenSink{ .ctx = &sctx, .emit = StreamCtx.emit };

    const parse_special = is_chat and chat_template.usesSpecialMarkers(ctx.gen.chatFormat());
    ctx.engine_lock.lockUncancelable(ctx.io);
    var res = ctx.gen.generate(gpa, ctx.io, prompt_text, max_tokens, temp, seed, budget, sink, parse_special) catch {
        ctx.engine_lock.unlock(ctx.io);
        // A disconnect aborts generation here. The prefill and every emitted
        // token were still computed, so bill them rather than letting an
        // aborted stream be free compute (security issue #30).
        if (ctx.meter) |m| _ = m.settle(client, reserved, sctx.emitted);
        wi.print("data: {{\"error\":{{\"message\":\"generation_failed\",\"type\":\"server_error\"}}}}\n\ndata: [DONE]\n\n", .{}) catch {};
        wi.flush() catch {};
        return;
    };
    ctx.engine_lock.unlock(ctx.io);
    defer res.deinit(gpa);

    if (ctx.meter) |m| {
        _ = m.settle(client, reserved, @intCast(res.prompt_tokens + res.completion_tokens));
    }

    const finish: []const u8 = if (res.stop) "stop" else "length";
    if (is_chat) {
        wi.print("data: {{\"id\":\"chatcmpl-loom-{d}\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{s}\"}}]}}\n\n", .{ id, model_esc, finish }) catch return;
    } else {
        wi.print("data: {{\"id\":\"cmpl-loom-{d}\",\"object\":\"text_completion\",\"created\":0,\"model\":\"{s}\",\"choices\":[{{\"index\":0,\"text\":\"\",\"finish_reason\":\"{s}\"}}]}}\n\n", .{ id, model_esc, finish }) catch return;
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
const DelegateResult = struct {
    text: []u8,
    prompt_tokens: usize,
    completion_tokens: usize,
    tok_per_s: f64,
};

fn delegateTarget(ctx: *Ctx) ?struct { addr: []u8, frac: f64 } {
    if (ctx.delegate_below <= 0) return null;
    const st = ctx.store orelse return null;
    const table = ctx.peers orelse return null;
    const total = st.manifest.nRanges();
    if (total == 0) return null;
    const mine = @as(f64, @floatFromInt(st.holdings.count())) / @as(f64, @floatFromInt(total));
    if (mine >= ctx.delegate_below) return null;
    const best = (table.warmest(ctx.gpa, stats.nowNs(ctx.io), total) catch return null) orelse return null;
    // A partial peer would just relocate the misses; require a warm one.
    if (best.frac < 0.9) {
        ctx.gpa.free(best.addr);
        return null;
    }
    return .{ .addr = best.addr, .frac = best.frac };
}

fn delegateGenerate(
    ctx: *Ctx,
    addr_str: []const u8,
    prompt: []const u8,
    max_tokens: usize,
    temp: f32,
    seed: u64,
    parse_special: bool,
) !DelegateResult {
    const gpa = ctx.gpa;
    const addr = try sync.PeerAddr.parse(addr_str);
    const peer = try sync.Peer.connect(gpa, ctx.io, addr);
    defer peer.close(gpa);
    const prompt_esc = try jsonEscapeAlloc(gpa, prompt);
    defer gpa.free(prompt_esc);
    try peer.send("GEN {{\"prompt\":\"{s}\",\"max_tokens\":{d},\"temperature\":{d:.3},\"seed\":{d},\"parse_special\":{}}}\n", .{
        prompt_esc, max_tokens, temp, seed, parse_special,
    });
    // the remote generation takes minutes; stretch this connection's reaper
    sockopt.refresh(ctx.io, peer.deadline, sockopt.GENERATE_TIMEOUT_S);
    const line = try peer.recvLine();
    if (!std.mem.startsWith(u8, line, "GENR ")) return error.DelegateRefused;
    const flen = std.fmt.parseInt(usize, fieldOf(line, "len") orelse return error.BadReply, 10) catch return error.BadReply;
    if (flen > 1 << 22) return error.BadReply;
    const pt = std.fmt.parseInt(usize, fieldOf(line, "prompt_tokens") orelse "0", 10) catch 0;
    const ct = std.fmt.parseInt(usize, fieldOf(line, "completion_tokens") orelse "0", 10) catch 0;
    const tps = std.fmt.parseFloat(f64, fieldOf(line, "tok_per_s") orelse "0") catch 0;
    const text = try gpa.alloc(u8, flen);
    errdefer gpa.free(text);
    try peer.r.interface.readSliceAll(text);
    return .{ .text = text, .prompt_tokens = pt, .completion_tokens = ct, .tok_per_s = tps };
}

const DraftPeer = struct {
    ctx: *Ctx,
    addr: sync.PeerAddr,
    peer: ?*sync.Peer = null,

    /// Lazy connect + reconnect: the local drafting between rounds can outlast
    /// the responder's read deadline, so the server reaping our idle
    /// connection is normal operation, not an error. Each verify round
    /// tolerates exactly one reconnect.
    fn ensure(self: *DraftPeer) !*sync.Peer {
        if (self.peer) |p| return p;
        const p = try sync.Peer.connect(self.ctx.gpa, self.ctx.io, self.addr);
        sockopt.refresh(self.ctx.io, p.deadline, sockopt.GENERATE_TIMEOUT_S);
        self.peer = p;
        return p;
    }

    fn drop(self: *DraftPeer) void {
        if (self.peer) |p| p.close(self.ctx.gpa);
        self.peer = null;
    }

    fn verify(opaque_self: *anyopaque, tokens: []const u32, draft: []const u32) anyerror!generator.DraftVerify {
        const self: *DraftPeer = @ptrCast(@alignCast(opaque_self));
        return self.verifyOnce(tokens, draft) catch {
            // stale connection (server reaped it while we drafted): one fresh
            // connection, one retry, then the error is real
            self.drop();
            return self.verifyOnce(tokens, draft);
        };
    }

    fn verifyOnce(self: *DraftPeer, tokens: []const u32, draft: []const u32) anyerror!generator.DraftVerify {
        const gpa = self.ctx.gpa;
        const peer = try self.ensure();
        errdefer self.drop();
        var aw = std.Io.Writer.Allocating.init(gpa);
        defer aw.deinit();
        try aw.writer.print("DRAFT {{\"ctx\":[", .{});
        for (tokens, 0..) |t, i| {
            if (i != 0) try aw.writer.print(",", .{});
            try aw.writer.print("{d}", .{t});
        }
        try aw.writer.print("],\"draft\":[", .{});
        for (draft, 0..) |t, i| {
            if (i != 0) try aw.writer.print(",", .{});
            try aw.writer.print("{d}", .{t});
        }
        try aw.writer.print("]}}\n", .{});
        try peer.send("{s}", .{aw.writer.buffered()});
        sockopt.refresh(self.ctx.io, peer.deadline, sockopt.GENERATE_TIMEOUT_S);
        const line = try peer.recvLine();
        if (!std.mem.startsWith(u8, line, "DRAFTR ")) return error.DraftRefused;
        const acc = std.fmt.parseInt(usize, fieldOf(line, "accepted") orelse return error.BadReply, 10) catch return error.BadReply;
        const corr = std.fmt.parseInt(u32, fieldOf(line, "correction") orelse return error.BadReply, 10) catch return error.BadReply;
        return .{ .accepted = acc, .correction = corr };
    }
};

/// Draft locally, verify remotely. Returns null when the acceptance
/// controller bailed (caller falls back to wholesale GEN); the few verified
/// tokens of a bailed attempt are discarded -- the bail fires within two
/// windows, so the waste is bounded and the GEN answer stays token-exact.
fn draftDelegate(
    ctx: *Ctx,
    addr_str: []const u8,
    prompt: []const u8,
    max_tokens: usize,
    parse_special: bool,
) !?DelegateResult {
    const gg = switch (ctx.gen.*) {
        .gguf => |p| p,
        else => return error.DraftUnsupported,
    };
    const gpa = ctx.gpa;
    const addr = try sync.PeerAddr.parse(addr_str);
    var dp = DraftPeer{ .ctx = ctx, .addr = addr };
    defer dp.drop();

    const t0 = stats.nowNs(ctx.io);
    ctx.engine_lock.lockUncancelable(ctx.io);
    const out = generator.generateDrafted(gg, gpa, prompt, max_tokens, parse_special, .{
        .ctx = @ptrCast(&dp),
        .call = DraftPeer.verify,
    }) catch |e| {
        ctx.engine_lock.unlock(ctx.io);
        return e;
    };
    ctx.engine_lock.unlock(ctx.io);
    const secs = @as(f64, @floatFromInt(stats.nowNs(ctx.io) - t0)) / 1e9;

    if (ctx.alpha_metrics) |am| am.recordDraft(out.rounds, out.drafted, out.accepted, out.final_gamma, out.bailed);
    if (out.bailed) {
        gpa.free(out.res.text);
        consoleLine(ctx, "draft-verify bailed after {d} round(s) (acceptance collapsed); wholesale delegation", .{out.rounds});
        return null;
    }
    const tok_s = if (secs > 0) @as(f64, @floatFromInt(out.res.completion_tokens)) / secs else 0;
    consoleLine(ctx, "served   draft-verify: {d} token(s), {d}/{d} drafts accepted over {d} round(s), gamma {d} ({d:.2} tok/s)", .{
        out.res.completion_tokens, out.accepted, out.drafted, out.rounds, out.final_gamma, tok_s,
    });
    return .{
        .text = @constCast(out.res.text),
        .prompt_tokens = out.res.prompt_tokens,
        .completion_tokens = out.res.completion_tokens,
        .tok_per_s = tok_s,
    };
}

/// "key=value" scan over a space-separated reply line.
fn fieldOf(line: []const u8, key: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, line, ' ');
    while (it.next()) |tok| {
        if (tok.len > key.len + 1 and std.mem.startsWith(u8, tok, key) and tok[key.len] == '=')
            return tok[key.len + 1 ..];
    }
    return null;
}

fn delegatedResponse(ctx: *Ctx, is_chat: bool, stream: bool, model_id: []const u8, dres: DelegateResult, wi: *Io.Writer) !?Response {
    const gpa = ctx.gpa;
    const content = try jsonEscapeAlloc(gpa, dres.text);
    defer gpa.free(content);
    const model_esc = try jsonEscapeAlloc(gpa, model_id);
    defer gpa.free(model_esc);
    const id = id_counter.fetchAdd(1, .monotonic) + 1;
    if (stream) {
        // Valid SSE with the whole delegated answer as one delta.
        try wi.print("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n", .{});
        try wi.print("data: {{\"id\":\"chatcmpl-loom-{d}\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"{s}\",\"loom_delegated\":true,\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"{s}\"}},\"finish_reason\":null}}]}}\n\n", .{ id, model_esc, content });
        try wi.print("data: {{\"id\":\"chatcmpl-loom-{d}\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"{s}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n", .{ id, model_esc });
        try wi.flush();
        return null;
    }
    const body = if (is_chat)
        try std.fmt.allocPrint(gpa,
            \\{{"id":"chatcmpl-loom-{d}","object":"chat.completion","created":0,"model":"{s}","loom_delegated":true,"choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
        , .{ id, model_esc, content, dres.prompt_tokens, dres.completion_tokens, dres.prompt_tokens + dres.completion_tokens })
    else
        try std.fmt.allocPrint(gpa,
            \\{{"id":"cmpl-loom-{d}","object":"text_completion","created":0,"model":"{s}","loom_delegated":true,"choices":[{{"index":0,"text":"{s}","finish_reason":"stop"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
        , .{ id, model_esc, content, dres.prompt_tokens, dres.completion_tokens, dres.prompt_tokens + dres.completion_tokens });
    return .{ .status = 200, .body = body };
}

fn consoleLine(ctx: *Ctx, comptime fmt: []const u8, args: anytype) void {
    const cw = ctx.console orelse return;
    const lk = ctx.console_lock orelse return;
    lk.lockUncancelable(ctx.io);
    defer lk.unlock(ctx.io);
    cw.print(fmt ++ "\n", args) catch {};
    cw.flush() catch {};
}

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

fn writeHttp(wi: *Io.Writer, status: u16, body: []const u8, ct: ContentType) !void {
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
    try wi.print("Content-Type: {s}\r\n", .{ct.mime()});
    try wi.print("Content-Length: {d}\r\n", .{body.len});
    try wi.print("Access-Control-Allow-Origin: *\r\n", .{});
    try wi.print("Connection: close\r\n\r\n", .{});
    try wi.print("{s}", .{body});
    try wi.flush();
}
