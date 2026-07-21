//! Light-node OpenAI delegation (SPEC.md client API, issue #16).
//!
//! Exposes an OpenAI-compatible HTTP endpoint on a low-memory device and
//! transparently proxies every request to a full node's OpenAI endpoint
//! (round-robin with failover), forcing the light node's own client id via
//! `Authorization: Bearer` so the full node meters it and a light node cannot be
//! turned into an open proxy (parallels the native force-client-id rule in
//! light.zig). Spend is tallied from the response `usage.total_tokens`.
//!
//! The light node runs NO engine; it is a metered, load-balancing reverse proxy
//! for the OpenAI surface.

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const sync = @import("../p2p/sync.zig");

pub const Options = struct {
    addr: []const u8,
    port: u16,
    backends: []const sync.PeerAddr, // full-node OpenAI endpoints
    client_id: []const u8,
};

pub const Ctx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    opts: Options,
    rr: usize = 0, // round-robin start across backends
    spent: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
};

const Conn = struct { ctx: *Ctx, stream: net.Stream };

const MAX_CONNS: u32 = 128;
var live_conns: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);

pub fn serve(ctx: *Ctx) !void {
    var address = try net.IpAddress.parse(ctx.opts.addr, ctx.opts.port);
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

fn handleConn(ctx: *Ctx, stream: net.Stream) !void {
    defer stream.close(ctx.io);
    var rbuf: [1 << 16]u8 = undefined;
    var wbuf: [1 << 16]u8 = undefined;
    var r = stream.reader(ctx.io, &rbuf);
    var w = stream.writer(ctx.io, &wbuf);
    const ri = &r.interface;
    const wi = &w.interface;

    const req = readRequest(ctx.gpa, ri) catch {
        try writeHttp(wi, 400, "{\"error\":{\"message\":\"malformed_request\",\"type\":\"invalid_request_error\"}}");
        return;
    };
    defer req.deinit(ctx.gpa);

    const resp = delegate(ctx, req) catch {
        try writeHttp(wi, 502, "{\"error\":{\"message\":\"no full node reachable\",\"type\":\"server_error\"}}");
        return;
    };
    defer ctx.gpa.free(resp.body);
    try writeHttp(wi, resp.status, resp.body);
}

// ---- incoming request (method, path, body; bearer is dropped) --------------

const Request = struct {
    method: []u8,
    path: []u8,
    body: []u8,

    fn deinit(self: Request, gpa: std.mem.Allocator) void {
        gpa.free(self.method);
        gpa.free(self.path);
        gpa.free(self.body);
    }
};

fn readRequest(gpa: std.mem.Allocator, ri: *Io.Reader) !Request {
    const line0 = trimCrlf(try ri.takeDelimiterInclusive('\n'));
    var it = std.mem.splitScalar(u8, line0, ' ');
    const method = try gpa.dupe(u8, it.next() orelse return error.BadRequest);
    errdefer gpa.free(method);
    const path = try gpa.dupe(u8, it.next() orelse return error.BadRequest);
    errdefer gpa.free(path);

    var content_length: usize = 0;
    while (true) {
        const h = trimCrlf(try ri.takeDelimiterInclusive('\n'));
        if (h.len == 0) break;
        if (headerIs(h, "content-length")) {
            content_length = std.fmt.parseInt(usize, std.mem.trim(u8, headerValue(h), " "), 10) catch 0;
        }
    }
    if (content_length > (8 << 20)) return error.BodyTooLarge;
    const body = try gpa.alloc(u8, content_length);
    errdefer gpa.free(body);
    if (content_length > 0) try ri.readSliceAll(body);
    return .{ .method = method, .path = path, .body = body };
}

// ---- delegation ------------------------------------------------------------

const Response = struct { status: u16, body: []u8 };

/// Forward to the first responsive backend (round-robin start), forcing our
/// client id; relay the response and tally spend.
fn delegate(ctx: *Ctx, req: Request) !Response {
    const n = ctx.opts.backends.len;
    if (n == 0) return error.NoBackends;
    const start = ctx.rr;
    ctx.rr +%= 1;

    var attempt: usize = 0;
    while (attempt < n) : (attempt += 1) {
        const backend = ctx.opts.backends[(start + attempt) % n];
        const resp = forwardTo(ctx, backend, req) catch continue;
        trackSpend(ctx, resp.body);
        return resp;
    }
    return error.NoBackendReachable;
}

fn forwardTo(ctx: *Ctx, backend: sync.PeerAddr, req: Request) !Response {
    const io = ctx.io;
    const gpa = ctx.gpa;
    const ip = try net.IpAddress.parse(backend.host, backend.port);
    const stream = try ip.connect(io, .{ .mode = .stream });
    defer stream.close(io);
    var rbuf: [1 << 16]u8 = undefined;
    var wbuf: [1 << 16]u8 = undefined;
    var r = stream.reader(io, &rbuf);
    var w = stream.writer(io, &wbuf);
    const ri = &r.interface;
    const wi = &w.interface;

    // Request with our forced bearer id (drops any caller-supplied credential).
    try wi.print("{s} {s} HTTP/1.1\r\n", .{ req.method, req.path });
    try wi.print("Host: {s}:{d}\r\n", .{ backend.host, backend.port });
    try wi.print("Authorization: Bearer {s}\r\n", .{ctx.opts.client_id});
    try wi.print("Content-Type: application/json\r\n", .{});
    try wi.print("Content-Length: {d}\r\n", .{req.body.len});
    try wi.print("Connection: close\r\n\r\n", .{});
    if (req.body.len > 0) try wi.print("{s}", .{req.body});
    try wi.flush();

    // Response: status code + Content-Length body.
    const status_line = trimCrlf(try ri.takeDelimiterInclusive('\n'));
    const status = parseStatus(status_line);
    var content_length: usize = 0;
    while (true) {
        const h = trimCrlf(try ri.takeDelimiterInclusive('\n'));
        if (h.len == 0) break;
        if (headerIs(h, "content-length")) {
            content_length = std.fmt.parseInt(usize, std.mem.trim(u8, headerValue(h), " "), 10) catch 0;
        }
    }
    const body = try gpa.alloc(u8, content_length);
    errdefer gpa.free(body);
    if (content_length > 0) try ri.readSliceAll(body);
    return .{ .status = status, .body = body };
}

/// Tally spend from the response `usage.total_tokens` (the light node's own view
/// of what it owes across backends), mirroring light.zig's cost tally.
fn trackSpend(ctx: *Ctx, body: []const u8) void {
    const key = "\"total_tokens\":";
    const idx = std.mem.indexOf(u8, body, key) orelse return;
    var end = idx + key.len;
    var cost: u64 = 0;
    while (end < body.len and body[end] >= '0' and body[end] <= '9') : (end += 1) {
        cost = cost * 10 + (body[end] - '0');
    }
    if (cost > 0) {
        const total = ctx.spent.fetchAdd(cost, .monotonic) + cost;
        std.debug.print("light-openai: spent {d} tokens this request ({d} total)\n", .{ cost, total });
    }
}

// ---- helpers ---------------------------------------------------------------

fn trimCrlf(s: []const u8) []const u8 {
    return std.mem.trimEnd(u8, s, "\r\n");
}

fn headerIs(h: []const u8, name: []const u8) bool {
    const colon = std.mem.indexOfScalar(u8, h, ':') orelse return false;
    return std.ascii.eqlIgnoreCase(std.mem.trim(u8, h[0..colon], " "), name);
}

fn headerValue(h: []const u8) []const u8 {
    const colon = std.mem.indexOfScalar(u8, h, ':') orelse return "";
    return h[colon + 1 ..];
}

/// "HTTP/1.1 200 OK" -> 200 (falls back to 502 on a malformed status line).
fn parseStatus(status_line: []const u8) u16 {
    var it = std.mem.splitScalar(u8, status_line, ' ');
    _ = it.next(); // HTTP/1.1
    const code = it.next() orelse return 502;
    return std.fmt.parseInt(u16, code, 10) catch 502;
}

fn writeHttp(wi: *Io.Writer, status: u16, body: []const u8) !void {
    const reason = switch (status) {
        200 => "OK",
        400 => "Bad Request",
        402 => "Payment Required",
        404 => "Not Found",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
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
