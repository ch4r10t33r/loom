//! Light node (SPEC.md node classes): local inference for low-memory devices.
//!
//! A light node holds NO weights, NO store, and runs NO engine — its memory
//! footprint is a few megabytes. It serves the exact same line-delimited JSON
//! RPC as a full node on localhost, and **delegates** every request to a full
//! node, with round-robin failover across the configured backends. It stamps
//! its client identity onto each request so the serving full node can meter
//! it (the compensation ledger lives on the full node; the light node tracks
//! its own spend from the cost/balance fields in responses).
//!
//! v1 backends are configured statically (`--full-nodes`); discovering
//! RPC-serving full nodes via the gossip mesh is a noted follow-up.

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const sync = @import("../p2p/sync.zig");
const sockopt = @import("../core/sockopt.zig");
const dns = @import("../p2p/dns.zig");

pub const Options = struct {
    rpc_addr: []const u8,
    rpc_port: u16,
    full_nodes: []const sync.PeerAddr, // full-node RPC endpoints
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
    var address = try net.IpAddress.parse(ctx.opts.rpc_addr, ctx.opts.rpc_port);
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

fn handleConn(ctx: *Ctx, stream: net.Stream) !void {
    defer stream.close(ctx.io);
    const dl = sockopt.trackServe(ctx.io, stream);
    defer sockopt.untrack(ctx.io, dl);
    var rbuf: [1 << 16]u8 = undefined;
    var wbuf: [1 << 16]u8 = undefined;
    var r = stream.reader(ctx.io, &rbuf);
    var w = stream.writer(ctx.io, &wbuf);

    while (true) {
        const raw = r.interface.takeDelimiterInclusive('\n') catch return;
        const line = std.mem.trimEnd(u8, raw, "\r\n");
        if (line.len == 0) continue;
        delegate(ctx, line, &w.interface) catch |e| {
            try w.interface.print("{{\"ok\":false,\"error\":\"{s}\"}}\n", .{@errorName(e)});
            try w.interface.flush();
        };
    }
}

/// Stamp our client id onto the request (unless the caller set one), forward
/// to the first responsive backend (round-robin start), relay the response.
fn delegate(ctx: *Ctx, line: []const u8, wi: *Io.Writer) !void {
    const gpa = ctx.gpa;

    // FORCE our configured client id (audit #6 P0-3): a caller-supplied
    // "client" field is never honored, so a light node cannot be used as an
    // open proxy to spend/mint under arbitrary identities on full nodes.
    // Parse the object, drop any inbound "client", stamp ours.
    var parsed = std.json.parseFromSlice(std.json.Value, gpa, line, .{}) catch
        return error.BadRequest;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadRequest;
    // Only inference passes through (security issue #30). Forwarding `method`
    // verbatim made a light node an anonymous oracle for brute-forcing the full
    // node's admin token, from behind the proxy's IP and unlogged.
    if (parsed.value.object.get("method")) |mv| {
        if (mv != .string or !std.mem.eql(u8, mv.string, "tab")) return error.MethodNotAllowed;
    }
    _ = parsed.value.object.orderedRemove("admin_token");
    try parsed.value.object.put(parsed.arena.allocator(), "client", .{ .string = ctx.opts.client_id });
    const fixed = try std.json.Stringify.valueAlloc(gpa, parsed.value, .{});
    defer gpa.free(fixed);

    const n_backends = ctx.opts.full_nodes.len;
    if (n_backends == 0) return error.NoFullNodes;
    const start = ctx.rr;
    ctx.rr +%= 1;

    var attempt: usize = 0;
    while (attempt < n_backends) : (attempt += 1) {
        const backend = ctx.opts.full_nodes[(start + attempt) % n_backends];
        var sent = false;
        const resp = forwardTo(ctx, backend, fixed, &sent) catch {
            // at-most-once across backends (security issue #159): a failure
            // after the request went out is ambiguous -- the backend may have
            // generated and charged -- so it must not be replayed elsewhere
            if (sent) return error.AmbiguousBackendFailure;
            continue;
        };
        defer gpa.free(resp);
        trackSpend(ctx, resp);
        try wi.print("{s}\n", .{resp});
        try wi.flush();
        return;
    }
    return error.NoFullNodeReachable;
}

fn forwardTo(ctx: *Ctx, backend: sync.PeerAddr, request: []const u8, sent: *bool) ![]u8 {
    const io = ctx.io;
    const ip = try dns.resolve(io, backend.host, backend.port);
    const stream = try ip.connect(io, .{ .mode = .stream });
    const dl = sockopt.trackPeer(io, stream);
    defer sockopt.untrack(io, dl);
    defer stream.close(io);
    var rbuf: [1 << 16]u8 = undefined;
    var wbuf: [1 << 16]u8 = undefined;
    var r = stream.reader(io, &rbuf);
    var w = stream.writer(io, &wbuf);

    sent.* = true; // commit point: bytes may reach the backend from here on
    try w.interface.print("{s}\n", .{request});
    try w.interface.flush();
    const line = std.mem.trimEnd(u8, try r.interface.takeDelimiterInclusive('\n'), "\r\n");
    return ctx.gpa.dupe(u8, line);
}

/// Pull "cost" out of the response to keep a local spend tally — the light
/// node's own view of what it owes across backends.
fn trackSpend(ctx: *Ctx, resp: []const u8) void {
    const key = "\"cost\":";
    const idx = std.mem.indexOf(u8, resp, key) orelse return;
    var end = idx + key.len;
    var cost: u64 = 0;
    while (end < resp.len and resp[end] >= '0' and resp[end] <= '9') : (end += 1) {
        cost = cost * 10 + (resp[end] - '0');
    }
    if (cost > 0) {
        const total = ctx.spent.fetchAdd(cost, .monotonic) + cost;
        std.debug.print("light: spent {d} tokens this request ({d} total)\n", .{ cost, total });
    }
}
