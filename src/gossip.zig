//! Gossip loop (ROADMAP #7, minimal in-house form): on a short interval, dial
//! every peer in the table, announce ourselves (address + manifest version +
//! holdings bitmap), and merge the peer's table into ours. Discovery is
//! transitive: if B knows A, then C — told only about B — learns A within one
//! round and the eager repair loop can immediately fetch from it.
//!
//! At LAN scale we exchange the full holdings bitmap; at swarm scale this
//! payload moves to a gossipsub topic and the ENR carries only the compact
//! summary (see ROADMAP #5/#7).

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const peers = @import("peers.zig");
const weights = @import("weights.zig");
const sync = @import("sync.zig");
const hashmod = @import("hash.zig");
const stats = @import("stats.zig");

pub const INTERVAL_NS: i96 = 3 * std.time.ns_per_s;

pub const Ctx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    table: *peers.Table,
    store: ?*weights.Store,
    advertise: []const u8,
};

/// One gossip exchange with one peer. Announces us, merges their peer list.
fn exchange(ctx: *Ctx, addr_str: []const u8) !void {
    const gpa = ctx.gpa;
    const io = ctx.io;
    const addr = try sync.PeerAddr.parse(addr_str);
    const ip = try net.IpAddress.parse(addr.host, addr.port);
    const stream = try ip.connect(io, .{ .mode = .stream });
    defer stream.close(io);

    var rbuf: [1 << 16]u8 = undefined;
    var wbuf: [4096]u8 = undefined;
    var r = stream.reader(io, &rbuf);
    var w = stream.writer(io, &wbuf);

    // announce
    const zero_version = "0" ** 64;
    if (ctx.store) |st| {
        const vhex = hashmod.toHex(st.manifest.version);
        const hhex = try st.holdings.toHex(gpa);
        defer gpa.free(hhex);
        try w.interface.print("GOSSIP addr={s} version={s} holdings={s}\n", .{ ctx.advertise, vhex, hhex });
    } else {
        try w.interface.print("GOSSIP addr={s} version={s} holdings=\n", .{ ctx.advertise, zero_version });
    }
    try w.interface.flush();

    // merge their list
    const hline = std.mem.trimEnd(u8, try r.interface.takeDelimiterInclusive('\n'), "\r\n");
    if (!std.mem.startsWith(u8, hline, "PEERS ")) return error.BadGossipResponse;
    const n = try std.fmt.parseInt(usize, hline[6..], 10);
    if (n > 4096) return error.BadGossipResponse;
    const now = stats.nowNs(io);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const line = std.mem.trimEnd(u8, try r.interface.takeDelimiterInclusive('\n'), "\r\n");
        const a = fieldOf(line, "addr") orelse continue;
        const v = fieldOf(line, "version") orelse continue;
        const h = fieldOf(line, "holdings") orelse "";
        _ = ctx.table.merge(a, v, h, now) catch continue;
    }
}

fn fieldOf(line: []const u8, key: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, line, ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, key) and tok.len > key.len and tok[key.len] == '=') {
            return tok[key.len + 1 ..];
        }
    }
    return null;
}

/// The loop: every INTERVAL_NS, exchange with every currently known peer.
/// Unreachable peers are skipped this round and retried next round (they stay
/// in the table — eager churn repair wants them retried, not forgotten).
pub fn loop(ctx: *Ctx) void {
    while (true) {
        Io.sleep(ctx.io, .{ .nanoseconds = INTERVAL_NS }, .awake) catch return;
        const addrs = ctx.table.snapshotAddrs(ctx.gpa) catch continue;
        defer {
            for (addrs) |a| ctx.gpa.free(a);
            ctx.gpa.free(addrs);
        }
        for (addrs) |a| {
            exchange(ctx, a) catch continue;
        }
    }
}
