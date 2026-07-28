//! Periodic one-line status report for a running node.
//!
//! A node is a long-lived background process, and the interesting facts about
//! it change without any request arriving: peers join and drop, gossip
//! discovers new holders, repair pulls shards in, and the local hit rate moves
//! as fetch-on-demand replicates hot experts. None of that shows up in a log
//! that only prints on events, so a node can look idle while its membership
//! churns underneath.
//!
//! One compact line at a fixed interval, in a stable field order so `grep` and
//! `awk` work on it:
//!
//!     status  peers 3  committee 2  shards 4021/6289 (63.9%)  local-hit 87.4%  up 12m
//!
//! Every field is read through the same locks the live code uses, so this is a
//! consistent snapshot rather than a torn read.

const std = @import("std");
const Io = std.Io;
const peers_mod = @import("../p2p/peers.zig");
const weights = @import("../p2p/weights.zig");
const generator = @import("generator.zig");
const stats = @import("../core/stats.zig");

pub const Reporter = struct {
    io: Io,
    out: *Io.Writer,
    /// Serializes writes with the rest of the node's console output.
    out_lock: *Io.Mutex,
    table: ?*peers_mod.Table,
    store: ?*weights.Store,
    gen: *generator.Generator,
    /// Members of this node's committee, as configured at join.
    committee: usize,
    /// Redundancy target, for the under-replication warning. 0 disables it.
    r_target: usize,
    /// Scratch for the per-shard holder count.
    gpa: std.mem.Allocator,
    interval_ns: u64,
    start_ns: i128,
    /// Set false to stop the loop; the thread exits within one interval.
    running: std.atomic.Value(bool) = std.atomic.Value(bool).init(true),

    pub fn stop(self: *Reporter) void {
        self.running.store(false, .monotonic);
    }

    fn line(self: *Reporter) void {
        const n_peers: usize = if (self.table) |t| t.liveCount() else 0;

        self.out_lock.lockUncancelable(self.io);
        defer self.out_lock.unlock(self.io);

        self.out.print("status  peers {d}", .{n_peers}) catch return;
        if (self.committee > 0) self.out.print("  committee {d}", .{self.committee}) catch return;

        if (self.store) |st| {
            const total = st.manifest.nRanges();
            const held = st.holdings.count();
            const pct: f64 = if (total == 0) 0 else 100.0 * @as(f64, @floatFromInt(held)) / @as(f64, @floatFromInt(total));
            self.out.print("  shards {d}/{d} ({d:.1}%)", .{ held, total, pct }) catch return;
        }

        // The condition that must never be silent: shards the swarm holds in
        // fewer copies than asked for. At zero live peers there is nothing to
        // say beyond "peers 0", which the line already carries.
        if (self.r_target > 1 and n_peers > 0) {
            if (self.store) |st| {
                const short = self.table.?.underReplicated(
                    self.gpa,
                    st.manifest.nRanges(),
                    self.r_target,
                    st.holdings.bits,
                ) catch null;
                if (short) |k| {
                    if (k > 0) self.out.print("  UNDER-REPLICATED {d} (want R={d})", .{ k, self.r_target }) catch return;
                }
            }
        }

        // Only meaningful once something has been served; printing 0.0% before
        // the first request reads as a fault rather than as "no data yet".
        const hit = self.gen.hitRate();
        if (hit > 0) self.out.print("  local-hit {d:.1}%", .{hit * 100.0}) catch return;

        self.out.print("  up {s}\n", .{fmtUptime(self.uptimeSecs())}) catch return;
        self.out.flush() catch return;
    }

    fn uptimeSecs(self: *Reporter) u64 {
        const now = stats.nowNs(self.io);
        if (now <= self.start_ns) return 0;
        return @intCast(@divTrunc(now - self.start_ns, std.time.ns_per_s));
    }
};

/// Compact uptime: seconds under a minute, then minutes, then hours, then
/// days. A wall-clock timestamp would be redundant next to whatever the
/// surrounding log system already stamps.
var uptime_buf: [24]u8 = undefined;
fn fmtUptime(secs: u64) []const u8 {
    if (secs < 60) return std.fmt.bufPrint(&uptime_buf, "{d}s", .{secs}) catch "?";
    if (secs < 3600) return std.fmt.bufPrint(&uptime_buf, "{d}m", .{secs / 60}) catch "?";
    if (secs < 86400) return std.fmt.bufPrint(&uptime_buf, "{d}h{d}m", .{ secs / 3600, (secs % 3600) / 60 }) catch "?";
    return std.fmt.bufPrint(&uptime_buf, "{d}d{d}h", .{ secs / 86400, (secs % 86400) / 3600 }) catch "?";
}

/// Thread body: report every `interval_ns` until stopped.
pub fn thread(r: *Reporter) void {
    while (r.running.load(.monotonic)) {
        // Sleep first, so the very first line lands after the node has
        // finished booting and has something worth reporting.
        Io.sleep(r.io, .{ .nanoseconds = @intCast(r.interval_ns) }, .awake) catch return;
        if (!r.running.load(.monotonic)) return;
        r.line();
    }
}

test "uptime formatting is compact and monotonic in unit size" {
    try std.testing.expectEqualStrings("0s", fmtUptime(0));
    try std.testing.expectEqualStrings("59s", fmtUptime(59));
    try std.testing.expectEqualStrings("1m", fmtUptime(60));
    try std.testing.expectEqualStrings("59m", fmtUptime(3599));
    try std.testing.expectEqualStrings("1h0m", fmtUptime(3600));
    try std.testing.expectEqualStrings("2h30m", fmtUptime(9000));
    try std.testing.expectEqualStrings("1d0h", fmtUptime(86400));
    try std.testing.expectEqualStrings("3d5h", fmtUptime(3 * 86400 + 5 * 3600));
}
