//! Opt-in alpha telemetry (--report-metrics). Strictly numeric operational
//! data: never prompts, never generated text, never chunk contents. The full
//! field list is documented in docs/ALPHA.md and is the contract with alpha
//! testers; adding a field means updating that document in the same change.
//!
//! Transport is the existing p2p line protocol: `METRICS <json>` to the
//! bootstrap peer, answered with OK (persisted) or ERR no_ingest (the peer
//! does not collect). One line a minute; a lost report is never retried,
//! because a gap in telemetry is not worth a reconnect storm.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const sync = @import("../p2p/sync.zig");
const weights = @import("../p2p/weights.zig");
const peers = @import("../p2p/peers.zig");
const stats = @import("../core/stats.zig");

/// Rolling generation aggregates, updated by the RPC and OpenAI surfaces
/// after each real generation. One mutex; two floats and a counter.
pub const Metrics = struct {
    io: Io,
    mu: Io.Mutex = .init,
    gens: u64 = 0,
    tok_s_sum: f64 = 0,
    hit_sum: f64 = 0,
    last_tok_s: f64 = 0,
    inflight: u32 = 0,
    inflight_since_ns: i128 = 0,
    // DSD draft-verify (whitepaper roadmap 6)
    draft_rounds: u64 = 0,
    draft_drafted: u64 = 0,
    draft_accepted: u64 = 0,
    draft_last_gamma: u64 = 0,
    draft_bails: u64 = 0,

    pub fn recordGen(self: *Metrics, tok_s: f64, hit_rate: f64) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        self.gens += 1;
        self.tok_s_sum += tok_s;
        self.hit_sum += hit_rate;
        self.last_tok_s = tok_s;
    }

    pub fn recordDraft(self: *Metrics, rounds: usize, drafted: usize, accepted: usize, gamma: usize, bailed: bool) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        self.draft_rounds += rounds;
        self.draft_drafted += drafted;
        self.draft_accepted += accepted;
        self.draft_last_gamma = gamma;
        if (bailed) self.draft_bails += 1;
    }

    /// For the status line: how many generations, and the latest speed.
    pub fn lastGen(self: *Metrics) struct { gens: u64, tok_s: f64 } {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        return .{ .gens = self.gens, .tok_s = self.last_tok_s };
    }

    pub fn beginGen(self: *Metrics, now_ns: i128) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        self.inflight += 1;
        if (self.inflight == 1) self.inflight_since_ns = now_ns;
    }

    pub fn endGen(self: *Metrics) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        if (self.inflight > 0) self.inflight -= 1;
    }

    /// Seconds the oldest in-flight generation has been running, or null when
    /// idle. The status line uses this so a busy node never looks dead.
    pub fn inflightSecs(self: *Metrics, now_ns: i128) ?u64 {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        if (self.inflight == 0) return null;
        return @intCast(@divTrunc(now_ns - self.inflight_since_ns, 1_000_000_000));
    }

    fn snapshot(self: *Metrics) struct {
        gens: u64,
        tok_s_avg: f64,
        hit_avg: f64,
        draft_rounds: u64,
        draft_drafted: u64,
        draft_accepted: u64,
        draft_gamma: u64,
        draft_bails: u64,
    } {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const n: f64 = @floatFromInt(@max(self.gens, 1));
        return .{
            .gens = self.gens,
            .tok_s_avg = self.tok_s_sum / n,
            .hit_avg = self.hit_sum / n,
            .draft_rounds = self.draft_rounds,
            .draft_drafted = self.draft_drafted,
            .draft_accepted = self.draft_accepted,
            .draft_gamma = self.draft_last_gamma,
            .draft_bails = self.draft_bails,
        };
    }
};

pub const ReporterCtx = struct {
    gpa: std.mem.Allocator,
    io: Io,
    metrics: *Metrics,
    store: ?*weights.Store,
    table: *peers.Table,
    target: sync.PeerAddr,
    network_id: u64,
    version: []const u8,
    hold_fraction: f64,
    /// Random per-boot id; no stable identity, no hostname, no address.
    boot_id: u64,
    started_ns: i128,
    interval_s: u64 = 60,
};

/// One report line. Returns the JSON length written into `buf`.
fn buildReport(ctx: *ReporterCtx, buf: []u8) ![]const u8 {
    const snap = ctx.metrics.snapshot();
    const up_s: u64 = @intCast(@divTrunc(stats.nowNs(ctx.io) - ctx.started_ns, 1_000_000_000));
    var held: usize = 0;
    var total: usize = 0;
    if (ctx.store) |st| {
        held = st.holdings.count();
        total = st.manifest.nRanges();
    }
    const n_peers = ctx.table.count();
    return std.fmt.bufPrint(buf, "{{\"id\":\"{x:0>16}\",\"v\":\"{s}\",\"os\":\"{s}\",\"arch\":\"{s}\"," ++
        "\"up_s\":{d},\"net\":{d},\"hold_target\":{d:.3},\"held\":{d},\"total\":{d}," ++
        "\"peers\":{d},\"gens\":{d},\"tok_s_avg\":{d:.3},\"hit_avg\":{d:.4}," ++
        "\"draft_rounds\":{d},\"draft_drafted\":{d},\"draft_accepted\":{d},\"draft_gamma\":{d},\"draft_bails\":{d}}}", .{
        ctx.boot_id,
        ctx.version,
        @tagName(builtin.os.tag),
        @tagName(builtin.cpu.arch),
        up_s,
        ctx.network_id,
        ctx.hold_fraction,
        held,
        total,
        n_peers,
        snap.gens,
        snap.tok_s_avg,
        snap.hit_avg,
        snap.draft_rounds,
        snap.draft_drafted,
        snap.draft_accepted,
        snap.draft_gamma,
        snap.draft_bails,
    });
}

/// Detached loop. Failures are ignored until the next tick.
pub fn reporterLoop(ctx: *ReporterCtx) void {
    while (true) {
        Io.sleep(ctx.io, .{ .nanoseconds = @intCast(ctx.interval_s * 1_000_000_000) }, .awake) catch return;
        var buf: [1024]u8 = undefined;
        const line = buildReport(ctx, &buf) catch continue;
        sendOnce(ctx, line) catch continue;
    }
}

fn sendOnce(ctx: *ReporterCtx, json: []const u8) !void {
    const peer = try sync.Peer.connect(ctx.gpa, ctx.io, ctx.target);
    defer peer.close(ctx.gpa);
    try peer.send("METRICS {s}\n", .{json});
    _ = try peer.recvLine(); // OK or ERR no_ingest; either way, done
}

test "report json stays under the wire cap and parses" {
    var threaded: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var m = Metrics{ .io = io };
    m.recordGen(12.5, 0.97);
    var table = peers.Table.init(std.testing.allocator, io, "127.0.0.1:1");
    defer table.deinit();
    var ctx = ReporterCtx{
        .gpa = std.testing.allocator,
        .io = io,
        .metrics = &m,
        .store = null,
        .table = &table,
        .target = undefined,
        .network_id = 1337,
        .version = "0.0.0-test",
        .hold_fraction = 0.2,
        .boot_id = 0xdeadbeef,
        .started_ns = stats.nowNs(io),
    };
    var buf: [1024]u8 = undefined;
    const line = try buildReport(&ctx, &buf);
    try std.testing.expect(line.len < 512);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, line, .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(i64, 1337), parsed.value.object.get("net").?.integer);
    try std.testing.expectEqual(@as(i64, 1), parsed.value.object.get("gens").?.integer);
}
