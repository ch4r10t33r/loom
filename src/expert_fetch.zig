//! The distributed tier in the token loop (issue #3): expert shards a node
//! doesn't hold are fetched from peers *during inference*, digest-verified,
//! and persisted into the local sparse store.
//!
//! Tier order for a partial expert-aligned store: shards are either held
//! locally (read via pread — the OS page cache is the RAM tier) or held only
//! by peers (fetch). The two sources are disjoint by construction, so there is
//! no ordering decision to measure yet; a bandwidth probe becomes meaningful
//! once heat-replication puts the same shard both locally and remotely.
//!
//! Fetched shards are written back through Store.writeRange: verified against
//! the manifest digest before touching disk, marked held, and advertised on
//! the next gossip round — fetch-on-demand doubles as organic heat
//! replication (hot experts gain holders by being used).
//!
//! Failure policy matches the eager-churn decision: holder selection spreads
//! round-robin across peers that advertise the shard (client-side spreading,
//! no single-holder hotspot), and a failed peer falls through to the next —
//! the token stalls only if *no* reachable peer holds the shard.

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const weights = @import("weights.zig");
const sync = @import("sync.zig");
const hashmod = @import("hash.zig");
const stats_mod = @import("stats.zig");

pub const Stats = struct {
    local: u64 = 0, // served from held shards (disk/page cache)
    fetched: u64 = 0, // fetched from a peer this run
    fetch_bytes: u64 = 0,
    fetch_ns: i128 = 0,
    fetch_failures: u64 = 0, // per-peer attempt failures (retried elsewhere)
};

pub const Source = struct {
    gpa: std.mem.Allocator,
    io: Io,
    store: *weights.Store,
    peers: []const sync.PeerAddr,
    /// reusable buffer for single-threaded get(); prefetch threads allocate
    scratch: []u8,
    stats: Stats = .{},
    stats_mutex: Io.Mutex = .init,
    rr: usize = 0, // round-robin start for holder spreading

    pub fn init(gpa: std.mem.Allocator, io: Io, store: *weights.Store, peers: []const sync.PeerAddr) !Source {
        return .{
            .gpa = gpa,
            .io = io,
            .store = store,
            .peers = peers,
            .scratch = try gpa.alloc(u8, @intCast(store.manifest.maxShardLen())),
        };
    }

    pub fn deinit(self: *Source) void {
        self.gpa.free(self.scratch);
    }

    /// Materialize shard `id`. Local tier first; else fetch, verify, persist.
    /// Returns a slice of `self.scratch` (valid until the next get()).
    pub fn get(self: *Source, id: usize) ![]const u8 {
        if (self.store.holdings.has(id)) {
            self.stats.local += 1;
            return self.store.readRange(id, self.scratch);
        }
        const n = try self.fetchShard(id, self.scratch);
        return self.scratch[0..n];
    }

    /// Parallel warm-up for one MoE layer's selected experts: fetch every
    /// missing shard concurrently (one thread + connection each) so the
    /// per-layer miss latency is max(fetch), not sum(fetch). After this,
    /// get() for these ids hits the local tier.
    pub fn prefetch(self: *Source, ids: []const usize) void {
        var missing_buf: [64]usize = undefined;
        var n_missing: usize = 0;
        for (ids) |id| {
            if (!self.store.holdings.has(id) and n_missing < missing_buf.len) {
                missing_buf[n_missing] = id;
                n_missing += 1;
            }
        }
        if (n_missing == 0) return;
        if (n_missing == 1) {
            _ = self.fetchShard(missing_buf[0], self.scratch) catch {};
            return;
        }
        var threads_buf: [64]?std.Thread = [_]?std.Thread{null} ** 64;
        for (missing_buf[0..n_missing], 0..) |id, i| {
            threads_buf[i] = std.Thread.spawn(.{}, prefetchOne, .{ self, id }) catch null;
        }
        for (threads_buf[0..n_missing]) |t| {
            if (t) |th| th.join();
        }
    }

    fn prefetchOne(self: *Source, id: usize) void {
        const buf = self.gpa.alloc(u8, @intCast(self.store.manifest.rangeLen(id))) catch return;
        defer self.gpa.free(buf);
        _ = self.fetchShard(id, buf) catch {};
    }

    /// Fetch shard `id` from the first peer (starting round-robin) that holds
    /// it. Digest-verified + persisted via writeRange; concurrent calls for
    /// distinct shards write disjoint extents. Returns bytes filled in `buf`.
    fn fetchShard(self: *Source, id: usize, buf: []u8) !usize {
        const m = &self.store.manifest;
        const want_len: usize = @intCast(m.rangeLen(id));
        if (want_len > buf.len) return error.ShardTooLarge;

        const t0 = stats_mod.nowNs(self.io);
        const start = blk: {
            const s = self.rr;
            self.rr +%= 1; // benign race under concurrent prefetch: only spreads
            break :blk s;
        };

        var attempt: usize = 0;
        while (attempt < self.peers.len) : (attempt += 1) {
            const addr = self.peers[(start + attempt) % self.peers.len];
            const n = self.fetchFromPeer(addr, id, buf[0..want_len]) catch {
                self.bumpFailure();
                continue;
            };
            // verify before it touches disk; writeRange re-checks + persists
            self.store.writeRange(id, buf[0..n]) catch {
                self.bumpFailure();
                continue;
            };
            self.stats_mutex.lockUncancelable(self.io);
            self.stats.fetched += 1;
            self.stats.fetch_bytes += n;
            self.stats.fetch_ns += stats_mod.nowNs(self.io) - t0;
            self.stats_mutex.unlock(self.io);
            return n;
        }
        return error.NoHolderReachable;
    }

    fn bumpFailure(self: *Source) void {
        self.stats_mutex.lockUncancelable(self.io);
        self.stats.fetch_failures += 1;
        self.stats_mutex.unlock(self.io);
    }

    fn fetchFromPeer(self: *Source, addr: sync.PeerAddr, id: usize, buf: []u8) !usize {
        const io = self.io;
        const ip = try net.IpAddress.parse(addr.host, addr.port);
        const stream = try ip.connect(io, .{ .mode = .stream });
        defer stream.close(io);
        var rbuf: [1 << 16]u8 = undefined;
        var wbuf: [1024]u8 = undefined;
        var r = stream.reader(io, &rbuf);
        var w = stream.writer(io, &wbuf);

        try w.interface.print("GETR {d}\n", .{id});
        try w.interface.flush();
        const hline = std.mem.trimEnd(u8, try r.interface.takeDelimiterInclusive('\n'), "\r\n");
        if (!std.mem.startsWith(u8, hline, "DATA ")) return error.PeerCannotServe;
        var len: usize = 0;
        var it = std.mem.splitScalar(u8, hline, ' ');
        while (it.next()) |tok| {
            if (std.mem.startsWith(u8, tok, "len=")) {
                len = try std.fmt.parseInt(usize, tok[4..], 10);
            }
        }
        if (len != buf.len) return error.BadShardLength;
        try r.interface.readSliceAll(buf);
        return len;
    }
};
