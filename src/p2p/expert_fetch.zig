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
const dns = @import("dns.zig");
const hashmod = @import("../core/hash.zig");
const stats_mod = @import("../core/stats.zig");
const wire = @import("wire.zig");

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
    /// SPEC.md query path: committee members are the allotted first-choice
    /// holders; `peers` is the mesh fallback (gossip-discovered or static).
    committee: []const sync.PeerAddr = &.{},
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
            // verify the local block before matmul (audit #5 P0-2); on failure
            // fall through to a peer fetch (the bit was cleared by the verify)
            if (self.store.readRangeVerified(id, self.scratch)) |data| {
                self.stats.local += 1;
                return data;
            } else |_| {}
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

        // SPEC.md order: committee first (round-robin spread), then the mesh
        const total = self.committee.len + self.peers.len;
        var attempt: usize = 0;
        while (attempt < total) : (attempt += 1) {
            const addr = if (attempt < self.committee.len)
                self.committee[(start + attempt) % self.committee.len]
            else
                self.peers[(attempt - self.committee.len)];
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

    /// One ExpertRequest/ExpertResponse exchange (SPEC.md wire messages v1).
    /// The request pins our manifest version; the server refuses others. The
    /// caller verifies the payload against the local manifest digest.
    fn fetchFromPeer(self: *Source, addr: sync.PeerAddr, id: usize, buf: []u8) !usize {
        const io = self.io;
        const gpa = self.gpa;
        const ip = try dns.resolve(io, addr.host, addr.port);
        const stream = try ip.connect(io, .{ .mode = .stream });
        defer stream.close(io);
        var rbuf: [1 << 16]u8 = undefined;
        var wbuf: [1024]u8 = undefined;
        var r = stream.reader(io, &rbuf);
        var w = stream.writer(io, &wbuf);

        const req = wire.ExpertRequest{
            .request_id = @intCast(id), // 1 request per connection: shard id doubles as request id
            .manifest_version = self.store.manifest.version,
            .shard_id = @intCast(id),
        };
        const body = try req.encodeBody(gpa);
        defer gpa.free(body);
        const frame = try wire.encodeFrame(gpa, .expert_request, body);
        defer gpa.free(frame);
        try wire.writeFrame(&w.interface, frame);

        const resp_raw = try wire.readFrameAlloc(gpa, &r.interface);
        defer gpa.free(resp_raw);
        const dec = try wire.decodeFrame(gpa, resp_raw);
        defer gpa.free(dec.body);
        if (dec.ty != .expert_response) return error.PeerCannotServe;
        const resp = try wire.ExpertResponse.parseBody(dec.body);
        if (resp.request_id != req.request_id or resp.shard_id != req.shard_id) return error.BadResponse;
        switch (resp.status) {
            .ok => {},
            .not_held => return error.PeerNotHeld,
            .version_mismatch => return error.PeerVersionMismatch,
            .busy => return error.PeerBusy,
            else => return error.PeerCannotServe,
        }
        if (resp.payload.len != buf.len) return error.BadShardLength;
        @memcpy(buf, resp.payload);
        return resp.payload.len;
    }
};
