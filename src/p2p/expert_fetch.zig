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
const sockopt = @import("../core/sockopt.zig");
const hashmod = @import("../core/hash.zig");
const stats_mod = @import("../core/stats.zig");
const wire = @import("wire.zig");
const expert_cache = @import("../engine/expert_cache.zig");

pub const Stats = struct {
    mapped: u64 = 0, // served zero-copy from the mmap'd store, no read at all
    ram: u64 = 0, // served from the verified-block cache, no read and no hash
    local: u64 = 0, // served from held shards (disk/page cache)
    fetched: u64 = 0, // fetched from a peer this run
    fetch_bytes: u64 = 0,
    fetch_ns: i128 = 0,
    fetch_failures: u64 = 0, // per-peer attempt failures (retried elsewhere)
    pilot_pred: u64 = 0, // experts predicted one layer ahead (PILOT)
    pilot_hit: u64 = 0, // predictions the real router then confirmed
};

/// Persistent bounded fetch workers (issue #170): jobs are shard ids, the
/// queue applies backpressure, concurrent requests for one shard collapse
/// into one fetch, and the per-layer path never touches Thread.spawn. The
/// workers park on a condition variable, so an idle pool costs nothing.
const Pool = struct {
    const WORKERS = 6;
    const QUEUE_CAP = 128;
    const NO_ID = std.math.maxInt(usize);

    mutex: Io.Mutex = .init,
    /// wakes workers when work arrives or stop flips
    work_cond: Io.Condition = .init,
    /// wakes waiters when a job leaves the pending set
    done_cond: Io.Condition = .init,
    queue: [QUEUE_CAP]usize = undefined,
    head: usize = 0,
    len: usize = 0,
    active: [WORKERS]usize = @splat(NO_ID),
    stop: bool = false,
    started: bool = false,
    threads: [WORKERS]?std.Thread = @splat(null),

    /// Under mutex: is `id` queued or being fetched right now?
    fn pendingLocked(p: *Pool, id: usize) bool {
        var i: usize = 0;
        while (i < p.len) : (i += 1) {
            if (p.queue[(p.head + i) % QUEUE_CAP] == id) return true;
        }
        for (p.active) |a| if (a == id) return true;
        return false;
    }

    /// Enqueue unless duplicate or full. Returns whether the id is now
    /// pending (true also for duplicates -- the caller's wait still works).
    fn submit(p: *Pool, io: Io, id: usize) bool {
        p.mutex.lockUncancelable(io);
        defer p.mutex.unlock(io);
        if (p.pendingLocked(id)) return true;
        if (p.len == QUEUE_CAP) return false;
        p.queue[(p.head + p.len) % QUEUE_CAP] = id;
        p.len += 1;
        p.work_cond.signal(io);
        return true;
    }
};

pub const Source = struct {
    pool: Pool = .{},
    gpa: std.mem.Allocator,
    io: Io,
    store: *weights.Store,
    /// SPEC.md query path: committee members are the allotted first-choice
    /// holders; `peers` is the mesh fallback (gossip-discovered or static).
    committee: []const sync.PeerAddr = &.{},
    peers: []const sync.PeerAddr,
    /// reusable buffer for single-threaded get(); prefetch threads allocate
    scratch: []u8,
    /// The local RAM tier. Without it every routed expert is re-read from
    /// disk *and* re-hashed on every token: a 27-layer model activating 7
    /// experts a layer does ~190 of those per token, ~950 MB hashed, which
    /// measured at ~0.9 s/token and swamped everything else in the forward
    /// pass. Verifying on first read still catches a poisoned block on disk
    /// (audit #5 P0-2); re-verifying a block already in our own address space
    /// on every subsequent access buys nothing.
    ///
    /// This is the tier the design has always specified as the top of the
    /// fetch order -- the loom-format engine had it, the GGUF path did not.
    cache: ?expert_cache.Lru = null,
    /// Store mutation counter last seen. A repair or peer fetch that rewrites
    /// a shard invalidates whatever we cached for it, so the whole cache is
    /// dropped when this moves. Coarse, and it should be rare.
    cache_seq: u64 = 0,
    stats: Stats = .{},
    stats_mutex: Io.Mutex = .init,
    rr: usize = 0, // round-robin start for holder spreading

    pub fn init(gpa: std.mem.Allocator, io: Io, store: *weights.Store, peers: []const sync.PeerAddr) !Source {
        return initCached(gpa, io, store, peers, 0);
    }

    /// `cache_bytes` sizes the RAM tier; 0 disables it. Slots are one
    /// maximum-length shard each, since a shard has to land contiguously to
    /// be handed to a matmul.
    pub fn initCached(
        gpa: std.mem.Allocator,
        io: Io,
        store: *weights.Store,
        peers: []const sync.PeerAddr,
        cache_bytes: usize,
    ) !Source {
        // Sized by the largest *expert* shard, not the largest shard: the
        // resident chunks are several times bigger and only ever read at
        // startup, so letting them set the slot size throws away most of the
        // budget.
        const slot = @as(usize, @intCast(store.manifest.maxExpertShardLen()));
        const slots = if (slot == 0) 0 else cache_bytes / slot;
        return .{
            .gpa = gpa,
            .io = io,
            .store = store,
            .peers = peers,
            .scratch = try gpa.alloc(u8, @intCast(store.manifest.maxShardLen())),
            .cache = if (slots == 0) null else try expert_cache.Lru.init(gpa, slots, slot),
            .cache_seq = store.holdingsSeq(),
        };
    }

    pub fn deinit(self: *Source) void {
        {
            self.pool.mutex.lockUncancelable(self.io);
            self.pool.stop = true;
            self.pool.mutex.unlock(self.io);
            self.pool.work_cond.broadcast(self.io);
        }
        for (self.pool.threads) |t| if (t) |th| th.join();
        self.gpa.free(self.scratch);
        if (self.cache) |*c| c.deinit(self.gpa);
    }

    /// Spawn the fetch workers on first use, so a Source that never prefetches
    /// (unit tests, local-only runs) never owns a thread.
    fn ensurePool(self: *Source) void {
        self.pool.mutex.lockUncancelable(self.io);
        defer self.pool.mutex.unlock(self.io);
        if (self.pool.started) return;
        self.pool.started = true;
        for (&self.pool.threads, 0..) |*t, i| {
            t.* = std.Thread.spawn(.{}, poolWorker, .{ self, i }) catch null;
        }
    }

    fn poolWorker(self: *Source, slot: usize) void {
        const buf = self.gpa.alloc(u8, @intCast(self.store.manifest.maxShardLen())) catch return;
        defer self.gpa.free(buf);
        const p = &self.pool;
        const io = self.io;
        while (true) {
            p.mutex.lockUncancelable(io);
            while (p.len == 0 and !p.stop) p.work_cond.waitUncancelable(io, &p.mutex);
            if (p.stop) {
                p.mutex.unlock(io);
                return;
            }
            const id = p.queue[p.head];
            p.head = (p.head + 1) % Pool.QUEUE_CAP;
            p.len -= 1;
            p.active[slot] = id;
            p.mutex.unlock(io);

            if (!self.store.holdings.has(id)) {
                _ = self.fetchShard(id, buf) catch {};
            }

            p.mutex.lockUncancelable(io);
            p.active[slot] = Pool.NO_ID;
            p.mutex.unlock(io);
            p.done_cond.broadcast(io);
        }
    }

    /// Block until none of `ids` is pending in the pool. "Pending" means
    /// queued or actively fetching; a job that failed its fetch completes the
    /// wait too -- the caller's get() retries (or errors) exactly as before.
    fn waitPool(self: *Source, ids: []const usize) void {
        const p = &self.pool;
        p.mutex.lockUncancelable(self.io);
        defer p.mutex.unlock(self.io);
        outer: while (true) {
            for (ids) |id| {
                if (p.pendingLocked(id)) {
                    p.done_cond.waitUncancelable(self.io, &p.mutex);
                    continue :outer;
                }
            }
            return;
        }
    }

    /// Whether pointers returned by `get` stay valid across later `get` calls.
    ///
    /// They do when a shard lands in the RAM cache or is read straight out of
    /// the mapping, and they do not when it lands in the single shared
    /// `scratch` buffer — there the next call overwrites the previous result.
    /// A caller that wants several experts live at once (recording a whole MoE
    /// layer into one command buffer) has to know which it is getting.
    pub fn stablePointers(self: *const Source, needed: usize) bool {
        if (self.store.map != null) return true;
        // The LRU is only stable for the caller if it can hold every expert the
        // caller is about to ask for: a cache smaller than the layer's
        // selection would evict the first expert to make room for the last.
        const c = self.cache orelse return false;
        return c.capacity >= needed;
    }

    /// Slots the RAM tier holds, for the startup banner.
    pub fn cacheSlots(self: *const Source) usize {
        return if (self.cache) |c| c.capacity else 0;
    }

    /// The three parts of an expert shard -- gate, up, down -- each as its own
    /// slice. A shard is three separate file extents, so this is the shape it
    /// naturally has; only the old copying path ever concatenated them.
    pub const Parts = struct { gate: []const u8, up: []const u8, down: []const u8 };

    /// Zero-copy shard access: pointers straight into the mapped store, valid
    /// as long as the store stays mapped and the shard stays held.
    ///
    /// Returns null when the store is not mapped, the shard is not held, it
    /// fails its digest, or it does not have exactly three extents -- in every
    /// one of those cases the caller falls back to `get`. This is the whole
    /// point of the exercise: a copy into a heap cache is a second resident
    /// copy of bytes the page cache already holds, and on a machine where the
    /// two do not both fit it costs far more than the copy itself.
    pub fn getMapped(self: *Source, id: usize) ?Parts {
        if (self.store.extentCount(id) != 3) return null;
        const g = self.store.extentSlice(id, 0) orelse return null;
        const u = self.store.extentSlice(id, 1) orelse return null;
        const d = self.store.extentSlice(id, 2) orelse return null;
        self.stats.mapped += 1;
        self.store.touch(id);
        return .{ .gate = g, .up = u, .down = d };
    }

    /// Materialize shard `id`. Local tier first; else fetch, verify, persist.
    /// Returns a slice of `self.scratch` (valid until the next get()).
    pub fn get(self: *Source, id: usize) ![]const u8 {
        // Cache slots are sized for an expert shard, and the resident chunks
        // are larger -- 16 MB against ~6 MB here. Reading one through the cache
        // wrote past the end of its slot: with safety off that is a silent heap
        // overflow, which is what it had been doing on every startup, since the
        // resident gate reads exactly these shards.
        if (self.cache) |*c| use_cache: {
            if (self.store.manifest.rangeLen(id) > c.block_bytes) break :use_cache;
            const seq = self.store.holdingsSeq();
            if (seq != self.cache_seq) {
                c.clear();
                self.cache_seq = seq;
            } else if (c.find(id)) |blk| {
                self.stats.ram += 1;
                self.store.touch(id);
                return blk[0..@intCast(self.store.manifest.rangeLen(id))];
            }
            const want: usize = @intCast(self.store.manifest.rangeLen(id));
            if (self.store.holdings.has(id)) {
                const slot = try c.reserve(id);
                if (self.store.readRangeVerified(id, slot[1][0..want])) |data| {
                    self.stats.local += 1;
                    self.store.touch(id);
                    return data;
                } else |_| {
                    // publish-after-verify: a poisoned block must not become a
                    // cache hit later (audit #5 P0-3)
                    c.abort(id);
                }
            }
            const n = try self.fetchShard(id, self.scratch);
            const slot = try c.reserve(id);
            @memcpy(slot[1][0..n], self.scratch[0..n]);
            return slot[1][0..n];
        }
        if (self.store.holdings.has(id)) {
            // verify the local block before matmul (audit #5 P0-2); on failure
            // fall through to a peer fetch (the bit was cleared by the verify)
            if (self.store.readRangeVerified(id, self.scratch)) |data| {
                self.stats.local += 1;
                self.store.touch(id);
                return data;
            } else |_| {}
        }
        const n = try self.fetchShard(id, self.scratch);
        return self.scratch[0..n];
    }

    /// Local tiers only -- RAM cache or held shards; never the network. The
    /// draft-local path uses this: the caller skips a missing expert entirely,
    /// so a cold node can draft candidate tokens without paying a single peer
    /// round trip. Same verify-before-use rules as get(); a shard that fails
    /// its digest reads as absent rather than fetched.
    pub fn getLocal(self: *Source, id: usize) ?[]const u8 {
        const want: usize = @intCast(self.store.manifest.rangeLen(id));
        if (self.cache) |*c| {
            if (want <= c.block_bytes) {
                const seq = self.store.holdingsSeq();
                if (seq != self.cache_seq) {
                    c.clear();
                    self.cache_seq = seq;
                } else if (c.find(id)) |blk| {
                    self.stats.ram += 1;
                    self.store.touch(id);
                    return blk[0..want];
                }
                if (!self.store.holdings.has(id)) return null;
                const slot = c.reserve(id) catch return null;
                if (self.store.readRangeVerified(id, slot[1][0..want])) |data| {
                    self.stats.local += 1;
                    self.store.touch(id);
                    return data;
                } else |_| {
                    c.abort(id);
                    return null;
                }
            }
        }
        if (!self.store.holdings.has(id)) return null;
        if (self.store.readRangeVerified(id, self.scratch)) |data| {
            self.stats.local += 1;
            self.store.touch(id);
            return data;
        } else |_| return null;
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
        // Submit to the persistent pool and wait; per-layer miss latency stays
        // max(fetch) up to the worker bound, without a single Thread.spawn on
        // the token path (issue #170). A full queue degrades to fetching the
        // overflow inline, which is backpressure rather than lost work.
        self.ensurePool();
        for (missing_buf[0..n_missing]) |id| {
            if (!self.pool.submit(self.io, id)) _ = self.fetchShard(id, self.scratch) catch {};
        }
        self.waitPool(missing_buf[0..n_missing]);
    }

    /// PILOT prefetch: fire-and-forget fetch of predicted next-layer experts
    /// while the current layer computes. Bounded and lossy by design -- when
    /// the pool is full the prediction is simply dropped, because a stalled
    /// main path would cost more than a missed prefetch. Fetched shards
    /// persist to the store, so even a mispredicted fetch warms the node.
    pub fn prefetchAsync(self: *Source, ids: []const usize) void {
        self.ensurePool();
        for (ids) |id| {
            if (self.store.holdings.has(id)) continue;
            // lossy by design: a full queue drops the prediction rather than
            // stalling the main path (same policy the thread cap enforced)
            _ = self.pool.submit(self.io, id);
        }
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
        const dl = sockopt.trackPeer(io, stream);
        defer sockopt.untrack(io, dl);
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

test "get: a shard larger than a cache slot bypasses the cache instead of overflowing it" {
    const gpa = std.testing.allocator;
    var thr: std.Io.Threaded = .init(gpa, .{});
    defer thr.deinit();
    const io = thr.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dir_buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});

    // One resident chunk of 4 KB and one expert shard of 1 KB: the cache sizes
    // its slots by the expert, so the resident chunk is four slots wide -- the
    // shape that was writing past the end of the slab.
    const resident_len: u64 = 4096;
    const expert_len: u64 = 1024;
    var blob = try gpa.alloc(u8, @intCast(resident_len + expert_len));
    defer gpa.free(blob);
    for (blob, 0..) |*b, k| b.* = @truncate(k *% 7);

    // Heap-allocated: the store takes ownership of the manifest and frees it.
    const digests = try gpa.dupe(hashmod.Digest, &.{
        hashmod.hashBlock(blob[0..@intCast(resident_len)]),
        hashmod.hashBlock(blob[@intCast(resident_len)..]),
    });
    const extents = try gpa.dupe(weights.Extent, &.{
        .{ .offset = 0, .len = resident_len },
        .{ .offset = resident_len, .len = expert_len },
    });
    const starts = try gpa.dupe(u32, &.{ 0, 1, 2 });
    const total = resident_len + expert_len;
    const version = try weights.computeVersion(gpa, .expert, total, 0, digests, extents, starts);
    const manifest = weights.Manifest{
        .mode = .expert,
        .version = version,
        .file_size = total,
        .range_size = 0,
        .n_resident = 1,
        .digests = digests,
        .extents = extents,
        .extent_start = starts,
    };

    var wanted = try weights.Holdings.initEmpty(gpa, 2);
    wanted.set(0);
    wanted.set(1);
    var store = try weights.createFromManifest(gpa, io, dir, manifest, wanted);
    defer store.deinit();
    try store.writeRange(0, blob[0..@intCast(resident_len)]);
    try store.writeRange(1, blob[@intCast(resident_len)..]);

    // Budget for exactly two expert-sized slots.
    var src = try Source.initCached(gpa, io, &store, &.{}, 2 * @as(usize, @intCast(expert_len)));
    defer src.deinit();
    try std.testing.expect(src.cache != null);

    const got = try src.get(0);
    try std.testing.expectEqualSlices(u8, blob[0..@intCast(resident_len)], got);

    // The expert still goes through the cache: the bypass is by size, not a
    // blanket disable.
    _ = try src.get(1);
    const before = src.stats.ram;
    _ = try src.get(1);
    try std.testing.expect(src.stats.ram > before);
}

test "fetch pool: prefetch of unfetchable shards completes, dedups, and shuts down clean" {
    // No peers are configured, so every pooled fetch fails -- the point is
    // the machinery: waiters must not deadlock on failed jobs, duplicate
    // submissions must collapse, and deinit must join the workers.
    const gpa = std.testing.allocator;
    var thr: std.Io.Threaded = .init(gpa, .{});
    defer thr.deinit();
    const io = thr.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dir_buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});

    const n_shards = 6;
    const shard_len: u64 = 512;
    var blob = try gpa.alloc(u8, @intCast(n_shards * shard_len));
    defer gpa.free(blob);
    for (blob, 0..) |*b, k| b.* = @truncate(k *% 13);

    var digests = try gpa.alloc(hashmod.Digest, n_shards);
    var extents = try gpa.alloc(weights.Extent, n_shards);
    var starts = try gpa.alloc(u32, n_shards + 1);
    for (0..n_shards) |i| {
        digests[i] = hashmod.hashBlock(blob[i * @as(usize, @intCast(shard_len)) ..][0..@intCast(shard_len)]);
        extents[i] = .{ .offset = @as(u64, @intCast(i)) * shard_len, .len = shard_len };
        starts[i] = @intCast(i);
    }
    starts[n_shards] = n_shards;
    const total = n_shards * shard_len;
    const version = try weights.computeVersion(gpa, .expert, total, 1, digests, extents, starts);
    const manifest = weights.Manifest{
        .mode = .expert,
        .version = version,
        .file_size = total,
        .range_size = 0,
        .n_resident = 1,
        .digests = digests,
        .extents = extents,
        .extent_start = starts,
    };
    var wanted = try weights.Holdings.initEmpty(gpa, n_shards);
    for (0..n_shards) |i| wanted.set(i);
    var store = try weights.createFromManifest(gpa, io, dir, manifest, wanted);
    defer store.deinit();
    // hold only shard 0; the rest are misses with nowhere to fetch from
    try store.writeRange(0, blob[0..@intCast(shard_len)]);

    var src = try Source.init(gpa, io, &store, &.{});
    defer src.deinit();

    const ids = [_]usize{ 1, 2, 3, 4, 5 };
    src.prefetch(&ids); // must return despite every fetch failing
    src.prefetch(&ids); // second round: same ids again, still no deadlock
    src.prefetchAsync(&ids); // lossy path exercises submit-without-wait
    src.prefetch(&.{ 1, 1, 1, 2 }); // duplicates collapse

    try std.testing.expect(!store.holdings.has(1));
    try std.testing.expect(store.holdings.has(0));
}
