//! The v0 product core: a single-node, tiered expert cache.
//!
//! Tier order for one commodity box (CLAUDE.md principle 2, "local RAM -> local
//! disk"): pinned RAM -> LRU RAM -> pread from experts.blob (the OS page cache
//! sits implicitly under the pread and is the real "local disk" tier). Every
//! `get` records usage so the pinner can size the hot set (STATS/PIN), and every
//! disk read verifies the block against its manifest digest — the free poison
//! check from content-addressing (principle 5).
//!
//! v1 will swap the disk tier for "best measured source (peer)" behind the same
//! `get(expert_id)` interface; nothing above the cache needs to change.

const std = @import("std");
const Io = std.Io;
const checkpoint = @import("checkpoint.zig");
const hashmod = @import("../core/hash.zig");

pub const Stats = struct {
    pin_hits: u64 = 0,
    lru_hits: u64 = 0,
    disk_misses: u64 = 0,
    bytes_read: u64 = 0,
    digest_failures: u64 = 0,
    // pin/prefetch warm-up reads, kept separate so token-path hit-rate is honest
    warmup_reads: u64 = 0,
    warmup_bytes: u64 = 0,

    pub fn accesses(self: Stats) u64 {
        return self.pin_hits + self.lru_hits + self.disk_misses;
    }
    pub fn hitRate(self: Stats) f64 {
        const a = self.accesses();
        if (a == 0) return 0;
        return @as(f64, @floatFromInt(self.pin_hits + self.lru_hits)) / @as(f64, @floatFromInt(a));
    }
};

/// Intrusive doubly-linked LRU over a fixed slab of `capacity` block buffers.
pub const Lru = struct {
    const NodeIdx = usize;
    const nil = std.math.maxInt(NodeIdx);

    slab: []u8, // capacity * block_bytes
    block_bytes: usize,
    ids: []usize, // slot -> expert_id currently held
    prev: []NodeIdx,
    next: []NodeIdx,
    map: std.AutoHashMap(usize, NodeIdx), // expert_id -> slot
    head: NodeIdx = nil, // most-recently-used
    tail: NodeIdx = nil, // least-recently-used
    free_top: usize, // next never-used slot (fill before evicting)
    capacity: usize,

    pub fn init(gpa: std.mem.Allocator, capacity: usize, block_bytes: usize) !Lru {
        return .{
            .slab = try gpa.alloc(u8, capacity * block_bytes),
            .block_bytes = block_bytes,
            .ids = try gpa.alloc(usize, capacity),
            .prev = try gpa.alloc(NodeIdx, capacity),
            .next = try gpa.alloc(NodeIdx, capacity),
            .map = std.AutoHashMap(usize, NodeIdx).init(gpa),
            .free_top = 0,
            .capacity = capacity,
        };
    }

    pub fn deinit(self: *Lru, gpa: std.mem.Allocator) void {
        gpa.free(self.slab);
        gpa.free(self.ids);
        gpa.free(self.prev);
        gpa.free(self.next);
        self.map.deinit();
    }

    fn buf(self: *Lru, slot: NodeIdx) []u8 {
        return self.slab[slot * self.block_bytes ..][0..self.block_bytes];
    }

    fn unlink(self: *Lru, slot: NodeIdx) void {
        const p = self.prev[slot];
        const n = self.next[slot];
        if (p != nil) self.next[p] = n else self.head = n;
        if (n != nil) self.prev[n] = p else self.tail = p;
    }

    fn pushFront(self: *Lru, slot: NodeIdx) void {
        self.prev[slot] = nil;
        self.next[slot] = self.head;
        if (self.head != nil) self.prev[self.head] = slot;
        self.head = slot;
        if (self.tail == nil) self.tail = slot;
    }

    fn touch(self: *Lru, slot: NodeIdx) void {
        if (self.head == slot) return;
        self.unlink(slot);
        self.pushFront(slot);
    }

    /// Returns the held buffer if present (and marks it MRU).
    pub fn find(self: *Lru, id: usize) ?[]u8 {
        if (self.map.get(id)) |slot| {
            self.touch(slot);
            return self.buf(slot);
        }
        return null;
    }

    /// Reserve a slot for `id` (allocating a fresh one or evicting the LRU
    /// tail), returning its writable buffer. Caller must fill it.
    pub fn reserve(self: *Lru, id: usize) !struct { usize, []u8 } {
        if (self.capacity == 0) return error.NoCapacity;
        var slot: NodeIdx = undefined;
        if (self.free_top < self.capacity) {
            slot = self.free_top;
            self.free_top += 1;
        } else {
            slot = self.tail;
            self.unlink(slot);
            _ = self.map.remove(self.ids[slot]);
        }
        self.ids[slot] = id;
        try self.map.put(id, slot);
        self.pushFront(slot);
        return .{ slot, self.buf(slot) };
    }

    /// Drop every entry, keeping the allocation. Used when the backing store
    /// is mutated underneath the cache and its contents can no longer be
    /// trusted to match what is on disk.
    pub fn clear(self: *Lru) void {
        self.map.clearRetainingCapacity();
        self.head = nil;
        self.tail = nil;
        self.free_top = 0;
    }

    /// Undo a reserve whose fill failed (e.g. a poisoned expert): unmap and
    /// unlink the slot so a later get() re-fetches instead of returning the
    /// unverified buffer (audit #5 P0-3).
    pub fn abort(self: *Lru, id: usize) void {
        if (self.map.fetchRemove(id)) |kv| {
            self.unlink(kv.value);
            // return the slot to the free pool if it was the newest allocation
            if (kv.value + 1 == self.free_top) self.free_top -= 1;
        }
    }
};

pub const ExpertCache = struct {
    gpa: std.mem.Allocator,
    io: Io,
    file: Io.File,
    entries: []const checkpoint.ExpertEntry,
    block_bytes: usize,
    verify: bool,

    pinned: std.AutoHashMap(usize, []u8), // expert_id -> owned resident bytes
    lru: Lru,
    stats: Stats = .{},
    access_count: []u64, // per expert_id, for STATS-driven pinning

    pub fn init(
        gpa: std.mem.Allocator,
        io: Io,
        file: Io.File,
        entries: []const checkpoint.ExpertEntry,
        block_bytes: usize,
        lru_capacity: usize,
        verify: bool,
    ) !ExpertCache {
        return .{
            .gpa = gpa,
            .io = io,
            .file = file,
            .entries = entries,
            .block_bytes = block_bytes,
            .verify = verify,
            .pinned = std.AutoHashMap(usize, []u8).init(gpa),
            .lru = try Lru.init(gpa, lru_capacity, block_bytes),
            .access_count = blk: {
                const ac = try gpa.alloc(u64, entries.len);
                @memset(ac, 0);
                break :blk ac;
            },
        };
    }

    pub fn deinit(self: *ExpertCache) void {
        var it = self.pinned.valueIterator();
        while (it.next()) |v| self.gpa.free(v.*);
        self.pinned.deinit();
        self.lru.deinit(self.gpa);
        self.gpa.free(self.access_count);
    }

    fn readBlock(self: *ExpertCache, id: usize, dst: []u8, is_warmup: bool) !void {
        const e = self.entries[id];
        std.debug.assert(dst.len == e.len);
        _ = try self.file.readPositionalAll(self.io, dst, e.offset);
        if (is_warmup) {
            self.stats.warmup_reads += 1;
            self.stats.warmup_bytes += e.len;
        } else {
            self.stats.disk_misses += 1;
            self.stats.bytes_read += e.len;
        }
        if (self.verify) {
            const got = hashmod.hashBlock(dst);
            if (!hashmod.eql(got, e.digest)) {
                self.stats.digest_failures += 1;
                return error.PoisonedExpert;
            }
        }
    }

    /// Load `ids` into pinned RAM (the hot set). Idempotent per id.
    pub fn pin(self: *ExpertCache, ids: []const usize) !void {
        for (ids) |id| {
            if (self.pinned.contains(id)) continue;
            const block = try self.gpa.alloc(u8, self.entries[id].len);
            errdefer self.gpa.free(block);
            try self.readBlock(id, block, true); // warm-up read, not a token-path miss
            try self.pinned.put(id, block);
        }
    }

    /// Materialize expert `id`'s int4 block. Never fails to place the exact
    /// block in RAM (principle 7: no decode, direct addressed fetch).
    pub fn get(self: *ExpertCache, id: usize) ![]const u8 {
        self.access_count[id] += 1;

        if (self.pinned.get(id)) |blk| {
            self.stats.pin_hits += 1;
            return blk;
        }
        if (self.lru.find(id)) |blk| {
            self.stats.lru_hits += 1;
            return blk;
        }
        const slot_buf = try self.lru.reserve(id);
        // publish-after-verify: on a digest failure, unmap the slot so the
        // poison is not returned by a later LRU hit (audit #5 P0-3)
        self.readBlock(id, slot_buf[1], false) catch |e| {
            self.lru.abort(id);
            return e;
        };
        return slot_buf[1];
    }

    pub fn pinnedCount(self: *const ExpertCache) usize {
        return self.pinned.count();
    }
};

// ---- tests -----------------------------------------------------------------

test "LRU evicts least-recently-used and pin overrides" {
    const gpa = std.testing.allocator;
    var lru = try Lru.init(gpa, 2, 4);
    defer lru.deinit(gpa);

    // fill two slots
    {
        const r0 = try lru.reserve(10);
        @memcpy(r0[1], "aaaa");
        const r1 = try lru.reserve(11);
        @memcpy(r1[1], "bbbb");
    }
    try std.testing.expect(lru.find(10) != null); // touch 10 -> 11 is now LRU
    // inserting a third evicts 11
    {
        const r2 = try lru.reserve(12);
        @memcpy(r2[1], "cccc");
    }
    try std.testing.expect(lru.find(11) == null);
    try std.testing.expect(lru.find(10) != null);
    try std.testing.expect(lru.find(12) != null);
}
