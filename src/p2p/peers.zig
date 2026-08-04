//! Dynamic peer table — the substrate for gossip (ROADMAP #7) and the majority
//! count hardforks need (#4).
//!
//! Each entry tracks a peer's advertised address, manifest version, and
//! holdings bitmap (hex). Entries arrive from two sources: peers announcing
//! themselves (`GOSSIP` requests we serve) and tables we receive when we gossip
//! outward. The table is shared by the gossip loop, the eager repair loop, and
//! P2P connection threads, so all access goes through a mutex.
//!
//! This is the LAN-scale, epidemic-exchange form of the "global gossip topic"
//! decision; swapping the transport for real gossipsub/discv5 later replaces
//! the exchange mechanics, not this table.

const std = @import("std");
const Io = std.Io;
const stats_mod = @import("../core/stats.zig");

pub const NO_COMMITTEE: u32 = 0xFFFF_FFFF;

/// How we came to know this entry, which decides whether it proves liveness.
///
///  - `first_hand`: we exchanged with this peer, or it connected to us.
///  - `hearsay`: another peer listed it. Enough to learn the address (and to
///    start a fresh entry's TTL, so a genuinely new peer gets a grace period
///    in which to be contacted), never enough to keep a stale one alive.
pub const Evidence = enum { first_hand, hearsay };

pub const PeerInfo = struct {
    addr: []u8, // owned "host:port"
    version_hex: [64]u8, // manifest version the peer advertises (zeros if none)
    holdings_hex: []u8, // owned; may be empty if peer has no store
    committee_id: u32, // NO_COMMITTEE when unknown / not in one
    holdings_seq: u64, // monotonic per-peer; a lower seq never overwrites bits
    last_seen_ns: i128,
    /// Whether we have ever *dialed this address successfully*. Hearsay --
    /// an address relayed inside another peer's batch -- is enough to try
    /// dialing, but not enough to send token-loop fetches to (security issue
    /// #144): an attacker who announces full bitmaps for addresses that do
    /// not answer would otherwise redirect every fetch into a timeout.
    dialed: bool = false,
};

/// Bound on distinct peers retained (audit #7 P1 unbounded table): beyond this
/// the coldest (oldest last_seen) entry is evicted on insert.
pub const MAX_PEERS: usize = 4096;

/// A peer not heard from for this long is no longer counted as a holder.
///
/// Gossip refreshes `last_seen_ns` every 3 s, so this is ~10 missed rounds.
/// The entry itself is *kept* — the eager-churn policy in gossip.zig never
/// forgets an address, so a peer that comes back is picked up again without
/// re-bootstrapping, and a node whose only seed has died does not become
/// unreachable by pruning it.
///
/// What staleness must gate is the *holdings claim*, which is a different
/// thing from the address. Liveness cannot be left to the committee
/// heartbeat: that monitors committee members, and the most important node in
/// a swarm need not be one. Measured on a three-node run, an origin that
/// seeded the model sat outside every committee while solely holding 332 of
/// 1664 expert shards; it was killed and six minutes later both survivors
/// still advertised it as a live holder, so every fetch for those shards
/// would have been aimed at a dead node and no repair was ever triggered.
pub const PEER_TTL_NS: i128 = 30 * std.time.ns_per_s;

pub const Table = struct {
    gpa: std.mem.Allocator,
    io: Io,
    mutex: Io.Mutex = .init,
    entries: std.ArrayList(PeerInfo) = .empty,
    self_addr: []const u8, // our own advertised addr — never inserted
    /// Exact hex length a holdings bitmap must have, i.e. ((nRanges+7)/8)*2.
    /// Zero means "store not attached yet, do not validate". Announces carry an
    /// attacker-chosen bitmap whose only bound was MAX_BODY_BYTES, so without
    /// this a peer could park ~128 MiB of hex per entry (security issue #28).
    expected_holdings_hex_len: usize = 0,

    pub fn init(gpa: std.mem.Allocator, io: Io, self_addr: []const u8) Table {
        return .{ .gpa = gpa, .io = io, .self_addr = self_addr };
    }

    pub fn deinit(self: *Table) void {
        for (self.entries.items) |e| {
            self.gpa.free(e.addr);
            self.gpa.free(e.holdings_hex);
        }
        self.entries.deinit(self.gpa);
    }

    /// Insert or refresh a peer. Skips our own address. Returns true if the
    /// entry was new (useful for logging discovery).
    pub fn merge(self: *Table, addr: []const u8, version_hex: []const u8, holdings_hex: []const u8, committee_id: u32, holdings_seq: u64, now_ns: i128, evidence: Evidence) !bool {
        if (std.mem.eql(u8, addr, self.self_addr)) return false;
        if (version_hex.len != 64) return error.BadVersionHex;
        // reject a bitmap that is not exactly one bit per shard (issue #28)
        if (self.expected_holdings_hex_len != 0 and holdings_hex.len != 0 and
            holdings_hex.len != self.expected_holdings_hex_len) return error.BadHoldingsLength;

        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

        for (self.entries.items) |*e| {
            if (std.mem.eql(u8, e.addr, addr)) {
                // Only direct contact refreshes liveness. A peer's table is
                // hearsay: it says an address exists, not that it answers.
                // Refreshing on hearsay makes death undetectable in any swarm
                // of three or more, because the survivors keep echoing the
                // dead node's entry to each other forever — measured, the
                // origin stayed "live" indefinitely after being killed.
                if (evidence == .first_hand) {
                    e.last_seen_ns = now_ns;
                    e.dialed = true;
                }
                // a strictly-lower seq must not clobber fresher holdings/version
                // (audit #7 P1); seq 0 (text GOSSIP) never regresses a known seq
                if (holdings_seq < e.holdings_seq) return false;
                @memcpy(&e.version_hex, version_hex);
                if (!std.mem.eql(u8, e.holdings_hex, holdings_hex)) {
                    self.gpa.free(e.holdings_hex);
                    e.holdings_hex = try self.gpa.dupe(u8, holdings_hex);
                }
                if (committee_id != NO_COMMITTEE) e.committee_id = committee_id;
                e.holdings_seq = holdings_seq;
                return false;
            }
        }
        // bounded insert: evict the coldest entry when full
        if (self.entries.items.len >= MAX_PEERS) self.evictColdest();
        var info = PeerInfo{
            .addr = try self.gpa.dupe(u8, addr),
            .version_hex = undefined,
            .holdings_hex = try self.gpa.dupe(u8, holdings_hex),
            .committee_id = committee_id,
            .holdings_seq = holdings_seq,
            .last_seen_ns = now_ns,
            .dialed = evidence == .first_hand,
        };
        @memcpy(&info.version_hex, version_hex);
        try self.entries.append(self.gpa, info);
        return true;
    }

    /// Addresses of known peers in `committee_id` — the gossip-derived
    /// committee view (SPEC.md): earlier members learn later joiners from
    /// their announces. Caller frees each string and the slice.
    pub fn committeeMembersAlloc(self: *Table, gpa: std.mem.Allocator, committee_id: u32) ![][]u8 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        var out = std.ArrayList([]u8).empty;
        errdefer {
            for (out.items) |a| gpa.free(a);
            out.deinit(gpa);
        }
        for (self.entries.items) |e| {
            if (e.committee_id == committee_id) {
                try out.append(gpa, try gpa.dupe(u8, e.addr));
            }
        }
        return out.toOwnedSlice(gpa);
    }

    /// Drop the entry with the oldest last_seen. Caller holds the mutex.
    fn evictColdest(self: *Table) void {
        if (self.entries.items.len == 0) return;
        var oldest: usize = 0;
        for (self.entries.items, 0..) |e, i| {
            if (e.last_seen_ns < self.entries.items[oldest].last_seen_ns) oldest = i;
        }
        const e = self.entries.swapRemove(oldest);
        self.gpa.free(e.addr);
        self.gpa.free(e.holdings_hex);
    }

    /// Addresses of peers whose advertised holdings bitmap covers shard `id`
    /// (audit #7 P1 holdings-filtered candidates). Bit i in the hex bitmap.
    /// Caller frees each string and the slice.
    pub fn holdersOf(self: *Table, gpa: std.mem.Allocator, id: usize) ![][]u8 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        var out = std.ArrayList([]u8).empty;
        errdefer {
            for (out.items) |a| gpa.free(a);
            out.deinit(gpa);
        }
        const byte_i = id / 8;
        const bit: u8 = @as(u8, 1) << @intCast(id % 8);
        const now = stats_mod.nowNs(self.io);
        for (self.entries.items) |e| {
            // A stale peer's bitmap is a claim about a node that may be gone.
            // Returning it would send every fetch for these shards to a dead
            // address to time out, and would hide from the repair loop that
            // the shard now has no holder at all.
            if (now - e.last_seen_ns > PEER_TTL_NS) continue;
            // Hearsay holdings never steer the hot path (issue #144).
            if (!e.dialed) continue;
            if (byte_i * 2 + 1 >= e.holdings_hex.len) continue;
            const b = std.fmt.parseInt(u8, e.holdings_hex[byte_i * 2 ..][0..2], 16) catch continue;
            if (b & bit != 0) try out.append(gpa, try gpa.dupe(u8, e.addr));
        }
        return out.toOwnedSlice(gpa);
    }

    /// Copy of a peer's advertised holdings bitmap (hex), or null if unknown.
    /// Caller frees. Used by death re-replication (audit #7 P1).
    pub fn peerBitmapAlloc(self: *Table, gpa: std.mem.Allocator, addr: []const u8) !?[]u8 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        for (self.entries.items) |e| {
            if (std.mem.eql(u8, e.addr, addr)) {
                if (e.holdings_hex.len == 0) return null;
                return try gpa.dupe(u8, e.holdings_hex);
            }
        }
        return null;
    }

    /// Copy of all peer addresses. Caller frees each string and the slice.
    pub fn snapshotAddrs(self: *Table, gpa: std.mem.Allocator) ![][]u8 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const out = try gpa.alloc([]u8, self.entries.items.len);
        var filled: usize = 0;
        errdefer {
            for (out[0..filled]) |a| gpa.free(a);
            gpa.free(out);
        }
        for (self.entries.items, 0..) |e, i| {
            out[i] = try gpa.dupe(u8, e.addr);
            filled += 1;
        }
        return out;
    }

    /// The live peer with the fullest holdings bitmap, for delegate-while-
    /// cold: a cold node forwards generations to the warmest peer it knows.
    /// Returns an owned addr copy plus the holdings fraction; null when no
    /// live peer advertises a store.
    pub fn warmest(self: *Table, gpa: std.mem.Allocator, now_ns: i128, total_shards: usize) !?struct { addr: []u8, frac: f64 } {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        var best_addr: ?[]const u8 = null;
        var best_bits: usize = 0;
        for (self.entries.items) |*ent| {
            if (now_ns - ent.last_seen_ns > PEER_TTL_NS) continue;
            if (ent.holdings_hex.len == 0) continue;
            var bits: usize = 0;
            for (ent.holdings_hex) |c| {
                const v = std.fmt.charToDigit(c, 16) catch continue;
                bits += @popCount(v);
            }
            if (bits > best_bits) {
                best_bits = bits;
                best_addr = ent.addr;
            }
        }
        const addr = best_addr orelse return null;
        // Fraction against the real shard count: the bitmap's byte padding
        // would otherwise deflate a full peer below any threshold.
        const denom: f64 = @floatFromInt(@max(total_shards, 1));
        return .{
            .addr = try gpa.dupe(u8, addr),
            .frac = @min(@as(f64, @floatFromInt(best_bits)) / denom, 1.0),
        };
    }

    pub fn count(self: *Table) usize {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.entries.items.len;
    }

    /// Number of shards in [0, n_ranges) that fewer than `min` *live* peers
    /// advertise. `own` is this node's own holdings, which are not in the
    /// table. Returns 0 when no peer has published a bitmap yet, since "no
    /// data" and "no coverage" must not read the same.
    ///
    /// This exists because under-replication was otherwise silent. A swarm
    /// where every node takes `--hold-fraction 0.40` cannot reach R=2 with
    /// only two joiners — 2 x 40% = 80% of one full copy — and the assignment
    /// is not wrong, the arithmetic is short. Measured on a three-node run:
    /// 332 of 1664 expert shards had exactly one holder under `--r-target 2`,
    /// and nothing said so.
    pub fn underReplicated(self: *Table, gpa: std.mem.Allocator, n_ranges: usize, min: usize, own_bits: ?[]const u8) !?usize {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (n_ranges == 0) return null;

        const counts = try gpa.alloc(u8, n_ranges);
        defer gpa.free(counts);
        @memset(counts, 0);

        const now = stats_mod.nowNs(self.io);
        var any = false;
        for (self.entries.items) |e| {
            if (now - e.last_seen_ns > PEER_TTL_NS) continue;
            if (e.holdings_hex.len == 0) continue;
            any = true;
            for (0..n_ranges) |id| {
                const byte_i = id / 8;
                if (byte_i * 2 + 1 >= e.holdings_hex.len) break;
                const b = std.fmt.parseInt(u8, e.holdings_hex[byte_i * 2 ..][0..2], 16) catch continue;
                if (b & (@as(u8, 1) << @intCast(id % 8)) != 0) counts[id] +|= 1;
            }
        }
        if (!any) return null;

        var short: usize = 0;
        for (counts, 0..) |c, id| {
            var n: usize = c;
            if (own_bits) |ob| {
                const bi = id / 8;
                if (bi < ob.len and ob[bi] & (@as(u8, 1) << @intCast(id % 8)) != 0) n += 1;
            }
            if (n < min) short += 1;
        }
        return short;
    }

    /// Peers heard from within PEER_TTL_NS. This is what the status line and
    /// the hardfork majority count should use; `count` includes tombstones the
    /// eager-churn policy is still retrying.
    pub fn liveCount(self: *Table) usize {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const now = stats_mod.nowNs(self.io);
        var n: usize = 0;
        for (self.entries.items) |e| {
            if (now - e.last_seen_ns <= PEER_TTL_NS) n += 1;
        }
        return n;
    }

    /// Serialize the table as `addr=<a> version=<v> holdings=<h>` lines into
    /// `list` (for the GOSSIP response). Caller provides/owns the list.
    pub fn dump(self: *Table, gpa: std.mem.Allocator, list: *std.ArrayList(u8)) !usize {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        for (self.entries.items) |e| {
            try list.print(gpa, "addr={s} version={s} holdings={s} committee={d} seq={d}\n", .{ e.addr, e.version_hex, e.holdings_hex, e.committee_id, e.holdings_seq });
        }
        return self.entries.items.len;
    }
};

test "table merges, dedupes, refuses self" {
    const gpa = std.testing.allocator;
    // a Table without real Io: mutex lockUncancelable needs io only when
    // contended; single-threaded tests never contend, but the API requires an
    // io. Use a Threaded instance.
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var t = Table.init(gpa, io, "127.0.0.1:9000");
    defer t.deinit();

    const v = "aa" ** 32;
    try std.testing.expect(try t.merge("127.0.0.1:9001", v, "ff01", 2, 1, 1, .first_hand)); // new
    try std.testing.expect(!try t.merge("127.0.0.1:9001", v, "ab00", NO_COMMITTEE, 2, 2, .first_hand)); // refresh keeps committee
    try std.testing.expect(!try t.merge("127.0.0.1:9000", v, "ff01", 2, 1, 3, .first_hand)); // self
    try std.testing.expect(t.count() == 1);

    var out = std.ArrayList(u8).empty;
    defer out.deinit(gpa);
    const n = try t.dump(gpa, &out);
    try std.testing.expect(n == 1);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "holdings=ab00") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "committee=2") != null);

    const members = try t.committeeMembersAlloc(gpa, 2);
    defer {
        for (members) |m| gpa.free(m);
        gpa.free(members);
    }
    try std.testing.expect(members.len == 1);
    try std.testing.expectEqualStrings("127.0.0.1:9001", members[0]);
}

test "merge rejects a holdings bitmap of the wrong size (issue #28)" {
    const gpa = std.testing.allocator;
    var t = Table.init(gpa, undefined, "self:1");
    defer t.deinit();
    t.expected_holdings_hex_len = 4; // 2 bytes of bitmap = 16 shards
    const v = "0" ** 64;
    // an oversized attacker bitmap is refused instead of being stored
    try std.testing.expectError(error.BadHoldingsLength, t.merge("peer:1", v, "00" ** 4096, 0, 1, 0, .first_hand));
    // the correctly sized one is accepted
    try std.testing.expect(try t.merge("peer:1", v, "ffff", 0, 1, 0, .first_hand));
}

test "a stale peer stops being a holder but keeps its table entry" {
    // The regression this pins: liveness was decided only by the committee
    // heartbeat, so a non-committee origin could die unnoticed and go on being
    // handed out as the holder of shards nobody else had.
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();

    var t = Table.init(gpa, threaded.io(), "127.0.0.1:9000");
    defer t.deinit();

    const now = stats_mod.nowNs(t.io);
    const v = [_]u8{'0'} ** 64;
    _ = try t.merge("127.0.0.1:9101", &v, "ff", NO_COMMITTEE, 1, now, .first_hand);

    {
        const h = try t.holdersOf(gpa, 3);
        defer {
            for (h) |a| gpa.free(a);
            gpa.free(h);
        }
        try std.testing.expectEqual(@as(usize, 1), h.len);
    }
    try std.testing.expectEqual(@as(usize, 1), t.liveCount());

    // Age it past the TTL by rewriting last_seen, as a missed gossip round would.
    t.entries.items[0].last_seen_ns = now - (PEER_TTL_NS + 1);

    {
        const h = try t.holdersOf(gpa, 3);
        defer {
            for (h) |a| gpa.free(a);
            gpa.free(h);
        }
        try std.testing.expectEqual(@as(usize, 0), h.len);
    }
    try std.testing.expectEqual(@as(usize, 0), t.liveCount());
    // ...but the address is retained, so eager-churn retry can still reach it.
    try std.testing.expectEqual(@as(usize, 1), t.count());
}

test "hearsay about a peer does not keep it alive" {
    // Three nodes, one dies: the two survivors go on listing it in the tables
    // they exchange. If those echoes refreshed liveness, no TTL could ever
    // fire and the death would be invisible -- which is what was measured
    // before Evidence existed.
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();

    var t = Table.init(gpa, threaded.io(), "127.0.0.1:9000");
    defer t.deinit();

    const now = stats_mod.nowNs(t.io);
    const v = [_]u8{'0'} ** 64;
    _ = try t.merge("127.0.0.1:9101", &v, "ff", NO_COMMITTEE, 1, now, .first_hand);
    t.entries.items[0].last_seen_ns = now - (PEER_TTL_NS + 1);

    // a survivor echoes the dead node's entry
    _ = try t.merge("127.0.0.1:9101", &v, "ff", NO_COMMITTEE, 2, now, .hearsay);
    try std.testing.expectEqual(@as(usize, 0), t.liveCount());
    // the echo may still carry newer holdings; only liveness is withheld
    try std.testing.expectEqual(@as(u64, 2), t.entries.items[0].holdings_seq);

    // direct contact revives it
    _ = try t.merge("127.0.0.1:9101", &v, "ff", NO_COMMITTEE, 3, now, .first_hand);
    try std.testing.expectEqual(@as(usize, 1), t.liveCount());
}

test "hearsay holdings never steer the hot path (security issue #144)" {
    // An attacker can put any address into another peer's AnnounceBatch with
    // a full bitmap. Trusting that for fetch selection sends every miss to a
    // black hole; requiring a successful dial first makes the claim cost the
    // attacker a reachable, answering node.
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    var t = Table.init(gpa, threaded.io(), "127.0.0.1:9000");
    defer t.deinit();

    const now = stats_mod.nowNs(t.io);
    const v = [_]u8{'0'} ** 64;
    // relayed claim only: known address, but never answered us
    _ = try t.merge("10.0.0.9:1", &v, "ff", NO_COMMITTEE, 1, now, .hearsay);
    {
        const h = try t.holdersOf(gpa, 3);
        defer {
            for (h) |a| gpa.free(a);
            gpa.free(h);
        }
        try std.testing.expectEqual(@as(usize, 0), h.len);
    }
    // it stays in the table (we may still dial it) and becomes a holder only
    // once a first-hand exchange proves it answers
    _ = try t.merge("10.0.0.9:1", &v, "ff", NO_COMMITTEE, 2, now, .first_hand);
    {
        const h = try t.holdersOf(gpa, 3);
        defer {
            for (h) |a| gpa.free(a);
            gpa.free(h);
        }
        try std.testing.expectEqual(@as(usize, 1), h.len);
    }
}
