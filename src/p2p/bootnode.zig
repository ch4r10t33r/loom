//! Bootnode committee registry (SPEC.md): assigns each joining node to a
//! shard committee and, within it, a want-set of the currently least-covered
//! expert shards — so committee completeness (every shard held by ≥1 member)
//! and the redundancy target R are met by construction, not probability.
//!
//! Lifecycle: joiners fill the first committee whose minimum expert-shard
//! coverage is below R; when all committees are saturated (min coverage ≥ R),
//! the next joiner opens a new one. Rejoins (same address) at the same
//! capacity are idempotent and return the original assignment; a rejoin at a
//! different capacity re-assigns within the same committee (issue #252).
//!
//! The registry is in-memory state on the bootnode; it is deliberately NOT in
//! the inference path and nothing depends on it after a node has joined.

const std = @import("std");
const Io = std.Io;
const weights = @import("weights.zig");

pub const JoinOut = struct {
    committee_id: usize,
    /// resident shards + assigned expert shards; caller owns
    assign: weights.Holdings,
    /// other members of the committee ("host:port" each); caller owns all
    members: [][]u8,

    pub fn deinit(self: *JoinOut, gpa: std.mem.Allocator) void {
        self.assign.deinit(gpa);
        for (self.members) |m| gpa.free(m);
        gpa.free(self.members);
    }
};

const Member = struct {
    addr: []u8, // owned
    assign: weights.Holdings, // owned
};

const Committee = struct {
    id: usize,
    members: std.ArrayList(Member) = .empty,
    coverage: []u16, // per expert shard (index 0 = shard n_resident)

    fn minCoverage(self: *const Committee) u16 {
        var m: u16 = std.math.maxInt(u16);
        for (self.coverage) |c| m = @min(m, c);
        return if (self.coverage.len == 0) 0 else m;
    }
};

pub const Registry = struct {
    gpa: std.mem.Allocator,
    io: Io,
    mutex: Io.Mutex = .init,
    n_shards: usize,
    n_resident: usize,
    r_target: u16,
    committees: std.ArrayList(Committee) = .empty,

    pub fn init(gpa: std.mem.Allocator, io: Io, n_shards: usize, n_resident: usize, r_target: u16) Registry {
        return .{ .gpa = gpa, .io = io, .n_shards = n_shards, .n_resident = n_resident, .r_target = r_target };
    }

    pub fn deinit(self: *Registry) void {
        for (self.committees.items) |*c| {
            for (c.members.items) |*m| {
                self.gpa.free(m.addr);
                m.assign.deinit(self.gpa);
            }
            c.members.deinit(self.gpa);
            self.gpa.free(c.coverage);
        }
        self.committees.deinit(self.gpa);
    }

    fn nExperts(self: *const Registry) usize {
        return self.n_shards - self.n_resident;
    }

    /// Assign `addr` to a committee with capacity for `capacity` expert
    /// shards. Least-covered shards first. A rejoin at the same capacity is
    /// idempotent; a rejoin at a different capacity re-assigns (issue #252:
    /// the old always-return-stored behavior meant a node that ever joined
    /// at the default fraction could never come back smaller -- every later
    /// `--hold-fraction` was silently ignored).
    pub fn join(self: *Registry, addr: []const u8, capacity: usize) !JoinOut {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

        const cap = @max(@as(usize, 1), @min(capacity, self.nExperts()));

        for (self.committees.items) |*c| {
            for (c.members.items) |*m| {
                if (std.mem.eql(u8, m.addr, addr)) {
                    const held = m.assign.count() - self.n_resident;
                    if (held != cap) {
                        // Release the old picks, then re-pick at the new
                        // size. Freeing drops each shard a coverage point,
                        // which puts the previously-held set first in the
                        // least-covered order again: a grow keeps the old
                        // set and extends it, a shrink keeps its
                        // least-covered core.
                        var i: usize = self.n_resident;
                        while (i < self.n_shards) : (i += 1) {
                            if (m.assign.has(i)) c.coverage[i - self.n_resident] -= 1;
                        }
                        m.assign.deinit(self.gpa);
                        m.assign = try self.assignExperts(c, cap);
                    }
                    const copy = weights.Holdings{
                        .bits = try self.gpa.dupe(u8, m.assign.bits),
                        .n = m.assign.n,
                    };
                    return self.buildOut(c, copy, addr);
                }
            }
        }

        // first committee with min coverage < r_target, else a new one
        var target: ?*Committee = null;
        for (self.committees.items) |*c| {
            if (c.minCoverage() < self.r_target) {
                target = c;
                break;
            }
        }
        if (target == null) {
            const coverage = try self.gpa.alloc(u16, self.nExperts());
            @memset(coverage, 0);
            try self.committees.append(self.gpa, .{
                .id = self.committees.items.len,
                .coverage = coverage,
            });
            target = &self.committees.items[self.committees.items.len - 1];
        }
        const c = target.?;

        var assign = try self.assignExperts(c, cap);
        errdefer assign.deinit(self.gpa);

        const member = Member{
            .addr = try self.gpa.dupe(u8, addr),
            .assign = .{ .bits = try self.gpa.dupe(u8, assign.bits), .n = assign.n },
        };
        try c.members.append(self.gpa, member);

        return self.buildOut(c, assign, addr);
    }

    /// Pick `cap` least-covered expert shards (stable by coverage, then
    /// index), bump their coverage, and return resident+picked. Caller owns.
    fn assignExperts(self: *Registry, c: *Committee, cap: usize) !weights.Holdings {
        const Idx = struct { cov: u16, idx: usize };
        const order = try self.gpa.alloc(Idx, self.nExperts());
        defer self.gpa.free(order);
        for (order, 0..) |*o, i| o.* = .{ .cov = c.coverage[i], .idx = i };
        std.mem.sort(Idx, order, {}, struct {
            fn lt(_: void, a: Idx, b: Idx) bool {
                if (a.cov != b.cov) return a.cov < b.cov;
                return a.idx < b.idx;
            }
        }.lt);

        var assign = try weights.Holdings.initEmpty(self.gpa, self.n_shards);
        errdefer assign.deinit(self.gpa);
        var i: usize = 0;
        while (i < self.n_resident) : (i += 1) assign.set(i); // resident mandatory
        for (order[0..cap]) |o| {
            assign.set(self.n_resident + o.idx);
            c.coverage[o.idx] += 1;
        }
        return assign;
    }

    /// Takes ownership of `assign` (already caller-owned at both call sites).
    fn buildOut(self: *Registry, c: *Committee, assign: weights.Holdings, self_addr: []const u8) !JoinOut {
        var members = std.ArrayList([]u8).empty;
        errdefer {
            for (members.items) |m| self.gpa.free(m);
            members.deinit(self.gpa);
        }
        for (c.members.items) |m| {
            if (std.mem.eql(u8, m.addr, self_addr)) continue;
            try members.append(self.gpa, try self.gpa.dupe(u8, m.addr));
        }
        return .{
            .committee_id = c.id,
            .assign = assign,
            .members = try members.toOwnedSlice(self.gpa),
        };
    }

    /// Debug summary: one line per committee.
    pub fn summary(self: *Registry, gpa: std.mem.Allocator, list: *std.ArrayList(u8)) !usize {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        for (self.committees.items) |*c| {
            var mx: u16 = 0;
            for (c.coverage) |cv| mx = @max(mx, cv);
            const mn = c.minCoverage();
            try list.print(gpa, "committee id={d} members={d} coverage_min={d} coverage_max={d} complete={} saturated={}\n", .{
                c.id, c.members.items.len, mn, mx, mn >= 1, mn >= self.r_target,
            });
        }
        return self.committees.items.len;
    }
};

test "committee assignment: least-covered first, completeness, then new committee" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // 2 resident + 8 expert shards, R=2
    var reg = Registry.init(gpa, io, 10, 2, 2);
    defer reg.deinit();

    // node1 takes 4 experts -> coverage [1,1,1,1,0,0,0,0]
    var j1 = try reg.join("n1:1", 4);
    defer j1.deinit(gpa);
    try std.testing.expect(j1.committee_id == 0);
    try std.testing.expect(j1.members.len == 0);
    try std.testing.expect(j1.assign.has(0) and j1.assign.has(1)); // resident
    try std.testing.expect(j1.assign.has(2) and j1.assign.has(5) and !j1.assign.has(6));

    // node2 takes 4 -> the uncovered ones first -> committee complete (min 1)
    var j2 = try reg.join("n2:1", 4);
    defer j2.deinit(gpa);
    try std.testing.expect(j2.committee_id == 0);
    try std.testing.expect(j2.members.len == 1);
    try std.testing.expect(j2.assign.has(6) and j2.assign.has(9));
    try std.testing.expect(!j2.assign.has(2)); // not the already-covered ones

    // node3 (8 experts) reinforces toward R=2 -> saturates committee 0
    var j3 = try reg.join("n3:1", 8);
    defer j3.deinit(gpa);
    try std.testing.expect(j3.committee_id == 0);

    // node4: committee 0 saturated -> new committee 1
    var j4 = try reg.join("n4:1", 4);
    defer j4.deinit(gpa);
    try std.testing.expect(j4.committee_id == 1);

    // rejoin is idempotent
    var j1b = try reg.join("n1:1", 4);
    defer j1b.deinit(gpa);
    try std.testing.expect(j1b.committee_id == 0);
    try std.testing.expect(std.mem.eql(u8, j1b.assign.bits, j1.assign.bits));
}

test "rejoin at a different capacity re-assigns (issue #252)" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // 2 resident + 8 expert shards, R=2
    var reg = Registry.init(gpa, io, 10, 2, 2);
    defer reg.deinit();

    var j1 = try reg.join("n1:1", 6);
    defer j1.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 8), j1.assign.count()); // 2 resident + 6

    // shrink: the regression -- the old code returned the stored 6-expert
    // assignment here, so a fresh low-holdings measurement node was
    // impossible once its address had ever joined at the default fraction
    var j2 = try reg.join("n1:1", 2);
    defer j2.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), j2.committee_id);
    try std.testing.expectEqual(@as(usize, 4), j2.assign.count()); // 2 resident + 2
    // the kept experts are a subset of the original assignment
    for (2..10) |i| {
        if (j2.assign.has(i)) try std.testing.expect(j1.assign.has(i));
    }

    // coverage bookkeeping stayed consistent: another joiner still gets a
    // full, valid assignment and the freed shards are available again
    var j3 = try reg.join("n2:1", 8);
    defer j3.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 10), j3.assign.count());

    // grow: back to 6 keeps the shrunken core and extends it
    var j4 = try reg.join("n1:1", 6);
    defer j4.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 8), j4.assign.count());
    for (2..10) |i| {
        if (j2.assign.has(i)) try std.testing.expect(j4.assign.has(i));
    }

    // same capacity stays idempotent
    var j5 = try reg.join("n1:1", 6);
    defer j5.deinit(gpa);
    try std.testing.expect(std.mem.eql(u8, j5.assign.bits, j4.assign.bits));
}
