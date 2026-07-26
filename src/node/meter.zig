//! Per-client metering on full nodes (SPEC.md node classes): light nodes
//! compensate full nodes for serviced inference. v1 implements the
//! *accounting* half — a per-client ledger (free quota + purchased credits −
//! usage, in token units) with hard enforcement (`payment_required` once the
//! allowance is exhausted) — and exposes the settlement seam: the `credit` op
//! accepts a payment proof that v1 does not verify (trusted-swarm), and a
//! real payment rail (invoice/receipt verification) replaces that check
//! without touching the ledger or the enforcement path.

const std = @import("std");
const Io = std.Io;

pub const Account = struct {
    used: u64 = 0, // tokens consumed (prompt + generated)
    credits: u64 = 0, // purchased allowance beyond the free quota
};

pub const Charge = struct {
    cost: u64,
    balance: u64, // remaining allowance after the charge
};

pub const Meter = struct {
    gpa: std.mem.Allocator,
    io: Io,
    mutex: Io.Mutex = .init,
    free_quota: u64, // per-client free allowance, token units
    accounts: std.StringHashMap(Account),
    max_accounts: usize, // memory-DoS bound on distinct client ids (audit #6 P0-2)

    pub fn init(gpa: std.mem.Allocator, io: Io, free_quota: u64) Meter {
        return .{
            .gpa = gpa,
            .io = io,
            .free_quota = free_quota,
            .accounts = std.StringHashMap(Account).init(gpa),
            .max_accounts = 100_000,
        };
    }

    pub fn deinit(self: *Meter) void {
        var it = self.accounts.keyIterator();
        while (it.next()) |k| self.gpa.free(k.*);
        self.accounts.deinit();
    }

    fn allowance(self: *const Meter, acc: Account) u64 {
        return (self.free_quota + acc.credits) -| acc.used;
    }

    /// Remaining allowance for `client` without charging.
    pub fn remaining(self: *Meter, client: []const u8) u64 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const acc = self.accounts.get(client) orelse Account{};
        return self.allowance(acc);
    }

    /// Charge `tokens` to `client`. The pre-request gate is `remaining() > 0`;
    /// the final request may overdraw by at most one generation (charged from
    /// actual usage, clamped at zero balance).
    pub fn charge(self: *Meter, client: []const u8, tokens: u64) !Charge {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (!self.accounts.contains(client) and self.accounts.count() >= self.max_accounts)
            return error.TooManyAccounts;
        const gop = try self.accounts.getOrPut(client);
        if (!gop.found_existing) {
            gop.key_ptr.* = self.gpa.dupe(u8, client) catch |e| {
                self.accounts.removeByPtr(gop.key_ptr);
                return e;
            };
            gop.value_ptr.* = .{};
        }
        gop.value_ptr.used += tokens;
        return .{ .cost = tokens, .balance = self.allowance(gop.value_ptr.*) };
    }

    /// Reserve up to `want` tokens for `client`, debiting them immediately
    /// (security issue #30). The old gate/charge split was a TOCTOU: N
    /// concurrent requests each read the full remaining allowance before any of
    /// them charged, so aggregate generation could reach N x the balance
    /// (bounded only by MAX_CONNS). Reserving under one lock makes the debit
    /// atomic with the check; `settle` returns whatever was not used.
    ///
    /// Returns 0 when the client is exhausted, which callers treat as
    /// payment_required.
    pub fn reserve(self: *Meter, client: []const u8, want: u64) !u64 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (!self.accounts.contains(client) and self.accounts.count() >= self.max_accounts)
            return error.TooManyAccounts;
        const gop = try self.accounts.getOrPut(client);
        if (!gop.found_existing) {
            gop.key_ptr.* = self.gpa.dupe(u8, client) catch |e| {
                // do not leave an entry with an uninitialised key behind
                self.accounts.removeByPtr(gop.key_ptr);
                return e;
            };
            gop.value_ptr.* = .{};
        }
        const avail = self.allowance(gop.value_ptr.*);
        const granted = @min(want, avail);
        gop.value_ptr.used += granted;
        return granted;
    }

    /// Return the unused part of a reservation. `actual` is what was really
    /// consumed; the difference is credited back. Always call this, including
    /// on the error path, or the client keeps paying for work it never got.
    pub fn settle(self: *Meter, client: []const u8, reserved: u64, actual: u64) Charge {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const acc = self.accounts.getPtr(client) orelse return .{ .cost = actual, .balance = 0 };
        if (reserved > actual) acc.used -|= (reserved - actual);
        return .{ .cost = actual, .balance = self.allowance(acc.*) };
    }

    /// Clamp a requested generation length to the client's remaining allowance
    /// (audit #6 P1): a single request can overdraw by at most the returned
    /// value, never the whole `max_tokens` the caller asked for.
    pub fn clampTokens(self: *Meter, client: []const u8, requested: u64) u64 {
        const rem = self.remaining(client);
        return @min(requested, rem);
    }

    /// Settlement seam: add purchased credits. v1 trusts the caller (the
    /// `proof` argument is recorded nowhere and verified by nothing);
    /// a payment rail replaces this function's trust, not its interface.
    pub fn credit(self: *Meter, client: []const u8, amount: u64) !u64 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        // same account-map bound `charge` enforces (security issue #30):
        // otherwise a leaked admin token grows the map without limit
        if (!self.accounts.contains(client) and self.accounts.count() >= self.max_accounts)
            return error.TooManyAccounts;
        const gop = try self.accounts.getOrPut(client);
        if (!gop.found_existing) {
            gop.key_ptr.* = self.gpa.dupe(u8, client) catch |e| {
                self.accounts.removeByPtr(gop.key_ptr);
                return e;
            };
            gop.value_ptr.* = .{};
        }
        gop.value_ptr.credits += amount;
        return self.allowance(gop.value_ptr.*);
    }

    /// Snapshot for a single client (used, balance).
    pub fn account(self: *Meter, client: []const u8) struct { used: u64, balance: u64 } {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const acc = self.accounts.get(client) orelse Account{};
        return .{ .used = acc.used, .balance = self.allowance(acc) };
    }
};

test "meter: quota, charge, exhaustion, credit top-up" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    var m = Meter.init(gpa, threaded.io(), 100);
    defer m.deinit();

    try std.testing.expect(m.remaining("alice") == 100);
    const c1 = try m.charge("alice", 30);
    try std.testing.expect(c1.cost == 30 and c1.balance == 70);
    _ = try m.charge("alice", 70);
    try std.testing.expect(m.remaining("alice") == 0);

    // exhausted: gate closes; other clients unaffected
    try std.testing.expect(m.remaining("bob") == 100);

    // settlement stub reopens the gate
    const bal = try m.credit("alice", 50);
    try std.testing.expect(bal == 50);
    const c2 = try m.charge("alice", 20);
    try std.testing.expect(c2.balance == 30);

    // overdraw on the final generation clamps at zero
    _ = try m.charge("alice", 1000);
    try std.testing.expect(m.remaining("alice") == 0);
}

test "reserve debits atomically: concurrent requests cannot each see the full balance (issue #30)" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    var m = Meter.init(gpa, threaded.io(), 100);
    defer m.deinit();

    // two "concurrent" requests: the old gate/charge split let both see 100
    const a = try m.reserve("alice", 80);
    const b = try m.reserve("alice", 80);
    try std.testing.expectEqual(@as(u64, 80), a);
    try std.testing.expectEqual(@as(u64, 20), b); // only what is left
    try std.testing.expectEqual(@as(u64, 0), m.remaining("alice"));

    // settling refunds the unused part of each reservation
    _ = m.settle("alice", a, 10);
    _ = m.settle("alice", b, 5);
    try std.testing.expectEqual(@as(u64, 85), m.remaining("alice"));
}

test "an aborted request still pays for work done (issue #30)" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    var m = Meter.init(gpa, threaded.io(), 100);
    defer m.deinit();

    const r = try m.reserve("bob", 50);
    // client disconnects after 7 tokens were emitted: bill those, refund rest
    const ch = m.settle("bob", r, 7);
    try std.testing.expectEqual(@as(u64, 7), ch.cost);
    try std.testing.expectEqual(@as(u64, 93), m.remaining("bob"));
}

test "exhausted client cannot reserve (issue #30)" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    var m = Meter.init(gpa, threaded.io(), 10);
    defer m.deinit();
    const r = try m.reserve("carol", 10);
    _ = m.settle("carol", r, 10); // fully consumed
    try std.testing.expectEqual(@as(u64, 0), try m.reserve("carol", 10));
}
