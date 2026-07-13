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

pub const PeerInfo = struct {
    addr: []u8, // owned "host:port"
    version_hex: [64]u8, // manifest version the peer advertises (zeros if none)
    holdings_hex: []u8, // owned; may be empty if peer has no store
    last_seen_ns: i128,
};

pub const Table = struct {
    gpa: std.mem.Allocator,
    io: Io,
    mutex: Io.Mutex = .init,
    entries: std.ArrayList(PeerInfo) = .empty,
    self_addr: []const u8, // our own advertised addr — never inserted

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
    pub fn merge(self: *Table, addr: []const u8, version_hex: []const u8, holdings_hex: []const u8, now_ns: i128) !bool {
        if (std.mem.eql(u8, addr, self.self_addr)) return false;
        if (version_hex.len != 64) return error.BadVersionHex;

        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);

        for (self.entries.items) |*e| {
            if (std.mem.eql(u8, e.addr, addr)) {
                @memcpy(&e.version_hex, version_hex);
                if (!std.mem.eql(u8, e.holdings_hex, holdings_hex)) {
                    self.gpa.free(e.holdings_hex);
                    e.holdings_hex = try self.gpa.dupe(u8, holdings_hex);
                }
                e.last_seen_ns = now_ns;
                return false;
            }
        }
        var info = PeerInfo{
            .addr = try self.gpa.dupe(u8, addr),
            .version_hex = undefined,
            .holdings_hex = try self.gpa.dupe(u8, holdings_hex),
            .last_seen_ns = now_ns,
        };
        @memcpy(&info.version_hex, version_hex);
        try self.entries.append(self.gpa, info);
        return true;
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

    pub fn count(self: *Table) usize {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.entries.items.len;
    }

    /// Serialize the table as `addr=<a> version=<v> holdings=<h>` lines into
    /// `list` (for the GOSSIP response). Caller provides/owns the list.
    pub fn dump(self: *Table, gpa: std.mem.Allocator, list: *std.ArrayList(u8)) !usize {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        for (self.entries.items) |e| {
            try list.print(gpa, "addr={s} version={s} holdings={s}\n", .{ e.addr, e.version_hex, e.holdings_hex });
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
    try std.testing.expect(try t.merge("127.0.0.1:9001", v, "ff01", 1)); // new
    try std.testing.expect(!try t.merge("127.0.0.1:9001", v, "ab00", 2)); // refresh
    try std.testing.expect(!try t.merge("127.0.0.1:9000", v, "ff01", 3)); // self
    try std.testing.expect(t.count() == 1);

    var out = std.ArrayList(u8).empty;
    defer out.deinit(gpa);
    const n = try t.dump(gpa, &out);
    try std.testing.expect(n == 1);
    try std.testing.expect(std.mem.indexOf(u8, out.items, "holdings=ab00") != null);
}
