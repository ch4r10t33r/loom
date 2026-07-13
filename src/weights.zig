//! Range-sharded weight store for GGUF distribution (ROADMAP.md v1 #2).
//!
//! A GGUF file is split into fixed-size byte ranges. Each range has a SHA-256
//! digest; the Merkle root over all range digests is the **model version id** —
//! the identity nodes gossip, advertise, and (later) hardfork on. A node holds a
//! subset of ranges recorded in a **holdings bitmap** (bit i = range i present);
//! the same bitmap, hex-encoded, is the compact holdings summary destined for
//! ENR metadata (300-byte ENR limit) and the gossip topic.
//!
//! Ranges are chosen *randomly* per node (`--hold-fraction`), so overlapping
//! holdings across nodes provide redundancy without a placement coordinator.
//!
//! On disk a store is a directory:
//!   model.gguf      the (possibly sparse) weight file
//!   ranges.manifest text: version, size, range_size, then one digest hex/line
//!   holdings.bitmap binary bitmap of held ranges

const std = @import("std");
const Io = std.Io;
const hashmod = @import("hash.zig");

pub const DEFAULT_RANGE_BYTES: u64 = 4 * 1024 * 1024;

pub const Manifest = struct {
    version: hashmod.Digest, // merkle root over range digests
    file_size: u64,
    range_size: u64,
    digests: []hashmod.Digest, // owned

    pub fn nRanges(self: *const Manifest) usize {
        return self.digests.len;
    }

    pub fn rangeLen(self: *const Manifest, i: usize) u64 {
        const off = @as(u64, i) * self.range_size;
        return @min(self.range_size, self.file_size - off);
    }

    pub fn deinit(self: *Manifest, gpa: std.mem.Allocator) void {
        gpa.free(self.digests);
    }
};

/// Compute the manifest of an existing (fully present) weight file.
pub fn buildManifest(gpa: std.mem.Allocator, io: Io, path: []const u8, range_size: u64) !Manifest {
    const f = try Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const file_size = (try f.stat(io)).size;
    if (file_size == 0) return error.EmptyFile;

    const n: usize = @intCast((file_size + range_size - 1) / range_size);
    const digests = try gpa.alloc(hashmod.Digest, n);
    errdefer gpa.free(digests);

    const buf = try gpa.alloc(u8, @intCast(range_size));
    defer gpa.free(buf);

    for (digests, 0..) |*d, i| {
        const off = @as(u64, i) * range_size;
        const len: usize = @intCast(@min(range_size, file_size - off));
        _ = try f.readPositionalAll(io, buf[0..len], off);
        d.* = hashmod.hashBlock(buf[0..len]);
    }

    return .{
        .version = try hashmod.merkleRoot(gpa, digests),
        .file_size = file_size,
        .range_size = range_size,
        .digests = digests,
    };
}

// ---- holdings bitmap ---------------------------------------------------------

pub const Holdings = struct {
    bits: []u8, // owned; ceil(n/8) bytes
    n: usize,

    pub fn initEmpty(gpa: std.mem.Allocator, n: usize) !Holdings {
        const bits = try gpa.alloc(u8, (n + 7) / 8);
        @memset(bits, 0);
        return .{ .bits = bits, .n = n };
    }

    pub fn initFull(gpa: std.mem.Allocator, n: usize) !Holdings {
        var h = try initEmpty(gpa, n);
        var i: usize = 0;
        while (i < n) : (i += 1) h.set(i);
        return h;
    }

    /// Random subset: each range held independently with probability `fraction`.
    /// Overlap across nodes with different seeds provides emergent redundancy.
    pub fn initRandom(gpa: std.mem.Allocator, n: usize, fraction: f32, seed: u64) !Holdings {
        var h = try initEmpty(gpa, n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        var i: usize = 0;
        while (i < n) : (i += 1) {
            if (rnd.float(f32) < fraction) h.set(i);
        }
        return h;
    }

    pub fn deinit(self: *Holdings, gpa: std.mem.Allocator) void {
        gpa.free(self.bits);
    }

    // set/has are atomic: the eager repair thread mutates holdings while P2P
    // connection threads read them concurrently.
    pub fn set(self: *Holdings, i: usize) void {
        _ = @atomicRmw(u8, &self.bits[i / 8], .Or, @as(u8, 1) << @intCast(i % 8), .monotonic);
    }

    pub fn has(self: *const Holdings, i: usize) bool {
        if (i >= self.n) return false;
        const b = @atomicLoad(u8, &self.bits[i / 8], .monotonic);
        return (b >> @intCast(i % 8)) & 1 == 1;
    }

    pub fn count(self: *const Holdings) usize {
        var c: usize = 0;
        for (self.bits) |*b| c += @popCount(@atomicLoad(u8, b, .monotonic));
        return c;
    }

    /// Hex encoding of the bitmap — the compact summary for ENR/gossip. Caller
    /// owns the result. For n ranges this is ceil(n/8)*2 chars.
    pub fn toHex(self: *const Holdings, gpa: std.mem.Allocator) ![]u8 {
        const out = try gpa.alloc(u8, self.bits.len * 2);
        const hex = "0123456789abcdef";
        for (self.bits, 0..) |b, i| {
            out[i * 2] = hex[b >> 4];
            out[i * 2 + 1] = hex[b & 0xf];
        }
        return out;
    }

    pub fn fromHex(gpa: std.mem.Allocator, s: []const u8, n: usize) !Holdings {
        const need = (n + 7) / 8;
        if (s.len != need * 2) return error.BadBitmapLength;
        const bits = try gpa.alloc(u8, need);
        errdefer gpa.free(bits);
        for (bits, 0..) |*b, i| {
            b.* = std.fmt.parseInt(u8, s[i * 2 ..][0..2], 16) catch return error.BadBitmapHex;
        }
        return .{ .bits = bits, .n = n };
    }
};

// ---- persistent store ----------------------------------------------------------

pub const Store = struct {
    gpa: std.mem.Allocator,
    io: Io,
    dir: []const u8, // owned
    manifest: Manifest,
    holdings: Holdings,
    /// The ranges this node *wants* to hold (its random subset). Holdings ⊆
    /// wanted; the eager repair loop works to close the gap.
    wanted: Holdings,
    file: Io.File, // model.gguf, open read/write

    pub fn deinit(self: *Store) void {
        self.file.close(self.io);
        self.manifest.deinit(self.gpa);
        self.holdings.deinit(self.gpa);
        self.wanted.deinit(self.gpa);
        self.gpa.free(self.dir);
    }

    /// Ranges still wanted but not held — what the repair loop chases.
    pub fn missingCount(self: *const Store) usize {
        var c: usize = 0;
        var i: usize = 0;
        while (i < self.manifest.nRanges()) : (i += 1) {
            if (self.wanted.has(i) and !self.holdings.has(i)) c += 1;
        }
        return c;
    }

    /// Read range `i` into `buf` (must be >= rangeLen(i)). Errors if not held.
    pub fn readRange(self: *Store, i: usize, buf: []u8) ![]u8 {
        if (!self.holdings.has(i)) return error.RangeNotHeld;
        const len: usize = @intCast(self.manifest.rangeLen(i));
        _ = try self.file.readPositionalAll(self.io, buf[0..len], @as(u64, i) * self.manifest.range_size);
        return buf[0..len];
    }

    /// Verify + write range `i`, marking it held. Rejects digest mismatches
    /// (the same free poison check the expert cache does).
    pub fn writeRange(self: *Store, i: usize, data: []const u8) !void {
        if (i >= self.manifest.nRanges()) return error.RangeOutOfBounds;
        if (data.len != self.manifest.rangeLen(i)) return error.BadRangeLength;
        const got = hashmod.hashBlock(data);
        if (!hashmod.eql(got, self.manifest.digests[i])) return error.RangeDigestMismatch;
        try self.file.writePositionalAll(self.io, data, @as(u64, i) * self.manifest.range_size);
        self.holdings.set(i);
    }

    pub fn saveSidecars(self: *Store) !void {
        try writeManifestFile(self.gpa, self.io, self.dir, &self.manifest);
        try writeBitmapFile(self.io, self.dir, "holdings.bitmap", &self.holdings);
        try writeBitmapFile(self.io, self.dir, "wanted.bitmap", &self.wanted);
    }
};

fn subPath(buf: []u8, dir: []const u8, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ dir, name });
}

fn writeManifestFile(gpa: std.mem.Allocator, io: Io, dir: []const u8, m: *const Manifest) !void {
    var text = std.ArrayList(u8).empty;
    defer text.deinit(gpa);
    try text.print(gpa, "version {s}\n", .{hashmod.toHex(m.version)});
    try text.print(gpa, "size {d}\n", .{m.file_size});
    try text.print(gpa, "range_size {d}\n", .{m.range_size});
    for (m.digests) |d| try text.print(gpa, "{s}\n", .{hashmod.toHex(d)});

    var pbuf: [4096]u8 = undefined;
    const p = try subPath(&pbuf, dir, "ranges.manifest");
    const f = try Io.Dir.cwd().createFile(io, p, .{ .truncate = true });
    defer f.close(io);
    try f.writeStreamingAll(io, text.items);
}

fn writeBitmapFile(io: Io, dir: []const u8, name: []const u8, h: *const Holdings) !void {
    var pbuf: [4096]u8 = undefined;
    const p = try subPath(&pbuf, dir, name);
    const f = try Io.Dir.cwd().createFile(io, p, .{ .truncate = true });
    defer f.close(io);
    try f.writeStreamingAll(io, h.bits);
}

fn loadBitmapFile(gpa: std.mem.Allocator, io: Io, dir: []const u8, name: []const u8, n: usize) !Holdings {
    var pbuf: [4096]u8 = undefined;
    const p = try subPath(&pbuf, dir, name);
    const f = try Io.Dir.cwd().openFile(io, p, .{});
    defer f.close(io);
    const size = (try f.stat(io)).size;
    if (size != (n + 7) / 8) return error.BadBitmapLength;
    const bits = try gpa.alloc(u8, @intCast(size));
    errdefer gpa.free(bits);
    _ = try f.readPositionalAll(io, bits, 0);
    return .{ .bits = bits, .n = n };
}

fn parseDigestHex(s: []const u8) !hashmod.Digest {
    if (s.len != 64) return error.BadDigestHex;
    var d: hashmod.Digest = undefined;
    for (&d, 0..) |*b, i| {
        b.* = std.fmt.parseInt(u8, s[i * 2 ..][0..2], 16) catch return error.BadDigestHex;
    }
    return d;
}

fn loadManifestFile(gpa: std.mem.Allocator, io: Io, dir: []const u8) !Manifest {
    var pbuf: [4096]u8 = undefined;
    const p = try subPath(&pbuf, dir, "ranges.manifest");
    const f = try Io.Dir.cwd().openFile(io, p, .{});
    defer f.close(io);
    const size = (try f.stat(io)).size;
    const text = try gpa.alloc(u8, size);
    defer gpa.free(text);
    _ = try f.readPositionalAll(io, text, 0);

    var it = std.mem.splitScalar(u8, text, '\n');
    const vline = it.next() orelse return error.BadManifest;
    if (!std.mem.startsWith(u8, vline, "version ")) return error.BadManifest;
    const version = try parseDigestHex(vline["version ".len..]);
    const sline = it.next() orelse return error.BadManifest;
    if (!std.mem.startsWith(u8, sline, "size ")) return error.BadManifest;
    const file_size = try std.fmt.parseInt(u64, sline["size ".len..], 10);
    const rline = it.next() orelse return error.BadManifest;
    if (!std.mem.startsWith(u8, rline, "range_size ")) return error.BadManifest;
    const range_size = try std.fmt.parseInt(u64, rline["range_size ".len..], 10);

    const n: usize = @intCast((file_size + range_size - 1) / range_size);
    const digests = try gpa.alloc(hashmod.Digest, n);
    errdefer gpa.free(digests);
    for (digests) |*d| {
        const line = it.next() orelse return error.BadManifest;
        d.* = try parseDigestHex(std.mem.trimEnd(u8, line, "\r"));
    }

    // integrity: recompute the root
    const root = try hashmod.merkleRoot(gpa, digests);
    if (!hashmod.eql(root, version)) return error.ManifestRootMismatch;

    return .{ .version = version, .file_size = file_size, .range_size = range_size, .digests = digests };
}

/// Open a store around an existing complete GGUF file (the origin/full holder).
/// The GGUF stays where it is (the store opens it in place); only the manifest
/// and full holdings bitmap sidecars are written into `store_dir`.
pub fn openFull(gpa: std.mem.Allocator, io: Io, gguf_path: []const u8, store_dir: []const u8, range_size: u64) !Store {
    try makePath(io, store_dir);
    var manifest = try buildManifest(gpa, io, gguf_path, range_size);
    errdefer manifest.deinit(gpa);
    var holdings = try Holdings.initFull(gpa, manifest.nRanges());
    errdefer holdings.deinit(gpa);
    var wanted = try Holdings.initFull(gpa, manifest.nRanges());
    errdefer wanted.deinit(gpa);

    const file = try Io.Dir.cwd().openFile(io, gguf_path, .{});
    var store = Store{
        .gpa = gpa,
        .io = io,
        .dir = try gpa.dupe(u8, store_dir),
        .manifest = manifest,
        .holdings = holdings,
        .wanted = wanted,
        .file = file,
    };
    try store.saveSidecars();
    return store;
}

/// Create an empty store from a manifest received from a peer (bootstrap path).
/// `model.gguf` is created in `store_dir`, sized, and filled by writeRange as
/// ranges arrive. Takes ownership of `manifest` and `wanted`.
pub fn createFromManifest(gpa: std.mem.Allocator, io: Io, store_dir: []const u8, manifest: Manifest, wanted: Holdings) !Store {
    try makePath(io, store_dir);
    var pbuf: [4096]u8 = undefined;
    const p = try subPath(&pbuf, store_dir, "model.gguf");
    const file = try Io.Dir.cwd().createFile(io, p, .{ .truncate = true, .read = true });
    errdefer file.close(io);
    try file.setLength(io, manifest.file_size);

    const holdings = try Holdings.initEmpty(gpa, manifest.nRanges());
    return .{
        .gpa = gpa,
        .io = io,
        .dir = try gpa.dupe(u8, store_dir),
        .manifest = manifest,
        .holdings = holdings,
        .wanted = wanted,
        .file = file,
    };
}

/// Reopen a previously synced store directory (manifest + bitmaps + model.gguf).
pub fn openDir(gpa: std.mem.Allocator, io: Io, store_dir: []const u8) !Store {
    var manifest = try loadManifestFile(gpa, io, store_dir);
    errdefer manifest.deinit(gpa);
    const n = manifest.nRanges();

    var holdings = try loadBitmapFile(gpa, io, store_dir, "holdings.bitmap", n);
    errdefer holdings.deinit(gpa);
    var wanted = loadBitmapFile(gpa, io, store_dir, "wanted.bitmap", n) catch |e| switch (e) {
        // older store without a wanted sidecar: wanted := current holdings
        error.FileNotFound => Holdings{ .bits = try gpa.dupe(u8, holdings.bits), .n = n },
        else => return e,
    };
    errdefer wanted.deinit(gpa);

    var pbuf: [4096]u8 = undefined;
    const mp = try subPath(&pbuf, store_dir, "model.gguf");
    const file = try Io.Dir.cwd().openFile(io, mp, .{ .mode = .read_write });

    return .{
        .gpa = gpa,
        .io = io,
        .dir = try gpa.dupe(u8, store_dir),
        .manifest = manifest,
        .holdings = holdings,
        .wanted = wanted,
        .file = file,
    };
}

fn makePath(io: Io, path: []const u8) !void {
    var i: usize = 1;
    while (i <= path.len) : (i += 1) {
        if (i != path.len and path[i] != '/') continue;
        const prefix = path[0..i];
        if (prefix.len == 0) continue;
        Io.Dir.cwd().createDir(io, prefix, .default_dir) catch |e| switch (e) {
            error.PathAlreadyExists => {},
            else => return e,
        };
    }
}

// ---- tests --------------------------------------------------------------------

test "holdings bitmap set/has/count and hex roundtrip" {
    const gpa = std.testing.allocator;
    var h = try Holdings.initEmpty(gpa, 20);
    defer h.deinit(gpa);
    h.set(0);
    h.set(7);
    h.set(8);
    h.set(19);
    try std.testing.expect(h.has(0) and h.has(7) and h.has(8) and h.has(19));
    try std.testing.expect(!h.has(1) and !h.has(18) and !h.has(20));
    try std.testing.expect(h.count() == 4);

    const hex = try h.toHex(gpa);
    defer gpa.free(hex);
    var h2 = try Holdings.fromHex(gpa, hex, 20);
    defer h2.deinit(gpa);
    try std.testing.expect(h2.has(0) and h2.has(7) and h2.has(8) and h2.has(19));
    try std.testing.expect(h2.count() == 4);
}

test "random holdings hit roughly the requested fraction" {
    const gpa = std.testing.allocator;
    var h = try Holdings.initRandom(gpa, 1000, 0.5, 99);
    defer h.deinit(gpa);
    const c = h.count();
    try std.testing.expect(c > 400 and c < 600);
    // different seed -> different (overlapping) subset: emergent redundancy
    var h2 = try Holdings.initRandom(gpa, 1000, 0.5, 100);
    defer h2.deinit(gpa);
    var overlap: usize = 0;
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        if (h.has(i) and h2.has(i)) overlap += 1;
    }
    try std.testing.expect(overlap > 100); // ~250 expected
}
