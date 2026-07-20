//! Range-sharded weight store for GGUF distribution (ROADMAP v1 #2, issue #2).
//!
//! Two sharding modes over one representation — every shard is an **extent
//! list** (byte spans of the GGUF file) with a SHA-256 digest; the Merkle root
//! over shard digests is the **model version id**:
//!
//!   fixed   fixed-size contiguous ranges (arch-oblivious fallback; one extent
//!           per shard). Used for non-MoE files and the demo fixture.
//!   expert  expert-aligned (the deployment mode): one shard per
//!           (layer, expert) covering its three slices of the 3D
//!           ffn_{gate,up,down}_exps tensors, plus the **resident bundle** —
//!           every non-expert byte of the file (header, attention, norms,
//!           shared experts, embeddings) chunked into ~16 MB shards that
//!           every node must hold. Every byte of the file belongs to exactly
//!           one shard, so the version root pins the whole file.
//!
//! Shard order in expert mode: resident chunks first (indices 0..n_resident),
//! then expert shards (layer-major, expert-minor). A fetched expert shard is
//! exactly the block a MoE matmul consumes (CLAUDE principle 7).

const std = @import("std");
const Io = std.Io;
const hashmod = @import("../core/hash.zig");
const gguf = @import("../gguf/gguf.zig");
const ggml = @import("../gguf/ggml.zig");

pub const DEFAULT_RANGE_BYTES: u64 = 4 * 1024 * 1024;
pub const RESIDENT_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

pub const Mode = enum { fixed, expert };

pub const Extent = struct { offset: u64, len: u64 };

pub const Manifest = struct {
    mode: Mode,
    version: hashmod.Digest, // merkle root over shard digests
    file_size: u64,
    range_size: u64, // fixed mode granularity (0 in expert mode)
    n_resident: usize, // expert mode: shards [0, n_resident) are mandatory
    digests: []hashmod.Digest, // owned
    extents: []Extent, // owned, flat
    extent_start: []u32, // owned, len nShards()+1; shard i owns [start[i], start[i+1])

    pub fn nRanges(self: *const Manifest) usize {
        return self.digests.len;
    }

    pub fn shardExtents(self: *const Manifest, i: usize) []const Extent {
        return self.extents[self.extent_start[i]..self.extent_start[i + 1]];
    }

    pub fn rangeLen(self: *const Manifest, i: usize) u64 {
        var n: u64 = 0;
        for (self.shardExtents(i)) |e| n += e.len;
        return n;
    }

    pub fn maxShardLen(self: *const Manifest) u64 {
        var m: u64 = 0;
        var i: usize = 0;
        while (i < self.nRanges()) : (i += 1) m = @max(m, self.rangeLen(i));
        return m;
    }

    pub fn deinit(self: *Manifest, gpa: std.mem.Allocator) void {
        gpa.free(self.digests);
        gpa.free(self.extents);
        gpa.free(self.extent_start);
    }

    /// Text serialization (the sidecar format and the MANIFESTFILE payload):
    ///   loomv2 <mode>
    ///   version <hex> / size <n> / range_size <n> / resident <n> / shards <n>
    ///   <digest_hex> <off>:<len>[,<off>:<len>...]   x n
    pub fn serialize(self: *const Manifest, gpa: std.mem.Allocator) ![]u8 {
        var text = std.ArrayList(u8).empty;
        errdefer text.deinit(gpa);
        try text.print(gpa, "loomv2 {s}\n", .{@tagName(self.mode)});
        try text.print(gpa, "version {s}\n", .{hashmod.toHex(self.version)});
        try text.print(gpa, "size {d}\n", .{self.file_size});
        try text.print(gpa, "range_size {d}\n", .{self.range_size});
        try text.print(gpa, "resident {d}\n", .{self.n_resident});
        try text.print(gpa, "shards {d}\n", .{self.nRanges()});
        var i: usize = 0;
        while (i < self.nRanges()) : (i += 1) {
            try text.print(gpa, "{s} ", .{hashmod.toHex(self.digests[i])});
            for (self.shardExtents(i), 0..) |e, k| {
                if (k != 0) try text.append(gpa, ',');
                try text.print(gpa, "{d}:{d}", .{ e.offset, e.len });
            }
            try text.append(gpa, '\n');
        }
        return text.toOwnedSlice(gpa);
    }
};

fn parseDigestHex(s: []const u8) !hashmod.Digest {
    if (s.len != 64) return error.BadDigestHex;
    var d: hashmod.Digest = undefined;
    for (&d, 0..) |*b, i| {
        b.* = std.fmt.parseInt(u8, s[i * 2 ..][0..2], 16) catch return error.BadDigestHex;
    }
    return d;
}

fn headerField(line: []const u8, key: []const u8) ![]const u8 {
    if (!std.mem.startsWith(u8, line, key) or line.len <= key.len or line[key.len] != ' ')
        return error.BadManifest;
    return line[key.len + 1 ..];
}

/// Parse `serialize` output. Verifies the Merkle root pins the digest set.
pub fn parseManifestBytes(gpa: std.mem.Allocator, bytes: []const u8) !Manifest {
    var it = std.mem.splitScalar(u8, bytes, '\n');
    const magic = it.next() orelse return error.BadManifest;
    const mode: Mode = if (std.mem.eql(u8, magic, "loomv2 fixed"))
        .fixed
    else if (std.mem.eql(u8, magic, "loomv2 expert"))
        .expert
    else
        return error.BadManifest;

    const version = try parseDigestHex(try headerField(it.next() orelse return error.BadManifest, "version"));
    const file_size = try std.fmt.parseInt(u64, try headerField(it.next() orelse return error.BadManifest, "size"), 10);
    const range_size = try std.fmt.parseInt(u64, try headerField(it.next() orelse return error.BadManifest, "range_size"), 10);
    const n_resident = try std.fmt.parseInt(usize, try headerField(it.next() orelse return error.BadManifest, "resident"), 10);
    const n = try std.fmt.parseInt(usize, try headerField(it.next() orelse return error.BadManifest, "shards"), 10);
    if (n == 0 or n > (1 << 26)) return error.BadManifest;

    const digests = try gpa.alloc(hashmod.Digest, n);
    errdefer gpa.free(digests);
    var extents = std.ArrayList(Extent).empty;
    errdefer extents.deinit(gpa);
    const extent_start = try gpa.alloc(u32, n + 1);
    errdefer gpa.free(extent_start);

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const line = std.mem.trimEnd(u8, it.next() orelse return error.BadManifest, "\r");
        const sp = std.mem.indexOfScalar(u8, line, ' ') orelse return error.BadManifest;
        digests[i] = try parseDigestHex(line[0..sp]);
        extent_start[i] = @intCast(extents.items.len);
        var eit = std.mem.splitScalar(u8, line[sp + 1 ..], ',');
        while (eit.next()) |tok| {
            if (tok.len == 0) continue;
            const colon = std.mem.indexOfScalar(u8, tok, ':') orelse return error.BadManifest;
            try extents.append(gpa, .{
                .offset = try std.fmt.parseInt(u64, tok[0..colon], 10),
                .len = try std.fmt.parseInt(u64, tok[colon + 1 ..], 10),
            });
        }
        if (extent_start[i] == extents.items.len) return error.BadManifest; // no extents
    }
    extent_start[n] = @intCast(extents.items.len);

    const root = try hashmod.merkleRoot(gpa, digests);
    if (!hashmod.eql(root, version)) return error.ManifestRootMismatch;

    return .{
        .mode = mode,
        .version = version,
        .file_size = file_size,
        .range_size = range_size,
        .n_resident = n_resident,
        .digests = digests,
        .extents = try extents.toOwnedSlice(gpa),
        .extent_start = extent_start,
    };
}

// ---- manifest building -------------------------------------------------------

fn digestShards(
    gpa: std.mem.Allocator,
    io: Io,
    file: Io.File,
    extents: []const Extent,
    extent_start: []const u32,
    digests: []hashmod.Digest,
) !void {
    // hash each shard's concatenated extent bytes
    var max_len: u64 = 0;
    for (digests, 0..) |_, i| {
        var l: u64 = 0;
        for (extents[extent_start[i]..extent_start[i + 1]]) |e| l += e.len;
        max_len = @max(max_len, l);
    }
    const buf = try gpa.alloc(u8, @intCast(max_len));
    defer gpa.free(buf);
    for (digests, 0..) |*d, i| {
        var pos: usize = 0;
        for (extents[extent_start[i]..extent_start[i + 1]]) |e| {
            _ = try file.readPositionalAll(io, buf[pos..][0..@intCast(e.len)], e.offset);
            pos += @intCast(e.len);
        }
        d.* = hashmod.hashBlock(buf[0..pos]);
    }
}

/// Fixed-size contiguous ranges (arch-oblivious fallback).
pub fn buildManifest(gpa: std.mem.Allocator, io: Io, path: []const u8, range_size: u64) !Manifest {
    const f = try Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const file_size = (try f.stat(io)).size;
    if (file_size == 0) return error.EmptyFile;

    const n: usize = @intCast((file_size + range_size - 1) / range_size);
    const digests = try gpa.alloc(hashmod.Digest, n);
    errdefer gpa.free(digests);
    const extents = try gpa.alloc(Extent, n);
    errdefer gpa.free(extents);
    const extent_start = try gpa.alloc(u32, n + 1);
    errdefer gpa.free(extent_start);

    for (extents, 0..) |*e, i| {
        const off = @as(u64, i) * range_size;
        e.* = .{ .offset = off, .len = @min(range_size, file_size - off) };
        extent_start[i] = @intCast(i);
    }
    extent_start[n] = @intCast(n);

    try digestShards(gpa, io, f, extents, extent_start, digests);

    return .{
        .mode = .fixed,
        .version = try hashmod.merkleRoot(gpa, digests),
        .file_size = file_size,
        .range_size = range_size,
        .n_resident = 0,
        .digests = digests,
        .extents = extents,
        .extent_start = extent_start,
    };
}

/// Expert-aligned sharding: parses the GGUF tensor table, emits one shard per
/// (layer, expert) over the 3D *_exps tensors, and chunks every remaining byte
/// of the file (the resident bundle) into RESIDENT_CHUNK_BYTES shards that
/// every node must hold. Errors with NoExpertTensors if the file has none.
pub fn buildExpertManifest(gpa: std.mem.Allocator, io: Io, path: []const u8) !Manifest {
    var parsed = try gguf.parse(gpa, io, path);
    defer parsed.deinit();

    const ExpTensor = struct { start: u64, per_expert: u64, n_expert: usize };
    const Layer = struct { gate: ?ExpTensor = null, up: ?ExpTensor = null, down: ?ExpTensor = null };

    // collect expert tensors by layer index
    var layers = std.AutoHashMap(usize, Layer).init(gpa);
    defer layers.deinit();
    var max_layer: usize = 0;

    for (parsed.tensors) |t| {
        if (t.dims.len != 3) continue;
        if (!std.mem.startsWith(u8, t.name, "blk.")) continue;
        const rest = t.name[4..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse continue;
        const layer_idx = std.fmt.parseInt(usize, rest[0..dot], 10) catch continue;
        const suffix = rest[dot + 1 ..];

        if (!ggml.Type.supported(t.ggml_type)) return error.UnsupportedTensorType;
        const ty: ggml.Type = @enumFromInt(t.ggml_type);
        const per: u64 = @intCast(t.dims[1] * ggml.rowBytes(ty, @intCast(t.dims[0])));
        const et = ExpTensor{
            .start = parsed.data_offset + t.offset,
            .per_expert = per,
            .n_expert = @intCast(t.dims[2]),
        };
        const g = try layers.getOrPut(layer_idx);
        if (!g.found_existing) g.value_ptr.* = .{};
        if (std.mem.eql(u8, suffix, "ffn_gate_exps.weight")) {
            g.value_ptr.gate = et;
        } else if (std.mem.eql(u8, suffix, "ffn_up_exps.weight")) {
            g.value_ptr.up = et;
        } else if (std.mem.eql(u8, suffix, "ffn_down_exps.weight")) {
            g.value_ptr.down = et;
        } else continue;
        max_layer = @max(max_layer, layer_idx);
    }
    if (layers.count() == 0) return error.NoExpertTensors;

    var extents = std.ArrayList(Extent).empty;
    errdefer extents.deinit(gpa);
    var starts = std.ArrayList(u32).empty;
    errdefer starts.deinit(gpa);

    // expert tensor regions, for computing the resident complement
    var regions = std.ArrayList(Extent).empty;
    defer regions.deinit(gpa);

    // resident shards first: complement of expert regions, chunked
    {
        var li: usize = 0;
        while (li <= max_layer) : (li += 1) {
            const l = layers.get(li) orelse continue;
            inline for (.{ l.gate, l.up, l.down }) |mt| {
                const t = mt orelse return error.IncompleteExpertLayer;
                try regions.append(gpa, .{ .offset = t.start, .len = t.per_expert * @as(u64, @intCast(t.n_expert)) });
            }
        }
        std.mem.sort(Extent, regions.items, {}, struct {
            fn lt(_: void, a: Extent, b: Extent) bool {
                return a.offset < b.offset;
            }
        }.lt);

        var pos: u64 = 0;
        var ri: usize = 0;
        while (pos < parsed.file_size) {
            const gap_end = if (ri < regions.items.len) regions.items[ri].offset else parsed.file_size;
            if (pos < gap_end) {
                // resident span [pos, gap_end): chunk it
                var p = pos;
                while (p < gap_end) {
                    const len = @min(RESIDENT_CHUNK_BYTES, gap_end - p);
                    try starts.append(gpa, @intCast(extents.items.len));
                    try extents.append(gpa, .{ .offset = p, .len = len });
                    p += len;
                }
            }
            if (ri < regions.items.len) {
                pos = regions.items[ri].offset + regions.items[ri].len;
                ri += 1;
            } else {
                pos = parsed.file_size;
            }
        }
    }
    const n_resident = starts.items.len;

    // expert shards: layer-major, expert-minor; extents = [gate, up, down] slices
    var li: usize = 0;
    while (li <= max_layer) : (li += 1) {
        const l = layers.get(li) orelse continue;
        const gate = l.gate.?;
        const up = l.up.?;
        const down = l.down.?;
        if (up.n_expert != gate.n_expert or down.n_expert != gate.n_expert) return error.IncompleteExpertLayer;
        var e: usize = 0;
        while (e < gate.n_expert) : (e += 1) {
            try starts.append(gpa, @intCast(extents.items.len));
            try extents.append(gpa, .{ .offset = gate.start + gate.per_expert * e, .len = gate.per_expert });
            try extents.append(gpa, .{ .offset = up.start + up.per_expert * e, .len = up.per_expert });
            try extents.append(gpa, .{ .offset = down.start + down.per_expert * e, .len = down.per_expert });
        }
    }

    const n = starts.items.len;
    try starts.append(gpa, @intCast(extents.items.len));
    const extent_start = try starts.toOwnedSlice(gpa);
    errdefer gpa.free(extent_start);
    const extents_flat = try extents.toOwnedSlice(gpa);
    errdefer gpa.free(extents_flat);

    const digests = try gpa.alloc(hashmod.Digest, n);
    errdefer gpa.free(digests);
    {
        const f = try Io.Dir.cwd().openFile(io, path, .{});
        defer f.close(io);
        try digestShards(gpa, io, f, extents_flat, extent_start, digests);
    }

    return .{
        .mode = .expert,
        .version = try hashmod.merkleRoot(gpa, digests),
        .file_size = parsed.file_size,
        .range_size = 0,
        .n_resident = n_resident,
        .digests = digests,
        .extents = extents_flat,
        .extent_start = extent_start,
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

    /// The bootstrap want-set: resident shards [0, n_resident) are always
    /// wanted; each expert shard independently with probability `fraction`.
    /// Overlap across nodes with different seeds provides emergent redundancy.
    pub fn initWanted(gpa: std.mem.Allocator, n: usize, n_resident: usize, fraction: f32, seed: u64) !Holdings {
        var h = try initEmpty(gpa, n);
        var i: usize = 0;
        while (i < n_resident) : (i += 1) h.set(i);
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
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
    /// owns the result.
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
    /// The shards this node *wants* to hold. Holdings ⊆ wanted; the eager
    /// repair loop works to close the gap. Resident shards are always wanted.
    wanted: Holdings,
    file: Io.File, // model.gguf, open read/write

    pub fn deinit(self: *Store) void {
        self.file.close(self.io);
        self.manifest.deinit(self.gpa);
        self.holdings.deinit(self.gpa);
        self.wanted.deinit(self.gpa);
        self.gpa.free(self.dir);
    }

    /// Shards still wanted but not held — what the repair loop chases.
    pub fn missingCount(self: *const Store) usize {
        var c: usize = 0;
        var i: usize = 0;
        while (i < self.manifest.nRanges()) : (i += 1) {
            if (self.wanted.has(i) and !self.holdings.has(i)) c += 1;
        }
        return c;
    }

    /// Read shard `i` (concatenated extents) into `buf`. Errors if not held.
    pub fn readRange(self: *Store, i: usize, buf: []u8) ![]u8 {
        if (!self.holdings.has(i)) return error.RangeNotHeld;
        var pos: usize = 0;
        for (self.manifest.shardExtents(i)) |e| {
            _ = try self.file.readPositionalAll(self.io, buf[pos..][0..@intCast(e.len)], e.offset);
            pos += @intCast(e.len);
        }
        return buf[0..pos];
    }

    /// Verify + write shard `i` (scattering to its extents), marking it held.
    /// Rejects digest mismatches (the free poison check).
    pub fn writeRange(self: *Store, i: usize, data: []const u8) !void {
        if (i >= self.manifest.nRanges()) return error.RangeOutOfBounds;
        if (data.len != self.manifest.rangeLen(i)) return error.BadRangeLength;
        const got = hashmod.hashBlock(data);
        if (!hashmod.eql(got, self.manifest.digests[i])) return error.RangeDigestMismatch;
        var pos: usize = 0;
        for (self.manifest.shardExtents(i)) |e| {
            try self.file.writePositionalAll(self.io, data[pos..][0..@intCast(e.len)], e.offset);
            pos += @intCast(e.len);
        }
        self.holdings.set(i);
    }

    pub fn saveSidecars(self: *Store) !void {
        const text = try self.manifest.serialize(self.gpa);
        defer self.gpa.free(text);
        try writeFileIn(self.io, self.dir, "ranges.manifest", text);
        try writeFileIn(self.io, self.dir, "holdings.bitmap", self.holdings.bits);
        try writeFileIn(self.io, self.dir, "wanted.bitmap", self.wanted.bits);
    }
};

fn subPath(buf: []u8, dir: []const u8, name: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ dir, name });
}

fn writeFileIn(io: Io, dir: []const u8, name: []const u8, bytes: []const u8) !void {
    var pbuf: [4096]u8 = undefined;
    const p = try subPath(&pbuf, dir, name);
    const f = try Io.Dir.cwd().createFile(io, p, .{ .truncate = true });
    defer f.close(io);
    try f.writeStreamingAll(io, bytes);
}

fn readFileIn(gpa: std.mem.Allocator, io: Io, dir: []const u8, name: []const u8) ![]u8 {
    var pbuf: [4096]u8 = undefined;
    const p = try subPath(&pbuf, dir, name);
    const f = try Io.Dir.cwd().openFile(io, p, .{});
    defer f.close(io);
    const size = (try f.stat(io)).size;
    const out = try gpa.alloc(u8, @intCast(size));
    errdefer gpa.free(out);
    _ = try f.readPositionalAll(io, out, 0);
    return out;
}

fn loadBitmapFile(gpa: std.mem.Allocator, io: Io, dir: []const u8, name: []const u8, n: usize) !Holdings {
    const bits = try readFileIn(gpa, io, dir, name);
    errdefer gpa.free(bits);
    if (bits.len != (n + 7) / 8) return error.BadBitmapLength;
    return .{ .bits = bits, .n = n };
}

/// Open a store around an existing complete GGUF file (the origin/full holder).
/// The GGUF stays where it is (opened in place); sidecars go to `store_dir`.
/// mode: .expert requires expert tensors; null = auto (expert if present).
pub fn openFull(gpa: std.mem.Allocator, io: Io, gguf_path: []const u8, store_dir: []const u8, range_size: u64, mode: ?Mode) !Store {
    try makePath(io, store_dir);
    var manifest: Manifest = switch (mode orelse .expert) {
        .expert => buildExpertManifest(gpa, io, gguf_path) catch |e| switch (e) {
            error.NoExpertTensors, error.NotGguf => if (mode == null)
                try buildManifest(gpa, io, gguf_path, range_size)
            else
                return e,
            else => return e,
        },
        .fixed => try buildManifest(gpa, io, gguf_path, range_size),
    };
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
/// Takes ownership of `manifest` and `wanted`.
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

/// Reopen a previously synced store directory.
pub fn openDir(gpa: std.mem.Allocator, io: Io, store_dir: []const u8) !Store {
    const mtext = try readFileIn(gpa, io, store_dir, "ranges.manifest");
    defer gpa.free(mtext);
    var manifest = try parseManifestBytes(gpa, mtext);
    errdefer manifest.deinit(gpa);
    const n = manifest.nRanges();

    var holdings = try loadBitmapFile(gpa, io, store_dir, "holdings.bitmap", n);
    errdefer holdings.deinit(gpa);
    var wanted = loadBitmapFile(gpa, io, store_dir, "wanted.bitmap", n) catch |e| switch (e) {
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

test "wanted set: resident always, experts by fraction" {
    const gpa = std.testing.allocator;
    var w = try Holdings.initWanted(gpa, 1000, 100, 0.5, 7);
    defer w.deinit(gpa);
    var i: usize = 0;
    while (i < 100) : (i += 1) try std.testing.expect(w.has(i)); // resident mandatory
    const expert_held = w.count() - 100;
    try std.testing.expect(expert_held > 350 and expert_held < 550); // ~450 of 900
}

test "manifest serialize/parse roundtrip with multi-extent shards" {
    const gpa = std.testing.allocator;
    var digests = [_]hashmod.Digest{ hashmod.hashBlock("a"), hashmod.hashBlock("b") };
    const version = try hashmod.merkleRoot(gpa, &digests);
    var extents = [_]Extent{
        .{ .offset = 0, .len = 100 },
        .{ .offset = 200, .len = 50 },
        .{ .offset = 300, .len = 10 },
    };
    var starts = [_]u32{ 0, 2, 3 }; // shard0 = 2 extents, shard1 = 1
    const m = Manifest{
        .mode = .expert,
        .version = version,
        .file_size = 310,
        .range_size = 0,
        .n_resident = 1,
        .digests = &digests,
        .extents = &extents,
        .extent_start = &starts,
    };
    const text = try m.serialize(gpa);
    defer gpa.free(text);

    var p = try parseManifestBytes(gpa, text);
    defer p.deinit(gpa);
    try std.testing.expect(p.mode == .expert);
    try std.testing.expect(p.n_resident == 1);
    try std.testing.expect(p.nRanges() == 2);
    try std.testing.expect(p.shardExtents(0).len == 2);
    try std.testing.expect(p.rangeLen(0) == 150);
    try std.testing.expect(p.rangeLen(1) == 10);
    try std.testing.expect(hashmod.eql(p.version, version));
}


test "expert manifest partitions the whole file: no gaps, no overlaps (issue #4.14)" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // a real (synthetic) deepseek2 GGUF with 3-D expert tensors + resident bytes
    const path = "test-partition.gguf";
    try gguf.writeDeepseekFixture(gpa, io, path, 7);
    defer Io.Dir.cwd().deleteFile(io, path) catch {};

    var m = try buildExpertManifest(gpa, io, path);
    defer m.deinit(gpa);

    // collect every extent across every shard, sort by offset, and assert the
    // union is exactly [0, file_size) with no overlaps.
    var extents = std.ArrayList(Extent).empty;
    defer extents.deinit(gpa);
    var i: usize = 0;
    while (i < m.nRanges()) : (i += 1) {
        for (m.shardExtents(i)) |e| try extents.append(gpa, e);
    }
    std.mem.sort(Extent, extents.items, {}, struct {
        fn lt(_: void, a: Extent, b: Extent) bool {
            return a.offset < b.offset;
        }
    }.lt);

    var cursor: u64 = 0;
    for (extents.items) |e| {
        try std.testing.expectEqual(cursor, e.offset); // no gap, no overlap
        cursor += e.len;
    }
    try std.testing.expectEqual(m.file_size, cursor); // covers to EOF

    // and the total sharded bytes equal the file size (independent check)
    var total: u64 = 0;
    i = 0;
    while (i < m.nRanges()) : (i += 1) total += m.rangeLen(i);
    try std.testing.expectEqual(m.file_size, total);
}
