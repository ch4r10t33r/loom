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
const builtin = @import("builtin");
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

    /// Longest *routed-expert* shard, i.e. ignoring the resident chunks.
    ///
    /// These differ by more than they look: resident chunks are fixed 16 MB
    /// slices of the dense prefix, while an expert shard is one expert's
    /// tensors, around 4.6 MB in a 16B model. A RAM cache that sizes every
    /// slot by the global maximum therefore wastes ~70% of its budget on
    /// padding and holds a third of the experts it could -- measured, 512
    /// slots for 1,664 expert shards and a 52% hit rate.
    pub fn maxExpertShardLen(self: *const Manifest) u64 {
        var m: u64 = 0;
        var i: usize = self.n_resident;
        while (i < self.nRanges()) : (i += 1) {
            const n = self.rangeLen(i);
            if (n > m) m = n;
        }
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
    // A declared shard count is honoured before any shard line is read, so a
    // ~200-byte manifest claiming 67M shards would allocate 2.3 GB and then
    // fail (security issue #29). Every shard line is at least
    // "<64 hex digest> <offset>:<len>\n" = 68 bytes, so bound n by what the
    // payload could possibly contain.
    const MIN_SHARD_LINE = 68;
    if (n > bytes.len / MIN_SHARD_LINE + 1) return error.BadManifest;

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
    if (n_resident > n) return error.BadManifest;

    // layout integrity: exact partition of the file + version binds the layout
    try validatePartition(gpa, extents.items, file_size);
    const root = try computeVersion(gpa, mode, file_size, n_resident, digests, extents.items, extent_start);
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

/// Version id = Merkle root over per-shard *layout-committing* leaves plus a
/// header leaf binding global fields. Binding extents/file_size/n_resident/mode
/// into the root means a peer cannot serve digests that verify while pointing
/// them at attacker-chosen offsets (audit #5 P0-1).
pub fn computeVersion(
    gpa: std.mem.Allocator,
    mode: Mode,
    file_size: u64,
    n_resident: usize,
    digests: []const hashmod.Digest,
    extents: []const Extent,
    extent_start: []const u32,
) !hashmod.Digest {
    var leaves = try gpa.alloc(hashmod.Digest, digests.len + 1);
    defer gpa.free(leaves);

    var hdr = std.ArrayList(u8).empty;
    defer hdr.deinit(gpa);
    try hdr.append(gpa, @intFromEnum(mode));
    var t8: [8]u8 = undefined;
    std.mem.writeInt(u64, &t8, file_size, .little);
    try hdr.appendSlice(gpa, &t8);
    std.mem.writeInt(u64, &t8, @intCast(n_resident), .little);
    try hdr.appendSlice(gpa, &t8);
    std.mem.writeInt(u64, &t8, @intCast(digests.len), .little);
    try hdr.appendSlice(gpa, &t8);
    leaves[0] = hashmod.hashBlock(hdr.items);

    for (digests, 0..) |d, i| {
        var leaf = std.ArrayList(u8).empty;
        defer leaf.deinit(gpa);
        try leaf.appendSlice(gpa, &d);
        for (extents[extent_start[i]..extent_start[i + 1]]) |e| {
            std.mem.writeInt(u64, &t8, e.offset, .little);
            try leaf.appendSlice(gpa, &t8);
            std.mem.writeInt(u64, &t8, e.len, .little);
            try leaf.appendSlice(gpa, &t8);
        }
        leaves[i + 1] = hashmod.hashBlock(leaf.items);
    }
    return hashmod.merkleRoot(gpa, leaves);
}

/// Every byte of [0, file_size) belongs to exactly one shard extent: no gaps,
/// no overlaps, no out-of-bounds (audit #5 P0-1). O(total extents log).
pub fn validatePartition(gpa: std.mem.Allocator, extents: []const Extent, file_size: u64) !void {
    const sorted = try gpa.dupe(Extent, extents);
    defer gpa.free(sorted);
    std.mem.sort(Extent, sorted, {}, struct {
        fn lt(_: void, a: Extent, b: Extent) bool {
            return a.offset < b.offset;
        }
    }.lt);
    var cursor: u64 = 0;
    for (sorted) |e| {
        if (e.len == 0) return error.EmptyExtent;
        if (e.offset != cursor) return error.ManifestNotAPartition; // gap or overlap
        const end = std.math.add(u64, e.offset, e.len) catch return error.ExtentOverflow;
        if (end > file_size) return error.ExtentPastEof;
        cursor = end;
    }
    if (cursor != file_size) return error.ManifestNotAPartition; // does not cover EOF
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
        .version = try computeVersion(gpa, .fixed, file_size, 0, digests, extents, extent_start),
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
        .version = try computeVersion(gpa, .expert, parsed.file_size, n_resident, digests, extents_flat, extent_start),
        .file_size = parsed.file_size,
        .range_size = 0,
        .n_resident = n_resident,
        .digests = digests,
        .extents = extents_flat,
        .extent_start = extent_start,
    };
}

// ---- holdings bitmap ---------------------------------------------------------

/// Blocks actually allocated to `fd`, in bytes, or null where the platform is
/// not covered. Distinct from the file's length: a hole-punched file keeps its
/// length and gives up its blocks, and that difference is the only way to see
/// whether eviction reached the disk.
fn allocatedBytes(fd: std.posix.fd_t) ?u64 {
    switch (builtin.os.tag) {
        .macos, .ios, .tvos, .watchos => {
            var st: std.c.Stat = undefined;
            if (std.c.fstat(fd, &st) != 0) return null;
            return @as(u64, @intCast(st.blocks)) * 512;
        },
        .linux => {
            // `std.c.Stat` is libc's layout and this binary may be musl or
            // glibc; statx is the kernel interface and is stable across both.
            var stx: std.os.linux.Statx = undefined;
            const rc = std.os.linux.statx(fd, "", std.os.linux.AT.EMPTY_PATH, .{ .BLOCKS = true }, &stx);
            if (std.os.linux.errno(rc) != .SUCCESS) return null;
            return @as(u64, stx.blocks) * 512;
        },
        else => return null,
    }
}

/// Release a byte range's blocks back to the filesystem, leaving a hole that
/// reads back as zeros without changing the file's length.
///
/// Best-effort by design: this is what turns "the node no longer holds this
/// shard" from a bit in a bitmap into free space on the disk, but a filesystem
/// that cannot punch holes is not an error -- the node is still correct, it
/// just does not get the space back. Nothing reads an evicted range: the
/// holdings bit is cleared first, and every read path checks it.
fn punchHole(fd: std.posix.fd_t, offset: u64, len: u64) void {
    if (len == 0) return;
    // Align inward to whole filesystem blocks. Shard extents start and end at
    // arbitrary byte offsets, and a partially-covered block cannot be freed --
    // some of it belongs to the neighbouring shard, which is very likely still
    // held. APFS rejects an unaligned range outright (EINVAL), so the first
    // version of this punched nothing at all and the store kept growing while
    // the holdings count stayed flat, which is exactly as confusing as it
    // sounds. Losing up to one block at each end is the whole cost.
    const block: u64 = 4096;
    const start = std.mem.alignForward(u64, offset, block);
    const end = std.mem.alignBackward(u64, offset + len, block);
    if (end <= start) return; // shard smaller than a block, or spanning none
    const off = start;
    const n = end - start;
    switch (builtin.os.tag) {
        .macos, .ios, .tvos, .watchos => {
            // Not in std.c: fpunchhole_t from <sys/fcntl.h>. The two leading
            // u32s are a flags word and explicit padding to keep fp_offset
            // 8-aligned, so they cannot be collapsed.
            const punchhole = extern struct {
                fp_flags: u32,
                reserved: u32,
                fp_offset: i64,
                fp_length: i64,
            };
            var arg = punchhole{
                .fp_flags = 0,
                .reserved = 0,
                .fp_offset = @intCast(off),
                .fp_length = @intCast(n),
            };
            _ = std.c.fcntl(fd, std.c.F.PUNCHHOLE, &arg);
        },
        .linux => {
            const PUNCH_HOLE = 0x02;
            const KEEP_SIZE = 0x01;
            _ = std.os.linux.fallocate(fd, PUNCH_HOLE | KEEP_SIZE, @intCast(off), @intCast(n));
        },
        else => {},
    }
}

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
    /// Choose this node's want-set: all resident shards, plus a random subset of
    /// the expert shards sized by `fraction`.
    ///
    /// The count is exact — `round(fraction * expert_shards)` — and only *which*
    /// shards get picked is random. An earlier version flipped an independent
    /// coin per shard, which makes the count binomial: with 5 shards and
    /// `--hold-fraction 0.4` that returned 0, 1, or 2 depending on the seed, and
    /// on the default seed it returned 0, i.e. a node holding nothing. At
    /// GLM scale (~19k shards) the binomial concentrates and nobody notices, but
    /// the flag should mean what it says at any size.
    pub fn initWanted(gpa: std.mem.Allocator, n: usize, n_resident: usize, fraction: f32, seed: u64) !Holdings {
        var h = try initEmpty(gpa, n);
        var i: usize = 0;
        while (i < n_resident) : (i += 1) h.set(i); // resident bundle is mandatory
        const pool = n - n_resident;
        if (pool == 0) return h;

        const frac = std.math.clamp(fraction, 0.0, 1.0);
        var want: usize = @intFromFloat(@round(@as(f64, @floatFromInt(pool)) * @as(f64, frac)));
        // asking for a non-zero fraction should never yield an empty node
        if (want == 0 and frac > 0) want = 1;
        if (want > pool) want = pool;
        if (want == pool) {
            while (i < n) : (i += 1) h.set(i);
            return h;
        }

        // partial Fisher-Yates over the expert-shard indices: exact count,
        // uniformly chosen, seeded so a restart re-picks the same set
        const idx = try gpa.alloc(usize, pool);
        defer gpa.free(idx);
        for (idx, 0..) |*v, k| v.* = n_resident + k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        var k: usize = 0;
        while (k < want) : (k += 1) {
            const j = k + rnd.uintLessThan(usize, pool - k);
            const tmp = idx[k];
            idx[k] = idx[j];
            idx[j] = tmp;
            h.set(idx[k]);
        }
        return h;
    }

    pub fn deinit(self: *Holdings, gpa: std.mem.Allocator) void {
        gpa.free(self.bits);
    }

    // set/has are atomic: the eager repair thread mutates holdings while P2P
    // connection threads read them concurrently.
    /// Drop every bit, keeping the allocation.
    pub fn clearAll(self: *Holdings) void {
        @memset(self.bits, 0);
    }

    pub fn set(self: *Holdings, i: usize) void {
        _ = @atomicRmw(u8, &self.bits[i / 8], .Or, @as(u8, 1) << @intCast(i % 8), .monotonic);
    }

    pub fn clear(self: *Holdings, i: usize) void {
        _ = @atomicRmw(u8, &self.bits[i / 8], .And, ~(@as(u8, 1) << @intCast(i % 8)), .monotonic);
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
        // atomic per byte: other threads RMW these bits concurrently, and
        // mixing plain loads with atomic RMWs on one location is a data race
        // (security issue #31)
        for (self.bits, 0..) |*bp, i| {
            const b = @atomicLoad(u8, bp, .monotonic);
            out[i * 2] = hex[b >> 4];
            out[i * 2 + 1] = hex[b & 0xf];
        }
        return out;
    }

    /// Consistent copy of the bitmap for advertising or hashing. Reading
    /// `bits` directly races the atomic RMWs done by set/clear, which yields a
    /// torn bitmap and a digest that does not match what was advertised
    /// (security issue #31). Caller owns the result.
    pub fn snapshotAlloc(self: *const Holdings, gpa: std.mem.Allocator) ![]u8 {
        const out = try gpa.alloc(u8, self.bits.len);
        for (self.bits, 0..) |*bp, i| out[i] = @atomicLoad(u8, bp, .monotonic);
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
    /// Truly monotonic sequence, bumped on every holdings mutation. Advertised
    /// on heartbeat/gossip so a stale bitmap can never clobber a fresh one —
    /// popcount is NOT monotonic now that verify-failures clear bits (#7 P1).
    seq: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    /// Guards `saveSidecars` only (see there). Never held across network I/O.
    sidecar_mutex: Io.Mutex = .init,
    /// Serializes cap enforcement. Shard *writes* need no lock -- they touch
    /// disjoint extents and flip one bit -- but choosing a victim reads the
    /// whole bitmap and then clears a bit, and two fetchers doing that at once
    /// can evict twice or pick the same victim.
    evict_mutex: Io.Mutex = .init,
    /// Most expert shards this store will hold. Resident chunks are mandatory,
    /// so they are neither counted nor evictable. Null means uncapped, which
    /// is what an origin holding a complete copy wants.
    ///
    /// Without this `--hold-fraction` bounded only the bootstrap fetch: a shard
    /// pulled from a peer at token time was persisted and marked held, nothing
    /// evicted it, and a node measured on a two-machine run went from 3.6% to
    /// 93.1% of the corpus while generating 24 tokens. That makes every serving
    /// node converge on a full replica and removes the reason to distribute the
    /// corpus at all.
    cap_experts: ?usize = null,
    /// Use counter per shard, for picking the coldest victim. Lazily allocated
    /// because an uncapped store never needs it.
    last_use: ?[]u32 = null,
    use_clock: u32 = 0,
    /// Read-only mapping of `file`, when one could be made. The token path
    /// reads weights straight out of this instead of copying them into a heap
    /// cache, which is both a copy and a second resident copy of bytes the
    /// page cache already holds. Measured on a 16 GB machine serving an 8.9 GB
    /// model, an 8 GB heap cache made the expert matmuls 2.8x slower (65 ->
    /// 180 ms/token) purely through memory pressure, while making the reads
    /// look fast -- the cost moved from the read to whoever next touched the
    /// pages.
    map: ?[]align(std.heap.page_size_min) const u8 = null,
    /// Shards whose digest has been checked against this mapping. Verifying on
    /// first use still catches a poisoned block (audit #5 P0-2); re-hashing
    /// ~950 MB per token to re-confirm bytes we already vouched for does not.
    /// Reset wholesale whenever `seq` moves, i.e. whenever a shard is written.
    /// Shards whose digest has been checked this session. Allocated lazily on
    /// first use so it applies to the pread path as well as the mapped one.
    verified: ?Holdings = null,
    verified_seq: u64 = 0,

    /// Map the backing file read-only. Best-effort: a store that cannot be
    /// mapped simply keeps using pread, so callers need no fallback of their
    /// own.
    pub fn mapReadOnly(self: *Store) void {
        if (self.map != null) return;
        const len = self.manifest.file_size;
        if (len == 0) return;
        const m = std.posix.mmap(
            null,
            @intCast(len),
            .{ .READ = true },
            .{ .TYPE = .SHARED },
            self.file.handle,
            0,
        ) catch return;
        self.map = m;
        self.verified = Holdings.initEmpty(self.gpa, self.manifest.nRanges()) catch {
            std.posix.munmap(m);
            self.map = null;
            return;
        };
        self.verified_seq = self.holdingsSeq();
    }

    /// The whole read-only mapping, for a compute backend that wants to make
    /// the file device-resident in one allocation rather than per extent.
    pub fn mapping(self: *const Store) ?[]const u8 {
        return self.map;
    }

    /// Extent `k` of shard `i` as a slice of the mapping, or null when the
    /// store is not mapped, the shard is not held, or it has not verified.
    ///
    /// The caller gets a pointer into the page cache: no copy, and the pages
    /// are shared with every other reader of the file rather than duplicated.
    pub fn extentSlice(self: *Store, i: usize, k: usize) ?[]const u8 {
        const map = self.map orelse return null;
        if (!self.holdings.has(i)) return null;
        if (!self.ensureVerified(i)) return null;
        const ex = self.manifest.shardExtents(i);
        if (k >= ex.len) return null;
        const off: usize = @intCast(ex[k].offset);
        const len: usize = @intCast(ex[k].len);
        if (off + len > map.len) return null;
        return map[off..][0..len];
    }

    /// Number of extents shard `i` has, so a caller can check the layout it
    /// expects before taking the zero-copy path.
    pub fn extentCount(self: *const Store, i: usize) usize {
        return self.manifest.shardExtents(i).len;
    }

    /// Hash shard `i` out of the mapping once, remembering the result. A
    /// mismatch clears the holdings bit exactly as `readRangeVerified` does,
    /// so eager repair re-fetches.
    /// True if shard `i` has already been verified this session. Also handles
    /// the generation reset, so callers need not.
    fn markVerifiedOnce(self: *Store, i: usize) bool {
        const seq = self.holdingsSeq();
        if (self.verified) |*v| {
            if (seq != self.verified_seq) {
                v.clearAll();
                self.verified_seq = seq;
                return false;
            }
            return v.has(i);
        }
        self.verified = Holdings.initEmpty(self.gpa, self.manifest.nRanges()) catch return false;
        self.verified_seq = seq;
        return false;
    }

    fn setVerified(self: *Store, i: usize) void {
        if (self.verified) |*v| v.set(i);
    }

    fn ensureVerified(self: *Store, i: usize) bool {
        var v = &(self.verified orelse return false);
        const seq = self.holdingsSeq();
        if (seq != self.verified_seq) {
            v.clearAll();
            self.verified_seq = seq;
        }
        if (v.has(i)) return true;

        const map = self.map orelse return false;
        var h = hashmod.blockHasher();
        for (self.manifest.shardExtents(i)) |e| {
            const off: usize = @intCast(e.offset);
            const len: usize = @intCast(e.len);
            if (off + len > map.len) return false;
            h.update(map[off..][0..len]);
        }
        var got: [32]u8 = undefined;
        h.final(&got);
        if (!hashmod.eql(got, self.manifest.digests[i])) {
            self.holdings.clear(i);
            _ = self.seq.fetchAdd(1, .monotonic);
            return false;
        }
        v.set(i);
        return true;
    }

    pub fn holdingsSeq(self: *Store) u64 {
        return self.seq.load(.monotonic);
    }

    pub fn deinit(self: *Store) void {
        self.file.close(self.io);
        if (self.last_use) |lu| self.gpa.free(lu);
        // Allocated lazily on the first verify, not only by `mapReadOnly`.
        if (self.verified) |*v| v.deinit(self.gpa);
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

    /// Read + re-hash shard `i` against its manifest digest before returning it
    /// (audit #5 P0-2): defends the local hot path and GETR/ExpertRequest serve
    /// against bitrot, torn pages, and a claimed-but-hole bitmap. On mismatch
    /// the bit is cleared so eager repair re-fetches a good copy.
    pub fn readRangeVerified(self: *Store, i: usize, buf: []u8) ![]u8 {
        const data = try self.readRange(i, buf);
        // Hash once per shard per session, not on every read.
        //
        // The property being defended (audit #5 P0-2) is that a poisoned or
        // bitrotted block on disk never reaches a matmul. Checking on first
        // read establishes that; checking again on the ten-thousandth read of
        // the same bytes, from the same file, in the same process, establishes
        // nothing new -- and it is not cheap. Measured on DeepSeek-V2-Lite,
        // 27 shards per token miss the RAM cache and each one hashes 4.6 MB,
        // about 80 ms/token of the 249 ms spent in `get`.
        //
        // The bit resets whenever `seq` moves, i.e. whenever any shard is
        // written or a verification fails, so a repaired or refetched shard is
        // always re-checked.
        if (self.markVerifiedOnce(i)) return data;
        const got = hashmod.hashBlock(data);
        if (!hashmod.eql(got, self.manifest.digests[i])) {
            self.holdings.clear(i);
            _ = self.seq.fetchAdd(1, .monotonic);
            return error.RangeDigestMismatch;
        }
        self.setVerified(i);
        return data;
    }

    /// Verify all currently-held shards; clear bits that fail (audit #5 P0-5).
    /// Returns the number of shards that failed and were cleared.
    pub fn auditHeld(self: *Store, scratch: []u8) !usize {
        var failed: usize = 0;
        var i: usize = 0;
        while (i < self.manifest.nRanges()) : (i += 1) {
            if (!self.holdings.has(i)) continue;
            _ = self.readRangeVerified(i, scratch) catch {
                failed += 1;
            };
        }
        return failed;
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
        _ = self.seq.fetchAdd(1, .monotonic);
        self.touch(i);
        self.enforceCap();
    }

    /// Cap this store to `n` expert shards, or uncap it with null.
    pub fn setCap(self: *Store, n: ?usize) void {
        self.cap_experts = n;
    }

    /// Note that shard `i` was used, so eviction can prefer colder ones.
    pub fn touch(self: *Store, i: usize) void {
        if (self.cap_experts == null) return;
        const lu = self.last_use orelse blk: {
            const b = self.gpa.alloc(u32, self.manifest.nRanges()) catch return;
            @memset(b, 0);
            self.last_use = b;
            break :blk b;
        };
        if (i >= lu.len) return;
        self.use_clock +%= 1;
        lu[i] = self.use_clock;
    }

    /// Expert shards currently held. Resident chunks are excluded: they are
    /// mandatory, so counting them would make the cap mean different things on
    /// models with different dense prefixes.
    pub fn heldExperts(self: *const Store) usize {
        var c: usize = 0;
        var i: usize = self.manifest.n_resident;
        while (i < self.manifest.nRanges()) : (i += 1) {
            if (self.holdings.has(i)) c += 1;
        }
        return c;
    }

    /// Stop holding shard `i`: clear the bit so it is neither read nor
    /// advertised, and release its blocks back to the filesystem.
    ///
    /// Clearing the bit is what makes the cap real; punching the hole is what
    /// makes it visible in `df`. The second is best-effort -- a filesystem
    /// without hole punching leaves the blocks allocated, and the node still
    /// behaves as though it does not hold the shard.
    pub fn evictRange(self: *Store, i: usize) void {
        if (i < self.manifest.n_resident) return; // mandatory
        if (!self.holdings.has(i)) return;
        self.holdings.clear(i);
        _ = self.seq.fetchAdd(1, .monotonic);
        for (self.manifest.shardExtents(i)) |e| punchHole(self.file.handle, e.offset, e.len);
        if (self.last_use) |lu| {
            if (i < lu.len) lu[i] = 0;
        }
    }

    /// Evict coldest-first until the cap is met. No-op when uncapped.
    pub fn enforceCap(self: *Store) void {
        const cap = self.cap_experts orelse return;
        self.evict_mutex.lockUncancelable(self.io);
        defer self.evict_mutex.unlock(self.io);
        while (self.heldExperts() > cap) {
            const victim = self.coldestHeld() orelse return;
            self.evictRange(victim);
        }
    }

    fn coldestHeld(self: *const Store) ?usize {
        var best: ?usize = null;
        var best_use: u32 = std.math.maxInt(u32);
        var i: usize = self.manifest.n_resident;
        while (i < self.manifest.nRanges()) : (i += 1) {
            if (!self.holdings.has(i)) continue;
            // Never used this session sorts coldest, which is what we want: a
            // shard fetched at bootstrap and never routed to is exactly the
            // one worth giving up.
            const u = if (self.last_use) |lu| (if (i < lu.len) lu[i] else 0) else 0;
            if (u < best_use) {
                best_use = u;
                best = i;
            }
        }
        return best;
    }

    /// Persist the manifest + bitmaps. Self-serializing (security issue #25):
    /// this is the only store mutation that is NOT safe to run concurrently
    /// (three whole-file rewrites), so it takes its own lock rather than
    /// forcing callers to hold a coarse lock across unrelated work. Shard
    /// writes need no lock: `writeRange` digest-verifies, writes disjoint
    /// extents, and flips an atomic holdings bit.
    pub fn saveSidecars(self: *Store) !void {
        self.sidecar_mutex.lockUncancelable(self.io);
        defer self.sidecar_mutex.unlock(self.io);
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

    var store = Store{
        .gpa = gpa,
        .io = io,
        .dir = try gpa.dupe(u8, store_dir),
        .manifest = manifest,
        .holdings = holdings,
        .wanted = wanted,
        .file = file,
    };
    // audit #5 P0-5: never trust a persisted holdings bitmap — re-hash every
    // claimed shard on open; failures are cleared (eager repair re-fetches).
    const scratch = try gpa.alloc(u8, @intCast(store.manifest.maxShardLen()));
    defer gpa.free(scratch);
    const failed = store.auditHeld(scratch) catch 0;
    if (failed > 0) {
        std.debug.print("store audit: {d} held shard(s) failed digest, cleared\n", .{failed});
        store.saveSidecars() catch {};
    }
    return store;
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
    // a real partition of [0,160): shard0 = {[0,100),[100,50)}, shard1 = [150,10)
    var extents = [_]Extent{
        .{ .offset = 0, .len = 100 },
        .{ .offset = 100, .len = 50 },
        .{ .offset = 150, .len = 10 },
    };
    var starts = [_]u32{ 0, 2, 3 }; // shard0 = 2 extents, shard1 = 1
    const version = try computeVersion(gpa, .expert, 160, 1, &digests, &extents, &starts);
    const m = Manifest{
        .mode = .expert,
        .version = version,
        .file_size = 160,
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
    try std.testing.expect(p.file_size == 160);
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

test "malicious manifest extents rejected on parse (audit #5 P0-1)" {
    const gpa = std.testing.allocator;
    var digests = [_]hashmod.Digest{ hashmod.hashBlock("a"), hashmod.hashBlock("b") };
    // overlapping extents: shard1 points back into shard0's bytes
    var extents = [_]Extent{
        .{ .offset = 0, .len = 100 },
        .{ .offset = 50, .len = 50 }, // overlap with shard0
    };
    var starts = [_]u32{ 0, 1, 2 };
    const version = try computeVersion(gpa, .expert, 100, 1, &digests, &extents, &starts);
    const m = Manifest{
        .mode = .expert,
        .version = version,
        .file_size = 100,
        .range_size = 0,
        .n_resident = 1,
        .digests = &digests,
        .extents = &extents,
        .extent_start = &starts,
    };
    const text = try m.serialize(gpa);
    defer gpa.free(text);
    try std.testing.expectError(error.ManifestNotAPartition, parseManifestBytes(gpa, text));

    // tampered extents whose Merkle-verifying digests still parse: flip an
    // offset in the serialized text but keep the (now-wrong) version -> the
    // recomputed layout-committed root no longer matches
    var digests2 = [_]hashmod.Digest{hashmod.hashBlock("x")};
    var ex2 = [_]Extent{.{ .offset = 0, .len = 64 }};
    var st2 = [_]u32{ 0, 1 };
    const v2 = try computeVersion(gpa, .expert, 64, 0, &digests2, &ex2, &st2);
    const m2 = Manifest{ .mode = .expert, .version = v2, .file_size = 64, .range_size = 0, .n_resident = 0, .digests = &digests2, .extents = &ex2, .extent_start = &st2 };
    const t2 = try m2.serialize(gpa);
    defer gpa.free(t2);
    // corrupt the extent "0:64" -> "0:63" (past-EOF/short) without touching version
    const bad = try std.mem.replaceOwned(u8, gpa, t2, "0:64", "0:60");
    defer gpa.free(bad);
    try std.testing.expectError(error.ManifestNotAPartition, parseManifestBytes(gpa, bad));
}

test "initWanted picks an exact count, not a binomial draw" {
    const gpa = std.testing.allocator;
    // 5 shards, no resident, 40% -> exactly 2, for EVERY seed
    for ([_]u64{ 1, 2, 3, 42, 99, 12345 }) |seed| {
        var h = try Holdings.initWanted(gpa, 5, 0, 0.4, seed);
        defer h.deinit(gpa);
        try std.testing.expectEqual(@as(usize, 2), h.count());
    }
    // resident shards are always wanted, and count toward neither the pool nor
    // the fraction
    var h2 = try Holdings.initWanted(gpa, 10, 4, 0.5, 7); // 4 resident + 3 of 6
    defer h2.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 7), h2.count());
    var r: usize = 0;
    while (r < 4) : (r += 1) try std.testing.expect(h2.has(r));

    // a non-zero fraction must never yield an empty node
    var h3 = try Holdings.initWanted(gpa, 100, 0, 0.001, 42);
    defer h3.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), h3.count());

    // the extremes
    var h4 = try Holdings.initWanted(gpa, 8, 0, 1.0, 42);
    defer h4.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 8), h4.count());
    var h5 = try Holdings.initWanted(gpa, 8, 0, 0.0, 42);
    defer h5.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), h5.count());
}

test "hold cap: holdings never exceed it, resident chunks survive, coldest goes first" {
    // The invariant the two-machine run showed missing. Without a cap,
    // --hold-fraction bounded the bootstrap fetch only and a node grew from
    // 3.6% to 93.1% of the corpus while serving 24 tokens, because every shard
    // fetched at token time was persisted and nothing evicted it.
    const gpa = std.testing.allocator;
    var thr: std.Io.Threaded = .init(gpa, .{});
    defer thr.deinit();
    const io = thr.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dir_buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});

    const n_resident = 2;
    const n_expert = 40; // >> cap, so the disk bound below discriminates
    const n = n_resident + n_expert;
    const each: u64 = 64 * 1024; // > one filesystem block, or nothing is punched

    var blob = try gpa.alloc(u8, @intCast(each * n));
    defer gpa.free(blob);
    for (blob, 0..) |*b, k| b.* = @truncate(k *% 31);

    const digests = try gpa.alloc(hashmod.Digest, n);
    const extents = try gpa.alloc(Extent, n);
    const starts = try gpa.alloc(u32, n + 1);
    for (0..n) |i| {
        const off = each * @as(u64, i);
        digests[i] = hashmod.hashBlock(blob[@intCast(off)..][0..@intCast(each)]);
        extents[i] = .{ .offset = off, .len = each };
        starts[i] = @intCast(i);
    }
    starts[n] = @intCast(n);
    const total = each * n;
    const version = try computeVersion(gpa, .expert, total, 0, digests, extents, starts);
    const manifest = Manifest{
        .mode = .expert,
        .version = version,
        .file_size = total,
        .range_size = 0,
        .n_resident = n_resident,
        .digests = digests,
        .extents = extents,
        .extent_start = starts,
    };
    const wanted = try Holdings.initFull(gpa, n);
    var store = try createFromManifest(gpa, io, dir, manifest, wanted);
    defer store.deinit();

    const cap = 3;
    store.setCap(cap);

    // Resident chunks first, as a real store writes them.
    for (0..n_resident) |i| try store.writeRange(i, blob[@intCast(each * i)..][0..@intCast(each)]);

    // Then every expert, as fetch-on-demand would. The cap has to hold across
    // all of them, not just at the end.
    for (n_resident..n) |i| {
        try store.writeRange(i, blob[@intCast(each * @as(u64, i))..][0..@intCast(each)]);
        try std.testing.expect(store.heldExperts() <= cap);
    }

    try std.testing.expectEqual(@as(usize, cap), store.heldExperts());
    // Mandatory chunks are never victims: evicting one makes the node unable to
    // run a forward pass at all.
    for (0..n_resident) |i| try std.testing.expect(store.holdings.has(i));

    // Coldest-first, where cold means least *recently* used: a shard that keeps
    // being routed to must survive an arbitrary number of arrivals. Touching it
    // once up front would not show this -- the arrivals are all more recent
    // than that touch, so it would be the correct victim.
    var keep: usize = 0;
    for (n_resident..n) |i| {
        if (store.holdings.has(i)) {
            keep = i;
            break;
        }
    }
    for (n_resident..n) |i| {
        if (i == keep) continue;
        // Before the arrival, not after: eviction happens inside writeRange, so
        // a touch afterwards is too late to save it -- which is also true of a
        // real router, where the shard is read to compute the token that then
        // triggers the next fetch.
        store.touch(keep);
        try store.writeRange(i, blob[@intCast(each * @as(u64, i))..][0..@intCast(each)]);
        try std.testing.expect(store.heldExperts() <= cap);
    }
    try std.testing.expect(store.holdings.has(keep));

    // The cap has to be visible on the disk, not only in the bitmap. Evicted
    // ranges are hole-punched, so the file's *allocated* blocks stay near what
    // is held rather than near what has ever passed through. The first version
    // punched unaligned ranges, which APFS rejects outright: holdings stayed
    // flat at 28.2% on a real node while the store grew 1.8 -> 7.6 GB.
    if (allocatedBytes(store.file.handle)) |allocated| {
        const held_bytes = each * (n_resident + cap);
        // The whole corpus is 42 ranges and the cap is 3, so an unpunched file
        // is ~8x this bound while a punched one is the held bytes plus a
        // partial block at each end of every hole. Checked by reverting the
        // alignment: without it the punch is refused and this fails.
        try std.testing.expect(allocated < held_bytes * 3);
    }
}
