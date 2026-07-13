//! GGUF (v3) file parsing — the interchange weight format for v1 distribution
//! (ROADMAP.md #2). This module reads the header, metadata KVs, and tensor
//! table, and locates the aligned data section. Distribution operates on byte
//! ranges of the whole file (see weights.zig), so per-tensor dequantization is
//! not needed here; wiring GGUF tensors into the inference engine is a later,
//! separate step.
//!
//! Layout (little-endian):
//!   u32 magic "GGUF" | u32 version | u64 tensor_count | u64 metadata_kv_count
//!   metadata KVs: { string key; u32 type; value }
//!   tensor infos: { string name; u32 n_dims; u64 dims[n]; u32 ggml_type; u64 offset }
//!   padding to `general.alignment` (default 32), then tensor data.

const std = @import("std");
const Io = std.Io;

pub const MAGIC: u32 = 0x46554747; // "GGUF" LE

pub const ValueType = enum(u32) {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    u32 = 4,
    i32 = 5,
    f32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    u64 = 10,
    i64 = 11,
    f64 = 12,
    _,
};

pub const TensorInfo = struct {
    name: []const u8, // owned by Parsed.arena
    dims: []u64, // owned by Parsed.arena
    ggml_type: u32,
    offset: u64, // relative to data_offset

    pub fn elemCount(self: TensorInfo) u64 {
        var n: u64 = 1;
        for (self.dims) |d| n *= d;
        return n;
    }
};

pub const MetaValue = union(enum) {
    int: i64,
    uint: u64,
    float: f64,
    boolean: bool,
    string: []const u8, // owned by Parsed.arena
    // typed arrays retained for tokenizer metadata; all owned by Parsed.arena
    array_str: []const []const u8,
    array_f32: []const f32,
    array_i32: []const i32,
    array: struct { elem_type: u32, count: u64 }, // other element types: skipped
};

pub const MetaKV = struct {
    key: []const u8, // owned by Parsed.arena
    value: MetaValue,
};

pub const Parsed = struct {
    arena: std.heap.ArenaAllocator,
    version: u32,
    alignment: u64,
    metadata: []MetaKV,
    tensors: []TensorInfo,
    data_offset: u64, // absolute file offset of the tensor data section
    file_size: u64,

    pub fn deinit(self: *Parsed) void {
        self.arena.deinit();
    }

    pub fn findMeta(self: *const Parsed, key: []const u8) ?MetaValue {
        for (self.metadata) |kv| {
            if (std.mem.eql(u8, kv.key, key)) return kv.value;
        }
        return null;
    }

    pub fn getUint(self: *const Parsed, key: []const u8) ?u64 {
        return switch (self.findMeta(key) orelse return null) {
            .uint => |v| v,
            .int => |v| if (v >= 0) @intCast(v) else null,
            else => null,
        };
    }

    pub fn getFloat(self: *const Parsed, key: []const u8) ?f64 {
        return switch (self.findMeta(key) orelse return null) {
            .float => |v| v,
            .uint => |v| @floatFromInt(v),
            .int => |v| @floatFromInt(v),
            else => null,
        };
    }

    pub fn getString(self: *const Parsed, key: []const u8) ?[]const u8 {
        return switch (self.findMeta(key) orelse return null) {
            .string => |s| s,
            else => null,
        };
    }

    pub fn findTensor(self: *const Parsed, name: []const u8) ?TensorInfo {
        for (self.tensors) |t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }
};

const Cursor = struct {
    buf: []const u8,
    pos: usize = 0,

    fn need(self: *Cursor, n: usize) ![]const u8 {
        if (self.pos + n > self.buf.len) return error.TruncatedGguf;
        const s = self.buf[self.pos..][0..n];
        self.pos += n;
        return s;
    }
    fn u32le(self: *Cursor) !u32 {
        return std.mem.readInt(u32, (try self.need(4))[0..4], .little);
    }
    fn u64le(self: *Cursor) !u64 {
        return std.mem.readInt(u64, (try self.need(8))[0..8], .little);
    }
    fn str(self: *Cursor) ![]const u8 {
        const len = try self.u64le();
        if (len > 1 << 24) return error.TruncatedGguf; // sanity: 16 MB string cap
        return self.need(@intCast(len));
    }
};

fn readValue(c: *Cursor, arena: std.mem.Allocator, vtype: u32) !MetaValue {
    return switch (@as(ValueType, @enumFromInt(vtype))) {
        .u8 => .{ .uint = (try c.need(1))[0] },
        .i8 => .{ .int = @as(i8, @bitCast((try c.need(1))[0])) },
        .u16 => .{ .uint = std.mem.readInt(u16, (try c.need(2))[0..2], .little) },
        .i16 => .{ .int = std.mem.readInt(i16, (try c.need(2))[0..2], .little) },
        .u32 => .{ .uint = try c.u32le() },
        .i32 => .{ .int = @as(i32, @bitCast(try c.u32le())) },
        .f32 => .{ .float = @as(f32, @bitCast(try c.u32le())) },
        .bool => .{ .boolean = (try c.need(1))[0] != 0 },
        .string => .{ .string = try arena.dupe(u8, try c.str()) },
        .u64 => .{ .uint = try c.u64le() },
        .i64 => .{ .int = @as(i64, @bitCast(try c.u64le())) },
        .f64 => .{ .float = @as(f64, @bitCast(try c.u64le())) },
        .array => blk: {
            const elem_type = try c.u32le();
            const count = try c.u64le();
            if (count > 1 << 24) return error.TruncatedGguf;
            switch (@as(ValueType, @enumFromInt(elem_type))) {
                .string => {
                    const items = try arena.alloc([]const u8, @intCast(count));
                    for (items) |*s| s.* = try arena.dupe(u8, try c.str());
                    break :blk .{ .array_str = items };
                },
                .f32 => {
                    const items = try arena.alloc(f32, @intCast(count));
                    for (items) |*v| v.* = @bitCast(try c.u32le());
                    break :blk .{ .array_f32 = items };
                },
                .i32 => {
                    const items = try arena.alloc(i32, @intCast(count));
                    for (items) |*v| v.* = @bitCast(try c.u32le());
                    break :blk .{ .array_i32 = items };
                },
                else => {
                    // skip contents (recursively handles arrays of any type)
                    var i: u64 = 0;
                    while (i < count) : (i += 1) _ = try readValue(c, arena, elem_type);
                    break :blk .{ .array = .{ .elem_type = elem_type, .count = count } };
                },
            }
        },
        _ => error.UnknownValueType,
    };
}

/// Parse the GGUF header/metadata/tensor table from the start of a file. Only
/// the header region is read into memory, never the tensor data.
pub fn parse(gpa: std.mem.Allocator, io: Io, path: []const u8) !Parsed {
    const f = try Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const file_size = (try f.stat(io)).size;

    // read up to 64 MB of header; real headers are far smaller
    const head_cap: usize = @intCast(@min(file_size, 64 * 1024 * 1024));
    const head = try gpa.alloc(u8, head_cap);
    defer gpa.free(head);
    _ = try f.readPositionalAll(io, head, 0);

    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    var c = Cursor{ .buf = head };
    if (try c.u32le() != MAGIC) return error.NotGguf;
    const version = try c.u32le();
    if (version < 2 or version > 3) return error.UnsupportedGgufVersion;
    const tensor_count = try c.u64le();
    const kv_count = try c.u64le();
    if (tensor_count > 1 << 20 or kv_count > 1 << 20) return error.TruncatedGguf;

    const metadata = try a.alloc(MetaKV, @intCast(kv_count));
    for (metadata) |*kv| {
        kv.key = try a.dupe(u8, try c.str());
        const vtype = try c.u32le();
        kv.value = try readValue(&c, a, vtype);
    }

    const tensors = try a.alloc(TensorInfo, @intCast(tensor_count));
    for (tensors) |*t| {
        t.name = try a.dupe(u8, try c.str());
        const n_dims = try c.u32le();
        if (n_dims > 8) return error.TruncatedGguf;
        t.dims = try a.alloc(u64, n_dims);
        for (t.dims) |*d| d.* = try c.u64le();
        t.ggml_type = try c.u32le();
        t.offset = try c.u64le();
    }

    var alignment: u64 = 32;
    for (metadata) |kv| {
        if (std.mem.eql(u8, kv.key, "general.alignment")) {
            switch (kv.value) {
                .uint => |v| alignment = v,
                .int => |v| if (v > 0) {
                    alignment = @intCast(v);
                },
                else => {},
            }
        }
    }
    if (alignment == 0 or (alignment & (alignment - 1)) != 0) return error.BadAlignment;

    const data_offset = std.mem.alignForward(u64, c.pos, alignment);
    if (data_offset > file_size) return error.TruncatedGguf;

    return .{
        .arena = arena,
        .version = version,
        .alignment = alignment,
        .metadata = metadata,
        .tensors = tensors,
        .data_offset = data_offset,
        .file_size = file_size,
    };
}

// ---- synthetic fixture writer ------------------------------------------------

/// Write a small, valid GGUF v3 file with deterministic f32 tensor data.
/// Used for tests and local multi-node demos (`loom gguf gen`).
pub fn writeFixture(gpa: std.mem.Allocator, io: Io, path: []const u8, seed: u64, data_mb: usize) !void {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(gpa);

    const alignment: u64 = 32;
    // two tensors filling ~data_mb MB total
    const total_elems: u64 = @intCast(data_mb * 1024 * 1024 / 4);
    const t0_elems = total_elems / 2;
    const t1_elems = total_elems - t0_elems;

    try appendU32(gpa, &buf, MAGIC);
    try appendU32(gpa, &buf, 3); // version
    try appendU64(gpa, &buf, 2); // tensor_count
    try appendU64(gpa, &buf, 3); // kv_count

    try appendStr(gpa, &buf, "general.architecture");
    try appendU32(gpa, &buf, @intFromEnum(ValueType.string));
    try appendStr(gpa, &buf, "loom-synthetic");

    try appendStr(gpa, &buf, "general.name");
    try appendU32(gpa, &buf, @intFromEnum(ValueType.string));
    try appendStr(gpa, &buf, "loom test fixture");

    try appendStr(gpa, &buf, "general.alignment");
    try appendU32(gpa, &buf, @intFromEnum(ValueType.u32));
    try appendU32(gpa, &buf, @intCast(alignment));

    // tensor infos (ggml type 0 = F32)
    try appendStr(gpa, &buf, "tok_embd.weight");
    try appendU32(gpa, &buf, 1);
    try appendU64(gpa, &buf, t0_elems);
    try appendU32(gpa, &buf, 0);
    try appendU64(gpa, &buf, 0);

    try appendStr(gpa, &buf, "blk.0.ffn.weight");
    try appendU32(gpa, &buf, 1);
    try appendU64(gpa, &buf, t1_elems);
    try appendU32(gpa, &buf, 0);
    try appendU64(gpa, &buf, t0_elems * 4);

    // pad to alignment
    while (buf.items.len % alignment != 0) try buf.append(gpa, 0);

    const f = try Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer f.close(io);
    try f.writeStreamingAll(io, buf.items);

    // stream tensor data in chunks (don't hold data_mb in RAM)
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var chunk: [64 * 1024]u8 = undefined;
    var remaining: u64 = total_elems * 4;
    while (remaining > 0) {
        const n: usize = @intCast(@min(remaining, chunk.len));
        var i: usize = 0;
        while (i + 4 <= n) : (i += 4) {
            const v: f32 = (rnd.float(f32) - 0.5) * 0.1;
            std.mem.writeInt(u32, chunk[i..][0..4], @bitCast(v), .little);
        }
        try f.writeStreamingAll(io, chunk[0..n]);
        remaining -= n;
    }
}

fn appendU32(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try buf.appendSlice(gpa, &b);
}
fn appendU64(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), v: u64) !void {
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, v, .little);
    try buf.appendSlice(gpa, &b);
}
fn appendStr(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), s: []const u8) !void {
    try appendU64(gpa, buf, s.len);
    try buf.appendSlice(gpa, s);
}
