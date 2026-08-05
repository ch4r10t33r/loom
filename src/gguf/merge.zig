//! Merge a `gguf-split` shard set into one canonical GGUF (issue #32).
//!
//! Every large published quant of a frontier model ships split
//! (`…-00001-of-00003.gguf`): DeepSeek-V4-Flash, GLM-5.2, and most 100 GB+
//! releases. Loom's whole distribution model — the expert-aligned manifest,
//! byte-range shards, digests, the HTTP mirror, peer fetch — is defined over
//! **one logical file**, so the cheapest correct way to consume a split model
//! is to make it one file, exactly as `llama-gguf-split --merge` does. Doing
//! it here removes llama.cpp from the operator's path (the devnet's GLM-Air
//! was merged with that tool by hand).
//!
//! Metadata is copied **verbatim** from the first shard, minus the `split.*`
//! keys: `gguf.readValue` deliberately skips array element types the engine
//! has no use for, so re-serializing the parsed form would silently drop
//! them. Tensor infos are re-emitted with offsets recomputed against the
//! merged data section, which is the only thing that actually changes.
//!
//! Not implemented here: reading a split set in place, without merging. That
//! would push a (file, offset) pair through the manifest, the p2p wire and
//! the store, and it buys only disk headroom on the machine that happens to
//! hold the origin copy.

const std = @import("std");
const Io = std.Io;
const gguf = @import("gguf.zig");
const ggml = @import("ggml.zig");

/// Bytes of tensor data copied per read/write chunk.
const CHUNK = 8 << 20;

pub fn run(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, first: []const u8, dest: []const u8) !void {
    var head = gguf.parse(gpa, io, first) catch |e| {
        return out.print("not a readable GGUF: {t}\n", .{e});
    };
    defer head.deinit();

    const split = gguf.splitInfo(&head) orelse {
        return out.print("{s} is not a split shard (no split.count); nothing to merge\n", .{first});
    };
    if (split.no != 0) {
        return out.print("pass the FIRST shard (this one is {d} of {d})\n", .{ split.no + 1, split.count });
    }

    try out.print("merging {d} shard(s) -> {s}\n", .{ split.count, dest });
    try out.flush();

    // ---- pass one: parse every shard, tally tensors and data bytes ----
    var shards = try gpa.alloc(gguf.Parsed, split.count);
    var opened: usize = 0;
    defer {
        for (shards[0..opened]) |*p| p.deinit();
        gpa.free(shards);
    }
    var paths = try gpa.alloc([]u8, split.count);
    defer {
        for (paths[0..opened]) |p| gpa.free(p);
        gpa.free(paths);
    }

    var total_tensors: u64 = 0;
    var pbuf: [4096]u8 = undefined;
    for (0..split.count) |i| {
        const p = try gguf.shardPath(&pbuf, first, @intCast(i), split.count);
        paths[i] = try gpa.dupe(u8, p);
        shards[i] = gguf.parse(gpa, io, p) catch |e| {
            try out.print("cannot read shard {d}: {s} ({t})\n", .{ i + 1, p, e });
            return e;
        };
        opened = i + 1;
        total_tensors += shards[i].tensors.len;
    }
    if (split.tensors_count != 0 and total_tensors != split.tensors_count) {
        try out.print(
            "shard set is incomplete: {d} tensors found, {d} declared\n",
            .{ total_tensors, split.tensors_count },
        );
        return error.IncompleteSplitSet;
    }

    // ---- build the merged header ----
    // Metadata: shard 0's KVs verbatim, minus split.* (which describe the
    // sharding, not the model).
    const head_bytes = try readAll(gpa, io, paths[0]);
    defer gpa.free(head_bytes);

    var kv_count: u64 = 0;
    for (head.metadata) |kv| {
        if (std.mem.startsWith(u8, kv.key, "split.")) continue;
        kv_count += 1;
    }

    var hdr = std.ArrayList(u8).empty;
    defer hdr.deinit(gpa);
    try appendU32(gpa, &hdr, 0x46554747); // "GGUF"
    try appendU32(gpa, &hdr, head.version);
    try appendU64(gpa, &hdr, total_tensors);
    try appendU64(gpa, &hdr, kv_count);
    for (head.metadata) |kv| {
        if (std.mem.startsWith(u8, kv.key, "split.")) continue;
        try hdr.appendSlice(gpa, head_bytes[@intCast(kv.span_start)..@intCast(kv.span_end)]);
    }

    // Tensor infos, in shard order, with offsets recomputed against one data
    // section. Each shard's data is placed at an aligned base.
    const align_to = head.alignment;
    var bases = try gpa.alloc(u64, split.count);
    defer gpa.free(bases);
    var running: u64 = 0;
    for (shards, 0..) |*p, i| {
        bases[i] = std.mem.alignForward(u64, running, align_to);
        const data_len = p.file_size - p.data_offset;
        running = bases[i] + data_len;
    }
    const merged_data_bytes = running;

    for (shards, 0..) |*p, i| {
        for (p.tensors) |t| {
            try appendStr(gpa, &hdr, t.name);
            try appendU32(gpa, &hdr, @intCast(t.dims.len));
            for (t.dims) |d| try appendU64(gpa, &hdr, d);
            try appendU32(gpa, &hdr, t.ggml_type);
            try appendU64(gpa, &hdr, bases[i] + t.offset);
        }
    }
    while (hdr.items.len % align_to != 0) try hdr.append(gpa, 0);

    // ---- write ----
    const f = try Io.Dir.cwd().createFile(io, dest, .{ .truncate = true });
    defer f.close(io);
    try f.writeStreamingAll(io, hdr.items);

    const buf = try gpa.alloc(u8, CHUNK);
    defer gpa.free(buf);
    var written: u64 = 0;
    for (shards, 0..) |*p, i| {
        // pad to this shard's aligned base
        const pad = bases[i] - written;
        if (pad > 0) {
            @memset(buf[0..@intCast(@min(pad, CHUNK))], 0);
            var left = pad;
            while (left > 0) {
                const n: usize = @intCast(@min(left, CHUNK));
                try f.writeStreamingAll(io, buf[0..n]);
                left -= n;
            }
            written = bases[i];
        }
        const src = try Io.Dir.cwd().openFile(io, paths[i], .{});
        defer src.close(io);
        var off = p.data_offset;
        const end = p.file_size;
        while (off < end) {
            const n: usize = @intCast(@min(end - off, CHUNK));
            _ = try src.readPositionalAll(io, buf[0..n], off);
            try f.writeStreamingAll(io, buf[0..n]);
            off += n;
            written += n;
        }
        try out.print("  shard {d}/{d} copied ({d:.1} GB total)\n", .{
            i + 1, split.count, @as(f64, @floatFromInt(written)) / (1 << 30),
        });
        try out.flush();
    }

    try out.print("merged {d} tensors, {d:.2} GB of tensor data -> {s}\n", .{
        total_tensors, @as(f64, @floatFromInt(merged_data_bytes)) / (1 << 30), dest,
    });
    try out.flush();
}

fn readAll(gpa: std.mem.Allocator, io: Io, path: []const u8) ![]u8 {
    const f = try Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const size = try f.length(io);
    // only the header region is ever needed; cap it the same way parse does
    const want: usize = @intCast(@min(size, 64 * 1024 * 1024));
    const buf = try gpa.alloc(u8, want);
    errdefer gpa.free(buf);
    _ = try f.readPositionalAll(io, buf, 0);
    return buf;
}

fn appendU32(gpa: std.mem.Allocator, b: *std.ArrayList(u8), v: u32) !void {
    var t: [4]u8 = undefined;
    std.mem.writeInt(u32, &t, v, .little);
    try b.appendSlice(gpa, &t);
}
fn appendU64(gpa: std.mem.Allocator, b: *std.ArrayList(u8), v: u64) !void {
    var t: [8]u8 = undefined;
    std.mem.writeInt(u64, &t, v, .little);
    try b.appendSlice(gpa, &t);
}
fn appendStr(gpa: std.mem.Allocator, b: *std.ArrayList(u8), s: []const u8) !void {
    try appendU64(gpa, b, s.len);
    try b.appendSlice(gpa, s);
}

/// Write `src` (a plain single-file GGUF) as a `count`-way split set under
/// `dir/<stem>-0000N-of-0000C.gguf`, mirroring what gguf-split produces:
/// shard 0 carries the metadata plus its share of tensors, and every shard
/// declares split.no / split.count / split.tensors.count.
fn writeSplitFixture(gpa: std.mem.Allocator, io: Io, src_path: []const u8, out_stem: []const u8, count: u16) !void {
    var p = try gguf.parse(gpa, io, src_path);
    defer p.deinit();
    const src = try readAll(gpa, io, src_path);
    defer gpa.free(src);
    const f = try Io.Dir.cwd().openFile(io, src_path, .{});
    defer f.close(io);

    const per = (p.tensors.len + count - 1) / count;
    for (0..count) |i| {
        const lo = @min(i * per, p.tensors.len);
        const hi = @min(lo + per, p.tensors.len);
        const mine = p.tensors[lo..hi];

        var hdr = std.ArrayList(u8).empty;
        defer hdr.deinit(gpa);
        try appendU32(gpa, &hdr, 0x46554747);
        try appendU32(gpa, &hdr, p.version);
        try appendU64(gpa, &hdr, mine.len);
        try appendU64(gpa, &hdr, p.metadata.len + 3); // + split.no/count/tensors.count
        for (p.metadata) |kv| try hdr.appendSlice(gpa, src[@intCast(kv.span_start)..@intCast(kv.span_end)]);
        try kvU16(gpa, &hdr, "split.no", @intCast(i));
        try kvU16(gpa, &hdr, "split.count", count);
        try kvU32kv(gpa, &hdr, "split.tensors.count", @intCast(p.tensors.len));

        // offsets restart at 0 in each shard, in tensor order
        var base: u64 = 0;
        for (mine) |t| {
            try appendStr(gpa, &hdr, t.name);
            try appendU32(gpa, &hdr, @intCast(t.dims.len));
            for (t.dims) |d| try appendU64(gpa, &hdr, d);
            try appendU32(gpa, &hdr, t.ggml_type);
            try appendU64(gpa, &hdr, base);
            base = std.mem.alignForward(u64, base + tensorBytes(t), p.alignment);
        }
        while (hdr.items.len % p.alignment != 0) try hdr.append(gpa, 0);

        var nbuf: [512]u8 = undefined;
        const name = try std.fmt.bufPrint(&nbuf, "{s}-{d:0>5}-of-{d:0>5}.gguf", .{ out_stem, i + 1, count });
        const of = try Io.Dir.cwd().createFile(io, name, .{ .truncate = true });
        defer of.close(io);
        try of.writeStreamingAll(io, hdr.items);

        var written: u64 = 0;
        for (mine) |t| {
            const want = std.mem.alignForward(u64, written, p.alignment) - written;
            if (want > 0) {
                const pad = try gpa.alloc(u8, @intCast(want));
                defer gpa.free(pad);
                @memset(pad, 0);
                try of.writeStreamingAll(io, pad);
                written += want;
            }
            const n: usize = @intCast(tensorBytes(t));
            const tb = try gpa.alloc(u8, n);
            defer gpa.free(tb);
            _ = try f.readPositionalAll(io, tb, p.data_offset + t.offset);
            try of.writeStreamingAll(io, tb);
            written += n;
        }
    }
}

fn tensorBytes(t: gguf.TensorInfo) u64 {
    const ty: ggml.Type = @enumFromInt(t.ggml_type);
    var rows: u64 = 1;
    for (t.dims[1..]) |d| rows *= d;
    return rows * ggml.rowBytes(ty, @intCast(t.dims[0]));
}

fn kvU16(gpa: std.mem.Allocator, b: *std.ArrayList(u8), key: []const u8, v: u16) !void {
    try appendStr(gpa, b, key);
    try appendU32(gpa, b, 3); // u16
    var t: [2]u8 = undefined;
    std.mem.writeInt(u16, &t, v, .little);
    try b.appendSlice(gpa, &t);
}
fn kvU32kv(gpa: std.mem.Allocator, b: *std.ArrayList(u8), key: []const u8, v: u32) !void {
    try appendStr(gpa, b, key);
    try appendU32(gpa, b, 4); // u32
    try appendU32(gpa, b, v);
}

test "a split set merges back into a byte-faithful single GGUF (issue #32)" {
    // The property that matters: every tensor's *bytes* survive the split and
    // merge unchanged, and the merged file is a normal GGUF with no split.*
    // keys -- because everything downstream (manifest, digests, mirror, peer
    // fetch) is defined over one logical file.
    const gpa = std.testing.allocator;
    var thr: std.Io.Threaded = .init(gpa, .{});
    defer thr.deinit();
    const io = thr.io();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var b1: [std.fs.max_path_bytes]u8 = undefined;
    const orig = try std.fmt.bufPrint(&b1, ".zig-cache/tmp/{s}/orig.gguf", .{tmp.sub_path});
    try gguf.writeMoeFixture(gpa, io, orig, 5, "llama");

    var b2: [std.fs.max_path_bytes]u8 = undefined;
    const stem = try std.fmt.bufPrint(&b2, ".zig-cache/tmp/{s}/part", .{tmp.sub_path});
    try writeSplitFixture(gpa, io, orig, stem, 3);

    var b3: [std.fs.max_path_bytes]u8 = undefined;
    const first = try std.fmt.bufPrint(&b3, "{s}-00001-of-00003.gguf", .{stem});
    var b4: [std.fs.max_path_bytes]u8 = undefined;
    const merged = try std.fmt.bufPrint(&b4, ".zig-cache/tmp/{s}/merged.gguf", .{tmp.sub_path});

    var sink = std.Io.Writer.Allocating.init(gpa);
    defer sink.deinit();
    try run(gpa, io, &sink.writer, first, merged);

    var a = try gguf.parse(gpa, io, orig);
    defer a.deinit();
    var m = try gguf.parse(gpa, io, merged);
    defer m.deinit();

    try std.testing.expectEqual(a.tensors.len, m.tensors.len);
    try std.testing.expect(gguf.splitInfo(&m) == null); // split.* dropped
    try std.testing.expectEqualStrings(
        a.getString("general.architecture").?,
        m.getString("general.architecture").?,
    );

    const fa = try Io.Dir.cwd().openFile(io, orig, .{});
    defer fa.close(io);
    const fm = try Io.Dir.cwd().openFile(io, merged, .{});
    defer fm.close(io);
    for (a.tensors, m.tensors) |ta, tm| {
        try std.testing.expectEqualStrings(ta.name, tm.name);
        try std.testing.expectEqual(ta.ggml_type, tm.ggml_type);
        const n: usize = @intCast(tensorBytes(ta));
        const ba = try gpa.alloc(u8, n);
        defer gpa.free(ba);
        const bm = try gpa.alloc(u8, n);
        defer gpa.free(bm);
        _ = try fa.readPositionalAll(io, ba, a.data_offset + ta.offset);
        _ = try fm.readPositionalAll(io, bm, m.data_offset + tm.offset);
        try std.testing.expectEqualSlices(u8, ba, bm);
    }
}

test "split shard naming round-trips" {
    var buf: [256]u8 = undefined;
    const p = "models/DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf";
    try std.testing.expectEqualStrings(
        "models/DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf",
        try gguf.shardPath(&buf, p, 0, 3),
    );
    try std.testing.expectEqualStrings(
        "models/DeepSeek-V4-Flash-0731-UD-IQ1_S-00003-of-00003.gguf",
        try gguf.shardPath(&buf, p, 2, 3),
    );
    try std.testing.expectError(error.NotSplitName, gguf.shardPath(&buf, "plain.gguf", 0, 3));
}
