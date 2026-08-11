//! Pre-gate predictor: a small trained head that, given the residual stream
//! after layer 0, predicts every later MoE layer's expert selection -- so a
//! partial holder can start fetching a token's whole expert working set with
//! per-layer lead time instead of discovering it layer by layer.
//!
//! This is PILOT's idea at full depth. PILOT (router-lookahead) predicts ONE
//! layer ahead, and the lever-9 replay showed one layer-compute of lead
//! cannot cover a multi-ms fetch; the trained head predicts all layers from
//! layer-0 state (measured on Qwen3-30B-A3B: 50.9% of every layer's top-8
//! against a 22.7% hot-set baseline, deepest layers best). Predictions feed
//! `Source.prefetchAsync`, which is residency-checked and lossy by design;
//! a mispredicted fetch still warms the store.
//!
//! File format (LPG1, little-endian): "LPG1", u32 hid, u32 width,
//! u32 n_pred (layers predicted, i.e. n_layers-1), u32 n_expert, then f32
//! arrays w1[width*hid] b1[width] w2[n_pred*n_expert*width]
//! b2[n_pred*n_expert] -- PyTorch Linear layout (row-major [out, in]),
//! written by scripts/pregate-export.py from the probe's checkpoint.
//!
//! Scratch note: the engine serves one request at a time (rpc.zig's
//! documented contract), so the inference scratch lives here rather than in
//! every State. Concurrent step() calls would race it.

const std = @import("std");
const Io = std.Io;
const backend = @import("../compute/backend.zig");

pub const Pregate = struct {
    hid: usize,
    width: usize,
    n_pred: usize,
    n_expert: usize,
    w1: []f32,
    b1: []f32,
    w2: []f32,
    b2: []f32,
    hbuf: []f32, // width
    logits: []f32, // n_pred * n_expert

    pub fn deinit(self: *Pregate, gpa: std.mem.Allocator) void {
        gpa.free(self.w1);
        gpa.free(self.b1);
        gpa.free(self.w2);
        gpa.free(self.b2);
        gpa.free(self.hbuf);
        gpa.free(self.logits);
    }
};

pub fn load(gpa: std.mem.Allocator, io: Io, path: []const u8) !Pregate {
    const f = try Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    var hdr: [20]u8 = undefined;
    if (try f.readPositionalAll(io, &hdr, 0) != hdr.len) return error.Truncated;
    if (!std.mem.eql(u8, hdr[0..4], "LPG1")) return error.BadMagic;
    const hid: usize = std.mem.readInt(u32, hdr[4..8], .little);
    const width: usize = std.mem.readInt(u32, hdr[8..12], .little);
    const n_pred: usize = std.mem.readInt(u32, hdr[12..16], .little);
    const n_expert: usize = std.mem.readInt(u32, hdr[16..20], .little);
    // Arbitrary sane caps: a corrupted header must not drive allocation.
    if (hid == 0 or hid > 32768 or width == 0 or width > 32768 or
        n_pred == 0 or n_pred > 512 or n_expert == 0 or n_expert > 1024)
        return error.BadHeader;

    var pg = Pregate{
        .hid = hid,
        .width = width,
        .n_pred = n_pred,
        .n_expert = n_expert,
        .w1 = undefined,
        .b1 = undefined,
        .w2 = undefined,
        .b2 = undefined,
        .hbuf = undefined,
        .logits = undefined,
    };
    var off: u64 = hdr.len;
    pg.w1 = try readF32s(gpa, io, f, &off, width * hid);
    errdefer gpa.free(pg.w1);
    pg.b1 = try readF32s(gpa, io, f, &off, width);
    errdefer gpa.free(pg.b1);
    pg.w2 = try readF32s(gpa, io, f, &off, n_pred * n_expert * width);
    errdefer gpa.free(pg.w2);
    pg.b2 = try readF32s(gpa, io, f, &off, n_pred * n_expert);
    errdefer gpa.free(pg.b2);
    pg.hbuf = try gpa.alloc(f32, width);
    errdefer gpa.free(pg.hbuf);
    pg.logits = try gpa.alloc(f32, n_pred * n_expert);
    return pg;
}

fn readF32s(gpa: std.mem.Allocator, io: Io, f: Io.File, off: *u64, n: usize) ![]f32 {
    const out = try gpa.alloc(f32, n);
    errdefer gpa.free(out);
    const bytes = std.mem.sliceAsBytes(out);
    if (try f.readPositionalAll(io, bytes, off.*) != bytes.len) return error.Truncated;
    off.* += bytes.len;
    return out;
}

/// tanh-approximated GELU. The head was trained with PyTorch's erf GELU;
/// the approximation differs by <1e-3, far below what reorders a top-k.
fn gelu(x: f32) f32 {
    const c: f32 = 0.7978845608; // sqrt(2/pi)
    return 0.5 * x * (1.0 + std.math.tanh(c * (x + 0.044715 * x * x * x)));
}

/// Head forward: fills `self.logits` with n_pred x n_expert scores. Takes
/// const self -- the scratch writes go through the slices, whose pointees
/// stay mutable -- so a `*const Model` can hold and use the head.
pub fn predict(self: *const Pregate, h: []const f32) void {
    backend.matvec(.f32, self.hbuf, std.mem.sliceAsBytes(self.w1), h, self.width, self.hid);
    for (self.hbuf, self.b1) |*v, b| v.* = gelu(v.* + b);
    backend.matvec(.f32, self.logits, std.mem.sliceAsBytes(self.w2), self.hbuf, self.n_pred * self.n_expert, self.width);
    for (self.logits, self.b2) |*v, b| v.* += b;
}

/// Top-k expert indices for predicted layer `pi` (0 = model layer 1), from
/// the logits `predict` filled. First-max-wins on ties, the router's order.
pub fn topk(self: *const Pregate, pi: usize, k: usize, out: []u16) usize {
    const lg = self.logits[pi * self.n_expert ..][0..self.n_expert];
    const n = @min(k, self.n_expert);
    var taken = std.StaticBitSet(1024).initEmpty();
    for (0..n) |j| {
        var best: usize = 0;
        var best_v = -std.math.inf(f32);
        for (lg, 0..) |v, e| {
            if (!taken.isSet(e) and v > best_v) {
                best_v = v;
                best = e;
            }
        }
        taken.set(best);
        out[j] = @intCast(best);
    }
    return n;
}

test "pregate load, predict and topk agree with a hand-computed reference" {
    const gpa = std.testing.allocator;
    var thr: std.Io.Threaded = .init(gpa, .{});
    defer thr.deinit();
    const io = thr.io();

    // Tiny head: hid=2, width=2, n_pred=2, n_expert=3.
    // w1 = identity, b1 = 0 -> hbuf = gelu(h).
    const f32s = [_]f32{
        1, 0, 0, 1, // w1
        0, 0, // b1
        1, 0, 0, 1, 1, 1, // w2 rows pi0: e0=[1,0] e1=[0,1] e2=[1,1]
        2, 0, 0, 2, -1, -1, // w2 rows pi1
        0, 0.5, 0, 0, 0, 0, // b2 (pi0 e1 boosted)
    };
    var bytes: [20 + f32s.len * 4]u8 = undefined;
    @memcpy(bytes[0..4], "LPG1");
    for ([_]u32{ 2, 2, 2, 3 }, 0..) |v, i|
        std.mem.writeInt(u32, bytes[4 + i * 4 ..][0..4], v, .little);
    for (f32s, 0..) |v, i|
        std.mem.writeInt(u32, bytes[20 + i * 4 ..][0..4], @bitCast(v), .little);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&pb, ".zig-cache/tmp/{s}/head.lpg", .{tmp.sub_path});
    {
        const f = try Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer f.close(io);
        try f.writeStreamingAll(io, &bytes);
    }

    var pg = try load(gpa, io, path);
    defer pg.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), pg.hid);
    try std.testing.expectEqual(@as(usize, 3), pg.n_expert);

    const h = [_]f32{ 2.0, -1.0 };
    predict(&pg, &h);
    const g2 = gelu(2.0); // ~1.954
    const gm1 = gelu(-1.0); // ~-0.159
    // pi0: e0=g2, e1=gm1+0.5, e2=g2+gm1 ; pi1: e0=2*g2, e1=2*gm1, e2=-(g2+gm1)
    try std.testing.expectApproxEqAbs(g2, pg.logits[0], 1e-5);
    try std.testing.expectApproxEqAbs(gm1 + 0.5, pg.logits[1], 1e-5);
    try std.testing.expectApproxEqAbs(g2 + gm1, pg.logits[2], 1e-5);
    try std.testing.expectApproxEqAbs(2 * g2, pg.logits[3], 1e-5);

    var out: [2]u16 = undefined;
    // pi0 ranking: e0=1.954 > e2=1.795 > e1=0.341 -> top2 = e0, e2
    try std.testing.expectEqual(@as(usize, 2), topk(&pg, 0, 2, &out));
    try std.testing.expectEqual(@as(u16, 0), out[0]);
    try std.testing.expectEqual(@as(u16, 2), out[1]);
    // pi1: e0=3.91 > e1=-0.32 > e2=-1.79 -> top2 = e0, e1
    _ = topk(&pg, 1, 2, &out);
    try std.testing.expectEqual(@as(u16, 0), out[0]);
    try std.testing.expectEqual(@as(u16, 1), out[1]);
}
