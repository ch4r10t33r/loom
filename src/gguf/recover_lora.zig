//! Recover-LoRA: small per-expert low-rank adapters trained to recover the
//! quality an aggressive weight quantization destroyed (recipe generalized
//! from Edge0, github.com/Edge0-AI/Edge0; philosophically a sibling of the
//! pre-gate head -- a bolt-on artifact beside a frozen model, no base
//! retraining, shipped as one file).
//!
//! Why it fits loom: the devnet serves Q2_K expert bodies because fetch
//! bytes are the currency, and Q2_K's quality damage is the price. The
//! adapters are tiny (rank r: 3*r*(dim+ffn) f16 elements per expert, ~135 KB
//! at r=8 on Qwen3-30B -- ~830 MB for all 6,144 experts) so they stay fully
//! RESIDENT on every node while the 2-bit expert bodies keep streaming:
//! quality moves toward a higher-bit quant at Q2_K's fetch bill. Adapters
//! never touch the fetch path or the store; they are dense local state.
//!
//! Math: W'x = Wx + B(Ax), PyTorch LoRA convention (A: [r, in], B: [out, r],
//! row-major), with the alpha/r scale baked into B at export time so the
//! engine does two plain matvecs and an add per projection.
//!
//! File format (LRA1, little-endian): "LRA1", u32 n_layers, u32 n_expert,
//! u32 rank, u32 dim, u32 ffn, then f16 data for every (layer, expert) pair
//! in layer-major order, each pair laid out as
//!   Ag[r*dim] Bg[ffn*r] Au[r*dim] Bu[ffn*r] Ad[r*ffn] Bd[dim*r]
//! (gate, up: dim -> ffn; down: ffn -> dim). Written by
//! scripts/recover-lora-train.py --export.
//!
//! Scratch note: the engine serves one request at a time (rpc.zig's
//! documented contract), so the inference scratch lives here rather than in
//! every State -- the same reasoning as pregate.zig.

const std = @import("std");
const Io = std.Io;

pub const MAX_RANK = 64;

pub const Rlora = struct {
    n_layers: usize,
    n_expert: usize,
    rank: usize,
    dim: usize,
    ffn: usize,
    /// the whole f16 payload, pair-addressed by arithmetic (no per-entry table)
    data: []align(2) u8,
    tmp_r: []f32, // rank
    tmp_out: []f32, // max(dim, ffn)

    pub fn deinit(self: *Rlora, gpa: std.mem.Allocator) void {
        gpa.free(self.data);
        gpa.free(self.tmp_r);
        gpa.free(self.tmp_out);
    }

    fn pairElems(rank: usize, dim: usize, ffn: usize) usize {
        return 3 * rank * (dim + ffn);
    }

    /// The six f16 sub-slices (raw bytes, ready for a Tensor) of one
    /// (layer, expert) adapter pair.
    pub const Pair = struct {
        ag: []const u8,
        bg: []const u8,
        au: []const u8,
        bu: []const u8,
        ad: []const u8,
        bd: []const u8,
    };

    pub fn pair(self: *const Rlora, layer: usize, expert: usize) Pair {
        const r = self.rank;
        const pe = pairElems(r, self.dim, self.ffn);
        var off = (layer * self.n_expert + expert) * pe * 2;
        const a_gu = r * self.dim * 2; // Ag/Au bytes
        const b_gu = self.ffn * r * 2; // Bg/Bu bytes
        const a_d = r * self.ffn * 2;
        const b_d = self.dim * r * 2;
        const d = self.data;
        const ag = d[off..][0..a_gu];
        off += a_gu;
        const bg = d[off..][0..b_gu];
        off += b_gu;
        const au = d[off..][0..a_gu];
        off += a_gu;
        const bu = d[off..][0..b_gu];
        off += b_gu;
        const ad = d[off..][0..a_d];
        off += a_d;
        const bd = d[off..][0..b_d];
        return .{ .ag = ag, .bg = bg, .au = au, .bu = bu, .ad = ad, .bd = bd };
    }
};

pub fn load(gpa: std.mem.Allocator, io: Io, path: []const u8) !Rlora {
    const f = try Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    var hdr: [24]u8 = undefined;
    if (try f.readPositionalAll(io, &hdr, 0) != hdr.len) return error.Truncated;
    if (!std.mem.eql(u8, hdr[0..4], "LRA1")) return error.BadMagic;
    const n_layers: usize = std.mem.readInt(u32, hdr[4..8], .little);
    const n_expert: usize = std.mem.readInt(u32, hdr[8..12], .little);
    const rank: usize = std.mem.readInt(u32, hdr[12..16], .little);
    const dim: usize = std.mem.readInt(u32, hdr[16..20], .little);
    const ffn: usize = std.mem.readInt(u32, hdr[20..24], .little);
    // A corrupted header must not drive allocation (the pregate rule).
    if (n_layers == 0 or n_layers > 512 or n_expert == 0 or n_expert > 1024 or
        rank == 0 or rank > MAX_RANK or dim == 0 or dim > 32768 or ffn == 0 or ffn > 65536)
        return error.BadHeader;

    const payload = n_layers * n_expert * Rlora.pairElems(rank, dim, ffn) * 2;
    const fsize = try f.length(io);
    // exact-size check: a truncated or padded file is a wrong file
    if (fsize != hdr.len + payload) return error.BadSize;

    const data = try gpa.alignedAlloc(u8, .fromByteUnits(2), payload);
    errdefer gpa.free(data);
    if (try f.readPositionalAll(io, data, hdr.len) != payload) return error.Truncated;

    const tmp_r = try gpa.alloc(f32, rank);
    errdefer gpa.free(tmp_r);
    const tmp_out = try gpa.alloc(f32, @max(dim, ffn));
    errdefer gpa.free(tmp_out);

    return .{
        .n_layers = n_layers,
        .n_expert = n_expert,
        .rank = rank,
        .dim = dim,
        .ffn = ffn,
        .data = data,
        .tmp_r = tmp_r,
        .tmp_out = tmp_out,
    };
}

// ---- tests -----------------------------------------------------------------

const t = std.testing;

fn writeHeader(buf: *[24]u8, n_layers: u32, n_expert: u32, rank: u32, dim: u32, ffn: u32) void {
    @memcpy(buf[0..4], "LRA1");
    std.mem.writeInt(u32, buf[4..8], n_layers, .little);
    std.mem.writeInt(u32, buf[8..12], n_expert, .little);
    std.mem.writeInt(u32, buf[12..16], rank, .little);
    std.mem.writeInt(u32, buf[16..20], dim, .little);
    std.mem.writeInt(u32, buf[20..24], ffn, .little);
}

test "LRA1 roundtrip: pair slice arithmetic addresses the right values" {
    const gpa = t.allocator;
    var thr: std.Io.Threaded = .init(gpa, .{});
    defer thr.deinit();
    const io = thr.io();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&pbuf, ".zig-cache/tmp/{s}/a.lra", .{tmp.sub_path});

    // 2 layers x 3 experts, rank 2, dim 4, ffn 3
    const nl = 2;
    const ne = 3;
    const r = 2;
    const dim = 4;
    const ffn = 3;
    const pe = 3 * r * (dim + ffn); // 42 f16 elems per pair
    var hdr: [24]u8 = undefined;
    writeHeader(&hdr, nl, ne, r, dim, ffn);
    // payload: element i (globally) holds f16(i % 251) -- distinct, exact in f16
    var payload: [nl * ne * pe * 2]u8 = undefined;
    for (0..nl * ne * pe) |i| {
        const v: f16 = @floatFromInt(i % 251);
        std.mem.writeInt(u16, payload[i * 2 ..][0..2], @bitCast(v), .little);
    }
    {
        const f = try Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer f.close(io);
        try f.writeStreamingAll(io, &hdr);
        try f.writeStreamingAll(io, &payload);
    }

    var ra = try load(gpa, io, path);
    defer ra.deinit(gpa);
    try t.expectEqual(@as(usize, r), ra.rank);

    // pair (1, 2) is the last: starts at element (1*3+2)*42 = 210. Its ag
    // holds elements 210..218; its bd (last b_d = dim*r = 8 elems) ends the file.
    const p = ra.pair(1, 2);
    const first_ag: f16 = @bitCast(std.mem.readInt(u16, p.ag[0..2], .little));
    try t.expectEqual(@as(f16, @floatFromInt(210 % 251)), first_ag);
    const last_bd: f16 = @bitCast(std.mem.readInt(u16, p.bd[p.bd.len - 2 ..][0..2], .little));
    try t.expectEqual(@as(f16, @floatFromInt((nl * ne * pe - 1) % 251)), last_bd);
    // slice lengths match the layout
    try t.expectEqual(@as(usize, r * dim * 2), p.ag.len);
    try t.expectEqual(@as(usize, ffn * r * 2), p.bg.len);
    try t.expectEqual(@as(usize, r * ffn * 2), p.ad.len);
    try t.expectEqual(@as(usize, dim * r * 2), p.bd.len);

    // wrong size is a wrong file
    {
        const f = try Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer f.close(io);
        try f.writeStreamingAll(io, &hdr);
        try f.writeStreamingAll(io, payload[0 .. payload.len - 2]);
    }
    try t.expectError(error.BadSize, load(gpa, io, path));
    // corrupted header caps
    var bad: [24]u8 = undefined;
    writeHeader(&bad, nl, ne, 9999, dim, ffn);
    {
        const f = try Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer f.close(io);
        try f.writeStreamingAll(io, &bad);
        try f.writeStreamingAll(io, &payload);
    }
    try t.expectError(error.BadHeader, load(gpa, io, path));
}
