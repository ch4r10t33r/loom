//! IQ (importance-matrix) and MXFP4 block decoders.
//!
//! These are *codebook* quantizations, not affine ones. Q4_0 and the K-quants
//! reconstruct a value arithmetically (`scale * q + min`), so their kernels are
//! a mask, a shift and a multiply-add. An IQ block instead stores an **index**
//! into a static grid table plus packed sign bits, and the value comes out of a
//! table lookup — there is nothing to derive, which is why `iq_tables.zig` is
//! transcribed from llama.cpp rather than computed.
//!
//! Consequence for review: a wrong table entry or a mis-shifted index does not
//! crash, it silently returns plausible-looking garbage weights. Every decoder
//! here is a line-by-line transliteration of the matching
//! `dequantize_row_*` in llama.cpp's `ggml/src/ggml-quants.c`, and the tests
//! check block sizes and decoded values against vectors taken from that
//! implementation.
//!
//! Decode only: loom never quantizes, so there are no encoders.

const std = @import("std");
const t = @import("iq_tables.zig");

pub const QK_K: usize = 256;
pub const QK_NL: usize = 32; // iq4_nl and mxfp4 block width

const IQ1S_DELTA: f32 = 0.125;

// Block sizes in bytes, matching the `static_assert`s in ggml-common.h.
pub const IQ2_XXS_BLOCK: usize = 2 + QK_K / 8 * 2; // 66
pub const IQ2_XS_BLOCK: usize = 2 + QK_K / 8 * 2 + QK_K / 32; // 74
pub const IQ2_S_BLOCK: usize = 2 + QK_K / 4 + QK_K / 16; // 82
pub const IQ3_XXS_BLOCK: usize = 2 + 3 * (QK_K / 8); // 98
pub const IQ3_S_BLOCK: usize = 2 + 13 * (QK_K / 32) + QK_K / 64; // 110
pub const IQ1_S_BLOCK: usize = 2 + QK_K / 8 + QK_K / 16; // 50
pub const IQ1_M_BLOCK: usize = QK_K / 8 + QK_K / 16 + QK_K / 32; // 56
pub const IQ4_XS_BLOCK: usize = 2 + 2 + QK_K / 64 + QK_K / 2; // 136
pub const IQ4_NL_BLOCK: usize = 2 + QK_NL / 2; // 18
pub const MXFP4_BLOCK: usize = 1 + QK_NL / 2; // 17

inline fn f16FromBytes(b: []const u8) f32 {
    const bits = std.mem.readInt(u16, b[0..2], .little);
    return @floatCast(@as(f16, @bitCast(bits)));
}

/// E8M0 shared exponent, halved. MXFP4's E2M1 value table is stored doubled
/// (see `kvalues_fp4`), so the scale carries the compensating 1/2.
/// Mirrors ggml_e8m0_to_fp32_half in ggml-impl.h.
inline fn e8m0ToF32Half(x: u8) f32 {
    // x < 2 lands in the denormal range, where shifting the exponent field
    // would underflow; llama.cpp uses precomputed patterns for those two.
    const bits: u32 = if (x < 2)
        @as(u32, 0x00200000) << @intCast(x)
    else
        @as(u32, x - 1) << 23;
    return @bitCast(bits);
}

/// Byte `j` of grid word `w` — the grids pack 8 (u64) or 4 (u32) codebook
/// values per entry, and llama.cpp reads them through a `uint8_t *`, i.e. in
/// little-endian byte order.
inline fn gbyte(w: u64, j: usize) u8 {
    return @truncate(w >> @intCast(8 * j));
}

/// Sign for lane `j` given a 7-bit sign index already mapped through
/// `ksigns_iq2xs`: bit j set means negate.
inline fn sgn(signs: u8, j: usize) f32 {
    return if (signs & t.kmask_iq2xs[j] != 0) -1.0 else 1.0;
}

// ---- 256-wide super-block decoders ------------------------------------------

pub fn dequantBlockIq2XXS(block: []const u8, vals: *[QK_K]f32) void {
    const d = f16FromBytes(block[0..2]);
    const qs = block[2..][0 .. QK_K / 4]; // 32 u16, read as pairs of u32
    var y: usize = 0;
    var ib32: usize = 0;
    while (ib32 < QK_K / 32) : (ib32 += 1) {
        const a0 = std.mem.readInt(u32, qs[8 * ib32 ..][0..4], .little);
        const a1 = std.mem.readInt(u32, qs[8 * ib32 + 4 ..][0..4], .little);
        const db = d * (0.5 + @as(f32, @floatFromInt(a1 >> 28))) * 0.25;
        var l: usize = 0;
        while (l < 4) : (l += 1) {
            const idx: u8 = @truncate(a0 >> @intCast(8 * l));
            const grid = t.iq2xxs_grid[idx];
            const signs = t.ksigns_iq2xs[(a1 >> @intCast(7 * l)) & 127];
            var j: usize = 0;
            while (j < 8) : (j += 1) {
                vals[y + j] = db * @as(f32, @floatFromInt(gbyte(grid, j))) * sgn(signs, j);
            }
            y += 8;
        }
    }
}

pub fn dequantBlockIq2XS(block: []const u8, vals: *[QK_K]f32) void {
    const d = f16FromBytes(block[0..2]);
    const qs = block[2..][0 .. QK_K / 4];
    const scales = block[2 + QK_K / 4 ..][0 .. QK_K / 32];
    var y: usize = 0;
    var ib32: usize = 0;
    while (ib32 < QK_K / 32) : (ib32 += 1) {
        const db = [2]f32{
            d * (0.5 + @as(f32, @floatFromInt(scales[ib32] & 0xf))) * 0.25,
            d * (0.5 + @as(f32, @floatFromInt(scales[ib32] >> 4))) * 0.25,
        };
        var l: usize = 0;
        while (l < 4) : (l += 1) {
            const q = std.mem.readInt(u16, qs[2 * (4 * ib32 + l) ..][0..2], .little);
            const grid = t.iq2xs_grid[q & 511];
            const signs = t.ksigns_iq2xs[q >> 9];
            const dl = db[l / 2];
            var j: usize = 0;
            while (j < 8) : (j += 1) {
                vals[y + j] = dl * @as(f32, @floatFromInt(gbyte(grid, j))) * sgn(signs, j);
            }
            y += 8;
        }
    }
}

pub fn dequantBlockIq2S(block: []const u8, vals: *[QK_K]f32) void {
    const d = f16FromBytes(block[0..2]);
    const qs_all = block[2..][0 .. QK_K / 4]; // first half indices, second half signs
    const qh = block[2 + QK_K / 4 ..][0 .. QK_K / 32];
    const scales = block[2 + QK_K / 4 + QK_K / 32 ..][0 .. QK_K / 32];
    var y: usize = 0;
    var ib32: usize = 0;
    while (ib32 < QK_K / 32) : (ib32 += 1) {
        const qs = qs_all[4 * ib32 ..][0..4];
        const signs = qs_all[QK_K / 8 + 4 * ib32 ..][0..4];
        const db = [2]f32{
            d * (0.5 + @as(f32, @floatFromInt(scales[ib32] & 0xf))) * 0.25,
            d * (0.5 + @as(f32, @floatFromInt(scales[ib32] >> 4))) * 0.25,
        };
        var l: usize = 0;
        while (l < 4) : (l += 1) {
            const hi = (@as(usize, qh[ib32]) << @intCast(8 - 2 * l)) & 0x300;
            const grid = t.iq2s_grid[@as(usize, qs[l]) | hi];
            const dl = db[l / 2];
            var j: usize = 0;
            while (j < 8) : (j += 1) {
                vals[y + j] = dl * @as(f32, @floatFromInt(gbyte(grid, j))) * sgn(signs[l], j);
            }
            y += 8;
        }
    }
}

pub fn dequantBlockIq3XXS(block: []const u8, vals: *[QK_K]f32) void {
    const d = f16FromBytes(block[0..2]);
    const qs = block[2..][0 .. QK_K / 4];
    const sas = block[2 + QK_K / 4 ..][0 .. QK_K / 8]; // scales and signs
    var y: usize = 0;
    var ib32: usize = 0;
    while (ib32 < QK_K / 32) : (ib32 += 1) {
        const aux = std.mem.readInt(u32, sas[4 * ib32 ..][0..4], .little);
        const db = d * (0.5 + @as(f32, @floatFromInt(aux >> 28))) * 0.5;
        const q = qs[8 * ib32 ..][0..8];
        var l: usize = 0;
        while (l < 4) : (l += 1) {
            const signs = t.ksigns_iq2xs[(aux >> @intCast(7 * l)) & 127];
            const g1 = t.iq3xxs_grid[q[2 * l + 0]];
            const g2 = t.iq3xxs_grid[q[2 * l + 1]];
            var j: usize = 0;
            while (j < 4) : (j += 1) {
                vals[y + j + 0] = db * @as(f32, @floatFromInt(gbyte(g1, j))) * sgn(signs, j);
                vals[y + j + 4] = db * @as(f32, @floatFromInt(gbyte(g2, j))) * sgn(signs, j + 4);
            }
            y += 8;
        }
    }
}

pub fn dequantBlockIq3S(block: []const u8, vals: *[QK_K]f32) void {
    const d = f16FromBytes(block[0..2]);
    const qs = block[2..][0 .. QK_K / 4];
    const qh = block[2 + QK_K / 4 ..][0 .. QK_K / 32];
    const signs = block[2 + QK_K / 4 + QK_K / 32 ..][0 .. QK_K / 8];
    const scales = block[2 + QK_K / 4 + QK_K / 32 + QK_K / 8 ..][0 .. QK_K / 64];
    var y: usize = 0;
    var ib32: usize = 0;
    // two 32-groups per scale byte, each with its own qh byte
    while (ib32 < QK_K / 32) : (ib32 += 2) {
        const sc = scales[ib32 / 2];
        const dbs = [2]f32{
            d * (1 + 2 * @as(f32, @floatFromInt(sc & 0xf))),
            d * (1 + 2 * @as(f32, @floatFromInt(sc >> 4))),
        };
        var half: usize = 0;
        while (half < 2) : (half += 1) {
            const db = dbs[half];
            const h = qh[ib32 + half];
            const q = qs[8 * (ib32 + half) ..][0..8];
            const sg = signs[4 * (ib32 + half) ..][0..4];
            var l: usize = 0;
            while (l < 4) : (l += 1) {
                const hi1 = (@as(usize, h) << @intCast(8 - 2 * l)) & 256;
                const hi2 = (@as(usize, h) << @intCast(7 - 2 * l)) & 256;
                const g1 = t.iq3s_grid[@as(usize, q[2 * l + 0]) | hi1];
                const g2 = t.iq3s_grid[@as(usize, q[2 * l + 1]) | hi2];
                var j: usize = 0;
                while (j < 4) : (j += 1) {
                    vals[y + j + 0] = db * @as(f32, @floatFromInt(gbyte(g1, j))) * sgn(sg[l], j);
                    vals[y + j + 4] = db * @as(f32, @floatFromInt(gbyte(g2, j))) * sgn(sg[l], j + 4);
                }
                y += 8;
            }
        }
    }
}

pub fn dequantBlockIq1S(block: []const u8, vals: *[QK_K]f32) void {
    const d = f16FromBytes(block[0..2]);
    const qs = block[2..][0 .. QK_K / 8];
    const qh_bytes = block[2 + QK_K / 8 ..][0 .. QK_K / 16];
    var y: usize = 0;
    var ib: usize = 0;
    while (ib < QK_K / 32) : (ib += 1) {
        const qh = std.mem.readInt(u16, qh_bytes[2 * ib ..][0..2], .little);
        const dl = d * @as(f32, @floatFromInt(2 * ((qh >> 12) & 7) + 1));
        const delta: f32 = if (qh & 0x8000 != 0) -IQ1S_DELTA else IQ1S_DELTA;
        var l: usize = 0;
        while (l < 4) : (l += 1) {
            const idx = @as(usize, qs[4 * ib + l]) | (@as(usize, (qh >> @intCast(3 * l)) & 7) << 8);
            const grid = t.iq1s_grid[idx];
            var j: usize = 0;
            while (j < 8) : (j += 1) {
                // iq1s_grid packs *signed* codebook values, unlike the iq2/iq3 grids
                const g: f32 = @floatFromInt(@as(i8, @bitCast(gbyte(grid, j))));
                vals[y + j] = dl * (g + delta);
            }
            y += 8;
        }
    }
}

pub fn dequantBlockIq1M(block: []const u8, vals: *[QK_K]f32) void {
    const qs = block[0 .. QK_K / 8];
    const qh = block[QK_K / 8 ..][0 .. QK_K / 16];
    const scales = block[QK_K / 8 + QK_K / 16 ..][0 .. QK_K / 32];
    // iq1_m has no block scale field: the f16 is scattered across the four
    // scale words, four bits at a time.
    var sc: [4]u16 = undefined;
    for (&sc, 0..) |*s, i| s.* = std.mem.readInt(u16, scales[2 * i ..][0..2], .little);
    const dbits: u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const d: f32 = @floatCast(@as(f16, @bitCast(dbits)));

    var y: usize = 0;
    var ib: usize = 0;
    while (ib < QK_K / 32) : (ib += 1) {
        const w = sc[ib / 2];
        const sh: u4 = @intCast(6 * (ib % 2));
        const dl1 = d * @as(f32, @floatFromInt(2 * ((w >> sh) & 0x7) + 1));
        const dl2 = d * @as(f32, @floatFromInt(2 * ((w >> (sh + 3)) & 0x7) + 1));
        const q = qs[4 * ib ..][0..4];
        const h = qh[2 * ib ..][0..2];
        const idx = [4]usize{
            @as(usize, q[0]) | ((@as(usize, h[0]) << 8) & 0x700),
            @as(usize, q[1]) | ((@as(usize, h[0]) << 4) & 0x700),
            @as(usize, q[2]) | ((@as(usize, h[1]) << 8) & 0x700),
            @as(usize, q[3]) | ((@as(usize, h[1]) << 4) & 0x700),
        };
        const delta = [4]f32{
            if (h[0] & 0x08 != 0) -IQ1S_DELTA else IQ1S_DELTA,
            if (h[0] & 0x80 != 0) -IQ1S_DELTA else IQ1S_DELTA,
            if (h[1] & 0x08 != 0) -IQ1S_DELTA else IQ1S_DELTA,
            if (h[1] & 0x80 != 0) -IQ1S_DELTA else IQ1S_DELTA,
        };
        var l: usize = 0;
        while (l < 4) : (l += 1) {
            const dl = if (l < 2) dl1 else dl2;
            const grid = t.iq1s_grid[idx[l]];
            var j: usize = 0;
            while (j < 8) : (j += 1) {
                const g: f32 = @floatFromInt(@as(i8, @bitCast(gbyte(grid, j))));
                vals[y + j] = dl * (g + delta[l]);
            }
            y += 8;
        }
    }
}

pub fn dequantBlockIq4XS(block: []const u8, vals: *[QK_K]f32) void {
    const d = f16FromBytes(block[0..2]);
    const scales_h = std.mem.readInt(u16, block[2..4], .little);
    const scales_l = block[4..][0 .. QK_K / 64];
    const qs = block[4 + QK_K / 64 ..][0 .. QK_K / 2];
    var y: usize = 0;
    var ib: usize = 0;
    while (ib < QK_K / 32) : (ib += 1) {
        // 6-bit scale: low nibble from scales_l, high 2 bits from scales_h
        const lo: u8 = (scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0xf;
        const hi: u8 = @intCast((scales_h >> @intCast(2 * ib)) & 3);
        const ls: i32 = @as(i32, lo) | (@as(i32, hi) << 4);
        const dl = d * @as(f32, @floatFromInt(ls - 32));
        const q = qs[16 * ib ..][0..16];
        var j: usize = 0;
        while (j < 16) : (j += 1) {
            vals[y + j + 0] = dl * @as(f32, @floatFromInt(t.kvalues_iq4nl[q[j] & 0xf]));
            vals[y + j + 16] = dl * @as(f32, @floatFromInt(t.kvalues_iq4nl[q[j] >> 4]));
        }
        y += 32;
    }
}

// ---- 32-wide block decoders --------------------------------------------------

pub fn dequantBlockIq4NL(block: []const u8, vals: *[QK_NL]f32) void {
    const d = f16FromBytes(block[0..2]);
    const qs = block[2..][0 .. QK_NL / 2];
    var j: usize = 0;
    while (j < QK_NL / 2) : (j += 1) {
        vals[j] = d * @as(f32, @floatFromInt(t.kvalues_iq4nl[qs[j] & 0xf]));
        vals[j + QK_NL / 2] = d * @as(f32, @floatFromInt(t.kvalues_iq4nl[qs[j] >> 4]));
    }
}

pub fn dequantBlockMxfp4(block: []const u8, vals: *[QK_NL]f32) void {
    const d = e8m0ToF32Half(block[0]);
    const qs = block[1..][0 .. QK_NL / 2];
    var j: usize = 0;
    while (j < QK_NL / 2) : (j += 1) {
        vals[j] = d * @as(f32, @floatFromInt(t.kvalues_fp4[qs[j] & 0xf]));
        vals[j + QK_NL / 2] = d * @as(f32, @floatFromInt(t.kvalues_fp4[qs[j] >> 4]));
    }
}

// ---- tests -------------------------------------------------------------------

test "decoders reproduce llama.cpp bit-for-bit on golden vectors" {
    // The load-bearing test for this file. Everything else here checks our own
    // reading of the format against itself; this checks it against the
    // implementation that produced the weights.
    const vectors = @import("iq_vectors.zig").vectors;
    var vals: [QK_K]f32 = undefined;
    for (vectors) |v| {
        const nblocks = v.input.len / v.block_bytes;
        var got: usize = 0;
        for (0..nblocks) |b| {
            const block = v.input[b * v.block_bytes ..][0..v.block_bytes];
            if (std.mem.eql(u8, v.name, "iq2_xxs")) dequantBlockIq2XXS(block, &vals) else if (std.mem.eql(u8, v.name, "iq2_xs")) dequantBlockIq2XS(block, &vals) else if (std.mem.eql(u8, v.name, "iq2_s")) dequantBlockIq2S(block, &vals) else if (std.mem.eql(u8, v.name, "iq3_xxs")) dequantBlockIq3XXS(block, &vals) else if (std.mem.eql(u8, v.name, "iq3_s")) dequantBlockIq3S(block, &vals) else if (std.mem.eql(u8, v.name, "iq1_s")) dequantBlockIq1S(block, &vals) else if (std.mem.eql(u8, v.name, "iq1_m")) dequantBlockIq1M(block, &vals) else if (std.mem.eql(u8, v.name, "iq4_xs")) dequantBlockIq4XS(block, &vals) else if (std.mem.eql(u8, v.name, "iq4_nl")) dequantBlockIq4NL(block, vals[0..QK_NL]) else if (std.mem.eql(u8, v.name, "mxfp4")) dequantBlockMxfp4(block, vals[0..QK_NL]) else unreachable;
            for (vals[0..v.width]) |f| {
                const want: f32 = @bitCast(v.expect[got]);
                if (f != want) {
                    std.debug.print("{s}: lane {d} got {d} want {d}\n", .{ v.name, got, f, want });
                    return error.DecoderMismatch;
                }
                got += 1;
            }
        }
        try std.testing.expectEqual(v.expect.len, got);
    }
}

test "table shapes match ggml-common.h" {
    try std.testing.expectEqual(@as(usize, 8), t.kmask_iq2xs.len);
    try std.testing.expectEqual(@as(usize, 128), t.ksigns_iq2xs.len);
    try std.testing.expectEqual(@as(usize, 256), t.iq2xxs_grid.len);
    try std.testing.expectEqual(@as(usize, 512), t.iq2xs_grid.len);
    try std.testing.expectEqual(@as(usize, 1024), t.iq2s_grid.len);
    try std.testing.expectEqual(@as(usize, 256), t.iq3xxs_grid.len);
    try std.testing.expectEqual(@as(usize, 512), t.iq3s_grid.len);
    try std.testing.expectEqual(@as(usize, 2048), t.iq1s_grid.len);
    try std.testing.expectEqual(@as(usize, 16), t.kvalues_iq4nl.len);
    try std.testing.expectEqual(@as(usize, 16), t.kvalues_fp4.len);
    // spot-check the two value tables, which are short enough to verify by eye
    try std.testing.expectEqualSlices(i8, &.{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 }, &t.kvalues_iq4nl);
    try std.testing.expectEqualSlices(i8, &.{ 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 }, &t.kvalues_fp4);
}

test "ksigns/kmask agree: bit j of the sign index negates lane j" {
    // ksigns_iq2xs[i] re-encodes i so that its low 7 bits are i and its parity
    // bit makes the popcount even. Lane signs therefore come straight from i.
    for (0..128) |i| {
        const s = t.ksigns_iq2xs[i];
        try std.testing.expectEqual(@as(u8, @intCast(i)), s & 127);
        try std.testing.expectEqual(@as(u32, 0), @popCount(s) % 2);
    }
}

test "e8m0 half-scale conversion" {
    // 127 is the exponent bias: 2^(127-127) halved = 0.5
    try std.testing.expectEqual(@as(f32, 0.5), e8m0ToF32Half(127));
    try std.testing.expectEqual(@as(f32, 1.0), e8m0ToF32Half(128));
    try std.testing.expectEqual(@as(f32, 2.0), e8m0ToF32Half(129));
    try std.testing.expectEqual(@as(f32, 0.25), e8m0ToF32Half(126));
    // the two denormal patterns the shift path cannot express
    try std.testing.expect(e8m0ToF32Half(0) > 0);
    try std.testing.expectEqual(e8m0ToF32Half(0) * 2, e8m0ToF32Half(1));
}

test "mxfp4 decodes the e2m1 value table at unit scale" {
    // e = 128 -> scale 1.0, and kvalues_fp4 is the doubled E2M1 table, so the
    // decoded values are exactly the table entries.
    var block: [MXFP4_BLOCK]u8 = undefined;
    block[0] = 128;
    for (0..16) |j| block[1 + j] = @intCast(j); // low nibble j, high nibble 0
    var vals: [QK_NL]f32 = undefined;
    dequantBlockMxfp4(&block, &vals);
    for (0..16) |j| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(t.kvalues_fp4[j])), vals[j]);
        try std.testing.expectEqual(@as(f32, 0), vals[j + 16]);
    }
}

test "iq4_nl decodes the non-linear value table at unit scale" {
    var block: [IQ4_NL_BLOCK]u8 = undefined;
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 1.0)), .little);
    for (0..16) |j| block[2 + j] = @as(u8, @intCast(j)) | (@as(u8, @intCast(15 - j)) << 4);
    var vals: [QK_NL]f32 = undefined;
    dequantBlockIq4NL(&block, &vals);
    for (0..16) |j| {
        try std.testing.expectEqual(@as(f32, @floatFromInt(t.kvalues_iq4nl[j])), vals[j]);
        try std.testing.expectEqual(@as(f32, @floatFromInt(t.kvalues_iq4nl[15 - j])), vals[j + 16]);
    }
}

test "iq4_xs scale is 6 bits split across scales_l and scales_h, biased by 32" {
    var block = [_]u8{0} ** IQ4_XS_BLOCK;
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 1.0)), .little);
    // sub-block 0: low nibble 0xf in scales_l[0], high bits 0b11 in scales_h
    // -> ls = 0x3f = 63, dl = 63 - 32 = 31
    block[4] = 0x0f;
    std.mem.writeInt(u16, block[2..4], 0x0003, .little);
    block[4 + QK_K / 64] = 0x0f; // first quant byte: both nibbles index 15
    var vals: [QK_K]f32 = undefined;
    dequantBlockIq4XS(&block, &vals);
    try std.testing.expectEqual(@as(f32, 31 * @as(f32, @floatFromInt(t.kvalues_iq4nl[15]))), vals[0]);
    // nibble 0 of every other byte -> kvalues[0], and sub-block 1 has scale
    // 0 - 32 = -32, so it is not silently zero either
    try std.testing.expectEqual(@as(f32, 31 * @as(f32, @floatFromInt(t.kvalues_iq4nl[0]))), vals[1]);
    try std.testing.expectEqual(@as(f32, -32 * @as(f32, @floatFromInt(t.kvalues_iq4nl[0]))), vals[32]);
}

test "iq2_xxs: grid index, sign index and the 4-bit block scale" {
    var block = [_]u8{0} ** IQ2_XXS_BLOCK;
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 1.0)), .little);
    // aux32[0] = grid indices for the four groups of 8; aux32[1] = sign
    // indices (7 bits each) in the low 28 bits, scale in the top 4.
    std.mem.writeInt(u32, block[2..6], 0x03020100, .little);
    std.mem.writeInt(u32, block[6..10], @as(u32, 5) << 28, .little);
    var vals: [QK_K]f32 = undefined;
    dequantBlockIq2XXS(&block, &vals);
    const db: f32 = (0.5 + 5.0) * 0.25;
    for (0..4) |l| {
        const grid = t.iq2xxs_grid[l];
        for (0..8) |j| {
            // sign index 0 -> ksigns_iq2xs[0] == 0 -> no lane negated
            try std.testing.expectEqual(db * @as(f32, @floatFromInt(gbyte(grid, j))), vals[8 * l + j]);
        }
    }
}

test "iq1_s: grid values are signed and the delta shifts every lane" {
    var block = [_]u8{0} ** IQ1_S_BLOCK;
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 1.0)), .little);
    // qh sub-block 0: scale bits (12..14) = 3 -> dl = 2*3+1 = 7, sign bit clear
    std.mem.writeInt(u16, block[2 + QK_K / 8 ..][0..2], @as(u16, 3) << 12, .little);
    var vals: [QK_K]f32 = undefined;
    dequantBlockIq1S(&block, &vals);
    const grid = t.iq1s_grid[0];
    for (0..8) |j| {
        const g: f32 = @floatFromInt(@as(i8, @bitCast(gbyte(grid, j))));
        try std.testing.expectEqual(7 * (g + IQ1S_DELTA), vals[j]);
    }
    // the negative-delta path must differ from the positive one
    std.mem.writeInt(u16, block[2 + QK_K / 8 ..][0..2], (@as(u16, 3) << 12) | 0x8000, .little);
    var neg: [QK_K]f32 = undefined;
    dequantBlockIq1S(&block, &neg);
    try std.testing.expect(neg[0] != vals[0]);
}

test "decoders cover every lane of a super-block" {
    // A decoder that walks its buffers wrongly typically leaves a tail
    // untouched; fill with a sentinel and require every lane to be written.
    const nan = std.math.nan(f32);
    var vals: [QK_K]f32 = .{nan} ** QK_K;
    const Case = struct { size: usize, f: *const fn ([]const u8, *[QK_K]f32) void };
    const cases = [_]Case{
        .{ .size = IQ2_XXS_BLOCK, .f = dequantBlockIq2XXS },
        .{ .size = IQ2_XS_BLOCK, .f = dequantBlockIq2XS },
        .{ .size = IQ2_S_BLOCK, .f = dequantBlockIq2S },
        .{ .size = IQ3_XXS_BLOCK, .f = dequantBlockIq3XXS },
        .{ .size = IQ3_S_BLOCK, .f = dequantBlockIq3S },
        .{ .size = IQ1_S_BLOCK, .f = dequantBlockIq1S },
        .{ .size = IQ1_M_BLOCK, .f = dequantBlockIq1M },
        .{ .size = IQ4_XS_BLOCK, .f = dequantBlockIq4XS },
    };
    var block = [_]u8{0x11} ** 256;
    for (cases) |c| {
        @memset(&vals, nan);
        c.f(block[0..c.size], &vals);
        for (vals) |v| try std.testing.expect(!std.math.isNan(v));
    }
}
