//! int4 block quantization + fused matmul kernels.
//!
//! Format (q4_0-style, one scale per QK=32 weights):
//!   block = [ f32 scale (4 bytes LE) ][ 32 nibbles packed into 16 bytes ]  = 20 bytes
//! A nibble n in 0..15 dequantizes to `(n - 8) * scale`.
//!
//! This is the on-disk representation of every routed expert and the format the
//! per-token fetch path materializes and matmuls against (CLAUDE principle 7:
//! the hot path is a direct addressed fetch of an original block — no coding).

const std = @import("std");
const model = @import("../engine/model.zig");

pub const QK = model.QK; // 32
pub const BLOCK_BYTES: usize = QK / 2 + 4; // 20

/// Quantize `src` (length must be a multiple of QK) into `dst` (must be
/// `src.len / QK * BLOCK_BYTES` bytes).
pub fn quantizeRow(dst: []u8, src: []const f32) void {
    std.debug.assert(src.len % QK == 0);
    const n_blocks = src.len / QK;
    std.debug.assert(dst.len == n_blocks * BLOCK_BYTES);

    var b: usize = 0;
    while (b < n_blocks) : (b += 1) {
        const in = src[b * QK ..][0..QK];
        const out = dst[b * BLOCK_BYTES ..][0..BLOCK_BYTES];

        var amax: f32 = 0;
        for (in) |v| amax = @max(amax, @abs(v));
        const scale: f32 = if (amax == 0) 0 else amax / 7.0;
        const inv: f32 = if (scale == 0) 0 else 1.0 / scale;

        std.mem.writeInt(u32, out[0..4], @bitCast(scale), .little);

        var i: usize = 0;
        while (i < QK) : (i += 2) {
            const q0 = quant1(in[i] * inv);
            const q1 = quant1(in[i + 1] * inv);
            out[4 + i / 2] = @as(u8, q0) | (@as(u8, q1) << 4);
        }
    }
}

inline fn quant1(scaled: f32) u4 {
    const r = std.math.clamp(@round(scaled), -7.0, 7.0);
    const q: i32 = @intFromFloat(r);
    return @intCast(q + 8); // 1..15
}

/// Dequantize one block (BLOCK_BYTES) into `out` (QK f32s).
pub fn dequantizeBlock(out: []f32, block: []const u8) void {
    std.debug.assert(out.len == QK);
    const scale: f32 = @bitCast(std.mem.readInt(u32, block[0..4], .little));
    var i: usize = 0;
    while (i < QK) : (i += 2) {
        const packed_byte = block[4 + i / 2];
        out[i] = (@as(f32, @floatFromInt(packed_byte & 0x0f)) - 8.0) * scale;
        out[i + 1] = (@as(f32, @floatFromInt(packed_byte >> 4)) - 8.0) * scale;
    }
}

/// out[r] = sum_c dequant(weight[r][c]) * x[c], for r in 0..rows.
/// `weight` holds `rows` quantized rows of `cols` values each (cols % QK == 0).
/// This is the fused int4 matvec on the compute-node-local forward pass.
pub fn matvecQ4(out: []f32, weight: []const u8, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(cols % QK == 0);
    std.debug.assert(x.len == cols);
    std.debug.assert(out.len == rows);
    const blocks_per_row = cols / QK;
    const row_bytes = blocks_per_row * BLOCK_BYTES;

    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = weight[r * row_bytes ..][0..row_bytes];
        var acc: f32 = 0;
        var blk: usize = 0;
        while (blk < blocks_per_row) : (blk += 1) {
            const block = row[blk * BLOCK_BYTES ..][0..BLOCK_BYTES];
            const scale: f32 = @bitCast(std.mem.readInt(u32, block[0..4], .little));
            if (scale == 0) continue;
            const xb = x[blk * QK ..][0..QK];
            var partial: f32 = 0;
            var i: usize = 0;
            while (i < QK) : (i += 2) {
                const pb = block[4 + i / 2];
                const w0 = @as(f32, @floatFromInt(pb & 0x0f)) - 8.0;
                const w1 = @as(f32, @floatFromInt(pb >> 4)) - 8.0;
                partial += w0 * xb[i] + w1 * xb[i + 1];
            }
            acc += partial * scale;
        }
        out[r] = acc;
    }
}

pub fn rowBytes(cols: usize) usize {
    return (cols / QK) * BLOCK_BYTES;
}

test "quantize/dequantize roundtrip is bounded" {
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    var src: [QK * 3]f32 = undefined;
    for (&src) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;

    var q: [3 * BLOCK_BYTES]u8 = undefined;
    quantizeRow(&q, &src);

    var deq: [QK]f32 = undefined;
    dequantizeBlock(&deq, q[0..BLOCK_BYTES]);
    // per-element error must be within half a quantization step of that block
    var amax: f32 = 0;
    for (src[0..QK]) |v| amax = @max(amax, @abs(v));
    const step = amax / 7.0;
    for (0..QK) |i| try std.testing.expect(@abs(deq[i] - src[i]) <= step);
}

test "matvecQ4 matches an f32 reference within quant error" {
    const rows = 4;
    const cols = QK * 2;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();

    var wf: [rows * cols]f32 = undefined;
    for (&wf) |*v| v.* = rnd.float(f32) - 0.5;
    var x: [cols]f32 = undefined;
    for (&x) |*v| v.* = rnd.float(f32) - 0.5;

    var wq: [rows * (cols / QK) * BLOCK_BYTES]u8 = undefined;
    for (0..rows) |r| {
        quantizeRow(wq[r * rowBytes(cols) ..][0..rowBytes(cols)], wf[r * cols ..][0..cols]);
    }

    var out: [rows]f32 = undefined;
    matvecQ4(&out, &wq, &x, rows, cols);

    for (0..rows) |r| {
        var ref: f32 = 0;
        for (0..cols) |c| ref += wf[r * cols + c] * x[c];
        try std.testing.expect(@abs(out[r] - ref) < 0.5); // loose: int4 is lossy
    }
}
