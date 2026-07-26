//! GGML tensor formats — the quantizations GGUF files actually ship.
//!
//! Supported: F32, F16, Q4_0 (18-byte blocks: f16 scale + 32 nibbles),
//! Q8_0 (34-byte blocks: f16 scale + 32 int8). Each gets a fused
//! matvec over the raw tensor bytes so weights are never dequantized
//! wholesale. Note these differ from loom's own q4 expert format
//! (quant.zig, f32 scales): here we implement GGML's layouts exactly.

const std = @import("std");

pub const Type = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q5_0 = 6,
    q8_0 = 8,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    _,

    pub fn supported(t: u32) bool {
        return switch (@as(Type, @enumFromInt(t))) {
            .f32, .f16, .q4_0, .q5_0, .q8_0, .q4_k, .q5_k, .q6_k => true,
            _ => false,
        };
    }
};

pub const QK_0: usize = 32; // block width for q4_0 / q8_0
const Q4_0_BLOCK: usize = 2 + QK_0 / 2; // f16 scale + 16 nibble bytes = 18
const Q5_0_BLOCK: usize = 2 + 4 + QK_0 / 2; // f16 scale + 32 high bits + nibbles = 22
const Q8_0_BLOCK: usize = 2 + QK_0; // f16 scale + 32 int8 = 34

pub const QK_K: usize = 256; // super-block width for K-quants
const Q4_K_BLOCK: usize = 2 + 2 + 12 + QK_K / 2; // d, dmin, 6-bit scales, nibbles = 144
const Q5_K_BLOCK: usize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2; // + high bits = 176
const Q6_K_BLOCK: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; // ql, qh, scales, d = 210

/// Bytes of one row of `n` elements in format `t`. `n` must be a multiple of
/// the block width for quantized types.
pub fn rowBytes(t: Type, n: usize) usize {
    return switch (t) {
        .f32 => n * 4,
        .f16 => n * 2,
        .q4_0 => (n / QK_0) * Q4_0_BLOCK,
        .q5_0 => (n / QK_0) * Q5_0_BLOCK,
        .q8_0 => (n / QK_0) * Q8_0_BLOCK,
        .q4_k => (n / QK_K) * Q4_K_BLOCK,
        .q5_k => (n / QK_K) * Q5_K_BLOCK,
        .q6_k => (n / QK_K) * Q6_K_BLOCK,
        _ => unreachable,
    };
}

pub fn tensorBytes(t: Type, ne0: usize, rows: usize) usize {
    return rows * rowBytes(t, ne0);
}

/// Block size in elements for `t` (1 for unquantized types). A quantized row
/// length must be a whole number of blocks: `rowBytes` divides, so a ragged
/// ne0 silently under-counts the bytes a kernel will actually walk.
pub fn blockElems(t: Type) usize {
    return switch (t) {
        .f32, .f16 => 1,
        .q4_0, .q5_0, .q8_0 => QK_0,
        .q4_k, .q5_k, .q6_k => QK_K,
        _ => 1,
    };
}

/// Overflow-checked `tensorBytes` for sizes read from an untrusted file
/// (security issue #29). Also rejects a row length that is not a whole number
/// of quantization blocks.
pub fn tensorBytesChecked(t: Type, ne0: usize, rows: usize, ne2: usize) !usize {
    const blk = blockElems(t);
    if (blk != 1 and ne0 % blk != 0) return error.BadTensorShape;
    const per_row = switch (t) {
        .f32 => try std.math.mul(usize, ne0, 4),
        .f16 => try std.math.mul(usize, ne0, 2),
        else => try std.math.mul(usize, ne0 / blk, blockBytes(t)),
    };
    const per_slice = try std.math.mul(usize, rows, per_row);
    return std.math.mul(usize, per_slice, ne2);
}

/// Bytes per quantization block for `t`.
pub fn blockBytes(t: Type) usize {
    return switch (t) {
        .q4_0 => Q4_0_BLOCK,
        .q5_0 => Q5_0_BLOCK,
        .q8_0 => Q8_0_BLOCK,
        .q4_k => Q4_K_BLOCK,
        .q5_k => Q5_K_BLOCK,
        .q6_k => Q6_K_BLOCK,
        else => 0,
    };
}

inline fn f16FromBytes(b: []const u8) f32 {
    const bits = std.mem.readInt(u16, b[0..2], .little);
    return @floatCast(@as(f16, @bitCast(bits)));
}

/// out[r] = sum_c W[r][c] * x[c] over a row-major tensor of `rows` rows and
/// `cols` columns stored in format `t` at `data`.
pub fn matvec(t: Type, out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(out.len == rows and x.len == cols);
    switch (t) {
        .f32 => matvecF32(out, data, x, rows, cols),
        .f16 => matvecF16(out, data, x, rows, cols),
        .q4_0 => matvecQ40(out, data, x, rows, cols),
        .q5_0 => matvecQ50(out, data, x, rows, cols),
        .q8_0 => matvecQ80(out, data, x, rows, cols),
        .q4_k, .q5_k, .q6_k => matvecK(t, out, data, x, rows, cols),
        _ => unreachable,
    }
}

/// K-quant matvec: dequantize one 256-wide super-block at a time and dot it.
fn matvecK(t: Type, out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(cols % QK_K == 0);
    const bs: usize = switch (t) {
        .q4_k => Q4_K_BLOCK,
        .q5_k => Q5_K_BLOCK,
        .q6_k => Q6_K_BLOCK,
        else => unreachable,
    };
    const blocks_per_row = cols / QK_K;
    const rb = blocks_per_row * bs;
    var vals: [QK_K]f32 = undefined;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const block = row[b * bs ..][0..bs];
            switch (t) {
                .q4_k => dequantBlockQ4K(block, &vals),
                .q5_k => dequantBlockQ5K(block, &vals),
                .q6_k => dequantBlockQ6K(block, &vals),
                else => unreachable,
            }
            const xb = x[b * QK_K ..][0..QK_K];
            var partial: f32 = 0;
            for (vals, xb) |v, xv| partial += v * xv;
            acc += partial;
        }
        out[r] = acc;
    }
}

/// 6-bit scale/min unpacking shared by q4_k/q5_k (llama.cpp get_scale_min_k4).
inline fn scaleMinK4(j: usize, scales: *const [12]u8, sc: *u8, m: *u8) void {
    if (j < 4) {
        sc.* = scales[j] & 63;
        m.* = scales[j + 4] & 63;
    } else {
        sc.* = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        m.* = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

fn dequantBlockQ4K(block: []const u8, vals: *[QK_K]f32) void {
    const d = f16FromBytes(block[0..2]);
    const dmin = f16FromBytes(block[2..4]);
    const scales: *const [12]u8 = block[4..16];
    const qs = block[16..][0 .. QK_K / 2];
    var is: usize = 0;
    var y: usize = 0;
    var q: usize = 0;
    var j: usize = 0;
    while (j < QK_K) : (j += 64) {
        var sc: u8 = undefined;
        var mn: u8 = undefined;
        scaleMinK4(is, scales, &sc, &mn);
        const d1 = d * @as(f32, @floatFromInt(sc));
        const m1 = dmin * @as(f32, @floatFromInt(mn));
        scaleMinK4(is + 1, scales, &sc, &mn);
        const d2 = d * @as(f32, @floatFromInt(sc));
        const m2 = dmin * @as(f32, @floatFromInt(mn));
        var l: usize = 0;
        while (l < 32) : (l += 1) {
            vals[y + l] = d1 * @as(f32, @floatFromInt(qs[q + l] & 0xF)) - m1;
        }
        l = 0;
        while (l < 32) : (l += 1) {
            vals[y + 32 + l] = d2 * @as(f32, @floatFromInt(qs[q + l] >> 4)) - m2;
        }
        y += 64;
        q += 32;
        is += 2;
    }
}

fn dequantBlockQ5K(block: []const u8, vals: *[QK_K]f32) void {
    const d = f16FromBytes(block[0..2]);
    const dmin = f16FromBytes(block[2..4]);
    const scales: *const [12]u8 = block[4..16];
    const qh = block[16..][0 .. QK_K / 8];
    const qs = block[16 + QK_K / 8 ..][0 .. QK_K / 2];
    var is: usize = 0;
    var y: usize = 0;
    var q: usize = 0;
    var hb1: u8 = 1;
    var hb2: u8 = 2;
    var j: usize = 0;
    while (j < QK_K) : (j += 64) {
        var sc: u8 = undefined;
        var mn: u8 = undefined;
        scaleMinK4(is, scales, &sc, &mn);
        const d1 = d * @as(f32, @floatFromInt(sc));
        const m1 = dmin * @as(f32, @floatFromInt(mn));
        scaleMinK4(is + 1, scales, &sc, &mn);
        const d2 = d * @as(f32, @floatFromInt(sc));
        const m2 = dmin * @as(f32, @floatFromInt(mn));
        var l: usize = 0;
        while (l < 32) : (l += 1) {
            const hi: f32 = if (qh[l] & hb1 != 0) 16 else 0;
            vals[y + l] = d1 * (@as(f32, @floatFromInt(qs[q + l] & 0xF)) + hi) - m1;
        }
        l = 0;
        while (l < 32) : (l += 1) {
            const hi: f32 = if (qh[l] & hb2 != 0) 16 else 0;
            vals[y + 32 + l] = d2 * (@as(f32, @floatFromInt(qs[q + l] >> 4)) + hi) - m2;
        }
        y += 64;
        q += 32;
        is += 2;
        hb1 <<= 2;
        hb2 <<= 2;
    }
}

fn dequantBlockQ6K(block: []const u8, vals: *[QK_K]f32) void {
    const ql_all = block[0 .. QK_K / 2];
    const qh_all = block[QK_K / 2 ..][0 .. QK_K / 4];
    const sc_all = block[QK_K / 2 + QK_K / 4 ..][0 .. QK_K / 16];
    const d = f16FromBytes(block[QK_K / 2 + QK_K / 4 + QK_K / 16 ..][0..2]);

    var y: usize = 0;
    var qlo: usize = 0;
    var qho: usize = 0;
    var sco: usize = 0;
    var n: usize = 0;
    while (n < QK_K) : (n += 128) {
        var l: usize = 0;
        while (l < 32) : (l += 1) {
            const is = l / 16;
            const ql = ql_all[qlo..];
            const qh = qh_all[qho..];
            const sc = sc_all[sco..];
            const q1: i32 = @as(i32, (ql[l] & 0xF) | ((qh[l] >> 0 & 3) << 4)) - 32;
            const q2: i32 = @as(i32, (ql[l + 32] & 0xF) | ((qh[l] >> 2 & 3) << 4)) - 32;
            const q3: i32 = @as(i32, (ql[l] >> 4) | ((qh[l] >> 4 & 3) << 4)) - 32;
            const q4: i32 = @as(i32, (ql[l + 32] >> 4) | ((qh[l] >> 6 & 3) << 4)) - 32;
            vals[y + l] = d * i8f(sc[is]) * @as(f32, @floatFromInt(q1));
            vals[y + l + 32] = d * i8f(sc[is + 2]) * @as(f32, @floatFromInt(q2));
            vals[y + l + 64] = d * i8f(sc[is + 4]) * @as(f32, @floatFromInt(q3));
            vals[y + l + 96] = d * i8f(sc[is + 6]) * @as(f32, @floatFromInt(q4));
        }
        y += 128;
        qlo += 64;
        qho += 32;
        sco += 8;
    }
}

inline fn i8f(b: u8) f32 {
    return @floatFromInt(@as(i8, @bitCast(b)));
}

fn matvecF32(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    const w: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, data[0 .. rows * cols * 4]));
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = w[r * cols ..][0..cols];
        var acc: f32 = 0;
        for (row, x) |wv, xv| acc += wv * xv;
        out[r] = acc;
    }
}

fn matvecF16(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    const w: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, data[0 .. rows * cols * 2]));
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = w[r * cols ..][0..cols];
        var acc: f32 = 0;
        for (row, x) |bits, xv| {
            acc += @as(f32, @floatCast(@as(f16, @bitCast(bits)))) * xv;
        }
        out[r] = acc;
    }
}

fn matvecQ40(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(cols % QK_0 == 0);
    const blocks_per_row = cols / QK_0;
    const rb = blocks_per_row * Q4_0_BLOCK;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const block = row[b * Q4_0_BLOCK ..][0..Q4_0_BLOCK];
            const scale = f16FromBytes(block[0..2]);
            if (scale == 0) continue;
            const xb = x[b * QK_0 ..][0..QK_0];
            var partial: f32 = 0;
            // GGML q4_0 nibble layout: byte j holds elements j (low nibble)
            // and j+16 (high nibble); value = (nibble - 8) * scale.
            var j: usize = 0;
            while (j < QK_0 / 2) : (j += 1) {
                const byte = block[2 + j];
                const lo = @as(f32, @floatFromInt(@as(i8, @intCast(byte & 0x0f)) - 8));
                const hi = @as(f32, @floatFromInt(@as(i8, @intCast(byte >> 4)) - 8));
                partial += lo * xb[j] + hi * xb[j + QK_0 / 2];
            }
            acc += partial * scale;
        }
        out[r] = acc;
    }
}

fn matvecQ50(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(cols % QK_0 == 0);
    const blocks_per_row = cols / QK_0;
    const rb = blocks_per_row * Q5_0_BLOCK;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const block = row[b * Q5_0_BLOCK ..][0..Q5_0_BLOCK];
            const scale = f16FromBytes(block[0..2]);
            if (scale == 0) continue;
            const qh = std.mem.readInt(u32, block[2..6], .little);
            const qs = block[6..][0 .. QK_0 / 2];
            const xb = x[b * QK_0 ..][0..QK_0];
            var partial: f32 = 0;
            var j: u5 = 0;
            while (true) {
                const xh0: u8 = @intCast(((qh >> j) << 4) & 0x10);
                const xh1: u8 = @intCast((qh >> (j + 12)) & 0x10);
                const w0 = @as(f32, @floatFromInt(@as(i32, (qs[j] & 0xF) | xh0) - 16));
                const w1 = @as(f32, @floatFromInt(@as(i32, (qs[j] >> 4) | xh1) - 16));
                partial += w0 * xb[j] + w1 * xb[@as(usize, j) + QK_0 / 2];
                if (j == 15) break;
                j += 1;
            }
            acc += partial * scale;
        }
        out[r] = acc;
    }
}

fn matvecQ80(out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(cols % QK_0 == 0);
    const blocks_per_row = cols / QK_0;
    const rb = blocks_per_row * Q8_0_BLOCK;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = data[r * rb ..][0..rb];
        var acc: f32 = 0;
        var b: usize = 0;
        while (b < blocks_per_row) : (b += 1) {
            const block = row[b * Q8_0_BLOCK ..][0..Q8_0_BLOCK];
            const scale = f16FromBytes(block[0..2]);
            if (scale == 0) continue;
            const xb = x[b * QK_0 ..][0..QK_0];
            var partial: f32 = 0;
            var j: usize = 0;
            while (j < QK_0) : (j += 1) {
                partial += @as(f32, @floatFromInt(@as(i8, @bitCast(block[2 + j])))) * xb[j];
            }
            acc += partial * scale;
        }
        out[r] = acc;
    }
}

/// Read row `r` of an F32 tensor directly (embeddings lookup).
pub fn f32Row(data: []const u8, r: usize, cols: usize) []const f32 {
    const w: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, data));
    return w[r * cols ..][0..cols];
}

/// Dequantize row `r` of a tensor in format `t` into `out` (embeddings lookup
/// for non-f32 token_embd).
pub fn dequantRow(t: Type, out: []f32, data: []const u8, r: usize, cols: usize) void {
    switch (t) {
        .f32 => @memcpy(out, f32Row(data, r, cols)),
        .f16 => {
            const w: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, data));
            for (out, w[r * cols ..][0..cols]) |*o, bits| {
                o.* = @floatCast(@as(f16, @bitCast(bits)));
            }
        },
        .q4_k, .q5_k, .q6_k => {
            const rb = rowBytes(t, cols);
            const row = data[r * rb ..][0..rb];
            const bs: usize = switch (t) {
                .q4_k => Q4_K_BLOCK,
                .q5_k => Q5_K_BLOCK,
                else => Q6_K_BLOCK,
            };
            var vals: [QK_K]f32 = undefined;
            var b: usize = 0;
            while (b * QK_K < cols) : (b += 1) {
                const block = row[b * bs ..][0..bs];
                switch (t) {
                    .q4_k => dequantBlockQ4K(block, &vals),
                    .q5_k => dequantBlockQ5K(block, &vals),
                    else => dequantBlockQ6K(block, &vals),
                }
                @memcpy(out[b * QK_K ..][0..QK_K], &vals);
            }
        },
        .q5_0 => {
            // embeddings are never q5_0 in practice; go through a 1-row matvec
            // against basis vectors would be wasteful, so walk blocks directly
            const rb = rowBytes(t, cols);
            const row = data[r * rb ..][0..rb];
            var b: usize = 0;
            while (b * QK_0 < cols) : (b += 1) {
                const block = row[b * Q5_0_BLOCK ..][0..Q5_0_BLOCK];
                const scale = f16FromBytes(block[0..2]);
                const qh = std.mem.readInt(u32, block[2..6], .little);
                const qs = block[6..][0 .. QK_0 / 2];
                const ob = out[b * QK_0 ..][0..QK_0];
                var j: u5 = 0;
                while (true) {
                    const xh0: u8 = @intCast(((qh >> j) << 4) & 0x10);
                    const xh1: u8 = @intCast((qh >> (j + 12)) & 0x10);
                    ob[j] = @as(f32, @floatFromInt(@as(i32, (qs[j] & 0xF) | xh0) - 16)) * scale;
                    ob[@as(usize, j) + QK_0 / 2] = @as(f32, @floatFromInt(@as(i32, (qs[j] >> 4) | xh1) - 16)) * scale;
                    if (j == 15) break;
                    j += 1;
                }
            }
        },
        .q4_0, .q8_0 => {
            // one-row matvec against basis vectors would be wasteful; walk blocks
            const rb = rowBytes(t, cols);
            const row = data[r * rb ..][0..rb];
            const bs: usize = if (t == .q4_0) Q4_0_BLOCK else Q8_0_BLOCK;
            var b: usize = 0;
            while (b * QK_0 < cols) : (b += 1) {
                const block = row[b * bs ..][0..bs];
                const scale = f16FromBytes(block[0..2]);
                const ob = out[b * QK_0 ..][0..QK_0];
                if (t == .q4_0) {
                    var j: usize = 0;
                    while (j < QK_0 / 2) : (j += 1) {
                        const byte = block[2 + j];
                        ob[j] = @as(f32, @floatFromInt(@as(i8, @intCast(byte & 0x0f)) - 8)) * scale;
                        ob[j + QK_0 / 2] = @as(f32, @floatFromInt(@as(i8, @intCast(byte >> 4)) - 8)) * scale;
                    }
                } else {
                    var j: usize = 0;
                    while (j < QK_0) : (j += 1) {
                        ob[j] = @as(f32, @floatFromInt(@as(i8, @bitCast(block[2 + j])))) * scale;
                    }
                }
            }
        },
        _ => unreachable,
    }
}

// ---- test-only quantizers (reference encoders for kernel validation) --------

fn quantizeQ40(dst: []u8, src: []const f32) void {
    std.debug.assert(src.len % QK_0 == 0);
    var b: usize = 0;
    while (b * QK_0 < src.len) : (b += 1) {
        const in = src[b * QK_0 ..][0..QK_0];
        const block = dst[b * Q4_0_BLOCK ..][0..Q4_0_BLOCK];
        var amax: f32 = 0;
        var vmax: f32 = 0;
        for (in) |v| {
            if (@abs(v) > amax) {
                amax = @abs(v);
                vmax = v;
            }
        }
        const scale: f32 = vmax / -8.0;
        const half: f16 = @floatCast(scale);
        std.mem.writeInt(u16, block[0..2], @bitCast(half), .little);
        const inv: f32 = if (scale != 0) 1.0 / scale else 0;
        var j: usize = 0;
        while (j < QK_0 / 2) : (j += 1) {
            const lo: u8 = @intCast(std.math.clamp(@as(i32, @intFromFloat(in[j] * inv + 8.5)), 0, 15));
            const hi: u8 = @intCast(std.math.clamp(@as(i32, @intFromFloat(in[j + QK_0 / 2] * inv + 8.5)), 0, 15));
            block[2 + j] = lo | (hi << 4);
        }
    }
}

fn quantizeQ80(dst: []u8, src: []const f32) void {
    std.debug.assert(src.len % QK_0 == 0);
    var b: usize = 0;
    while (b * QK_0 < src.len) : (b += 1) {
        const in = src[b * QK_0 ..][0..QK_0];
        const block = dst[b * Q8_0_BLOCK ..][0..Q8_0_BLOCK];
        var amax: f32 = 0;
        for (in) |v| amax = @max(amax, @abs(v));
        const scale: f32 = amax / 127.0;
        const half: f16 = @floatCast(scale);
        std.mem.writeInt(u16, block[0..2], @bitCast(half), .little);
        const inv: f32 = if (scale != 0) 1.0 / scale else 0;
        for (in, 0..) |v, j| {
            block[2 + j] = @bitCast(@as(i8, @intCast(std.math.clamp(@as(i32, @intFromFloat(@round(v * inv))), -127, 127))));
        }
    }
}

test "q4_0 and q8_0 matvec match f32 reference within quant error" {
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const rows = 4;
    const cols = QK_0 * 2;

    var wf: [rows * cols]f32 = undefined;
    for (&wf) |*v| v.* = rnd.float(f32) - 0.5;
    var x: [cols]f32 = undefined;
    for (&x) |*v| v.* = rnd.float(f32) - 0.5;

    var ref: [rows]f32 = undefined;
    matvecF32(&ref, std.mem.sliceAsBytes(&wf), &x, rows, cols);

    var q4: [rows * (cols / QK_0) * Q4_0_BLOCK]u8 = undefined;
    var q8: [rows * (cols / QK_0) * Q8_0_BLOCK]u8 = undefined;
    for (0..rows) |r| {
        quantizeQ40(q4[r * rowBytes(.q4_0, cols) ..][0..rowBytes(.q4_0, cols)], wf[r * cols ..][0..cols]);
        quantizeQ80(q8[r * rowBytes(.q8_0, cols) ..][0..rowBytes(.q8_0, cols)], wf[r * cols ..][0..cols]);
    }

    var out4: [rows]f32 = undefined;
    var out8: [rows]f32 = undefined;
    matvec(.q4_0, &out4, &q4, &x, rows, cols);
    matvec(.q8_0, &out8, &q8, &x, rows, cols);

    for (0..rows) |r| {
        try std.testing.expect(@abs(out4[r] - ref[r]) < 0.6); // int4 is lossy
        try std.testing.expect(@abs(out8[r] - ref[r]) < 0.05); // int8 is close
    }
}

test "q4_k dequant with handcrafted block" {
    var block = [_]u8{0} ** Q4_K_BLOCK;
    // d = 1.0, dmin = 0.0
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 1.0)), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(@as(f16, 0.0)), .little);
    // sub-blocks 0..3: sc = 1 (scales[0..4]=1), min = 0 (scales[4..8]=0)
    block[4] = 1;
    block[5] = 1;
    block[6] = 1;
    block[7] = 1;
    // qs = 0x31: low nibble 1, high nibble 3
    for (block[16..]) |*b| b.* = 0x31;

    var vals: [QK_K]f32 = undefined;
    dequantBlockQ4K(&block, &vals);
    // first 128 elems: groups of (32 x low=1, 32 x high=3); last 128: sc=0 -> 0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vals[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), vals[32], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vals[64], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), vals[96], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vals[128], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vals[255], 1e-4);

    // matvec against ones = sum = 32*(1+3)*2 = 256
    const x = [_]f32{1.0} ** QK_K;
    var out: [1]f32 = undefined;
    matvec(.q4_k, &out, &block, &x, 1, QK_K);
    try std.testing.expectApproxEqAbs(@as(f32, 256.0), out[0], 1e-2);
}

test "q6_k dequant with handcrafted block" {
    var block = [_]u8{0} ** Q6_K_BLOCK;
    // ql=0, qh=0 -> q = -32 everywhere; scales all 1; d = 0.5
    for (block[QK_K / 2 + QK_K / 4 ..][0 .. QK_K / 16]) |*b| b.* = 1;
    std.mem.writeInt(u16, block[QK_K / 2 + QK_K / 4 + QK_K / 16 ..][0..2], @bitCast(@as(f16, 0.5)), .little);

    var vals: [QK_K]f32 = undefined;
    dequantBlockQ6K(&block, &vals);
    for (vals) |v| try std.testing.expectApproxEqAbs(@as(f32, -16.0), v, 1e-4);
}

test "q5_k high bit adds 16" {
    var block = [_]u8{0} ** Q5_K_BLOCK;
    std.mem.writeInt(u16, block[0..2], @bitCast(@as(f16, 1.0)), .little);
    std.mem.writeInt(u16, block[2..4], @bitCast(@as(f16, 0.0)), .little);
    block[4] = 1; // sc(sub-block 0) = 1
    // qh bit0 set for l=0 -> element 0 gets +16; qs all zero
    block[16] = 1;
    var vals: [QK_K]f32 = undefined;
    dequantBlockQ5K(&block, &vals);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), vals[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vals[1], 1e-4);
}

test "f16 matvec and dequantRow" {
    const cols = 4;
    var bits: [2 * cols]u16 = undefined;
    const vals = [_]f32{ 1.0, -2.0, 0.5, 4.0, 0.25, 3.0, -1.0, 2.0 };
    for (&bits, vals) |*b, v| b.* = @bitCast(@as(f16, @floatCast(v)));
    const x = [_]f32{ 1, 1, 1, 1 };
    var out: [2]f32 = undefined;
    matvec(.f16, &out, std.mem.sliceAsBytes(&bits), &x, 2, cols);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 4.25), out[1], 1e-3);

    var row: [cols]f32 = undefined;
    dequantRow(.f16, &row, std.mem.sliceAsBytes(&bits), 1, cols);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), row[0], 1e-3);
}

test "tensorBytesChecked rejects overflow and ragged quant rows (issue #29)" {
    // ne0 * 4 must not wrap: 1<<62 f32 elements overflows usize
    try std.testing.expectError(error.Overflow, tensorBytesChecked(.f32, 1 << 62, 4, 1));
    // rows * per_row must not wrap
    try std.testing.expectError(error.Overflow, tensorBytesChecked(.f32, 1 << 40, 1 << 40, 1));
    // a quantized row must be a whole number of blocks, else rowBytes
    // under-counts the bytes the kernel actually walks
    try std.testing.expectError(error.BadTensorShape, tensorBytesChecked(.q4_0, 33, 1, 1));
    // sane shapes still compute
    try std.testing.expectEqual(@as(usize, 4 * 16), try tensorBytesChecked(.f32, 16, 1, 1));
    try std.testing.expectEqual(@as(usize, Q4_0_BLOCK * 2), try tensorBytesChecked(.q4_0, 64, 1, 1));
}
