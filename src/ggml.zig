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
    q8_0 = 8,
    _,

    pub fn supported(t: u32) bool {
        return switch (@as(Type, @enumFromInt(t))) {
            .f32, .f16, .q4_0, .q8_0 => true,
            _ => false,
        };
    }
};

pub const QK_0: usize = 32; // block width for q4_0 / q8_0
const Q4_0_BLOCK: usize = 2 + QK_0 / 2; // f16 scale + 16 nibble bytes = 18
const Q8_0_BLOCK: usize = 2 + QK_0; // f16 scale + 32 int8 = 34

/// Bytes of one row of `n` elements in format `t`. `n` must be a multiple of
/// the block width for quantized types.
pub fn rowBytes(t: Type, n: usize) usize {
    return switch (t) {
        .f32 => n * 4,
        .f16 => n * 2,
        .q4_0 => (n / QK_0) * Q4_0_BLOCK,
        .q8_0 => (n / QK_0) * Q8_0_BLOCK,
        _ => unreachable,
    };
}

pub fn tensorBytes(t: Type, ne0: usize, rows: usize) usize {
    return rows * rowBytes(t, ne0);
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
        .q8_0 => matvecQ80(out, data, x, rows, cols),
        _ => unreachable,
    }
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
