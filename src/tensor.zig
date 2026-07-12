//! Small f32 tensor helpers for the resident (dense) compute path: RMSNorm,
//! softmax, SwiGLU, dense matvec, and rotary position embedding. The dense
//! sub-model (attention, norms, dense FFN, shared expert) stays resident and is
//! kept in f32 here for clarity; only the streamed routed experts are int4.

const std = @import("std");

/// out = x / rms(x) * weight, with rms over `x`. In-place-safe if out != x aliasing rules hold.
pub fn rmsnorm(out: []f32, x: []const f32, weight: []const f32, eps: f32) void {
    std.debug.assert(out.len == x.len and x.len == weight.len);
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const scale = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    for (out, x, weight) |*o, v, w| o.* = v * scale * w;
}

/// In-place numerically-stable softmax.
pub fn softmax(x: []f32) void {
    var m: f32 = -std.math.inf(f32);
    for (x) |v| m = @max(m, v);
    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - m);
        sum += v.*;
    }
    const inv = 1.0 / sum;
    for (x) |*v| v.* *= inv;
}

pub inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// out[r] = sum_c weight[r*cols + c] * x[c].
pub fn matvec(out: []f32, weight: []const f32, x: []const f32, rows: usize, cols: usize) void {
    std.debug.assert(x.len == cols and out.len == rows);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = weight[r * cols ..][0..cols];
        var acc: f32 = 0;
        for (row, x) |w, v| acc += w * v;
        out[r] = acc;
    }
}

pub fn add(dst: []f32, src: []const f32) void {
    for (dst, src) |*d, s| d.* += s;
}

/// SwiGLU FFN activation: out = silu(gate) * up, elementwise.
pub fn swiglu(out: []f32, gate: []const f32, up: []const f32) void {
    for (out, gate, up) |*o, g, u| o.* = silu(g) * u;
}

/// Apply rotary embedding in-place to a `rope_dim`-length slice at position
/// `pos`. NeoX/half-split convention: element i pairs with i + rope_dim/2.
pub fn rope(vec: []f32, pos: usize, theta: f32) void {
    const half = vec.len / 2;
    var i: usize = 0;
    while (i < half) : (i += 1) {
        const exponent = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(vec.len));
        const freq = std.math.pow(f32, theta, -exponent);
        const angle = @as(f32, @floatFromInt(pos)) * freq;
        const cos = @cos(angle);
        const sin = @sin(angle);
        const a = vec[i];
        const b = vec[i + half];
        vec[i] = a * cos - b * sin;
        vec[i + half] = a * sin + b * cos;
    }
}

test "rmsnorm normalizes to unit rms when weight is 1" {
    var x = [_]f32{ 3, -3, 3, -3 };
    var w = [_]f32{ 1, 1, 1, 1 };
    var out: [4]f32 = undefined;
    rmsnorm(&out, &x, &w, 1e-6);
    var ss: f32 = 0;
    for (out) |v| ss += v * v;
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), ss, 1e-3); // rms==1 -> sum sq == n
}

test "softmax sums to one" {
    var x = [_]f32{ 1, 2, 3, 4 };
    softmax(&x);
    var s: f32 = 0;
    for (x) |v| s += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), s, 1e-6);
}

test "rope preserves norm" {
    var v = [_]f32{ 0.5, -0.3, 0.9, 0.1 };
    const before = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    rope(&v, 5, 10000.0);
    const after = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    try std.testing.expectApproxEqAbs(before, after, 1e-4);
}
