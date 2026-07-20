//! Token sampling. Greedy by default (deterministic, for the correctness
//! harness); temperature sampling when a positive temperature is set.

const std = @import("std");
const tensor = @import("../core/tensor.zig");

pub fn argmax(logits: []const f32) usize {
    var best: usize = 0;
    var best_v: f32 = -std.math.inf(f32);
    for (logits, 0..) |v, i| if (v > best_v) {
        best_v = v;
        best = i;
    };
    return best;
}

/// `temp <= 0` is greedy. Otherwise softmax(logits/temp) then inverse-CDF sample.
pub fn sample(scratch: []f32, logits: []const f32, temp: f32, rnd: std.Random) usize {
    if (temp <= 0) return argmax(logits);
    std.debug.assert(scratch.len == logits.len);
    const inv = 1.0 / temp;
    for (scratch, logits) |*s, v| s.* = v * inv;
    tensor.softmax(scratch);
    var r = rnd.float(f32);
    for (scratch, 0..) |p, i| {
        r -= p;
        if (r <= 0) return i;
    }
    return scratch.len - 1;
}

test "argmax finds the max index" {
    const l = [_]f32{ -1, 3, 2, 3.5, 0 };
    try std.testing.expect(argmax(&l) == 3);
}
