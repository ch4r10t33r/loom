//! Byte-level tokenizer (vocab = 256). Enough to drive the engine end-to-end on
//! the synthetic model. The real GLM-5.2 BPE tokenizer is out of scope for v0
//! and is a prerequisite for the token-exact oracle comparison.

const std = @import("std");

pub const VOCAB: usize = 256;

pub fn encode(gpa: std.mem.Allocator, text: []const u8) ![]usize {
    const out = try gpa.alloc(usize, text.len);
    for (text, 0..) |b, i| out[i] = b;
    return out;
}

pub fn decodeByte(token: usize) u8 {
    return @intCast(token & 0xff);
}

test "roundtrip bytes" {
    const gpa = std.testing.allocator;
    const toks = try encode(gpa, "hi");
    defer gpa.free(toks);
    try std.testing.expect(toks.len == 2);
    try std.testing.expect(decodeByte(toks[0]) == 'h');
}
