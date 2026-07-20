//! Content addressing and Merkle-rooting for the expert corpus.
//!
//! Principle 5 (CLAUDE.md): expert blocks are keyed by hash and the checkpoint
//! manifest is Merkle-rooted, giving a free integrity check against poisoned or
//! corrupted experts. We use SHA-256 for both the leaf digests and the tree.

const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Digest = [32]u8;

pub fn hashBlock(bytes: []const u8) Digest {
    var d: Digest = undefined;
    Sha256.hash(bytes, &d, .{});
    return d;
}

fn hashPair(a: Digest, b: Digest) Digest {
    var h = Sha256.init(.{});
    h.update(&a);
    h.update(&b);
    var d: Digest = undefined;
    h.final(&d);
    return d;
}

/// Merkle root over an ordered list of leaf digests. Odd nodes at any level are
/// promoted (duplicated) upward — the classic Bitcoin-style rule. An empty list
/// hashes to all-zero.
pub fn merkleRoot(gpa: std.mem.Allocator, leaves: []const Digest) !Digest {
    if (leaves.len == 0) return std.mem.zeroes(Digest);
    if (leaves.len == 1) return leaves[0];

    var level = try gpa.alloc(Digest, leaves.len);
    defer gpa.free(level);
    @memcpy(level, leaves);
    var len = leaves.len;

    while (len > 1) {
        var w: usize = 0;
        var r: usize = 0;
        while (r < len) : (r += 2) {
            const left = level[r];
            const right = if (r + 1 < len) level[r + 1] else left;
            level[w] = hashPair(left, right);
            w += 1;
        }
        len = w;
    }
    return level[0];
}

pub fn eql(a: Digest, b: Digest) bool {
    return std.mem.eql(u8, &a, &b);
}

pub fn toHex(d: Digest) [64]u8 {
    var out: [64]u8 = undefined;
    const hex = "0123456789abcdef";
    for (d, 0..) |byte, i| {
        out[i * 2] = hex[byte >> 4];
        out[i * 2 + 1] = hex[byte & 0xf];
    }
    return out;
}

test "merkle root is order-sensitive and stable" {
    const gpa = std.testing.allocator;
    const a = hashBlock("expert-a");
    const b = hashBlock("expert-b");
    const c = hashBlock("expert-c");

    const r1 = try merkleRoot(gpa, &.{ a, b, c });
    const r2 = try merkleRoot(gpa, &.{ a, b, c });
    try std.testing.expect(eql(r1, r2)); // deterministic

    const r3 = try merkleRoot(gpa, &.{ b, a, c });
    try std.testing.expect(!eql(r1, r3)); // order matters

    const single = try merkleRoot(gpa, &.{a});
    try std.testing.expect(eql(single, a));
}
