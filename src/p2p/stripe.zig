//! Striped range groups: systematic Reed-Solomon over GF(2^8) for the
//! propagation plane (whitepaper: "EC & RLNC -- for propagation and
//! durability, not the hot path"; the construction follows commonware's
//! stripes write-up, minus the multi-core decode machinery loom's
//! minutes-scale bulk events do not need).
//!
//! Geometry: consecutive manifest ranges form groups of K_DATA; each group
//! has M_PARITY parity pieces, computed over the ranges zero-padded to the
//! group's longest range. Any k of the k+m pieces reconstruct the group.
//! The generator is [I; C] with C a Cauchy matrix derived only from (k, m,
//! row, column), so parity is deterministic: every full holder computes
//! byte-identical parity, and pieces from different peers compose.
//!
//! Trust model, deliberately minimal: parity pieces are NOT separately
//! committed. Reconstructed DATA ranges are verified against the manifest's
//! per-range digests (the store's existing root of trust) before they touch
//! disk, so a polluted parity piece costs one failed decode and a retry with
//! different sources, never a corrupt store. This is the property that makes
//! plain RS safe from untrusted relays where recoding schemes (RLNC) need
//! homomorphic hashing: pieces are fixed objects, and the final artifact is
//! checked against a commitment the swarm already agrees on.
//!
//! Never in the token loop (principle 7): callers are the bulk sync and
//! future rollout/durability tiers only.

const std = @import("std");

pub const K_DATA: usize = 8;
pub const M_PARITY: usize = 2;
pub const MAX_PIECES: usize = 32; // k ceiling for the fixed work arrays

// ---- GF(2^8), polynomial 0x11d --------------------------------------------

const gf_tables = blk: {
    @setEvalBranchQuota(10_000);
    var exp: [512]u8 = undefined;
    var log: [256]u8 = undefined;
    var x: usize = 1;
    for (0..255) |i| {
        exp[i] = @intCast(x);
        log[x] = @intCast(i);
        x <<= 1;
        if (x >= 256) x ^= 0x11d;
    }
    for (255..512) |i| exp[i] = exp[i - 255];
    log[0] = 0; // never read: mul/inv guard zero explicitly
    break :blk .{ .exp = exp, .log = log };
};

fn gfMul(a: u8, b: u8) u8 {
    if (a == 0 or b == 0) return 0;
    return gf_tables.exp[@as(usize, gf_tables.log[a]) + gf_tables.log[b]];
}

fn gfInv(a: u8) u8 {
    std.debug.assert(a != 0);
    return gf_tables.exp[255 - @as(usize, gf_tables.log[a])];
}

/// Cauchy coefficient for parity row `r` (0..m) against data column `j`
/// (0..k): inv(x_r ^ y_j) with x = {0..m-1}, y = {m..m+k-1}. The sets are
/// disjoint so the denominator is never zero, and every square submatrix of
/// a Cauchy matrix is nonsingular -- the property that makes any k of k+m
/// pieces decode.
pub fn coeff(r: usize, j: usize, m: usize) u8 {
    return gfInv(@as(u8, @intCast(r)) ^ @as(u8, @intCast(m + j)));
}

// ---- encode ----------------------------------------------------------------

/// out[0..data.len] ^= c * data, elementwise. The streaming form of
/// encodeRow: a server reads one range at a time and folds it in, so peak
/// memory is two pieces, not k+1. Bytes of `out` past data.len are the
/// zero padding of a short range and stay untouched (0 * c = 0).
pub fn accumulate(c: u8, data: []const u8, out: []u8) void {
    for (out[0..data.len], data) |*o, b| o.* ^= gfMul(c, b);
}

/// One parity piece: out = sum_j coeff(r, j) * data[j]. All slices equal
/// length; out is fully overwritten. Computing rows independently lets a
/// server answer a single STRIPE request without building the others.
pub fn encodeRow(r: usize, m: usize, data: []const []const u8, out: []u8) void {
    @memset(out, 0);
    for (data, 0..) |d, j| accumulate(coeff(r, j, m), d, out);
}

// ---- decode ----------------------------------------------------------------

pub const Piece = struct {
    /// 0..k-1 = data piece with that index; k..k+m-1 = parity row (index-k).
    index: usize,
    bytes: []const u8,
};

/// Reconstruct all k data pieces from any k distinct pieces. `out` holds k
/// destination slices, every slice (and every piece) the same length.
/// A *wrong-but-well-formed* piece yields wrong output; the caller catches
/// that by digest verification (see module doc).
pub fn decode(k: usize, m: usize, pieces: []const Piece, out: []const []u8) !void {
    if (k == 0 or k > MAX_PIECES or out.len != k) return error.BadGeometry;
    const plen = out[0].len;
    for (out) |o| if (o.len != plen) return error.BadGeometry;

    // Take the first k distinct, well-formed piece indices offered.
    var used: [MAX_PIECES]Piece = undefined;
    var seen = std.StaticBitSet(2 * MAX_PIECES).initEmpty();
    var n: usize = 0;
    for (pieces) |p| {
        if (n == k) break;
        if (p.index >= k + m or seen.isSet(p.index)) continue;
        if (p.bytes.len != plen) return error.BadPiece;
        seen.set(p.index);
        used[n] = p;
        n += 1;
    }
    if (n < k) return error.NotEnoughPieces;

    // Generator rows for the pieces in hand.
    var mat: [MAX_PIECES][MAX_PIECES]u8 = undefined;
    for (used[0..k], 0..) |p, row| {
        for (0..k) |col| {
            mat[row][col] = if (p.index < k)
                (if (p.index == col) 1 else 0)
            else
                coeff(p.index - k, col, m);
        }
    }

    // Invert via Gauss-Jordan over GF(256).
    var inv: [MAX_PIECES][MAX_PIECES]u8 = undefined;
    for (0..k) |i| {
        for (0..k) |j| inv[i][j] = if (i == j) 1 else 0;
    }
    for (0..k) |col| {
        var pivot = col;
        while (pivot < k and mat[pivot][col] == 0) pivot += 1;
        if (pivot == k) return error.SingularMatrix; // unreachable for distinct indices
        if (pivot != col) {
            std.mem.swap([MAX_PIECES]u8, &mat[pivot], &mat[col]);
            std.mem.swap([MAX_PIECES]u8, &inv[pivot], &inv[col]);
        }
        const pinv = gfInv(mat[col][col]);
        for (0..k) |j| {
            mat[col][j] = gfMul(mat[col][j], pinv);
            inv[col][j] = gfMul(inv[col][j], pinv);
        }
        for (0..k) |row| {
            if (row == col or mat[row][col] == 0) continue;
            const f = mat[row][col];
            for (0..k) |j| {
                mat[row][j] ^= gfMul(f, mat[col][j]);
                inv[row][j] ^= gfMul(f, inv[col][j]);
            }
        }
    }

    // data[i] = sum_row inv[i][row] * used[row].bytes
    for (0..k) |i| {
        @memset(out[i], 0);
        for (0..k) |row| {
            const c = inv[i][row];
            if (c == 0) continue;
            for (out[i], used[row].bytes) |*o, b| o.* ^= gfMul(c, b);
        }
    }
}

// ---- tests -----------------------------------------------------------------

test "gf field sanity: a * inv(a) == 1 for every nonzero a" {
    var a: usize = 1;
    while (a < 256) : (a += 1) {
        try std.testing.expectEqual(@as(u8, 1), gfMul(@intCast(a), gfInv(@intCast(a))));
    }
}

test "every 4-of-6 erasure pattern reconstructs (k=4, m=2)" {
    const gpa = std.testing.allocator;
    const k = 4;
    const m = 2;
    const len = 97; // odd on purpose
    var prng = std.Random.DefaultPrng.init(0x57121);
    const rnd = prng.random();

    var storage: [k + m][]u8 = undefined;
    for (&storage) |*s| s.* = try gpa.alloc(u8, len);
    defer for (storage) |s| gpa.free(s);
    for (storage[0..k]) |s| rnd.bytes(s);
    var data_const: [k][]const u8 = undefined;
    for (0..k) |i| data_const[i] = storage[i];
    for (0..m) |r| encodeRow(r, m, &data_const, storage[k + r]);

    var out_storage: [k][]u8 = undefined;
    for (&out_storage) |*s| s.* = try gpa.alloc(u8, len);
    defer for (out_storage) |s| gpa.free(s);

    // every subset of size k from k+m pieces
    var mask: usize = 0;
    while (mask < (1 << (k + m))) : (mask += 1) {
        if (@popCount(mask) != k) continue;
        var pieces: [k]Piece = undefined;
        var n: usize = 0;
        for (0..k + m) |i| {
            if (mask & (@as(usize, 1) << @intCast(i)) != 0) {
                pieces[n] = .{ .index = i, .bytes = storage[i] };
                n += 1;
            }
        }
        var outs: [k][]u8 = out_storage;
        try decode(k, m, &pieces, &outs);
        for (0..k) |i| try std.testing.expectEqualSlices(u8, storage[i], out_storage[i]);
    }
}

test "k=8 m=2 random erasures at range-like sizes reconstruct" {
    const gpa = std.testing.allocator;
    const k = K_DATA;
    const m = M_PARITY;
    const len = 4096;
    var prng = std.Random.DefaultPrng.init(0xC0DE);
    const rnd = prng.random();

    var storage: [k + m][]u8 = undefined;
    for (&storage) |*s| s.* = try gpa.alloc(u8, len);
    defer for (storage) |s| gpa.free(s);
    for (storage[0..k]) |s| rnd.bytes(s);
    var data_const: [k][]const u8 = undefined;
    for (0..k) |i| data_const[i] = storage[i];
    for (0..m) |r| encodeRow(r, m, &data_const, storage[k + r]);

    var out_storage: [k][]u8 = undefined;
    for (&out_storage) |*s| s.* = try gpa.alloc(u8, len);
    defer for (out_storage) |s| gpa.free(s);

    // drop two random data pieces, replace with the two parity pieces
    var trial: usize = 0;
    while (trial < 20) : (trial += 1) {
        const drop_a = rnd.uintLessThan(usize, k);
        var drop_b = rnd.uintLessThan(usize, k);
        while (drop_b == drop_a) drop_b = rnd.uintLessThan(usize, k);
        var pieces: [k]Piece = undefined;
        var n: usize = 0;
        for (0..k) |i| {
            if (i == drop_a or i == drop_b) continue;
            pieces[n] = .{ .index = i, .bytes = storage[i] };
            n += 1;
        }
        pieces[n] = .{ .index = k, .bytes = storage[k] };
        pieces[n + 1] = .{ .index = k + 1, .bytes = storage[k + 1] };
        var outs: [k][]u8 = out_storage;
        try decode(k, m, &pieces, &outs);
        for (0..k) |i| try std.testing.expectEqualSlices(u8, storage[i], out_storage[i]);
    }
}

test "parity is deterministic and error paths hold" {
    const gpa = std.testing.allocator;
    const len = 64;
    var d0: [len]u8 = undefined;
    var d1: [len]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    prng.random().bytes(&d0);
    prng.random().bytes(&d1);
    const data = [_][]const u8{ &d0, &d1 };
    var p_a: [len]u8 = undefined;
    var p_b: [len]u8 = undefined;
    encodeRow(0, 1, &data, &p_a);
    encodeRow(0, 1, &data, &p_b);
    try std.testing.expectEqualSlices(u8, &p_a, &p_b);

    const out0 = try gpa.alloc(u8, len);
    defer gpa.free(out0);
    const out1 = try gpa.alloc(u8, len);
    defer gpa.free(out1);
    var outs = [_][]u8{ out0, out1 };
    const one = [_]Piece{.{ .index = 0, .bytes = &d0 }};
    try std.testing.expectError(error.NotEnoughPieces, decode(2, 1, &one, &outs));
    var short: [len - 1]u8 = undefined;
    const bad = [_]Piece{ .{ .index = 0, .bytes = &d0 }, .{ .index = 1, .bytes = &short } };
    try std.testing.expectError(error.BadPiece, decode(2, 1, &bad, &outs));
}
