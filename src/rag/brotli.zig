//! Brotli, vendored and statically linked (build.zig compiles the pinned
//! v1.1.0 C sources into every target). Used for AT-REST chunk text only --
//! the wire applies adaptive snappy per frame, and at-rest format is a
//! purely local choice. Unlike FAISS this is a build-time dependency by
//! deliberate contrast: pure C with zero dependencies earns static linking;
//! C++ + BLAS + OpenMP does not.
const std = @import("std");

extern fn BrotliEncoderCompress(quality: c_int, lgwin: c_int, mode: c_int, input_size: usize, input: [*]const u8, encoded_size: *usize, encoded: [*]u8) c_int;
extern fn BrotliDecoderDecompress(encoded_size: usize, encoded: [*]const u8, decoded_size: *usize, decoded: [*]u8) c_int;

/// Compress into an owned buffer; null when the result would not be smaller
/// (same keep-only-when-it-pays rule as the frame encoder's snappy).
pub fn compressAlloc(gpa: std.mem.Allocator, data: []const u8) ?[]u8 {
    if (data.len == 0) return null;
    var out_len: usize = data.len; // only accepted if strictly smaller anyway
    const buf = gpa.alloc(u8, out_len) catch return null;
    // quality 5, window 22, mode 1 (TEXT): the ratio/speed point for prose.
    if (BrotliEncoderCompress(5, 22, 1, data.len, data.ptr, &out_len, buf.ptr) != 1 or out_len >= data.len) {
        gpa.free(buf);
        return null;
    }
    if (gpa.realloc(buf, out_len)) |shrunk| return shrunk else |_| return buf[0..out_len];
}

/// Decompress into an owned buffer of exactly `raw_len` (stored beside the
/// chunk); null on any mismatch.
pub fn decompressAlloc(gpa: std.mem.Allocator, data: []const u8, raw_len: usize) ?[]u8 {
    const buf = gpa.alloc(u8, raw_len) catch return null;
    var out_len: usize = raw_len;
    if (BrotliDecoderDecompress(data.len, data.ptr, &out_len, buf.ptr) != 1 or out_len != raw_len) {
        gpa.free(buf);
        return null;
    }
    return buf;
}

test "vendored brotli round-trips and actually compresses prose" {
    const gpa = std.testing.allocator;
    const text = "the weaver wove and wove and wove the same thread again " ** 16;
    const c = compressAlloc(gpa, text).?; // vendored: must be present AND pay
    defer gpa.free(c);
    try std.testing.expect(c.len < text.len / 2);
    const d = decompressAlloc(gpa, c, text.len).?;
    defer gpa.free(d);
    try std.testing.expectEqualStrings(text, d);
}
