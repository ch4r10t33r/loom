//! Brotli via dlopen, the same trade as the FAISS and Vulkan loaders: no
//! build dependency, no runtime requirement. Used for AT-REST chunk text
//! only -- the wire already applies adaptive snappy per frame, and at-rest
//! format is a purely local choice, so nodes with and without the library
//! interoperate unchanged. Absent library = text stored raw.
const std = @import("std");

const B = struct {
    BrotliEncoderCompress: *const fn (c_int, c_int, c_int, usize, [*]const u8, *usize, [*]u8) callconv(.c) c_int,
};
const D = struct {
    BrotliDecoderDecompress: *const fn (usize, [*]const u8, *usize, [*]u8) callconv(.c) c_int,
};
var be: B = undefined;
var bd: D = undefined;
var state: enum { unprobed, absent, loaded } = .unprobed;

fn load() bool {
    if (state == .unprobed) {
        state = .absent;
        blk: {
            var enc = std.DynLib.open("libbrotlienc.so.1") catch
                std.DynLib.open("libbrotlienc.dylib") catch break :blk;
            var dec = std.DynLib.open("libbrotlidec.so.1") catch
                std.DynLib.open("libbrotlidec.dylib") catch break :blk;
            be.BrotliEncoderCompress = enc.lookup(@TypeOf(be.BrotliEncoderCompress), "BrotliEncoderCompress") orelse break :blk;
            bd.BrotliDecoderDecompress = dec.lookup(@TypeOf(bd.BrotliDecoderDecompress), "BrotliDecoderDecompress") orelse break :blk;
            state = .loaded;
        }
    }
    return state == .loaded;
}

/// Compress into an owned buffer; null when the library is absent or the
/// result would not be smaller (same keep-only-when-it-pays rule as the
/// frame encoder's snappy).
pub fn compressAlloc(gpa: std.mem.Allocator, data: []const u8) ?[]u8 {
    if (!load() or data.len == 0) return null;
    var out_len: usize = data.len; // only accepted if strictly smaller anyway
    const buf = gpa.alloc(u8, out_len) catch return null;
    // quality 5, window 22, mode 1 (TEXT): the ratio/speed point for prose.
    if (be.BrotliEncoderCompress(5, 22, 1, data.len, data.ptr, &out_len, buf.ptr) != 1 or out_len >= data.len) {
        gpa.free(buf);
        return null;
    }
    if (gpa.realloc(buf, out_len)) |shrunk| return shrunk else |_| return buf[0..out_len];
}

/// Decompress into an owned buffer of exactly `raw_len` (stored beside the
/// chunk); null on any mismatch.
pub fn decompressAlloc(gpa: std.mem.Allocator, data: []const u8, raw_len: usize) ?[]u8 {
    if (!load()) return null;
    const buf = gpa.alloc(u8, raw_len) catch return null;
    var out_len: usize = raw_len;
    if (bd.BrotliDecoderDecompress(data.len, data.ptr, &out_len, buf.ptr) != 1 or out_len != raw_len) {
        gpa.free(buf);
        return null;
    }
    return buf;
}
