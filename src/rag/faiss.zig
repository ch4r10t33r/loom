//! FAISS via dlopen, the same trade as the Vulkan loader: loom must build
//! with no FAISS SDK and run with no FAISS library, declining cleanly in
//! both cases. The surface is the C API's flat inner-product index --
//! vectors are L2-normalized by the store, so inner product is cosine.
//! When the library is absent the store's exact scan serves instead; FAISS
//! is an accelerator, never a correctness dependency.
const std = @import("std");

const F = struct {
    faiss_IndexFlatIP_new_with: *const fn (*?*anyopaque, i64) callconv(.c) c_int,
    faiss_Index_add: *const fn (?*anyopaque, i64, [*]const f32) callconv(.c) c_int,
    faiss_Index_search: *const fn (?*anyopaque, i64, [*]const f32, i64, [*]f32, [*]i64) callconv(.c) c_int,
    faiss_Index_ntotal: *const fn (?*anyopaque) callconv(.c) i64,
};
var f: F = undefined;
var state: enum { unprobed, absent, loaded } = .unprobed;

fn load() bool {
    if (state == .unprobed) {
        state = .absent;
        blk: {
            var lib = std.DynLib.open("libfaiss_c.so") catch
                std.DynLib.open("libfaiss_c.dylib") catch break :blk;
            inline for (@typeInfo(F).@"struct".fields) |fld| {
                @field(f, fld.name) = lib.lookup(fld.type, fld.name) orelse break :blk;
            }
            state = .loaded;
        }
    }
    return state == .loaded;
}

pub const Index = struct {
    handle: ?*anyopaque,

    /// Null when FAISS is unavailable or refuses the dimension; callers use
    /// the exact scan instead.
    pub fn init(dim: usize) ?Index {
        if (!load()) return null;
        var h: ?*anyopaque = null;
        if (f.faiss_IndexFlatIP_new_with(&h, @intCast(dim)) != 0) return null;
        return .{ .handle = h };
    }

    pub fn add(self: *Index, vec: []const f32) bool {
        return f.faiss_Index_add(self.handle, 1, vec.ptr) == 0;
    }

    /// Top-k by inner product; returns the number of valid results. `ids`
    /// and `scores` must hold k entries.
    pub fn search(self: *Index, query: []const f32, k: usize, scores: []f32, ids: []i64) usize {
        if (f.faiss_Index_search(self.handle, 1, query.ptr, @intCast(k), scores.ptr, ids.ptr) != 0) return 0;
        var n: usize = 0;
        while (n < k and ids[n] >= 0) n += 1;
        return n;
    }

    pub fn count(self: *Index) usize {
        return @intCast(@max(f.faiss_Index_ntotal(self.handle), 0));
    }
};
