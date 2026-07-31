//! Vulkan backend: the CPU implementation with a device underneath the
//! matvec family. Everything the seam exports resolves to the CPU fallback
//! except `matvec`, which runs on the device when `--gpu-ops` asks, the
//! quantization has a kernel, and the shape fits -- the same bring-up the
//! Metal backend had, and verified the same way, against the exact
//! dequantize-then-dot oracle.
//!
//! On llvmpipe (the only Vulkan this project currently has) that proves
//! correctness and nothing else; every performance decision waits for
//! physical hardware. The kernel and this file already carry the Metal
//! effort's transferable lessons -- oracle before use, decline loudly on
//! shape, one counter that cannot lie -- because those were the expensive
//! part to learn.
const std = @import("std");
const cpu = @import("../compute/cpu.zig");
const ggml = @import("../gguf/ggml.zig");
const vk = @import("../vulkan/device.zig");

// SPIR-V is a stream of u32 words and vkCreateShaderModule takes a *u32, but
// @embedFile guarantees byte alignment only -- the cast panicked on the first
// real run. A comptime copy into an aligned array fixes it for good.
const dmmv_q4k_raw = @embedFile("../shaders/vulkan/dmmv_q4k.spv");
const dmmv_q4k_spv: [dmmv_q4k_raw.len]u8 align(4) = dmmv_q4k_raw.*;
const dmmv_q5_0_raw = @embedFile("../shaders/vulkan/dmmv_q5_0.spv");
const dmmv_q5_0_spv: [dmmv_q5_0_raw.len]u8 align(4) = dmmv_q5_0_raw.*;
const dmmv_q8_0_raw = @embedFile("../shaders/vulkan/dmmv_q8_0.spv");
const dmmv_q8_0_spv: [dmmv_q8_0_raw.len]u8 align(4) = dmmv_q8_0_raw.*;
const dmmv_q6_k_raw = @embedFile("../shaders/vulkan/dmmv_q6_k.spv");
const dmmv_q6_k_spv: [dmmv_q6_k_raw.len]u8 align(4) = dmmv_q6_k_raw.*;
const dmmv_f32_raw = @embedFile("../shaders/vulkan/dmmv_f32.spv");
const dmmv_f32_spv: [dmmv_f32_raw.len]u8 align(4) = dmmv_f32_raw.*;

pub const ExpertRef = cpu.ExpertRef;
pub const LayerSpec = cpu.LayerSpec;
pub const MAX_BATCH = cpu.MAX_BATCH;
pub const MlaFrameCfg = cpu.MlaFrameCfg;
pub const MlaLayerDesc = cpu.MlaLayerDesc;
pub const RoutedCfg = cpu.RoutedCfg;
pub const Shape = cpu.Shape;
pub const WeightRef = cpu.WeightRef;
pub const add = cpu.add;
pub const attnHeads = cpu.attnHeads;
pub const attnInit = cpu.attnInit;
pub const axpy = cpu.axpy;
pub const beginFrame = cpu.beginFrame;
pub const calibrate = cpu.calibrate;
pub const calibrateAttn = cpu.calibrateAttn;
pub const dequantRow = cpu.dequantRow;
pub const disableAttn = cpu.disableAttn;
pub const dotF32 = cpu.dotF32;
pub const enableKvMirror = cpu.enableKvMirror;
pub const endFrame = cpu.endFrame;
pub const ffnBlock = cpu.ffnBlock;
pub const frameLoadX = cpu.frameLoadX;
pub const frameOpen = cpu.frameOpen;
pub const frameStoreX = cpu.frameStoreX;
pub const hasKvCache = cpu.hasKvCache;
pub const hasMlaCache = cpu.hasMlaCache;
pub const kvAppend = cpu.kvAppend;
pub const lastVerdict = cpu.lastVerdict;
pub const layerBlock = cpu.layerBlock;
pub const materializeArenas = cpu.materializeArenas;
pub const matmul = cpu.matmul;
pub const mlaAppend = cpu.mlaAppend;
pub const mlaAttnHeads = cpu.mlaAttnHeads;
pub const mlaInit = cpu.mlaInit;
pub const mlaLayerTail = cpu.mlaLayerTail;
pub const mlaReadCache = cpu.mlaReadCache;
pub const mlaSetWk = cpu.mlaSetWk;
pub const mlaTokenFrame = cpu.mlaTokenFrame;
pub const moeFfnBlock = cpu.moeFfnBlock;
pub const moeFfnBlockRouted = cpu.moeFfnBlockRouted;
pub const moeRoute = cpu.moeRoute;
pub const registerArena = cpu.registerArena;
pub const releaseKvCache = cpu.releaseKvCache;
pub const rmsnorm = cpu.rmsnorm;
pub const sigmoid = cpu.sigmoid;
pub const softmax = cpu.softmax;
pub const swiglu = cpu.swiglu;

pub var use_gpu_ops: bool = false;
pub var arena_error: ?[]const u8 = null;
pub const useGpuOps = &use_gpu_ops;

var dev: ?vk.Device = null;
const Pipes = struct { q4_k: u64 = 0, q5_0: u64 = 0, q8_0: u64 = 0, q6_k: u64 = 0, f32: u64 = 0 };
var pipes = Pipes{};
var xbuf: ?vk.Buffer = null;
var obuf: ?vk.Buffer = null;
var cmdbufs: usize = 0;

// Weights uploaded once, keyed by host pointer -- the replacement for the
// bring-up's re-upload-per-call. A length mismatch on a known pointer means
// the allocation was reused and triggers a fresh upload; the old device
// buffer is leaked, which is acceptable because the model path uploads each
// immutable mmap'd tensor exactly once. Same-pointer-same-length reuse with
// *different* contents would be served stale -- impossible for mmap'd
// weights, and the oracle test keeps its per-type buffers distinct.
const CachedWeight = struct { buf: vk.Buffer, len: usize };
var wcache: std.AutoHashMapUnmanaged(usize, CachedWeight) = .empty;

pub fn takeCmdBufCount() usize {
    const n = cmdbufs;
    cmdbufs = 0;
    return n;
}

pub fn parallelBegin(n: usize) void {
    cpu.parallelBegin(n);
    if (dev != null) return;
    dev = vk.Device.init() catch return;
    // All pipelines or none: a partial set would make one quantization
    // silently slower or, worse, mask a broken kernel behind its fallback.
    const d = &dev.?;
    pipes = .{
        .q4_k = d.pipeline(&dmmv_q4k_spv) catch return fail(),
        .q5_0 = d.pipeline(&dmmv_q5_0_spv) catch return fail(),
        .q8_0 = d.pipeline(&dmmv_q8_0_spv) catch return fail(),
        .q6_k = d.pipeline(&dmmv_q6_k_spv) catch return fail(),
        .f32 = d.pipeline(&dmmv_f32_spv) catch return fail(),
    };
}

fn fail() void {
    dev = null;
    pipes = .{};
}

pub const parallelEnd = cpu.parallelEnd;

fn pipeFor(t: ggml.Type, cols: usize) ?u64 {
    return switch (t) {
        .q4_k => if (cols % 256 == 0) pipes.q4_k else null,
        .q6_k => if (cols % 256 == 0) pipes.q6_k else null,
        .q5_0 => if (cols % 32 == 0) pipes.q5_0 else null,
        .q8_0 => if (cols % 32 == 0) pipes.q8_0 else null,
        .f32 => pipes.f32,
        else => null,
    };
}

fn weightBuffer(d: *vk.Device, data: []const u8) ?vk.Buffer {
    const gop = wcache.getOrPut(std.heap.page_allocator, @intFromPtr(data.ptr)) catch return null;
    if (gop.found_existing and gop.value_ptr.len == data.len) return gop.value_ptr.buf;
    const buf = d.alloc(data.len) catch {
        if (!gop.found_existing) _ = wcache.remove(@intFromPtr(data.ptr));
        return null;
    };
    @memcpy(buf.slice(u8)[0..data.len], data);
    gop.value_ptr.* = .{ .buf = buf, .len = data.len };
    return buf;
}

/// The dmmv family (q4_k, q5_0, q8_0, q6_k, f32), one submission per call.
/// Weights live on the device after their first use; activations stage
/// through grow-only host-visible buffers. Anything without a kernel falls
/// to the CPU exactly as `cpu.matvec` would.
pub fn matvec(t: ggml.Type, out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    if (!use_gpu_ops) return cpu.matvec(t, out, data, x, rows, cols);
    const d = &(dev orelse return cpu.matvec(t, out, data, x, rows, cols));
    const pipe = pipeFor(t, cols) orelse return cpu.matvec(t, out, data, x, rows, cols);
    const wb = weightBuffer(d, data) orelse return cpu.matvec(t, out, data, x, rows, cols);
    if (xbuf == null or xbuf.?.len < x.len * 4) xbuf = d.alloc(x.len * 4) catch return cpu.matvec(t, out, data, x, rows, cols);
    if (obuf == null or obuf.?.len < out.len * 4) obuf = d.alloc(out.len * 4) catch return cpu.matvec(t, out, data, x, rows, cols);
    @memcpy(xbuf.?.slice(f32)[0..x.len], x);
    const push = extern struct { rows: u32, cols: u32 }{ .rows = @intCast(rows), .cols = @intCast(cols) };
    d.dispatchWait(pipe, &.{ wb, xbuf.?, obuf.? }, std.mem.asBytes(&push), @intCast(rows)) catch
        return cpu.matvec(t, out, data, x, rows, cols);
    cmdbufs += 1;
    @memcpy(out, obuf.?.slice(f32)[0..out.len]);
}

test "vulkan dmmv family agrees with the exact cpu reference" {
    // The bring-up oracle, identical in intent to the Metal ones: pinned
    // block scales so f32 summation is conditioned, random quants, an f64
    // dequantize-then-dot reference, and the submission counter as proof the
    // device path actually ran -- a silent fallback to `cpu.matvec` is
    // int8-approximate and presents as a slightly wrong kernel.
    const gpa = std.testing.allocator;
    const cols = 2048;
    const rows = 96;

    parallelBegin(1);
    defer parallelEnd();
    if (dev == null) return error.SkipZigTest;
    const saved = use_gpu_ops;
    use_gpu_ops = true;
    defer use_gpu_ops = saved;

    var prng = std.Random.DefaultPrng.init(0x0741C);
    const rnd = prng.random();
    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (x) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    // Per type: block size and the byte offsets of the half-precision scales
    // to pin. f32 has no blocks and is filled with real floats instead.
    const Case = struct { t: ggml.Type, block: usize, d_off: usize, dmin_off: ?usize };
    const cases = [_]Case{
        .{ .t = .q4_k, .block = 144, .d_off = 0, .dmin_off = 2 },
        .{ .t = .q5_0, .block = 22, .d_off = 0, .dmin_off = null },
        .{ .t = .q8_0, .block = 34, .d_off = 0, .dmin_off = null },
        .{ .t = .q6_k, .block = 210, .d_off = 208, .dmin_off = null },
        .{ .t = .f32, .block = 0, .d_off = 0, .dmin_off = null },
    };

    // Weight buffers stay allocated across the loop so the pointer-keyed
    // device cache never sees a recycled address with fresh contents.
    var datas: [cases.len][]u8 = undefined;
    var n_alloc: usize = 0;
    defer for (datas[0..n_alloc]) |dd| gpa.free(dd);

    for (cases) |case| {
        const data = try gpa.alloc(u8, ggml.tensorBytes(case.t, cols, rows));
        datas[n_alloc] = data;
        n_alloc += 1;
        if (case.t == .f32) {
            const wf = std.mem.bytesAsSlice(f32, data);
            for (wf) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
        } else {
            rnd.bytes(data);
            var b: usize = 0;
            while (b + case.block <= data.len) : (b += case.block) {
                std.mem.writeInt(u16, data[b + case.d_off ..][0..2], 0x2C00, .little);
                if (case.dmin_off) |o| std.mem.writeInt(u16, data[b + o ..][0..2], 0x2800, .little);
            }
        }

        const got = try gpa.alloc(f32, rows);
        defer gpa.free(got);
        @memset(got, std.math.nan(f32));
        const before = cmdbufs;
        matvec(case.t, got, data, x, rows, cols);
        try std.testing.expectEqual(before + 1, cmdbufs); // device path ran, no silent fallback

        const row = try gpa.alloc(f32, cols);
        defer gpa.free(row);
        for (0..rows) |r| {
            if (case.t == .f32) {
                const wf = std.mem.bytesAsSlice(f32, data);
                for (row, wf[r * cols ..][0..cols]) |*dst, src| dst.* = src;
            } else {
                cpu.dequantRow(case.t, row, data, r, cols);
            }
            var acc64: f64 = 0;
            var mass64: f64 = 0;
            for (row, x) |wv, xv| {
                acc64 += @as(f64, wv) * @as(f64, xv);
                mass64 += @abs(@as(f64, wv) * @as(f64, xv));
            }
            const want: f32 = @floatCast(acc64);
            const tol: f32 = @floatCast(mass64 * 1e-5);
            std.testing.expectApproxEqAbs(want, got[r], tol) catch |e| {
                std.debug.print("vulkan {s} row {d}: reference {d} vs gpu {d}\n", .{ @tagName(case.t), r, want, got[r] });
                return e;
            };
        }
    }
}
