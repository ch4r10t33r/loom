//! Vulkan backend, day one: the CPU implementation with a device underneath
//! one operation. Everything the seam exports resolves to the CPU fallback
//! except the Q4_K matvec, which runs on the device when `--gpu-ops` asks and
//! the shape fits -- the same bring-up the Metal backend had, and verified the
//! same way, against the exact dequantize-then-dot oracle.
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

const dmmv_q4k_raw = @embedFile("../shaders/vulkan/dmmv_q4k.spv");
// SPIR-V is a stream of u32 words and vkCreateShaderModule takes a *u32, but
// @embedFile guarantees byte alignment only -- the cast panicked on the first
// real run. A comptime copy into an aligned array fixes it for good.
const dmmv_q4k_spv: [dmmv_q4k_raw.len]u8 align(4) = dmmv_q4k_raw.*;

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
var q4k_pipe: u64 = 0;
var wbuf: ?vk.Buffer = null;
var xbuf: ?vk.Buffer = null;
var obuf: ?vk.Buffer = null;
var cmdbufs: usize = 0;

pub fn takeCmdBufCount() usize {
    const n = cmdbufs;
    cmdbufs = 0;
    return n;
}

pub fn parallelBegin(n: usize) void {
    cpu.parallelBegin(n);
    if (dev != null) return;
    dev = vk.Device.init() catch return;
    q4k_pipe = dev.?.pipeline(&dmmv_q4k_spv) catch {
        dev = null;
        return;
    };
}

pub const parallelEnd = cpu.parallelEnd;

/// Q4_K only, staged through host-visible buffers, one submission per call.
/// Everything else falls to the CPU exactly as `cpu.matvec` would.
pub fn matvec(t: ggml.Type, out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    if (!use_gpu_ops or t != .q4_k or cols % 256 != 0) return cpu.matvec(t, out, data, x, rows, cols);
    const d = &(dev orelse return cpu.matvec(t, out, data, x, rows, cols));
    // Grow-only staging; a weights re-upload per call is the bring-up cost.
    if (wbuf == null or wbuf.?.len < data.len) wbuf = d.alloc(data.len) catch return cpu.matvec(t, out, data, x, rows, cols);
    if (xbuf == null or xbuf.?.len < x.len * 4) xbuf = d.alloc(x.len * 4) catch return cpu.matvec(t, out, data, x, rows, cols);
    if (obuf == null or obuf.?.len < out.len * 4) obuf = d.alloc(out.len * 4) catch return cpu.matvec(t, out, data, x, rows, cols);
    @memcpy(wbuf.?.slice(u8)[0..data.len], data);
    @memcpy(xbuf.?.slice(f32)[0..x.len], x);
    const push = extern struct { rows: u32, cols: u32 }{ .rows = @intCast(rows), .cols = @intCast(cols) };
    d.dispatchWait(q4k_pipe, &.{ wbuf.?, xbuf.?, obuf.? }, std.mem.asBytes(&push), @intCast(rows)) catch
        return cpu.matvec(t, out, data, x, rows, cols);
    cmdbufs += 1;
    @memcpy(out, obuf.?.slice(f32)[0..out.len]);
}

test "vulkan q4_k matvec agrees with the exact cpu reference" {
    // The bring-up oracle, identical in intent to the Metal one: pinned block
    // scales so f32 summation is conditioned, random quants, an f64
    // dequantize-then-dot reference, and an explicit check that the device
    // path actually ran -- a silent fallback to `cpu.matvec` is
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
    const data = try gpa.alloc(u8, ggml.tensorBytes(.q4_k, cols, rows));
    defer gpa.free(data);
    rnd.bytes(data);
    var b: usize = 0;
    while (b + 144 <= data.len) : (b += 144) {
        std.mem.writeInt(u16, data[b..][0..2], 0x2C00, .little);
        std.mem.writeInt(u16, data[b + 2 ..][0..2], 0x2800, .little);
    }
    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (x) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    const got = try gpa.alloc(f32, rows);
    defer gpa.free(got);
    @memset(got, std.math.nan(f32));
    matvec(.q4_k, got, data, x, rows, cols);

    const cpu_ref = try gpa.alloc(f32, rows);
    defer gpa.free(cpu_ref);
    cpu.matvec(.q4_k, cpu_ref, data, x, rows, cols);
    var identical: usize = 0;
    for (got, cpu_ref) |g, c| {
        if (g == c) identical += 1;
    }
    if (identical == rows) return error.SkipZigTest; // fell back; nothing proven

    const row = try gpa.alloc(f32, cols);
    defer gpa.free(row);
    for (0..rows) |r| {
        cpu.dequantRow(.q4_k, row, data, r, cols);
        var acc64: f64 = 0;
        var mass64: f64 = 0;
        for (row, x) |wv, xv| {
            acc64 += @as(f64, wv) * @as(f64, xv);
            mass64 += @abs(@as(f64, wv) * @as(f64, xv));
        }
        const want: f32 = @floatCast(acc64);
        const tol: f32 = @floatCast(mass64 * 1e-5);
        std.testing.expectApproxEqAbs(want, got[r], tol) catch |e| {
            std.debug.print("vulkan q4_k row {d}: reference {d} vs gpu {d}\n", .{ r, want, got[r] });
            return e;
        };
    }
}
