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
const dmmv_q4k_id_raw = @embedFile("../shaders/vulkan/dmmv_q4k_id.spv");
const dmmv_q4k_id_spv: [dmmv_q4k_id_raw.len]u8 align(4) = dmmv_q4k_id_raw.*;
const dmmv_q5_0_id_raw = @embedFile("../shaders/vulkan/dmmv_q5_0_id.spv");
const dmmv_q5_0_id_spv: [dmmv_q5_0_id_raw.len]u8 align(4) = dmmv_q5_0_id_raw.*;
const dmmv_q8_0_id_raw = @embedFile("../shaders/vulkan/dmmv_q8_0_id.spv");
const dmmv_q8_0_id_spv: [dmmv_q8_0_id_raw.len]u8 align(4) = dmmv_q8_0_id_raw.*;
const moe_route_raw = @embedFile("../shaders/vulkan/moe_route.spv");
const moe_route_spv: [moe_route_raw.len]u8 align(4) = moe_route_raw.*;
const swiglu_slots_raw = @embedFile("../shaders/vulkan/swiglu_slots.spv");
const swiglu_slots_spv: [swiglu_slots_raw.len]u8 align(4) = swiglu_slots_raw.*;
const moe_reduce_dev_raw = @embedFile("../shaders/vulkan/moe_reduce_dev.spv");
const moe_reduce_dev_spv: [moe_reduce_dev_raw.len]u8 align(4) = moe_reduce_dev_raw.*;

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
const Pipes = struct {
    q4_k: u64 = 0,
    q5_0: u64 = 0,
    q8_0: u64 = 0,
    q6_k: u64 = 0,
    f32: u64 = 0,
    q4_k_id: u64 = 0,
    q5_0_id: u64 = 0,
    q8_0_id: u64 = 0,
    route: u64 = 0,
    swiglu_slots: u64 = 0,
    reduce_dev: u64 = 0,
};
var pipes = Pipes{};
var xbuf: ?vk.Buffer = null;
var obuf: ?vk.Buffer = null;
var cmdbufs: usize = 0;

// Routed-MoE working set: ids/gates written by the route kernel and read by
// the id kernels without a host round trip; slot buffers for every selected
// expert's gate/up/down activations; the reduce accumulator. Grow-only, like
// the matvec staging.
const MAX_MOE_EXPERTS = 16;
var route_in: ?vk.Buffer = null; // [logits][bias], packed
var ids_buf: ?vk.Buffer = null;
var gates_buf: ?vk.Buffer = null;
var slots: [3]?vk.Buffer = .{ null, null, null };
var accbuf: ?vk.Buffer = null;

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
        .q4_k_id = d.pipeline(&dmmv_q4k_id_spv) catch return fail(),
        .q5_0_id = d.pipeline(&dmmv_q5_0_id_spv) catch return fail(),
        .q8_0_id = d.pipeline(&dmmv_q8_0_id_spv) catch return fail(),
        .route = d.pipeline(&moe_route_spv) catch return fail(),
        .swiglu_slots = d.pipeline(&swiglu_slots_spv) catch return fail(),
        .reduce_dev = d.pipeline(&moe_reduce_dev_spv) catch return fail(),
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

fn idPipeFor(t: ggml.Type, cols: usize) ?u64 {
    return switch (t) {
        .q4_k => if (cols % 256 == 0) pipes.q4_k_id else null,
        .q5_0 => if (cols % 32 == 0) pipes.q5_0_id else null,
        .q8_0 => if (cols % 32 == 0) pipes.q8_0_id else null,
        else => null,
    };
}

fn ensure(buf: *?vk.Buffer, d: *vk.Device, len: usize) ?vk.Buffer {
    if (buf.* == null or buf.*.?.len < len) buf.* = d.alloc(len) catch return null;
    return buf.*;
}

const RoutePush = extern struct { n_expert: u32, n_used: u32, gating: u32, weights_norm: u32, has_bias: u32, weights_scale: f32 };
const IdPush = extern struct { rows: u32, cols: u32, n_used: u32, plane_stride: u32, x_stride: u32 };

/// Stage logits (and bias, packed after them) and run the route kernel; ids
/// and gates stay in their device buffers for the id kernels to read.
fn routeDispatch(d: *vk.Device, logits: []const f32, bias: ?[]const f32, n_used: usize, gating_sigmoid: bool, weights_norm: bool, weights_scale: f32) bool {
    const n_expert = logits.len;
    if (n_expert > 256 or n_used == 0 or n_used > MAX_MOE_EXPERTS) return false;
    const rin = ensure(&route_in, d, n_expert * 2 * 4) orelse return false;
    const idb = ensure(&ids_buf, d, MAX_MOE_EXPERTS * 4) orelse return false;
    const gb = ensure(&gates_buf, d, MAX_MOE_EXPERTS * 4) orelse return false;
    @memcpy(rin.slice(f32)[0..n_expert], logits);
    if (bias) |b| @memcpy(rin.slice(f32)[n_expert..][0..n_expert], b);
    const push = RoutePush{
        .n_expert = @intCast(n_expert),
        .n_used = @intCast(n_used),
        .gating = if (gating_sigmoid) 1 else 0,
        .weights_norm = if (weights_norm) 1 else 0,
        .has_bias = if (bias != null) 1 else 0,
        .weights_scale = weights_scale,
    };
    d.dispatchWait(pipes.route, &.{ rin, idb, gb }, std.mem.asBytes(&push), 1) catch return false;
    cmdbufs += 1;
    return true;
}

pub fn moeRoute(logits: []const f32, bias: ?[]const f32, ids_out: []u32, gates_out: []f32, gating_sigmoid: bool, weights_norm: bool, weights_scale: f32) bool {
    if (!use_gpu_ops) return false;
    const d = &(dev orelse return false);
    if (ids_out.len != gates_out.len) return false;
    if (!routeDispatch(d, logits, bias, ids_out.len, gating_sigmoid, weights_norm, weights_scale)) return false;
    @memcpy(ids_out, ids_buf.?.slice(u32)[0..ids_out.len]);
    @memcpy(gates_out, gates_buf.?.slice(f32)[0..gates_out.len]);
    return true;
}

/// A whole routed MoE layer on the device: route, every selected expert's
/// gate/up/SwiGLU/down, the gated reduce, and the shared expert. The same
/// assembly as the Metal `moeFfnBlockRouted`, with one bring-up difference:
/// each step is its own submit-and-wait rather than one recorded command
/// buffer, because `dispatchWait` is all the device layer has. What matters
/// on llvmpipe is that the ids and gates never return to the host between
/// route and reduce -- the kernels are what a real GPU will run; the
/// submission shape is not.
pub fn moeFfnBlockRouted(normed: []const f32, logits: []const f32, bias: ?[]const f32, gate_w: WeightRef, up_w: WeightRef, down_w: WeightRef, shexp: ?[3]WeightRef, ffn: usize, shexp_ffn: usize, cfg: RoutedCfg, out: []f32) bool {
    if (!use_gpu_ops) return false;
    const d = &(dev orelse return false);
    const dim = normed.len;
    if (out.len != dim or cfg.n_expert != logits.len) return false;
    if (cfg.n_used == 0 or cfg.n_used > MAX_MOE_EXPERTS) return false;

    // Resolve everything before dispatching anything: a decline halfway
    // through would leave stale slot buffers behind a `true`.
    const gp = idPipeFor(gate_w.ty, dim) orelse return false;
    const up = idPipeFor(up_w.ty, dim) orelse return false;
    const dp = idPipeFor(down_w.ty, ffn) orelse return false;
    var sh_pipes: [3]u64 = undefined;
    if (shexp) |sw| {
        sh_pipes[0] = pipeFor(sw[0].ty, dim) orelse return false;
        sh_pipes[1] = pipeFor(sw[1].ty, dim) orelse return false;
        sh_pipes[2] = pipeFor(sw[2].ty, shexp_ffn) orelse return false;
    }
    const gw = weightBuffer(d, gate_w.data) orelse return false;
    const uw = weightBuffer(d, up_w.data) orelse return false;
    const dw = weightBuffer(d, down_w.data) orelse return false;
    const widest = @max(ffn, shexp_ffn);
    const s1 = ensure(&slots[0], d, @max(cfg.n_used * ffn, widest) * 4) orelse return false;
    const s2 = ensure(&slots[1], d, @max(cfg.n_used * ffn, widest) * 4) orelse return false;
    const s3 = ensure(&slots[2], d, @max(cfg.n_used, 1) * dim * 4) orelse return false;
    const acc = ensure(&accbuf, d, dim * 4) orelse return false;
    const xb = ensure(&xbuf, d, dim * 4) orelse return false;
    @memcpy(xb.slice(f32)[0..dim], normed);

    if (!routeDispatch(d, logits, bias, cfg.n_used, cfg.gating_sigmoid, cfg.weights_norm, cfg.weights_scale)) return false;

    const d_gate = IdPush{ .rows = @intCast(ffn), .cols = @intCast(dim), .n_used = @intCast(cfg.n_used), .plane_stride = @intCast(gate_w.data.len / cfg.n_expert), .x_stride = 0 };
    const d_up = IdPush{ .rows = @intCast(ffn), .cols = @intCast(dim), .n_used = @intCast(cfg.n_used), .plane_stride = @intCast(up_w.data.len / cfg.n_expert), .x_stride = 0 };
    const d_down = IdPush{ .rows = @intCast(dim), .cols = @intCast(ffn), .n_used = @intCast(cfg.n_used), .plane_stride = @intCast(down_w.data.len / cfg.n_expert), .x_stride = @intCast(ffn) };
    const n_sw = extern struct { n: u32 }{ .n = @intCast(cfg.n_used * ffn) };
    const n_red = extern struct { n: u32, dim: u32 }{ .n = @intCast(cfg.n_used), .dim = @intCast(dim) };

    d.dispatchWait(gp, &.{ gw, xb, s1, ids_buf.? }, std.mem.asBytes(&d_gate), @intCast(cfg.n_used * ffn)) catch return false;
    d.dispatchWait(up, &.{ uw, xb, s2, ids_buf.? }, std.mem.asBytes(&d_up), @intCast(cfg.n_used * ffn)) catch return false;
    d.dispatchWait(pipes.swiglu_slots, &.{ s1, s2 }, std.mem.asBytes(&n_sw), @intCast((cfg.n_used * ffn + 63) / 64)) catch return false;
    d.dispatchWait(dp, &.{ dw, s1, s3, ids_buf.? }, std.mem.asBytes(&d_down), @intCast(cfg.n_used * dim)) catch return false;
    d.dispatchWait(pipes.reduce_dev, &.{ acc, s3, gates_buf.? }, std.mem.asBytes(&n_red), @intCast((dim + 63) / 64)) catch return false;
    cmdbufs += 5;
    @memcpy(out, acc.slice(f32)[0..dim]);

    if (shexp) |sw| {
        const shw = [3]?vk.Buffer{ weightBuffer(d, sw[0].data), weightBuffer(d, sw[1].data), weightBuffer(d, sw[2].data) };
        for (shw) |w| if (w == null) return false;
        const d_sh = extern struct { rows: u32, cols: u32 }{ .rows = @intCast(shexp_ffn), .cols = @intCast(dim) };
        const d_shd = extern struct { rows: u32, cols: u32 }{ .rows = @intCast(dim), .cols = @intCast(shexp_ffn) };
        const n_shsw = extern struct { n: u32 }{ .n = @intCast(shexp_ffn) };
        d.dispatchWait(sh_pipes[0], &.{ shw[0].?, xb, s1 }, std.mem.asBytes(&d_sh), @intCast(shexp_ffn)) catch return false;
        d.dispatchWait(sh_pipes[1], &.{ shw[1].?, xb, s2 }, std.mem.asBytes(&d_sh), @intCast(shexp_ffn)) catch return false;
        d.dispatchWait(pipes.swiglu_slots, &.{ s1, s2 }, std.mem.asBytes(&n_shsw), @intCast((shexp_ffn + 63) / 64)) catch return false;
        d.dispatchWait(sh_pipes[2], &.{ shw[2].?, s1, s3 }, std.mem.asBytes(&d_shd), @intCast(dim)) catch return false;
        cmdbufs += 4;
        for (out, s3.slice(f32)[0..dim]) |*o, v| o.* += v;
    }
    return true;
}

test "vulkan moe_route agrees with the host route" {
    // The two silent-when-wrong details -- bias shifting selection but not
    // gates, and the f16-clamped renormalization -- are exactly what a
    // differential against the shipped host router catches.
    const moe = @import("../gguf/moe.zig");
    const gpa = std.testing.allocator;
    const n_expert = 64;
    const n_used = 6;

    parallelBegin(1);
    defer parallelEnd();
    if (dev == null) return error.SkipZigTest;
    const saved = use_gpu_ops;
    use_gpu_ops = true;
    defer use_gpu_ops = saved;

    var prng = std.Random.DefaultPrng.init(0x0740E);
    const rnd = prng.random();
    const logits = try gpa.alloc(f32, n_expert);
    defer gpa.free(logits);
    for (logits) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;
    const bias = try gpa.alloc(f32, n_expert);
    defer gpa.free(bias);
    for (bias) |*v| v.* = (rnd.float(f32) - 0.5) * 0.5;

    const Case = struct { sigmoid: bool, norm: bool, scale: f32, bias: bool };
    const cases = [_]Case{
        .{ .sigmoid = true, .norm = true, .scale = 1.0, .bias = true },
        .{ .sigmoid = true, .norm = true, .scale = 1.0, .bias = false },
        .{ .sigmoid = false, .norm = true, .scale = 1.0, .bias = false },
        .{ .sigmoid = false, .norm = false, .scale = 2.5, .bias = false },
    };
    for (cases) |case| {
        var ids: [n_used]u32 = undefined;
        var gates: [n_used]f32 = undefined;
        const before = cmdbufs;
        try std.testing.expect(moeRoute(logits, if (case.bias) bias else null, &ids, &gates, case.sigmoid, case.norm, case.scale));
        try std.testing.expectEqual(before + 1, cmdbufs);

        var sel: [n_used]moe.Selected = undefined;
        moe.route(.{
            .n_expert = n_expert,
            .n_used = n_used,
            .gating = if (case.sigmoid) .sigmoid else .softmax,
            .weights_norm = case.norm,
            .weights_scale = case.scale,
        }, logits, if (case.bias) bias else null, &sel);
        for (0..n_used) |k| {
            try std.testing.expectEqual(sel[k].expert, ids[k]);
            try std.testing.expectApproxEqAbs(sel[k].gate, gates[k], 1e-5);
        }
    }
}

test "vulkan routed moe block agrees with the exact reference" {
    // The rental-ready differential: route on the device, all three id
    // kernels (q4_k gate, q8_0 up, q5_0 down), device SwiGLU and gated
    // reduce, shared expert -- against an f64 dequantize-everything
    // reference that shares nothing with the kernels but the weights.
    const moe = @import("../gguf/moe.zig");
    const gpa = std.testing.allocator;
    const dim = 512;
    const ffn = 256;
    const shexp_ffn = 256;
    const n_expert = 8;
    const n_used = 3;

    parallelBegin(1);
    defer parallelEnd();
    if (dev == null) return error.SkipZigTest;
    const saved = use_gpu_ops;
    use_gpu_ops = true;
    defer use_gpu_ops = saved;

    var prng = std.Random.DefaultPrng.init(0x0740F);
    const rnd = prng.random();
    const pin = struct {
        fn f(data: []u8, block: usize, d_off: usize, dmin: bool) void {
            var b: usize = 0;
            while (b + block <= data.len) : (b += block) {
                std.mem.writeInt(u16, data[b + d_off ..][0..2], 0x2C00, .little);
                if (dmin) std.mem.writeInt(u16, data[b + 2 ..][0..2], 0x2800, .little);
            }
        }
    }.f;
    const gw = try gpa.alloc(u8, n_expert * ggml.tensorBytes(.q4_k, dim, ffn));
    defer gpa.free(gw);
    const uw = try gpa.alloc(u8, n_expert * ggml.tensorBytes(.q8_0, dim, ffn));
    defer gpa.free(uw);
    const dw = try gpa.alloc(u8, n_expert * ggml.tensorBytes(.q5_0, ffn, dim));
    defer gpa.free(dw);
    const sg = try gpa.alloc(u8, ggml.tensorBytes(.q4_k, dim, shexp_ffn));
    defer gpa.free(sg);
    const su = try gpa.alloc(u8, ggml.tensorBytes(.q4_k, dim, shexp_ffn));
    defer gpa.free(su);
    const sd = try gpa.alloc(u8, ggml.tensorBytes(.q4_k, shexp_ffn, dim));
    defer gpa.free(sd);
    for ([_][]u8{ gw, sg, su, sd }) |data| {
        rnd.bytes(data);
        pin(data, 144, 0, true);
    }
    rnd.bytes(uw);
    pin(uw, 34, 0, false);
    rnd.bytes(dw);
    pin(dw, 22, 0, false);

    const normed = try gpa.alloc(f32, dim);
    defer gpa.free(normed);
    for (normed) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    const logits = try gpa.alloc(f32, n_expert);
    defer gpa.free(logits);
    for (logits) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;
    const bias = try gpa.alloc(f32, n_expert);
    defer gpa.free(bias);
    for (bias) |*v| v.* = (rnd.float(f32) - 0.5) * 0.5;

    const cfg = RoutedCfg{ .n_expert = n_expert, .n_used = n_used, .gating_sigmoid = true, .weights_norm = true, .weights_scale = 1.0 };
    const got = try gpa.alloc(f32, dim);
    defer gpa.free(got);
    @memset(got, std.math.nan(f32));
    const before = cmdbufs;
    try std.testing.expect(moeFfnBlockRouted(
        normed,
        logits,
        bias,
        .{ .ty = .q4_k, .data = gw },
        .{ .ty = .q8_0, .data = uw },
        .{ .ty = .q5_0, .data = dw },
        .{ .{ .ty = .q4_k, .data = sg }, .{ .ty = .q4_k, .data = su }, .{ .ty = .q4_k, .data = sd } },
        ffn,
        shexp_ffn,
        cfg,
        got,
    ));
    try std.testing.expectEqual(before + 10, cmdbufs); // route + 5 routed + 4 shared

    // f64 reference: host route, then dequantize-everything expert math.
    var sel: [n_used]moe.Selected = undefined;
    moe.route(.{ .n_expert = n_expert, .n_used = n_used, .gating = .sigmoid, .weights_norm = true, .weights_scale = 1.0 }, logits, bias, &sel);
    const want = try gpa.alloc(f64, dim);
    defer gpa.free(want);
    @memset(want, 0);
    const row = try gpa.alloc(f32, dim);
    defer gpa.free(row);
    const h = try gpa.alloc(f64, @max(ffn, shexp_ffn));
    defer gpa.free(h);
    const expert = struct {
        fn silu(g: f64) f64 {
            return g / (1.0 + @exp(-g));
        }
        fn f(gate_pl: []const u8, gate_ty: ggml.Type, up_pl: []const u8, up_ty: ggml.Type, down_pl: []const u8, down_ty: ggml.Type, x_in: []const f32, width: usize, weight: f64, acc: []f64, row_buf: []f32, h_buf: []f64) void {
            const dim2 = x_in.len;
            for (0..width) |r| {
                cpu.dequantRow(gate_ty, row_buf[0..dim2], gate_pl, r, dim2);
                var g: f64 = 0;
                for (row_buf[0..dim2], x_in) |wv, xv| g += @as(f64, wv) * @as(f64, xv);
                cpu.dequantRow(up_ty, row_buf[0..dim2], up_pl, r, dim2);
                var u: f64 = 0;
                for (row_buf[0..dim2], x_in) |wv, xv| u += @as(f64, wv) * @as(f64, xv);
                h_buf[r] = silu(g) * u;
            }
            for (0..dim2) |r| {
                cpu.dequantRow(down_ty, row_buf[0..width], down_pl, r, width);
                var o: f64 = 0;
                for (row_buf[0..width], h_buf[0..width]) |wv, hv| o += @as(f64, wv) * hv;
                acc[r] += weight * o;
            }
        }
    };
    const gate_plane = gw.len / n_expert;
    const up_plane = uw.len / n_expert;
    const down_plane = dw.len / n_expert;
    for (sel) |s| {
        expert.f(gw[s.expert * gate_plane ..][0..gate_plane], .q4_k, uw[s.expert * up_plane ..][0..up_plane], .q8_0, dw[s.expert * down_plane ..][0..down_plane], .q5_0, normed, ffn, s.gate, want, row, h);
    }
    expert.f(sg, .q4_k, su, .q4_k, sd, .q4_k, normed, shexp_ffn, 1.0, want, row, h);

    var mass: f64 = 0;
    for (want) |v| mass += @abs(v);
    const tol: f32 = @floatCast((mass / dim) * 1e-3);
    for (want, got, 0..) |a, b, k| {
        std.testing.expectApproxEqAbs(@as(f32, @floatCast(a)), b, tol) catch |e| {
            std.debug.print("routed block dim {d}: reference {d} vs gpu {d}\n", .{ k, a, b });
            return e;
        };
    }
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
