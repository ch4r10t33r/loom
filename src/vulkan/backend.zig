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
const mla_absorb_raw = @embedFile("../shaders/vulkan/mla_absorb.spv");
const mla_absorb_spv: [mla_absorb_raw.len]u8 align(4) = mla_absorb_raw.*;
const mla_attn_raw = @embedFile("../shaders/vulkan/mla_attn.spv");
const mla_attn_spv: [mla_attn_raw.len]u8 align(4) = mla_attn_raw.*;
const mla_wsum_raw = @embedFile("../shaders/vulkan/mla_wsum.spv");
const mla_wsum_spv: [mla_wsum_raw.len]u8 align(4) = mla_wsum_raw.*;
const rmsnorm_raw = @embedFile("../shaders/vulkan/rmsnorm.spv");
const rmsnorm_spv: [rmsnorm_raw.len]u8 align(4) = rmsnorm_raw.*;
const add_vec_raw = @embedFile("../shaders/vulkan/add_vec.spv");
const add_vec_spv: [add_vec_raw.len]u8 align(4) = add_vec_raw.*;

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
pub const kvAppend = cpu.kvAppend;
pub const lastVerdict = cpu.lastVerdict;
pub const layerBlock = cpu.layerBlock;
pub const materializeArenas = cpu.materializeArenas;
pub const matmul = cpu.matmul;
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
    absorb: u64 = 0,
    attn: u64 = 0,
    wsum: u64 = 0,
    rmsnorm: u64 = 0,
    add: u64 = 0,
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
        .absorb = d.pipeline(&mla_absorb_spv) catch return fail(),
        .attn = d.pipeline(&mla_attn_spv) catch return fail(),
        .wsum = d.pipeline(&mla_wsum_spv) catch return fail(),
        .rmsnorm = d.pipeline(&rmsnorm_spv) catch return fail(),
        .add = d.pipeline(&add_vec_spv) catch return fail(),
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
    // Device-local: VRAM on a discrete GPU (staged through `upload`), plain
    // mapped memory on llvmpipe/UMA. Must not be mid-recording -- `upload`
    // submits its own transfer -- which is why callers resolve every weight
    // before recording anything.
    const buf = d.allocDevice(data.len) catch {
        if (!gop.found_existing) _ = wcache.remove(@intFromPtr(data.ptr));
        return null;
    };
    d.upload(buf, data) catch {
        if (!gop.found_existing) _ = wcache.remove(@intFromPtr(data.ptr));
        return null;
    };
    gop.value_ptr.* = .{ .buf = buf, .len = data.len };
    return buf;
}

// A standalone matvec submission costs on the order of a millisecond of
// submit-and-wait on a discrete GPU regardless of size, so below some weight
// size the CPU wins on latency even though the GPU wins on bandwidth.
// Measured on an RTX 3060 (DeepSeek-V2-Lite Q4_K_M, 32 tokens): every matvec
// on the device 1.5 tok/s; cutover at 2 MB / 8 MB / 32 MB all 4.3; nothing
// but the fused MoE chain 4.1; CPU-only 3.1. The curve is flat across the
// 2-32 MB span because the model has no matvec between 3.6 MB (attention
// projections, CPU-cheap) and the 172 MB lm_head (GPU-cheap), so 2 MB is the
// default and LOOM_VK_MIN_BYTES overrides it for re-measurement on other
// hardware.
const default_min_bytes: usize = 2_000_000;
var min_bytes: ?usize = null;
fn matvecMinBytes() usize {
    if (min_bytes == null) {
        min_bytes = default_min_bytes;
        if (std.c.getenv("LOOM_VK_MIN_BYTES")) |s| {
            min_bytes = std.fmt.parseInt(usize, std.mem.span(s), 10) catch default_min_bytes;
        }
    }
    return min_bytes.?;
}

/// The dmmv family (q4_k, q5_0, q8_0, q6_k, f32), one submission per call.
/// Weights live on the device after their first use; activations stage
/// through grow-only host-visible buffers. Anything without a kernel falls
/// to the CPU exactly as `cpu.matvec` would.
pub fn matvec(t: ggml.Type, out: []f32, data: []const u8, x: []const f32, rows: usize, cols: usize) void {
    if (!use_gpu_ops or data.len < matvecMinBytes()) return cpu.matvec(t, out, data, x, rows, cols);
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
const IdPush = extern struct { rows: u32, cols: u32, n_used: u32, plane_stride: u32, x_stride: u32, base: u32 = 0 };

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

    // Shared-expert weights resolve here too: `upload` submits its own
    // transfer, so every weight must be resident before recording opens.
    var shw: [3]vk.Buffer = undefined;
    if (shexp) |sw| {
        for (sw, 0..) |w, i| shw[i] = weightBuffer(d, w.data) orelse return false;
    }

    // Route staging, packed [logits][bias] -- host writes to mapped memory,
    // legal any time before submit.
    const n_expert = cfg.n_expert;
    const rin = ensure(&route_in, d, n_expert * 2 * 4) orelse return false;
    const idb = ensure(&ids_buf, d, MAX_MOE_EXPERTS * 4) orelse return false;
    const gb = ensure(&gates_buf, d, MAX_MOE_EXPERTS * 4) orelse return false;
    if (n_expert > 256) return false;
    @memcpy(rin.slice(f32)[0..n_expert], logits);
    if (bias) |b| @memcpy(rin.slice(f32)[n_expert..][0..n_expert], b);
    const rpush = RoutePush{
        .n_expert = @intCast(n_expert),
        .n_used = @intCast(cfg.n_used),
        .gating = if (cfg.gating_sigmoid) 1 else 0,
        .weights_norm = if (cfg.weights_norm) 1 else 0,
        .has_bias = if (bias != null) 1 else 0,
        .weights_scale = cfg.weights_scale,
    };

    const d_gate = IdPush{ .rows = @intCast(ffn), .cols = @intCast(dim), .n_used = @intCast(cfg.n_used), .plane_stride = @intCast(gate_w.data.len / cfg.n_expert), .x_stride = 0 };
    const d_up = IdPush{ .rows = @intCast(ffn), .cols = @intCast(dim), .n_used = @intCast(cfg.n_used), .plane_stride = @intCast(up_w.data.len / cfg.n_expert), .x_stride = 0 };
    const d_down = IdPush{ .rows = @intCast(dim), .cols = @intCast(ffn), .n_used = @intCast(cfg.n_used), .plane_stride = @intCast(down_w.data.len / cfg.n_expert), .x_stride = @intCast(ffn) };
    const n_sw = extern struct { n: u32 }{ .n = @intCast(cfg.n_used * ffn) };
    const n_red = extern struct { n: u32, dim: u32 }{ .n = @intCast(cfg.n_used), .dim = @intCast(dim) };

    // The whole chain as one submission, barriers where a stage reads what
    // the previous one wrote -- the shape Metal proved out, minus only the
    // cross-layer fusion.
    const c = d.beginCmd() catch return false;
    d.record(c, pipes.route, &.{ rin, idb, gb }, std.mem.asBytes(&rpush), 1) catch return false;
    d.barrier(c);
    d.record(c, gp, &.{ gw, xb, s1, idb }, std.mem.asBytes(&d_gate), @intCast(cfg.n_used * ffn)) catch return false;
    d.record(c, up, &.{ uw, xb, s2, idb }, std.mem.asBytes(&d_up), @intCast(cfg.n_used * ffn)) catch return false;
    d.barrier(c);
    d.record(c, pipes.swiglu_slots, &.{ s1, s2 }, std.mem.asBytes(&n_sw), @intCast((cfg.n_used * ffn + 63) / 64)) catch return false;
    d.barrier(c);
    d.record(c, dp, &.{ dw, s1, s3, idb }, std.mem.asBytes(&d_down), @intCast(cfg.n_used * dim)) catch return false;
    d.barrier(c);
    d.record(c, pipes.reduce_dev, &.{ acc, s3, gb }, std.mem.asBytes(&n_red), @intCast((dim + 63) / 64)) catch return false;
    if (shexp != null) {
        const d_sh = extern struct { rows: u32, cols: u32 }{ .rows = @intCast(shexp_ffn), .cols = @intCast(dim) };
        const d_shd = extern struct { rows: u32, cols: u32 }{ .rows = @intCast(dim), .cols = @intCast(shexp_ffn) };
        const n_shsw = extern struct { n: u32 }{ .n = @intCast(shexp_ffn) };
        d.barrier(c);
        d.record(c, sh_pipes[0], &.{ shw[0], xb, s1 }, std.mem.asBytes(&d_sh), @intCast(shexp_ffn)) catch return false;
        d.record(c, sh_pipes[1], &.{ shw[1], xb, s2 }, std.mem.asBytes(&d_sh), @intCast(shexp_ffn)) catch return false;
        d.barrier(c);
        d.record(c, pipes.swiglu_slots, &.{ s1, s2 }, std.mem.asBytes(&n_shsw), @intCast((shexp_ffn + 63) / 64)) catch return false;
        d.barrier(c);
        d.record(c, sh_pipes[2], &.{ shw[2], s1, s3 }, std.mem.asBytes(&d_shd), @intCast(dim)) catch return false;
    }
    d.submitWait(c) catch return false;
    cmdbufs += 1;

    @memcpy(out, acc.slice(f32)[0..dim]);
    if (shexp != null) {
        for (out, s3.slice(f32)[0..dim]) |*o, v| o.* += v;
    }
    return true;
}

// ---- MLA attention on the device ------------------------------------------
//
// The port of the Metal backend's compressed-cache attention: absorb
// (q_abs = W_k^T q_nope), scores+softmax, weighted sum in compressed space,
// and W_v as an id-kernel dispatch with identity ids -- one submission for
// the lot, and `mlaLayerTail` extends that same recording through the output
// projection, residual, FFN norm, router and the routed MoE chain, so a
// whole MoE layer's tail is one submit-and-wait.
//
// The c and k_rope caches share ONE buffer (c rows first, k_rope rows after
// at `mla_r0`) because the attention kernel's four bindings are all taken;
// per-layer rows are addressed by push-constant element offsets. The cache
// is host-visible: the engine's host cache stays authoritative and mlaAppend
// mirrors rows in with a memcpy, costing no submission.

const MLA_MAX_SEQ = 4096; // the attention kernel's shared-memory bound

const MlaWant = struct { layers: usize, ctx: usize, kvr: usize, rope: usize };
var mla_want: ?MlaWant = null;
var mla_cache: ?vk.Buffer = null;
var mla_r0: usize = 0; // element offset of the k_rope region
var mla_wk: std.ArrayListUnmanaged(vk.Buffer) = .empty; // per layer: full dequant kv_b, f32, device-local
var vids: ?vk.Buffer = null; // identity ids for the W_v id dispatch

// Attention working set, grow-only host-visible like the rest.
var qbuf: ?vk.Buffer = null; // gathered [n_heads][nope] then [n_heads][rope]
var qabs: ?vk.Buffer = null;
var probs_buf: ?vk.Buffer = null;
var olat: ?vk.Buffer = null;
var hout: ?vk.Buffer = null;
var projb: ?vk.Buffer = null;
var xres: ?vk.Buffer = null;
var normw: ?vk.Buffer = null;
var normed_buf: ?vk.Buffer = null;
var lbuf: ?vk.Buffer = null; // router logits (device-written) + bias (host-written)

pub fn mlaInit(layers: usize, ctx_len: usize, kvr: usize, rope: usize) bool {
    // Records the shape; buffers are made on first use. Declining a context
    // the attention kernel cannot hold beats truncating one.
    if (ctx_len > MLA_MAX_SEQ) {
        mla_want = null;
        return false;
    }
    mla_want = .{ .layers = layers, .ctx = ctx_len, .kvr = kvr, .rope = rope };
    return true;
}

fn ensureMlaCache(d: *vk.Device) ?MlaWant {
    const w = mla_want orelse return null;
    if (mla_cache == null) {
        const c_elems = w.layers * w.ctx * w.kvr;
        const r_elems = w.layers * w.ctx * w.rope;
        mla_cache = d.alloc((c_elems + r_elems) * 4) catch return null;
        mla_r0 = c_elems;
    }
    return w;
}

pub fn hasMlaCache() bool {
    const d = &(dev orelse return false);
    return ensureMlaCache(d) != null;
}

/// Mirror one position's compressed row into the device cache. The engine's
/// host cache stays authoritative: this can decline at any point and the
/// host fallback must still be correct.
pub fn mlaAppend(li: usize, pos: usize, c_kv: []const f32, k_rope: []const f32) bool {
    const d = &(dev orelse return false);
    const w = ensureMlaCache(d) orelse return false;
    if (pos >= w.ctx or c_kv.len != w.kvr or k_rope.len != w.rope) return false;
    const cache = mla_cache.?;
    @memcpy(cache.slice(f32)[(li * w.ctx + pos) * w.kvr ..][0..w.kvr], c_kv);
    @memcpy(cache.slice(f32)[mla_r0 + (li * w.ctx + pos) * w.rope ..][0..w.rope], k_rope);
    return true;
}

pub fn mlaReadCache(li: usize, pos: usize, c_kv: []f32, k_rope: []f32) bool {
    const d = &(dev orelse return false);
    const w = ensureMlaCache(d) orelse return false;
    if (pos >= w.ctx or c_kv.len != w.kvr or k_rope.len != w.rope) return false;
    const cache = mla_cache.?;
    @memcpy(c_kv, cache.slice(f32)[(li * w.ctx + pos) * w.kvr ..][0..w.kvr]);
    @memcpy(k_rope, cache.slice(f32)[mla_r0 + (li * w.ctx + pos) * w.rope ..][0..w.rope]);
    return true;
}

/// One layer's kv_b, dequantized to f32, onto the device -- the absorb
/// kernel strides past the W_v rows, so the layout matches the tensor.
/// Layers must arrive in order.
pub fn mlaSetWk(li: usize, wk_f32: []const f32) bool {
    const d = &(dev orelse return false);
    if (li != mla_wk.items.len) return false;
    const b = d.allocDevice(wk_f32.len * 4) catch return false;
    d.upload(b, std.mem.sliceAsBytes(wk_f32)) catch return false;
    mla_wk.append(std.heap.page_allocator, b) catch return false;
    return true;
}

const AbsorbPush = extern struct { n_heads: u32, nope: u32, kvr: u32, stride: u32 };
const AttnPush = extern struct { n_heads: u32, kvr: u32, rope: u32, seq: u32, qr_off: u32, c_base: u32, r_base: u32, scale: f32 };
const WsumPush = extern struct { n_heads: u32, kvr: u32, seq: u32, c_base: u32 };

/// Everything the attention recording needs resolved and staged, or null to
/// decline before anything is recorded.
const AttnPrep = struct {
    w: MlaWant,
    vp: u64,
    vw: vk.Buffer,
    row_bytes: usize,
    seq: usize,
};

fn prepAttn(d: *vk.Device, li: usize, pos: usize, q_nope: []const f32, q_rope: []const f32, kv_b: WeightRef, n_heads: usize, nope: usize, v_head_dim: usize) ?AttnPrep {
    if (!use_gpu_ops) return null;
    const w = ensureMlaCache(d) orelse return null;
    if (li >= mla_wk.items.len) return null;
    const seq = pos + 1;
    if (seq > w.ctx) return null;
    if (q_nope.len != n_heads * nope or q_rope.len != n_heads * w.rope) return null;
    const vp = idPipeFor(kv_b.ty, w.kvr) orelse return null;
    const vw = weightBuffer(d, kv_b.data) orelse return null;
    const row_bytes = kv_b.data.len / (n_heads * (nope + v_head_dim));
    if (vids == null) {
        const b = d.alloc(64 * 4) catch return null;
        for (0..64) |k| b.slice(u32)[k] = @intCast(k);
        vids = b;
    }
    _ = ensure(&qbuf, d, (q_nope.len + q_rope.len) * 4) orelse return null;
    _ = ensure(&qabs, d, n_heads * w.kvr * 4) orelse return null;
    _ = ensure(&probs_buf, d, n_heads * w.ctx * 4) orelse return null;
    _ = ensure(&olat, d, n_heads * w.kvr * 4) orelse return null;
    _ = ensure(&hout, d, n_heads * v_head_dim * 4) orelse return null;
    @memcpy(qbuf.?.slice(f32)[0..q_nope.len], q_nope);
    @memcpy(qbuf.?.slice(f32)[q_nope.len..][0..q_rope.len], q_rope);
    return .{ .w = w, .vp = vp, .vw = vw, .row_bytes = row_bytes, .seq = seq };
}

/// Record absorb -> attention -> weighted sum -> W_v; the result lands in
/// `hout` on the device.
fn recordAttn(d: *vk.Device, c: vk.Device.Cmd, p: AttnPrep, li: usize, n_heads: usize, nope: usize, v_head_dim: usize, scale: f32) vk.Error!void {
    const w = p.w;
    const apush = AbsorbPush{ .n_heads = @intCast(n_heads), .nope = @intCast(nope), .kvr = @intCast(w.kvr), .stride = @intCast(nope + v_head_dim) };
    try d.record(c, pipes.absorb, &.{ mla_wk.items[li], qbuf.?, qabs.? }, std.mem.asBytes(&apush), @intCast(n_heads * w.kvr));
    d.barrier(c);
    const tpush = AttnPush{
        .n_heads = @intCast(n_heads),
        .kvr = @intCast(w.kvr),
        .rope = @intCast(w.rope),
        .seq = @intCast(p.seq),
        .qr_off = @intCast(n_heads * nope),
        .c_base = @intCast(li * w.ctx * w.kvr),
        .r_base = @intCast(mla_r0 + li * w.ctx * w.rope),
        .scale = scale,
    };
    try d.record(c, pipes.attn, &.{ qabs.?, qbuf.?, mla_cache.?, probs_buf.? }, std.mem.asBytes(&tpush), @intCast(n_heads));
    d.barrier(c);
    const wpush = WsumPush{ .n_heads = @intCast(n_heads), .kvr = @intCast(w.kvr), .seq = @intCast(p.seq), .c_base = @intCast(li * w.ctx * w.kvr) };
    try d.record(c, pipes.wsum, &.{ probs_buf.?, mla_cache.?, olat.? }, std.mem.asBytes(&wpush), @intCast(n_heads * w.kvr));
    d.barrier(c);
    // W_v with identity ids: head h's value rows are a plane at stride
    // (nope + vd) rows, offset nope rows in, its input its own o_latent.
    const vpush = IdPush{
        .rows = @intCast(v_head_dim),
        .cols = @intCast(w.kvr),
        .n_used = @intCast(n_heads),
        .plane_stride = @intCast((nope + v_head_dim) * p.row_bytes),
        .x_stride = @intCast(w.kvr),
        .base = @intCast(nope * p.row_bytes),
    };
    try d.record(c, p.vp, &.{ p.vw, olat.?, hout.?, vids.? }, std.mem.asBytes(&vpush), @intCast(n_heads * v_head_dim));
}

/// Every head's MLA attention over the compressed cache, W_v applied -- one
/// submission. False declines to the engine's host loop.
pub fn mlaAttnHeads(li: usize, pos: usize, q_nope: []const f32, q_rope: []const f32, kv_b: WeightRef, out: []f32, n_heads: usize, nope: usize, v_head_dim: usize, scale: f32) bool {
    const d = &(dev orelse return false);
    if (out.len != n_heads * v_head_dim) return false;
    const p = prepAttn(d, li, pos, q_nope, q_rope, kv_b, n_heads, nope, v_head_dim) orelse return false;
    const c = d.beginCmd() catch return false;
    recordAttn(d, c, p, li, n_heads, nope, v_head_dim, scale) catch return false;
    d.submitWait(c) catch return false;
    cmdbufs += 1;
    @memcpy(out, hout.?.slice(f32)[0..out.len]);
    return true;
}

/// A whole MLA layer's tail as one submission: attention (absorb, scores,
/// weighted sum, W_v), output projection, residual, FFN norm, router, and
/// the routed MoE chain with shared expert. In: the residual stream and this
/// position's queries; out: the residual stream after the layer. The
/// attention head (projections, norms, rope, cache append) stays on the
/// host, exactly as Metal's tail began.
pub fn mlaLayerTail(li: usize, pos: usize, x: []f32, q_nope: []const f32, q_rope: []const f32, kv_b: WeightRef, attn_out_w: WeightRef, ffn_norm: []const f32, eps: f32, router_w: WeightRef, router_bias: ?[]const f32, gate_w: WeightRef, up_w: WeightRef, down_w: WeightRef, shexp: ?[3]WeightRef, ffn: usize, shexp_ffn: usize, cfg: RoutedCfg, n_heads: usize, nope: usize, v_head_dim: usize, scale: f32) bool {
    const d = &(dev orelse return false);
    const dim = x.len;
    if (ffn_norm.len != dim or cfg.n_expert > 256) return false;
    if (cfg.n_used == 0 or cfg.n_used > MAX_MOE_EXPERTS) return false;

    // Resolve and stage everything before recording anything: attention...
    const p = prepAttn(d, li, pos, q_nope, q_rope, kv_b, n_heads, nope, v_head_dim) orelse return false;
    // ...the projections around it...
    const ap = pipeFor(attn_out_w.ty, n_heads * v_head_dim) orelse return false;
    const aw = weightBuffer(d, attn_out_w.data) orelse return false;
    const rp = pipeFor(router_w.ty, dim) orelse return false;
    const rw = weightBuffer(d, router_w.data) orelse return false;
    // ...and the routed chain, same resolution as moeFfnBlockRouted.
    const gp = idPipeFor(gate_w.ty, dim) orelse return false;
    const up = idPipeFor(up_w.ty, dim) orelse return false;
    const dp = idPipeFor(down_w.ty, ffn) orelse return false;
    var sh_pipes: [3]u64 = undefined;
    var shw: [3]vk.Buffer = undefined;
    if (shexp) |sw| {
        sh_pipes[0] = pipeFor(sw[0].ty, dim) orelse return false;
        sh_pipes[1] = pipeFor(sw[1].ty, dim) orelse return false;
        sh_pipes[2] = pipeFor(sw[2].ty, shexp_ffn) orelse return false;
        for (sw, 0..) |w2, i| shw[i] = weightBuffer(d, w2.data) orelse return false;
    }
    const gw = weightBuffer(d, gate_w.data) orelse return false;
    const uw = weightBuffer(d, up_w.data) orelse return false;
    const dw = weightBuffer(d, down_w.data) orelse return false;
    const widest = @max(ffn, shexp_ffn);
    const s1 = ensure(&slots[0], d, @max(cfg.n_used * ffn, widest) * 4) orelse return false;
    const s2 = ensure(&slots[1], d, @max(cfg.n_used * ffn, widest) * 4) orelse return false;
    const s3 = ensure(&slots[2], d, @max(cfg.n_used, 1) * dim * 4) orelse return false;
    const acc = ensure(&accbuf, d, dim * 4) orelse return false;
    const idb = ensure(&ids_buf, d, MAX_MOE_EXPERTS * 4) orelse return false;
    const gb = ensure(&gates_buf, d, MAX_MOE_EXPERTS * 4) orelse return false;
    const xr = ensure(&xres, d, dim * 4) orelse return false;
    const pj = ensure(&projb, d, dim * 4) orelse return false;
    const nw = ensure(&normw, d, dim * 4) orelse return false;
    const nb = ensure(&normed_buf, d, dim * 4) orelse return false;
    const lb = ensure(&lbuf, d, cfg.n_expert * 2 * 4) orelse return false;
    @memcpy(xr.slice(f32)[0..dim], x);
    @memcpy(nw.slice(f32)[0..dim], ffn_norm);
    // The router kernel writes logits into [0..n_expert); the bias rides in
    // the same buffer at [n_expert..), host-written before submit.
    if (router_bias) |b| @memcpy(lb.slice(f32)[cfg.n_expert..][0..cfg.n_expert], b);

    const d_proj = extern struct { rows: u32, cols: u32 }{ .rows = @intCast(dim), .cols = @intCast(n_heads * v_head_dim) };
    const d_router = extern struct { rows: u32, cols: u32 }{ .rows = @intCast(cfg.n_expert), .cols = @intCast(dim) };
    const n_add = extern struct { n: u32 }{ .n = @intCast(dim) };
    const d_norm = extern struct { n: u32, eps: f32 }{ .n = @intCast(dim), .eps = eps };
    const rpush = RoutePush{
        .n_expert = @intCast(cfg.n_expert),
        .n_used = @intCast(cfg.n_used),
        .gating = if (cfg.gating_sigmoid) 1 else 0,
        .weights_norm = if (cfg.weights_norm) 1 else 0,
        .has_bias = if (router_bias != null) 1 else 0,
        .weights_scale = cfg.weights_scale,
    };
    const d_gate = IdPush{ .rows = @intCast(ffn), .cols = @intCast(dim), .n_used = @intCast(cfg.n_used), .plane_stride = @intCast(gate_w.data.len / cfg.n_expert), .x_stride = 0 };
    const d_up = IdPush{ .rows = @intCast(ffn), .cols = @intCast(dim), .n_used = @intCast(cfg.n_used), .plane_stride = @intCast(up_w.data.len / cfg.n_expert), .x_stride = 0 };
    const d_down = IdPush{ .rows = @intCast(dim), .cols = @intCast(ffn), .n_used = @intCast(cfg.n_used), .plane_stride = @intCast(down_w.data.len / cfg.n_expert), .x_stride = @intCast(ffn) };
    const n_sw = extern struct { n: u32 }{ .n = @intCast(cfg.n_used * ffn) };
    const n_red = extern struct { n: u32, dim: u32 }{ .n = @intCast(cfg.n_used), .dim = @intCast(dim) };

    const c = d.beginCmd() catch return false;
    recordAttn(d, c, p, li, n_heads, nope, v_head_dim, scale) catch return false;
    d.barrier(c);
    d.record(c, ap, &.{ aw, hout.?, pj }, std.mem.asBytes(&d_proj), @intCast(dim)) catch return false;
    d.barrier(c);
    d.record(c, pipes.add, &.{ xr, pj }, std.mem.asBytes(&n_add), @intCast((dim + 63) / 64)) catch return false;
    d.barrier(c);
    d.record(c, pipes.rmsnorm, &.{ xr, nw, nb }, std.mem.asBytes(&d_norm), 1) catch return false;
    d.barrier(c);
    d.record(c, rp, &.{ rw, nb, lb }, std.mem.asBytes(&d_router), @intCast(cfg.n_expert)) catch return false;
    d.barrier(c);
    // The routed chain, identical in shape to moeFfnBlockRouted's recording,
    // reading its activation from the device-resident `nb`.
    d.record(c, pipes.route, &.{ lb, idb, gb }, std.mem.asBytes(&rpush), 1) catch return false;
    d.barrier(c);
    d.record(c, gp, &.{ gw, nb, s1, idb }, std.mem.asBytes(&d_gate), @intCast(cfg.n_used * ffn)) catch return false;
    d.record(c, up, &.{ uw, nb, s2, idb }, std.mem.asBytes(&d_up), @intCast(cfg.n_used * ffn)) catch return false;
    d.barrier(c);
    d.record(c, pipes.swiglu_slots, &.{ s1, s2 }, std.mem.asBytes(&n_sw), @intCast((cfg.n_used * ffn + 63) / 64)) catch return false;
    d.barrier(c);
    d.record(c, dp, &.{ dw, s1, s3, idb }, std.mem.asBytes(&d_down), @intCast(cfg.n_used * dim)) catch return false;
    d.barrier(c);
    d.record(c, pipes.reduce_dev, &.{ acc, s3, gb }, std.mem.asBytes(&n_red), @intCast((dim + 63) / 64)) catch return false;
    d.barrier(c);
    d.record(c, pipes.add, &.{ xr, acc }, std.mem.asBytes(&n_add), @intCast((dim + 63) / 64)) catch return false;
    if (shexp != null) {
        const d_sh = extern struct { rows: u32, cols: u32 }{ .rows = @intCast(shexp_ffn), .cols = @intCast(dim) };
        const d_shd = extern struct { rows: u32, cols: u32 }{ .rows = @intCast(dim), .cols = @intCast(shexp_ffn) };
        const n_shsw = extern struct { n: u32 }{ .n = @intCast(shexp_ffn) };
        d.barrier(c);
        d.record(c, sh_pipes[0], &.{ shw[0], nb, s1 }, std.mem.asBytes(&d_sh), @intCast(shexp_ffn)) catch return false;
        d.record(c, sh_pipes[1], &.{ shw[1], nb, s2 }, std.mem.asBytes(&d_sh), @intCast(shexp_ffn)) catch return false;
        d.barrier(c);
        d.record(c, pipes.swiglu_slots, &.{ s1, s2 }, std.mem.asBytes(&n_shsw), @intCast((shexp_ffn + 63) / 64)) catch return false;
        d.barrier(c);
        d.record(c, sh_pipes[2], &.{ shw[2], s1, s3 }, std.mem.asBytes(&d_shd), @intCast(dim)) catch return false;
        d.barrier(c);
        d.record(c, pipes.add, &.{ xr, s3 }, std.mem.asBytes(&n_add), @intCast((dim + 63) / 64)) catch return false;
    }
    d.submitWait(c) catch return false;
    cmdbufs += 1;
    @memcpy(x, xr.slice(f32)[0..dim]);
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
    try std.testing.expectEqual(before + 1, cmdbufs); // the whole chain is one submission

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

test "vulkan mla attention agrees with the exact reference" {
    // The whole compressed-cache attention -- absorb, scores, softmax,
    // weighted sum, W_v -- against an f64 reference that shares nothing with
    // the kernels but the cache contents and the quantized kv_b. Layer 1 of
    // 2, with decoy rows in layer 0's cache region, so a wrong per-layer
    // base offset cannot pass.
    const gpa = std.testing.allocator;
    const n_heads = 4;
    const nope = 32;
    const vd = 16;
    const kvr = 64;
    const rope = 16;
    const ctx = 64;
    const seq = 5;
    const li = 1;
    const scale: f32 = 0.25;
    const stride = nope + vd;

    parallelBegin(1);
    defer parallelEnd();
    if (dev == null) return error.SkipZigTest;
    const saved = use_gpu_ops;
    use_gpu_ops = true;
    defer use_gpu_ops = saved;

    var prng = std.Random.DefaultPrng.init(0x1207A);
    const rnd = prng.random();
    const kvb = try gpa.alloc(u8, ggml.tensorBytes(.q8_0, kvr, n_heads * stride));
    defer gpa.free(kvb);
    rnd.bytes(kvb);
    var b: usize = 0;
    while (b + 34 <= kvb.len) : (b += 34) std.mem.writeInt(u16, kvb[b..][0..2], 0x2C00, .little);

    try std.testing.expect(mlaInit(2, ctx, kvr, rope));
    const wkbuf = try gpa.alloc(f32, n_heads * stride * kvr);
    defer gpa.free(wkbuf);
    for (0..n_heads * stride) |r| cpu.dequantRow(.q8_0, wkbuf[r * kvr ..][0..kvr], kvb, r, kvr);
    try std.testing.expect(mlaSetWk(0, wkbuf));
    try std.testing.expect(mlaSetWk(1, wkbuf));

    const c_rows = try gpa.alloc(f32, seq * kvr);
    defer gpa.free(c_rows);
    const r_rows = try gpa.alloc(f32, seq * rope);
    defer gpa.free(r_rows);
    for (c_rows) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    for (r_rows) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    var decoy_c: [kvr]f32 = undefined;
    var decoy_r: [rope]f32 = undefined;
    for (0..seq) |t| {
        for (&decoy_c) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
        for (&decoy_r) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
        try std.testing.expect(mlaAppend(0, t, &decoy_c, &decoy_r));
        try std.testing.expect(mlaAppend(li, t, c_rows[t * kvr ..][0..kvr], r_rows[t * rope ..][0..rope]));
    }

    const q_nope = try gpa.alloc(f32, n_heads * nope);
    defer gpa.free(q_nope);
    const q_rope = try gpa.alloc(f32, n_heads * rope);
    defer gpa.free(q_rope);
    for (q_nope) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    for (q_rope) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    const got = try gpa.alloc(f32, n_heads * vd);
    defer gpa.free(got);
    @memset(got, std.math.nan(f32));
    const before = cmdbufs;
    try std.testing.expect(mlaAttnHeads(li, seq - 1, q_nope, q_rope, .{ .ty = .q8_0, .data = kvb }, got, n_heads, nope, vd, scale));
    try std.testing.expectEqual(before + 1, cmdbufs);

    // f64 reference off the dequantized rows.
    const row = try gpa.alloc(f32, kvr);
    defer gpa.free(row);
    var mass: f64 = 0;
    var worst: f64 = 0;
    for (0..n_heads) |h| {
        var qa: [kvr]f64 = @splat(0);
        for (0..nope) |r| {
            cpu.dequantRow(.q8_0, row, kvb, h * stride + r, kvr);
            for (&qa, row) |*a, wv| a.* += @as(f64, q_nope[h * nope + r]) * wv;
        }
        var probs: [seq]f64 = undefined;
        var mx: f64 = -std.math.inf(f64);
        for (0..seq) |t| {
            var s: f64 = 0;
            for (qa, c_rows[t * kvr ..][0..kvr]) |a, cv| s += a * cv;
            for (0..rope) |i| s += @as(f64, q_rope[h * rope + i]) * r_rows[t * rope + i];
            probs[t] = s * scale;
            mx = @max(mx, probs[t]);
        }
        var sum: f64 = 0;
        for (&probs) |*pv| {
            pv.* = @exp(pv.* - mx);
            sum += pv.*;
        }
        var ol: [kvr]f64 = @splat(0);
        for (0..seq) |t| {
            for (&ol, c_rows[t * kvr ..][0..kvr]) |*a, cv| a.* += (probs[t] / sum) * cv;
        }
        for (0..vd) |r| {
            cpu.dequantRow(.q8_0, row, kvb, h * stride + nope + r, kvr);
            var o: f64 = 0;
            for (row, ol) |wv, av| o += @as(f64, wv) * av;
            mass += @abs(o);
            worst = @max(worst, @abs(o - got[h * vd + r]));
        }
    }
    const tol = (mass / @as(f64, n_heads * vd)) * 1e-3;
    if (worst > tol) {
        std.debug.print("mla attention worst diff {d} tol {d}\n", .{ worst, tol });
        return error.MlaAttnMismatch;
    }
}

test "vulkan mla layer tail agrees with its constituents" {
    // Tail-in-one-submission against the same layer assembled from verified
    // pieces: device attention, exact f64 projection, host norm and router,
    // and the verified routed block. What this pins down is the plumbing --
    // buffer reuse, barrier placement, residual order -- which is exactly
    // where partial-output bugs lived on Metal.
    const gpa = std.testing.allocator;
    const n_heads = 4;
    const nope = 32;
    const vd = 16;
    const kvr = 64;
    const rope = 16;
    const ctx = 64;
    const seq = 4;
    const li = 0;
    const scale: f32 = 0.25;
    const stride = nope + vd;
    const dim = 512;
    const ffn = 256;
    const shexp_ffn = 256;
    const n_expert = 8;
    const n_used = 3;
    const eps: f32 = 1e-6;

    parallelBegin(1);
    defer parallelEnd();
    if (dev == null) return error.SkipZigTest;
    const saved = use_gpu_ops;
    use_gpu_ops = true;
    defer use_gpu_ops = saved;

    var prng = std.Random.DefaultPrng.init(0x1207B);
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
    const kvb = try gpa.alloc(u8, ggml.tensorBytes(.q8_0, kvr, n_heads * stride));
    defer gpa.free(kvb);
    const proj = try gpa.alloc(u8, ggml.tensorBytes(.q8_0, n_heads * vd, dim));
    defer gpa.free(proj);
    for ([_][]u8{ kvb, proj }) |data| {
        rnd.bytes(data);
        pin(data, 34, 0, false);
    }
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
    const router = try gpa.alloc(f32, n_expert * dim);
    defer gpa.free(router);
    for (router) |*v| v.* = (rnd.float(f32) - 0.5) * 0.05;
    const bias = try gpa.alloc(f32, n_expert);
    defer gpa.free(bias);
    for (bias) |*v| v.* = (rnd.float(f32) - 0.5) * 0.5;
    const norm_w = try gpa.alloc(f32, dim);
    defer gpa.free(norm_w);
    for (norm_w) |*v| v.* = 1.0;

    try std.testing.expect(mlaInit(1, ctx, kvr, rope));
    // A fresh cache shape may follow an earlier test's: reset the recorded
    // buffers so ensureMlaCache rebuilds for this geometry.
    mla_cache = null;
    mla_wk.clearRetainingCapacity();
    const wkbuf = try gpa.alloc(f32, n_heads * stride * kvr);
    defer gpa.free(wkbuf);
    for (0..n_heads * stride) |r| cpu.dequantRow(.q8_0, wkbuf[r * kvr ..][0..kvr], kvb, r, kvr);
    try std.testing.expect(mlaSetWk(0, wkbuf));
    for (0..seq) |t| {
        var c_row: [kvr]f32 = undefined;
        var r_row: [rope]f32 = undefined;
        for (&c_row) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
        for (&r_row) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
        try std.testing.expect(mlaAppend(li, t, &c_row, &r_row));
    }

    const q_nope = try gpa.alloc(f32, n_heads * nope);
    defer gpa.free(q_nope);
    const q_rope = try gpa.alloc(f32, n_heads * rope);
    defer gpa.free(q_rope);
    for (q_nope) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    for (q_rope) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
    const x0 = try gpa.alloc(f32, dim);
    defer gpa.free(x0);
    for (x0) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    const cfg = RoutedCfg{ .n_expert = n_expert, .n_used = n_used, .gating_sigmoid = true, .weights_norm = true, .weights_scale = 1.0 };
    const shexp: [3]WeightRef = .{ .{ .ty = .q4_k, .data = sg }, .{ .ty = .q4_k, .data = su }, .{ .ty = .q4_k, .data = sd } };

    // ---- tail ----
    const x_tail = try gpa.dupe(f32, x0);
    defer gpa.free(x_tail);
    const before = cmdbufs;
    try std.testing.expect(mlaLayerTail(li, seq - 1, x_tail, q_nope, q_rope, .{ .ty = .q8_0, .data = kvb }, .{ .ty = .q8_0, .data = proj }, norm_w, eps, .{ .ty = .f32, .data = std.mem.sliceAsBytes(router) }, bias, .{ .ty = .q4_k, .data = gw }, .{ .ty = .q8_0, .data = uw }, .{ .ty = .q5_0, .data = dw }, shexp, ffn, shexp_ffn, cfg, n_heads, nope, vd, scale));
    try std.testing.expectEqual(before + 1, cmdbufs);

    // ---- constituents ----
    const head_out = try gpa.alloc(f32, n_heads * vd);
    defer gpa.free(head_out);
    try std.testing.expect(mlaAttnHeads(li, seq - 1, q_nope, q_rope, .{ .ty = .q8_0, .data = kvb }, head_out, n_heads, nope, vd, scale));
    const x_ref = try gpa.dupe(f32, x0);
    defer gpa.free(x_ref);
    const row = try gpa.alloc(f32, n_heads * vd);
    defer gpa.free(row);
    for (0..dim) |r| {
        cpu.dequantRow(.q8_0, row, proj, r, n_heads * vd);
        var acc64: f64 = 0;
        for (row, head_out) |wv, hv| acc64 += @as(f64, wv) * @as(f64, hv);
        x_ref[r] += @floatCast(acc64);
    }
    const normed = try gpa.alloc(f32, dim);
    defer gpa.free(normed);
    cpu.rmsnorm(normed, x_ref, norm_w, eps);
    const logits = try gpa.alloc(f32, n_expert);
    defer gpa.free(logits);
    for (0..n_expert) |e| {
        var acc64: f64 = 0;
        for (router[e * dim ..][0..dim], normed) |wv, nv| acc64 += @as(f64, wv) * @as(f64, nv);
        logits[e] = @floatCast(acc64);
    }
    const ffn_out = try gpa.alloc(f32, dim);
    defer gpa.free(ffn_out);
    try std.testing.expect(moeFfnBlockRouted(normed, logits, bias, .{ .ty = .q4_k, .data = gw }, .{ .ty = .q8_0, .data = uw }, .{ .ty = .q5_0, .data = dw }, shexp, ffn, shexp_ffn, cfg, ffn_out));
    for (x_ref, ffn_out) |*a, v| a.* += v;

    var mass: f32 = 0;
    for (x_ref) |v| mass += @abs(v);
    const tol = (mass / @as(f32, @floatFromInt(dim))) * 1e-3;
    for (x_ref, x_tail, 0..) |a, bb, k| {
        std.testing.expectApproxEqAbs(a, bb, tol) catch |e| {
            std.debug.print("layer tail dim {d}: constituents {d} vs tail {d}\n", .{ k, a, bb });
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
    // The test tensors sit under the size cutover; pin it to zero so every
    // kernel actually runs (the submission-counter check would catch a
    // silent fallback as a failure, not a skip).
    const saved_min = min_bytes;
    min_bytes = 0;
    defer min_bytes = saved_min;

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
