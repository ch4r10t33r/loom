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
const mla_rope_raw = @embedFile("../shaders/vulkan/mla_rope.spv");
const mla_rope_spv: [mla_rope_raw.len]u8 align(4) = mla_rope_raw.*;
const copy_f32_raw = @embedFile("../shaders/vulkan/copy_f32.spv");
const copy_f32_spv: [copy_f32_raw.len]u8 align(4) = copy_f32_raw.*;

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
    rope: u64 = 0,
    copy: u64 = 0,
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
// *different* contents is served stale -- impossible for mmap'd weights, but
// very possible for TESTS that free and reallocate same-shape tensors, where
// the allocator hands back the same address run-dependently. That was a
// 1-in-5 flake on real hardware, misread as a barrier race for a day; every
// test that builds weight tensors clears this cache first.
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
        .rope = d.pipeline(&mla_rope_spv) catch return fail(),
        .copy = d.pipeline(&copy_f32_spv) catch return fail(),
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

const AbsorbPush = extern struct { n_heads: u32, nope: u32, kvr: u32, stride: u32, q_stride: u32, q_off: u32 };
const AttnPush = extern struct { n_heads: u32, kvr: u32, rope: u32, seq: u32, qr_stride: u32, qr_off: u32, c_base: u32, r_base: u32, scale: f32 };
const WsumPush = extern struct { n_heads: u32, kvr: u32, seq: u32, c_base: u32 };
const NormPush = extern struct { n: u32, eps: f32, x_off: u32 = 0, out_off: u32 = 0 };
const RopePush = extern struct { n_vec: u32, rope: u32, stride: u32, offset: u32, pos: u32, buf_off: u32, base: f32, yarn_factor: f32, yarn_orig_ctx: f32 };
const CopyPush = extern struct { n: u32, in_off: u32, out_off: u32 };

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
fn recordAttn(d: *vk.Device, c: vk.Device.Cmd, p: AttnPrep, li: usize, n_heads: usize, nope: usize, v_head_dim: usize, scale: f32, qsrc: vk.Buffer, q_stride: usize, q_off: usize, qr_stride: usize, qr_off: usize) vk.Error!void {
    const w = p.w;
    const apush = AbsorbPush{ .n_heads = @intCast(n_heads), .nope = @intCast(nope), .kvr = @intCast(w.kvr), .stride = @intCast(nope + v_head_dim), .q_stride = @intCast(q_stride), .q_off = @intCast(q_off) };
    try d.record(c, pipes.absorb, &.{ mla_wk.items[li], qsrc, qabs.? }, std.mem.asBytes(&apush), @intCast(n_heads * w.kvr));
    d.barrier(c);
    const tpush = AttnPush{
        .n_heads = @intCast(n_heads),
        .kvr = @intCast(w.kvr),
        .rope = @intCast(w.rope),
        .seq = @intCast(p.seq),
        .qr_stride = @intCast(qr_stride),
        .qr_off = @intCast(qr_off),
        .c_base = @intCast(li * w.ctx * w.kvr),
        .r_base = @intCast(mla_r0 + li * w.ctx * w.rope),
        .scale = scale,
    };
    try d.record(c, pipes.attn, &.{ qabs.?, qsrc, mla_cache.?, probs_buf.? }, std.mem.asBytes(&tpush), @intCast(n_heads));
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
    recordAttn(d, c, p, li, n_heads, nope, v_head_dim, scale, qbuf.?, nope, 0, p.w.rope, n_heads * nope) catch return false;
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
    const d_norm = NormPush{ .n = @intCast(dim), .eps = eps };
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
    recordAttn(d, c, p, li, n_heads, nope, v_head_dim, scale, qbuf.?, nope, 0, p.w.rope, n_heads * nope) catch return false;
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

var qfull: ?vk.Buffer = null; // [n_heads][kd] straight from the q projection
var kva_buf: ?vk.Buffer = null; // kvr + rope, the kv_a projection

/// The whole token as one submission: for every layer the attention head
/// (attn norm, q and kv_a projections, kv_a norm into the cache row, rope on
/// q and on the cached k_rope), the compressed-cache attention, the output
/// projection and residual, and the FFN -- routed MoE with shared expert or
/// dense -- then the final norm and lm_head. In: the residual stream after
/// embedding. Out: the stream and the logits. The device cache holds this
/// position's rows afterwards; the engine mirrors them back to its host
/// cache, which stays authoritative for any fallback.
///
/// The port of Metal's `mlaTokenFrame`, dispatch for dispatch; buffer
/// offsets become push-constant offsets throughout.
pub fn mlaTokenFrame(descs: []const MlaLayerDesc, fc: MlaFrameCfg, x: []f32, pos: usize, out_norm: []const f32, lm_head: WeightRef, logits: []f32) bool {
    const d = &(dev orelse return false);
    if (!use_gpu_ops) return false;
    // LOOM_NO_FRAME bisects against the per-layer tail in one binary.
    if (std.c.getenv("LOOM_NO_FRAME") != null) return false;
    const w = ensureMlaCache(d) orelse return false;
    if (pos + 1 > w.ctx or descs.len > mla_wk.items.len) return false;
    const dim = fc.dim;
    const kd = fc.nope + fc.rope;
    const seq = pos + 1;
    const vocab = logits.len;
    if (x.len != dim or out_norm.len != dim) return false;
    if (fc.routed.n_used == 0 or fc.routed.n_used > MAX_MOE_EXPERTS or fc.routed.n_expert > 256) return false;
    if (fc.kvr != w.kvr or fc.rope != w.rope) return false;

    // Resolve every layer -- pipelines, device weights, uploaded norms --
    // before a single dispatch is recorded. Norms go through the same
    // pointer-keyed cache as weights: uploaded once, tiny.
    const P = struct {
        anw: vk.Buffer,
        qp: u64,
        qw: vk.Buffer,
        kap: u64,
        kaw: vk.Buffer,
        kanw: vk.Buffer,
        vp: u64,
        vw: vk.Buffer,
        row_bytes: usize,
        pp: u64,
        pw: vk.Buffer,
        fnw: vk.Buffer,
        rp: u64 = 0,
        rw: vk.Buffer = undefined,
        gp: u64 = 0,
        gw: vk.Buffer = undefined,
        up: u64 = 0,
        uw: vk.Buffer = undefined,
        dp: u64 = 0,
        dw: vk.Buffer = undefined,
        sh_pipes: [3]u64 = @splat(0),
        shw: [3]vk.Buffer = undefined,
        dqp: [3]u64 = @splat(0),
        dqw: [3]vk.Buffer = undefined,
    };
    var plans: [64]P = undefined;
    if (descs.len > plans.len) return false;
    var s12_len: usize = 1;
    for (descs, 0..) |dd, i| {
        var p: P = .{ .anw = undefined, .qp = undefined, .qw = undefined, .kap = undefined, .kaw = undefined, .kanw = undefined, .vp = undefined, .vw = undefined, .row_bytes = undefined, .pp = undefined, .pw = undefined, .fnw = undefined };
        p.anw = weightBuffer(d, std.mem.sliceAsBytes(dd.attn_norm)) orelse return false;
        p.qp = pipeFor(dd.wq.ty, dim) orelse return false;
        p.qw = weightBuffer(d, dd.wq.data) orelse return false;
        p.kap = pipeFor(dd.kv_a.ty, dim) orelse return false;
        p.kaw = weightBuffer(d, dd.kv_a.data) orelse return false;
        p.kanw = weightBuffer(d, std.mem.sliceAsBytes(dd.kv_a_norm)) orelse return false;
        p.vp = idPipeFor(dd.kv_b.ty, fc.kvr) orelse return false;
        p.vw = weightBuffer(d, dd.kv_b.data) orelse return false;
        p.row_bytes = dd.kv_b.data.len / (fc.n_heads * (fc.nope + fc.v_head_dim));
        p.pp = pipeFor(dd.attn_out.ty, fc.n_heads * fc.v_head_dim) orelse return false;
        p.pw = weightBuffer(d, dd.attn_out.data) orelse return false;
        p.fnw = weightBuffer(d, std.mem.sliceAsBytes(dd.ffn_norm)) orelse return false;
        if (dd.is_moe) {
            p.rp = pipeFor(dd.router.ty, dim) orelse return false;
            p.rw = weightBuffer(d, dd.router.data) orelse return false;
            p.gp = idPipeFor(dd.gate.ty, dim) orelse return false;
            p.gw = weightBuffer(d, dd.gate.data) orelse return false;
            p.up = idPipeFor(dd.up.ty, dim) orelse return false;
            p.uw = weightBuffer(d, dd.up.data) orelse return false;
            p.dp = idPipeFor(dd.down.ty, dd.ffn) orelse return false;
            p.dw = weightBuffer(d, dd.down.data) orelse return false;
            s12_len = @max(s12_len, fc.routed.n_used * dd.ffn);
            if (dd.shexp) |sw| {
                p.sh_pipes[0] = pipeFor(sw[0].ty, dim) orelse return false;
                p.sh_pipes[1] = pipeFor(sw[1].ty, dim) orelse return false;
                p.sh_pipes[2] = pipeFor(sw[2].ty, dd.shexp_ffn) orelse return false;
                for (sw, 0..) |w2, k| p.shw[k] = weightBuffer(d, w2.data) orelse return false;
                s12_len = @max(s12_len, dd.shexp_ffn);
            }
        } else {
            p.dqp[0] = pipeFor(dd.dgate.ty, dim) orelse return false;
            p.dqp[1] = pipeFor(dd.dup.ty, dim) orelse return false;
            p.dqp[2] = pipeFor(dd.ddown.ty, dd.dffn) orelse return false;
            p.dqw[0] = weightBuffer(d, dd.dgate.data) orelse return false;
            p.dqw[1] = weightBuffer(d, dd.dup.data) orelse return false;
            p.dqw[2] = weightBuffer(d, dd.ddown.data) orelse return false;
            s12_len = @max(s12_len, dd.dffn);
        }
        plans[i] = p;
    }
    const onw = weightBuffer(d, std.mem.sliceAsBytes(out_norm)) orelse return false;
    const lmp = pipeFor(lm_head.ty, dim) orelse return false;
    const lmw = weightBuffer(d, lm_head.data) orelse return false;
    if (vids == null) {
        const b = d.alloc(64 * 4) catch return false;
        for (0..64) |k| b.slice(u32)[k] = @intCast(k);
        vids = b;
    }
    const xr = ensure(&xres, d, dim * 4) orelse return false;
    const nb = ensure(&normed_buf, d, dim * 4) orelse return false;
    const qf = ensure(&qfull, d, fc.n_heads * kd * 4) orelse return false;
    const kva = ensure(&kva_buf, d, (fc.kvr + fc.rope) * 4) orelse return false;
    _ = ensure(&qabs, d, fc.n_heads * fc.kvr * 4) orelse return false;
    _ = ensure(&probs_buf, d, fc.n_heads * w.ctx * 4) orelse return false;
    _ = ensure(&olat, d, fc.n_heads * fc.kvr * 4) orelse return false;
    _ = ensure(&hout, d, fc.n_heads * fc.v_head_dim * 4) orelse return false;
    const pj = ensure(&projb, d, dim * 4) orelse return false;
    const lb = ensure(&lbuf, d, @max(fc.routed.n_expert, 1) * 2 * 4) orelse return false;
    const idb = ensure(&ids_buf, d, MAX_MOE_EXPERTS * 4) orelse return false;
    const gb = ensure(&gates_buf, d, MAX_MOE_EXPERTS * 4) orelse return false;
    const s1 = ensure(&slots[0], d, s12_len * 4) orelse return false;
    const s2 = ensure(&slots[1], d, s12_len * 4) orelse return false;
    const s3 = ensure(&slots[2], d, @max(fc.routed.n_used, 1) * dim * 4) orelse return false;
    const acc = ensure(&accbuf, d, dim * 4) orelse return false;
    const ob = ensure(&obuf, d, vocab * 4) orelse return false;
    @memcpy(xr.slice(f32)[0..dim], x);

    const nd = NormPush{ .n = @intCast(dim), .eps = fc.eps };
    const n_add = extern struct { n: u32 }{ .n = @intCast(dim) };
    const Dims2 = extern struct { rows: u32, cols: u32 };
    const d_q = Dims2{ .rows = @intCast(fc.n_heads * kd), .cols = @intCast(dim) };
    const d_ka = Dims2{ .rows = @intCast(fc.kvr + fc.rope), .cols = @intCast(dim) };
    const d_proj = Dims2{ .rows = @intCast(dim), .cols = @intCast(fc.n_heads * fc.v_head_dim) };
    const d_router = Dims2{ .rows = @intCast(fc.routed.n_expert), .cols = @intCast(dim) };
    const rpush = RoutePush{
        .n_expert = @intCast(fc.routed.n_expert),
        .n_used = @intCast(fc.routed.n_used),
        .gating = if (fc.routed.gating_sigmoid) 1 else 0,
        .weights_norm = if (fc.routed.weights_norm) 1 else 0,
        .has_bias = 0, // buildFrameDescs declines biased-router models
        .weights_scale = fc.routed.weights_scale,
    };
    const apush = AbsorbPush{ .n_heads = @intCast(fc.n_heads), .nope = @intCast(fc.nope), .kvr = @intCast(fc.kvr), .stride = @intCast(fc.nope + fc.v_head_dim), .q_stride = @intCast(kd), .q_off = 0 };
    const rope_q = RopePush{ .n_vec = @intCast(fc.n_heads), .rope = @intCast(fc.rope), .stride = @intCast(kd), .offset = @intCast(fc.nope), .pos = @intCast(pos), .buf_off = 0, .base = fc.rope_base, .yarn_factor = fc.yarn_factor, .yarn_orig_ctx = fc.yarn_orig_ctx };

    const c = d.beginCmd() catch return false;
    for (descs, 0..) |dd, li| {
        const p = &plans[li];
        const c_pos: usize = (li * w.ctx + pos) * fc.kvr;
        const r_pos: usize = mla_r0 + (li * w.ctx + pos) * fc.rope;
        // ---- head ----
        d.record(c, pipes.rmsnorm, &.{ xr, p.anw, nb }, std.mem.asBytes(&nd), 1) catch return false;
        d.barrier(c);
        d.record(c, p.qp, &.{ p.qw, nb, qf }, std.mem.asBytes(&d_q), @intCast(fc.n_heads * kd)) catch return false;
        d.record(c, p.kap, &.{ p.kaw, nb, kva }, std.mem.asBytes(&d_ka), @intCast(fc.kvr + fc.rope)) catch return false;
        d.barrier(c);
        // c_kv: norm straight into the cache row; k_rope: copy then rotate there.
        const nd_kv = NormPush{ .n = @intCast(fc.kvr), .eps = fc.eps, .x_off = 0, .out_off = @intCast(c_pos) };
        d.record(c, pipes.rmsnorm, &.{ kva, p.kanw, mla_cache.? }, std.mem.asBytes(&nd_kv), 1) catch return false;
        const cp = CopyPush{ .n = @intCast(fc.rope), .in_off = @intCast(fc.kvr), .out_off = @intCast(r_pos) };
        d.record(c, pipes.copy, &.{ kva, mla_cache.? }, std.mem.asBytes(&cp), @intCast((fc.rope + 63) / 64)) catch return false;
        d.record(c, pipes.rope, &.{qf}, std.mem.asBytes(&rope_q), @intCast((fc.n_heads * fc.rope / 2 + 31) / 32)) catch return false;
        d.barrier(c);
        const rope_k = RopePush{ .n_vec = 1, .rope = @intCast(fc.rope), .stride = 0, .offset = 0, .pos = @intCast(pos), .buf_off = @intCast(r_pos), .base = fc.rope_base, .yarn_factor = fc.yarn_factor, .yarn_orig_ctx = fc.yarn_orig_ctx };
        d.record(c, pipes.rope, &.{mla_cache.?}, std.mem.asBytes(&rope_k), @intCast((fc.rope / 2 + 31) / 32)) catch return false;
        d.barrier(c);
        // ---- attention, q read in place: nope at h*kd, rope at h*kd+nope ----
        d.record(c, pipes.absorb, &.{ mla_wk.items[li], qf, qabs.? }, std.mem.asBytes(&apush), @intCast(fc.n_heads * fc.kvr)) catch return false;
        d.barrier(c);
        const tpush = AttnPush{ .n_heads = @intCast(fc.n_heads), .kvr = @intCast(fc.kvr), .rope = @intCast(fc.rope), .seq = @intCast(seq), .qr_stride = @intCast(kd), .qr_off = @intCast(fc.nope), .c_base = @intCast(li * w.ctx * fc.kvr), .r_base = @intCast(mla_r0 + li * w.ctx * fc.rope), .scale = fc.scale };
        d.record(c, pipes.attn, &.{ qabs.?, qf, mla_cache.?, probs_buf.? }, std.mem.asBytes(&tpush), @intCast(fc.n_heads)) catch return false;
        d.barrier(c);
        const wpush = WsumPush{ .n_heads = @intCast(fc.n_heads), .kvr = @intCast(fc.kvr), .seq = @intCast(seq), .c_base = @intCast(li * w.ctx * fc.kvr) };
        d.record(c, pipes.wsum, &.{ probs_buf.?, mla_cache.?, olat.? }, std.mem.asBytes(&wpush), @intCast(fc.n_heads * fc.kvr)) catch return false;
        d.barrier(c);
        const vpush = IdPush{ .rows = @intCast(fc.v_head_dim), .cols = @intCast(fc.kvr), .n_used = @intCast(fc.n_heads), .plane_stride = @intCast((fc.nope + fc.v_head_dim) * p.row_bytes), .x_stride = @intCast(fc.kvr), .base = @intCast(fc.nope * p.row_bytes) };
        d.record(c, p.vp, &.{ p.vw, olat.?, hout.?, vids.? }, std.mem.asBytes(&vpush), @intCast(fc.n_heads * fc.v_head_dim)) catch return false;
        d.barrier(c);
        // ---- projection, residual, FFN norm ----
        d.record(c, p.pp, &.{ p.pw, hout.?, pj }, std.mem.asBytes(&d_proj), @intCast(dim)) catch return false;
        d.barrier(c);
        d.record(c, pipes.add, &.{ xr, pj }, std.mem.asBytes(&n_add), @intCast((dim + 63) / 64)) catch return false;
        d.barrier(c);
        d.record(c, pipes.rmsnorm, &.{ xr, p.fnw, nb }, std.mem.asBytes(&nd), 1) catch return false;
        d.barrier(c);
        if (dd.is_moe) {
            d.record(c, p.rp, &.{ p.rw, nb, lb }, std.mem.asBytes(&d_router), @intCast(fc.routed.n_expert)) catch return false;
            d.barrier(c);
            d.record(c, pipes.route, &.{ lb, idb, gb }, std.mem.asBytes(&rpush), 1) catch return false;
            d.barrier(c);
            const d_gate = IdPush{ .rows = @intCast(dd.ffn), .cols = @intCast(dim), .n_used = @intCast(fc.routed.n_used), .plane_stride = @intCast(dd.gate.data.len / fc.routed.n_expert), .x_stride = 0 };
            const d_up = IdPush{ .rows = @intCast(dd.ffn), .cols = @intCast(dim), .n_used = @intCast(fc.routed.n_used), .plane_stride = @intCast(dd.up.data.len / fc.routed.n_expert), .x_stride = 0 };
            const d_down = IdPush{ .rows = @intCast(dim), .cols = @intCast(dd.ffn), .n_used = @intCast(fc.routed.n_used), .plane_stride = @intCast(dd.down.data.len / fc.routed.n_expert), .x_stride = @intCast(dd.ffn) };
            d.record(c, p.gp, &.{ p.gw, nb, s1, idb }, std.mem.asBytes(&d_gate), @intCast(fc.routed.n_used * dd.ffn)) catch return false;
            d.record(c, p.up, &.{ p.uw, nb, s2, idb }, std.mem.asBytes(&d_up), @intCast(fc.routed.n_used * dd.ffn)) catch return false;
            d.barrier(c);
            const n_sw = extern struct { n: u32 }{ .n = @intCast(fc.routed.n_used * dd.ffn) };
            d.record(c, pipes.swiglu_slots, &.{ s1, s2 }, std.mem.asBytes(&n_sw), @intCast((fc.routed.n_used * dd.ffn + 63) / 64)) catch return false;
            d.barrier(c);
            d.record(c, p.dp, &.{ p.dw, s1, s3, idb }, std.mem.asBytes(&d_down), @intCast(fc.routed.n_used * dim)) catch return false;
            d.barrier(c);
            const n_red = extern struct { n: u32, dim: u32 }{ .n = @intCast(fc.routed.n_used), .dim = @intCast(dim) };
            d.record(c, pipes.reduce_dev, &.{ acc, s3, gb }, std.mem.asBytes(&n_red), @intCast((dim + 63) / 64)) catch return false;
            d.barrier(c);
            if (dd.shexp != null) {
                const d_sh = Dims2{ .rows = @intCast(dd.shexp_ffn), .cols = @intCast(dim) };
                const d_shd = Dims2{ .rows = @intCast(dim), .cols = @intCast(dd.shexp_ffn) };
                const n_shsw = extern struct { n: u32 }{ .n = @intCast(dd.shexp_ffn) };
                d.record(c, p.sh_pipes[0], &.{ p.shw[0], nb, s1 }, std.mem.asBytes(&d_sh), @intCast(dd.shexp_ffn)) catch return false;
                d.record(c, p.sh_pipes[1], &.{ p.shw[1], nb, s2 }, std.mem.asBytes(&d_sh), @intCast(dd.shexp_ffn)) catch return false;
                d.barrier(c);
                d.record(c, pipes.swiglu_slots, &.{ s1, s2 }, std.mem.asBytes(&n_shsw), @intCast((dd.shexp_ffn + 63) / 64)) catch return false;
                d.barrier(c);
                d.record(c, p.sh_pipes[2], &.{ p.shw[2], s1, s3 }, std.mem.asBytes(&d_shd), @intCast(dim)) catch return false;
                d.barrier(c);
                d.record(c, pipes.add, &.{ acc, s3 }, std.mem.asBytes(&n_add), @intCast((dim + 63) / 64)) catch return false;
                d.barrier(c);
            }
            d.record(c, pipes.add, &.{ xr, acc }, std.mem.asBytes(&n_add), @intCast((dim + 63) / 64)) catch return false;
        } else {
            const d_f = Dims2{ .rows = @intCast(dd.dffn), .cols = @intCast(dim) };
            const d_fd = Dims2{ .rows = @intCast(dim), .cols = @intCast(dd.dffn) };
            const n_f = extern struct { n: u32 }{ .n = @intCast(dd.dffn) };
            d.record(c, p.dqp[0], &.{ p.dqw[0], nb, s1 }, std.mem.asBytes(&d_f), @intCast(dd.dffn)) catch return false;
            d.record(c, p.dqp[1], &.{ p.dqw[1], nb, s2 }, std.mem.asBytes(&d_f), @intCast(dd.dffn)) catch return false;
            d.barrier(c);
            d.record(c, pipes.swiglu_slots, &.{ s1, s2 }, std.mem.asBytes(&n_f), @intCast((dd.dffn + 63) / 64)) catch return false;
            d.barrier(c);
            d.record(c, p.dqp[2], &.{ p.dqw[2], s1, s3 }, std.mem.asBytes(&d_fd), @intCast(dim)) catch return false;
            d.barrier(c);
            d.record(c, pipes.add, &.{ xr, s3 }, std.mem.asBytes(&n_add), @intCast((dim + 63) / 64)) catch return false;
        }
        d.barrier(c);
    }
    // ---- final norm and lm_head ----
    d.record(c, pipes.rmsnorm, &.{ xr, onw, nb }, std.mem.asBytes(&nd), 1) catch return false;
    d.barrier(c);
    const d_lm = Dims2{ .rows = @intCast(vocab), .cols = @intCast(dim) };
    d.record(c, lmp, &.{ lmw, nb, ob }, std.mem.asBytes(&d_lm), @intCast(vocab)) catch return false;
    d.submitWait(c) catch return false;
    cmdbufs += 1;
    @memcpy(x, xr.slice(f32)[0..dim]);
    @memcpy(logits, ob.slice(f32)[0..vocab]);
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
    wcache.clearRetainingCapacity(); // test tensors may recycle freed addresses

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
    wcache.clearRetainingCapacity(); // test tensors may recycle freed addresses

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
    wcache.clearRetainingCapacity(); // test tensors may recycle freed addresses

    var prng = std.Random.DefaultPrng.init(0x1207A);
    const rnd = prng.random();
    const kvb = try gpa.alloc(u8, ggml.tensorBytes(.q8_0, kvr, n_heads * stride));
    defer gpa.free(kvb);
    rnd.bytes(kvb);
    var b: usize = 0;
    while (b + 34 <= kvb.len) : (b += 34) std.mem.writeInt(u16, kvb[b..][0..2], 0x2C00, .little);

    try std.testing.expect(mlaInit(2, ctx, kvr, rope));
    mla_cache = null;
    mla_wk.clearRetainingCapacity();
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
    wcache.clearRetainingCapacity(); // test tensors may recycle freed addresses

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

test "vulkan mla rope kernel matches the host ropeApply, plain and yarn, strided" {
    // Rotates in place where q lives -- stride kd, offset nope, family base
    // buf_off -- so a wrong stride rotates the wrong halves of the wrong
    // heads, which is exactly as silent as every other rope bug.
    const deepseek = @import("../gguf/deepseek.zig");
    const gpa = std.testing.allocator;
    const n_vec = 4;
    const rope = 16;
    const stride = 48;
    const offset = 24;
    const buf_off = 8;
    const pos = 37;

    parallelBegin(1);
    defer parallelEnd();
    const d = &(dev orelse return error.SkipZigTest);
    const saved = use_gpu_ops;
    use_gpu_ops = true;
    defer use_gpu_ops = saved;
    wcache.clearRetainingCapacity(); // test tensors may recycle freed addresses

    var prng = std.Random.DefaultPrng.init(0x1207E);
    const rnd = prng.random();
    const host = try gpa.alloc(f32, buf_off + n_vec * stride);
    defer gpa.free(host);
    for (host) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    inline for (.{ 1.0, 40.0 }) |yf| {
        const dev_buf = d.alloc(host.len * 4) catch return error.SkipZigTest;
        @memcpy(dev_buf.slice(f32)[0..host.len], host);
        const want = try gpa.dupe(f32, host);
        defer gpa.free(want);
        var cfg: deepseek.Config = undefined;
        cfg.rope_base = 10000.0;
        cfg.yarn_factor = yf;
        cfg.yarn_orig_ctx = 4096.0;
        cfg.yarn_log_mul = 0;
        for (0..n_vec) |k| deepseek.ropeApplyForTest(cfg, want[buf_off + k * stride + offset ..][0..rope], pos);

        const push = RopePush{ .n_vec = n_vec, .rope = rope, .stride = stride, .offset = offset, .pos = pos, .buf_off = buf_off, .base = 10000.0, .yarn_factor = yf, .yarn_orig_ctx = 4096.0 };
        d.dispatchWait(pipes.rope, &.{dev_buf}, std.mem.asBytes(&push), (n_vec * rope / 2 + 31) / 32) catch return error.SkipZigTest;

        const got = dev_buf.slice(f32)[0..host.len];
        for (want, got, 0..) |a, b, k| {
            std.testing.expectApproxEqAbs(a, b, 1e-5) catch |e| {
                std.debug.print("rope yf={d} idx {d}: host {d} vs gpu {d}\n", .{ yf, k, a, b });
                return e;
            };
        }
    }
}

test "vulkan whole-token frame agrees with the f64 constituents" {
    // Two layers -- dense then MoE-with-shared-expert -- through the whole
    // frame: head projections, norm-into-cache, rope on device, attention,
    // FFN, final norm, q6_k lm_head. The reference recomputes the token in
    // f64 from the quantized tensors, sharing nothing with the kernels; the
    // cache rows the frame wrote are checked too, because a frame that
    // attends correctly but caches garbage only fails on the NEXT token.
    const deepseek = @import("../gguf/deepseek.zig");
    const moe = @import("../gguf/moe.zig");
    const gpa = std.testing.allocator;
    const n_heads = 4;
    const nope = 32;
    const vd = 16;
    const kd = nope + 16;
    const kvr = 64;
    const rope = 16;
    const ctx = 64;
    const pos = 3;
    const dim = 512;
    const dffn = 192;
    const ffn = 256;
    const shexp_ffn = 256;
    const n_expert = 8;
    const n_used = 3;
    const vocab = 128;
    const stride = nope + vd;
    const eps: f32 = 1e-6;
    const scale: f32 = 0.25;

    parallelBegin(1);
    defer parallelEnd();
    if (dev == null) return error.SkipZigTest;
    const saved = use_gpu_ops;
    use_gpu_ops = true;
    defer use_gpu_ops = saved;
    wcache.clearRetainingCapacity(); // test tensors may recycle freed addresses

    var prng = std.Random.DefaultPrng.init(0x1207F);
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
    const q8 = struct {
        fn alloc(g: std.mem.Allocator, r: std.Random, cols: usize, rows: usize) ![]u8 {
            const data = try g.alloc(u8, ggml.tensorBytes(.q8_0, cols, rows));
            r.bytes(data);
            var b: usize = 0;
            while (b + 34 <= data.len) : (b += 34) std.mem.writeInt(u16, data[b..][0..2], 0x2C00, .little);
            return data;
        }
    };

    // Per-layer attention tensors, distinct so a cross-layer mixup cannot pass.
    var kvb: [2][]u8 = undefined;
    var wq: [2][]u8 = undefined;
    var kva_w: [2][]u8 = undefined;
    var attn_out: [2][]u8 = undefined;
    var attn_norm: [2][]f32 = undefined;
    var kva_norm: [2][]f32 = undefined;
    var ffn_norm: [2][]f32 = undefined;
    for (0..2) |l| {
        kvb[l] = try q8.alloc(gpa, rnd, kvr, n_heads * stride);
        wq[l] = try q8.alloc(gpa, rnd, dim, n_heads * kd);
        kva_w[l] = try q8.alloc(gpa, rnd, dim, kvr + rope);
        attn_out[l] = try q8.alloc(gpa, rnd, n_heads * vd, dim);
        attn_norm[l] = try gpa.alloc(f32, dim);
        kva_norm[l] = try gpa.alloc(f32, kvr);
        ffn_norm[l] = try gpa.alloc(f32, dim);
        for (attn_norm[l]) |*v| v.* = 0.5 + rnd.float(f32);
        for (kva_norm[l]) |*v| v.* = 0.5 + rnd.float(f32);
        for (ffn_norm[l]) |*v| v.* = 0.5 + rnd.float(f32);
    }
    defer for (0..2) |l| {
        gpa.free(kvb[l]);
        gpa.free(wq[l]);
        gpa.free(kva_w[l]);
        gpa.free(attn_out[l]);
        gpa.free(attn_norm[l]);
        gpa.free(kva_norm[l]);
        gpa.free(ffn_norm[l]);
    };
    // Layer 0 dense FFN; layer 1 routed with shared expert.
    const dgate = try q8.alloc(gpa, rnd, dim, dffn);
    defer gpa.free(dgate);
    const dup = try q8.alloc(gpa, rnd, dim, dffn);
    defer gpa.free(dup);
    const ddown = try q8.alloc(gpa, rnd, dffn, dim);
    defer gpa.free(ddown);
    const gw = try gpa.alloc(u8, n_expert * ggml.tensorBytes(.q4_k, dim, ffn));
    defer gpa.free(gw);
    rnd.bytes(gw);
    pin(gw, 144, 0, true);
    const uw = try q8.alloc(gpa, rnd, dim, n_expert * ffn);
    defer gpa.free(uw);
    const dw = try gpa.alloc(u8, n_expert * ggml.tensorBytes(.q5_0, ffn, dim));
    defer gpa.free(dw);
    rnd.bytes(dw);
    pin(dw, 22, 0, false);
    const sg = try gpa.alloc(u8, ggml.tensorBytes(.q4_k, dim, shexp_ffn));
    defer gpa.free(sg);
    const su = try gpa.alloc(u8, ggml.tensorBytes(.q4_k, dim, shexp_ffn));
    defer gpa.free(su);
    const sd = try gpa.alloc(u8, ggml.tensorBytes(.q4_k, shexp_ffn, dim));
    defer gpa.free(sd);
    for ([_][]u8{ sg, su, sd }) |data| {
        rnd.bytes(data);
        pin(data, 144, 0, true);
    }
    const router = try gpa.alloc(f32, n_expert * dim);
    defer gpa.free(router);
    for (router) |*v| v.* = (rnd.float(f32) - 0.5) * 0.05;
    const out_norm = try gpa.alloc(f32, dim);
    defer gpa.free(out_norm);
    for (out_norm) |*v| v.* = 0.5 + rnd.float(f32);
    const lmh = try gpa.alloc(u8, ggml.tensorBytes(.q6_k, dim, vocab));
    defer gpa.free(lmh);
    rnd.bytes(lmh);
    pin(lmh, 210, 208, false);

    try std.testing.expect(mlaInit(2, ctx, kvr, rope));
    mla_cache = null;
    mla_wk.clearRetainingCapacity();
    const wkbuf = try gpa.alloc(f32, n_heads * stride * kvr);
    defer gpa.free(wkbuf);
    for (0..2) |l| {
        for (0..n_heads * stride) |r| cpu.dequantRow(.q8_0, wkbuf[r * kvr ..][0..kvr], kvb[l], r, kvr);
        try std.testing.expect(mlaSetWk(l, wkbuf));
    }
    // Prior positions, distinct per layer.
    var hist_c: [2][pos][kvr]f32 = undefined;
    var hist_r: [2][pos][rope]f32 = undefined;
    for (0..2) |l| for (0..pos) |t| {
        for (&hist_c[l][t]) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
        for (&hist_r[l][t]) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
        try std.testing.expect(mlaAppend(l, t, &hist_c[l][t], &hist_r[l][t]));
    };

    const x0 = try gpa.alloc(f32, dim);
    defer gpa.free(x0);
    for (x0) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;

    var descs: [2]MlaLayerDesc = undefined;
    for (0..2) |l| {
        descs[l] = .{
            .attn_norm = attn_norm[l],
            .wq = .{ .ty = .q8_0, .data = wq[l] },
            .kv_a = .{ .ty = .q8_0, .data = kva_w[l] },
            .kv_a_norm = kva_norm[l],
            .kv_b = .{ .ty = .q8_0, .data = kvb[l] },
            .attn_out = .{ .ty = .q8_0, .data = attn_out[l] },
            .ffn_norm = ffn_norm[l],
            .is_moe = l == 1,
        };
    }
    descs[0].dgate = .{ .ty = .q8_0, .data = dgate };
    descs[0].dup = .{ .ty = .q8_0, .data = dup };
    descs[0].ddown = .{ .ty = .q8_0, .data = ddown };
    descs[0].dffn = dffn;
    descs[1].router = .{ .ty = .f32, .data = std.mem.sliceAsBytes(router) };
    descs[1].gate = .{ .ty = .q4_k, .data = gw };
    descs[1].up = .{ .ty = .q8_0, .data = uw };
    descs[1].down = .{ .ty = .q5_0, .data = dw };
    descs[1].ffn = ffn;
    descs[1].shexp = .{ .{ .ty = .q4_k, .data = sg }, .{ .ty = .q4_k, .data = su }, .{ .ty = .q4_k, .data = sd } };
    descs[1].shexp_ffn = shexp_ffn;
    const fc = MlaFrameCfg{
        .dim = dim,
        .n_heads = n_heads,
        .nope = nope,
        .rope = rope,
        .kvr = kvr,
        .v_head_dim = vd,
        .eps = eps,
        .scale = scale,
        .rope_base = 10000.0,
        .yarn_factor = 1.0,
        .yarn_orig_ctx = 4096.0,
        .routed = .{ .n_expert = n_expert, .n_used = n_used, .gating_sigmoid = true, .weights_norm = true, .weights_scale = 1.0 },
    };

    const x_frame = try gpa.dupe(f32, x0);
    defer gpa.free(x_frame);
    const logits_frame = try gpa.alloc(f32, vocab);
    defer gpa.free(logits_frame);
    @memset(logits_frame, std.math.nan(f32));
    const before = cmdbufs;
    try std.testing.expect(mlaTokenFrame(&descs, fc, x_frame, pos, out_norm, .{ .ty = .q6_k, .data = lmh }, logits_frame));
    try std.testing.expectEqual(before + 1, cmdbufs); // the whole token, one submission
    // Determinism probe: the same token twice must be bit-identical --
    // divergence here is a synchronization race, the exact failure mode the
    // barrier comment in device.zig documents.
    {
        const x2 = try gpa.dupe(f32, x0);
        defer gpa.free(x2);
        const l2 = try gpa.alloc(f32, vocab);
        defer gpa.free(l2);
        try std.testing.expect(mlaTokenFrame(&descs, fc, x2, pos, out_norm, .{ .ty = .q6_k, .data = lmh }, l2));
        var ndiff: usize = 0;
        var first: usize = 0;
        for (x_frame, x2, 0..) |a, b2, k| {
            if (a != b2) {
                if (ndiff == 0) first = k;
                ndiff += 1;
            }
        }
        std.debug.print("DBG determinism: {d}/{d} dims differ (first {d}); run1[0..4]={any}, run2[0..4]={any}\n", .{ ndiff, dim, first, x_frame[0..4], x2[0..4] });
    }

    // ---- f64 reference ----
    const xr = try gpa.alloc(f64, dim);
    defer gpa.free(xr);
    for (xr, x0) |*a, v| a.* = v;
    const nrm = struct {
        fn f(out: []f64, in: []const f64, wgt: []const f32, e: f32) void {
            var ss: f64 = 0;
            for (in) |v| ss += v * v;
            const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(in.len)) + e);
            for (out, in, wgt) |*o, v, wv| o.* = v * @as(f64, wv) * inv;
        }
    }.f;
    const mv64 = struct {
        fn f(g: std.mem.Allocator, ty: ggml.Type, data: []const u8, xin: []const f64, rows: usize, cols: usize, out: []f64) !void {
            const rbuf = try g.alloc(f32, cols);
            defer g.free(rbuf);
            for (0..rows) |r| {
                cpu.dequantRow(ty, rbuf, data, r, cols);
                var a: f64 = 0;
                for (rbuf, xin[0..cols]) |wv, xv| a += @as(f64, wv) * xv;
                out[r] = a;
            }
        }
    }.f;
    var rope_cfg: deepseek.Config = undefined;
    rope_cfg.rope_base = 10000.0;
    rope_cfg.yarn_factor = 1.0;
    rope_cfg.yarn_orig_ctx = 4096.0;
    rope_cfg.yarn_log_mul = 0;

    const normed = try gpa.alloc(f64, dim);
    defer gpa.free(normed);
    const scratch = try gpa.alloc(f64, dim * 4);
    defer gpa.free(scratch);
    const rowbuf = try gpa.alloc(f32, dim);
    defer gpa.free(rowbuf);
    for (0..2) |l| {
        nrm(normed, xr, attn_norm[l], eps);
        // head: q, kv_a, cache row, rope (host rope runs in f32, as the engine's does)
        const q64 = scratch[0 .. n_heads * kd];
        try mv64(gpa, .q8_0, wq[l], normed, n_heads * kd, dim, q64);
        var q32: [n_heads * kd]f32 = undefined;
        for (&q32, q64) |*o, v| o.* = @floatCast(v);
        for (0..n_heads) |h| deepseek.ropeApplyForTest(rope_cfg, q32[h * kd + nope ..][0..rope], pos);
        var kva64: [kvr + rope]f64 = undefined;
        try mv64(gpa, .q8_0, kva_w[l], normed, kvr + rope, dim, &kva64);
        var c_new64: [kvr]f64 = undefined;
        nrm(&c_new64, kva64[0..kvr], kva_norm[l], eps);
        var c_new: [kvr]f32 = undefined;
        for (&c_new, c_new64) |*o, v| o.* = @floatCast(v);
        var r_new: [rope]f32 = undefined;
        for (&r_new, kva64[kvr..]) |*o, v| o.* = @floatCast(v);
        deepseek.ropeApplyForTest(rope_cfg, &r_new, pos);
        // the cache rows the frame wrote must match these
        var c_dev: [kvr]f32 = undefined;
        var r_dev: [rope]f32 = undefined;
        try std.testing.expect(mlaReadCache(l, pos, &c_dev, &r_dev));
        for (c_new, c_dev) |a, b| try std.testing.expectApproxEqAbs(a, b, 2e-3);
        for (r_new, r_dev) |a, b| try std.testing.expectApproxEqAbs(a, b, 2e-3);
        // attention over pos+1 rows
        const head_out = scratch[dim .. dim + n_heads * vd];
        const wkrow = try gpa.alloc(f32, kvr);
        defer gpa.free(wkrow);
        for (0..n_heads) |h| {
            var qa: [kvr]f64 = @splat(0);
            for (0..nope) |r| {
                cpu.dequantRow(.q8_0, wkrow, kvb[l], h * stride + r, kvr);
                for (&qa, wkrow) |*a, wv| a.* += q64[h * kd + r] * wv;
            }
            var probs: [pos + 1]f64 = undefined;
            var mx: f64 = -std.math.inf(f64);
            for (0..pos + 1) |t| {
                const ct: []const f32 = if (t < pos) &hist_c[l][t] else &c_new;
                const rt: []const f32 = if (t < pos) &hist_r[l][t] else &r_new;
                var sc: f64 = 0;
                for (qa, ct) |a, cv| sc += a * cv;
                for (0..rope) |i| sc += @as(f64, q32[h * kd + nope + i]) * rt[i];
                probs[t] = sc * scale;
                mx = @max(mx, probs[t]);
            }
            var sum: f64 = 0;
            for (&probs) |*pv| {
                pv.* = @exp(pv.* - mx);
                sum += pv.*;
            }
            var ol: [kvr]f64 = @splat(0);
            for (0..pos + 1) |t| {
                const ct: []const f32 = if (t < pos) &hist_c[l][t] else &c_new;
                for (&ol, ct) |*a, cv| a.* += (probs[t] / sum) * cv;
            }
            for (0..vd) |r| {
                cpu.dequantRow(.q8_0, wkrow, kvb[l], h * stride + nope + r, kvr);
                var o: f64 = 0;
                for (wkrow, ol) |wv, av| o += @as(f64, wv) * av;
                head_out[h * vd + r] = o;
            }
        }
        const proj = scratch[dim * 2 .. dim * 3];
        try mv64(gpa, .q8_0, attn_out[l], head_out, dim, n_heads * vd, proj);
        for (xr, proj) |*a, v| a.* += v;
        nrm(normed, xr, ffn_norm[l], eps);
        const silu = struct {
            fn f(g: f64) f64 {
                return g / (1.0 + @exp(-g));
            }
        }.f;
        if (l == 0) {
            const g64 = try gpa.alloc(f64, dffn);
            defer gpa.free(g64);
            const u64_ = try gpa.alloc(f64, dffn);
            defer gpa.free(u64_);
            try mv64(gpa, .q8_0, dgate, normed, dffn, dim, g64);
            try mv64(gpa, .q8_0, dup, normed, dffn, dim, u64_);
            for (g64, u64_) |*g, u| g.* = silu(g.*) * u;
            const down64 = scratch[dim * 3 ..];
            try mv64(gpa, .q8_0, ddown, g64, dim, dffn, down64);
            for (xr, down64) |*a, v| a.* += v;
        } else {
            var logits32: [n_expert]f32 = undefined;
            for (0..n_expert) |e| {
                var a: f64 = 0;
                for (router[e * dim ..][0..dim], normed) |wv, nv| a += @as(f64, wv) * nv;
                logits32[e] = @floatCast(a);
            }
            var sel: [n_used]moe.Selected = undefined;
            moe.route(.{ .n_expert = n_expert, .n_used = n_used, .gating = .sigmoid, .weights_norm = true, .weights_scale = 1.0 }, &logits32, null, &sel);
            const g64 = try gpa.alloc(f64, ffn);
            defer gpa.free(g64);
            const u64_ = try gpa.alloc(f64, ffn);
            defer gpa.free(u64_);
            const down64 = scratch[dim * 3 ..];
            const gate_plane = gw.len / n_expert;
            const up_plane = uw.len / n_expert;
            const down_plane = dw.len / n_expert;
            const ffn_acc = try gpa.alloc(f64, dim);
            defer gpa.free(ffn_acc);
            @memset(ffn_acc, 0);
            for (sel) |se| {
                try mv64(gpa, .q4_k, gw[se.expert * gate_plane ..][0..gate_plane], normed, ffn, dim, g64);
                try mv64(gpa, .q8_0, uw[se.expert * up_plane ..][0..up_plane], normed, ffn, dim, u64_);
                for (g64, u64_) |*g, u| g.* = silu(g.*) * u;
                try mv64(gpa, .q5_0, dw[se.expert * down_plane ..][0..down_plane], g64, dim, ffn, down64);
                for (ffn_acc, down64) |*a, v| a.* += @as(f64, se.gate) * v;
            }
            const sg64 = try gpa.alloc(f64, shexp_ffn);
            defer gpa.free(sg64);
            const su64 = try gpa.alloc(f64, shexp_ffn);
            defer gpa.free(su64);
            try mv64(gpa, .q4_k, sg, normed, shexp_ffn, dim, sg64);
            try mv64(gpa, .q4_k, su, normed, shexp_ffn, dim, su64);
            for (sg64, su64) |*g, u| g.* = silu(g.*) * u;
            try mv64(gpa, .q4_k, sd, sg64, dim, shexp_ffn, down64);
            for (ffn_acc, down64) |*a, v| a.* += v;
            for (xr, ffn_acc) |*a, v| a.* += v;
        }
    }
    nrm(normed, xr, out_norm, eps);
    const logits_ref = try gpa.alloc(f64, vocab);
    defer gpa.free(logits_ref);
    try mv64(gpa, .q6_k, lmh, normed, vocab, dim, logits_ref);

    var mass: f64 = 0;
    for (xr) |v| mass += @abs(v);
    const tol_x: f32 = @floatCast((mass / dim) * 2e-3);
    for (xr, x_frame, 0..) |a, b, k| {
        std.testing.expectApproxEqAbs(@as(f32, @floatCast(a)), b, tol_x) catch |e| {
            std.debug.print("frame x dim {d}: reference {d} vs frame {d}\n", .{ k, a, b });
            return e;
        };
    }
    var lmass: f64 = 0;
    for (logits_ref) |v| lmass += @abs(v);
    const tol_l: f32 = @floatCast((lmass / vocab) * 2e-3);
    for (logits_ref, logits_frame, 0..) |a, b, k| {
        std.testing.expectApproxEqAbs(@as(f32, @floatCast(a)), b, tol_l) catch |e| {
            std.debug.print("frame logit {d}: reference {d} vs frame {d}\n", .{ k, a, b });
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
    wcache.clearRetainingCapacity(); // test tensors may recycle freed addresses
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
