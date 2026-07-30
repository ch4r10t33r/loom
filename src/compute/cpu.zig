//! The CPU backend: SIMD kernels over mmap'd weights, row-parallel across a
//! worker pool.
//!
//! This is a binding layer, not an implementation — the kernels live in
//! `gguf/ggml.zig` (quantized matrix ops) and `core/tensor.zig` (elementwise).
//! Collecting them here is what lets `backend.zig` swap the whole set for a
//! GPU without every engine knowing.
//!
//! It also stays the correctness oracle once GPU backends exist: every GPU
//! kernel is checked against its CPU counterpart, the same way the codebook
//! decoders are checked against llama.cpp's.

const std = @import("std");
const ggml = @import("../gguf/ggml.zig");
const tensor = @import("../core/tensor.zig");

pub const matvec = ggml.matvec;
pub const matmul = ggml.matmul;
pub const dequantRow = ggml.dequantRow;

pub const WeightRef = struct { ty: ggml.Type, data: []const u8 };

pub const ExpertRef = struct { gate: WeightRef, up: WeightRef, down: WeightRef, weight: f32, ffn: usize };

/// Declined: the CPU path has no command buffer to amortize, so grouping a
/// layer's experts would buy nothing the caller's own loop does not already do.
pub fn moeFfnBlock(normed: []const f32, experts: []const ExpertRef, out: []f32) bool {
    _ = .{ normed, experts, out };
    return false;
}

/// Declined: host memory is already addressable, so there is nothing to register.
pub fn registerArena(mem: []const u8) bool {
    _ = mem;
    return false;
}

pub fn materializeArenas() usize {
    return 0;
}

pub fn takeCmdBufCount() usize {
    return 0;
}

/// The CPU engine keeps its own compressed cache and needs no device mirror.
pub fn mlaInit(layers: usize, ctx_len: usize, kvr: usize, rope: usize) bool {
    _ = .{ layers, ctx_len, kvr, rope };
    return false;
}

pub fn mlaAppend(li: usize, pos: usize, c_kv: []const f32, k_rope: []const f32) bool {
    _ = .{ li, pos, c_kv, k_rope };
    return false;
}

pub fn hasMlaCache() bool {
    return false;
}

pub const RoutedCfg = struct { n_expert: usize, n_used: usize, gating_sigmoid: bool, weights_norm: bool, weights_scale: f32 };

pub fn mlaLayerTail(li: usize, pos: usize, x: []f32, q_nope: []const f32, q_rope: []const f32, kv_b: WeightRef, attn_out_w: WeightRef, ffn_norm: []const f32, eps: f32, router_w: WeightRef, router_bias: ?[]const f32, gate_w: WeightRef, up_w: WeightRef, down_w: WeightRef, shexp: ?[3]WeightRef, ffn: usize, shexp_ffn: usize, cfg: RoutedCfg, n_heads: usize, nope: usize, v_head_dim: usize, scale: f32) bool {
    _ = .{ li, pos, x, q_nope, q_rope, kv_b, attn_out_w, ffn_norm, eps, router_w, router_bias, gate_w, up_w, down_w, shexp, ffn, shexp_ffn, cfg, n_heads, nope, v_head_dim, scale };
    return false;
}

pub fn moeFfnBlockRouted(normed: []const f32, logits: []const f32, bias: ?[]const f32, gate_w: WeightRef, up_w: WeightRef, down_w: WeightRef, shexp: ?[3]WeightRef, ffn: usize, shexp_ffn: usize, cfg: RoutedCfg, out: []f32) bool {
    _ = .{ normed, logits, bias, gate_w, up_w, down_w, shexp, ffn, shexp_ffn, cfg, out };
    return false;
}

pub const MlaLayerDesc = struct { attn_norm: []const f32, wq: WeightRef, kv_a: WeightRef, kv_a_norm: []const f32, kv_b: WeightRef, attn_out: WeightRef, ffn_norm: []const f32, is_moe: bool, router: WeightRef = .{ .ty = .f32, .data = &.{} }, gate: WeightRef = .{ .ty = .f32, .data = &.{} }, up: WeightRef = .{ .ty = .f32, .data = &.{} }, down: WeightRef = .{ .ty = .f32, .data = &.{} }, shexp: ?[3]WeightRef = null, ffn: usize = 0, shexp_ffn: usize = 0, dgate: WeightRef = .{ .ty = .f32, .data = &.{} }, dup: WeightRef = .{ .ty = .f32, .data = &.{} }, ddown: WeightRef = .{ .ty = .f32, .data = &.{} }, dffn: usize = 0 };
pub const MlaFrameCfg = struct { dim: usize, n_heads: usize, nope: usize, rope: usize, kvr: usize, v_head_dim: usize, eps: f32, scale: f32, rope_base: f32, yarn_factor: f32, yarn_orig_ctx: f32, routed: RoutedCfg };

pub fn mlaTokenFrame(descs: []const MlaLayerDesc, fc: MlaFrameCfg, x: []f32, pos: usize, out_norm: []const f32, lm_head: WeightRef, logits: []f32) bool {
    _ = .{ descs, fc, x, pos, out_norm, lm_head, logits };
    return false;
}

pub fn mlaReadCache(li: usize, pos: usize, c_kv: []f32, k_rope: []f32) bool {
    _ = .{ li, pos, c_kv, k_rope };
    return false;
}

pub fn moeRoute(logits: []const f32, bias: ?[]const f32, ids_out: []u32, gates_out: []f32, gating_sigmoid: bool, weights_norm: bool, weights_scale: f32) bool {
    _ = .{ logits, bias, ids_out, gates_out, gating_sigmoid, weights_norm, weights_scale };
    return false;
}

pub fn mlaSetWk(li: usize, wk_f32: []const f32) bool {
    _ = .{ li, wk_f32 };
    return false;
}

pub fn mlaAttnHeads(li: usize, pos: usize, q_nope: []const f32, q_rope: []const f32, kv_b: WeightRef, out: []f32, n_heads: usize, nope: usize, v_head_dim: usize, scale: f32) bool {
    _ = .{ li, pos, q_nope, q_rope, kv_b, out, n_heads, nope, v_head_dim, scale };
    return false;
}

pub var arena_error: ?[]const u8 = null;

/// The CPU has no device cache to set up and no fused attention to offer; the
/// engine's own path is already the implementation.
pub fn attnInit(layers: usize, ctx_len: usize, kvd: usize) bool {
    _ = .{ layers, ctx_len, kvd };
    return false;
}
pub fn disableAttn() void {}
pub fn enableKvMirror() void {}
pub fn releaseKvCache() void {}
pub fn hasKvCache() bool {
    return false;
}

pub const LayerSpec = struct {
    li: usize,
    pos: usize,
    attn_norm: []const f32,
    ffn_norm: []const f32,
    eps: f32,
    wq: WeightRef,
    wk: WeightRef,
    wv: WeightRef,
    wo: WeightRef,
    gate: WeightRef,
    up: WeightRef,
    down: WeightRef,
    dim: usize,
    ffn: usize,
    n_heads: usize,
    n_kv_heads: usize,
    hd: usize,
    rope_dim: usize,
    rope_base: f32,
    rope_neox: bool,
    attn_scale: f32,
};

/// The CPU has no frame: there is no submission to amortize, so every one of
/// these declines and the engine runs its own path.
pub fn beginFrame() bool {
    return false;
}
pub fn endFrame() void {}
pub fn frameOpen() bool {
    return false;
}
pub fn layerBlock(s: LayerSpec) bool {
    _ = s;
    return false;
}
pub fn frameLoadX(x: []const f32) bool {
    _ = x;
    return false;
}
pub fn frameStoreX(x: []f32) bool {
    _ = x;
    return false;
}
pub var frames_submitted: u64 = 0;
pub var dispatches_submitted: u64 = 0;
pub fn kvAppend(li: usize, pos: usize, k_new: []const f32, v_new: []const f32) bool {
    _ = .{ li, pos, k_new, v_new };
    return false;
}
pub fn attnHeads(
    li: usize,
    pos: usize,
    q: []const f32,
    out: []f32,
    n_heads: usize,
    n_kv_heads: usize,
    hd: usize,
    scale: f32,
) bool {
    _ = .{ li, pos, q, out, n_heads, n_kv_heads, hd, scale };
    return false;
}

pub var use_gpu_ops: bool = false;

pub const Shape = struct { data: []const u8, ty: ggml.Type, rows: usize, cols: usize };
pub const Verdict = struct { ran: bool, matvec_min_rows: usize, matvec_used: bool, ffn_used: bool, prefill_used: bool = false, attn_used: bool = false, attn_gpu_ms: f64 = 0, attn_cpu_ms: f64 = 0, ffn_gpu_ms: f64 = 0, ffn_cpu_ms: f64 = 0 };

/// Nothing to calibrate: with one backend there is no choice to measure.
pub fn calibrate(gpa: std.mem.Allocator, dim: usize, ffn: usize, shapes: []const Shape, triple: ?[3]Shape) void {
    _ = .{ gpa, dim, ffn, shapes, triple };
}
pub fn calibrateAttn(gpa: std.mem.Allocator, n_heads: usize, n_kv_heads: usize, hd: usize, seq: usize) void {
    _ = .{ gpa, n_heads, n_kv_heads, hd, seq };
}
pub fn lastVerdict() Verdict {
    return .{ .ran = false, .matvec_min_rows = 0, .matvec_used = false, .ffn_used = false };
}

/// The CPU declines the fused block, always.
///
/// Fusing buys a GPU one command buffer instead of six; it buys the CPU
/// nothing, because every op already runs on the same cores over the same
/// memory and the engine's own FFN sequence is exactly this. Declining keeps
/// that sequence as the single implementation rather than copying it here to
/// satisfy the seam.
pub fn ffnBlock(
    x: []f32,
    norm_w: []const f32,
    eps: f32,
    gate_w: WeightRef,
    up_w: WeightRef,
    down_w: WeightRef,
    ffn: usize,
) bool {
    _ = .{ x, norm_w, eps, gate_w, up_w, down_w, ffn };
    return false;
}
pub const dotF32 = ggml.dotF32;
pub const axpy = ggml.axpy;
pub const MAX_BATCH = ggml.MAX_BATCH;
pub const parallelBegin = ggml.parallelBegin;
pub const parallelEnd = ggml.parallelEnd;

pub const rmsnorm = tensor.rmsnorm;
pub const softmax = tensor.softmax;
pub const swiglu = tensor.swiglu;
pub const add = tensor.add;
pub const sigmoid = tensor.sigmoid;
