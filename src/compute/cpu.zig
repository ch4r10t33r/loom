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

/// The CPU has no device cache to set up and no fused attention to offer; the
/// engine's own path is already the implementation.
pub fn attnInit(layers: usize, ctx_len: usize, kvd: usize) bool {
    _ = .{ layers, ctx_len, kvd };
    return false;
}
pub fn disableAttn() void {}
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
