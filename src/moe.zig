//! Feed-forward paths: the dense FFN (first `n_dense_layers` layers) and the
//! MoE layer (DeepSeek-V3-style sigmoid router, top-k routed experts + an
//! always-active shared expert).
//!
//! The routed experts are the only weights that stream: each selected expert's
//! int4 block is materialized through the ExpertCache and matmul'd in place
//! (CLAUDE principle 7 — direct addressed fetch, no coding on the hot path).
//! The shared expert lives in the resident dense set.

const std = @import("std");
const model = @import("model.zig");
const tensor = @import("tensor.zig");
const quant = @import("quant.zig");
const ckpt = @import("checkpoint.zig");
const ExpertCache = @import("expert_cache.zig").ExpertCache;
const ModelConfig = model.ModelConfig;

/// SwiGLU FFN over f32 weights (dense layers and shared experts).
pub fn denseFFN(
    scratch: std.mem.Allocator,
    cfg: ModelConfig,
    gate_w: []const f32,
    up_w: []const f32,
    down_w: []const f32,
    ffn_dim: usize,
    x: []const f32,
    out: []f32,
) !void {
    const gate = try scratch.alloc(f32, ffn_dim);
    const up = try scratch.alloc(f32, ffn_dim);
    tensor.matvec(gate, gate_w, x, ffn_dim, cfg.hidden);
    tensor.matvec(up, up_w, x, ffn_dim, cfg.hidden);
    const act = try scratch.alloc(f32, ffn_dim);
    tensor.swiglu(act, gate, up);
    tensor.matvec(out, down_w, act, cfg.hidden, ffn_dim);
}

/// One int4 routed expert: SwiGLU over its streamed block.
fn expertFFN(
    scratch: std.mem.Allocator,
    cfg: ModelConfig,
    block: []const u8,
    x: []const f32,
    out: []f32,
) !void {
    const h = cfg.hidden;
    const f = cfg.moe_ffn;
    const gate_bytes = ModelConfig.q4MatrixBytes(f, h);
    const up_bytes = ModelConfig.q4MatrixBytes(f, h);
    const gate_w = block[0..gate_bytes];
    const up_w = block[gate_bytes .. gate_bytes + up_bytes];
    const down_w = block[gate_bytes + up_bytes ..];

    const gate = try scratch.alloc(f32, f);
    const up = try scratch.alloc(f32, f);
    quant.matvecQ4(gate, gate_w, x, f, h);
    quant.matvecQ4(up, up_w, x, f, h);
    const act = try scratch.alloc(f32, f);
    tensor.swiglu(act, gate, up);
    quant.matvecQ4(out, down_w, act, h, f);
}

const Selected = struct { expert: usize, gate: f32 };

/// DeepSeek-V3 / noaux_tc routing: sigmoid over router logits, pick the top-k
/// experts, renormalize their gates to sum to 1. (No auxiliary loss / bias term
/// is modeled in v0; the synthetic checkpoint has none.)
fn route(cfg: ModelConfig, router_logits: []const f32, sel: []Selected) void {
    const k = cfg.n_routed;
    // sigmoid scores
    var chosen: usize = 0;
    var used = [_]bool{false} ** 512; // n_experts bounded by 512 in practice
    while (chosen < k) : (chosen += 1) {
        var best: usize = 0;
        var best_score: f32 = -std.math.inf(f32);
        for (router_logits, 0..) |logit, e| {
            if (used[e]) continue;
            const s = tensor.sigmoid(logit);
            if (s > best_score) {
                best_score = s;
                best = e;
            }
        }
        used[best] = true;
        sel[chosen] = .{ .expert = best, .gate = best_score };
    }
    // renormalize gates over the selected set
    var sum: f32 = 0;
    for (sel[0..k]) |s| sum += s.gate;
    if (sum > 0) for (sel[0..k]) |*s| {
        s.gate /= sum;
    };
}

/// MoE layer forward. `moe_layer` is the index among MoE layers (0-based), used
/// to address experts in the cache.
pub fn moeFFN(
    scratch: std.mem.Allocator,
    cfg: ModelConfig,
    lw: ckpt.LayerWeights,
    cache: *ExpertCache,
    moe_layer: usize,
    x: []const f32,
    out: []f32,
) !void {
    std.debug.assert(cfg.n_experts <= 512);
    const router_logits = try scratch.alloc(f32, cfg.n_experts);
    tensor.matvec(router_logits, lw.router, x, cfg.n_experts, cfg.hidden);

    var sel_buf: [64]Selected = undefined;
    const sel = sel_buf[0..cfg.n_routed];
    route(cfg, router_logits, sel);

    @memset(out, 0);
    const e_out = try scratch.alloc(f32, cfg.hidden);

    // routed experts (streamed)
    for (sel) |s| {
        const id = ckpt.expertId(cfg, moe_layer, s.expert);
        const block = try cache.get(id);
        try expertFFN(scratch, cfg, block, x, e_out);
        for (out, e_out) |*o, v| o.* += s.gate * v;
    }

    // shared expert(s), always active, resident
    const f = cfg.moe_ffn;
    var si: usize = 0;
    while (si < cfg.n_shared) : (si += 1) {
        const gw = lw.shared_gate[si * f * cfg.hidden ..][0 .. f * cfg.hidden];
        const uw = lw.shared_up[si * f * cfg.hidden ..][0 .. f * cfg.hidden];
        const dw = lw.shared_down[si * cfg.hidden * f ..][0 .. cfg.hidden * f];
        try denseFFN(scratch, cfg, gw, uw, dw, f, x, e_out);
        tensor.add(out, e_out);
    }
}

test "router selects exactly n_routed distinct experts, gates sum to 1" {
    const cfg = model.tinyShape();
    var logits: [32]f32 = undefined;
    for (&logits, 0..) |*l, i| l.* = @floatFromInt(@as(i32, @intCast(i)) - 16);
    var sel_buf: [64]Selected = undefined;
    const sel = sel_buf[0..cfg.n_routed];
    route(cfg, logits[0..cfg.n_experts], sel);

    var sum: f32 = 0;
    for (sel) |s| sum += s.gate;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    // distinct
    for (sel, 0..) |a, i| for (sel[i + 1 ..]) |b| try std.testing.expect(a.expert != b.expert);
    // highest-logit expert (31) must be chosen
    var found = false;
    for (sel) |s| if (s.expert == 31) {
        found = true;
    };
    try std.testing.expect(found);
}
