//! One node-local forward pass step: embed -> [attention + FFN] x layers ->
//! final norm -> logits. Compute is entirely node-local (CLAUDE principle 1);
//! the only thing that "arrives" mid-pass is routed expert blocks, pulled
//! through the ExpertCache inside the MoE layers.

const std = @import("std");
const model = @import("model.zig");
const tensor = @import("tensor.zig");
const ckpt = @import("checkpoint.zig");
const attention = @import("attention.zig");
const moe = @import("moe.zig");
const ExpertCache = @import("expert_cache.zig").ExpertCache;
const ModelConfig = model.ModelConfig;

pub const State = struct {
    cfg: ModelConfig,
    weights: ckpt.Weights,
    kv: *attention.KVCache,
    cache: *ExpertCache,
    /// Reset once per token; all per-token temporaries allocate here.
    arena: *std.heap.ArenaAllocator,

    // persistent residual buffer, reused across tokens
    x: []f32,
};

/// Run one token at `pos`, writing vocab logits into `logits_out`.
pub fn step(s: *State, token: usize, pos: usize, logits_out: []f32) !void {
    _ = s.arena.reset(.retain_capacity);
    const scratch = s.arena.allocator();
    const cfg = s.cfg;
    const h = cfg.hidden;
    const w = s.weights;

    // embedding
    @memcpy(s.x, w.token_embedding[token * h ..][0..h]);

    const normed = try scratch.alloc(f32, h);
    const attn_out = try scratch.alloc(f32, h);
    const ffn_out = try scratch.alloc(f32, h);

    for (w.layers, 0..) |lw, li| {
        // attention block (pre-norm + residual)
        tensor.rmsnorm(normed, s.x, lw.input_norm, 1e-6);
        try attention.forward(scratch, cfg, lw, s.kv, li, pos, normed, attn_out);
        tensor.add(s.x, attn_out);

        // FFN block (pre-norm + residual)
        tensor.rmsnorm(normed, s.x, lw.post_attn_norm, 1e-6);
        if (lw.is_moe) {
            const moe_layer = li - cfg.n_dense_layers;
            try moe.moeFFN(scratch, cfg, lw, s.cache, moe_layer, normed, ffn_out);
        } else {
            try moe.denseFFN(scratch, cfg, lw.ffn_gate, lw.ffn_up, lw.ffn_down, cfg.dense_ffn, normed, ffn_out);
        }
        tensor.add(s.x, ffn_out);
    }

    // final norm + lm head
    const final = try scratch.alloc(f32, h);
    tensor.rmsnorm(final, s.x, w.final_norm, 1e-6);
    tensor.matvec(logits_out, w.lm_head, final, cfg.vocab_size, h);
}
