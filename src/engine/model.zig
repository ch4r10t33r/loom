//! Model configuration for a GLM-5.2-shaped MoE transformer.
//!
//! The engine is MoE-agnostic: every dimension here is read from the checkpoint
//! manifest at load time, so the same binary runs the tiny synthetic model used
//! for the v0 harness and (given real weights + a matching manifest) the full
//! 744B GLM-5.2 shape. `glmShape()` and `tinyShape()` are the two presets.

const std = @import("std");

pub const QK: usize = 32; // int4 quantization block size (values per scale)

pub const ModelConfig = extern struct {
    // Token / embedding
    vocab_size: u32,
    hidden: u32,

    // Layer counts. Layers [0, n_dense) are dense FFN; the rest are MoE.
    n_dense_layers: u32,
    n_moe_layers: u32,

    // Attention (MLA: q/kv-LoRA + partial RoPE)
    n_heads: u32,
    nope_dim: u32, // per-head dims without RoPE
    rope_dim: u32, // per-head dims carrying RoPE
    v_head_dim: u32,
    q_lora_rank: u32,
    kv_lora_rank: u32,

    // Feed-forward widths
    dense_ffn: u32, // hidden dim of the dense (first-3-layers) FFN
    moe_ffn: u32, // hidden dim of a single routed/shared expert

    // Mixture of experts
    n_experts: u32, // routed experts per MoE layer
    n_routed: u32, // top-k routed experts activated per token (GLM: 8)
    n_shared: u32, // always-active shared experts (GLM: 1)

    // Sampling / positional
    max_seq_len: u32,
    rope_theta: f32,

    pub fn nLayers(self: ModelConfig) usize {
        return self.n_dense_layers + self.n_moe_layers;
    }

    pub fn headDim(self: ModelConfig) usize {
        return self.nope_dim + self.rope_dim;
    }

    /// Bytes of one int4 (q4_0-style) matrix: `rows * ceil(cols/QK) * (QK/2 + 4)`.
    /// Each block is 4 bytes of f32 scale followed by QK 4-bit weights.
    pub fn q4MatrixBytes(rows: usize, cols: usize) usize {
        const blocks_per_row = (cols + QK - 1) / QK;
        return rows * blocks_per_row * (QK / 2 + 4);
    }

    /// Bytes of one routed expert block on disk (gate + up + down, all int4).
    pub fn expertBytes(self: ModelConfig) usize {
        const h = self.hidden;
        const f = self.moe_ffn;
        return q4MatrixBytes(f, h) // gate  [moe_ffn x hidden]
        + q4MatrixBytes(f, h) // up    [moe_ffn x hidden]
        + q4MatrixBytes(h, f); // down  [hidden x moe_ffn]
    }

    /// Working-set bytes streamed per token: n_moe_layers * n_routed experts.
    pub fn perTokenExpertBytes(self: ModelConfig) usize {
        return self.n_moe_layers * self.n_routed * self.expertBytes();
    }

    pub fn totalRoutedExperts(self: ModelConfig) usize {
        return self.n_moe_layers * self.n_experts;
    }

    /// Ceilings on attacker-controlled dimensions (security issue #164). A
    /// manifest is just a file: without them, a checkpoint can name a model
    /// whose layout arithmetic overflows or whose allocation is absurd, and
    /// the first symptom is a trap or an OOM at load.
    pub const MAX_DIM = 1 << 20; // any single dimension
    pub const MAX_LAYERS = 1 << 12;
    pub const MAX_EXPERTS = 1 << 20; // per layer
    pub const MAX_TOTAL_EXPERTS = 1 << 24; // layers x experts

    pub fn validate(self: ModelConfig) !void {
        if (self.hidden % QK != 0) return error.HiddenNotBlockAligned;
        if (self.moe_ffn % QK != 0) return error.MoeFfnNotBlockAligned;
        if (self.dense_ffn % QK != 0) return error.DenseFfnNotBlockAligned;
        if (self.n_routed > self.n_experts) return error.RoutedExceedsExperts;
        if (self.headDim() == 0 or self.n_heads == 0) return error.BadAttention;

        // Bounded dimensions first, so every product below is safe to form.
        if (self.hidden == 0 or self.hidden > MAX_DIM) return error.DimOutOfRange;
        if (self.moe_ffn > MAX_DIM or self.dense_ffn > MAX_DIM) return error.DimOutOfRange;
        if (self.vocab_size == 0 or self.vocab_size > MAX_DIM) return error.DimOutOfRange;
        if (self.n_heads > MAX_DIM or self.headDim() > MAX_DIM) return error.DimOutOfRange;
        if (self.nope_dim > MAX_DIM or self.rope_dim > MAX_DIM or self.v_head_dim > MAX_DIM) return error.DimOutOfRange;
        if (self.n_dense_layers > MAX_LAYERS or self.n_moe_layers > MAX_LAYERS) return error.LayersOutOfRange;
        if (self.n_experts > MAX_EXPERTS) return error.ExpertsOutOfRange;

        // Checked products: a silent wrap here becomes an undersized
        // allocation indexed with the unwrapped value.
        const total = std.math.mul(usize, self.n_moe_layers, self.n_experts) catch return error.ExpertsOutOfRange;
        if (total > MAX_TOTAL_EXPERTS) return error.ExpertsOutOfRange;
        _ = std.math.mul(usize, self.hidden, self.moe_ffn) catch return error.DimOutOfRange;
        _ = std.math.mul(usize, self.hidden, self.vocab_size) catch return error.DimOutOfRange;
    }
};

/// The real GLM-5.2 shape (design inputs from CLAUDE.md). Needs real weights to
/// run; `expertBytes()` here is ~19 MB and `perTokenExpertBytes()` ~11.4 GB.
pub fn glmShape() ModelConfig {
    return .{
        .vocab_size = 151552,
        .hidden = 5120,
        .n_dense_layers = 3,
        .n_moe_layers = 75,
        .n_heads = 128,
        .nope_dim = 128,
        .rope_dim = 64,
        .v_head_dim = 128,
        .q_lora_rank = 1536,
        .kv_lora_rank = 512,
        .dense_ffn = 12288,
        .moe_ffn = 1536,
        .n_experts = 256,
        .n_routed = 8,
        .n_shared = 1,
        .max_seq_len = 8192,
        .rope_theta = 10000.0,
    };
}

/// A small, runnable shape for the v0 harness on commodity hardware. Preserves
/// GLM's structure (dense-then-MoE layers, top-k routing, 1 shared expert, MLA).
pub fn tinyShape() ModelConfig {
    return .{
        .vocab_size = 256, // byte-level tokenizer
        .hidden = 256,
        .n_dense_layers = 2,
        .n_moe_layers = 6,
        .n_heads = 4,
        .nope_dim = 32,
        .rope_dim = 32,
        .v_head_dim = 64,
        .q_lora_rank = 128,
        .kv_lora_rank = 96,
        .dense_ffn = 512,
        .moe_ffn = 512,
        .n_experts = 32,
        .n_routed = 4,
        .n_shared = 1,
        .max_seq_len = 512,
        .rope_theta = 10000.0,
    };
}

test "config sizing is self-consistent" {
    const c = tinyShape();
    try c.validate();
    try std.testing.expect(c.nLayers() == 8);
    try std.testing.expect(c.headDim() == 64);
    // expert bytes must be a multiple of a block (20 bytes) and > 0
    try std.testing.expect(c.expertBytes() > 0);
    try std.testing.expect(c.expertBytes() % (QK / 2 + 4) == 0);
}

test "config validation rejects out-of-range dimensions and overflowing products (issue #164)" {
    var cfg = tinyShape();
    try cfg.validate();

    // a dimension large enough that layout products would wrap
    var big = cfg;
    big.hidden = ModelConfig.MAX_DIM + QK;
    try std.testing.expectError(error.DimOutOfRange, big.validate());

    var many = cfg;
    many.n_moe_layers = ModelConfig.MAX_LAYERS + 1;
    try std.testing.expectError(error.LayersOutOfRange, many.validate());

    var experts = cfg;
    experts.n_experts = ModelConfig.MAX_EXPERTS + 1;
    try std.testing.expectError(error.ExpertsOutOfRange, experts.validate());
}
