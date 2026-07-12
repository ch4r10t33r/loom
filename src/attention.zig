//! Multi-head Latent Attention (MLA), the GLM-5.2 attention.
//!
//! Query and key/value are produced through low-rank (LoRA) projections and RoPE
//! is applied only to a `rope_dim` slice of each head (partial RoPE). The KV
//! cache stores just the compressed latent `c_kv` (kv_lora_rank) plus the shared
//! `k_rope` (rope_dim) per position — ~kv_lora_rank+rope_dim floats/token, the
//! 57x-smaller KV that makes distributing experts (not KV) the right axis.
//!
//! Per step we recompute per-head k_nope and v from the cached latent (the
//! straightforward, correctness-first MLA; the "absorbed" fast path is a later
//! optimization). Scalar throughout.

const std = @import("std");
const model = @import("model.zig");
const tensor = @import("tensor.zig");
const ckpt = @import("checkpoint.zig");
const ModelConfig = model.ModelConfig;

pub const KVCache = struct {
    cfg: ModelConfig,
    c_kv: []f32, // [n_layers][max_seq][kv_lora_rank]
    k_rope: []f32, // [n_layers][max_seq][rope_dim]
    len: usize = 0, // positions filled so far

    pub fn init(gpa: std.mem.Allocator, cfg: ModelConfig) !KVCache {
        const nl = cfg.nLayers();
        return .{
            .cfg = cfg,
            .c_kv = try gpa.alloc(f32, nl * cfg.max_seq_len * cfg.kv_lora_rank),
            .k_rope = try gpa.alloc(f32, nl * cfg.max_seq_len * cfg.rope_dim),
        };
    }
    pub fn deinit(self: *KVCache, gpa: std.mem.Allocator) void {
        gpa.free(self.c_kv);
        gpa.free(self.k_rope);
    }

    fn cKvAt(self: *KVCache, layer: usize, pos: usize) []f32 {
        const r = self.cfg.kv_lora_rank;
        const base = (layer * self.cfg.max_seq_len + pos) * r;
        return self.c_kv[base .. base + r];
    }
    fn kRopeAt(self: *KVCache, layer: usize, pos: usize) []f32 {
        const r = self.cfg.rope_dim;
        const base = (layer * self.cfg.max_seq_len + pos) * r;
        return self.k_rope[base .. base + r];
    }
};

/// Compute attention for one token at `pos` (0-based) in `layer`. Writes the
/// hidden-sized result into `out`. `x` is the (already normalized) input.
pub fn forward(
    scratch: std.mem.Allocator,
    cfg: ModelConfig,
    lw: ckpt.LayerWeights,
    kv: *KVCache,
    layer: usize,
    pos: usize,
    x: []const f32,
    out: []f32,
) !void {
    const H = cfg.n_heads;
    const nope = cfg.nope_dim;
    const rope = cfg.rope_dim;
    const hd = cfg.headDim(); // nope + rope
    const vhd = cfg.v_head_dim;
    const qr = cfg.q_lora_rank;
    const kvr = cfg.kv_lora_rank;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    // --- query path: x -> q_a -> norm -> q_b -> [H*hd] ---
    const q_a = try scratch.alloc(f32, qr);
    tensor.matvec(q_a, lw.q_a_proj, x, qr, cfg.hidden);
    const q_a_n = try scratch.alloc(f32, qr);
    tensor.rmsnorm(q_a_n, q_a, lw.q_a_norm, 1e-6);
    const q = try scratch.alloc(f32, H * hd);
    tensor.matvec(q, lw.q_b_proj, q_a_n, H * hd, qr);

    // --- kv path: x -> [c_kv | k_rope], norm c_kv, rope k_rope, store in cache ---
    const kv_a = try scratch.alloc(f32, kvr + rope);
    tensor.matvec(kv_a, lw.kv_a_proj, x, kvr + rope, cfg.hidden);
    const c_kv_store = kv.cKvAt(layer, pos);
    tensor.rmsnorm(c_kv_store, kv_a[0..kvr], lw.kv_a_norm, 1e-6);
    const k_rope_store = kv.kRopeAt(layer, pos);
    @memcpy(k_rope_store, kv_a[kvr .. kvr + rope]);
    tensor.rope(k_rope_store, pos, cfg.rope_theta);

    // rope the per-head q_rope slice (second half of each head's hd)
    var h: usize = 0;
    while (h < H) : (h += 1) {
        const q_rope = q[h * hd + nope ..][0..rope];
        tensor.rope(q_rope, pos, cfg.rope_theta);
    }

    const seq = pos + 1;
    kv.len = @max(kv.len, seq);

    // per-head k_nope and v are recomputed from the cached latent per position.
    const kv_b_head = nope + vhd; // rows of kv_b_proj per head
    const k_nope = try scratch.alloc(f32, nope);
    const v_t = try scratch.alloc(f32, vhd);
    const scores = try scratch.alloc(f32, seq);
    const head_out = try scratch.alloc(f32, H * vhd);

    h = 0;
    while (h < H) : (h += 1) {
        const q_nope = q[h * hd ..][0..nope];
        const q_rope = q[h * hd + nope ..][0..rope];
        // kv_b rows for this head: [nope rows -> k_nope][vhd rows -> v], each over kvr
        const kb_base = h * kv_b_head;

        var t: usize = 0;
        while (t < seq) : (t += 1) {
            const c_kv_t = kv.cKvAt(layer, t);
            // k_nope = kv_b_proj[kb_base .. kb_base+nope] @ c_kv_t
            tensor.matvec(k_nope, lw.kv_b_proj[kb_base * kvr ..][0 .. nope * kvr], c_kv_t, nope, kvr);
            const k_rope_t = kv.kRopeAt(layer, t);
            var dot: f32 = 0;
            for (q_nope, k_nope) |a, b| dot += a * b;
            for (q_rope, k_rope_t) |a, b| dot += a * b;
            scores[t] = dot * scale;
        }
        tensor.softmax(scores[0..seq]);

        const ho = head_out[h * vhd ..][0..vhd];
        @memset(ho, 0);
        t = 0;
        while (t < seq) : (t += 1) {
            const c_kv_t = kv.cKvAt(layer, t);
            // v = kv_b_proj[kb_base+nope .. kb_base+nope+vhd] @ c_kv_t
            tensor.matvec(v_t, lw.kv_b_proj[(kb_base + nope) * kvr ..][0 .. vhd * kvr], c_kv_t, vhd, kvr);
            const w = scores[t];
            for (ho, v_t) |*o, vv| o.* += w * vv;
        }
    }

    // output projection: [H*vhd] -> hidden
    tensor.matvec(out, lw.o_proj, head_out, cfg.hidden, H * vhd);
}
