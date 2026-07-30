// Multi-head Latent Attention, one head per threadgroup.
//
// The MLA identity is what makes this different from `attn.metal`, and it is
// the reason the cache is small enough to keep on the device at all. Keys and
// values are never materialized:
//
//   score_t = q_nope . (W_k c_t) + q_rope . k_rope_t
//           = (W_k^T q_nope) . c_t + q_rope . k_rope_t
//
// The absorbed query `W_k^T q_nope` is computed once per head on the host side
// (a kvr-wide vector), so this kernel dots it straight against the compressed
// cache row. The value side stays compressed too:
//
//   o_latent = sum_t p_t c_t          (kvr wide, not v_head_dim wide)
//
// and `W_v` is applied afterwards as an ordinary matvec. So per position this
// touches kvr + rope floats rather than n_heads * (nope + v_head_dim), which
// for DeepSeek-V2-Lite is 576 against 16 * 320.
//
// Layout note: the compressed cache is shared by every head -- there is one
// c_kv row per position, not one per head -- so `c_cache` and `krope_cache`
// carry no head stride. Only the query side is per head.
#include <metal_stdlib>
using namespace metal;

// Scores live in threadgroup memory, so the context this kernel serves is
// bounded by it: 4096 floats = 16 KB, half of Metal's 32 KB per threadgroup.
// The caller checks this and declines rather than truncating a sequence.
#define MAX_SEQ 4096

struct MlaDims {
    uint n_heads;
    uint kvr;   // kv_lora_rank: width of a compressed cache row
    uint rope;  // rope_dim: width of a shared rope key
    uint seq;   // positions 0..seq-1 are valid
    // q_rope layout: floats between heads and floats before each head's rope
    // section. Gathered layout is (rope, 0); in-place device q is (kd, nope).
    uint qr_stride;
    uint qr_off;
    float scale;
};

kernel void mla_attn_head(
    device const float *q_absorbed [[buffer(0)]], // n_heads * kvr
    device const float *q_rope     [[buffer(1)]], // n_heads * rope
    device const float *c_cache    [[buffer(2)]], // [seq][kvr] for this layer
    device const float *krope_cache[[buffer(3)]], // [seq][rope] for this layer
    device float       *out        [[buffer(4)]], // n_heads * kvr, compressed
    constant MlaDims   &d          [[buffer(5)]],
    uint  hg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]],
    uint  nt   [[threads_per_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]])
{
    threadgroup float scores[MAX_SEQ];
    // One slot per SIMD group. simd_max/simd_sum first and threadgroup memory
    // only across groups: a hand-rolled tree over threadgroup memory drops
    // lanes when the group size is not a power of two, and does it silently.
    threadgroup float red[32];

    const uint h = hg;
    if (h >= d.n_heads) return;
    device const float *qa = q_absorbed + (ulong)h * d.kvr;
    device const float *qr = q_rope + (ulong)h * d.qr_stride + d.qr_off;

    // ---- scores --------------------------------------------------------------
    // Both halves of the MLA score, against one cache row each.
    for (uint t = tid; t < d.seq; t += nt) {
        device const float *ct = c_cache + (ulong)t * d.kvr;
        device const float *kt = krope_cache + (ulong)t * d.rope;
        float s = 0.0f;
        for (uint i = 0; i < d.kvr; i++) s += qa[i] * ct[i];
        for (uint i = 0; i < d.rope; i++) s += qr[i] * kt[i];
        scores[t] = s * d.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- softmax -------------------------------------------------------------
    // Max first. Subtracting it is not cosmetic: attention logits reach
    // magnitudes where a bare exp() overflows to inf and the head becomes NaN.
    float local_max = -INFINITY;
    for (uint t = tid; t < d.seq; t += nt) local_max = max(local_max, scores[t]);
    local_max = simd_max(local_max);
    if (lane == 0) red[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY;
    for (uint i = 0; i < nsg; i++) mx = max(mx, red[i]);

    float local_sum = 0.0f;
    for (uint t = tid; t < d.seq; t += nt) {
        const float e = exp(scores[t] - mx);
        scores[t] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) red[sgid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint i = 0; i < nsg; i++) sum += red[i];
    const float inv = 1.0f / sum;

    // ---- weighted sum, in compressed space ----------------------------------
    // One thread per compressed dimension, each walking the whole sequence, so
    // the cache reads are contiguous across the threadgroup at every step. The
    // result is o_latent; W_v is applied to it outside this kernel.
    for (uint i = tid; i < d.kvr; i += nt) {
        float acc = 0.0f;
        for (uint t = 0; t < d.seq; t++) acc += scores[t] * c_cache[(ulong)t * d.kvr + i];
        out[(ulong)h * d.kvr + i] = acc * inv;
    }
}
