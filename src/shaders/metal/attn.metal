// Grouped-query attention over a device-resident KV cache.
//
// One threadgroup per query head, and the whole head -- scores, softmax, and
// the weighted sum over V -- in a single kernel. Splitting those into three
// dispatches would be the obvious decomposition and the wrong one: each stage
// feeds the next, so they would need barriers between dispatches, and on this
// backend the unit that costs real time is the submission rather than the
// kernel. Keeping a head whole means the scores never leave threadgroup memory.
//
// The cache is indexed [t][kvd] with the layer's base offset applied by the
// host, so this kernel sees one layer's slice. Query head h reads KV head
// h / (n_heads / n_kv_heads) -- that division is the whole of GQA.
#include <metal_stdlib>
using namespace metal;

// Scores live in threadgroup memory, so the context this kernel can serve is
// bounded by it: 4096 floats = 16 KB, half of Metal's 32 KB per threadgroup.
// The host declines longer sequences rather than silently truncating
// attention, which would be a correctness bug that still produces
// fluent-looking text.
//
// 2048 was too low to be useful: a node's default context cap is 4096, and
// Mistral-7B advertises 32768, so every real model fell straight back to the
// host path with no indication of why.
#define MAX_SEQ 4096

struct AttnDims {
    uint n_heads;
    uint n_kv_heads;
    uint hd;   // head dim
    uint seq;  // positions 0..seq-1 are valid
    uint kvd;  // n_kv_heads * hd, the cache row stride
    float scale;
};

kernel void attn_head(
    device const float *q       [[buffer(0)]], // n_heads * hd
    device const float *k_cache [[buffer(1)]], // [seq][kvd] for this layer
    device const float *v_cache [[buffer(2)]],
    device float       *out     [[buffer(3)]], // n_heads * hd
    constant AttnDims  &d       [[buffer(4)]],
    uint  hg   [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]],
    uint  nt   [[threads_per_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]])
{
    threadgroup float scores[MAX_SEQ];
    // One slot per SIMD group. The two-level reduction below is simd_max /
    // simd_sum first and threadgroup memory only across groups: a hand-rolled
    // tree over threadgroup memory silently drops lanes when the group size is
    // not a power of two, which is a bug that survives every smoke test.
    threadgroup float red[32];

    const uint h = hg;
    if (h >= d.n_heads) return;
    const uint kvh = h / (d.n_heads / d.n_kv_heads);
    device const float *qh = q + (ulong)h * d.hd;

    // ---- scores: one dot product per cached position -------------------------
    for (uint t = tid; t < d.seq; t += nt) {
        device const float *kt = k_cache + (ulong)t * d.kvd + (ulong)kvh * d.hd;
        float s = 0.0f;
        for (uint i = 0; i < d.hd; i++) s += qh[i] * kt[i];
        scores[t] = s * d.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- softmax -------------------------------------------------------------
    // Max first, then exponentiate against it. Subtracting the max is not
    // cosmetic: attention logits routinely reach magnitudes where a bare exp()
    // overflows to inf and the whole head becomes NaN.
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

    // ---- weighted sum over V -------------------------------------------------
    // One thread per head dimension: each walks the whole sequence, so the
    // reads of v_cache are contiguous across the threadgroup at every step.
    for (uint i = tid; i < d.hd; i += nt) {
        float acc = 0.0f;
        for (uint t = 0; t < d.seq; t++) {
            device const float *vt = v_cache + (ulong)t * d.kvd + (ulong)kvh * d.hd;
            acc += scores[t] * vt[i];
        }
        out[(ulong)h * d.hd + i] = acc * inv;
    }
}
