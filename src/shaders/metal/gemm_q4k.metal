// Q4_K batched matrix-vector: prefill.
//
// This is the one regime where a GPU should win outright on unified memory, and
// the reason is arithmetic intensity rather than any property of the kernel. A
// single-token matvec reads a whole weight matrix to produce one output vector,
// so it moves far more bytes than it does flops and both processors run into
// the same memory bus. A batch of N tokens reads the *same* weight matrix and
// produces N output vectors, so every byte fetched does N times the work. Past
// some N the problem stops being bandwidth-bound and becomes compute-bound,
// which is what a GPU is built for.
//
// So the loop nest is ordered to make that reuse explicit: one SIMD group per
// output row, unpack each weight sub-block exactly once, and immediately dot it
// against all N activation vectors before moving on. The weights are read once
// per row for the whole batch, not once per row per token.
//
// Lane layout matches dmmv_q4k: four consecutive quant bytes per lane, with a
// branchless scale unpack because the sub-block index varies across the group.
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define Q4_K_BLOCK 144
#define MAX_BATCH 8

struct GemmDims {
    uint rows;
    uint cols;
    uint n; // batch size, <= MAX_BATCH
};

// 6-bit scale/min unpack. Branchless: `j` varies with the lane, so a real
// branch would make the SIMD group execute both sides. `s[j & 3]` is `s[j - 4]`
// whenever j >= 4 and a harmless in-bounds read otherwise.
static inline void scale_min_k4(uint j, device const uchar *s, thread uchar &sc, thread uchar &m) {
    const uchar a = s[j];
    const uchar b = s[j + 4];
    const uchar c = s[j & 3];
    const bool low = j < 4;
    sc = low ? (uchar)(a & 63) : (uchar)((b & 0xF) | ((c >> 6) << 4));
    m  = low ? (uchar)(b & 63) : (uchar)((b >> 4) | ((a >> 6) << 4));
}

kernel void gemm_q4k(
    device const uchar   *weights [[buffer(0)]],
    device const float   *xs      [[buffer(1)]], // n vectors of `cols`, contiguous
    device float         *out     [[buffer(2)]], // n vectors of `rows`, contiguous
    constant GemmDims    &dims    [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]])
{
    const uint row = tgid * nsg + sgid;
    if (row >= dims.rows) return;

    const uint blocks = dims.cols / QK_K;
    device const uchar *w = weights + (ulong)row * blocks * Q4_K_BLOCK;

    const uint h = lane >> 3;
    const uint o = (lane & 7) * 4;

    // One accumulator per batch element, held in registers for the whole row.
    float acc[MAX_BATCH];
    for (uint k = 0; k < MAX_BATCH; k++) acc[k] = 0.0f;

    const uint n = min(dims.n, (uint)MAX_BATCH);

    for (uint b = 0; b < blocks; b++) {
        device const uchar *blk = w + b * Q4_K_BLOCK;
        const float d    = (float)((device const half *)blk)[0];
        const float dmin = (float)((device const half *)blk)[1];
        device const uchar *scales = blk + 4;
        device const uchar *qs     = blk + 16;

        uchar sc0, mn0, sc1, mn1;
        scale_min_k4(2 * h + 0, scales, sc0, mn0);
        scale_min_k4(2 * h + 1, scales, sc1, mn1);

        // Unpacked once, then reused across the whole batch. This is the line
        // that makes prefill different from decode.
        const uchar4 q = (uchar4)(*(device const packed_uchar4 *)(qs + h * 32 + o));
        const float4 wlo = (float4)(q & 0x0F) * (d * (float)sc0) - (dmin * (float)mn0);
        const float4 whi = (float4)(q >> 4) * (d * (float)sc1) - (dmin * (float)mn1);

        for (uint k = 0; k < n; k++) {
            device const float *xb = xs + (ulong)k * dims.cols + b * QK_K;
            // packed_*: `xs` may be an offset into a wrapped host allocation,
            // and a misaligned aligned-vector load is a fault, not a slow path.
            const float4 xlo = (float4)(*(device const packed_float4 *)(xb + (2 * h + 0) * 32 + o));
            const float4 xhi = (float4)(*(device const packed_float4 *)(xb + (2 * h + 1) * 32 + o));
            acc[k] += dot(wlo, xlo) + dot(whi, xhi);
        }
    }

    for (uint k = 0; k < n; k++) {
        const float v = simd_sum(acc[k]);
        if (lane == 0) out[(ulong)k * dims.rows + row] = v;
    }
}
