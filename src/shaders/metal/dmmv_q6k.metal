// Q6_K dequantize-multiply-matrix-vector.
//
// Needed before the fused FFN block is usable on anything real: llama.cpp's
// `Q4_K_M` mixture puts `ffn_down` in Q6_K while gate and up stay Q4_K, so a
// block that only accepts Q4_K declines every actual checkpoint.
//
// Layout of one 256-value super-block (210 bytes):
//
//   ql      [0,128)   low 4 bits of each value, two values per byte
//   qh      [128,192) high 2 bits of each value, four values per byte
//   scales  [192,208) sixteen int8 scales, one per 16 values
//   d       [208,210) half, the super-block scale
//
// It is organised as two 128-value halves; each half holds four 32-value runs
// whose low nibbles come from two 32-byte spans of `ql` and whose high bits all
// come from one 32-byte span of `qh` at shifts 0, 2, 4, 6.
//
// The trap worth naming: **a 32-value run spans two scale groups.** Scales are
// per 16 values, so lanes below position 16 in a run use a different scale from
// those above it. Folding a run under one scale is silently wrong -- the output
// stays plausible and every Q6_K tensor is quietly degraded. The CPU kernel
// carries the same warning.
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define Q6_K_BLOCK 210

struct Dims {
    uint rows;
    uint cols;
};

kernel void dmmv_q6k(
    device const uchar *weights [[buffer(0)]],
    device const float *x       [[buffer(1)]],
    device float       *out     [[buffer(2)]],
    constant Dims      &dims    [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]])
{
    const uint row = tgid * nsg + sgid;
    if (row >= dims.rows) return;

    const uint blocks = dims.cols / QK_K;
    device const uchar *w = weights + (ulong)row * blocks * Q6_K_BLOCK;

    // Four runs per half, eight lanes per run, four consecutive values each --
    // the same four-bytes-per-lane shape the Q4_K kernel uses, for the same
    // reason: a one-byte load leaves most of the memory pipeline idle.
    const uint k = lane >> 3;        // which run within the half
    const uint i = (lane & 7) * 4;   // value offset within the run
    // Everything below is a select rather than a branch: `k` varies across the
    // SIMD group, so a real branch would make the group execute both sides.
    const uint ql_off = (k & 1) * 32;
    const uint hshift = k * 2;
    const uint sc_i = k * 2 + (i >> 4); // the run's two scale groups split at 16

    float acc = 0.0f;
    for (uint b = 0; b < blocks; b++) {
        device const uchar *blk = w + b * Q6_K_BLOCK;
        device const uchar *ql_all = blk;
        device const uchar *qh_all = blk + QK_K / 2;
        device const char  *sc_all = (device const char *)(blk + QK_K / 2 + QK_K / 4);
        const float d = (float)((device const half *)(blk + QK_K / 2 + QK_K / 4 + QK_K / 16))[0];
        device const float *xb = x + b * QK_K;

        for (uint n = 0; n < 2; n++) {
            device const uchar *ql = ql_all + n * 64;
            device const uchar *qh = qh_all + n * 32;

            // packed_*: `x` may be an offset into a wrapped host allocation, and
            // a misaligned aligned-vector load is a fault, not a slow path.
            const uchar4 lo = (uchar4)(*(device const packed_uchar4 *)(ql + ql_off + i));
            const uchar4 hi = (uchar4)(*(device const packed_uchar4 *)(qh + i));
            const uchar4 base = (k < 2) ? (lo & 0x0F) : (lo >> 4);
            const uchar4 q = base | (uchar4)(((hi >> hshift) & 3) << 4);

            const float4 xv = (float4)(*(device const packed_float4 *)(xb + (n * 4 + k) * 32 + i));
            const float sc = d * (float)sc_all[n * 8 + sc_i];
            acc += sc * dot((float4)q - 32.0f, xv);
        }
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
