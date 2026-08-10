// Q2_K dequantize-multiply-matrix-vector.
//
// Same shape as dmmv_q4k.metal: one SIMD group per NR0 output rows, lanes
// arranged so that at any instant the group reads consecutive bytes. Q2_K
// packs FOUR 2-bit values per byte (shifts 0/2/4/6), where shift s of byte
// half*32+i feeds value half*128 + s*32 + i -- so one byte read feeds four
// scaled multiply-adds against four different 32-value x windows, and the
// per-16 scale/min pair for each lands at scales[half*8 + s*2 + (i>>4)].
//
// Lane mapping: lane owns TWO consecutive qs bytes (32 lanes x 2 B = the
// whole 64-byte qs array, one coalesced transaction), giving float2 math
// throughout. A lane's byte pair never straddles a 16-value scale group
// (offsets are even), so each shift needs exactly one scale byte.
//
// Layout (84 bytes): scales[16] | qs[64] | d:half | dmin:half. The min term
// never involves q, so per-window activation sums apply the whole
// -dmin*mn contribution as scalars, exactly as in dmmv_q4k.metal.
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define Q2_K_BLOCK 84

#ifndef NR0
#define NR0 4
#endif

struct Dims {
    uint rows;
    uint cols;
};

kernel void dmmv_q2k(
    device const uchar *weights [[buffer(0)]],
    device const float *x       [[buffer(1)]],
    device float       *out     [[buffer(2)]],
    constant Dims      &dims    [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]])
{
    const uint first_row = (tgid * nsg + sgid) * NR0;
    if (first_row >= dims.rows) return;

    const bool have_r1 = (first_row + 1) < dims.rows;
    const bool have_r2 = (first_row + 2) < dims.rows;
    const bool have_r3 = (first_row + 3) < dims.rows;

    const uint blocks = dims.cols / QK_K;
    const uint row_bytes = blocks * Q2_K_BLOCK;

    // Two consecutive qs bytes per lane: half hh, byte index bi within it.
    const uint bo = lane * 2;
    const uint hh = bo >> 5;
    const uint bi = bo & 31;
    // Per-16 scale group for shift s: hh*8 + s*2 + (bi>>4).
    const uint jbase = hh * 8 + (bi >> 4);

    float4 acc = float4(0.0f);
    device const uchar *w0 = weights + (ulong)first_row * row_bytes;
    device const uchar *w1 = w0 + row_bytes;
    device const uchar *w2 = w1 + row_bytes;
    device const uchar *w3 = w2 + row_bytes;

    for (uint b = 0; b < blocks; b++) {
        device const float *xb = x + b * QK_K + hh * 128 + bi;
        // The four x windows this lane's byte pair feeds, loaded once and
        // shared by all NR0 rows; their sums carry the min terms.
        const float2 x0 = (float2)(*(device const packed_float2 *)(xb + 0 * 32));
        const float2 x1 = (float2)(*(device const packed_float2 *)(xb + 1 * 32));
        const float2 x2 = (float2)(*(device const packed_float2 *)(xb + 2 * 32));
        const float2 x3 = (float2)(*(device const packed_float2 *)(xb + 3 * 32));
        const float sy0 = x0.x + x0.y;
        const float sy1 = x1.x + x1.y;
        const float sy2 = x2.x + x2.y;
        const float sy3 = x3.x + x3.y;

#define ROW(W, A)                                                                    \
        {                                                                            \
            device const uchar *blk = W + b * Q2_K_BLOCK;                            \
            device const uchar *sc = blk;                                            \
            const float d = (float)((device const half *)(blk + 80))[0];             \
            const float dmin = (float)((device const half *)(blk + 80))[1];          \
            const uchar2 q = (uchar2)(*(device const packed_uchar2 *)(blk + 16 + bo)); \
            const uchar s0 = sc[jbase + 0];                                          \
            const uchar s1 = sc[jbase + 2];                                          \
            const uchar s2 = sc[jbase + 4];                                          \
            const uchar s3 = sc[jbase + 6];                                          \
            const float2 c0 = float2(q.x & 3, q.y & 3);                              \
            const float2 c1 = float2((q.x >> 2) & 3, (q.y >> 2) & 3);                \
            const float2 c2 = float2((q.x >> 4) & 3, (q.y >> 4) & 3);                \
            const float2 c3 = float2(q.x >> 6, q.y >> 6);                            \
            const float sd = (float)(s0 & 0xF) * dot(c0, x0) +                        \
                             (float)(s1 & 0xF) * dot(c1, x1) +                        \
                             (float)(s2 & 0xF) * dot(c2, x2) +                        \
                             (float)(s3 & 0xF) * dot(c3, x3);                         \
            const float sm = (float)(s0 >> 4) * sy0 + (float)(s1 >> 4) * sy1 +        \
                             (float)(s2 >> 4) * sy2 + (float)(s3 >> 4) * sy3;         \
            A += d * sd - dmin * sm;                                                 \
        }

        ROW(w0, acc.x)
        if (have_r1) ROW(w1, acc.y)
        if (have_r2) ROW(w2, acc.z)
        if (have_r3) ROW(w3, acc.w)
#undef ROW
    }

    const float v0 = simd_sum(acc.x);
    const float v1 = simd_sum(acc.y);
    const float v2 = simd_sum(acc.z);
    const float v3 = simd_sum(acc.w);
    if (lane == 0) {
        out[first_row] = v0;
        if (have_r1) out[first_row + 1] = v1;
        if (have_r2) out[first_row + 2] = v2;
        if (have_r3) out[first_row + 3] = v3;
    }
}
