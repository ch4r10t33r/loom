// Q2_K matrix-vector for a *selected* expert, indexed on the device.
//
// Slot logic identical to dmmv_q4k_id.metal: one dispatch covers every
// selected expert, a SIMD group derives its slot from row / rows and reads
// plane ids[slot]. The per-row arithmetic is identical to dmmv_q2k.metal --
// deliberately, plane for plane, which is what its test asserts. See that
// file for the layout and lane-mapping commentary.
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define Q2_K_BLOCK 84

#ifndef NR0
#define NR0 4
#endif

struct IdDims {
    uint rows;         // rows per expert
    uint cols;
    uint n_used;       // selected experts in this dispatch
    uint plane_stride; // bytes between expert planes
    uint x_stride;     // floats between slots' activation vectors; 0 = shared
};

kernel void dmmv_q2k_id(
    device const uchar *weights [[buffer(0)]], // the whole expert tensor
    device const float *x       [[buffer(1)]],
    device float       *out     [[buffer(2)]], // n_used * rows
    device const uint  *ids     [[buffer(3)]], // n_used expert indices
    constant IdDims    &dims    [[buffer(4)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]])
{
    const uint gr = (tgid * nsg + sgid) * NR0; // global first row
    const uint slot = gr / dims.rows;
    if (slot >= dims.n_used) return;
    const uint first_row = gr - slot * dims.rows;
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

    device const uchar *plane = weights + (ulong)ids[slot] * dims.plane_stride;
    device const float *xs = x + (ulong)slot * dims.x_stride;

    float4 acc = float4(0.0f);
    device const uchar *w0 = plane + (ulong)first_row * row_bytes;
    device const uchar *w1 = w0 + row_bytes;
    device const uchar *w2 = w1 + row_bytes;
    device const uchar *w3 = w2 + row_bytes;

    for (uint b = 0; b < blocks; b++) {
        device const float *xb = xs + b * QK_K + hh * 128 + bi;
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
        device float *o_slot = out + (ulong)slot * dims.rows;
        o_slot[first_row] = v0;
        if (have_r1) o_slot[first_row + 1] = v1;
        if (have_r2) o_slot[first_row + 2] = v2;
        if (have_r3) o_slot[first_row + 3] = v3;
    }
}
