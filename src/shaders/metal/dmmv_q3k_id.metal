// Q3_K matrix-vector for a *selected* expert, indexed on the device.
//
// Slot logic identical to dmmv_q4k_id.metal: one dispatch covers every
// selected expert, a SIMD group derives its slot from row / rows and reads
// plane ids[slot]. The per-row arithmetic is identical to dmmv_q3k.metal --
// deliberately, plane for plane, which is what its test asserts. See that
// file for the layout and lane-mapping commentary.
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define Q3_K_BLOCK 110

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

static inline uint ld_u32(device const uchar *p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

// One 6-bit scale (biased by 32) from the unpacked aux words. `j` varies per
// lane, so the word pick is a ternary chain rather than an indexed array --
// an indexed uint[4] would be addressed dynamically and spill to
// thread-private memory (see the accumulator note in dmmv_q4k.metal).
static inline float q3k_scale(uint a0, uint a1, uint a2, uint a3, uint j) {
    const uint sel = j >> 2;
    const uint w = sel == 0 ? a0 : sel == 1 ? a1 : sel == 2 ? a2 : a3;
    return (float)(int)((w >> ((j & 3) * 8)) & 0xFF) - 32.0f;
}

kernel void dmmv_q3k_id(
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
    const uint row_bytes = blocks * Q3_K_BLOCK;

    const uint bo = lane * 2;
    const uint hh = bo >> 5;
    const uint bi = bo & 31;
    const uint jbase = hh * 8 + (bi >> 4);
    // hmask bit for window (hh, s) is 1 << (hh*4 + s); the byte index is the
    // within-half position bi, shared by both halves.
    const uchar m0 = (uchar)(1u << (hh * 4 + 0));
    const uchar m1 = (uchar)(1u << (hh * 4 + 1));
    const uchar m2 = (uchar)(1u << (hh * 4 + 2));
    const uchar m3 = (uchar)(1u << (hh * 4 + 3));

    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;

    device const uchar *plane = weights + (ulong)ids[slot] * dims.plane_stride;
    device const float *xs = x + (ulong)slot * dims.x_stride;

    float4 acc = float4(0.0f);
    device const uchar *w0 = plane + (ulong)first_row * row_bytes;
    device const uchar *w1 = w0 + row_bytes;
    device const uchar *w2 = w1 + row_bytes;
    device const uchar *w3 = w2 + row_bytes;

    for (uint b = 0; b < blocks; b++) {
        device const float *xb = xs + b * QK_K + hh * 128 + bi;
        const float2 x0 = (float2)(*(device const packed_float2 *)(xb + 0 * 32));
        const float2 x1 = (float2)(*(device const packed_float2 *)(xb + 1 * 32));
        const float2 x2 = (float2)(*(device const packed_float2 *)(xb + 2 * 32));
        const float2 x3 = (float2)(*(device const packed_float2 *)(xb + 3 * 32));

#define ROW(W, A)                                                                      \
        {                                                                              \
            device const uchar *blk = W + b * Q3_K_BLOCK;                              \
            const uchar2 hm = (uchar2)(*(device const packed_uchar2 *)(blk + bi));     \
            const uchar2 q = (uchar2)(*(device const packed_uchar2 *)(blk + 32 + bo)); \
            const uint u0 = ld_u32(blk + 96);                                          \
            const uint u1 = ld_u32(blk + 100);                                         \
            const uint u2 = ld_u32(blk + 104);                                         \
            const uint a0 = (u0 & kmask2) | (((u2 >> 0) & kmask1) << 4);               \
            const uint a1 = (u1 & kmask2) | (((u2 >> 2) & kmask1) << 4);               \
            const uint a2 = ((u0 >> 4) & kmask2) | (((u2 >> 4) & kmask1) << 4);        \
            const uint a3 = ((u1 >> 4) & kmask2) | (((u2 >> 6) & kmask1) << 4);        \
            const float d = (float)((device const half *)(blk + 108))[0];              \
            const float2 w0v = float2(q.x & 3, q.y & 3) -                              \
                float2((hm.x & m0) ? 0.0f : 4.0f, (hm.y & m0) ? 0.0f : 4.0f);          \
            const float2 w1v = float2((q.x >> 2) & 3, (q.y >> 2) & 3) -                \
                float2((hm.x & m1) ? 0.0f : 4.0f, (hm.y & m1) ? 0.0f : 4.0f);          \
            const float2 w2v = float2((q.x >> 4) & 3, (q.y >> 4) & 3) -                \
                float2((hm.x & m2) ? 0.0f : 4.0f, (hm.y & m2) ? 0.0f : 4.0f);          \
            const float2 w3v = float2(q.x >> 6, q.y >> 6) -                            \
                float2((hm.x & m3) ? 0.0f : 4.0f, (hm.y & m3) ? 0.0f : 4.0f);          \
            A += d * (q3k_scale(a0, a1, a2, a3, jbase + 0) * dot(w0v, x0) +            \
                      q3k_scale(a0, a1, a2, a3, jbase + 2) * dot(w1v, x1) +            \
                      q3k_scale(a0, a1, a2, a3, jbase + 4) * dot(w2v, x2) +            \
                      q3k_scale(a0, a1, a2, a3, jbase + 6) * dot(w3v, x3));            \
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
