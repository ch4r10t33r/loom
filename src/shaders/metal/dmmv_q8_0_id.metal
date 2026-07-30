// Q8_0 matvec for a selected expert, indexed on the device.
//
// The id wrapper `dmmv_q4k_id.metal` puts around `dmmv_q4k`, applied to
// Q8_0. Needed because `ffn_down_exps` is Q8_0 in half of
// DeepSeek-V2-Lite's layers: without it the down projection keeps host
// pointers, the selected ids still have to come back, and a layer's attention
// and its FFN still cannot share a command buffer -- which is the whole point
// of routing on the device.
//
// One row per SIMD group, as in `dmmv_q8_0.metal`, and the per-block
// arithmetic is identical to it on purpose: the two must agree bit for bit,
// which is what the test asserts plane for plane.
#include <metal_stdlib>
using namespace metal;

#define QK 32
#define BLOCK 34

struct IdDims {
    uint rows;
    uint cols;
    uint n_used;
    uint plane_stride;
};

kernel void dmmv_q8_0_id(
    device const uchar *weights [[buffer(0)]],
    device const float *x       [[buffer(1)]],
    device float       *out     [[buffer(2)]],
    device const uint  *ids     [[buffer(3)]],
    constant IdDims    &dims    [[buffer(4)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]])
{
    const uint gr = tgid * nsg + sgid;
    const uint slot = gr / dims.rows;
    if (slot >= dims.n_used) return;
    const uint row = gr - slot * dims.rows;

    const uint blocks = dims.cols / QK;
    device const uchar *plane = weights + (ulong)ids[slot] * dims.plane_stride;
    device const uchar *w = plane + (ulong)row * blocks * BLOCK;

    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += QK) {
        device const uchar *blk = w + b * BLOCK;
        device const float *xb = x + b * QK;
        const float d = (float)((device const half *)blk)[0];
        device const char *qs = (device const char *)(blk + 2);
        float sum = 0.0f;
        for (uint i = 0; i < QK; i++) sum += (float)qs[i] * xb[i];
        acc += d * sum;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[(ulong)slot * dims.rows + row] = acc;
}
