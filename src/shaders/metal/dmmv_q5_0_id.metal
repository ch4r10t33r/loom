// Q5_0 matvec for a selected expert, indexed on the device.
//
// The id wrapper `dmmv_q4k_id.metal` puts around `dmmv_q4k`, applied to
// Q5_0. Needed because `ffn_down_exps` is Q5_0 in half of
// DeepSeek-V2-Lite's layers: without it the down projection keeps host
// pointers, the selected ids still have to come back, and a layer's attention
// and its FFN still cannot share a command buffer -- which is the whole point
// of routing on the device.
//
// One row per SIMD group, as in `dmmv_q5_0.metal`, and the per-block
// arithmetic is identical to it on purpose: the two must agree bit for bit,
// which is what the test asserts plane for plane.
#include <metal_stdlib>
using namespace metal;

#define QK 32
#define BLOCK 22

struct IdDims {
    uint rows;
    uint cols;
    uint n_used;
    uint plane_stride;
    // Floats between consecutive slots' activation vectors. Zero means one
    // vector shared by every slot, which is the gate/up case -- all experts
    // read the same normed input. The down projection cannot share: each
    // slot's input is its own expert's SwiGLU output, so it passes the ffn
    // width here and slot s reads x + s*x_stride.
    uint x_stride;
};

kernel void dmmv_q5_0_id(
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
    device const float *xs = x + (ulong)slot * dims.x_stride;
    device const uchar *w = plane + (ulong)row * blocks * BLOCK;

    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += QK) {
        device const uchar *blk = w + b * BLOCK;
        device const float *xb = xs + b * QK;
        const float d = (float)((device const half *)blk)[0];
        const uint qh = (uint)blk[2] | ((uint)blk[3] << 8) | ((uint)blk[4] << 16) | ((uint)blk[5] << 24);
        device const uchar *qs = blk + 6;
        float sum_qx = 0.0f;
        float sum_x = 0.0f;
        for (uint i = 0; i < 16; i++) {
            const uchar byte = qs[i];
            const float q_lo = (float)((byte & 0x0F) | (uchar)(((qh >> i) & 1u) << 4));
            const float q_hi = (float)((byte >> 4) | (uchar)(((qh >> (i + 16)) & 1u) << 4));
            const float x_lo = xb[i];
            const float x_hi = xb[i + 16];
            sum_qx += q_lo * x_lo + q_hi * x_hi;
            sum_x += x_lo + x_hi;
        }
        acc += d * (sum_qx - 16.0f * sum_x);
    }

    acc = simd_sum(acc);
    if (lane == 0) out[(ulong)slot * dims.rows + row] = acc;
}
