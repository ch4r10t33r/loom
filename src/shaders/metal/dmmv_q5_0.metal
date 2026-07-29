// Q5_0 dequantize-multiply-matrix-vector.
//
// This is the type that actually blocks DeepSeek-V2-Lite Q4_K_M: its
// `ffn_down_exps` -- one of the three tensors in every routed expert -- is
// Q5_0 in some layers and Q8_0 in others. With no kernel for either,
// `moeFfnBlock` declined at `pipelineFor(e.down.ty)` and every MoE layer in
// the checkpoint fell back to the host, so the whole batched path was
// unreachable on the one model it was written for. (An earlier reading of the
// same checkpoint said Q5_1, which it contains none of.)
//
// Layout of one 32-value block (22 bytes):
//
//   d   [0,2)   half, scale
//   qh  [2,6)   u32, the fifth bit of each of the 32 values
//   qs  [6,22)  16 bytes, low nibbles: value i in the low nibble of byte i,
//               value i+16 in the high nibble
//
// A value is `d * (q - 16)` with q unsigned 0..31 -- symmetric around 16
// rather than affine, which is the one difference from Q5_1: there is no
// per-block minimum, but the -16 has to come off every element.
#include <metal_stdlib>
using namespace metal;

#define QK 32
#define Q5_0_BLOCK 22

struct Dims {
    uint rows;
    uint cols;
};

kernel void dmmv_q5_0(
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

    const uint blocks = dims.cols / QK;
    device const uchar *w = weights + (ulong)row * blocks * Q5_0_BLOCK;

    // One block per lane, striding by the SIMD width, as in dmmv_q5_1: with no
    // shared per-sub-block scales there is nothing to amortize by splitting a
    // block across lanes, and a lane that owns a whole block does strictly
    // less redundant work.
    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += QK) {
        device const uchar *blk = w + b * Q5_0_BLOCK;
        const float d = (float)((device const half *)blk)[0];
        // Byte-wise: the block is 22 bytes, so `blk + 2` is only 2-aligned and
        // a `uint` load there faults rather than running slowly.
        const uint qh = (uint)blk[2] | ((uint)blk[3] << 8) | ((uint)blk[4] << 16) | ((uint)blk[5] << 24);
        device const uchar *qs = blk + 6;
        device const float *xb = x + b * QK;

        float sum_qx = 0.0f; // sum(q * x)
        float sum_x = 0.0f;  // sum(x): carries the -16 offset for the block
        for (uint i = 0; i < 16; i++) {
            const uchar byte = qs[i];
            const float q_lo = (float)((byte & 0x0F) | (uchar)(((qh >> i) & 1u) << 4));
            const float q_hi = (float)((byte >> 4) | (uchar)(((qh >> (i + 16)) & 1u) << 4));
            const float x_lo = xb[i];
            const float x_hi = xb[i + 16];
            sum_qx += q_lo * x_lo + q_hi * x_hi;
            sum_x += x_lo + x_hi;
        }
        // d*(q-16) summed over the block separates into d*sum(q*x) - 16*d*sum(x),
        // so the offset never enters the per-element product.
        acc += d * (sum_qx - 16.0f * sum_x);
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
