// Q5_1 dequantize-multiply-matrix-vector.
//
// A missing quant kernel does not cost the speed of that one tensor, it costs
// every operation that would have shared its command buffer: the MoE block
// declines the whole layer if any of an expert's three tensors has no
// pipeline, and the expert FFN is ~40% of a token.
//
// This kernel was written believing DeepSeek-V2-Lite Q4_K_M stored
// `ffn_down_exps` as Q5_1. Reading the file's tensor types says otherwise --
// it is Q5_0 and Q8_0, and contains no Q5_1 at all -- so this one covers other
// checkpoints and `dmmv_q5_0.metal` is what unblocked that model. The habit
// worth keeping is reading the checkpoint rather than the quant mix's name.
//
// Layout of one 32-value block (24 bytes):
//
//   d   [0,2)   half, scale
//   m   [2,4)   half, minimum
//   qh  [4,8)   u32, the fifth bit of each of the 32 values
//   qs  [8,24)  16 bytes, low nibbles: value i in the low nibble of byte i,
//               value i+16 in the high nibble
//
// A value is `d*q + m` with q unsigned 0..31 — affine rather than symmetric,
// so unlike Q4_K there is no per-sub-block scale to unpack and no bias term to
// separate out. That makes this the simplest of the family.
#include <metal_stdlib>
using namespace metal;

#define QK 32
#define Q5_1_BLOCK 24

struct Dims {
    uint rows;
    uint cols;
};

kernel void dmmv_q5_1(
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
    device const uchar *w = weights + (ulong)row * blocks * Q5_1_BLOCK;

    // One block per lane, striding by the SIMD width: the 32 lanes cover 32
    // consecutive blocks, so each step touches a contiguous 768-byte run.
    // Splitting a block across lanes was tried first and buys nothing here —
    // unlike Q4_K there are no shared per-sub-block scales to amortise, so a
    // lane that owns a whole block does strictly less redundant work.
    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += QK) {
        device const uchar *blk = w + b * Q5_1_BLOCK;
        const float d = (float)((device const half *)blk)[0];
        const float m = (float)((device const half *)blk)[1];
        // A plain aligned load is safe: blocks are 24 bytes, so `blk + 4` is a
        // multiple of 4 for every block and every row. (MSL has no
        // `packed_uint` — only the vector types have packed forms.)
        const uint qh = ((device const uint *)(blk + 4))[0];
        device const uchar *qs = blk + 8;
        device const float *xb = x + b * QK;

        float sum_qx = 0.0f; // sum(q * x)
        float sum_x = 0.0f;  // sum(x): the affine minimum applies to every lane
        for (uint i = 0; i < 16; i++) {
            const uchar byte = qs[i];
            const float q_lo = (float)((byte & 0x0F) | (uchar)(((qh >> i) & 1u) << 4));
            const float q_hi = (float)((byte >> 4) | (uchar)(((qh >> (i + 16)) & 1u) << 4));
            const float x_lo = xb[i];
            const float x_hi = xb[i + 16];
            sum_qx += q_lo * x_lo + q_hi * x_hi;
            sum_x += x_lo + x_hi;
        }
        // d*q + m summed over the block separates: the minimum never enters
        // the per-element product.
        acc += d * sum_qx + m * sum_x;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
