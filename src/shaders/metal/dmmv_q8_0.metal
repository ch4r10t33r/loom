// Q8_0 dequantize-multiply-matrix-vector.
//
// The other half of what blocks DeepSeek-V2-Lite Q4_K_M: `ffn_down_exps` is
// Q8_0 wherever it is not Q5_0, and `ffn_down` in the dense prefix is Q8_0
// throughout. Both are 2.2 G elements of the checkpoint.
//
// Layout of one 32-value block (34 bytes):
//
//   d   [0,2)   half, scale
//   qs  [2,34)  32 signed bytes
//
// A value is `d * q` with q already signed, so there is no unpacking at all --
// this is the simplest kernel in the family, and the one whose arithmetic
// intensity is worst: 34 bytes read per 32 multiply-adds. It is bandwidth
// bound end to end, which is the regime where the GPU's advantage over the
// CPU is smallest; it earns its place by not forcing the layer back to the
// host rather than by being fast in isolation.
#include <metal_stdlib>
using namespace metal;

#define QK 32
#define Q8_0_BLOCK 34

struct Dims {
    uint rows;
    uint cols;
};

kernel void dmmv_q8_0(
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
    device const uchar *w = weights + (ulong)row * blocks * Q8_0_BLOCK;

    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += QK) {
        device const uchar *blk = w + b * Q8_0_BLOCK;
        const float d = (float)((device const half *)blk)[0];
        // `blk + 2` is 2-aligned at best across a 34-byte stride, so the quants
        // are read a byte at a time rather than as a vector.
        device const char *qs = (device const char *)(blk + 2);
        device const float *xb = x + b * QK;

        float sum = 0.0f;
        for (uint i = 0; i < QK; i++) sum += (float)qs[i] * xb[i];
        acc += d * sum;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
