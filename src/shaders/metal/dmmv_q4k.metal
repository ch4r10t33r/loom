// Q4_K dequantize-multiply-matrix-vector.
//
// One SIMD group (32 lanes) per output row, each lane taking whole 32-value
// sub-blocks, reduced with simd_sum.
//
// The two shapes that were tried first and are worth not repeating:
//
//   one thread per row  -- adjacent lanes then read addresses a whole row
//     apart (1152 bytes for a 2048-wide tensor), so every access is
//     uncoalesced and the memory system sees a scatter. Correct, and slower
//     than eight CPU threads even with dispatch overhead removed.
//
//   one threadgroup per row with a tree reduction over threadgroup memory --
//     a tree reduction whose group size is not a power of two silently drops
//     lanes, and the output stays plausible. simd_sum has no such hazard: it
//     is one instruction over a fixed 32-lane group, and needs no threadgroup
//     memory or barriers at all.
//
// Lanes 2k and 2k+1 read the same 32 packed bytes (low and high nibbles), so
// each pair shares a cache line and consecutive pairs walk consecutive bytes.
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define Q4_K_BLOCK 144
#define SIMD_W 32

struct Dims {
    uint rows;
    uint cols;
};

// 6-bit scale/min unpack, identical to the CPU scaleMinK4.
static inline void scale_min_k4(uint j, device const uchar *s, thread uchar &sc, thread uchar &m) {
    if (j < 4) {
        sc = s[j] & 63;
        m  = s[j + 4] & 63;
    } else {
        sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        m  = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

kernel void dmmv_q4k(
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

    const uint blocks = dims.cols / QK_K;      // super-blocks in this row
    const uint nsub   = blocks * 8;            // 32-value sub-blocks
    device const uchar *w = weights + (ulong)row * blocks * Q4_K_BLOCK;

    float acc = 0.0f;
    for (uint s = lane; s < nsub; s += SIMD_W) {
        const uint b = s >> 3;                 // super-block
        const uint j = s & 7;                  // sub-block within it
        device const uchar *blk = w + b * Q4_K_BLOCK;
        const float d    = (float)((device const half *)blk)[0];
        const float dmin = (float)((device const half *)blk)[1];
        uchar sc, mn;
        scale_min_k4(j, blk + 4, sc, mn);
        device const uchar *src = blk + 16 + (j >> 1) * 32;
        device const float *xb  = x + b * QK_K + j * 32;
        const bool high = (j & 1) != 0;

        float sum = 0.0f, xsum = 0.0f;
        for (uint i = 0; i < 32; i++) {
            const uchar q = high ? (src[i] >> 4) : (src[i] & 0x0F);
            const float xv = xb[i];
            sum  += (float)q * xv;
            xsum += xv;
        }
        acc += d * (float)sc * sum - dmin * (float)mn * xsum;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
