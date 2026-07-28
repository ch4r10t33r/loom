// Q4_K dequantize-multiply-matrix-vector.
//
// One thread per output row. The obvious alternative -- one threadgroup per
// row, with the group reducing partial sums -- was tried first and was wrong:
// a tree reduction over a group size that is not a power of two silently drops
// lanes, and the result still looks like plausible numbers. There is little to
// gain from it either, since a row has only cols/256 super-blocks (eight, for
// a 2048-wide tensor) and rows are plentiful, so parallelism across rows is
// already ample.
//
// Mirrors the CPU kernel value for value; that is how it is validated.
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define Q4_K_BLOCK 144

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
    uint row [[thread_position_in_grid]])
{
    if (row >= dims.rows) return;

    const uint blocks = dims.cols / QK_K;
    device const uchar *w = weights + (ulong)row * blocks * Q4_K_BLOCK;

    float acc = 0.0f;
    for (uint b = 0; b < blocks; b++) {
        device const uchar *blk = w + b * Q4_K_BLOCK;
        const float d    = (float)((device const half *)blk)[0];
        const float dmin = (float)((device const half *)blk)[1];
        device const uchar *scales = blk + 4;
        device const uchar *qs     = blk + 16;
        device const float *xb     = x + b * QK_K;

        for (uint j = 0; j < 8; j++) {
            uchar sc, mn;
            scale_min_k4(j, scales, sc, mn);
            device const uchar *src = qs + (j / 2) * 32;
            const bool high = (j & 1) != 0;
            float sum = 0.0f, xsum = 0.0f;
            for (uint i = 0; i < 32; i++) {
                const uchar q = high ? (src[i] >> 4) : (src[i] & 0x0F);
                const float xv = xb[j * 32 + i];
                sum  += (float)q * xv;
                xsum += xv;
            }
            acc += d * (float)sc * sum - dmin * (float)mn * xsum;
        }
    }
    out[row] = acc;
}
