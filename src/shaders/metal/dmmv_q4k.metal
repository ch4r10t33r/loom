// Q4_K dequantize-multiply-matrix-vector.
//
// One SIMD group per output row. Within a super-block, lane `i` handles
// element `i` of each 32-byte group, so at any instant the 32 lanes read 32
// *consecutive* bytes — one coalesced transaction instead of a scatter.
//
// Three shapes have been tried here; the two that failed are worth not
// repeating.
//
//   one thread per row -- adjacent lanes read addresses a whole row apart
//     (1152 bytes for a 2048-wide tensor). Correct, and slower than eight CPU
//     threads even with all dispatch overhead removed.
//
//   one SIMD group per row, each lane taking whole 32-value sub-blocks -- a
//     lane then walks 32 consecutive bytes on its own while its neighbours
//     walk unrelated regions, so the group still issues a scatter every step,
//     and the inner loop is scalar.
//
// Each byte carries two values: the low nibble belongs to sub-block 2h and the
// high nibble to sub-block 2h+1, which have different scales. So one load
// feeds two scaled multiply-adds, and the byte is never read twice.
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

        for (uint h = 0; h < 4; h++) {
            uchar sc0, mn0, sc1, mn1;
            scale_min_k4(2 * h + 0, scales, sc0, mn0);
            scale_min_k4(2 * h + 1, scales, sc1, mn1);

            // 32 lanes read 32 consecutive bytes
            const uchar byte = qs[h * 32 + lane];
            const float xlo = xb[(2 * h + 0) * 32 + lane];
            const float xhi = xb[(2 * h + 1) * 32 + lane];

            acc += (d * (float)sc0 * (float)(byte & 0x0F) - dmin * (float)mn0) * xlo;
            acc += (d * (float)sc1 * (float)(byte >> 4)   - dmin * (float)mn1) * xhi;
        }
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
