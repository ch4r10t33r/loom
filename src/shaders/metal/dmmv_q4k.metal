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
//
// Branchless. With one lane per byte every lane had the same `j` and the
// `j < 4` test was uniform across the SIMD group; now that a lane owns four
// consecutive bytes, `j` varies with the lane and a real branch would make
// the group execute both sides. `s[j & 3]` is `s[j - 4]` whenever j >= 4 and
// a harmless in-bounds read otherwise.
static inline void scale_min_k4(uint j, device const uchar *s, thread uchar &sc, thread uchar &m) {
    const uchar a = s[j];
    const uchar b = s[j + 4];
    const uchar c = s[j & 3];
    const bool low = j < 4;
    sc = low ? (uchar)(a & 63) : (uchar)((b & 0xF) | ((c >> 6) << 4));
    m  = low ? (uchar)(b & 63) : (uchar)((b >> 4) | ((a >> 6) << 4));
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

    // Each lane takes four *consecutive* quant bytes instead of one. The 32
    // lanes still cover the same 128 bytes of a super-block, but in one step
    // rather than four, and each step is a 4-byte load rather than a 1-byte
    // one. That is what the earlier version left on the table: it was
    // perfectly coalesced but issued a quarter-width load, so the inner loop
    // ran four times as often with four times the address arithmetic and four
    // times the scale unpacking, and never had enough loads in flight to
    // cover their own latency. A pure streaming-read kernel reaches ~110 GB/s
    // on an M5; the byte-at-a-time version of this one reached 61.
    //
    //   lane -> group h = lane/8, byte offset o = (lane%8)*4 within that group
    //
    // h is fixed per lane, so the two 6-bit scale/min pairs are unpacked once
    // per super-block instead of eight times.
    const uint h = lane >> 3;
    const uint o = (lane & 7) * 4;

    // Two things were tried here and are not kept, both measured on an M5 at
    // cols=2048: unrolling by two on independent accumulators (+6% at 131k
    // rows, -30% between 5k and 32k), and hoisting this body into a
    // `static inline` helper (-30% at 16k). The loop body stays written out.
    float acc = 0.0f;
    for (uint b = 0; b < blocks; b++) {
        device const uchar *blk = w + b * Q4_K_BLOCK;
        const float d    = (float)((device const half *)blk)[0];
        const float dmin = (float)((device const half *)blk)[1];
        device const uchar *scales = blk + 4;
        device const uchar *qs     = blk + 16;
        device const float *xb     = x + b * QK_K;

        uchar sc0, mn0, sc1, mn1;
        scale_min_k4(2 * h + 0, scales, sc0, mn0);
        scale_min_k4(2 * h + 1, scales, sc1, mn1);

        // packed_* rather than the aligned vector types: `x` may be an offset
        // into a wrapped host allocation, and a misaligned float4 load is a
        // fault, not a slow path.
        const uchar4 q   = (uchar4)(*(device const packed_uchar4 *)(qs + h * 32 + o));
        const float4 xlo = (float4)(*(device const packed_float4 *)(xb + (2 * h + 0) * 32 + o));
        const float4 xhi = (float4)(*(device const packed_float4 *)(xb + (2 * h + 1) * 32 + o));

        const float dlo = d * (float)sc0, blo = dmin * (float)mn0;
        const float dhi = d * (float)sc1, bhi = dmin * (float)mn1;

        acc += dot(dlo * (float4)(q & 0x0F) - blo, xlo);
        acc += dot(dhi * (float4)(q >> 4)   - bhi, xhi);
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
