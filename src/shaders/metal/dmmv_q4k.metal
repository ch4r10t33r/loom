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

// Rows per SIMD group. Two, so that one load of the activation vector feeds
// two output rows: x traffic per row halves, the per-block scale unpack is
// shared, and the two accumulator chains give the scheduler something to
// interleave. Taken from ZINC, whose kernel reaches 109 GB/s at 27 MB where the
// one-row-per-group version of this one managed 37.
// Rows a SIMD group covers. Overridable so the same source compiles as two
// pipelines: four rows wins where the weights are cache-resident, because what
// costs there is re-reading the activation vector per group rather than
// streaming weights, and two rows wins once the matrix is large enough that
// having more groups in flight matters more. Measured on an M5, Q4_K,
// 2048 columns:
//
//   rows     NR0=2      NR0=4
//   1408   33.2 GB/s  52.9 GB/s
//   2816   38.2       42.4
//   5632   62.1       56.5
//  65536  113.1      104.7
#ifndef NR0
#define NR0 4
#endif

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
    const uint first_row = (tgid * nsg + sgid) * NR0;
    if (first_row >= dims.rows) return;
    const bool have_r1 = (first_row + 1) < dims.rows;
    const bool have_r2 = (first_row + 2) < dims.rows;
    const bool have_r3 = (first_row + 3) < dims.rows;

    const uint blocks = dims.cols / QK_K;
    const uint row_bytes = blocks * Q4_K_BLOCK;

    // Each lane takes four *consecutive* quant bytes rather than one: a
    // byte-at-a-time version was perfectly coalesced but issued a quarter-width
    // load, so the loop ran four times as often and never had enough reads in
    // flight to cover their latency.
    //
    //   lane -> group h = lane/8, byte offset o = (lane%8)*4 within that group
    //
    // h is fixed per lane, so each row's two 6-bit scale/min pairs are unpacked
    // once per super-block rather than eight times.
    const uint h = lane >> 3;
    const uint o = (lane & 7) * 4;

    // A float2 rather than float[NR0]. An indexed array here is addressed
    // dynamically and the compiler puts it in thread-private memory; the two
    // accumulators then live on the stack instead of in registers, which costs
    // more than the two-row sharing gains. The same reasoning applies to the
    // scales below -- they stay device pointers rather than being unpacked into
    // a local uchar[12], which was measured 5-10% slower across every size for
    // exactly this reason.
    float4 acc = float4(0.0f);
    device const uchar *w0 = weights + (ulong)first_row * row_bytes;
    device const uchar *w1 = w0 + row_bytes;
    device const uchar *w2 = w1 + row_bytes;
    device const uchar *w3 = w2 + row_bytes;

    for (uint b = 0; b < blocks; b++) {
        device const float *xb = x + b * QK_K;
        // Loaded once and used by all NR0 rows -- the whole point of NR0.
        // At the expert-FFN shapes the weights are cache-resident, so what
        // costs is re-reading this vector per group, not streaming weights.
        // packed_*: `x` may be an offset into a wrapped host allocation, and a
        // misaligned aligned-vector load is a fault, not a slow path.
        const float4 xlo = (float4)(*(device const packed_float4 *)(xb + (2 * h + 0) * 32 + o));
        const float4 xhi = (float4)(*(device const packed_float4 *)(xb + (2 * h + 1) * 32 + o));
        // Q4_K is d*sc*q - dmin*m and the second term does not involve q at
        // all, so summing the activations once per block applies the whole
        // -dmin*m contribution as two scalars instead of a vector subtract per
        // element. Shared across both rows as well.
        const float sy_lo = xlo.x + xlo.y + xlo.z + xlo.w;
        const float sy_hi = xhi.x + xhi.y + xhi.z + xhi.w;

        {
            device const uchar *blk = w0 + b * Q4_K_BLOCK;
            const float d = (float)((device const half *)blk)[0];
            const float dmin = (float)((device const half *)blk)[1];
            uchar sc0, mn0, sc1, mn1;
            scale_min_k4(2 * h + 0, blk + 4, sc0, mn0);
            scale_min_k4(2 * h + 1, blk + 4, sc1, mn1);
            const uchar4 q = (uchar4)(*(device const packed_uchar4 *)(blk + 16 + h * 32 + o));
            acc.x += d * ((float)sc0 * dot((float4)(q & 0x0F), xlo) + (float)sc1 * dot((float4)(q >> 4), xhi)) -
                     dmin * ((float)mn0 * sy_lo + (float)mn1 * sy_hi);
        }
        if (have_r1) {
            device const uchar *blk = w1 + b * Q4_K_BLOCK;
            const float d = (float)((device const half *)blk)[0];
            const float dmin = (float)((device const half *)blk)[1];
            uchar sc0, mn0, sc1, mn1;
            scale_min_k4(2 * h + 0, blk + 4, sc0, mn0);
            scale_min_k4(2 * h + 1, blk + 4, sc1, mn1);
            const uchar4 q = (uchar4)(*(device const packed_uchar4 *)(blk + 16 + h * 32 + o));
            acc.y += d * ((float)sc0 * dot((float4)(q & 0x0F), xlo) + (float)sc1 * dot((float4)(q >> 4), xhi)) -
                     dmin * ((float)mn0 * sy_lo + (float)mn1 * sy_hi);
        }
        if (have_r2) {
            device const uchar *blk = w2 + b * Q4_K_BLOCK;
            const float d = (float)((device const half *)blk)[0];
            const float dmin = (float)((device const half *)blk)[1];
            uchar sc0, mn0, sc1, mn1;
            scale_min_k4(2 * h + 0, blk + 4, sc0, mn0);
            scale_min_k4(2 * h + 1, blk + 4, sc1, mn1);
            const uchar4 q = (uchar4)(*(device const packed_uchar4 *)(blk + 16 + h * 32 + o));
            acc.z += d * ((float)sc0 * dot((float4)(q & 0x0F), xlo) + (float)sc1 * dot((float4)(q >> 4), xhi)) -
                     dmin * ((float)mn0 * sy_lo + (float)mn1 * sy_hi);
        }
        if (have_r3) {
            device const uchar *blk = w3 + b * Q4_K_BLOCK;
            const float d = (float)((device const half *)blk)[0];
            const float dmin = (float)((device const half *)blk)[1];
            uchar sc0, mn0, sc1, mn1;
            scale_min_k4(2 * h + 0, blk + 4, sc0, mn0);
            scale_min_k4(2 * h + 1, blk + 4, sc1, mn1);
            const uchar4 q = (uchar4)(*(device const packed_uchar4 *)(blk + 16 + h * 32 + o));
            acc.w += d * ((float)sc0 * dot((float4)(q & 0x0F), xlo) + (float)sc1 * dot((float4)(q >> 4), xhi)) -
                     dmin * ((float)mn0 * sy_lo + (float)mn1 * sy_hi);
        }
    }

    const float v0 = simd_sum(acc.x);
    const float v1 = simd_sum(acc.y);
    const float v2 = simd_sum(acc.z);
    const float v3 = simd_sum(acc.w);
    if (lane == 0) {
        out[first_row] = v0;
        if (have_r1) out[first_row + 1] = v1;
        if (have_r2) out[first_row + 2] = v2;
        if (have_r3) out[first_row + 3] = v3;
    }
}
