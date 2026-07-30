// Q4_K matrix-vector for a *selected* expert, indexed on the device.
//
// The shape ggml uses: `ggml_mul_mat_id(ctx0, w, cur, ids)` treats the expert
// tensor as one 3D block and picks a plane with an id tensor that never
// returns to the host. loom's `ffn_gate_exps` is already that layout -- planes
// of `ne1 * rowBytes(ty, ne0)` bytes -- so the only thing missing was a kernel
// that takes the id from a buffer rather than the plane's address from the
// host.
//
// Why it matters is not the dispatch count inside the MoE block, which already
// batches. It is that as long as the *host* chooses the experts, there is a
// read-back between attention and the FFN, and the two cannot share a command
// buffer. Device-side ids are what let a whole layer become one submission.
//
// One dispatch covers every selected expert: the grid is `n_used * rows`, and
// a SIMD group derives its slot from `row / rows`, so slot s reads plane
// `ids[s]` and writes rows `[s*rows, (s+1)*rows)` of the output.
//
// The per-row arithmetic is identical to dmmv_q4k.metal -- four consecutive
// quant bytes per lane, branchless 6-bit scale unpack -- and deliberately so:
// this kernel must agree with it plane for plane, which is what its test
// asserts.
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define Q4_K_BLOCK 144

#ifndef NR0
#define NR0 4
#endif

struct IdDims {
    uint rows;         // rows per expert
    uint cols;
    uint n_used;       // selected experts in this dispatch
    uint plane_stride; // bytes between expert planes
    // Floats between consecutive slots' activation vectors. Zero means one
    // vector shared by every slot -- the gate/up case, where all experts read
    // the same normed input. The down projection cannot share: each slot's
    // input is its own expert's SwiGLU output, so it passes the ffn width and
    // slot s reads x + s*x_stride.
    uint x_stride;
};

// 6-bit scale/min unpack. Branchless because `j` varies with the lane and a
// real branch would make the SIMD group execute both sides.
static inline void scale_min_k4(uint j, device const uchar *s, thread uchar &sc, thread uchar &m) {
    const uchar a = s[j];
    const uchar b = s[j + 4];
    const uchar c = s[j & 3];
    const bool low = j < 4;
    sc = low ? (uchar)(a & 63) : (uchar)((b & 0xF) | ((c >> 6) << 4));
    m  = low ? (uchar)(b & 63) : (uchar)((b >> 4) | ((a >> 6) << 4));
}

kernel void dmmv_q4k_id(
    device const uchar *weights [[buffer(0)]], // the whole expert tensor
    device const float *x       [[buffer(1)]], // one activation vector
    device float       *out     [[buffer(2)]], // n_used * rows
    device const uint  *ids     [[buffer(3)]], // n_used expert indices
    constant IdDims    &dims    [[buffer(4)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]])
{
    const uint gr = (tgid * nsg + sgid) * NR0; // global first row
    const uint slot = gr / dims.rows;
    if (slot >= dims.n_used) return;
    const uint first_row = gr - slot * dims.rows;
    if (first_row >= dims.rows) return;

    // A group must not straddle two experts: rows per expert is a multiple of
    // NR0 for every real width (1408 and 2816 both divide by 4), and a shape
    // where it is not would silently mix two planes into one output row.
    const bool have_r1 = (first_row + 1) < dims.rows;
    const bool have_r2 = (first_row + 2) < dims.rows;
    const bool have_r3 = (first_row + 3) < dims.rows;

    const uint blocks = dims.cols / QK_K;
    const uint row_bytes = blocks * Q4_K_BLOCK;
    device const uchar *plane = weights + (ulong)ids[slot] * dims.plane_stride;
    device const float *xs = x + (ulong)slot * dims.x_stride;

    const uint h = lane >> 3;
    const uint o = (lane & 7) * 4;

    float4 acc = float4(0.0f);
    device const uchar *w0 = plane + (ulong)first_row * row_bytes;
    device const uchar *w1 = w0 + row_bytes;
    device const uchar *w2 = w1 + row_bytes;
    device const uchar *w3 = w2 + row_bytes;

    for (uint b = 0; b < blocks; b++) {
        device const float *xb = xs + b * QK_K;
        const float4 xlo = (float4)(*(device const packed_float4 *)(xb + (2 * h + 0) * 32 + o));
        const float4 xhi = (float4)(*(device const packed_float4 *)(xb + (2 * h + 1) * 32 + o));
        const float sy_lo = xlo.x + xlo.y + xlo.z + xlo.w;
        const float sy_hi = xhi.x + xhi.y + xhi.z + xhi.w;

#define ROW(W, A)                                                                              \
        {                                                                                      \
            device const uchar *blk = W + b * Q4_K_BLOCK;                                      \
            const float d = (float)((device const half *)blk)[0];                              \
            const float dmin = (float)((device const half *)blk)[1];                           \
            uchar sc0, mn0, sc1, mn1;                                                          \
            scale_min_k4(2 * h + 0, blk + 4, sc0, mn0);                                        \
            scale_min_k4(2 * h + 1, blk + 4, sc1, mn1);                                        \
            const uchar4 q = (uchar4)(*(device const packed_uchar4 *)(blk + 16 + h * 32 + o)); \
            A += d * ((float)sc0 * dot((float4)(q & 0x0F), xlo) +                               \
                      (float)sc1 * dot((float4)(q >> 4), xhi)) -                                \
                 dmin * ((float)mn0 * sy_lo + (float)mn1 * sy_hi);                              \
        }

        ROW(w0, acc.x)
        if (have_r1) ROW(w1, acc.y)
        if (have_r2) ROW(w2, acc.z)
        if (have_r3) ROW(w3, acc.w)
#undef ROW
    }

    const float v0 = simd_sum(acc.x);
    const float v1 = simd_sum(acc.y);
    const float v2 = simd_sum(acc.z);
    const float v3 = simd_sum(acc.w);
    if (lane == 0) {
        device float *o_slot = out + (ulong)slot * dims.rows;
        o_slot[first_row] = v0;
        if (have_r1) o_slot[first_row + 1] = v1;
        if (have_r2) o_slot[first_row + 2] = v2;
        if (have_r3) o_slot[first_row + 3] = v3;
    }
}
