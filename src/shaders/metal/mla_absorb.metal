// q_abs = W_k^T q_nope, per head.
//
// The absorption identity is what lets MLA attention read the compressed cache
// directly (see mla_attn.metal). Computing it was the last piece of a layer
// still running on the host: a loop that dequantized one `kv_b` row at a time
// and axpy'd it, which forced a synchronization in the middle of every layer
// whatever else was recorded around it.
//
// `W_k` arrives pre-dequantized to f32 and device-resident. That is a
// deliberate trade: `kv_b` is 8 MB a layer against ~45 MB of experts, it is
// read on every layer of every token, and holding it as f32 turns this from a
// per-element K-quant unpack -- which needs block lookup per output column --
// into a plain transposed matvec.
//
// Transposed because the tensor is row-major over kvr: row r of W_k is
// contiguous, and the output is a sum over rows weighted by q_nope[r]. So each
// thread owns output columns and strides down the rows, which keeps every read
// contiguous across the threadgroup.
#include <metal_stdlib>
using namespace metal;

struct AbsorbDims {
    uint n_heads;
    uint nope;     // rows of W_k per head
    uint kvr;      // columns: the compressed width
    uint stride;   // rows per head in the source tensor, (nope + v_head_dim)
    // Floats between heads in `q`, and floats before each head's nope section.
    // The engine's gathered layout is (nope, 0); on-device q is laid out as
    // [n_heads][kd] straight from the projection, which is (kd, 0) -- reading
    // it in place is what removes the host gather.
    uint q_stride;
    uint q_off;
};

kernel void mla_absorb(
    device const float *wk  [[buffer(0)]], // [n_heads * stride][kvr], f32
    device const float *q   [[buffer(1)]], // [n_heads][nope]
    device float       *out [[buffer(2)]], // [n_heads][kvr]
    constant AbsorbDims &d  [[buffer(3)]],
    uint hg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt  [[threads_per_threadgroup]])
{
    const uint h = hg;
    if (h >= d.n_heads) return;

    // This head's W_k block: the first `nope` rows of its slice. The remaining
    // v_head_dim rows are W_v, which this kernel must not touch -- hence the
    // explicit stride rather than assuming the two are the same size.
    device const float *wh = wk + (ulong)h * d.stride * d.kvr;
    device const float *qh = q + (ulong)h * d.q_stride + d.q_off;

    for (uint i = tid; i < d.kvr; i += nt) {
        float acc = 0.0f;
        for (uint r = 0; r < d.nope; r++) acc += qh[r] * wh[(ulong)r * d.kvr + i];
        out[(ulong)h * d.kvr + i] = acc;
    }
}
