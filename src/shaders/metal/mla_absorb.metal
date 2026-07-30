// q_abs = W_k^T q_nope, per head.
//
// The absorption identity is what lets MLA attention read the compressed cache
// directly (see mla_attn.metal). Computing it was the last piece of a layer
// still running on the host: a loop that dequantized one `kv_b` row at a time
// and axpy'd it, which forced a synchronization in the middle of every layer
// whatever else was recorded around it.
//
// `W_k` arrives pre-dequantized to f16 and device-resident: the same trade as
// f32 -- no per-element K-quant unpack in the transposed access -- at half the
// bandwidth, which at ~113 MB per token as f32 was the largest single
// reducible read outside the experts themselves. f16 rounding of a weight
// already quantized to ~4.5 bits is noise against the quantization itself.
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

// One SIMD group per output element, the nope-sum split across lanes.
//
// The first version gave each *head* a threadgroup -- sixteen threadgroups on
// a device that wants hundreds, every thread crawling tens of kilobytes
// serially. The work is n_heads * kvr independent dot products over nope
// terms; gridding one SIMD group per output makes it 8,192 groups on the real
// model, and the column reads within a group are contiguous across lanes.
kernel void mla_absorb(
    device const half  *wk  [[buffer(0)]], // [n_heads * stride][kvr], f16
    device const float *q   [[buffer(1)]],
    device float       *out [[buffer(2)]], // [n_heads][kvr]
    constant AbsorbDims &d  [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  nsg  [[simdgroups_per_threadgroup]])
{
    const uint g = tgid * nsg + sgid;
    const uint h = g / d.kvr;
    if (h >= d.n_heads) return;
    const uint i = g - h * d.kvr;

    device const half *wh = wk + (ulong)h * d.stride * d.kvr;
    device const float *qh = q + (ulong)h * d.q_stride + d.q_off;

    float acc = 0.0f;
    for (uint r = lane; r < d.nope; r += 32) acc += qh[r] * (float)wh[(ulong)r * d.kvr + i];
    acc = simd_sum(acc);
    if (lane == 0) out[(ulong)h * d.kvr + i] = acc;
}
