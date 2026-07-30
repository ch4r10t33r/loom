// F32 matrix-vector. Exists for the small f32 tensors a checkpoint keeps
// unquantized -- the MoE router (n_expert rows over dim) is the one on the
// token path, and without this kernel the fused layer declines at exactly
// that dispatch, silently, which is how every other gate in this series
// presented too.
#include <metal_stdlib>
using namespace metal;

struct Dims {
    uint rows;
    uint cols;
};

kernel void dmmv_f32(
    device const float *weights [[buffer(0)]],
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
    device const float *w = weights + (ulong)row * dims.cols;
    float acc = 0.0f;
    for (uint i = lane; i < dims.cols; i += 32) acc += w[i] * x[i];
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
