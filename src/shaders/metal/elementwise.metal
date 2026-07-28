// Elementwise and norm kernels.
//
// These cost almost nothing to run and exist only so activations can stay in
// device memory between matmuls. Leaving RMSNorm on the host would force a
// round trip between every pair of matvecs, and a round trip is ~262 us
// against a matvec kernel of ~18 us -- the cheap op is the one that must not
// be stranded on the wrong side of the bus.
#include <metal_stdlib>
using namespace metal;

struct NormDims { uint n; float eps; };

// One threadgroup, one SIMD group: these vectors are a few thousand elements.
kernel void rmsnorm(
    device const float *x    [[buffer(0)]],
    device const float *w    [[buffer(1)]],
    device float       *out  [[buffer(2)]],
    constant NormDims  &d    [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]])
{
    float ss = 0.0f;
    for (uint i = lane; i < d.n; i += 32) ss += x[i] * x[i];
    ss = simd_sum(ss);
    const float inv = rsqrt(ss / (float)d.n + d.eps);
    for (uint i = lane; i < d.n; i += 32) out[i] = x[i] * inv * w[i];
}

struct Len { uint n; };

// SwiGLU: silu(gate) * up, elementwise.
kernel void swiglu(
    device const float *gate [[buffer(0)]],
    device const float *up   [[buffer(1)]],
    device float       *out  [[buffer(2)]],
    constant Len       &d    [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= d.n) return;
    const float g = gate[i];
    out[i] = (g / (1.0f + exp(-g))) * up[i];
}

// Residual add, in place on `acc`.
kernel void add_inplace(
    device float       *acc [[buffer(0)]],
    device const float *v   [[buffer(1)]],
    constant Len       &d   [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= d.n) return;
    acc[i] += v[i];
}
