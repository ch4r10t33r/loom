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

// silu(a) * b over `slots` consecutive vectors of `n` floats each -- the MoE
// block's per-slot SwiGLU as one dispatch instead of one per expert. The
// slices are disjoint, so the only thing the split dispatches bought was
// encode time and a longer serial chain.
kernel void swiglu_slots(
    device float       *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    constant uint2     &d [[buffer(2)]], // .x = n per slot, .y = slots
    uint gid [[thread_position_in_grid]])
{
    if (gid >= d.x * d.y) return;
    const float g = a[gid];
    a[gid] = (g / (1.0f + exp(-g))) * b[gid];
}

// out = in, n floats. Exists for cache writes inside a recorded frame: the
// k_rope section of kv_a has to land in the device cache before the rope
// kernel rotates it there, and a host memcpy mid-frame is a synchronization.
kernel void copy_f32(
    device const float *in  [[buffer(0)]],
    device float       *out [[buffer(1)]],
    constant uint      &n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) out[gid] = in[gid];
}
