// acc += alpha * v  — the MoE weighted-sum step.
//
// Trivial arithmetic, and it has to be a kernel for the same reason RoPE did:
// it sits between two expert FFNs, so doing it on the host would end the
// command buffer and put every expert back on its own submission. A MoE layer
// activating seven experts is seven FFN blocks plus seven of these, and the
// whole point is that they cost one submission between them rather than
// fourteen.
#include <metal_stdlib>
using namespace metal;

struct AccDims {
    uint n;
    float alpha;
};

kernel void scaled_add(
    device float       *acc [[buffer(0)]],
    device const float *v   [[buffer(1)]],
    constant AccDims   &d   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= d.n) return;
    acc[gid] += d.alpha * v[gid];
}

// acc = 0. Cheaper than a host memset plus upload, and again avoids ending the
// buffer between layers.
kernel void zero_fill(
    device float     *acc [[buffer(0)]],
    constant AccDims &d   [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= d.n) return;
    acc[gid] = 0.0f;
}

// acc = sum_e weight[e] * out[e*dim + i], for every expert at once.
//
// Replaces one `scaled_add` per expert. Those all write the same accumulator,
// so they need a barrier between each -- and a barrier is a pipeline drain, not
// a fence on a buffer. A MoE layer ran ~30 of them; this makes the whole
// accumulation one dispatch and no barriers.
#define MAX_E 16

struct ReduceDims {
    uint n;      // experts
    uint dim;
    float w[MAX_E];
};

kernel void moe_reduce(
    device float       *acc  [[buffer(0)]],
    device const float *outs [[buffer(1)]],
    constant ReduceDims &d   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= d.dim) return;
    float a = 0.0f;
    for (uint e = 0; e < d.n; e++) a += d.w[e] * outs[e * d.dim + gid];
    acc[gid] = a;
}
