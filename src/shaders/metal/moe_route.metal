// MoE routing: score the router logits, pick the top-k experts, emit their ids
// and gates — all on the device.
//
// This is `ggml_argsort_top_k`'s role in `build_moe_ffn`, and it is the last
// host step between a layer's attention and its FFN. While the host picks the
// experts, the two cannot share a command buffer however much else is
// recorded, because the ids have to come back before the expert matmuls can be
// encoded. With this and `dmmv_q4k_id.metal` the ids never leave the device.
//
// Mirrors `moe.route` exactly, including two things that are silent when got
// wrong:
//
//   - `bias` (DeepSeek-V3's noaux_tc trick) shifts the *selection* scores only.
//     The emitted gate comes from the unbiased probability. Reversing that
//     still produces text, just worse.
//   - the renormalization divisor is clamped to the smallest normal f16 rather
//     than tested against zero, which is what llama.cpp does, so an
//     all-but-zero gate vector scales the same way here as there.
//
// One threadgroup, and the selection is a single thread's serial scan. Top-k
// over 64 experts picking 6 is ~384 comparisons: parallelizing it would cost
// more in barriers than it saves, and a partial-sort bug here is expensive to
// find because the model still runs.
#include <metal_stdlib>
using namespace metal;

#define MAX_EXPERTS 256
#define MAX_SELECTED 16

struct RouteDims {
    uint n_expert;
    uint n_used;
    uint gating;      // 0 = softmax, 1 = sigmoid
    uint weights_norm;// 0 or 1
    uint has_bias;    // 0 or 1
    float weights_scale;
};

kernel void moe_route(
    device const float *logits [[buffer(0)]], // n_expert
    device const float *bias   [[buffer(1)]], // n_expert, read only if has_bias
    device uint        *ids    [[buffer(2)]], // n_used
    device float       *gates  [[buffer(3)]], // n_used
    constant RouteDims &d      [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scores[MAX_EXPERTS];
    threadgroup float choice[MAX_EXPERTS];

    if (d.n_expert > MAX_EXPERTS || d.n_used > MAX_SELECTED) return;

    if (d.gating == 1) {
        // sigmoid: independent per expert, so no reduction is needed
        for (uint e = tid; e < d.n_expert; e += 32) {
            scores[e] = 1.0f / (1.0f + exp(-logits[e]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        // softmax: max then normalize, both on one thread. n_expert is 64-256
        // and this runs once per layer, against an FFN that moves megabytes.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float mx = -INFINITY;
            for (uint e = 0; e < d.n_expert; e++) mx = max(mx, logits[e]);
            float sum = 0.0f;
            for (uint e = 0; e < d.n_expert; e++) {
                const float v = exp(logits[e] - mx);
                scores[e] = v;
                sum += v;
            }
            const float inv = 1.0f / sum;
            for (uint e = 0; e < d.n_expert; e++) scores[e] *= inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid != 0) return;

    for (uint e = 0; e < d.n_expert; e++) {
        choice[e] = scores[e] + (d.has_bias ? bias[e] : 0.0f);
    }

    // Selection, marking taken experts by driving their choice score to -inf
    // rather than keeping a separate used[] — same result, one array less.
    for (uint k = 0; k < d.n_used; k++) {
        uint best = 0;
        float best_v = -INFINITY;
        for (uint e = 0; e < d.n_expert; e++) {
            if (choice[e] > best_v) {
                best_v = choice[e];
                best = e;
            }
        }
        choice[best] = -INFINITY;
        ids[k] = best;
        gates[k] = scores[best]; // unbiased, deliberately
    }

    if (d.weights_norm) {
        float sum = 0.0f;
        for (uint k = 0; k < d.n_used; k++) sum += gates[k];
        const float denom = max(sum, 6.103515625e-5f); // smallest normal f16
        for (uint k = 0; k < d.n_used; k++) gates[k] /= denom;
    }
    if (d.weights_scale != 1.0f) {
        for (uint k = 0; k < d.n_used; k++) gates[k] *= d.weights_scale;
    }
}
