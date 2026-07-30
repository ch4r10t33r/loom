// DeepSeek's NORM-style rope, with the YaRN branch: adjacent pairs (2i, 2i+1),
// frequency by pair index, and -- when yarn_factor > 1 -- per-pair mixing
// between extrapolated and interpolated angles on the ramp llama.cpp uses.
//
// A faithful port of `ropeApply` in deepseek.zig, and tested against it
// directly, because rope is the classic silent failure: get it wrong and
// attention becomes partially position-blind, which reads as slightly worse
// text rather than as an error.
//
// Rotates `n_vec` vectors in place, each `rope` floats wide, laid out at
// `stride` floats apart starting `offset` floats in -- so the q heads can be
// rotated where they live (stride kd, offset nope) without a host gather.
#include <metal_stdlib>
using namespace metal;

struct RopeDims {
    uint n_vec;
    uint rope;    // floats per vector; pairs = rope/2
    uint stride;  // floats between vectors
    uint offset;  // floats before the first vector's rope section
    uint pos;
    float base;
    float yarn_factor;   // <= 1 means plain
    float yarn_orig_ctx;
};

kernel void mla_rope(
    device float    *v [[buffer(0)]],
    constant RopeDims &d [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    const uint pairs = d.rope / 2;
    const uint vec = gid / pairs;
    const uint i = gid % pairs;
    if (vec >= d.n_vec) return;
    device float *p = v + (ulong)vec * d.stride + d.offset + 2 * i;

    const float dim = (float)d.rope;
    const float exponent = (float)(2 * i) / dim;
    const float freq = pow(d.base, -exponent);
    const float theta_extrap = (float)d.pos * freq;
    float theta = theta_extrap;
    if (d.yarn_factor > 1.0f) {
        const float two_pi = 2.0f * M_PI_F;
        const float low = max(0.0f, floor(dim * log(d.yarn_orig_ctx / (32.0f * two_pi)) / (2.0f * log(d.base))));
        const float high = min(dim - 1.0f, ceil(dim * log(d.yarn_orig_ctx / (1.0f * two_pi)) / (2.0f * log(d.base))));
        const float theta_interp = theta_extrap / d.yarn_factor;
        const float y = ((float)i - low) / max(0.001f, high - low);
        const float ramp_mix = 1.0f - clamp(y, 0.0f, 1.0f);
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
    }
    const float c = cos(theta);
    const float sn = sin(theta);
    const float a = p[0];
    const float b = p[1];
    p[0] = a * c - b * sn;
    p[1] = a * sn + b * c;
}
