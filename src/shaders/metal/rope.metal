// RoPE, and the KV-cache write that follows it.
//
// These two exist for one reason: they are the only operations in a GQA layer
// that had no Metal kernel, and a single host-side operation in the middle of a
// token forces the command buffer to be committed and split. Measured on this
// machine, a command buffer costs ~262 us while a matvec kernel is ~18 us, and
// ZINC's own build flips from 53 to 11.8 tok/s when its per-dispatch timing
// probe forces a commit between dispatches. So a kernel here is not worth
// having because RoPE is slow -- it is trivial -- but because its absence
// prices the whole token at per-operation submission.
#include <metal_stdlib>
using namespace metal;

struct RopeDims {
    uint n_heads;  // heads in this tensor (q: n_heads, k: n_kv_heads)
    uint hd;       // head dim, the stride between heads
    uint rope_dim; // leading dims of each head that rotate; the rest pass through
    uint pos;      // token position
    float base;    // rope frequency base
    uint neox;     // 0 = NORM (adjacent pairs), 1 = NEOX (split-half pairs)
};

// theta for pair index i, matching the CPU's ropeTheta: base^(-i/rope_dim).
static inline float2 rope_theta(uint i, uint rope_dim, uint pos, float base) {
    const float exponent = -(float)i / (float)rope_dim;
    const float freq = pow(base, exponent);
    const float angle = (float)pos * freq;
    return float2(cos(angle), sin(angle));
}

// One thread per rotated *pair*. NORM pairs (2i, 2i+1); NEOX pairs
// (i, i + rope_dim/2). Swapping the two produces fluent-looking, wrong output,
// which is why the style is pinned per architecture rather than guessed -- the
// CPU carries the same warning.
kernel void rope_apply(
    device float      *vec  [[buffer(0)]],
    constant RopeDims &d    [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    const uint pairs = d.rope_dim / 2;
    const uint total = d.n_heads * pairs;
    if (gid >= total) return;

    const uint h = gid / pairs;
    const uint p = gid % pairs;
    device float *head = vec + (ulong)h * d.hd;

    uint ia, ib, ti;
    if (d.neox != 0) {
        ia = p;
        ib = p + pairs;
        ti = 2 * p;
    } else {
        ia = 2 * p;
        ib = 2 * p + 1;
        ti = 2 * p;
    }
    const float2 t = rope_theta(ti, d.rope_dim, d.pos, d.base);
    const float a = head[ia];
    const float b = head[ib];
    head[ia] = a * t.x - b * t.y;
    head[ib] = a * t.y + b * t.x;
}

struct KvWriteDims {
    uint kvd; // n_kv_heads * head_dim: the elements to copy
    uint row; // destination element offset = (li * ctx + pos) * kvd
};

// Append this token's k and v into the device cache. A copy, but it has to
// happen on the device: k and v were just produced by a dispatch, and pulling
// them to the host to memcpy would end the command buffer.
kernel void kv_write(
    device const float   *k   [[buffer(0)]],
    device const float   *v   [[buffer(1)]],
    device float         *kc  [[buffer(2)]],
    device float         *vc  [[buffer(3)]],
    constant KvWriteDims &d   [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= d.kvd) return;
    kc[(ulong)d.row + gid] = k[gid];
    vc[(ulong)d.row + gid] = v[gid];
}
