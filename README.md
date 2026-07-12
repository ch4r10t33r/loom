# Loom v0 — single-node expert-streaming MoE inference

Zig implementation of the **v0** phase from [`CLAUDE.md`](CLAUDE.md): a GLM-5.2-shaped
Mixture-of-Experts transformer whose routed experts are **streamed from a
content-addressed on-disk store** through a tiered cache (pinned hot-set → LRU →
page-cache/pread), with usage measurement (STATS), pinning (PIN), a disk profiler
(iobench), and free integrity checks from a Merkle-rooted manifest.

Compute is entirely node-local; the only thing that "arrives" mid-forward is a
19 MB int4 expert block, materialized directly (no coding on the hot path —
principle 7).

## Toolchain

Targets **Zig 0.16.0**. With [anyzig](https://github.com/marler8997/anyzig) the
pinned version resolves automatically from `build.zig.zon`.

```sh
zig build                      # debug binary -> zig-out/bin/loom
zig build -Doptimize=ReleaseFast
zig build test                 # unit tests (quant roundtrip, LRU, router, merkle, layout, ...)
```

## Quick start

```sh
# 1. generate a small synthetic checkpoint (runs on a laptop)
loom gen /tmp/ckpt

# 2. inspect it: config, sizes, Merkle-root + digest verification
loom info /tmp/ckpt

# 3. run inference; log tok/s, cache hit-rate, RSS; record usage for pinning
loom run /tmp/ckpt --prompt "Loom weaves" --max-tokens 32 --ram-gb 1 --stats /tmp/usage.stats

# 4. re-run pinning the hot set measured in step 3 (STATS -> PIN)
loom run /tmp/ckpt --prompt "Loom weaves" --max-tokens 32 --ram-gb 1 --pin-gb 0.05 --stats /tmp/usage.stats

# 5. disk profile: the parallel 19 MB random reads the engine issues on a miss
loom iobench /tmp/ckpt/experts.blob --threads 8 --block-mb 1 --reads 64
```

`run` also reads env overrides: `MODEL`, `RAM_BUDGET_GB`, `PIN_GB`, `MAX_TOKENS`,
`TEMP`, `SEED`, `STATS`.

## What's implemented

| Module | Role |
|---|---|
| `model.zig` | `ModelConfig`; GLM-5.2 shape + a runnable `tiny` shape; expert/working-set sizing |
| `quant.zig` | int4 (q4_0-style, 32 weights/scale) quantize + fused `matvecQ4` — the expert kernel |
| `tensor.zig` | RMSNorm, softmax, SwiGLU, dense matvec, partial RoPE |
| `attention.zig` | **MLA**: q/kv-LoRA, partial RoPE, compressed-latent KV cache (~kv_lora+rope floats/token) |
| `moe.zig` | DeepSeek-V3 sigmoid router (top-k routed + always-on shared expert); streamed int4 expert FFN |
| `checkpoint.zig` | on-disk format (manifest + dense.blob + experts.blob), content-addressed + Merkle-rooted, dedup |
| `expert_cache.zig` | **the v0 core**: pinned hot-set → LRU → pread, per-expert usage, per-block digest verify |
| `forward.zig` | one node-local forward step wiring dense + MoE layers |
| `engine.zig` | load + own resident set/KV/cache; RAM-budget → cache sizing; generate |
| `stats.zig` | RSS, usage histogram, `STATS`→`PIN` hot-set selection, raw-count persistence |
| `iobench.zig` | parallel random block reads → GB/s |
| `gen_checkpoint.zig` | deterministic synthetic checkpoint generator |
| `sampler.zig`, `tokenizer.zig` | greedy/temperature sampling; byte-level tokenizer |

## v0 acceptance mapping

| Acceptance item (CLAUDE.md) | Status |
|---|---|
| working forward (streamed experts, pinned hot-set + LRU + page cache) | ✅ |
| `STATS`/`PIN` hit-rate measurement | ✅ measure-then-pin across two runs |
| `iobench` disk profile | ✅ |
| per-turn tok/s, expert hit-rate, RSS logged | ✅ |
| runs within a declared RAM budget without OOM | ✅ `--ram-gb` sizes the LRU |
| content-addressed + Merkle integrity (blocks a poisoned expert) | ✅ per-block SHA-256 + Merkle root |
| **token-exact vs a `transformers` oracle on real weights** | ❌ **remaining** — see below |

## The remaining validation gap (honest)

The engine is structurally faithful (MLA, sigmoid top-k routing + shared expert,
int4 experts, the exact tiered fetch path) and numerically self-consistent, but it
is **not yet validated token-exact against a `transformers` oracle on real GLM-5.2
weights**. Closing that needs three things this repo does not ship:

1. **Real weights** converted to this int4 layout (colibri's `convert_fp8_to_int4.py`
   feeds the same content-addressed store; `gen --glm` describes the real shape but
   allocating its ~365 GB expert corpus is not a laptop operation).
2. **The real BPE tokenizer** (v0 uses a byte-level placeholder, vocab 256).
3. **Exact MLA/router hyperparameters** matched to the reference (RoPE convention,
   `noaux_tc` bias term, scaling factors), then a fixed-seed/prompt token-diff.

The synthetic path exists precisely so every other piece — streaming cache, int4
kernels, MLA, routing, STATS/PIN, iobench, Merkle — runs and is testable today; drop
in a real checkpoint and nothing in the engine changes.

## Deliberately out of scope for v0

Distribution (v1: expert directory, heat-aware replication, measured tier-order peer
fetch, EC cold tier, bulk propagation) and untrusted-peer verification (v2: RLNC +
homomorphic-hash pollution defense). The `get(expert_id)` cache interface is the seam
v1 slides the peer tier behind.
