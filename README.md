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

## Run a node

`loom node` loads a model, serves inference over RPC, and answers expert-directory
queries over P2P. Every parameter has a default, so it runs with zero args (it
auto-generates a synthetic `tiny` model into `~/.cache/loom/models`):

```sh
loom node
# loom node --model <local dir | tiny | [hf:]org/repo[@rev]>
#           --rpc-addr 127.0.0.1 --rpc-port 8770
#           --p2p-addr 0.0.0.0   --p2p-port 8771
#           --ram-gb 4 --pin-gb 0 --seed 42 [--stats FILE] [--no-verify]
```

**RPC** — line-delimited JSON over TCP (keep-alive; connections handled
concurrently, generation serialized):

```sh
printf '{"prompt":"Loom weaves","max_tokens":24}\n' | nc 127.0.0.1 8770
# -> {"ok":true,"generated":24,"text":"...","tokens":[...],"tok_per_s":X,"hit_rate":Y}
```

**P2P** — minimal peer directory (the v1 "who-has expert" seed):

```sh
printf 'HELLO\nHAS 5\nPING\n' | nc 127.0.0.1 8771
# LOOM/0 experts=192 unique_bytes=47185920
# PRESENT 5 off=1228800 len=245760 sha256=c1a8...
# PONG
```

`--model org/repo` downloads a **loom-format** checkpoint (`manifest.loom` +
`dense.blob` + `experts.blob`) from the Hugging Face Hub over HTTPS into the cache.
Converting raw GLM-5.2 safetensors (FP8→int4) remains a separate offline step.

## GGUF weight distribution (v1, first cut)

A GGUF file is split into fixed-size byte ranges (SHA-256 per range; the Merkle
root over range digests is the **model version id**). Nodes hold random subsets
of ranges — overlap across nodes gives redundancy — and new nodes sync from
peers instead of re-downloading from origin:

```sh
loom gguf gen /tmp/model.gguf --data-mb 8      # synthetic GGUF fixture
loom gguf info /tmp/model.gguf --range-mb 1    # metadata + range manifest

# node A: origin, serves the full GGUF
loom node --gguf /tmp/model.gguf --range-mb 1 --p2p-port 8771

# node B: boots by syncing a random half of the ranges from A,
# digest-verifying every range against the manifest root
loom node --bootstrap 127.0.0.1:8771 --hold-fraction 0.5 --p2p-port 8781
```

P2P weight ops: `MANIFEST` (version/size/ranges), `DIGESTS` (bulk),
`HOLDINGS` (hex bitmap, bit i = holds range i — the compact summary destined
for ENR metadata + gossip), `GETR <i>` (range bytes; `ERR not_held` otherwise).
Syncing from a partial holder takes what's available and reports the shortfall.

See [ROADMAP.md](ROADMAP.md) for where this is headed (ENR/gossip advertising,
churn repair, majority hardforks).

## Offline tools

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
| `node.zig` | `loom node`: resolve model → engine → RPC + P2P servers |
| `hf.zig` | model resolver: local dir / synthetic `tiny` / Hugging Face download |
| `rpc.zig` | JSON-over-TCP inference server (concurrent conns, serialized generate) |
| `p2p.zig` | peer directory + weight-range serving (`HAS`, `MANIFEST`, `DIGESTS`, `HOLDINGS`, `GETR`) |
| `gguf.zig` | GGUF v2/v3 parser (header, metadata, tensor table) + synthetic fixture writer |
| `weights.zig` | range-sharded weight store: manifest, Merkle version id, holdings bitmap, verified range IO |
| `sync.zig` | boot-time peer sync client (`--bootstrap`): manifest → digests → verified range fetch |
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
