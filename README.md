# Loom — distributed expert cache & weight-sharing node for large MoE inference

Loom is a research prototype (Zig, no runtime dependencies) exploring the design
in [`CLAUDE.md`](CLAUDE.md): for a huge Mixture-of-Experts model like GLM-5.2,
keep **compute node-local** and make the **weights** the thing that moves —
streamed from disk through a tiered cache on one box (v0), and shared across
nodes as content-addressed byte ranges of a GGUF file (v1, in progress — see
[`ROADMAP.md`](ROADMAP.md)).

One binary, `loom`, provides:

| Command | Purpose |
|---|---|
| [`loom node`](#loom-node--run-an-inference--weight-sharing-node) | the daemon: load a model, serve inference over RPC, share/sync/repair GGUF weight ranges over P2P with gossip discovery |
| [`loom run`](#loom-run--one-shot-local-inference) | one-shot local inference against a loom checkpoint (no servers) |
| [`loom gen`](#loom-gen--generate-a-synthetic-checkpoint) / [`loom info`](#loom-info--inspect--verify-a-checkpoint) | create / inspect+verify loom-format checkpoints |
| [`loom gguf`](#loom-gguf--gguf-tools-gen--info--run) | GGUF tools: make a fixture, inspect a file, **run a llama-architecture GGUF model** |
| [`loom iobench`](#loom-iobench--disk-profiler) | disk profiler for the random-read pattern the engine issues |

## Build

Targets **Zig 0.16.0** (pinned in `build.zig.zon`; [anyzig](https://github.com/marler8997/anyzig)
resolves it automatically).

```sh
zig build                        # debug binary -> zig-out/bin/loom
zig build -Doptimize=ReleaseFast # ~10x faster inference
zig build test                   # unit tests
zig build run -- <args>          # build + run in one step
```

## 60-second tour

```sh
# inference on a synthetic checkpoint, then as a served node
loom node                                   # zero-config: generates a tiny model, serves RPC :8770 / P2P :8771
printf '{"prompt":"hello","max_tokens":16}\n' | nc -w 3 127.0.0.1 8770

# run a real llama-architecture GGUF model (download any small GGUF first)
curl -LO https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf
loom gguf run stories15M-q4_0.gguf --prompt "Once upon a time"
```

---

## `loom node` — run an inference + weight-sharing node

```
loom node [--model SPEC] [--rpc-addr A] [--rpc-port P] [--p2p-addr A] [--p2p-port P]
          [--ram-gb X] [--pin-gb Y] [--seed S] [--stats FILE] [--no-verify]
          [--gguf FILE | --bootstrap HOST:PORT] [--peers H:P,H:P,...]
          [--hold-fraction F] [--range-mb M] [--advertise HOST:PORT]
```

Starts a long-running node that (a) loads a loom-format model and serves
inference over **RPC**, (b) optionally participates in **GGUF weight
distribution** over **P2P**, with gossip-based peer discovery and eager churn
repair. Every flag has a default, so `loom node` alone works.

### Model & engine flags

| Flag | Default | Meaning |
|---|---|---|
| `--model SPEC` | `tiny` | What to load for inference. `tiny` = auto-generated synthetic checkpoint (cached in `~/.cache/loom/models/tiny`); a **local directory** containing `manifest.loom`; or a **Hugging Face repo** `[hf:]org/repo[@rev]` holding a loom-format checkpoint — downloaded over HTTPS on first use, then served from the local cache (local-first resolution). |
| `--ram-gb X` | `4.0` | RAM budget. After the resident dense weights + KV cache, the remainder is split between the pinned hot-set and the LRU expert cache. The node never allocates expert storage beyond this. |
| `--pin-gb Y` | `0` | Portion of the budget for **pinning** the hottest experts, chosen from a prior `--stats` file (measure-then-pin; see `loom run`). |
| `--seed S` | `42` | Sampling seed (also the seed for random range selection when bootstrapping). |
| `--stats FILE` | off | Persist per-expert usage counts on exit; a later run with `--pin-gb` reads this file to choose the hot set. |
| `--no-verify` | verify on | Skip SHA-256 verification of expert blocks on disk reads (trusted-storage fast path). With verification on, a corrupted/poisoned expert fails the read with `PoisonedExpert`. |

Env overrides (flags win): `MODEL`, `RAM_BUDGET_GB`, `PIN_GB`, `SEED`, `STATS`.

### Server flags

| Flag | Default | Meaning |
|---|---|---|
| `--rpc-addr` / `--rpc-port` | `127.0.0.1` / `8770` | Where the JSON inference RPC listens. Bind `0.0.0.0` to accept remote clients. |
| `--p2p-addr` / `--p2p-port` | `0.0.0.0` / `8771` | Where the P2P line protocol listens (expert directory, weight ranges, gossip). |
| `--advertise HOST:PORT` | `127.0.0.1:<p2p-port>` | The dialable address this node announces to peers via gossip. Set it to your LAN/public address when peers are on other machines. |

### Weight-distribution flags (GGUF plane)

| Flag | Default | Meaning |
|---|---|---|
| `--gguf FILE` | off | Act as an **origin/full holder**: split `FILE` into ranges, compute the manifest (SHA-256 per range; Merkle root = **model version id**), hold and serve all ranges. Mutually exclusive with `--bootstrap`. |
| `--bootstrap HOST:PORT` | off | Boot by **syncing ranges from a peer**: adopt its manifest (root-verified), pick a random subset of ranges, fetch each one digest-verified. Also seeds the gossip peer table. |
| `--peers H:P,H:P,...` | none | Additional known peers (seed the gossip table; used by bootstrap and repair). |
| `--hold-fraction F` | `1.0` | Fraction of ranges this node wants to hold, chosen randomly (seeded by `--seed`, so a restart re-picks the same set). Independent random subsets across nodes overlap → emergent redundancy. |
| `--range-mb M` | `4.0` | Range size when *building* a fresh manifest (origin only; bootstrappers adopt the peer's value). |

Two loops run alongside the servers:

- **Gossip (every 3 s):** dial every known peer, announce `addr/version/holdings`,
  merge the peer's table back. Discovery is transitive — a node told only about
  B learns everything B knows within one round.
- **Eager churn repair (every 2 s):** whenever `wanted − held ≠ ∅`, retry every
  peer in the live table (including ones discovered via gossip and ones
  currently down — they're retried, not forgotten). Peers advertising a
  different manifest version are refused wholesale (the hardfork guard).

### RPC protocol (inference)

Line-delimited JSON over TCP; connections are handled concurrently, generation
is serialized. Request fields other than `prompt` are optional:

```sh
printf '{"prompt":"Loom weaves","max_tokens":24,"temp":0.0,"seed":7}\n' | nc -w 3 127.0.0.1 8770
# {"ok":true,"generated":24,"text":"...","tokens":[144,55,...],"tok_per_s":9.35,"hit_rate":0.8158}
```

`temp <= 0` = greedy (deterministic); `hit_rate` is the expert-cache hit rate
for the connection so far. Errors come back as `{"ok":false,"error":"..."}`.

### P2P protocol (line-based; one command per line)

| Request | Response | Purpose |
|---|---|---|
| `PING` | `PONG` | liveness |
| `HELLO` | `LOOM/0 experts=<n> unique_bytes=<b>` | node summary |
| `HAS <id>` | `PRESENT <id> off=.. len=.. sha256=..` \| `ERR range` | expert-directory query (v1 seed) |
| `MANIFEST` | `MANIFEST version=<hex> size=<b> ranges=<n> range_size=<b>` | weight manifest (`ERR no_store` if no GGUF attached) |
| `DIGEST <i>` / `DIGESTS` | one / all range digests | verification data |
| `HOLDINGS` | `HOLDINGS <hex bitmap>` | which ranges this node holds (bit i = range i) — the compact summary destined for ENR metadata |
| `GETR <i>` | `DATA <i> len=<l> sha256=<hex>` + raw bytes \| `ERR not_held` | fetch one range |
| `GOSSIP addr=.. version=.. holdings=..` | `PEERS <n>` + n × `addr=.. version=.. holdings=..` | announce yourself, receive the responder's entry + peer table |
| `TABLE` | same as `GOSSIP` response | inspect the peer table without announcing |

```sh
printf 'HELLO\nHAS 5\nPING\n' | nc -w 2 127.0.0.1 8771
# LOOM/0 experts=192 unique_bytes=47185920
# PRESENT 5 off=1228800 len=245760 sha256=c1a8c95b...
# PONG
```

### Worked example: a 3-node weight-sharing swarm (one machine)

```sh
loom gguf gen /tmp/model.gguf --data-mb 8          # synthetic GGUF to distribute

# terminal 1 — node A: origin, holds all 9 ranges (8 MB / 1 MB ranges)
loom node --gguf /tmp/model.gguf --range-mb 1 \
          --rpc-port 8770 --p2p-port 8771 --advertise 127.0.0.1:8771

# terminal 2 — node B: syncs a random ~half of the ranges from A
HOME=/tmp/nodeB loom node --bootstrap 127.0.0.1:8771 --hold-fraction 0.5 \
          --rpc-port 8780 --p2p-port 8781 --advertise 127.0.0.1:8781

# terminal 3 — node C: wants ALL ranges but is told only about B.
# Gossip teaches C about A transitively; eager repair fetches the ranges
# B doesn't have from A — a peer C was never configured with.
HOME=/tmp/nodeC loom node --bootstrap 127.0.0.1:8781 --hold-fraction 1.0 \
          --rpc-port 8790 --p2p-port 8791 --advertise 127.0.0.1:8791

# watch C converge to full holdings, and inspect everyone's peer tables
printf 'HOLDINGS\n' | nc -w 2 127.0.0.1 8791     # -> HOLDINGS ff01 (all 9 bits set)
printf 'TABLE\n'    | nc -w 2 127.0.0.1 8771     # A has learned B and C via gossip
```

(Separate `HOME`s only because all three nodes share one machine's model cache;
on real machines this isn't needed.)

---

## `loom run` — one-shot local inference

```
loom run <dir> [--prompt STR] [--max-tokens N] [--ram-gb X] [--pin-gb Y]
               [--temp T] [--seed S] [--stats FILE] [--no-verify]
```

Loads a loom checkpoint, generates, prints text + metrics, exits. No servers —
this is the measurement harness for the expert-streaming engine.

| Flag | Default | Meaning |
|---|---|---|
| `<dir>` | (or `MODEL` env) | checkpoint directory (`manifest.loom` + `dense.blob` + `experts.blob`) |
| `--prompt STR` | `"Loom"` | prompt (byte-level tokenizer on the synthetic model) |
| `--max-tokens N` | `32` | tokens to generate |
| `--temp T` | `0` | `<=0` greedy; else temperature sampling |
| `--ram-gb` / `--pin-gb` / `--seed` / `--stats` / `--no-verify` | as in `node` | same engine knobs |

The **measure-then-pin** loop — run once recording usage, run again pinning the
measured hot set:

```sh
loom gen /tmp/ckpt
loom run /tmp/ckpt --prompt "Loom weaves" --ram-gb 1 --stats /tmp/usage.stats
# ... hit_rate=0.892 disk_misses=91 ...
loom run /tmp/ckpt --prompt "Loom weaves" --ram-gb 1 --pin-gb 0.05 --stats /tmp/usage.stats
# ... pinned=91 hit_rate=1.000 disk_misses=0 (91 warm-up reads reported separately) ...
```

Each run logs tok/s, pin/LRU/disk hit breakdown, bytes read, digest failures,
and peak RSS. `--stats` also writes `<FILE>.txt`, a human-readable heat
histogram (rank, expert id, count, cumulative coverage, cumulative pin bytes).

---

## `loom gen` — generate a synthetic checkpoint

```
loom gen <dir> [--glm] [--seed N]
```

Writes a deterministic, self-consistent loom-format checkpoint: dense resident
weights + int4 routed experts, content-addressed (SHA-256 per expert, dedup)
with a Merkle-rooted manifest.

| Flag | Default | Meaning |
|---|---|---|
| `--glm` | off | use the real GLM-5.2 shape (78 layers, 256 experts/layer, ~19 MB/expert — describes the real deployment; generating its ~370 GB corpus is not a laptop operation). Default is the runnable `tiny` shape (8 layers, 32 experts, ~245 KB/expert). |
| `--seed N` | `42` | RNG seed; same seed → bit-identical checkpoint |

```sh
loom gen /tmp/ckpt --seed 7
```

## `loom info` — inspect & verify a checkpoint

```
loom info <dir>
```

Prints the model shape, expert sizing (per-token working set), the manifest's
Merkle root, and **verifies** it: recomputes the root over the expert index and
spot-checks an expert block's digest against the blob on disk.

```sh
loom info /tmp/ckpt
#   experts=32/layer, top-4 routed + 1 shared; expert_bytes=245760
#   merkle_root=b56ac498... merkle_check=OK expert0_digest_check=OK
```

---

## `loom gguf` — GGUF tools (`gen` / `info` / `run`)

### `loom gguf gen` — synthetic GGUF fixture

```
loom gguf gen <file> [--seed N] [--data-mb M] [--arch deepseek2]
```

Writes a small valid GGUF v3 file. Default (`demo`): metadata + two f32 tensors
of deterministic data — a distribution payload, **not** a runnable model.
`--arch deepseek2`: a structurally faithful tiny **deepseek2** model (MLA with
q-LoRA, 1 dense + 2 MoE layers, sigmoid gating + selection bias + shared
expert, byte-fallback SPM vocab) with random weights — runnable with
`loom gguf run`, used to validate the deepseek2 engine without a multi-GB
download.

### `loom gguf info` — inspect any GGUF

```
loom gguf info <file> [--range-mb M]              # default range preview: 4 MB
```

Prints version/alignment/data offset, all metadata KVs (including tokenizer
array sizes), the tensor table (name/type/dims/offset), and the distribution
manifest this file would get: range count + Merkle-root **version id**.

```sh
loom gguf info stories15M-q4_0.gguf --range-mb 1
```

### `loom gguf run` — run a llama-architecture GGUF model

```
loom gguf run <file.gguf> [--prompt STR] [--max-tokens N] [--temp T] [--seed S] [--ctx N]
```

| Flag | Default |
|---|---|
| `--prompt STR` | `"Once upon a time"` |
| `--max-tokens N` | `128` (stops early on EOS) |
| `--temp T` | `0` (greedy) |
| `--seed S` | `42` |

Real inference over the mmap'd file, dispatched on `general.architecture`:

- **`llama`** — GQA attention, NORM-style RoPE, SwiGLU.
- **`deepseek2`** (DeepSeek V2/V3, Kimi K2, GLM-class MoE) — MLA attention
  (q-LoRA, compressed-KV latent cache, decoupled NORM-rope head), MoE FFN
  (sigmoid/softmax gating, noaux_tc selection bias, top-k with renormalized
  scaled gates, shared experts, leading dense layers), YaRN context-extension
  scaling, and the gpt2-style **byte-level BPE tokenizer** from GGUF merges.
  **Validated on real weights**: DeepSeek-V2-Lite Q4_K_M (15.7B MoE, 27 MLA
  layers, 64 experts) produces correct factual completions ("The capital of
  France is Paris.") on one CPU core.

GGML tensor types: **F32 / F16 / Q4_0 / Q5_0 / Q8_0 / Q4_K / Q5_K / Q6_K**
(fused matvec on the raw mmap'd bytes — no wholesale dequantization).
Tokenizers: SentencePiece (score-merge, byte fallback) and byte-level BPE
(merge ranks, gpt2 byte table), selected by `tokenizer.ggml.model`. Output
streams as it generates. `--ctx N` (default 4096) caps the KV allocation for
models advertising 100k+ contexts.

```sh
curl -LO https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf
loom gguf run stories15M-q4_0.gguf --prompt "Once upon a time" --max-tokens 100
# output: <s> Once upon a time, there was a little girl named Lily. She loved to
# play outside and explore the world around her...
# ---- 5 prompt + 100 generated tokens in 0.45s (235.2 tok/s) ----
```

Validated against real reference models (`ggml-org/models` tinyllamas:
stories260K F32/GQA, stories15M Q4_0 and Q8_0) — coherent English confirms
kernels, attention, RoPE convention, and tokenizer simultaneously. Serving GGUF
models through `loom node`'s RPC is the next integration step.

---

## Target deployment: GLM 5.2 on a swarm of 16 GB machines (plan)

The end-state this repo is building toward (see [ROADMAP.md](ROADMAP.md);
expert-aligned shards are [#2](https://github.com/ch4r10t33r/loom/issues/2),
token-loop peer fetch is [#3](https://github.com/ch4r10t33r/loom/issues/3)):
GLM 5.2 — 744B-param MoE, 75 MoE layers × 256 experts, 8+1 active — served by
a swarm of commodity machines, onboarded by a **bootnode** that assigns each
joiner the most under-replicated shards first (coverage-aware, from gossiped
holdings; never in the inference path, no permanent dependency).

**Sharding.** Shard unit = one expert block (~19 MB int4: the gate/up/down
matrices of one expert, an extent-list over the GGUF's 3D expert tensors):

| Quantity | Value |
|---|---|
| Routed shards | **19,200** (75 layers × 256 experts) |
| Shard size | **~19 MB** (int4) |
| Routed corpus | ~370 GB, stored once across the swarm |
| Resident bundle | ~10 GB (attention, shared experts, embeddings, routers) — held **in full by every node** |
| Logical GGUF size | **~380 GB**; each node's copy is a *sparse* file with only resident + held bytes real |
| Manifest / holdings bitmap | ~1 MB / 2.4 KB (bitmap gossiped whole; ENR carries its hash) |

**Per-node plan.** Held shards are a *disk* budget (pread-served), not RAM:

| | Minimum | Recommended |
|---|---|---|
| RAM | **16 GB** (≈10 GB mmap'd dense + ~0.7 GB MLA KV @8k ctx + 2–3 GB expert cache) | 24–32 GB |
| Disk | **75 GB NVMe** (10 GB resident + ~50 GB shards + slack) | 150 GB NVMe |
| Experts held | ~1,300 (25 GB) | **~2,600 (≈50 GB, `--hold-gb 50`)** — 13.7% of the corpus |
| Network | 1 GbE (capacity win only) | 10 GbE+ (latency win too) |

**Swarm sizing** (replication R over the 370 GB corpus; R=3 target, R=2 floor):
at 50 GB/node, **R=2 needs ≥15 nodes, R=3 needs ≥22**; at 100 GB/node, 8 nodes
reach R=2. The bootnode's least-replicated-first assignment makes coverage
guaranteed rather than probabilistic; gossip + eager repair maintain R after
churn.

Honest caveats: the engine's kernels are still scalar (SIMD pending), single
low-latency streams can't fully hide per-layer fetch latency (this is a
serving-throughput architecture — batch and pin), and on 1 GbE the win is
*fitting the model at all*, not speed (see CLAUDE.md's bandwidth table).

## `loom iobench` — disk profiler

```
loom iobench <file> [--threads N] [--block-mb M] [--reads R]
```

| Flag | Default | Meaning |
|---|---|---|
| `--threads N` | CPU count | concurrent readers |
| `--block-mb M` | `19` | read size — 19 MB is one GLM-5.2 expert block |
| `--reads R` | `64` | random reads per thread |

Measures the exact access pattern the engine issues on a cache miss (parallel
large random reads) in GB/s. Per CLAUDE.md, the local-disk vs. network tier
order must be **measured, not assumed** — this is the measuring tool.

```sh
loom iobench /tmp/ckpt/experts.blob --threads 8 --block-mb 1 --reads 64
#   512 reads, 0.54 GB in 0.029s => 18.75 GB/s
```

---

## Source map

| Module | Role |
|---|---|
| `model.zig` | `ModelConfig`; GLM-5.2 shape + runnable `tiny` shape; expert/working-set sizing |
| `quant.zig` | loom int4 expert format (f32 scale/32 weights) quantize + fused matvec |
| `tensor.zig` | RMSNorm, softmax, SwiGLU, dense matvec, partial RoPE |
| `attention.zig` | MLA: q/kv-LoRA, partial RoPE, compressed-latent KV cache |
| `moe.zig` | DeepSeek-V3 sigmoid router (top-k + shared expert); streamed int4 expert FFN |
| `checkpoint.zig` | loom on-disk format: content-addressed, Merkle-rooted, deduped |
| `expert_cache.zig` | tiered expert cache: pinned hot-set → LRU → pread, usage stats, digest verify |
| `forward.zig` / `engine.zig` | forward step wiring; engine lifecycle, RAM-budget → cache sizing |
| `node.zig` | `loom node` orchestration: model → engine → RPC/P2P/gossip/repair |
| `hf.zig` | model resolver: local dir / synthetic / Hugging Face download (local-first) |
| `rpc.zig` | JSON-over-TCP inference server (concurrent connections, serialized generate) |
| `p2p.zig` | P2P line protocol: expert directory, weight ranges, gossip |
| `weights.zig` | range-sharded GGUF store: manifest, version id, holdings/wanted bitmaps, verified IO |
| `sync.zig` | peer sync client: manifest adoption, root verification, multi-peer range fetch |
| `peers.zig` | dynamic peer table shared by gossip/repair/P2P threads |
| `gossip.zig` | 3 s gossip loop: announce self, merge peers-of-peers |
| `gguf.zig` | GGUF v2/v3 parser (metadata incl. tokenizer arrays, tensor table) + fixture writer |
| `ggml.zig` | GGML kernels: F32/F16/Q4_0/Q8_0 fused matvec + row dequant |
| `llama.zig` | llama-arch engine over mmap'd GGUF: GQA, NORM RoPE, SwiGLU, SPM tokenizer |
| `deepseek.zig` | deepseek2-arch engine (Kimi/DeepSeek/GLM): MLA + MoE routing over mmap'd GGUF |
| `stats.zig` | RSS, usage histograms, STATS→PIN hot-set selection |
| `iobench.zig` | parallel random-read disk profiler |
| `gen_checkpoint.zig` | deterministic synthetic checkpoint generator |
| `hash.zig` | SHA-256 content addressing + Merkle root |
| `sampler.zig` / `tokenizer.zig` | greedy/temperature sampling; byte-level tokenizer (synthetic model) |

## Status & honest gaps

- **v0 (single-node expert streaming): done**, except token-exact validation
  against a `transformers` oracle on real GLM-5.2 weights — that needs the real
  converted weights (~370 GB), the real BPE tokenizer, and exact MLA/router
  hyperparameters. The synthetic model exercises every code path; a real
  checkpoint drops in without engine changes.
- **v1 (distributed weight sharing): first cuts working** — GGUF range
  sharding, multi-peer boot sync, gossip discovery, eager churn repair, and
  the version guard that will enforce hardforks. Remaining: ENR integration,
  real gossipsub transport, majority-hardfork coordination, serving GGUF
  models through the node's RPC. See [`ROADMAP.md`](ROADMAP.md) for the
  requirements of record and decisions (random+redundant placement, ENR +
  global gossip topic, maximally eager repair).
- **v2 (untrusted peers)** is design-only (see `CLAUDE.md`).
