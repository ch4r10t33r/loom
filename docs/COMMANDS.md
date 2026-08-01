# Command tour

Worked examples for every `loom` command, with real output. This is the
narrative companion to [`CLI.md`](CLI.md), which lists every flag with its
default and when to change it.

## `loom node` — run an inference + weight-sharing node

*Every flag for this command: [docs/CLI.md](CLI.md#loom-node).*

```
loom node [--model SPEC] [--rpc-addr A] [--rpc-port P] [--openai-addr A] [--openai-port P]
          [--p2p-addr A] [--p2p-port P] [--ram-gb X] [--pin-gb Y] [--seed S]
          [--stats FILE] [--no-verify] [--gguf FILE | --bootstrap HOST:PORT]
          [--peers H:P,H:P,...] [--hold-fraction F] [--range-mb M] [--advertise HOST:PORT]
```

Starts a long-running node. It loads a loom-format model and serves inference
over the native RPC and an optional OpenAI-compatible HTTP API, and it can
also take part in GGUF weight distribution over P2P, with gossip-based peer
discovery and eager churn repair. Every flag has a default, so `loom node`
alone works.

### Model & engine flags

| Flag | Default | Meaning |
|---|---|---|
| `--model SPEC` | `tiny` | What to load for inference. `tiny` = auto-generated synthetic checkpoint (cached in `~/.cache/loom/models/tiny`); a local directory containing `manifest.loom`; or a Hugging Face repo `[hf:]org/repo[@rev]` holding a loom-format checkpoint, downloaded over HTTPS on first use and then served from the local cache (local-first resolution). |
| `--ram-gb X` | `4.0` | RAM budget. After the resident dense weights + KV cache, the remainder is split between the pinned hot-set and the LRU expert cache. The node never allocates expert storage beyond this. |
| `--pin-gb Y` | `0` | Portion of the budget for pinning the hottest experts, chosen from a prior `--stats` file (measure-then-pin; see `loom run`). |
| `--seed S` | `42` | Sampling seed (also the seed for random range selection when bootstrapping). |
| `--stats FILE` | off | Persist per-expert usage counts on exit; a later run with `--pin-gb` reads this file to choose the hot set. |
| `--no-verify` | verify on | Skip SHA-256 verification of expert blocks on disk reads (trusted-storage fast path). With verification on, a corrupted/poisoned expert fails the read with `PoisonedExpert`. |

Env overrides (flags win): `MODEL`, `RAM_BUDGET_GB`, `PIN_GB`, `SEED`, `STATS`.

### Server flags

| Flag | Default | Meaning |
|---|---|---|
| `--rpc-addr` / `--rpc-port` | `127.0.0.1` / `8770` | Where the JSON inference RPC listens. Bind `0.0.0.0` to accept remote clients. |
| `--openai-addr` / `--openai-port` | `<rpc-addr>` / `0` (off) | Where the OpenAI-compatible HTTP API listens. Set `--openai-port` to enable it (e.g. `8772`). Serves the same engine as the RPC, metered by `Authorization: Bearer` client id. |
| `--ctx N` | `4096` | Context-length cap when serving a distributed GGUF engine. |
| `--chat-format F` | auto | Override the chat template (`deepseek`/`chatml`/`llama2`/`llama3`/`gemma`/`mistral`/`generic`); default auto-detects from GGUF metadata. |
| `--p2p-addr` / `--p2p-port` | `0.0.0.0` / `8771` | Where the P2P line protocol listens (expert directory, weight ranges, gossip). |
| `--advertise HOST:PORT` | `127.0.0.1:<p2p-port>` | The dialable address this node announces to peers via gossip. Set it to your LAN/public address when peers are on other machines. |

### Weight-distribution flags (GGUF plane)

| Flag | Default | Meaning |
|---|---|---|
| `--gguf FILE` | off | Act as an origin/full holder: split `FILE` into ranges, compute the manifest (SHA-256 per range; Merkle root = model version id), hold and serve all ranges. Mutually exclusive with `--bootstrap`. |
| `--bootstrap HOST:PORT` | off | Boot by syncing ranges from a peer: adopt its manifest (root-verified), pick a random subset of ranges, fetch each one digest-verified. Also seeds the gossip peer table. |
| `--peers H:P,H:P,...` | none | Additional known peers (seed the gossip table; used by bootstrap and repair). |
| `--hold-fraction F` | `1.0` | Fraction of ranges this node wants to hold, chosen randomly (seeded by `--seed`, so a restart re-picks the same set). Independent random subsets across nodes overlap → emergent redundancy. |
| `--range-mb M` | `4.0` | Range size when *building* a fresh manifest (origin only; bootstrappers adopt the peer's value). |
| `--r-target N` | `2` | Committee redundancy target when acting as bootnode (expert-sharded origin). |

**Committees (SPEC.md).** An expert-sharded origin automatically acts as the
bootnode: `JOIN` assigns each connecting node to a shard committee and
a least-covered-first want-set, so every committee converges to holding the
complete shard set with redundancy `--r-target` (default 2) by construction.
Joiners sync from committee members first, then the bootnode; committee
members heartbeat each other every 5 s and log liveness transitions. When all
committees are saturated, the next joiner opens a new one. See
[spec/SPEC.md](../spec/SPEC.md) for the full p2p-layer spec (roles, invariants, query
path, wire protocol).

Two loops run alongside the servers:

- **Gossip (every 3 s):** dial every known peer, exchange binary Announce /
  AnnounceBatch frames (addr, committee id, manifest version, holdings seq +
  bitmap; snappy-compressed). Discovery is transitive. Announces carry
  committee ids, so the table doubles as a gossip-derived committee view:
  earlier committee members discover later joiners automatically and
  start heartbeating them.
- **Eager churn repair (every 2 s):** whenever `wanted − held ≠ ∅`, retry every
  peer in the live table, including ones discovered via gossip and ones
  currently down (down peers stay in the table and get retried). Peers
  advertising a different manifest version are refused wholesale (the
  hardfork guard).

### RPC protocol (inference)

Line-delimited JSON over TCP; connections are handled concurrently, generation
is serialized. Request fields other than `prompt` are optional:

```sh
printf '{"prompt":"Loom weaves","max_tokens":24,"temp":0.0,"seed":7}\n' | nc -w 3 127.0.0.1 8770
# {"ok":true,"generated":24,"text":"...","tokens":[144,55,...],"tok_per_s":9.35,"hit_rate":0.8158}
```

`temp <= 0` = greedy (deterministic); `hit_rate` is the expert-cache hit rate
for the connection so far. Errors come back as `{"ok":false,"error":"..."}`.

### OpenAI-compatible API

Enable with `--openai-port` to let off-the-shelf clients (OpenWebUI, Continue,
aider, the OpenAI SDKs, curl) talk to the node with no adapter. It serves the
same engine as the native RPC and is metered by the same ledger, using the
`Authorization: Bearer <token>` header as the client id (out-of-band, so it is
not forgeable by prompt content). Routes: `GET /v1/models`,
`POST /v1/chat/completions`, `POST /v1/completions`, `GET /health`.

```sh
loom node --openai-port 8772 &

curl -s http://127.0.0.1:8772/v1/chat/completions \
  -H 'Authorization: Bearer sk-alice' \
  -d '{"model":"tiny","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"seed":7}'
# {"id":"chatcmpl-loom-1","object":"chat.completion","created":0,"model":"tiny",
#  "choices":[{"index":0,"message":{"role":"assistant","content":"..."},"finish_reason":"length"}],
#  "usage":{"prompt_tokens":19,"completion_tokens":16,"total_tokens":35}}
```

Output is byte-identical to the native RPC path for the same prompt/seed/params.
`usage` is the real token count and the ledger's cost unit; an exhausted client
gets HTTP `402`. `stream:true` streams the completion as OpenAI Server-Sent
Events (`text/event-stream`: one `chat.completion.chunk` / `text_completion`
event per token, then `data: [DONE]`), over both the loom and distributed-GGUF
engines:

```sh
curl -sN http://127.0.0.1:8772/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":true}'
# data: {...,"choices":[{"index":0,"delta":{"role":"assistant"},...}]}
# data: {...,"choices":[{"index":0,"delta":{"content":"..."},...}]}   (per token)
# data: {...,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
# data: [DONE]
```

Chat `messages[]` are rendered with the model's chat template. The format is
auto-detected from the GGUF `tokenizer.chat_template` metadata (or the arch),
overridable with `--chat-format {deepseek|chatml|llama2|llama3|gemma|mistral|generic}`.
Marker tokens (chatml `<|im_start|>`, llama3 `<|start_header_id|>`, gemma
`<start_of_turn>`, control tokens) are tokenized atomically to their ids by the
special-token matcher (`gguf/special.zig`) when the model's vocab defines them.

### P2P protocol (line-based; one command per line)

| Request | Response | Purpose |
|---|---|---|
| `PING` | `PONG` | liveness |
| `HELLO` | `LOOM/0 experts=<n> unique_bytes=<b>` | node summary |
| `HAS <id>` | `PRESENT <id> off=.. len=.. sha256=..` \| `ERR range` | expert-directory query (v1 seed) |
| `MANIFEST` | `MANIFEST version=.. size=.. ranges=.. range_size=.. mode=<fixed\|expert> resident=<n>` | manifest summary (`ERR no_store` if no GGUF attached) |
| `MANIFESTFILE` | `MANIFESTFILE len=<n>` + serialized manifest | full manifest (digests + extent lists) — what bootstrap adopts |
| `DIGEST <i>` / `DIGESTS` | one / all range digests | verification data |
| `HOLDINGS` | `HOLDINGS <hex bitmap>` | which ranges this node holds (bit i = range i); the compact summary destined for ENR metadata |
| `GETR <i>` | `DATA <i> len=<l> sha256=<hex>` + raw bytes \| `ERR not_held` | fetch one range |
| `JOIN addr=.. fraction=..` | `COMMITTEE id=.. members=.. assign=<hex>` | bootnode: committee + assigned want-set |
| `COMMITTEES` | per-committee coverage summary | bootnode debug |
| `FRAME <len>` + frame | `FRAME <len>` + frame | binary wire messages v1 (SPEC.md): heartbeat (committee state: version, holdings seq/digest, load), expert request/response (status: ok/not_held/version_mismatch/busy), adaptive snappy |
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

*Every flag for this command: [docs/CLI.md](CLI.md#loom-run).*

```
loom run <dir> [--prompt STR] [--max-tokens N] [--ram-gb X] [--pin-gb Y]
               [--temp T] [--seed S] [--stats FILE] [--no-verify]
```

Loads a loom checkpoint, generates, prints text + metrics, exits. No servers;
this is the measurement harness for the expert-streaming engine.

| Flag | Default | Meaning |
|---|---|---|
| `<dir>` | (or `MODEL` env) | checkpoint directory (`manifest.loom` + `dense.blob` + `experts.blob`) |
| `--prompt STR` | `"Loom"` | prompt (byte-level tokenizer on the synthetic model) |
| `--max-tokens N` | `32` | tokens to generate |
| `--temp T` | `0` | `<=0` greedy; else temperature sampling |
| `--ram-gb` / `--pin-gb` / `--seed` / `--stats` / `--no-verify` | as in `node` | same engine knobs |

The measure-then-pin loop: run once recording usage, then run again pinning
the measured hot set:

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

## `loom light` — delegating light node

*Every flag for this command: [docs/CLI.md](CLI.md#loom-light).*

```
loom light [--full-nodes H:RPC_PORT[,...]] [--openai-port P --openai-full-nodes H:OPENAI_PORT[,...]]
           [--rpc-addr A] [--rpc-port P] [--openai-addr A] [--client-id ID]
```

For low-memory devices: holds no weights, no store, no engine (megabytes of
footprint). It exposes the native line-JSON RPC and/or an OpenAI-compatible
HTTP API and delegates every request to a full node, round-robin with
failover. It forces its `--client-id` on each request, so a caller cannot
spend under another identity (a caller-supplied `client` field or
`Authorization` bearer is dropped). Configure at least one surface:
`--full-nodes` for the native RPC (default port 8768), and/or `--openai-port`
with `--openai-full-nodes` for the OpenAI surface.

Full nodes meter clients: the native RPC responses carry `cost` (prompt +
generated tokens) and `balance`, the OpenAI responses carry the `usage` object;
when a client's allowance (`--free-quota` on the full node + credits − usage)
hits zero, requests get `payment_required` / HTTP `402` until credited via the
settlement stub (`{"method":"credit","client":ID,"amount":N}` — proof
verification is the planned payment-rail integration point;
`{"method":"tab","client":ID}` shows the ledger).

```sh
# native RPC delegation
loom node --free-quota 5000 &                # full node, metered
loom light --full-nodes 127.0.0.1:8770 --client-id alice &
printf '{"prompt":"hi","max_tokens":16}\n' | nc -w 3 127.0.0.1 8768
# {"ok":true,...,"cost":18,"balance":4982}

# OpenAI delegation (light node proxies to a full node's OpenAI port)
loom node --openai-port 8772 --free-quota 5000 &
loom light --openai-port 9000 --openai-full-nodes 127.0.0.1:8772 --client-id alice &
curl -s http://127.0.0.1:9000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
# metered as 'alice' on the full node, regardless of any bearer the caller sends
```

## `loom gen` — generate a synthetic checkpoint

*Every flag for this command: [docs/CLI.md](CLI.md#loom-gen).*

```
loom gen <dir> [--glm] [--seed N]
```

Writes a deterministic, self-consistent loom-format checkpoint: dense resident
weights + int4 routed experts, content-addressed (SHA-256 per expert, dedup)
with a Merkle-rooted manifest.

| Flag | Default | Meaning |
|---|---|---|
| `--glm` | off | use the real GLM-5.2 shape (78 layers, 256 experts/layer, ~19 MB/expert). This describes the real deployment; generating its ~370 GB corpus is impractical on a laptop. Default is the runnable `tiny` shape (8 layers, 32 experts, ~245 KB/expert). |
| `--seed N` | `42` | RNG seed; same seed → bit-identical checkpoint |

```sh
loom gen /tmp/ckpt --seed 7
```

## `loom info` — inspect & verify a checkpoint

*Every flag for this command: [docs/CLI.md](CLI.md#loom-info).*

```
loom info <dir>
```

Prints the model shape, expert sizing (per-token working set), the manifest's
Merkle root, and verifies it: recomputes the root over the expert index and
spot-checks an expert block's digest against the blob on disk.

```sh
loom info /tmp/ckpt
#   experts=32/layer, top-4 routed + 1 shared; expert_bytes=245760
#   merkle_root=b56ac498... merkle_check=OK expert0_digest_check=OK
```

---

## `loom gguf` — GGUF tools (`gen` / `info` / `shard` / `run`)

*Every flag for this command: [docs/CLI.md](CLI.md#loom-gguf).*

### `loom gguf gen` — synthetic GGUF fixture

```
loom gguf gen <file> [--seed N] [--data-mb M] [--arch A]
```

Writes a small valid GGUF v3 file. Default (`demo`): metadata + two f32 tensors
of deterministic data — a distribution payload, not a runnable model.

`--arch deepseek2|llama|qwen2moe|qwen3moe|glm4moe` writes a structurally
faithful tiny model of that architecture with random weights — runnable with
`loom gguf run` and shardable by expert, so the engines and the whole
distribution path can be validated without a multi-GB download. Each fixture
carries exactly the optional pieces the real architecture has, so together they
cover every branch of the shared engine:

| `--arch` | What it exercises |
|---|---|
| `deepseek2` | MLA with q-LoRA, 1 dense + 2 MoE layers, sigmoid gating, selection bias, shared expert |
| `llama` | Mixtral shape: softmax routing, no bias, no Q/K norm, no shared expert, NORM RoPE |
| `qwen2moe` | QKV biases, sigmoid-gated shared expert, and the one arch that does not renormalize gates |
| `qwen3moe` | Q/K norm, and a head_dim that is deliberately not `dim / n_heads` |
| `glm4moe` | QKV biases, Q/K norm, sigmoid routing with selection bias, plain shared expert, `post_attention_norm` in place of `ffn_norm`, a leading dense layer, partial RoPE, and a trailing NextN block the forward pass must skip |

### `loom gguf info` — inspect any GGUF

```
loom gguf info <file> [--range-mb M]              # default range preview: 4 MB
```

Prints version/alignment/data offset, all metadata KVs (including tokenizer
array sizes), the tensor table (name/type/dims/offset), and the distribution
manifest this file would get: range count + Merkle-root version id.

```sh
loom gguf info stories15M-q4_0.gguf --range-mb 1
```

### `loom gguf shard` — expert-aligned shard manifest

```
loom gguf shard <file>
```

The sharding tool. It parses the GGUF tensor table and builds the
expert-aligned manifest: one shard per (layer, expert) as a 3-extent list over
the `ffn_{gate,up,down}_exps` tensors, with the resident bundle chunked at
16 MB. Then it prints the summary. `loom node --gguf` runs the same split
automatically (expert mode when the file has expert tensors, fixed ranges
otherwise).

```sh
loom gguf shard DeepSeek-V2-Lite.Q4_K_M.gguf
#   shards         1737 total = 73 resident + 1664 expert
#   resident       0.780 GB in 73 chunks (held by every node)
#   expert shards  4.98..6.02 MB (avg 5.46 MB), 8.87 GB routed corpus
#   metadata       manifest 204.3 KB, holdings bitmap 218 B
```

For GLM 5.2 this yields the planned 19,200 expert shards (~19 MB each) + a
~10 GB resident bundle. `--hold-fraction` applies to expert shards only;
resident shards are always in every node's want-set.

### `loom gguf run` — run a GGUF model

```
loom gguf run <file.gguf> [--prompt STR] [--max-tokens N] [--temp T] [--seed S] [--ctx N]
```

| Flag | Default |
|---|---|
| `--prompt STR` | `"Once upon a time"` |
| `--max-tokens N` | `128` (stops early on EOS) |
| `--temp T` | `0` (greedy) |
| `--seed S` | `42` |

Real inference over the mmap'd file, dispatched on `general.architecture`.
Two engines:

- **`deepseek2`** (DeepSeek V2/V3, Kimi K2) — MLA attention (q-LoRA,
  compressed-KV latent cache, decoupled NORM-rope head), MoE FFN, YaRN
  context-extension scaling.
  Validated on real weights: DeepSeek-V2-Lite Q4_K_M (15.7B MoE, 27 MLA
  layers, 64 experts) produces correct factual completions ("The capital of
  France is Paris.") on one CPU core.
- **`llama` / `qwen2moe` / `qwen3moe` / `glm4moe`** — one shared GQA engine.
  These differ only in optional pieces bolted onto the same skeleton, so the
  engine detects them from the tensors the file contains rather than from a
  table of per-architecture assumptions: QKV biases, per-head Q/K RMSNorm
  before RoPE, a dense or mixture-of-experts FFN, a shared expert that may be
  sigmoid-gated, leading dense layers, a post-attention norm standing in for
  `ffn_norm`, and trailing MTP/NextN blocks that are skipped. The one fact
  pinned per architecture is the RoPE style (adjacent-pair NORM for `llama`,
  split-half NEOX for the rest), because getting it wrong produces fluent but
  wrong output rather than an error.

MoE routing is shared by both engines (`src/gguf/moe.zig`): sigmoid or softmax
gating, noaux_tc selection bias, top-k with optionally renormalized and scaled
gates, shared experts.

GGML tensor types — every quantization llama.cpp ships except the ternary TQ
types and NVFP4:

| Family | Types |
|---|---|
| float | `F32`, `F16` |
| legacy | `Q4_0`, `Q5_0`, `Q8_0` |
| K-quants | `Q4_K`, `Q5_K`, `Q6_K` |
| IQ (codebook) | `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS` |
| microscaling | `MXFP4` |

Affine types (float/legacy/K) use a fused matvec on the raw mmap'd bytes, with
no wholesale dequantization. `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K` and `Q8_0`
additionally quantize the activation vector to int8 once per matvec and dot it
against the packed weights as integers, which removes the dequantize step
entirely and is worth 2x to 9x depending on the format. The IQ and MXFP4 types
are codebook quants: a block stores an index into a static grid table rather
than a value to scale, so they decode through `src/gguf/iq.zig` against tables
transcribed from llama.cpp. Because a wrong table entry there would silently
corrupt weights instead of failing, all ten decoders are checked bit-for-bit
against llama.cpp's own `dequantize_row_*` output on golden vectors
(`src/gguf/iq_vectors.zig`).
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
stories260K F32/GQA, stories15M Q4_0 and Q8_0, DeepSeek-V2-Lite Q4_K_M);
coherent English confirms kernels, attention, RoPE convention, and tokenizer
simultaneously. `loom node` can also serve the distributed GGUF (deepseek2)
engine directly over its RPC and OpenAI surfaces — see [Serving a distributed
GGUF model through the node](#serving-a-distributed-gguf-model-through-the-node).

### Distributed run: inference from a partial store

Point `gguf run` at a store directory (instead of a .gguf) and give it peers.
Held shards come from the local sparse file; missing experts are fetched from
peers inside the token loop, in parallel per MoE layer, round-robin across
holders, digest-verified before touching disk, then persisted. The node's
holdings grow with use and gossip advertises them, so fetch-on-demand doubles
as organic heat replication.

```sh
# node A serves the full expert-sharded model
loom node --gguf DeepSeek-V2-Lite.Q4_K_M.gguf --p2p-port 8771

# this machine holds a 33% store (bootstrapped earlier); missing experts
# stream from A during inference — output is token-identical to a full copy
loom gguf run ~/.cache/loom/models/gguf-synced --peers 127.0.0.1:8771 \
     --prompt "The capital of France is" --max-tokens 8
# output: The capital of France is Paris.
# expert tiers: local=2184 peer_fetched=641 (3512.6 MB, avg 29.2 ms/fetch) failures=0
# holdings grew 573 -> 1214 shards (fetched experts persisted + advertised)
```

### Serving a distributed GGUF model through the node

`loom gguf run` above is a one-shot CLI. The node serves the same distributed
engine as a long-running service over its RPC and OpenAI surfaces: when
`loom node` is given an expert-sharded GGUF (`--gguf` origin, or `--bootstrap`
to sync a partial store) and its resident bundle is complete, it serves the
deepseek2 engine and fetches missing experts from peers inside the token loop.
Otherwise it serves the loom-format `--model`. `--ctx N` caps the context
length (default 4096).

```sh
# origin: holds the whole expert-sharded model, serves it + acts as bootnode
loom node --gguf DeepSeek-V2-Lite.Q4_K_M.gguf \
          --rpc-port 8770 --openai-port 8772 --p2p-port 8771 --advertise 127.0.0.1:8771

# partial node: syncs ~30% of experts, serves the same model; the missing
# experts stream from the origin during generation (output identical to a full copy)
HOME=/tmp/nodeB loom node --bootstrap 127.0.0.1:8771 --hold-fraction 0.3 \
          --rpc-port 8780 --openai-port 8782 --p2p-port 8781 --advertise 127.0.0.1:8781

curl -s http://127.0.0.1:8782/v1/completions -d '{"prompt":"the","max_tokens":8}'
# a hit_rate below 1.0 on the partial node's RPC response is the token-loop
# peer fetch at work (experts read from the origin, not held locally)
```

Store mutation from the token-loop fetch and the eager-repair loop is serialized
on one engine mutex. The first request on a cold partial node is slow (many
sequential cold expert fetches); it warms as fetched experts are persisted.
This is the serving-first behavior the design calls for. Chat requests are
rendered with the model's detected chat template (`--chat-format` to
override); `--ctx N` caps context length.

---

## `loom iobench` — disk profiler

*Every flag for this command: [docs/CLI.md](CLI.md#loom-iobench).*

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
order must be measured rather than assumed; this is the measuring tool.

```sh
loom iobench /tmp/ckpt/experts.blob --threads 8 --block-mb 1 --reads 64
#   512 reads, 0.54 GB in 0.029s => 18.75 GB/s
```

---
