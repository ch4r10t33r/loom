# Loom CLI reference

Every command and every flag, with what it does and why you would change it.
The [README](../README.md) is the tour; this is the detail.

One binary, `loom`, with seven commands:

| Command | One line |
|---|---|
| [`loom node`](#loom-node) | the daemon: serve inference, share weights with peers |
| [`loom light`](#loom-light) | a no-weights node that delegates to full nodes |
| [`loom run`](#loom-run) | one-shot inference against a loom-format checkpoint |
| [`loom gen`](#loom-gen) | write a synthetic loom-format checkpoint |
| [`loom info`](#loom-info) | inspect a checkpoint and verify its Merkle root |
| [`loom gguf`](#loom-gguf) | GGUF tools: `gen`, `info`, `shard`, `run` |
| [`loom iobench`](#loom-iobench) | disk profiler for the engine's read pattern |

## Conventions

**Flags beat environment variables.** Where a flag lists an env override, the
env var supplies the default and an explicit flag wins.

**Sizes are decimal.** `--ram-gb 1.5` and `--range-mb 0.5` are valid; they are
converted to bytes internally.

**Two model formats, deliberately distinct:**
- **loom format** — a directory with `manifest.loom`, `dense.blob`,
  `experts.blob`. This is the v0 single-node format used by `loom run`,
  `loom gen`, `loom info`, and `loom node --model`.
- **GGUF** — the interchange format real models ship in, used by `loom gguf`
  and by the distribution plane (`loom node --gguf` / `--bootstrap`).

They are not interchangeable: `--model` takes loom format, `--gguf` takes GGUF.

---

## `loom node`

The long-running daemon. It does up to three jobs at once:

1. **Serves inference** over a line-JSON RPC and, optionally, an
   OpenAI-compatible HTTP API.
2. **Participates in GGUF weight distribution** over P2P — holding a subset of
   expert shards, serving them to peers, and fetching the ones it lacks *during
   inference*.
3. **Acts as a bootnode** when it holds a complete expert-sharded GGUF,
   assigning joiners to committees.

Every flag has a default, so bare `loom node` works: it generates a tiny
synthetic model and serves it on `127.0.0.1:8770`.

```
loom node [--model SPEC] [--rpc-addr A] [--rpc-port P]
          [--openai-addr A] [--openai-port P] [--ctx N] [--chat-format F]
          [--p2p-addr A] [--p2p-port P] [--ram-gb X] [--pin-gb Y]
          [--seed S] [--stats FILE] [--no-verify]
          [--gguf FILE | --bootstrap HOST:PORT]
          [--peers H:P,...] [--hold-fraction F] [--range-mb M]
          [--advertise HOST:PORT] [--r-target N]
          [--free-quota TOKENS] [--admin-token TOK]
```

### Which engine actually serves

This trips people up, so it is worth stating plainly. The node picks **one**
engine at startup:

- If an **expert-sharded GGUF store** is attached (via `--gguf` or
  `--bootstrap`) **and** its resident bundle is complete, it serves the
  **distributed GGUF (deepseek2)** engine, fetching missing experts from peers
  at token time.
- Otherwise it serves the **loom-format** model from `--model`.

The startup banner tells you which one you got.

### Model and engine

| Flag | Default | What it does |
|---|---|---|
| `--model SPEC` | `tiny` | The **loom-format** model to serve. Three forms: `tiny` generates a small synthetic checkpoint (cached under `~/.cache/loom/models/tiny`) — useful for smoke tests with no download; a **local directory** containing `manifest.loom`; or a **Hugging Face repo** `[hf:]org/repo[@rev]`, downloaded over HTTPS on first use then served from cache. Resolution is local-first, so a directory that already exists is never re-fetched. Env: `MODEL`. |
| `--ram-gb X` | `4.0` | Total RAM budget for weights. Resident dense weights and the KV cache come out of it first; whatever is left is split between the pinned hot set and the LRU expert cache. The node will not allocate expert storage beyond this, so it is the knob that keeps a box from OOMing. Env: `RAM_BUDGET_GB`. |
| `--pin-gb Y` | `0` | How much of the budget to spend **pinning** the hottest experts permanently in RAM. Pinning is chosen from a `--stats` file written by an earlier run, so the workflow is measure-then-pin: run once with `--stats`, then again with `--pin-gb`. Env: `PIN_GB`. |
| `--seed S` | `42` | Sampling seed, and also the seed for random shard selection when bootstrapping — so a restarted node re-picks the same shards. Env: `SEED`. |
| `--stats FILE` | off | On exit, write per-expert access counts to `FILE` plus a human-readable heat histogram to `FILE.txt` (rank, expert id, count, cumulative coverage, cumulative pin bytes). This is the input to `--pin-gb`. Env: `STATS`. |
| `--no-verify` | verification on | Skip the SHA-256 check on expert blocks read from disk. Verification is what turns a corrupted or poisoned expert into a clean `PoisonedExpert` error instead of silently wrong output. Only disable it if you trust the storage and need the cycles. |
| `--ctx N` | `4096` | Context-length cap when serving the distributed GGUF engine. Caps the model's own context, never raises it. Lower it to cut KV-cache memory. |
| `--chat-format F` | auto-detect | Chat template for `messages[]` on the OpenAI surface: `deepseek`, `chatml`, `llama2`, `llama3`, `gemma`, `mistral`, `generic`. Auto-detected from the GGUF `tokenizer.chat_template`; override when detection guesses wrong. |

### Serving surfaces

| Flag | Default | What it does |
|---|---|---|
| `--rpc-addr A` / `--rpc-port P` | `127.0.0.1` / `8770` | The line-delimited JSON inference RPC. Bind `0.0.0.0` to accept remote clients — but see the warning below. |
| `--openai-addr A` / `--openai-port P` | `<rpc-addr>` / `0` (**off**) | The OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`). Off until you set a port. Serves the same engine and the same ledger as the RPC; client identity comes from `Authorization: Bearer`. |
| `--p2p-addr A` / `--p2p-port P` | `0.0.0.0` / `8771` | Where the P2P protocol listens: shard serving, gossip, heartbeats, and bootnode `JOIN`. |
| `--advertise HOST:PORT` | `127.0.0.1:<p2p-port>` | The address this node tells peers to dial it on. **Set this whenever peers are on other machines** — the default only works on one box. Accepts an IP or a hostname (Compose service names and Kubernetes Service DNS both work). |

> **Neither serving surface has TLS or authentication.** Binding them to
> `0.0.0.0` exposes an unauthenticated inference endpoint on every interface.
> Keep them on loopback, or put an authenticating reverse proxy in front. See
> "Transport security" in [spec/SPEC.md](../spec/SPEC.md).

### Weight distribution (the GGUF plane)

`--gguf` and `--bootstrap` are mutually exclusive: one makes you the origin, the
other makes you a joiner.

| Flag | Default | What it does |
|---|---|---|
| `--gguf FILE` | off | Be the **origin**: shard `FILE`, compute the manifest (SHA-256 per shard, Merkle root = the model version id), and hold and serve everything. If the file is MoE, sharding is expert-aligned and this node also becomes the **bootnode** for the swarm. |
| `--bootstrap HOST:PORT` | off | **Join** an existing swarm: adopt that peer's manifest (root-verified), get a committee and shard assignment from the bootnode, then fetch the assigned shards, each digest-verified. Also seeds the gossip table. |
| `--peers H:P,...` | none | Extra known peers, comma-separated. Seeds the gossip table and gives bootstrap and repair more sources to try. |
| `--hold-fraction F` | `1.0` | Fraction of expert shards this node wants to hold, `0.0`–`1.0`. This is the capacity knob: `0.3` on a small box means it holds ~30% and streams the rest from peers at token time. Resident (non-expert) shards are always held in full regardless. |
| `--range-mb M` | `4.0` | Shard size when *building* a fresh manifest, for non-MoE files that use fixed-size ranges. Ignored when joining (you adopt the peer's layout) and for expert-aligned sharding (where the shard is one expert). |
| `--r-target N` | `2` | Redundancy target when acting as bootnode: how many committee members should hold each shard before the committee counts as saturated. Higher means more copies and more resilience, at more storage. |

### Metering

| Flag | Default | What it does |
|---|---|---|
| `--free-quota TOKENS` | `100000` | Free token allowance per client id before requests are refused with `payment_required` / HTTP 402. A token is one prompt token processed or one token generated. |
| `--admin-token TOK` | empty (**credit disabled**) | Gates the `credit` operation that tops up a client's balance. With no token set the operation is disabled entirely; there is no default to guess. This is the seam a real payment rail replaces. |

---

## `loom light`

A node with **no weights, no store, and no engine** — a few megabytes of
footprint, for a device that cannot hold a model. It exposes the same APIs
locally and forwards every request to a full node, round-robin with failover.

It **forces its own client id** on every forwarded request (a caller-supplied
`client` field or bearer token is dropped), so a light node cannot be used as an
open proxy to spend under someone else's identity. Only inference and the
read-only `tab` operation are forwarded.

```
loom light [--full-nodes H:P[,...]]
           [--openai-port P --openai-full-nodes H:P[,...]]
           [--rpc-addr A] [--rpc-port P] [--openai-addr A] [--client-id ID]
```

Configure at least one surface, or the command exits with usage.

| Flag | Default | What it does |
|---|---|---|
| `--full-nodes H:P,...` | none | Full-node **RPC** endpoints to delegate to. Enables the native RPC surface. |
| `--openai-full-nodes H:P,...` | none | Full-node **OpenAI** endpoints to delegate to. Required if you set `--openai-port`. |
| `--rpc-addr A` / `--rpc-port P` | `127.0.0.1` / `8768` | Where the local RPC listens. Note the port differs from a full node's `8770`, so both can run on one box. |
| `--openai-addr A` / `--openai-port P` | `<rpc-addr>` / `0` (off) | Where the local OpenAI-compatible endpoint listens. |
| `--client-id ID` | `light-anon` | The identity stamped on every forwarded request. Full nodes meter against this, so give each light node its own id if you want per-device accounting. |

---

## `loom run`

One-shot inference against a loom-format checkpoint. No servers, no P2P — it
loads, generates, prints timings, and exits. This is the command for measuring
cache behaviour.

```
loom run <dir> [--prompt STR] [--max-tokens N] [--ram-gb X] [--pin-gb Y]
                [--temp T] [--seed S] [--stats FILE] [--no-verify]
```

`<dir>` is a directory containing `manifest.loom`. Required.

| Flag | Default | What it does |
|---|---|---|
| `--prompt STR` | built-in | The prompt to generate from. |
| `--max-tokens N` | `32` | How many tokens to generate. Env: `MAX_TOKENS`. |
| `--ram-gb X` | `4.0` | RAM budget, as in `loom node`. Lower it to force cache misses and see the streaming path work. Env: `RAM_BUDGET_GB`. |
| `--pin-gb Y` | `0` | Pinned hot-set budget, chosen from `--stats`. Env: `PIN_GB`. |
| `--temp T` | `0.0` | Sampling temperature. `0` (or below) is greedy and therefore deterministic — use it when comparing runs. Env: `TEMP`. |
| `--seed S` | `42` | Sampling seed. Only matters when `--temp > 0`. Env: `SEED`. |
| `--stats FILE` | off | Write per-expert access counts plus `FILE.txt` heat histogram. Env: `STATS`. |
| `--no-verify` | verification on | Skip per-block SHA-256 verification. |

Each run reports tok/s, the pin/LRU/disk hit breakdown, bytes read, digest
failures, and peak RSS. The measure-then-pin loop:

```sh
loom run /tmp/ckpt --prompt "Loom weaves" --ram-gb 1 --stats /tmp/usage.stats
loom run /tmp/ckpt --prompt "Loom weaves" --ram-gb 1 --pin-gb 0.05 --stats /tmp/usage.stats
```

---

## `loom gen`

Writes a synthetic loom-format checkpoint — random weights with a real
structure. Output is gibberish; the point is exercising the machinery
(sharding, caching, distribution) without downloading hundreds of gigabytes.

```
loom gen <dir> [--glm] [--seed N]
```

| Flag | Default | What it does |
|---|---|---|
| `<dir>` | required | Where to write the checkpoint. |
| `--glm` | off (tiny shape) | Use **GLM-5.2's** shape (75 MoE layers x 256 experts) instead of the tiny one. Useful for sizing and layout experiments; it is large. |
| `--seed N` | `42` | Weight-generation seed, so a checkpoint is reproducible. |

---

## `loom info`

Prints a checkpoint's configuration and **verifies its Merkle root** by
recomputing it over the expert index. Use it to confirm a checkpoint is intact.

```
loom info <dir>
```

Reports hidden size, layer counts, expert counts, per-token working-set bytes,
and whether the stored root matches the recomputed one.

---

## `loom gguf`

Tools for GGUF files — the format real models ship in.

```
loom gguf gen   <file> [--seed N] [--data-mb M] [--arch A]
loom gguf info  <file> [--range-mb M]
loom gguf shard <file>
loom gguf run   <file.gguf | store-dir> [--prompt STR] [--max-tokens N]
                [--temp T] [--seed S] [--ctx N]
                [--committee H:P,...] [--peers H:P,...]
```

### `gguf gen` — write a synthetic GGUF fixture

| Flag | Default | What it does |
|---|---|---|
| `--arch A` | `demo` | Architecture to emit. `deepseek2` produces an MLA + MoE structure (the DeepSeek/Kimi family shape). `llama`, `qwen2moe`, `qwen3moe` and `glm4moe` produce GQA + MoE structures, each with exactly the optional pieces that architecture really has, so the fixtures exercise every branch of the shared engine. Anything else emits the plain `demo` tensor blob. These are what the expert-sharding and distribution tests run against, no multi-GB download needed. |
| `--seed N` | `42` | Weight seed. |
| `--data-mb M` | `8` | Approximate payload size, for the non-deepseek2 fixture. |

### `gguf info` — inspect a file

Prints architecture, metadata, tensor table, and the shard layout that would be
produced.

| Flag | Default | What it does |
|---|---|---|
| `--range-mb M` | `4.0` | Shard size to assume when reporting the fixed-range layout. |

### `gguf shard` — show the expert-aligned split

Computes the expert-aligned manifest: one shard per (layer, expert) plus the
resident bundle chunked at 16 MB, with digests and the Merkle version id. Prints
the layout without writing anything. `loom node --gguf` runs the same split
automatically.

### `gguf run` — run a GGUF model

Runs a `.gguf` file directly, dispatching on `general.architecture`:
`deepseek2` uses the MLA engine, and `llama` (including Mixtral), `qwen2moe`,
`qwen3moe` and `glm4moe` use the shared GQA engine. Point it at a **store
directory** instead and it runs *distributed*: held shards come from the local
sparse file and missing experts are fetched from peers inside the token loop.
The store's own model file selects the engine, so both families distribute.

| Flag | Default | What it does |
|---|---|---|
| `--prompt STR` | `Once upon a time` | The prompt. |
| `--max-tokens N` | `128` | Tokens to generate. |
| `--temp T` | `0.0` | Temperature; `0` is greedy/deterministic. |
| `--seed S` | `42` | Sampling seed. |
| `--ctx N` | `4096` | Context-length cap. Caps the model's own value, never raises it. |
| `--peers H:P,...` | none | Peers to fetch missing shards from (store-directory mode). The mesh fallback. |
| `--committee H:P,...` | none | Committee members, tried **before** `--peers`. This is the SPEC query path: committee first, then mesh. |

---

## `loom iobench`

Profiles the disk with the access pattern the engine actually issues —
parallel random reads of expert-sized blocks — rather than a generic sequential
benchmark. Use it to decide whether a box's storage can keep up, and to size
`--ram-gb` against real device throughput.

```
loom iobench <file> [--threads N] [--block-mb M] [--reads R]
```

| Flag | Default | What it does |
|---|---|---|
| `<file>` | required | File to read from. Any large file works; an `experts.blob` is representative. |
| `--threads N` | CPU count | Concurrent readers. Models the parallel per-layer expert fetches. |
| `--block-mb M` | `19` | Block size in MB. The default is one int4 expert block for a GLM-class model, which is the read size that matters. |
| `--reads R` | `64` | Reads per thread. |

---

## Environment variables

Only these are read, and a flag always wins:

| Variable | Used by | Equivalent flag |
|---|---|---|
| `MODEL` | `node`, `run` | `--model` / `<dir>` |
| `RAM_BUDGET_GB` | `node`, `run` | `--ram-gb` |
| `PIN_GB` | `node`, `run` | `--pin-gb` |
| `MAX_TOKENS` | `run` | `--max-tokens` |
| `TEMP` | `run` | `--temp` |
| `SEED` | `node`, `run` | `--seed` |
| `STATS` | `node`, `run` | `--stats` |
| `HOME` | all | Chooses the cache root (`$HOME/.cache/loom/models`, falling back to `./.loom-cache`). Setting it per-process is how the docs run several nodes on one box with separate stores. |

## See also

- [README](../README.md) — overview, quick start, protocol examples
- [spec/SPEC.md](../spec/SPEC.md) — the p2p layer: roles, committees, wire messages, trust model
- [docs/ROADMAP.md](ROADMAP.md) — status and what is planned
