# Loom — distributed expert cache & weight-sharing node for large MoE inference

[![CI](https://github.com/ch4r10t33r/loom/actions/workflows/ci.yml/badge.svg)](https://github.com/ch4r10t33r/loom/actions/workflows/ci.yml)

Loom is a research prototype (Zig, no runtime dependencies) exploring the design
in [`CLAUDE.md`](CLAUDE.md): for a huge Mixture-of-Experts model like GLM-5.2,
keep **compute node-local** and make the **weights** the thing that moves —
streamed from disk through a tiered cache on one box (v0), and shared across
nodes as content-addressed byte ranges of a GGUF file (v1, in progress — see
[`docs/ROADMAP.md`](docs/ROADMAP.md)).

One binary, `loom`, provides:

> **Walkthrough — shard a real MoE model across a swarm:
> [`docs/MULTI-NODE-WALKTHROUGH.md`](docs/MULTI-NODE-WALKTHROUGH.md)** — which
> models work, how to shard them, and how to run several nodes on one host.
>
> **Two physical machines:
> [`docs/TWO-MACHINE-TEST.md`](docs/TWO-MACHINE-TEST.md)** — measure your link
> first, then run a real cross-machine swarm and get honest tok/s numbers.
>
> **Full CLI reference: [`docs/CLI.md`](docs/CLI.md)** — every command and every
> flag, with defaults, env overrides, and when you would change them. The
> sections below are a tour; that document is the detail.

| Command | Purpose |
|---|---|
| [`loom node`](#loom-node--run-an-inference--weight-sharing-node) | the daemon: load a model, serve inference over RPC, share/sync/repair GGUF weight ranges over P2P with gossip discovery |
| [`loom run`](#loom-run--one-shot-local-inference) | one-shot local inference against a loom checkpoint (no servers) |
| [`loom gen`](#loom-gen--generate-a-synthetic-checkpoint) / [`loom info`](#loom-info--inspect--verify-a-checkpoint) | create / inspect+verify loom-format checkpoints |
| [`loom gguf`](#loom-gguf--gguf-tools-gen--info--shard--run) | GGUF tools: make a fixture, inspect a file, **shard by expert**, run deepseek2/llama/qwen/glm models |
| [chat UI](#chat-ui) | a single-page chat app compiled into the binary, served at `http://127.0.0.1:8555` |
| [`loom light`](#loom-light--delegating-light-node) | **light node**: no weights/engine; delegates the native RPC and/or OpenAI API to full nodes, metered by them |
| [`loom iobench`](#loom-iobench--disk-profiler) | disk profiler for the random-read pattern the engine issues |

## Install

Prebuilt binaries are attached to each
[release](https://github.com/ch4r10t33r/loom/releases). No runtime
dependencies — the Linux builds are statically linked against musl, so they run
on any distro.

| platform | asset suffix |
|---|---|
| macOS, Apple silicon | `aarch64-macos` |
| macOS, Intel | `x86_64-macos` |
| Linux, x86-64 | `x86_64-linux` |
| Linux, arm64 | `aarch64-linux` |

Windows is not built yet.

One line, which detects your platform, verifies the download against the
release checksums, and installs to `/usr/local/bin`:

```sh
curl -fsSL https://raw.githubusercontent.com/ch4r10t33r/loom/main/install.sh | sh
```

It takes `--version vX.Y.Z` and `--dir PATH`, or the same as `LOOM_VERSION` and
`LOOM_INSTALL_DIR`. On Windows it stops with an explanation and points at WSL2,
where the Linux build works. While the repository is private it needs a
`GITHUB_TOKEN`, or a signed-in `gh`.

By hand instead:

```sh
VER=v0.1.0; PLAT=aarch64-macos       # pick yours from the table
curl -fsSL -O https://github.com/ch4r10t33r/loom/releases/download/$VER/loom-$VER-$PLAT.tar.gz
curl -fsSL -O https://github.com/ch4r10t33r/loom/releases/download/$VER/SHA256SUMS
shasum -a 256 -c SHA256SUMS --ignore-missing
tar -xzf loom-$VER-$PLAT.tar.gz
sudo mv loom-$VER-$PLAT/loom /usr/local/bin/
loom version
```

`loom version` reports the version, the commit it was built from, and its
target triple — worth quoting in any bug report.

On macOS, Gatekeeper quarantines unsigned downloads. Clear it with
`xattr -d com.apple.quarantine /usr/local/bin/loom`, or right-click and Open
once. The binaries are not code-signed or notarized.

Docker images are published to `ghcr.io/ch4r10t33r/loom:latest` on every push
to `main`.

## Performance

Kernels are SIMD (`@Vector`, so portable rather than per-architecture
intrinsics) and matvecs run row-parallel across a worker pool. Measured on a
10-core Apple M5, TinyLlama 1.1B Q4_K_M, median of three 64-token runs:

| | tok/s | vs baseline |
|---|---|---|
| scalar, single-threaded (original) | 1.1 | 1x |
| SIMD, single-threaded | 6.0 | 5.5x |
| SIMD + 8 threads | 26.0 | 23.6x |
| + int8 activations, 1 thread | 15.0 | 13.6x |
| + int8 activations, 8 threads | 51.9 | 47x |
| **+ vectorized attention** | **57.8** | **53x** |

Prefill is batched separately, because the prompt is known up front: the same
unpacked weight serves several tokens instead of being unpacked once per
token. A 145-token prefill on the same model, best of three:

| `--batch` | time |
|---|---|
| 1 (off) | 2.65 s |
| 4 | 2.01 s |
| **8** | **1.87 s** |

That is 1.4x rather than the 2.4x the kernel microbenchmark shows in
isolation, because attention is quadratic in prompt length and is not batched
— it is bound by the KV cache rather than by weight reads, so there is nothing
to amortize there.

The last step is the one worth understanding. Profiling the kernels showed
**76-92% of a quantized matvec was the dequantize, not the dot** — each block
was expanded into a 256-float scratch buffer and pushed through L1 only to be
read straight back. Quantizing the *activation* vector to int8 once per matvec
and dotting it against the packed weights as integers removes that buffer
entirely and uses lanes four times as wide.

That step is an approximation the earlier ones were not: the activation
carries roughly 0.4% per-element error, so results are no longer bit-identical
to the dequantize path. Over a long dot those errors are independent and
largely cancel, and it is what every production CPU inference engine does.

Row-splitting is exact, not approximate: each output row is one independent
dot product, so there is no reassociation and results are bit-identical to the
serial path at any thread count. The tests assert that rather than assuming it.

Thread count defaults to `cpu_count - 2` — a node keeps p2p, gossip and repair
threads running during a generation, and past that point oversubscription
costs more than the extra cores return. Override with `--threads N`;
`--threads 1` disables the pool.

This is still CPU-only, and the CPU path is not exhausted: Q4_K runs at about
5.8 GB/s against roughly 70 GB/s of achievable bandwidth. GPU backends —
Metal on Apple, Vulkan on Linux and Windows — are planned in
[`docs/GPU-BACKENDS.md`](docs/GPU-BACKENDS.md), following the structure
[ZINC](https://github.com/zolotukhin/zinc) used for its Apple Silicon
bring-up.

## Benchmarking

```sh
loom bench          # kernel timings plus invariant checks
loom bench --check  # non-zero exit if an invariant fails; CI runs this
```

CI gates on **invariants** ("batching beats unbatched", "threads beat one
thread", "the selected path beats the alternative") rather than on wall-clock,
because a shared runner's absolute timings are noise while those relationships
are properties of the code. See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## Chat UI

`loom node` serves a small chat app at **`http://127.0.0.1:8555`** (change with
`--ui-port`, disable with `--ui-port 0`). It is a single page compiled into the
binary, so there is nothing to install and nothing to configure:

```sh
loom node --gguf model.gguf
# then open http://127.0.0.1:8555
```

Streaming replies, temperature and max-token controls, and per-response
timings that separate the two rates worth knowing apart — **decode speed** and
**time to first token**. On a partial node the first token also pays for every
expert fetch the prefill needed, so folding them together understates
steady-state throughput badly.

The header shows the live peer count and the local-hit rate, polled every few
seconds, so you can watch a node discover peers while you use it.

It runs on its own listener but shares the HTTP implementation and the
generator with the OpenAI API, so the page is same-origin with the endpoint it
calls: no CORS, and no host to configure in the page. Like the API it has **no
TLS and no authentication**, so it binds to loopback by default; do not widen
`--ui-addr` without an authenticating proxy in front.

If the node is serving one of loom's synthetic fixtures, the UI says so. Those
have random weights and answer with meaningless text by construction, which
otherwise reads as a broken model rather than one that was never trained.

## Node console

A running node prints a status line at a fixed interval (`--status-secs`,
default 30, `0` to disable), because the interesting facts change without any
request arriving:

```
status  peers 3  committee 2  shards 4021/6289 (63.9%)  local-hit 87.4%  up 12m
```

Stable field order, so `grep` and `awk` work on it.

## Build from source

Targets **Zig 0.16.0** (pinned in `build.zig.zon`; [anyzig](https://github.com/marler8997/anyzig)
resolves it automatically).

```sh
zig build                        # debug binary -> zig-out/bin/loom
zig build -Doptimize=ReleaseFast # ~10x faster inference
zig build test                   # unit tests
zig build run -- <args>          # build + run in one step
```

Cross-compiling is a flag, since Zig ships every target:

```sh
zig build -Doptimize=ReleaseFast -Dtarget=x86_64-linux-musl -p out
```

Releases are cut by pushing a tag (`v0.1.0`), which builds all four targets and
publishes them; see [`.github/workflows/release.yml`](.github/workflows/release.yml).

## Docker

A multi-stage [`Dockerfile`](Dockerfile) builds a ReleaseFast binary (Zig 0.16.0
is fetched by anyzig at build time) into a ~120 MB Debian-slim runtime image.

A prebuilt image is published to the GitHub Container Registry on every push to
`main`, tagged `latest` and with the commit SHA (pin the SHA for anything you
need to reproduce or roll back):

```sh
docker pull ghcr.io/ch4r10t33r/loom:latest
```

Only images that pass the CI smoke test are published, and the image pushed is
the one that was tested, not a rebuild. To build locally instead:

```sh
docker build -t loom:dev .

# run a single node (serves the built-in synthetic model).
# Ports are published to 127.0.0.1: the RPC and OpenAI surfaces have no TLS and
# no authentication, so do not expose them on a routable interface without an
# authenticating proxy in front.
docker run --rm -p 127.0.0.1:8770:8770 -p 127.0.0.1:8772:8772 loom:dev \
  node --rpc-addr 0.0.0.0 --openai-port 8772
curl -s localhost:8772/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

The image exposes `8770` (RPC), `8771` (P2P), `8772` (OpenAI), runs as a non-root
user, and persists the model cache in the `/home/loom/.cache/loom` volume. The
base image is pinned by digest and the build verifies the compiler tarball's
SHA-256, so neither a moved tag nor a swapped release asset can change what is
built. The compose demo additionally drops all capabilities, sets
`no-new-privileges`, and caps memory/PIDs.

**Distributed swarm** — [`docker-compose.yml`](docker-compose.yml) brings up a
two-node swarm (an origin that generates + serves a synthetic deepseek2 GGUF, and
a partial node that holds ~30% of the experts and fetches the rest from the
origin at token time), no external model needed:

```sh
docker compose up --build
# node2's OpenAI port; a hit_rate < 1 on the RPC response = token-loop peer fetch
curl -s localhost:8782/v1/completions -d '{"prompt":"the","max_tokens":8}'
```

Peers can be addressed by IP **or hostname** — the p2p layer resolves names via a
minimal DNS client ([`src/p2p/dns.zig`](src/p2p/dns.zig): IP literal, then
`/etc/hosts`, then a UDP A-query to `/etc/resolv.conf`'s first nameserver), so the
compose demo addresses peers by Compose service name (`origin`), and Kubernetes
Service DNS works the same way.

## 60-second tour

```sh
# inference on a synthetic checkpoint, then as a served node
loom node                                   # zero-config: generates a tiny model, serves RPC :8770 / P2P :8771
printf '{"prompt":"hello","max_tokens":16}\n' | nc -w 3 127.0.0.1 8770

# run a real GGUF model (download any small GGUF first)
curl -LO https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf
loom gguf run stories15M-q4_0.gguf --prompt "Once upon a time"
```

---

## `loom node` — run an inference + weight-sharing node

*Every flag for this command: [docs/CLI.md](docs/CLI.md#loom-node).*

```
loom node [--model SPEC] [--rpc-addr A] [--rpc-port P] [--openai-addr A] [--openai-port P]
          [--p2p-addr A] [--p2p-port P] [--ram-gb X] [--pin-gb Y] [--seed S]
          [--stats FILE] [--no-verify] [--gguf FILE | --bootstrap HOST:PORT]
          [--peers H:P,H:P,...] [--hold-fraction F] [--range-mb M] [--advertise HOST:PORT]
```

Starts a long-running node that (a) loads a loom-format model and serves
inference over the native **RPC** and an optional **OpenAI-compatible HTTP
API**, (b) optionally participates in **GGUF weight distribution** over **P2P**,
with gossip-based peer discovery and eager churn repair. Every flag has a
default, so `loom node` alone works.

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
| `--openai-addr` / `--openai-port` | `<rpc-addr>` / `0` (off) | Where the **OpenAI-compatible HTTP API** listens. Set `--openai-port` to enable it (e.g. `8772`). Serves the same engine as the RPC, metered by `Authorization: Bearer` client id. |
| `--ctx N` | `4096` | Context-length cap when serving a distributed GGUF engine. |
| `--chat-format F` | auto | Override the chat template (`deepseek`/`chatml`/`llama2`/`llama3`/`gemma`/`mistral`/`generic`); default auto-detects from GGUF metadata. |
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
| `--r-target N` | `2` | Committee redundancy target when acting as bootnode (expert-sharded origin). |

**Committees (SPEC.md).** An expert-sharded origin automatically acts as the
**bootnode**: `JOIN` assigns each connecting node to a **shard committee** and
a least-covered-first want-set, so every committee converges to holding the
complete shard set with redundancy `--r-target` (default 2) by construction.
Joiners sync from committee members first, then the bootnode; committee
members heartbeat each other every 5 s and log liveness transitions. When all
committees are saturated, the next joiner opens a new one. See
[spec/SPEC.md](spec/SPEC.md) for the full p2p-layer spec (roles, invariants, query
path, wire protocol).

Two loops run alongside the servers:

- **Gossip (every 3 s):** dial every known peer, exchange binary Announce /
  AnnounceBatch frames (addr, committee id, manifest version, holdings seq +
  bitmap; snappy-compressed). Discovery is transitive, and because announces
  carry committee ids the table doubles as the **gossip-derived committee
  view** — earlier committee members discover later joiners automatically and
  start heartbeating them.
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
| `HOLDINGS` | `HOLDINGS <hex bitmap>` | which ranges this node holds (bit i = range i) — the compact summary destined for ENR metadata |
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

*Every flag for this command: [docs/CLI.md](docs/CLI.md#loom-run).*

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

## `loom light` — delegating light node

*Every flag for this command: [docs/CLI.md](docs/CLI.md#loom-light).*

```
loom light [--full-nodes H:RPC_PORT[,...]] [--openai-port P --openai-full-nodes H:OPENAI_PORT[,...]]
           [--rpc-addr A] [--rpc-port P] [--openai-addr A] [--client-id ID]
```

For low-memory devices: holds no weights, no store, no engine (megabytes of
footprint). It exposes the native line-JSON RPC and/or an **OpenAI-compatible
HTTP API**, and transparently delegates every request to a full node — round-robin
with failover — forcing its `--client-id` so a caller cannot spend under another
identity (a caller-supplied `client` field or `Authorization` bearer is dropped).
Configure at least one surface: `--full-nodes` for the native RPC (default port
8768), and/or `--openai-port` with `--openai-full-nodes` for the OpenAI surface.

Full nodes **meter** clients: the native RPC responses carry `cost` (prompt +
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

*Every flag for this command: [docs/CLI.md](docs/CLI.md#loom-gen).*

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

*Every flag for this command: [docs/CLI.md](docs/CLI.md#loom-info).*

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

## `loom gguf` — GGUF tools (`gen` / `info` / `shard` / `run`)

*Every flag for this command: [docs/CLI.md](docs/CLI.md#loom-gguf).*

### `loom gguf gen` — synthetic GGUF fixture

```
loom gguf gen <file> [--seed N] [--data-mb M] [--arch A]
```

Writes a small valid GGUF v3 file. Default (`demo`): metadata + two f32 tensors
of deterministic data — a distribution payload, **not** a runnable model.

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
| `qwen2moe` | QKV biases, sigmoid-**gated** shared expert, and the one arch that does *not* renormalize gates |
| `qwen3moe` | Q/K norm, and a head_dim that is deliberately not `dim / n_heads` |
| `glm4moe` | QKV biases, Q/K norm, sigmoid routing with selection bias, plain shared expert, `post_attention_norm` in place of `ffn_norm`, a leading dense layer, partial RoPE, and a trailing NextN block the forward pass must skip |

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

### `loom gguf shard` — expert-aligned shard manifest

```
loom gguf shard <file>
```

The sharding tool: parses the GGUF tensor table and builds the expert-aligned
manifest — one shard per (layer, expert) as a 3-extent list over the
`ffn_{gate,up,down}_exps` tensors, resident bundle chunked at 16 MB — then
prints the summary. `loom node --gguf` runs the same split automatically
(expert mode when the file has expert tensors, fixed ranges otherwise).

```sh
loom gguf shard DeepSeek-V2-Lite.Q4_K_M.gguf
#   shards         1737 total = 73 resident + 1664 expert
#   resident       0.780 GB in 73 chunks (held by every node)
#   expert shards  4.98..6.02 MB (avg 5.46 MB), 8.87 GB routed corpus
#   metadata       manifest 204.3 KB, holdings bitmap 218 B
```

For GLM 5.2 this yields the planned 19,200 expert shards (~19 MB each) + a
~10 GB resident bundle. `--hold-fraction` applies to *expert* shards only —
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
  **Validated on real weights**: DeepSeek-V2-Lite Q4_K_M (15.7B MoE, 27 MLA
  layers, 64 experts) produces correct factual completions ("The capital of
  France is Paris.") on one CPU core.
- **`llama` / `qwen2moe` / `qwen3moe` / `glm4moe`** — one shared GQA engine.
  These differ only in optional pieces bolted onto the same skeleton, so the
  engine detects them from the tensors the file contains rather than from a
  table of per-architecture beliefs: QKV biases, per-head Q/K RMSNorm before
  RoPE, a dense or mixture-of-experts FFN, a shared expert that may be
  sigmoid-gated, leading dense layers, a post-attention norm standing in for
  `ffn_norm`, and trailing MTP/NextN blocks that are skipped. The one fact
  pinned per architecture is the RoPE style — adjacent-pair NORM for `llama`,
  split-half NEOX for the rest — because getting it wrong produces fluent but
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
no wholesale dequantization. The IQ and MXFP4 types are *codebook* quants —
a block stores an index into a static grid table rather than a value to scale —
so they decode through `src/gguf/iq.zig` against tables transcribed from
llama.cpp. Because a wrong table entry there would silently corrupt weights
instead of failing, all ten decoders are checked **bit-for-bit against
llama.cpp's own `dequantize_row_*` output** on golden vectors
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
stories260K F32/GQA, stories15M Q4_0 and Q8_0, DeepSeek-V2-Lite Q4_K_M) —
coherent English confirms kernels, attention, RoPE convention, and tokenizer
simultaneously. `loom node` can also serve the distributed GGUF (deepseek2)
engine directly over its RPC and OpenAI surfaces — see [Serving a distributed
GGUF model through the node](#serving-a-distributed-gguf-model-through-the-node).

### Distributed run: inference from a partial store

Point `gguf run` at a **store directory** (instead of a .gguf) and give it
peers: held shards come from the local sparse file; missing experts are
**fetched from peers inside the token loop** — in parallel per MoE layer,
round-robin across holders, digest-verified before touching disk, then
persisted (so the node's holdings grow with use and gossip advertises them:
fetch-on-demand doubles as organic heat replication).

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

`loom gguf run` above is a one-shot CLI. The **node** serves the same
distributed engine as a long-running service over its RPC and OpenAI surfaces:
when `loom node` is given an expert-sharded GGUF (`--gguf` origin, or
`--bootstrap` to sync a partial store) and its resident bundle is complete, it
serves the deepseek2 engine and **fetches missing experts from peers inside the
token loop**. Otherwise it serves the loom-format `--model`. `--ctx N` caps the
context length (default 4096).

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
sequential cold expert fetches); it warms as fetched experts are persisted. This
is the serving-first, latency-later behavior the design calls for. Chat requests
are rendered with the model's detected chat template (`--chat-format` to
override); `--ctx N` caps context length.

---

## Target deployment: GLM 5.2 on a swarm of 16 GB machines (plan)

The end-state this repo is building toward (see [docs/ROADMAP.md](docs/ROADMAP.md);
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

*Every flag for this command: [docs/CLI.md](docs/CLI.md#loom-iobench).*

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

## Repository layout

```
whitepaper/   the living whitepaper (updated with every design decision)
spec/         p2p-layer specification (SPEC.md)
docs/         roadmap and planning documents
src/main.zig  CLI entry point
src/core/     primitives: hashing/Merkle, tensor math, int4 quant, stats, iobench
src/engine/   the loom-format MoE engine (MLA, router, expert cache, checkpoints)
src/gguf/     GGUF plane: parser, GGML + IQ kernels, MLA + GQA engines, shared MoE routing, BPE
src/p2p/      distribution: wire frames, gossip, committees, sync, token-loop fetch
src/node/     the daemon: node orchestration, RPC server, model resolver
```

## Source map

| Module | Role |
|---|---|
| `engine/model.zig` | `ModelConfig`; GLM-5.2 shape + runnable `tiny` shape; expert/working-set sizing |
| `core/quant.zig` | loom int4 expert format (f32 scale/32 weights) quantize + fused matvec |
| `core/tensor.zig` | RMSNorm, softmax, SwiGLU, dense matvec, partial RoPE |
| `engine/attention.zig` | MLA: q/kv-LoRA, partial RoPE, compressed-latent KV cache |
| `engine/moe.zig` | DeepSeek-V3 sigmoid router (top-k + shared expert); streamed int4 expert FFN |
| `engine/checkpoint.zig` | loom on-disk format: content-addressed, Merkle-rooted, deduped |
| `engine/expert_cache.zig` | tiered expert cache: pinned hot-set → LRU → pread, usage stats, digest verify |
| `engine/forward.zig` / `engine/engine.zig` | forward step wiring; engine lifecycle, RAM-budget → cache sizing |
| `node/node.zig` | `loom node` orchestration: model → engine → RPC/P2P/gossip/repair |
| `node/hf.zig` | model resolver: local dir / synthetic / Hugging Face download (local-first) |
| `node/generator.zig` | generation abstraction over the loom-format engine and the distributed GGUF engines (MLA and GQA); both serve paths call it |
| `gguf/chat_template.zig` | per-model chat-template detection + rendering for OpenAI `messages[]` (deepseek/chatml/llama2/llama3/gemma/mistral/generic) |
| `gguf/special.zig` | special-token matcher: splices control / user-defined tokens (chat markers) to atomic ids during BPE + SPM encoding |
| `node/rpc.zig` | JSON-over-TCP inference server (concurrent connections, serialized generate) |
| `node/openai.zig` | OpenAI-compatible HTTP API: `/v1/chat/completions`, `/v1/completions`, `/v1/models` (shares the engine + meter with `rpc.zig`) |
| `node/light.zig` | light-node native-RPC delegator (forces client id, round-robin failover) |
| `node/light_openai.zig` | light-node OpenAI delegator: metered reverse proxy to full-node OpenAI endpoints |
| `p2p/p2p.zig` | P2P line protocol: expert directory, weight ranges, gossip |
| `p2p/weights.zig` | range-sharded GGUF store: manifest, version id, holdings/wanted bitmaps, verified IO |
| `p2p/sync.zig` | peer sync client: manifest adoption, root verification, multi-peer range fetch |
| `p2p/peers.zig` | dynamic peer table shared by gossip/repair/P2P threads |
| `p2p/gossip.zig` | 3 s gossip loop: announce self, merge peers-of-peers |
| `p2p/dns.zig` | minimal DNS resolver for peer hostnames (IP literal / /etc/hosts / UDP A-query); Zig std has none |
| `gguf/gguf.zig` | GGUF v2/v3 parser (metadata incl. tokenizer arrays, tensor table) + fixture writer |
| `gguf/ggml.zig` | GGML kernels: F32/F16/Q4_0/Q8_0 fused matvec + row dequant |
| `gguf/llama.zig` | llama-arch engine over mmap'd GGUF: GQA, NORM RoPE, SwiGLU, SPM tokenizer |
| `gguf/deepseek.zig` | deepseek2-arch engine (Kimi/DeepSeek/GLM): MLA + MoE routing over mmap'd GGUF |
| `core/stats.zig` | RSS, usage histograms, STATS→PIN hot-set selection |
| `core/iobench.zig` | parallel random-read disk profiler |
| `engine/gen_checkpoint.zig` | deterministic synthetic checkpoint generator |
| `core/hash.zig` | SHA-256 content addressing + Merkle root |
| `engine/sampler.zig` / `tokenizer.zig` | greedy/temperature sampling; byte-level tokenizer (synthetic model) |

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
  models through the node's RPC. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the
  requirements of record and decisions (random+redundant placement, ENR +
  global gossip topic, maximally eager repair).
- **v2 (untrusted peers)** is design-only (see `CLAUDE.md`).
