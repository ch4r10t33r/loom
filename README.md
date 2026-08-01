# Loom — distributed expert cache for large MoE inference

[![CI](https://github.com/ch4r10t33r/loom/actions/workflows/ci.yml/badge.svg)](https://github.com/ch4r10t33r/loom/actions/workflows/ci.yml)

Loom runs huge Mixture-of-Experts models (GLM, DeepSeek, Qwen, Kimi) on
swarms of commodity machines. The thesis: in a model like GLM 5.2, only
~11 GB of expert weights out of ~370 GB are touched per token, so **compute
stays node-local and the weights are what move** — streamed from disk
through a tiered cache on one box, and shared across nodes as
content-addressed byte ranges of a GGUF file, with gossip discovery, eager
churn repair, and Merkle-verified integrity.

One static binary (Zig, no runtime dependencies), CPU SIMD + Metal (macOS)
or Vulkan (Linux/Windows) GPU backends, and an OpenAI-compatible API.

- **Why / design:** [whitepaper](whitepaper/WHITEPAPER.md) · [roadmap](docs/ROADMAP.md) · [p2p spec](spec/SPEC.md)
- **Status:** v0 (single-node expert streaming) done; v1 (distributed weight
  sharing) working across real machines; v2 (untrusted peers) design-only.

## Quickstart

Install (macOS/Linux; detects your platform, verifies checksums, installs to
`/usr/local/bin` — on Windows grab the release zip):

```sh
curl -fsSL https://raw.githubusercontent.com/ch4r10t33r/loom/main/install.sh | sh
```

Join the devnet (installs loom if missing, syncs its share of expert shards
from peers — no model download — and serves an OpenAI API on `:8772`):

```sh
curl -fsSL https://raw.githubusercontent.com/ch4r10t33r/loom/main/scripts/join-devnet.sh | sh
```

Or a 60-second local tour, no network:

```sh
# zero-config node: generates a tiny model, serves RPC :8770 / P2P :8771
loom node
printf '{"prompt":"hello","max_tokens":16}\n' | nc -w 3 127.0.0.1 8770

# run a real GGUF model (any small GGUF works)
curl -LO https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf
loom gguf run stories15M-q4_0.gguf --prompt "Once upon a time"
```

A chat UI is compiled into the binary: run `loom node` and open
`http://127.0.0.1:8555` ([details](docs/CHAT-UI.md)).

## Build from source

Targets **Zig 0.16.0** (pinned in `build.zig.zon`;
[anyzig](https://github.com/marler8997/anyzig) resolves it automatically).

```sh
zig build                        # debug binary -> zig-out/bin/loom
zig build -Doptimize=ReleaseSafe # what releases ship: fast, bounds checks on
zig build test                   # unit tests
zig build -Dgpu=vulkan           # GPU backend on Linux/Windows (Metal is automatic on macOS)
zig build -Dtarget=x86_64-linux-musl -Doptimize=ReleaseSafe -p out   # cross-compile
```

Releases are ReleaseSafe, not ReleaseFast: loom is a daemon that writes
peer-supplied weight shards into buffers, and the ~11% speed difference is
whether an out-of-bounds write is caught or executed. Releases are cut by
pushing a tag; see
[`.github/workflows/release.yml`](.github/workflows/release.yml). Docker
images (linux/amd64 + arm64) publish to `ghcr.io/ch4r10t33r/loom:latest`
([docker guide](docs/DOCKER.md)).

## Commands

| Command | Purpose |
|---|---|
| [`loom node`](docs/COMMANDS.md#loom-node--run-an-inference--weight-sharing-node) | the daemon: load a model, serve inference (native RPC + OpenAI API), share/sync/repair GGUF weight ranges over P2P |
| [`loom gguf`](docs/COMMANDS.md#loom-gguf--gguf-tools-gen--info--shard--run) | GGUF tools: inspect any GGUF, **shard by expert**, run deepseek2/llama/qwen/glm models locally or from a partial store |
| [`loom light`](docs/COMMANDS.md#loom-light--delegating-light-node) | light node: no weights or engine; delegates RPC/OpenAI to full nodes, metered |
| [`loom run`](docs/COMMANDS.md#loom-run--one-shot-local-inference) | one-shot local inference against a loom checkpoint (no servers) |
| [`loom gen`](docs/COMMANDS.md#loom-gen--generate-a-synthetic-checkpoint) / [`loom info`](docs/COMMANDS.md#loom-info--inspect--verify-a-checkpoint) | create / inspect + verify loom-format checkpoints |
| [`loom bench`](docs/PERFORMANCE.md) / [`loom iobench`](docs/COMMANDS.md#loom-iobench--disk-profiler) | kernel invariants / disk profiler for the engine's random-read pattern |

`loom node --network devnet|testnet|mainnet` selects a
[named network](docs/NETWORKS.md) — each network serves exactly one model and
rejects peers from others, chainId-style. Every command and flag, with
defaults: [CLI reference](docs/CLI.md). Worked examples with real output:
[command tour](docs/COMMANDS.md).

## Documentation

- [Command tour](docs/COMMANDS.md) — every command, worked examples, protocols (RPC, OpenAI, P2P wire)
- [CLI reference](docs/CLI.md) — every flag, defaults, env overrides
- [Install details](docs/INSTALL.md) — manual install, checksum verification, Gatekeeper
- [Multi-node walkthrough](docs/MULTI-NODE-WALKTHROUGH.md) — shard a real MoE model across a swarm on one host
- [Two-machine test](docs/TWO-MACHINE-TEST.md) — real cross-machine swarm with honest tok/s
- [Named networks](docs/NETWORKS.md) — devnet / testnet / mainnet, models and policies
- [Performance](docs/PERFORMANCE.md) — CPU SIMD numbers, batching, benchmarking rules
- [GPU backends](docs/GPU-BACKENDS.md) — Metal and Vulkan design and status
- [Chat UI & node console](docs/CHAT-UI.md)
- [Docker](docs/DOCKER.md)
- [GLM 5.2 target deployment plan](docs/GLM52-PLAN.md) · [readiness audit](docs/GLM52-READINESS.md)
- [Repository layout & source map](docs/SOURCE-MAP.md)
- [Roadmap](docs/ROADMAP.md) · [p2p spec](spec/SPEC.md) · [whitepaper](whitepaper/WHITEPAPER.md)

## Status & honest gaps

- **v0 (single-node expert streaming): done**, except token-exact validation
  against a `transformers` oracle on real GLM-5.2 weights — that needs the
  real converted weights (~370 GB). The synthetic model exercises every code
  path; a real checkpoint drops in without engine changes.
- **v1 (distributed weight sharing): first cuts working** — GGUF range
  sharding, multi-peer boot sync, gossip discovery, eager churn repair,
  network ids, RAG chunk gossip. Remaining: ENR integration, real gossipsub
  transport, majority-hardfork coordination. See the
  [roadmap](docs/ROADMAP.md).
- **v2 (untrusted peers)** is design-only.
