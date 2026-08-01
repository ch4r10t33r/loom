# Loom

[![CI](https://github.com/ch4r10t33r/loom/actions/workflows/ci.yml/badge.svg)](https://github.com/ch4r10t33r/loom/actions/workflows/ci.yml)

Loom runs large Mixture-of-Experts models (GLM, DeepSeek, Qwen, Kimi) on
groups of ordinary machines. The idea is simple. In a model like GLM 5.2,
each token touches only about 11 GB of the ~370 GB of expert weights, so
there is no need for any one machine to hold the whole model. Compute stays
local to each node. What moves between nodes is weights: content-addressed
byte ranges of a GGUF file, found through gossip, repaired on churn, and
verified against a Merkle manifest before use.

Loom is a single static Zig binary with no runtime dependencies. It has SIMD
CPU kernels, a Metal backend on macOS, a Vulkan backend on Linux and
Windows, and serves an OpenAI-compatible API.

Design rationale is in the [whitepaper](whitepaper/WHITEPAPER.md), current
status in the [roadmap](docs/ROADMAP.md), and the wire protocol in the
[p2p spec](spec/SPEC.md). Single-node expert streaming (v0) is done.
Distributed weight sharing (v1) works across real machines. Untrusted peers
(v2) is a design only.

## Quickstart

Install on macOS or Linux (detects your platform, verifies checksums,
installs to `/usr/local/bin`; on Windows download the release zip instead):

```sh
curl -fsSL https://raw.githubusercontent.com/ch4r10t33r/loom/main/install.sh | sh
```

Join the devnet. This installs loom if it is missing, then starts a node
that downloads its share of the model's GGUF shards from peers to local
disk (about a fifth of the ~73 GB GLM-4.5-Air checkpoint by default) and
serves an OpenAI API on `:8772`:

```sh
curl -fsSL https://raw.githubusercontent.com/ch4r10t33r/loom/main/scripts/join-devnet.sh | sh
```

Or take a 60-second local tour with no network involved:

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

## Requirements

- **CPU**: any 64-bit x86-64 or arm64. The kernels are portable SIMD. A GPU
  is optional and used automatically where present: Metal on Apple silicon,
  Vulkan (`-Dgpu=vulkan` builds) on Linux and Windows.
- **RAM**: 8 GB minimum, 16 GB comfortable. `--ram-gb` (default 4) caps the
  weight cache, not the whole process.
- **Disk**: joining a network downloads a shard of the model's GGUF file to
  the local machine (under the node's home directory) and serves it back to
  peers. Space scales with `--hold-fraction`: your fraction of the routed
  experts, plus the resident chunks every node holds (about 10% of the
  model). For the devnet's ~73 GB GLM-4.5-Air at the default hold-fraction
  of 0.2, budget about 25 GB free.
- **Network**: any. A faster link shortens cold-miss expert fetches and the
  initial sync, nothing else. Peers behind NAT work in dial-out-only mode.

## Build from source

Loom targets Zig 0.16.0, pinned in `build.zig.zon`.
[anyzig](https://github.com/marler8997/anyzig) resolves the version
automatically.

```sh
zig build                        # debug binary -> zig-out/bin/loom
zig build -Doptimize=ReleaseSafe # what releases ship: fast, bounds checks on
zig build test                   # unit tests
zig build -Dgpu=vulkan           # GPU backend on Linux/Windows (Metal is automatic on macOS)
zig build -Dtarget=x86_64-linux-musl -Doptimize=ReleaseSafe -p out   # cross-compile
```

Releases are built ReleaseSafe rather than ReleaseFast. Loom is a daemon
that writes peer-supplied weight shards into buffers, and the roughly 11%
speed difference between the two modes is the difference between catching
an out-of-bounds write and executing it. Releases are cut by pushing a tag;
see [`.github/workflows/release.yml`](.github/workflows/release.yml).
Docker images (linux/amd64 and arm64) publish to
`ghcr.io/ch4r10t33r/loom:latest` ([docker guide](docs/DOCKER.md)).

## Commands

| Command | Purpose |
|---|---|
| [`loom node`](docs/COMMANDS.md#loom-node--run-an-inference--weight-sharing-node) | the daemon: load a model, serve inference (native RPC + OpenAI API), share/sync/repair GGUF weight ranges over P2P |
| [`loom gguf`](docs/COMMANDS.md#loom-gguf--gguf-tools-gen--info--shard--run) | GGUF tools: inspect any GGUF, shard by expert, run deepseek2/llama/qwen/glm models locally or from a partial store |
| [`loom light`](docs/COMMANDS.md#loom-light--delegating-light-node) | light node: no weights or engine; delegates RPC/OpenAI to full nodes, metered |
| [`loom run`](docs/COMMANDS.md#loom-run--one-shot-local-inference) | one-shot local inference against a loom checkpoint (no servers) |
| [`loom gen`](docs/COMMANDS.md#loom-gen--generate-a-synthetic-checkpoint) / [`loom info`](docs/COMMANDS.md#loom-info--inspect--verify-a-checkpoint) | create / inspect + verify loom-format checkpoints |
| [`loom bench`](docs/PERFORMANCE.md) / [`loom iobench`](docs/COMMANDS.md#loom-iobench--disk-profiler) | kernel invariants / disk profiler for the engine's random-read pattern |

`loom node --network devnet|testnet|mainnet` selects a
[named network](docs/NETWORKS.md). Each network serves exactly one model
and rejects peers from other networks, the same way a chain id works.
Every command and flag is listed in the [CLI reference](docs/CLI.md);
worked examples with real output are in the
[command tour](docs/COMMANDS.md).

## Documentation

- [Command tour](docs/COMMANDS.md) — every command, worked examples, protocols (RPC, OpenAI, P2P wire)
- [CLI reference](docs/CLI.md) — every flag, defaults, env overrides
- [Install details](docs/INSTALL.md) — manual install, checksum verification, Gatekeeper
- [Multi-node walkthrough](docs/MULTI-NODE-WALKTHROUGH.md) — shard a real MoE model across a swarm on one host
- [Two-machine test](docs/TWO-MACHINE-TEST.md) — a real cross-machine swarm with measured tok/s
- [Named networks](docs/NETWORKS.md) — devnet / testnet / mainnet, models and policies
- [Performance](docs/PERFORMANCE.md) — CPU SIMD numbers, batching, benchmarking rules
- [GPU backends](docs/GPU-BACKENDS.md) — Metal and Vulkan design and status
- [Chat UI & node console](docs/CHAT-UI.md)
- [Docker](docs/DOCKER.md)
- [GLM 5.2 target deployment plan](docs/GLM52-PLAN.md) · [readiness audit](docs/GLM52-READINESS.md)
- [Repository layout & source map](docs/SOURCE-MAP.md)
- [Roadmap](docs/ROADMAP.md) · [p2p spec](spec/SPEC.md) · [whitepaper](whitepaper/WHITEPAPER.md)

## License

[Apache 2.0](LICENSE). Redistributions must retain the copyright and
attribution notices in [NOTICE](NOTICE).
