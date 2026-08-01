# Target deployment: GLM 5.2 on a swarm of 16 GB machines (plan)

The end-state this repo is building toward (see [docs/ROADMAP.md](ROADMAP.md);
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
