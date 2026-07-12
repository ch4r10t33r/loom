# CLAUDE.md — Loom (working name): distributed expert cache for large MoE inference

> Handoff doc for Claude Code. Working name **Loom**. Target model: **GLM 5.2** (Z.ai, MIT license); architecture is MoE-agnostic. This revision replaces the earlier pipeline-parallel design: the distribution unit is now the **expert**, not the layer.

## Core thesis

GLM 5.2 is ~744B total params but only ~40B activate per token, and only ~11.4 GB of weights (the routed experts) change from token to token. The dense part (attention, shared experts, embeddings ≈ 17B params ≈ 10 GB at int4) is small and stays resident on every compute node. The ~19k routed experts (~370 GB at int4) are an **immutable, content-addressed, sparsely-accessed blob store**.

Therefore: **compute stays node-local** (each box runs its own full forward pass), and the only thing crossing the network is expert blocks — cacheable, prefetchable, replicable. No activations in a serial cross-node critical path. This is strictly better than pipeline-parallel for MoE, and it degrades gracefully to single-node streaming (colibri) when peers are unavailable.

KV cache is a non-issue for this model: GLM 5.2 uses MLA, so KV is ~576 floats/token (57× smaller than dense MHA). Do not build KV distribution. The thing to distribute is experts.

## Model facts (design inputs)

- 744B total / ~40B active; 256 experts, **8 routed + 1 shared** per MoE layer.
- 78 blocks: first 3 dense FFN, remaining **75 MoE layers**; MTP head at layer 78.
- Per token: 75 layers × 8 routed experts = **600 expert reads ≈ 11.4 GB** at int4 (~19 MB/expert).
- Dense resident set ≈ 10 GB int4. Total routed store ≈ 370 GB int4.
- Attention: MLA (q/kv-LoRA, partial RoPE) + DeepSeek-style sparse indexer (`glm_moe_dsa`, needs `transformers>=5.3.0`). Compressed KV ≈ 576 floats/token.
- Router: DeepSeek-V3-style sigmoid (noaux_tc), shared expert always active.
- Native MTP: multi-token-prediction head drafts tokens, verified in one batched forward.

## The one counterintuitive constraint — read before designing the fetch path

Peer-RAM is **not** reliably faster than local disk on commodity gear. Bandwidth math for the 11.4 GB/token working set:

| source | effective bw | per-token cost |
|---|---|---|
| gigabit ethernet | ~0.11 GB/s | ~90 s/token — useless |
| 25 GbE | ~3.1 GB/s | ~3.7 s/token |
| 100 GbE / RDMA | ~12 GB/s | ~0.9 s/token |
| local PCIe5 NVMe (19 MB random reads) | ~3–8 GB/s | comparable to 25 GbE |
| local RAM (pinned/cached) | — | free |

**Consequences, non-negotiable:**
- The tier order (`local RAM → peer → local disk` vs `local RAM → local disk → peer`) is a **per-deployment function of measured network bw vs. local disk bw**. Probe both at startup; do not hardcode "prefer network."
- The network tier wins decisively only on a fast fabric (100 GbE / RDMA / IB). On 25 GbE it ties local NVMe; on gigabit it's pointless.
- What distributing experts actually buys is **capacity + dedup, not raw speed**: the cluster's *collective* RAM holds a working set no single commodity box can (8×32 GB ≈ 256 GB, most of the 370 GB set stays hot *somewhere*), and the 370 GB is stored once (sharded + hot-replicated) instead of per-node. If the fabric is slow you keep the capacity benefit but lose the speed benefit, and single-node colibri may beat it per-user. Deployments must be told this.
- This is a **throughput-first / serving** architecture, not a snappy-single-chat one (see latency-hiding below).

## Non-negotiable design principles

1. **Experts flow, not activations, not KV.** Compute is node-local. Cross-node traffic is immutable expert blocks only.
2. **Measured tier order.** Fetch each block from the lowest *effective-latency* source, decided from startup probes, not assumptions.
3. **≥2 sources per expert.** Every expert reachable via replica or EC; one node leaving must never strand a layer.
4. **Heat-proportional replication.** Hottest experts on multiple nodes (a single home for a hot expert becomes a load hotspot under concurrency). Cold experts to an EC'd tier.
5. **Content-addressed + Merkle-rooted.** Expert blocks keyed by hash; checkpoint manifest is a Merkle root → free integrity check against poisoned experts.
6. **Throughput hides latency; pinning lowers misses.** Stay in the batched regime; pin the measured hot set.
7. **Coding never touches the per-token fetch path.** A token needs *that exact 19 MB block* materialized now to matmul. EC/RLNC require collecting *k* packets + decoding before any block is usable — same inner-loop mistake as remote KV. Coding lives strictly in the propagation/seeding/durability plane; the hot path is direct addressed fetch of the original block.

## Latency-hiding (why it's serving-first)

The router only reveals a layer's experts once you reach that layer, so a peer fetch for layer L sits in the critical path unless predicted — and you can't cleanly prefetch L+1 (its routing depends on L's output). You therefore cannot fully hide per-layer miss latency for a single low-latency stream. Levers, in order:
- **Pin the hot set.** Expert usage is ~Zipfian; a modest pinned set covers most activations. Measure first: colibri's `STATS=` records usage, `PIN_GB=` pins the hottest.
- **Batch-union throughput.** With continuous batching, a MoE layer needs the union of experts across the batch; prefetch next microbatch's experts while computing the current. Latency dissolves into throughput.
- **Speculative prefetch (research lever).** Cheap next-layer routing predictor or MTP drafts to prefetch likely experts; pay only on misprediction. Prototype, don't bet on it.

Failure modes to engineer against: **hot-expert hotspotting** (→ heat-proportional replication + client-side spreading) and **single-stream prefetch stall** (→ batched regime + aggressive pinning).

## Coding: EC & RLNC — for propagation and durability, not the hot path

Two distinct axes, often conflated. Neither is allowed in the per-token fetch path (principle 7). Both operate on the immutable ~370 GB expert corpus.

**Axis 1 — durability (storage).** EC (Reed–Solomon, any *k* of *n*) keeps the cold expert tier alive under node loss at lower overhead than full replication. Hot experts stay *replicated* (fast direct reads); cold experts are EC'd. This is the storage tier from v1.

**Axis 2 — propagation (dissemination).** Coding to move bytes across the swarm efficiently in one-to-many / many-to-many bulk transfers: onboarding a fresh node (pull tens of GB), rolling out a re-quantized model version, and burst heat-replication (a cold expert goes hot, fan out from 2 holders to 20). These are exactly the transfers where BitTorrent's rarest-piece / endgame stalls appear; rateless coding erases them — any holder can mint a fresh useful packet, so nobody waits on the last specific piece.

**EC vs RLNC for propagation:**
- **Fixed-rate EC (RS):** any *k* of *n* reconstruct; great for multi-source download + storage, but *end-to-end* — an intermediate relay can't recombine without decoding first. Fine for source→peer fan-out.
- **RLNC:** rateless *and* recodable — intermediate nodes recode received packets into new coded packets *without decoding*. The network-coding multicast gain; compounds over multi-hop meshes; zero who-has-what state, so it shrugs off churn. Plain EC cannot do this.
- **Systematic variants (either):** send originals first, coded repair packets after. **Default pick** — originals stay directly readable, preserving random-access single-expert fetch, while the coded tail gives loss repair. Resolves the "coded for propagation vs. readable for the hot path" tension.

**Where each belongs in Loom:**
- **LAN (v0/v1): RLNC is over-engineering.** Recoding's gain scales with mesh depth; a LAN is fat, flat, near-single-hop. Paying Galois-field encode/decode + coefficient overhead for a coupon-collector problem that barely exists is a bad trade. LAN path stays: direct addressed fetch (hot) + **systematic EC or plain chunked multi-source** (bulk/onboarding), with the DHT naming which fragments exist.
- **WAN / high-churn / DePIN (v2): RLNC's real payoff.** Deep multi-hop mesh, constant join/drop, no stable topology — rateless + recoding dominates for propagating the corpus and version updates across a public swarm.

**RLNC's tax on untrusted peers — the pollution attack (must price in before v2).** A malicious node emits a well-formed but wrong coded packet; because decoding linearly mixes a generation, one bad packet corrupts the *entire* generation, and post-mix you can't identify the culprit. Per-block Merkle hashing does **not** catch this — corruption only surfaces after decode. Defense needs pollution-resistant integrity: **homomorphic hashing** (Krohn–Freedman–Mazières) or homomorphic signatures/MACs that verify each *coded* packet against a commitment *before* mixing. Real crypto, real per-packet cost — belongs in the v2 verification design, not bolted onto propagation. So RLNC on a public swarm is coupled to v2: you get its churn benefits only once you've paid for pollution resistance.

**Net:** systematic EC for LAN bulk + cold-storage durability; RLNC reserved for WAN/untrusted propagation, gated behind homomorphic-hash pollution defense. *(This is the default chosen here — flip it if you want RLNC in v1, but then the LAN path pays coding cost for little gain.)*

## Reuse vs. build

**Reuse (technique or code):**

| Concern | Use | Notes |
|---|---|---|
| Expert-streaming technique, MLA/MoE int4 kernels, batch-union MoE, MTP decode | **colibri** (`JustVugg/colibri`) | Reference/fork. PoC caveat: 0-star, 4-commit, one-person; CPU/AVX2 only; forward validated token-exact vs. a *tiny-random* oracle; numbers above its dev box are estimates. Reuse the design + kernels, re-validate on real weights. |
| Usage measurement + pinning | **colibri `STATS` / `PIN`** | Get real per-workload hit-rates before sizing anything. |
| Disk microbenchmark | **colibri `iobench.c`** | Measures the parallel 19 MB random reads the engine actually issues. |
| FP8→int4 conversion (disk-safe, resumable) | **colibri `convert_fp8_to_int4.py`** | Streams one shard at a time; the 756 GB FP8 checkpoint never fully lands. |
| GPU compute path (if not CPU-only) | **SGLang** (primary) / **vLLM** | colibri is CPU-only; for GPU nodes wrap an engine for the dense part + resident hot experts. Verify `glm_moe_dsa` support. |
| Quantized checkpoints | existing **FP8 / NVFP4** builds | Don't build quant. |
| Directory + content-addressed store | **Hyperbee / HyperDHT / Hyperswarm** | `expert_id → holder_set`; Merkle-rooted manifest. |
| Expert-block transport on fast fabric | **Mooncake Transfer Engine / NIXL / RDMA** | For the 100 GbE/IB deployments. |
| Cold-tier durability + integrity | **Reed-Solomon EC lib + Merkle/SSZ** | EC the immutable cold experts; SSZ/EIP-6466 Merkleization reusable for the manifest. |
| LAN bulk propagation / onboarding | **Systematic RS-EC** or plain chunked multi-source | Originals readable + coded repair tail; DHT names fragments. No RLNC on LAN. |
| WAN/untrusted propagation (v2) | **RLNC lib** (rateless + recoding) | Reserved for deep-mesh/churny swarm. Requires the pollution defense below. |
| Pollution-resistant coded integrity (v2) | **Homomorphic hashing (KFM) / homomorphic MAC** | Verify each coded packet pre-mix; plain Merkle can't. Real per-packet crypto cost. |

**Build from scratch (the product core):**

1. **Distributed expert directory** — `expert_id → holder_set` with live heat tracking, over Hyperbee/HyperDHT.
2. **Heat-aware placement + replication manager** — rendezvous/consistent hashing for homes; replicate hot experts by heat; migrate cold experts to the EC tier.
3. **Measured tier-order fetch policy** — startup probe of local-disk vs. per-peer bandwidth; per-block pick the lowest effective-latency source; parallelize the 8 fetches/layer; local RAM/LRU/page-cache/NVMe as colibri does.
4. **Prefetcher** — pin from `STATS`; batch-union prefetch for next microbatch; optional speculative predictor.
5. **Fault tolerance** — enforce ≥2 sources; reroute on node drop without stalling in-flight tokens.
6. **Propagation layer** — systematic-EC bulk transfer for LAN onboarding/version rollout/burst heat-replication (v1); RLNC + homomorphic-hash pollution defense for the WAN/untrusted tier (v2). Never in the token loop.
7. **Verification layer (v2)** — see phase v2.
8. **Control plane** — observability, admission control, metering. The commercial surface.

## Phases

### v0 — Single-node expert streaming (fork colibri's technique)
**Goal:** GLM 5.2 answering on one commodity box; experts streamed from local disk with pinned hot-set + LRU + page cache.
**Deliverables:** working forward (reuse/fork colibri or re-implement its kernels on the GPU path); `STATS`/`PIN` hit-rate measurement; `iobench` disk profile; correctness validated token-exact vs. a `transformers` oracle on real (not tiny-random) weights for a fixed seed/prompt.
**Acceptance:** correct output; per-turn tok/s, expert hit-rate, RSS logged; runs within a declared RAM budget without OOM.
**Reuse:** colibri end-to-end. **Build:** GPU compute path if targeting GPU; real-weight validation harness.

### v1 — Distributed heat-aware expert cache
**Goal:** pool cluster RAM into a content-addressed expert cache; nodes fetch routed experts from local RAM → best source (measured) → local disk; store the 370 GB once.
**Deliverables:** expert directory + heat tracking; heat-proportional replication; EC cold tier with Merkle manifest; **systematic-EC (or chunked multi-source) bulk propagation** for node onboarding, version rollout, and burst heat-replication; measured tier-order fetch with parallel per-layer fetches; batch-union prefetch; ≥2-source fault tolerance with reroute.
**Acceptance:** on a workload with a large shared expert set, aggregate cold-miss-to-disk rate drops vs. v0 single-node; measured tok/s under continuous batching; correctness identical to v0; a killed node strands no layer; hotspot experts show balanced load across replicas; a fresh node reaches a hot-set-ready state in bounded time via bulk propagation.
**Reuse:** Hyperbee/HyperDHT, Mooncake/NIXL/RDMA, RS-EC (durability + systematic bulk), SSZ Merkle. **Build:** directory, placement/replication, propagation, fetch policy, prefetcher, fault tolerance.

### v2 — Untrusted-peer verification
**Goal:** let peers outside the trust boundary serve experts/compute without silently corrupting output.
**Deliverables:** weight/expert integrity via Merkle root (blocks a poisoned expert on read — mostly free from v1); **RLNC-based WAN propagation** (rateless + recoding for deep-mesh/churny swarm) **gated behind homomorphic-hash pollution defense** (verify each coded packet pre-mix); pluggable compute verification — redundant recompute + M-of-N voting on sampled tokens; TEE attestation of peer binaries; zkML interface stub. Reputation/slashing hook if a compute market is layered on later. Cross-reference Synod verification.
**Acceptance:** injected faulty expert/peer detected within a bounded window; injected polluted coded packet rejected pre-mix (generation not corrupted); overhead of RLNC + homomorphic hashing quantified; trust policy configurable per deployment.

## Tech stack summary
- **Engine:** colibri technique (CPU) and/or SGLang/vLLM (GPU); `transformers>=5.3.0` for `glm_moe_dsa`.
- **Weights:** FP8/NVFP4 checkpoints → int4 via colibri converter.
- **p2p / directory:** Hyperswarm + HyperDHT + Hyperbee.
- **Transport:** Mooncake Transfer Engine / NIXL / RDMA on fast fabrics.
- **Storage & coding:** RS-EC (cold durability + systematic LAN bulk), heat-based replication (hot), Merkle/SSZ manifest; RLNC + homomorphic hashing (WAN/untrusted propagation, v2). No coding in the token loop.
- **Daemon/router:** Rust or Zig (aligns with zig-libp2p / zquic); Python only where wrapping an engine.

## Open questions / risks
- **Slow fabric kills the network tier.** On ≤25 GbE the win is capacity/dedup, not speed. Size expectations to measured bandwidth.
- **colibri is a PoC.** Re-validate on real weights; treat as reference, not production.
- **Single-stream latency** can't be fully hidden — this is serving-first. If interactive single-user is a hard requirement, that's local-pinned colibri, not the distributed path.
- **Hotspotting** under concurrency needs heat-proportional replication from day one, not as an afterthought.
- **Pipeline-parallel is demoted** to a fallback (dense sub-model, or non-MoE targets), not the primary axis.
- **RLNC is coupled to v2, not free.** Its churn benefits on a public swarm require homomorphic-hash pollution defense (real per-packet crypto cost). Default keeps RLNC out of v0/v1; LAN uses systematic EC / chunked multi-source. Revisit only if a WAN/DePIN deployment is actually on the roadmap.
- **No coding in the token loop, ever.** EC/RLNC decode-before-use makes them fatal in the per-token fetch path; keep them in propagation/durability only.
