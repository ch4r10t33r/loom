# Loom: Running Frontier Mixture-of-Experts Models on Commodity Swarms

**Version 0.10 — 2026-07-20 (living document)**

> This whitepaper is maintained alongside the code. Every design decision is
> reflected here: the affected section is updated and a dated entry is appended
> to the [Decision Log](#appendix-a--decision-log). Where this document, the
> [spec](../spec/SPEC.md), and the [roadmap](../docs/ROADMAP.md) differ in
> detail, the spec governs the p2p layer and the roadmap governs status; this
> document governs intent.

## Abstract

**What is proven.** Loom keeps **compute node-local** and makes **weights the
only thing that moves**: an MoE model is sharded into content-addressed
*expert blocks*, distributed with committee-based redundancy, verified by
Merkle digests at every hop, gossiped, repaired eagerly, and fetched into the
token loop on demand. The claims we have demonstrated, on real weights, are
scoped precisely: the inference engines produce correct output on reference
checkpoints (tinyllamas; DeepSeek-V2-Lite, a 15.7 B-parameter MoE, Q4_K_M);
and a node holding **33 % of that model's shards produced token-identical,
factually correct completions while fetching the missing experts from a
single LAN peer mid-inference**, every block digest-verified before use.
This is a *correctness and integration* result obtained with scalar kernels
(~0.3 tok/s on one CPU core) on localhost — not a throughput or
network-performance result (§6, §10).

**What is targeted.** The design goal is serving frontier MoE models
(GLM 5.2-class: 744 B parameters, 19,200 experts) on swarms of ordinary
16–32 GB machines. That goal is *architecturally supported but not yet
demonstrated*: GLM 5.2 has never been run, kernels lack SIMD, batching is
unimplemented. And the bandwidth arithmetic must be read before the vision:
on ordinary gigabit Ethernet, distribution buys **capacity — the model fits
where it never could — not speed**; competitive throughput additionally
requires 10 GbE+, an aggressively pinned hot set, and batched serving. §10
maintains the exact measured/unmeasured boundary.

## 1. Motivation

Three facts collide:

1. **The best open models are MoE.** Kimi K2 (~1 T total / ~32 B active),
   DeepSeek V3 (671 B / 37 B), GLM 5.2 (744 B / ~40 B). The frontier of
   open-weight capability activates 3–5 % of its parameters per token.
2. **The weights don't fit anywhere cheap.** At int4, GLM 5.2's routed experts
   alone are ~370 GB. A 16 GB laptop cannot hold them; even a 192 GB
   workstation cannot.
3. **The activated set is small and skewed.** Per token, GLM 5.2 reads
   600 expert blocks (~11.4 GB), drawn Zipf-like from the 19,200-block corpus
   — a *sparsely accessed, immutable, content-addressable blob store*.

The conclusion Loom is built on: this is a **distributed storage and caching
problem wearing an inference costume**. The corpus should be stored once
across a swarm, replicated by policy, verified cryptographically, and paged
into compute on demand — the way distributed filesystems treat cold blocks,
not the way tensor-parallel runtimes treat shards.

## 2. Core thesis: experts flow — not activations, not KV

Existing approaches to multi-machine inference split the *computation*:
pipeline-parallel and layer-split systems (Petals, exo) place layers on
different machines and ship **activations** across the network every layer of
every token. These systems work and ship today — the comparison here is
structural, not a claim that they fail. Activation-shipping puts the network
in the serial critical path: total latency is the sum over layers of the
slowest link, and a slow or lost peer stalls every token in flight.

Loom inverts this. Every node runs the **complete forward pass** locally. The
only thing that ever crosses the network is an **expert block**: immutable,
content-addressed, cacheable, prefetchable, and identical for every user of
the model. Three consequences:

- **Failure is soft.** A lost peer costs a re-fetch from a replica, not a
  broken pipeline. A lone node still works — it just streams from its own
  disk.
- **Caching compounds.** Hot experts end up resident on many nodes; the
  network is touched only on the cold tail.
- **KV never moves.** The target models use Multi-head Latent Attention
  (MLA), which compresses the KV cache to ~576 floats per token per layer —
  a few GB even at long context. Distributing KV would put mutable,
  latency-critical, per-session state on the network for negligible capacity
  gain; Loom forbids it by design.

The honest counterweight, measured and documented from day one: on slow
networks (1 GbE), distribution buys **capacity, not speed** — the swarm lets
the model *fit* where it never could, while throughput requires 10 GbE+,
batching, and a pinned hot set. Loom is a serving-throughput architecture
first, an interactive-latency one second.

## 3. System architecture

Loom is a single Zig binary organized into five planes:

| Plane | Folder | Role |
|---|---|---|
| Core | `src/core/` | SHA-256/Merkle, tensor math, int4 quantization, measurement |
| Inference engines | `src/engine/`, `src/gguf/` | node-local forward passes (below) |
| Distribution | `src/p2p/` | shard store, wire protocol, gossip, committees, repair |
| Daemon | `src/node/` | node orchestration, RPC serving, model resolution |
| CLI | `src/main.zig` | `loom node / gguf / run / gen / info / iobench` |

Two inference engines share the kernels:

- **`llama` architecture** — GQA attention, NORM RoPE, SwiGLU; validated
  against reference models (tinyllamas) in F32, Q4_0, Q8_0.
- **`deepseek2` architecture** — the Kimi/DeepSeek/GLM family: MLA attention
  (q/kv-LoRA, compressed-KV latent cache, decoupled rope head), sparse MoE
  routing (sigmoid/softmax gating, noaux_tc selection bias, shared experts),
  YaRN context extension, byte-level BPE tokenizer from GGUF metadata.
  Validated on real DeepSeek-V2-Lite weights (Q4_K_M).

Weights stay in their GGML quantized formats (F32/F16/Q4_0/Q5_0/Q8_0/
Q4_K/Q5_K/Q6_K) in a read-only memory map; every matmul is a fused kernel
over the raw bytes — nothing is dequantized wholesale.

## 4. Expert-aligned sharding

The distribution unit equals the compute unit. A **shard is one expert
block**: the gate/up/down matrices of a single expert in a single layer,
expressed as an extent list over the GGUF file's 3-D expert tensors. A fetched
shard is exactly what a MoE matmul consumes — no assembly, no coding, no
partial reads in the hot path.

Everything that is *not* a routed expert — attention, norms, shared experts,
embeddings, router weights, the GGUF header — forms the **resident bundle**,
chunked into 16 MB shards that **every node holds in full**. This is what
keeps compute node-local: any node can run the dense path and the router
alone; only routed-expert reads may leave the machine.

Every byte of the file belongs to exactly one shard. Each shard carries a
SHA-256 digest; the Merkle root over the digest list is the **model version
id** — the identity nodes gossip, pin their requests to, and (planned)
hardfork on. Integrity is therefore free at every hop: any shard from any
peer is verified against the local manifest before it touches disk or a
matmul.

**GLM 5.2 numbers** (the target deployment):

| Quantity | Value |
|---|---|
| Routed shards | 19,200 (75 MoE layers × 256 experts) |
| Shard size | ~19 MB int4 |
| Routed corpus | ~370 GB, stored once across the swarm |
| Resident bundle | ~10 GB, on every node |
| Logical GGUF | ~380 GB; each node's copy is sparse (~60 GB real bytes at defaults) |
| Manifest / holdings bitmap | ~1 MB / 2.4 KB |

Measured on real DeepSeek-V2-Lite Q4_K_M (10.4 GB): 1,664 expert shards of
4.98–6.02 MB plus 73 resident chunks (0.78 GB), manifest 204 KB, bitmap 218 B.

## 5. The p2p layer: committees, gossip, and the mesh

Full detail in the [spec](../spec/SPEC.md); the shape:

**Bootnode.** In v1 the bootnode is a **trusted placement service** — this
is stated plainly rather than by analogy. It serves the manifest and assigns
each joining node to a **shard committee** with a **least-covered-first**
want-set, so coverage is achieved by construction rather than probability.
Two properties hold and two do not: it is *not* in the token-loop inference
path, and joined nodes survive its death (gossip-derived membership, mesh
routing); but until a committee reaches completeness the bootnode *is* in
the availability path (it covers coverage gaps as origin holder), and a
malicious bootnode could assign adversarial or incomplete want-sets — v1
assumes it is operated by the swarm operator. Verifiable assignment (signed
manifests and assignments, multiple registries, coverage challenges) is
future work, recorded in the spec's threat model.

**Committees.** A committee is a group of nodes that *collectively holds the
complete shard set* — a self-sufficient serving cell. Invariants: every shard
has ≥ 1 holder per committee (completeness); the bootnode fills committees
toward a redundancy target R (default 2); when all committees are saturated,
the next joiner opens a new one. Members exchange **heartbeats** (5 s):
liveness plus manifest version, a monotonic holdings sequence number + bitmap
digest, and a load hint for client-side spreading. Committee membership is
**gossip-derived**, not static: announces carry committee ids, so earlier
members discover later joiners automatically, and the view survives bootnode
death.

**Global gossip mesh.** Independently of committees, every node announces its
holdings (full bitmap + version + committee id) on the gossip network every
3 s; receivers merge announces into a peer table — the **mesh**. Discovery is
transitive. ENR records (planned) will carry only the compact summary
(version + holdings seq + digest, fitting the 300-byte ENR limit); the gossip
topic carries the full bitmap.

**Query path.** A node needing a shard asks (1) committee members first,
round-robin across replicas; (2) the mesh as fallback; every response is
digest-verified. A request fails only if *no reachable peer anywhere* holds
the shard — and the **eager repair loop** (2 s) works continuously to make
that state transient, re-fetching any unsatisfied want-set from every known
peer. Churn policy is deliberately maximal: dead peers are retried, never
forgotten; a returning holder is drained within one tick.

**Wire messages v1.** All structured messages are binary frames (8-byte
envelope, adaptive snappy: compressed only when smaller — holdings bitmaps
compress > 4×, quantized payloads ship raw): Heartbeat, Announce,
AnnounceBatch, ExpertRequest/ExpertResponse with typed statuses
(`ok / not_held / version_mismatch / busy / bad_request`). Requests pin the
manifest version — a peer on a different model version is refused wholesale,
which is the hardfork guard operating at the message level. Responses carry
no digest by design: the local manifest is the only trust root.

## 6. Distributed inference in the token loop — a correctness and integration result

The piece that makes the rest matter: a node whose local store lacks an
expert **fetches it mid-token**. Inside each MoE layer, after the router
selects its experts, missing shards are prefetched **in parallel** (per-layer
miss latency is max(fetch), not sum), pulled committee-first with round-robin
spreading, digest-verified, then **persisted into the local sparse store** —
so the node's holdings grow with use and are advertised on the next gossip
round. Fetch-on-demand is thereby also **organic heat replication**: hot
experts gain holders because they are hot.

Measured result (single peer, localhost, scalar kernels): a node holding
**573 of 1,737 shards (33 %)** of DeepSeek-V2-Lite completed
*"The capital of France is" → "Paris."* — streaming 641 experts (3.5 GB,
29 ms average) with zero failures, token-identical to a full-copy run, and
finishing at 1,214 shards held. The architecture's core promise — *compute
local, experts flow, correctness preserved* — holds on real weights.

**What this result does and does not validate.** It validates digest-verified
mid-token fetch, sparse-store persistence, shard/engine alignment, and
end-to-end correctness under a partial store. It does **not** validate:
miss latency on 1 GbE/10 GbE under contention; concurrent fetches from many
requesters (hot-expert hotspotting); committee failover under real RTT and
loss; or whether Zipf skew plus pinning keeps token-loop misses rare enough
for acceptable throughput. Those are network-bound measurements that have
not been made; the analytical model in §9 states the expectations they must
be tested against.

## 7. Trust model

**v1 (current): trusted-swarm with cryptographic *intra-manifest* integrity.**
Content addressing + the Merkle-rooted manifest make corruption and poisoning
detectable for free *relative to a manifest you already trust*: a bad shard
fails its digest before use, from disk or from any peer, and version pinning
prevents cross-model mixing. Two boundaries must be stated honestly:

- **Which manifest is trusted is not free.** Content addressing secures
  integrity *within* a manifest, not the choice *of* manifest. A node
  bootstraps its Merkle root from the bootnode's `MANIFESTFILE` (or operator
  config); a hostile bootstrap could pin an alternate,
  internally-consistent-but-poisoned root. v1's answer is trust-on-first-use
  plus operator configuration; signed roots / out-of-band pinning are future
  work. The spec's *Manifest trust bootstrap* section is normative here.
- **Holdings claims are unverified.** Peers can lie about which shards they
  hold (the heartbeat carries a *digest* of the bitmap, not a proof of
  possession). Completeness, committee-first routing, and load spreading all
  assume honest bitmaps. A lying holder is caught only on `GETR` (wrong or
  absent bytes → fall through to the next peer); there is no reputation,
  challenge, or probe-on-suspicion yet. **Committee completeness is a
  construction-time property under honest participation, not a
  cryptographically enforced invariant.**

Beyond these, v1 does not defend against availability attacks (refuse
service, mesh poisoning with fake holders, RPC flooding) — the swarm is
assumed cooperative and operator-run.

**v2 (designed, not built): untrusted peers.** Planned per the
[design doc](../CLAUDE.md): RLNC-based WAN propagation gated behind
homomorphic-hash pollution defenses, redundant recompute with M-of-N voting
on sampled tokens, TEE attestation, and a zkML interface stub. Coding
(EC/RLNC) is confined to propagation and durability planes — principle 7
forbids it in the token loop, where only direct addressed fetch of original
blocks is permitted.

## 8. Node classes and the service economy

Loom distinguishes two node classes:

**Full nodes** hold the resident bundle plus their assigned expert shards,
join committees, serve shards to peers, and run inference. They are the
supply side of the network.

**Light nodes** are the demand side: anyone wanting local inference on a
low-memory device (a laptop, a phone-class box) runs one. A light node holds
*no weights, no store, and no engine* — its footprint is a few megabytes. It
exposes the **same local RPC** a full node does, so applications are
oblivious to the difference, and **delegates** every request to a full node
with round-robin failover across its configured backends. It stamps a client
identity onto each request.

**Compensation.** Full nodes must be compensated by light nodes for serviced
requests. What v1 ships is a **trusted-operator accounting demo** — the
ledger mechanics and enforcement path, with *none* of the adversarial
hardening a real service economy needs. Specifically, v1 is trivially
gameable and says so: client ids are self-asserted strings (rotate the id,
reset the free quota); ledgers are per-provider (N full nodes ⇒ N× free
quota under round-robin); the `credit` op accepts an unverified proof
(anyone who can reach the RPC can mint credits); and there are no signed
receipts, so neither side can prove the other's ledger wrong. Cryptographic
client identity (at minimum a shared-secret HMAC, properly a keypair),
proof-of-payment verification, and signed usage receipts are prerequisites
before "compensation" can be described as implemented. The overdraw window
is bounded: at most one request's `prompt + max_tokens`. What exists today:

- Every full node runs a per-client **metering ledger**: allowance =
  free quota (`--free-quota`, default 1 M tokens) + purchased credits −
  usage, where usage is prompt-plus-generated tokens. Every metered response
  carries `cost` and `balance`; an exhausted client receives
  `payment_required` before any compute is spent. Each full node meters its
  own service — ledgers are per-provider, not global.
- The `credit` RPC op (`{"method":"credit","client":…,"amount":…,"proof":…}`)
  is the settlement integration point: v1 accepts the proof unverified
  (trusted swarm); a payment rail (invoice/receipt verification, on-chain or
  Lightning-style) replaces exactly that trust check without touching the
  ledger, the gate, or the wire format. A `tab` op exposes any client's
  used/balance.
- Light nodes independently tally their own spend from response `cost`
  fields — both sides of the ledger exist from day one, which is the
  precondition for any dispute-free settlement scheme later.

Measured end-to-end: a light node delegated inference transparently
(identical protocol, `cost`/`balance` appended), spread load round-robin
across two full nodes with independent ledgers, hit a deterministic
`payment_required` when its only backend's allowance was exhausted, and
resumed service after a `credit` top-up.

**Light nodes are thin clients, not swarm participants.** They contribute no
storage, no serving, and no redundancy; every token they consume is load on
a full node. Capacity planning must size the full-node fleet for aggregate
light-node QPS — light nodes amplify demand and add nothing to supply.

## 9. Deployment model and sizing

Held shards are a **disk** budget; RAM carries only the compute working set.

| | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB (≈10 GB mmap'd dense + ~0.7 GB MLA KV + 2–3 GB expert cache) | 24–32 GB |
| Disk | 75 GB NVMe (resident + ~50 GB shards) | 150 GB NVMe |
| Experts held | ~1,300 (25 GB) | ~2,600 (50 GB) — 13.7 % of GLM 5.2 |
| Network | 1 GbE (capacity win) | 10 GbE+ (latency win too) |

Swarm sizing against R over the 370 GB corpus (R = 3 target, R = 2 floor —
see the spec): at 50 GB/node, **completeness (R = 1) needs ≥ 8 nodes, R = 2
needs ≥ 15, R = 3 needs ≥ 22**; 8 nodes at 100 GB reach R = 2. The
bootnode's least-covered-first assignment makes these guarantees
constructive.

### Performance model (analytical — not yet measured)

Per-token time ≈ `t_compute + misses × t_fetch`, where `misses` is the number
of routed experts (of 600 for GLM 5.2) not in the local store/page cache, and
`t_fetch ≈ RTT + shard_size / bandwidth` per miss (fetches within a layer
parallelize to max, not sum, across that layer's ≤ 8 misses). Indicative
cold-miss cost for a 19 MB shard: ~160 ms on 1 GbE, ~16 ms on 10 GbE, before
protocol overhead. The implications are stated, not hidden: with a cold
working set on 1 GbE, a single stream is seconds per token — unusable
interactively; serving becomes viable exactly insofar as pinning + Zipf skew
+ organic replication drive the steady-state miss rate toward zero and
batching amortizes what remains. **No network-tier measurements exist yet**
(the table of GbE-tier × miss-rate × tok/s is future work); until they do,
every throughput expectation should be derived from this model, not from
the localhost result in §6.

Positioning, stated honestly: *run trillion-parameter open MoE models on the
machines you already have* — a pooling story, not a magic-laptop story. One
node alone gets disk-streaming; the value curve starts at a handful of boxes
on a LAN. The differentiators against layer-split systems are structural
(experts-not-activations, soft failure, verified weights, self-healing
membership); the differentiator against single-box streaming is capacity.

## 10. Validation status

**Proven, on real weights or live multi-node runs:**
llama + deepseek2 engines produce correct output on reference checkpoints
(tinyllamas; DeepSeek-V2-Lite Q4_K_M — including the NORM-rope finding, K-
quants, BPE, YaRN); expert-aligned sharding round-trips a real 10.4 GB model;
bootstrap, gossip discovery, committee formation/saturation, heartbeat death
detection, eager repair, mesh fallback, and 33 %-store distributed inference
all demonstrated end-to-end.

**Honest gaps:** GLM 5.2 itself has not been run (no public weights
converted; the deepseek2 path is the architectural proxy); kernels are
scalar (no SIMD) — ~0.3 tok/s on a 15.7 B model on one core; single-sequence
only (no batching); the loom-format v0 engine lacks token-exact oracle
validation; distributed inference is served via `loom gguf run`, not yet
through the node's RPC; ENR and hardfork coordination are designed but
unimplemented.

## 11. Roadmap

Near-term: heartbeat-triggered re-replication (a dead member's shards
re-covered proactively by its committee); hardfork coordination (majority of
nodes adopting a new manifest version, with the version guard already
enforcing isolation); serving the distributed engine through node RPC; SIMD
kernels; continuous batching. Mid-term: ENR integration (`blockblaz/enr`) and
gossipsub/discv5 transports — the containers and table semantics are already
transport-agnostic. Long-term: the v2 trust layer (§7). Authoritative
tracking: [docs/ROADMAP.md](../docs/ROADMAP.md).

## Appendix A — Decision Log

Every design decision is recorded here with its date and rationale. Newest
last. *This log is append-only; superseded decisions are struck through, not
deleted.*

| Date | Decision | Rationale |
|---|---|---|
| 2026-07-11 | Distribution unit = the **expert**, not the layer; compute node-local; never distribute KV | MoE sparsity + MLA's tiny KV; avoids activations in the serial network path (CLAUDE.md core thesis) |
| 2026-07-11 | Content-addressed shards + Merkle-rooted manifest | Free integrity/poisoning check at every hop; manifest root doubles as model version id |
| 2026-07-11 | No coding (EC/RLNC) in the token loop, ever | Decode-before-use adds latency exactly where it cannot be hidden; coding confined to propagation/durability |
| 2026-07-12 | Target toolchain: Zig 0.16.0 (dev), matching the user's zeam project | Shared toolchain + reusable deps (enr, ssz, snappy) outweigh 0.16 API churn |
| 2026-07-12 | v1 direction: GGUF as interchange format; ENR + gossip + req-resp (Ethereum-style) p2p; majority hardforks; boot-time peer sync | User requirements of record; supersedes Hyperswarm/Hyperbee sketch |
| 2026-07-12 | Weight advertising: **both** ENR (compact summary — 300 B limit → bitmap digest + seq) and a global gossip topic (full bitmap) | ENR for discovery-time selection; gossip for live freshness |
| 2026-07-13 | Churn repair: **maximally eager** — always-on loop retries all known peers; dead peers retried, never forgotten | User decision ("as eagerly as possible"); lazy repair-on-miss rejected |
| 2026-07-13 | Gossip: epidemic announce-and-exchange over the peer table; transport swappable for gossipsub later | Table semantics are transport-independent; LAN-scale first |
| 2026-07-19 | deepseek2 rope is **NORM (adjacent-pair)**, not NEOX | Empirical: DeepSeek's (d/2,2)-transpose before rotate_half nets to adjacent-pair on stored layout; NEOX degenerates |
| 2026-07-19 | Response payloads carry **no digest**; the requester verifies against its own manifest | The manifest is the only trust root; a peer-supplied digest adds nothing |
| 2026-07-19 | Shard = one expert block (extent list over 3-D exps tensors) + mandatory 16 MB resident chunks; ~19 MB for GLM 5.2 → 19,200 shards | Shard unit must equal the matmul's consumption unit (principle 7); metadata stays trivial |
| 2026-07-19 | Bootnode assigns committees **least-covered-first** toward R = 3 target / R = 2 floor; saturated committees spawn new ones | Coverage by construction instead of probabilistic overlap; Ethereum-bootnode discipline (out of inference path) |
| 2026-07-19 | Fetched shards are **persisted** (not LRU-evicted): fetch-on-demand = organic heat replication | Disk is the cheap resource; hot experts should gain holders by being used |
| 2026-07-20 | Wire messages v1: binary frames, **adaptive snappy** (compress only when smaller); heartbeat carries seq+digest+load but **not** the bitmap; the bitmap rides gossip announces | Quantized payloads are incompressible; heartbeats stay ~90 B; staleness detected cheaply |
| 2026-07-20 | Committee membership is **gossip-derived** (announces carry committee id), heartbeat set = seed ∪ table view | Fixes static-at-join membership; no bootnode push protocol; survives bootnode death |
| 2026-07-20 | Two node classes: **light nodes** (no weights/engine; same local RPC; delegate to full nodes with failover) and **full nodes** (shards + inference) | Local inference for low-memory devices without weakening the supply side |
| 2026-07-20 | Compensation: per-client **metering ledger on each full node** (free quota + credits − token usage), `payment_required` enforcement, `credit` op as the unverified-in-v1 settlement seam | Accounting must precede payment rails; ledgers are per-provider; both sides tally independently |
| 2026-07-20 | Docs honesty pass (adversarial review #4): abstract split into proven/targeted with capacity-first caveat up front; §6 relabeled correctness-not-throughput; v1 metering re-scoped as a trusted-operator demo with the id-rotation / multi-provider / unverified-credit games named; bootnode documented as a v1 trusted placement service (not an Ethereum analogy); trust §7 split into intra-manifest integrity vs manifest-choice and holdings-honesty | Documents must be adversarially honest before external readers; scope claims to evidence |
| 2026-07-20 | Redundancy is **R = 3 target / R = 2 floor**, default `--r-target 2`; completeness = R ≥ 1 | Resolve the R inconsistency across docs; completeness and redundancy are distinct thresholds |
| 2026-07-20 | Placement is **least-covered-first committee assignment**; random `--hold-fraction` is the legacy/no-bootnode fallback | One live placement policy; random ranges deprecated to fallback status |
| 2026-07-20 | **Binary frames are normative**; the line protocol is legacy/debug (still serves `JOIN`/`GETR`/`GOSSIP`/`TABLE` today during migration) | Prevent forked implementations; state migration status |

## Appendix B — Provenance

Loom's design draws on: colibri (expert streaming technique), llama.cpp/GGML
(GGUF format, quantization layouts, reference kernels semantics), DeepSeek
V2/V3 papers (MLA, noaux_tc routing), Ethereum networking (bootnodes, ENR,
gossip, committees), and content-addressed storage systems (Merkle
verification, manifest-pinned versions). Implementation is original Zig with
one dependency (blockblaz/zig-snappy).
