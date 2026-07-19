# Loom roadmap

## v0 — single-node expert streaming ✅ (this repo, working)

- int4 expert-streaming engine (pin → LRU → pread), MLA attention, sigmoid top-k
  MoE routing, content-addressed + Merkle-rooted checkpoint store.
- `loom node`: RPC inference server (JSON over TCP) + minimal P2P directory
  (`HELLO`/`COUNT`/`HAS <id>`/`PING`), model resolution from local dir /
  synthetic `tiny` / Hugging Face download.
- Remaining v0 gap: token-exact validation against a `transformers` oracle on
  real GLM weights (needs real converted weights + BPE tokenizer).

## v1 — distributed weight sharing (requirements of record)

These supersede the original CLAUDE.md v1 sketch where they conflict.
The p2p layer's committee spec lives in [SPEC.md](SPEC.md) — **✅ first cut
implemented** (`bootnode.zig` registry, JOIN protocol, assigned want-sets,
committee-first sync, 5 s heartbeats with death detection, committee-then-mesh
fetch order; **wire messages v1**: binary frames with adaptive snappy —
heartbeat carries committee id, manifest version, holdings seq+digest, and a
load hint; expert request/response carries version-pinned shard queries with
typed statuses; `wire.zig` + blockblaz/zig-snappy). Verified live: B+C formed a complete committee with
complementary assignments; D saturated it to R=2; E opened committee 1;
C detected B's death via heartbeat; a fetch with a dead committee peer fell
through to the mesh and completed inference.

1. **Local-first model resolution.** If a model is already present in local data
   storage, load it from there; only download from remote otherwise.
   (Implemented in `hf.zig::resolve`; a standing contract as features grow.)
2. **GGUF weight distribution.** ✅ *first cut implemented.* A loaded **GGUF**
   file's weights are distributed across multiple nodes. Ranges are chosen
   **randomly** (`--hold-fraction F`, seeded), and the same range may be held by
   several nodes for **redundancy** (independent random subsets overlap).
   Implemented: `gguf.zig` (v2/v3 parser + fixture writer), `weights.zig` (range
   manifest, SHA-256/range, Merkle-root version id, holdings bitmap), P2P ops
   `MANIFEST`/`DIGEST`/`DIGESTS`/`HOLDINGS`/`GETR`.
3. **Boot-time peer sync.** ✅ *first cut implemented* (`sync.zig`,
   `loom node --bootstrap host:port`). A new node requests weight ranges from a
   peer via the request-response protocol; every range is digest-verified before
   touching disk and the digest set is verified against the advertised Merkle
   root. Syncing from a *partial* holder takes what's available and reports the
   shortfall — the seam where multi-peer sync + churn repair (#6) plug in.
4. **Hardfork upgrades.** A majority of nodes agreeing on a new version of the
   GGUF file triggers a hardfork — a coordinated model-version upgrade across
   the swarm.
5. **ENR weight advertising** *(decided)*. Each peer reports the weights it holds
   as part of its **ENR + metadata** — ENR is the discoverability mechanism, so
   nodes can deliberately peer with nodes holding *different* ranges
   (coverage-seeking peering). Constraint: ENR records are limited to **300
   bytes**, so the ENR entry must be a compact summary (e.g. a range bitmap /
   holdings-manifest hash + sequence number), not the full holdings list.
   *Progress:* the holdings bitmap + hex encoding used by the `HOLDINGS` P2P op
   is exactly this summary (1 bit/range); ENR integration itself still todo.
6. **Churn repair** *(decided: maximally eager; ✅ first cut implemented)*. When a
   peer disconnects — or a node's wanted range set is unsatisfied for any
   reason — the node seeks replacement holders **as eagerly as possible**: an
   always-on repair loop (2 s interval, `node.zig`) retries all known peers
   rather than waiting for a miss. Pairs with over-provisioning (random
   overlapping holdings) so single disconnects rarely leave a range with no
   holder. Multi-peer bootstrap (`--bootstrap` + `--peers`) fills shortfalls
   across peers; `fetchFromPeer` refuses peers on a different manifest version
   (the hardfork guard, #4). Verified live: a node that booted 5/9 against a
   dead peer recovered to 9/9 within one repair tick of the peer returning.
   Still todo: peers discovered via ENR/gossip instead of a static list.
7. **Gossip advertising** *(decided: alongside ENR; ✅ first cut implemented)*.
   Per-node weight holdings are **also** advertised on a **global gossip topic**.
   Division of labor: ENR = the compact, discovery-time summary; gossip = live,
   detailed holdings updates (range acquisitions/drops) without waiting for ENR
   re-resolution. Implemented (LAN-scale epidemic form): a mutex-guarded peer
   table (`peers.zig`) shared by all loops; a `GOSSIP` P2P op
   (announce-and-exchange in one round trip: caller's addr/version/holdings
   merged in, responder's self entry + full table returned); a 3 s gossip loop
   (`gossip.zig`); churn repair now draws candidates from the live table.
   Verified live: C, told only about B, learned origin A transitively within one
   gossip round and repaired 4 missing ranges from it — and A symmetrically
   learned B and C. Swapping this for real gossipsub/discv5 replaces the
   transport, not the table.

### Token-loop peer fetch (✅ first cut implemented — issue #3)

`expert_fetch.zig` + `loom gguf run <store-dir> --peers`: a node running a
deepseek2 model from a *partial* expert-sharded store fetches missing experts
from peers inside the token loop — parallel per-layer prefetch (miss latency =
max, not sum), round-robin holder spreading, per-peer fallback, digest-verify
before disk, fetched shards persisted + advertised (organic heat replication).
Verified on real DeepSeek-V2-Lite: a 33% store (573/1737 shards) produced the
correct completion ("Paris."), streaming 641 experts / 3.5 GB from one peer
with zero failures; token-identical to a full-copy run on the fixture. Still
todo: holder discovery from the gossip table instead of --peers; measured
disk-vs-peer ordering once heat replication puts shards in both tiers; RAM
LRU for beyond-disk-budget caching; serving the distributed engine via node
RPC.

### GGUF → inference (✅ first cut implemented)

The engine runs llama-architecture and **deepseek2-architecture** GGUF models
directly (`loom gguf run`), dispatched on `general.architecture`. GGML
F32/F16/Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K fused kernels over the mmap'd file
(`ggml.zig`); llama: GQA + NORM RoPE + SwiGLU (`llama.zig`); deepseek2 (the
Kimi/DeepSeek/GLM MoE family): MLA + MoE routing + YaRN + byte-level BPE
tokenizer (`deepseek.zig`, `bpe.zig`). Validated on real weights: tinyllamas
(F32/Q4_0/Q8_0, coherent stories) and **DeepSeek-V2-Lite Q4_K_M** (15.7B MoE —
correct factual completions on one CPU core). Note: deepseek2 rope is
NORM-style (adjacent pairs) — DeepSeek's reference interleave-view before
rotate_half nets out to that; NEOX produces degenerate output. Still todo:
serve GGUF models through `loom node`'s RPC; batch >1; SIMD kernels;
IQ-quants.

### Bootnode onboarding + GLM 5.2 sharding plan (decided)

Base model: **GLM 5.2** (744B MoE; 75 MoE layers × 256 experts, 8+1 active).

**Sharding: ✅ implemented** (`weights.zig` expert mode, inspectable via
`loom gguf shard`, auto-selected by `loom node --gguf`; shards are extent
lists; the resident bundle is chunked at 16 MB and always in every node's
want-set — `--hold-fraction` applies to expert shards only. Verified on the
deepseek2 fixture and on real DeepSeek-V2-Lite Q4_K_M: 1,664 expert +
73 resident shards, two-node bootstrap holds all resident + a fraction of
experts, 3-extent shards digest-verified.)

**Shard unit = one expert block ≈ 19 MB int4** (gate/up/down of one expert,
extent-list over the GGUF's 3D expert tensors) → **19,200 routed shards**
(~370 GB corpus) plus one **resident bundle** (~10 GB: attention, shared
experts, embeddings, routers) that *every* node holds in full. Rationale: the
shard must equal the unit the matmul consumes (principle 7 — token-time fetch
is one direct addressed read, no assembly); per-layer shards (4.9 GB) are too
lumpy for redundancy; metadata stays trivial (holdings bitmap 2.4 KB — gossip
carries it whole, ENR carries its hash + seq; manifest ~1 MB).

**16 GB feasibility.** Held shards are a *disk* budget (pread-served), not RAM.
RAM: ~10 GB dense resident (mmap) + ~0.7 GB MLA KV (8k ctx) + 2–3 GB
pinned/LRU expert cache + OS ≈ fits in 16 GB (floor; comfortable at 24–32 GB).
Disk: contribution `--hold-gb`, default ~50 GB ≈ 2,600 shards.

**Redundancy targets.** Replication factor **R = 3 target, R = 2 hard floor**
(principle 3). Swarm sizing: N × hold ≥ R × 370 GB → 16 nodes × 50 GB or
8 × 100 GB reaches R=2; ~22 × 50 GB reaches R=3.

**Bootnode.** An onboarding + coverage accountant, deliberately *not* in the
inference path and *not* a permanent dependency (Ethereum-bootnode discipline):
1. serves the expert-aligned manifest + resident bundle to joining nodes;
2. assigns each joiner the **most under-replicated shards first**
   (capacity-aware, computed from live gossip holdings) — upgrading random
   holdings to guaranteed-coverage assignment;
3. watches per-shard replica counts from the gossip table and nudges
   re-replication when a shard falls below R (eager repair does the pulling);
4. after boot, nodes depend only on gossip + repair — a dead bootnode strands
   nothing already joined.
Decentralization path (later): deterministic assignment via rendezvous hashing
+ coverage computed by every node from gossiped bitmaps; heat-proportional
extra replicas on top (see tension below).

### Design tensions to resolve before implementation

- **Random vs. heat-aware placement.** CLAUDE.md prescribes heat-proportional
  replication (hot experts replicated more); the requirements above say random
  ranges + redundancy. These can compose (random baseline placement, heat-driven
  extra replicas), but the choice must be made explicitly, not blended.
- **GGUF vs. the loom checkpoint format.** The engine currently reads the
  content-addressed manifest/dense/experts layout. GGUF interop means either a
  GGUF reader that maps tensors into the existing store, or replacing the store
  format outright.
- **Networking stack.** ENR + gossip topics + request-response implies an
  Ethereum-style p2p stack (discv5/libp2p family — candidate reuse:
  `blockblaz/enr`, gossipsub, the zeam networking layer) rather than the
  Hyperswarm/Hyperbee stack originally sketched in CLAUDE.md.

## v2 — untrusted-peer verification (unchanged)

Merkle-root integrity on read (already free from v0's store), RLNC WAN
propagation gated behind homomorphic-hash pollution defense, redundant
recompute + M-of-N voting, TEE attestation. See CLAUDE.md.
