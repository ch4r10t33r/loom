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

1. **Local-first model resolution.** If a model is already present in local data
   storage, load it from there; only download from remote otherwise.
   (Implemented in `hf.zig::resolve`; a standing contract as features grow.)
2. **GGUF weight distribution.** A loaded **GGUF** file's weights are distributed
   across multiple nodes. Ranges are chosen **randomly**, and the same range may
   be held by several nodes for **redundancy**.
3. **Boot-time peer sync.** A new node requests weight ranges from its peers via
   a **request-response protocol** rather than re-downloading from origin.
4. **Hardfork upgrades.** A majority of nodes agreeing on a new version of the
   GGUF file triggers a hardfork — a coordinated model-version upgrade across
   the swarm.
5. **ENR weight advertising** *(open question)*. Each peer reports the weights it
   holds as part of its **ENR + metadata**, so nodes can deliberately peer with
   nodes holding *different* ranges (coverage-seeking peering).
6. **Churn repair** *(open question)*. When a peer disconnects, a node actively
   seeks a replacement peer that holds the required weight range.
7. **Gossip advertising** *(open question)*. Whether per-node weight holdings
   should be advertised via a **global gossip topic** instead of — or alongside —
   ENR metadata.

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
