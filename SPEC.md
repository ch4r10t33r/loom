# Loom p2p layer — spec

Architecture spec for the weight-distribution p2p layer. CLAUDE.md holds the
inference-side design; ROADMAP.md tracks status. Where this document and older
sketches conflict, this document wins for the p2p layer.

## Roles

**Bootnode** — the onboarding authority (Ethereum-bootnode discipline: never in
the inference path, no permanent dependency after join). Typically the origin
holder of the full GGUF. Responsibilities:

1. Serve the expert-aligned manifest (shard digests + extents, Merkle-rooted
   version id) to joining nodes.
2. **Assign each connecting node to a shard committee**, and within that
   committee assign it a shard want-set (least-covered shards first, up to the
   node's declared capacity).
3. Track per-committee coverage so redundancy targets are met by construction,
   not probability.

**Node** — holds the resident bundle (mandatory) plus its assigned expert
shards; serves them to peers; runs inference locally, fetching experts it
lacks at token time.

## Shard committees

A **committee** is a group of nodes that *collectively holds the complete
shard set* — a self-sufficient serving cell for one model version.

- **Completeness invariant:** each committee must have **at least one node
  holding every shard** of the GGUF set. Until a forming committee reaches
  completeness, the mesh (and the bootnode, which holds everything) covers the
  gaps.
- **Intra-committee redundancy:** more than one node in a committee may hold
  the same shard. The bootnode fills a committee toward a redundancy target
  `R` (default 2): joiners are assigned the currently least-covered shards.
- **Committee lifecycle:** joiners go to the first committee whose minimum
  shard coverage is below `R`. When every shard in every existing committee
  has coverage ≥ `R` (saturated), the next joiner opens a new committee.
- **Heartbeats:** every node maintains a regular heartbeat (`PING`/`PONG`,
  5 s interval) with its committee members and tracks their liveness.
  A member that stops answering is marked dead locally (and its shards
  become candidates for re-replication via the eager repair loop).

## Global gossip network

Independent of committees, **every node regularly announces the weights it
holds on the global gossip topic** (addr + manifest version + holdings
bitmap, epidemic exchange, 3 s interval). Receiving peers merge these
announcements into their peer table — the **mesh**.

The mesh serves two purposes:

1. **Discovery** — transitive: a node learns holders it was never configured
   with.
2. **Fallback routing** — see query path below.

ENR (planned) carries the compact summary (holdings-bitmap hash + seq); the
gossip topic carries the full bitmap.

## Query path for a weight / expert

When a node needs a shard it does not hold (boot-time sync, eager repair, or
an expert fetch inside the token loop):

1. **Committee first:** ask the allotted peer(s) within its own committee
   that hold the shard (round-robin across replicas — client-side spreading).
2. **Mesh fallback:** if no committee holder responds, query any peer from
   the gossip-built mesh whose announced holdings cover the shard.
3. Every response is digest-verified against the manifest before use;
   a shard obtained at token time is persisted and announced on the next
   gossip round (fetch-on-demand = organic heat replication).

A request fails only when neither the committee nor the mesh has a reachable
holder — and the eager repair loop keeps working to make that state
transient.

## Wire protocol (current line-protocol form)

| Op | Response | Purpose |
|---|---|---|
| `JOIN addr=<h:p> fraction=<f>` | `COMMITTEE id=<n> members=<a,b,...> assign=<hex bitmap>` | bootnode assigns committee + want-set (capacity ≈ f × expert shards) |
| `COMMITTEES` | summary lines | bootnode debug: per-committee members, min/max coverage, complete/saturated |
| `MANIFESTFILE` | serialized manifest | digests + extent lists, root-verified by the client |
| `GETR <i>` | shard bytes | fetch one shard (committee or mesh) |
| `GOSSIP addr=.. version=.. holdings=..` | peer table | the global announce + mesh exchange |
| `PING` | `PONG` | heartbeat |

The same ops move to gossipsub/discv5 transports later; the semantics above
are the spec.

## Sizing (GLM 5.2 target; see ROADMAP for derivation)

19,200 expert shards (~19 MB each) + ~10 GB resident bundle. A complete
committee at 50 GB/node needs ≥8 nodes (≈2,600 expert shards each); at `R=2`
inside one committee, ≥15. Holdings bitmap 2.4 KB; manifest ~1 MB.
