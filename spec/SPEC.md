# Loom p2p layer — spec

Architecture spec for the weight-distribution p2p layer. [../CLAUDE.md](../CLAUDE.md) holds the
inference-side design; [../docs/ROADMAP.md](../docs/ROADMAP.md) tracks status. Where this document and older
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

**Full node** — holds the resident bundle (mandatory) plus its assigned
expert shards; serves them to peers; runs inference locally, fetching experts
it lacks at token time; **meters** the inference it serves (below).

**Light node** — no weights, no store, no engine; runs on low-memory devices.
Serves the same local JSON RPC as a full node and **delegates** every request
to a full node (round-robin with failover across configured backends; gossip
discovery of RPC-serving full nodes is a follow-up). Stamps its client id on
each request so the serving full node can meter it.

## Metering & compensation

Full nodes are compensated by light nodes for serviced requests. Per-client
ledger on each full node (per-provider, not global):

```
allowance(client) = free_quota + credits(client) − used(client)
used unit         = prompt tokens processed + tokens generated
```

- Gate: a request from a client with `allowance == 0` is refused with
  `{"ok":false,"error":"payment_required"}` before any compute. The final
  request may overdraw by one generation (charged from actuals, clamped).
- Every metered response appends `"cost"` and `"balance"`.
- `{"method":"credit","client":C,"amount":N,"proof":P}` adds credits.
  **v1 does not verify `proof`** (trusted swarm); a settlement rail replaces
  exactly this verification. `{"method":"tab","client":C}` returns
  used/balance.
- Light nodes tally their own spend from response `cost` fields, so both
  sides of the ledger exist independently.

**v1 is a trusted-operator accounting demo, not a settlement system.**
Hardened since the initial cut: `credit` is now **admin-token gated** (no
token ⇒ the op is disabled; a payment rail replaces the token check),
`max_tokens` is **clamped to the client's remaining allowance** (overdraw
bounded to the prompt), the account map is **capped** and the default free
quota lowered, and the **light node force-stamps its own client id** (a
caller-supplied `client` is dropped, so light nodes can't be open proxies).
Still gaps, none defended today: client ids are self-asserted strings
(rotate to reset the free quota); ledgers are per-provider (round-robin
across N nodes ⇒ N× free quota); no signed receipts. Prerequisites for real
compensation: client keypair/HMAC identity, payment-proof verification,
signed usage receipts.


## Trust model & manifest bootstrap (v1)

v1 assumes a **cooperative, operator-run swarm** with cryptographic integrity
*relative to a trusted manifest*. This section is normative about what is and
is not defended.

**Assets:** model correctness (served weights match the intended model);
availability (a needed shard is reachable); metering integrity (accounting is
honest). **Adversaries considered:** corrupt storage / bit-rot; a peer
serving wrong bytes; a peer lying about holdings; RPC abuse. **Out of scope
for v1** (deferred to v2, see [../CLAUDE.md](../CLAUDE.md)): Byzantine
compute, Sybil identities, coordinated availability attacks, an untrusted
bootnode, payment fraud.

**What content addressing buys.** Every shard is verified against its
manifest digest before disk or matmul, so a wrong-bytes peer or corrupt disk
is caught for free — *given a trusted manifest*. Version pinning
(`manifest_version` on every ExpertRequest and on gossip/heartbeat) refuses
cross-version peers, isolating hardforks and alternate swarms.

**Manifest trust bootstrap.** Content addressing secures integrity *within* a
manifest, not the *choice* of manifest — a poisoned-but-internally-consistent
alternate root would verify against itself. v1 establishes the root by:

1. **Origin/operator config** — the operator running `loom node --gguf` on a
   known-good file is the root of trust; its Merkle root is authoritative for
   that swarm.
2. **Trust-on-first-use (TOFU)** — a bootstrapping node adopts the root from
   the first responsive peer's `MANIFESTFILE` and pins it; every subsequent
   peer must match (`PeerVersionMismatch` otherwise). A hostile *first*
   contact can pin a bad root — the known v1 weakness.

On root mismatch a node refuses the peer wholesale (it never mixes roots).
Future hardening (not in v1): operator-signed roots distributed out-of-band,
multiple independent registries, and a coverage-challenge protocol so a
committee can prove completeness rather than assert it.

**Bootnode as a trusted placement service.** The bootnode assigns committees
and want-sets and, until a committee is complete, backstops coverage as
origin holder — so it *is* trusted for placement and *is* in the availability
path pre-completeness. It is *not* in the token-loop inference path, and
joined nodes survive its death. A malicious bootnode can strand or bias
committees; v1 assumes it is honest and operator-run. Verifiable/redundant
assignment is future work.

**False-holdings failure mode.** Holdings bitmaps (gossip announces,
heartbeat digests) are unverified claims — no proof of possession. A lying
holder is detected only reactively: a `GETR`/ExpertRequest returns
`not_held` or wrong bytes, and the requester falls through to the next peer
(digest verification blocks the wrong bytes; availability degrades, integrity
does not). There is no reputation, periodic challenge, or probe-on-suspicion
yet — so **committee completeness is a construction-time property under
honest participation, not a cryptographic guarantee.** Adding a sampled-`GETR`
challenge + holder reputation is the planned mitigation.

## Resource governance & failure modes (v1 status)

Several safety limits are **specified here but only partially enforced in v1**;
they are called out so implementations and readers do not assume protection
that is absent.

**Serving SLA under incomplete committees.** Completeness at R = 1 is one
death from a hole, and heartbeat detection lags up to one interval (5 s).
During a gap, a token-loop fetch for the missing shard: (1) tries committee
holders, (2) falls through to the mesh, (3) if no reachable holder anywhere,
**the token-loop fetch errors** (inference fails loudly; it is not silently
wrong). "Can still serve via the mesh" and "the committee invariant holds"
are distinct states — a node may serve while its committee is technically
incomplete, and eager repair (2 s) works to restore the invariant. This is
eventual, not synchronous with death.

**Weight integrity, end to end.** The version id now **binds the shard layout**
(extents, file_size, n_resident, mode) into the Merkle root, and parse rejects
any manifest that is not an exact partition of `[0, file_size)` — a peer can no
longer serve digests that verify while pointing at attacker-chosen offsets. The
**hot path re-verifies**: `readRangeVerified` re-hashes on local read and on
`GETR`/`ExpertRequest` serve (clearing the bit + triggering repair on
mismatch); store-open **re-audits every held shard**; the distributed run
**fails closed** unless all resident shards are held+verified; the loom-format
ExpertCache **publishes a slot only after a successful verify**. All
implemented and tested (a corrupted on-disk shard is caught on open, cleared,
and re-fetched).

**Disk cap for organic replicas.** Fetched shards are persisted (organic heat
replication), which without a bound drives busy nodes toward the full corpus.
Still a v1 gap (**not yet enforced**): a `max_shard_bytes` cap above which
opportunistically-fetched shards are LRU-evicted while **assigned want-set
shards are pinned**. Organic replicas are cache and do **not** count toward a
committee's R target. Until the cap ships, operators must size disk for
worst-case growth on hot nodes.

**Admission control & priority.** ExpertResponse carries a `busy` status, but
v1 has no rate limiting, per-peer quota, or bandwidth reservation. The
intended priority order (v1 gap): **token-loop fetch > eager repair > bulk
sync**, with per-peer request quotas, so a light node or attacker cannot soak
expert-serving capacity and starve inference. Not implemented; documented as
required.

**Peer-table bounds & mesh authenticity.** The peer table is now **bounded**
(`MAX_PEERS`, coldest-`last_seen` eviction on overflow) and holdings updates
carry a **monotonic sequence** so a stale/replayed bitmap cannot clobber a
fresh one (implemented). Frame decode is snappy-capped (`decodeWithMax`,
`MAX_BODY_BYTES`) against decompression bombs, and connection handlers are
semaphore-bounded against thread-per-accept floods (implemented). Still a v1
gap: **authenticated announces** (fake-holder mesh poisoning is possible on an
untrusted network) and exponential backoff / probation on repeatedly-dead
peers. v1 runs LAN-scale and operator-trusted.

**Transport security.** The RPC and P2P transports are plaintext TCP with no
TLS and no authentication — acceptable for a lab/LAN operator-run swarm,
unacceptable for a public "service economy." TLS + peer authentication is a
prerequisite for any untrusted deployment.

**Failure-mode catalog (v1 behavior):**

| Failure | v1 behavior |
|---|---|
| Bootnode down (post-join) | No effect on joined nodes; gossip mesh + repair continue; new joins blocked until it returns or another registry exists |
| Network partition | Each side serves from its own holdings + reachable mesh; repair reconverges on heal |
| Peer lies about holdings | Caught on fetch (`not_held`/wrong bytes → digest reject → next peer); availability dips, integrity preserved |
| Disk full | Writes fail; the shard stays unheld and is retried; no cap/eviction yet (see above) |
| Version skew mid-fetch | ExpertRequest pins version; a re-versioned peer returns `version_mismatch` → fall through |
| Committee hole (member died) | Token fetch → mesh fallback; errors only if no holder anywhere; repair restores R eventually |
| RPC/expert flooding | No admission control in v1; capacity can be soaked (documented gap) |

## Architecture diagrams

### Full node

```
                                ┌─────────────────────────────────────────────────────┐
                                │                      FULL NODE                      │
                                │                                                     │
   apps / light nodes ─────────▶│  ┌───────────────┐      ┌──────────────────────┐    │
   {"prompt",...,"client":ID}   │  │  RPC server   │─────▶│   Metering ledger    │    │
   ◀── {..,"cost","balance"} ───│  │  (JSON, TCP)  │ gate │ quota+credits−usage  │    │
                                │  └──────┬────────┘      │ payment_required     │    │
                                │         │ generate      │ credit / tab ops     │    │
                                │         ▼               └──────────────────────┘    │
                                │  ┌────────────────────────────────┐                 │
                                │  │        Inference engine        │                 │
                                │  │  dense path (mmap resident):   │                 │
                                │  │  MLA attn · router · shared    │                 │
                                │  │  KV cache (compressed, local)  │                 │
                                │  └───────┬────────────────────────┘                 │
                                │          │ routed-expert reads (per MoE layer)      │
                                │          ▼                                          │
                                │  ┌────────────────────────────────┐                 │
                                │  │  Expert access (tiered)        │                 │
                                │  │  1. held shard → pread         │                 │
                                │  │     (page cache = RAM tier)    │                 │
                                │  │  2. miss → parallel prefetch ──┼──────────┐      │
                                │  └───────┬────────────────────────┘          │      │
                                │          │ verified writes                   │      │
                                │          ▼                                   ▼      │
                                │  ┌────────────────────────────────┐   ┌───────────┐ │
                                │  │  Weight store (sparse GGUF)    │   │  Expert   │ │
                                │  │  manifest (digests+extents,    │   │  fetch    │ │
                                │  │  Merkle version id)            │   │ committee │ │
                                │  │  holdings / wanted bitmaps     │   │ first,    │ │
                                │  └───────────────▲────────────────┘   │ then mesh │ │
                                │                  │ serve / repair     └─────┬─────┘ │
                                │  ┌───────────────┴────────────────┐         │       │
   peers ──────────────────────▶│  │  P2P server (TCP, frames+text) │         │       │
   GETR/ExpertRequest, GOSSIP,  │  │  shard serving · gossip ·      │         │       │
   Heartbeat, JOIN (bootnode)   │  │  heartbeat answers · JOIN      │         │       │
                                │  └───────────────┬────────────────┘         │       │
                                │                  │                          │       │
                                │  ┌───────────────▼────────────────┐         │       │
                                │  │  Peer table (mesh)             │◀────────┘       │
                                │  │  addr·version·holdings·        │  holder lookup  │
                                │  │  committee id                  │                 │
                                │  └───────▲────────▲───────▲───────┘                 │
                                │          │        │       │                         │
                                │   ┌──────┴──┐ ┌───┴────┐ ┌┴─────────┐               │
                                │   │ gossip  │ │ heart- │ │  eager   │               │
                                │   │ loop 3s │ │ beat 5s│ │ repair 2s│               │
                                │   └─────────┘ └────────┘ └──────────┘               │
                                └─────────────────────────────────────────────────────┘
```

### Light node

```
                       ┌───────────────────────────────────────────┐
                       │                LIGHT NODE                 │
                       │   (no weights · no store · no engine)     │
                       │                                           │
  local apps ─────────▶│  ┌─────────────────┐   ┌───────────────┐  │
  {"prompt",...}       │  │ Local RPC       │──▶│  Delegator    │  │
  ◀── full response ───│  │ (same protocol) │   │ stamp client  │  │
      + cost/balance   │  └─────────────────┘   │ id · round-   │  │
                       │                        │ robin +       │  │
                       │  ┌─────────────────┐   │ failover      │  │
                       │  │  Spend tally    │◀──┤               │  │
                       │  │ (from "cost")   │   └───────┬───────┘  │
                       │  └─────────────────┘           │          │
                       └────────────────────────────────┼──────────┘
                                                        │ metered JSON RPC
                                     ┌──────────────────┼──────────────────┐
                                     ▼                  ▼                  ▼
                               ┌───────────┐      ┌───────────┐      ┌───────────┐
                               │ FULL NODE │      │ FULL NODE │      │ FULL NODE │
                               │ (ledger A)│      │ (ledger B)│      │ (ledger C)│
                               └───────────┘      └───────────┘      └───────────┘
                                 per-provider ledgers: each full node meters
                                 the service it renders
```

### Swarm topology

```
                       ┌────────────┐  JOIN → committee id,
        new node ─────▶│  BOOTNODE  │  members, assigned want-set
                       │ (registry, │  (least-covered-first)
                       │  full GGUF)│  never in the inference path
                       └─────┬──────┘
                             │ manifest + initial shards
        ┌────────────────────┼─────────────────────┐
        ▼                    ▼                     ▼
  ┌───────────┐  heartbeat ┌───────────┐         ┌───────────┐
  │ FULL NODE │◀──── 5s ──▶│ FULL NODE │   ...   │ FULL NODE │   committee 0
  │  shards   │  shard     │  shards   │         │  shards   │   (complete: every
  │  A..M     │  fetch     │  N..Z     │         │  A..K     │    shard ≥1 holder)
  └─────▲─────┘            └─────▲─────┘         └─────▲─────┘
        │                        │                     │
        └────────────────────────┼─────────────────────┘
                                 │ global gossip mesh (3 s announces:
                                 │ addr · version · committee · holdings bitmap)
        ┌────────────────────────┼─────────────────────┐
        ▼                        ▼                     ▼
  ┌───────────┐            ┌───────────┐         ┌───────────┐
  │ FULL NODE │            │ FULL NODE │   ...   │ light     │   committee 1 +
  │           │            │           │         │ nodes     │   light clients
  └───────────┘            └───────────┘         └───────────┘
```

## Shard committees

A **committee** is a group of nodes that *collectively holds the complete
shard set* — a self-sufficient serving cell for one model version.

- **Completeness invariant:** each committee must have **at least one node
  holding every shard** of the GGUF set. Until a forming committee reaches
  completeness, the mesh (and the bootnode, which holds everything) covers the
  gaps.
- **Intra-committee redundancy:** more than one node in a committee may hold
  the same shard. The bootnode fills a committee toward a redundancy target
  `R` (**R = 3 target, R = 2 floor; `--r-target` default 2**): joiners are
  assigned the currently least-covered shards. *Completeness* (every shard
  held by ≥ 1 member, i.e. R ≥ 1) and *redundancy* (R ≥ 2) are distinct
  thresholds. Only **assigned** holdings count toward R; opportunistic
  organic replicas (§ Resource governance) are cache, not coverage.
- **Committee lifecycle:** joiners go to the first committee whose minimum
  shard coverage is below `R`. When every shard in every existing committee
  has coverage ≥ `R` (saturated), the next joiner opens a new committee.
- **Heartbeats:** every node maintains a regular heartbeat (wire Heartbeat
  frames, 5 s interval) with its committee members — the member set is the
  join-time seed plus the gossip-derived committee view, so membership stays
  current as the committee grows. A member that stops answering is marked
  dead locally (and its shards become candidates for re-replication via the
  eager repair loop).

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

## Wire messages v1 (binary frames, snappy)

The three structured message families — heartbeat, gossip announce, and the
expert request/response — are binary frames. All integers little-endian.
Text ops remain for debugging; frames ride the same TCP connection behind a
`FRAME <len>` header line.

**Frame envelope** (8-byte header + body):

```
magic:    "LM"          (2 bytes)
type:     u8            (message type, below)
flags:    u8            bit0 = body is snappy-compressed
body_len: u32           length of the (possibly compressed) body
body:     [body_len]u8
```

Compression policy: the encoder compresses the body with snappy and keeps the
compressed form only if it is smaller (adaptive). In practice: holdings
bitmaps and manifests compress well; quantized expert payloads are
high-entropy and usually ship raw. Decoders must handle both.

**Heartbeat — type 0x01 (request), 0x02 (response).** Committee-internal,
every 5 s. Carries liveness plus exactly the state a committee member needs
to act on: version drift detection, holdings freshness, and a load hint for
client-side spreading. One exchange refreshes both sides (the response is
the same container).

```
proto:            u8       (= 1)
committee_id:     u32      (0xFFFFFFFF = not in a committee)
manifest_version: [32]u8   (Merkle root; zeros = no store)
holdings_seq:     u64      (monotonic; bumps on every holdings change)
holdings_digest:  [32]u8   (sha256 of the holdings bitmap)
load:             u16      (in-flight expert requests being served)
sent_at_ns:       i64      (sender clock; RTT/skew measurement)
addr_len:         u8
addr:             [addr_len]u8   (sender's advertised host:port)
```

Rationale: the full bitmap does NOT ride the heartbeat — `holdings_seq` +
`holdings_digest` let the receiver detect staleness cheaply; when they
change, the fresh bitmap arrives via the next gossip announce (or a pull).
A committee member is marked dead after missed heartbeats; its shards
become re-replication candidates.

**Announce — type 0x03.** The global gossip record (every 3 s): what a node
tells the network about the weights it holds. This is the record receivers
merge into their mesh table.

```
proto:            u8
committee_id:     u32
manifest_version: [32]u8
holdings_seq:     u64
addr_len:         u8
addr:             [addr_len]u8
bitmap_len:       u32
holdings_bitmap:  [bitmap_len]u8   (1 bit per shard; frame-level snappy
                                    typically compresses this well)
```

Receivers keep the entry with the highest `holdings_seq` per addr. ENR
(planned) carries `manifest_version + holdings_seq + holdings_digest` only
(fits the 300-byte limit); the gossip announce carries the full bitmap.

**AnnounceBatch — type 0x04.** The gossip exchange response: the responder's
own Announce followed by its whole table, so one round trip both announces
and syncs the mesh view. Frame-level snappy compresses across the similar
bitmaps of many peers.

```
count:   u32
entries: count x { len: u32, body: [len]u8 }   (each an Announce body)
```

Because announces carry `committee_id`, the mesh table doubles as the
**gossip-derived committee view**: committee membership is not static-at-join —
earlier members learn later joiners from their announces, and the heartbeat
loop targets `seed ∪ {table entries with my committee_id}`. This view (not
the bootnode) is what members rely on after joining, so a dead bootnode
still strands nothing.

**ExpertRequest — type 0x10.** Ask a remote peer (committee member first,
then mesh) for one expert shard the requester doesn't hold.

```
proto:            u8
request_id:       u64      (echoed in the response; enables pipelining)
manifest_version: [32]u8   (hard guard: server refuses other versions)
shard_id:         u32      (manifest shard index — the expert block)
```

**ExpertResponse — type 0x11.**

```
request_id:  u64    (echo)
status:      u8     (0 ok | 1 not_held | 2 version_mismatch |
                     3 busy | 4 bad_request)
shard_id:    u32
payload_len: u32
payload:     [payload_len]u8   (the complete expert block: gate|up|down
                                extents concatenated; empty unless status=0)
```

The requester MUST verify the payload against its own manifest digest for
`shard_id` before use — the response deliberately carries no digest; the
manifest is the only trust root. `busy` tells the client to spread to
another replica (pairs with the heartbeat `load` hint).

## Wire protocol (legacy / debug line form)

> **Normative transport is the binary frame protocol above.** This
> line-delimited form is transitional: it still serves `JOIN`, `GETR`,
> `GOSSIP`/`TABLE`, `PING`, and the `MANIFEST*` ops today (migration in
> progress), and is convenient for `nc`-driven debugging. New message types
> are defined as frames; do not add line ops.

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

## Sizing (GLM 5.2 target)

Derivation and the full deployment plan live in
[../docs/ROADMAP.md](../docs/ROADMAP.md).

19,200 expert shards (~19 MB each) + a resident bundle. Committee sizing at
50 GB/node: **completeness (R ≥ 1) needs ≥ 8 nodes** (≈2,600 expert shards
each), **R = 2 needs ≥ 15**, **R = 3 needs ≥ 22**. Holdings bitmap 2.4 KB;
manifest ~1 MB.

**The resident bundle is the real floor, and the 16 GB minimum is
conditional on it.** Every node holds it in full, in RAM (mmap'd). It is
*not* free — DeepSeek/GLM shared experts and embeddings dominate it, and MLA
KV grows with context. A per-tensor-class breakdown for the target GGUF
(embeddings, attention projections incl. q/kv-LoRA, shared-expert FFNs,
router, norms) and a KV-bytes-vs-context table **must be published from the
real converted GLM 5.2 GGUF before the 16 GB row in the deployment table is
asserted as a default** rather than a conditional target. For
DeepSeek-V2-Lite the measured resident bundle is 0.78 GB (73 chunks); the
GLM 5.2 ~10 GB figure is an int4 estimate pending the real file.
