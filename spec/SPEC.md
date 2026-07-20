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

## Sizing (GLM 5.2 target; see ../docs/ROADMAP.md for derivation)

19,200 expert shards (~19 MB each) + ~10 GB resident bundle. A complete
committee at 50 GB/node needs ≥8 nodes (≈2,600 expert shards each); at `R=2`
inside one committee, ≥15. Holdings bitmap 2.4 KB; manifest ~1 MB.
