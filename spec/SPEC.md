# Loom p2p layer: spec

Architecture spec for the weight-distribution p2p layer. [../CLAUDE.md](../CLAUDE.md) holds the
inference-side design; [../docs/ROADMAP.md](../docs/ROADMAP.md) tracks status. Where this document and older
sketches conflict, this document wins for the p2p layer.

## Roles

**Bootnode:** the onboarding authority, following Ethereum-bootnode discipline
(never in the inference path, no permanent dependency after join). Typically the
origin holder of the full GGUF. Responsibilities:

1. Serve the expert-aligned manifest (shard digests plus extents, Merkle-rooted
   version id) to joining nodes.
2. Assign each connecting node to a shard committee, and within that committee
   assign it a shard want-set (least-covered shards first, up to the node's
   declared capacity).
3. Track per-committee coverage so redundancy targets are met by construction.

**Full node:** holds the resident bundle (mandatory) plus its assigned expert
shards; serves them to peers; runs inference locally, fetching experts it lacks
at token time; meters the inference it serves (below).

**Light node:** no weights, no store, no engine; runs on low-memory devices.
Serves the same local JSON RPC as a full node and delegates every request to a
full node (round-robin with failover across configured backends; gossip discovery
of RPC-serving full nodes is a follow-up). Stamps its client id on each request so
the serving full node can meter it.

## Client API (north-facing)

A node exposes two request-serving surfaces over TCP. Both sit in front of the
same engine and the same metering ledger; both are orthogonal to the p2p wire
protocol (the south-facing expert-fetch frames), which this section leaves
unchanged.

1. **Native JSON RPC** (`--rpc-port`, default 8770). The line-delimited
   `{"prompt":...}` protocol described under Metering below. This is the
   internal protocol light nodes delegate over.
2. **OpenAI-compatible HTTP API** (`--openai-port`, off by default in v1). An
   HTTP/1.1 endpoint that speaks the OpenAI schema so existing clients
   (OpenWebUI, Continue, aider, the OpenAI SDKs, curl) talk to a Loom node with
   no adapter. Routes: `GET /v1/models`, `POST /v1/chat/completions`,
   `POST /v1/completions`, `GET /health`. It is a thin translation layer over the same
   `engine.generate` path and the same `Meter`.

**Choice of the OpenAI schema over MCP.** MCP is a tool/context protocol; a node
serving completions is more naturally an MCP client (a model that can call
tools) than an MCP server. The OpenAI HTTP schema is the de-facto local-inference
serving contract (llama.cpp server, ollama, vLLM, ZINC all expose it), so it is
the correct north-facing surface for request serving. MCP-client support is a
separate, later concern and does not belong on the serving path.

**Identity and metering.** The OpenAI surface carries client identity in the
`Authorization: Bearer <token>` header rather than a request-body field. The
bearer token is the client id and credit key, so it is out-of-band from the
prompt and not forgeable by prompt content, an improvement over the native RPC's
self-asserted `client` string. Usage maps directly onto the OpenAI `usage`
object (`prompt_tokens` + `completion_tokens`), which is already the ledger's
cost unit; the `model` field maps to the manifest version and a mismatch is
refused the same way a cross-version peer is.

**Request-to-prompt mapping.** `messages[]` are rendered to the model's prompt
via its chat template (read from GGUF metadata alongside the tokenizer).
Token-by-token generation maps to OpenAI SSE streaming (`stream: true`) as
`data:` chunks terminated by `data: [DONE]`.

**v1 status.** The component ([src/node/openai.zig](../src/node/openai.zig))
implements the HTTP transport, routing, the OpenAI request/response structs,
bearer-token identity, and real generation for `POST /v1/chat/completions` and
`POST /v1/completions`: the loaded model runs over the shared engine mutex and
returns a real `usage` object, metered by bearer id, byte-identical to the native
RPC path for the same prompt/seed. `stream:true` streams the completion as OpenAI
Server-Sent Events (one `data:` chunk per token, then `[DONE]`) over both the
loom and distributed-GGUF engines. Light nodes delegate the OpenAI surface to
full nodes (metered reverse proxy). Chat `messages[]` are rendered with the
model's chat template, auto-detected from the GGUF `tokenizer.chat_template`
metadata (overridable via `--chat-format`). Marker tokens (chatml, llama3,
gemma, control tokens) tokenize atomically to their ids via the special-token
matcher when the model's vocab defines them. Special-token parsing is off for raw
prompts and for text-marker chat formats (deepseek/llama2/mistral), so untrusted
input cannot inject a control token; it is on only for special-marker chat
scaffolds (chatml/llama3/gemma), where making message content injection-safe too
requires segment encoding (follow-up).

## Delegated generation (optional)

`GEN <json>` runs one generation on the receiving node and returns it:
`GENR ok=1 prompt_tokens=<n> completion_tokens=<n> tok_per_s=<f>
hit_rate=<f> len=<n>` followed by exactly `len` bytes of generated text.
The JSON object carries `prompt` (string, <=64 KiB), `max_tokens`
(<=2048), `temperature`, `seed`, and `parse_special` (bool; the sender
renders its own chat template and says whether the scaffold uses special
markers). A node whose engine is not yet serving answers `ERR not_ready`;
a node with no engine, `ERR no_engine`. Delegation shares v1's trust
plane: the answer is unauthenticated text from a trusted-swarm peer, and
requests are not metered. A cold node (holdings fraction under its
`--delegate-below`, default 0.5) forwards to the warmest live peer whose
holdings fraction is at least 0.9, and falls back to local generation on
any failure.

## Draft verification (optional, DSD)

`DRAFT <json>` verifies a window of speculatively drafted tokens against the
receiving node's exact model (greedy). The JSON object carries `ctx` (array of
token ids, the full context so far: prompt plus every token already emitted;
<=32768 ids) and `draft` (array of token ids, <=8). Reply:
`DRAFTR ok=1 accepted=<n> correction=<id>` — the target model agrees with the
first `accepted` draft tokens, and `correction` is its own greedy token at the
first disagreement (or the bonus token after the window when everything was
accepted). The emitted stream `ctx ++ draft[0..accepted] ++ correction` is
therefore token-identical to the receiving node generating alone.

The verification KV cache persists between requests and is keyed by nothing:
the receiver reuses the longest common prefix of the previous request's token
history and `ctx`, re-feeding only the delta, so consecutive windows of one
generation cost one incremental forward each. There is no session id; the
context is the session. A node whose engine is not serving answers
`ERR not_ready`; no engine, `ERR no_engine`; ids out of vocab range or
over-long arrays, `ERR bad_draft`. Same trust plane as `GEN`: token ids from a
trusted-swarm peer, unmetered. Greedy only — a sampled (temperature > 0)
generation must use `GEN`.

## Batched shard fetch

`PACKR <id,id,...>` (<=256 ids) streams, per id, either
`DATA <i> len=<l> sha256=<hex>` followed by the raw bytes, or
`ABSENT <i>`, and finishes with `END`. Semantics per shard are identical
to `GETR`; the batch exists because a round trip per shard puts RTT times
the shard count on the critical path of every sync. Receivers verify
every shard against their own manifest digests, exactly as with `GETR`,
and degrade to per-shard `GETR` when a peer answers `ERR unknown`.

## Heat (sync ordering hint)

`HEAT` asks a peer which shards it serves most. The reply is one line,
`HEAT n=<k> ids=<csv>`, the ids of up to 2048 shards this peer has served
via `GETR`, descending by serve count; a node with no counts replies
`HEAT n=0 ids=`. The counts are per-process and reset on restart. A syncing
client fetches mandatory resident chunks first, then the peer's heat list,
then the remaining wanted shards in index order. The hint is best-effort:
an unknown-command error or an empty list degrades to index order, and a
lying peer can only reorder a download whose every shard is still
digest-verified against the manifest.

## Alpha telemetry (optional)

`METRICS <json>` carries one node's opt-in operational report to a peer that
collects it. A node started with `--alpha-ingest <path>` appends the JSON
payload as one line to that file and replies `OK`; every other node replies
`ERR no_ingest` and stores nothing. Payloads over 8192 bytes are refused
with `ERR bad_metrics`; a collector whose file exceeds 256 MB refuses with
`ERR ingest_failed`. The payload is a single JSON object whose complete
field list is documented in docs/ALPHA.md; it never contains prompts,
generated text, or RAG content. Reports are fire-and-forget: a sender does
not retry a failed report.

## Metering and compensation

Full nodes are compensated by light nodes for serviced requests. Each full node
keeps a per-client ledger (per-provider, not global):

```
allowance(client) = free_quota + credits(client) - used(client)
used unit         = prompt tokens processed + tokens generated
```

- Gate: a request from a client with `allowance == 0` is refused with
  `{"ok":false,"error":"payment_required"}` before any compute. The final request
  may overdraw by one generation (charged from actuals, clamped).
- Every metered response appends `"cost"` and `"balance"`.
- `{"method":"credit","client":C,"amount":N,"proof":P}` adds credits. v1 does not
  verify `proof` (trusted swarm); a settlement rail replaces exactly this
  verification. `{"method":"tab","client":C}` returns used/balance.
- Light nodes tally their own spend from response `cost` fields, so both sides of
  the ledger exist independently.

v1 is a trusted-operator accounting demo, not a settlement system. Hardened
since the initial cut: `credit` is now admin-token gated (no token means the op is
disabled; a payment rail replaces the token check), `max_tokens` is clamped to the
client's remaining allowance (overdraw bounded to the prompt), the account map is
capped, the default free quota is lowered, and the light node force-stamps its own
client id (a caller-supplied `client` is dropped, so light nodes cannot be open
proxies). Gaps that remain undefended today: client ids are self-asserted strings
(rotate to reset the free quota); ledgers are per-provider (round-robin across N
nodes gives N times the free quota); no signed receipts. Prerequisites for real
compensation: client keypair/HMAC identity, payment-proof verification, and signed
usage receipts.


## Trust model and manifest bootstrap (v1)

v1 assumes a cooperative, operator-run swarm with cryptographic integrity
relative to a trusted manifest. This section is normative about what is and is not
defended.

**Assets:** model correctness (served weights match the intended model);
availability (a needed shard is reachable); metering integrity (accounting is
honest). **Adversaries considered:** corrupt storage or bit-rot; a peer serving
wrong bytes; a peer lying about holdings; RPC abuse. **Out of scope for v1**
(deferred to v2, see [../CLAUDE.md](../CLAUDE.md)): Byzantine compute, Sybil
identities, coordinated availability attacks, an untrusted bootnode, payment
fraud.

**What content addressing provides.** Every shard is verified against its manifest
digest before disk or matmul, so a wrong-bytes peer or corrupt disk is caught for
free, given a trusted manifest. Version pinning (`manifest_version` on every
ExpertRequest and on gossip/heartbeat) refuses cross-version peers, isolating
hardforks and alternate swarms.

**Manifest trust bootstrap.** Content addressing secures integrity within a
manifest, not the choice of manifest: a poisoned-but-internally-consistent
alternate root would verify against itself. v1 establishes the root by:

1. **Origin/operator config.** The operator running `loom node --gguf` on a
   known-good file is the root of trust; its Merkle root is authoritative for that
   swarm.
2. **Trust-on-first-use (TOFU).** A bootstrapping node adopts the root from the
   first responsive peer's `MANIFESTFILE` and pins it; every subsequent peer must
   match (`PeerVersionMismatch` otherwise). A hostile first contact can pin a bad
   root; this is a known v1 weakness.

On root mismatch a node refuses the peer wholesale (it never mixes roots). Future
hardening (not in v1): operator-signed roots distributed out-of-band, multiple
independent registries, and a coverage-challenge protocol so a committee can prove
completeness rather than assert it.

**Bootnode as a trusted placement service.** The bootnode assigns committees and
want-sets and, until a committee is complete, backstops coverage as origin holder,
so it is trusted for placement and is in the availability path pre-completeness. It
is not in the token-loop inference path, and joined nodes survive its death. A
malicious bootnode can strand or bias committees; v1 assumes it is honest and
operator-run. Verifiable/redundant assignment is future work.

**False-holdings failure mode.** Holdings bitmaps (gossip announces, heartbeat
digests) are unverified claims with no proof of possession. A lying holder is
detected only reactively: a `GETR`/ExpertRequest returns `not_held` or wrong
bytes, and the requester falls through to the next peer (digest verification
blocks the wrong bytes; availability degrades, integrity does not). There is no
reputation, periodic challenge, or probe-on-suspicion yet, so committee
completeness is a construction-time property under honest participation, not a
cryptographic guarantee. Adding a sampled-`GETR` challenge plus holder
reputation is the planned mitigation.

## Resource governance and failure modes (v1 status)

Several safety limits are specified here but only partially enforced in v1. They
are called out so implementations and readers do not assume protection that is
absent.

**Serving SLA under incomplete committees.** At R = 1 a single member death opens
a coverage hole, and heartbeat detection lags up to one interval (5 s). During a
gap, a token-loop fetch for the missing shard first tries committee holders, then
falls through to the mesh; if no reachable holder exists anywhere the token-loop
fetch errors, so inference fails with an error rather than producing wrong
output. "Can still serve via the mesh" and "the committee invariant holds" are
distinct states: a node may serve while its committee is incomplete, and the
eager repair loop (2 s) restores the invariant eventually rather than
synchronously with the death.

**Weight integrity, end to end.** The version id now binds the shard layout
(extents, file_size, n_resident, mode) into the Merkle root, and parse rejects any
manifest that is not an exact partition of `[0, file_size)`, so a peer can no
longer serve digests that verify while pointing at attacker-chosen offsets. The
hot path re-verifies: `readRangeVerified` re-hashes on local read and on
`GETR`/`ExpertRequest` serve (clearing the bit and triggering repair on mismatch);
store-open re-audits every held shard; the distributed run fails closed unless all
resident shards are held and verified; the loom-format ExpertCache publishes a slot
only after a successful verify. All implemented and tested (a corrupted on-disk
shard is caught on open, cleared, and re-fetched).

**Disk cap for organic replicas.** Fetched shards are persisted (organic heat
replication), which without a bound drives busy nodes toward the full corpus. Still
a v1 gap (not yet enforced): a `max_shard_bytes` cap above which
opportunistically-fetched shards are LRU-evicted while assigned want-set shards are
pinned. Organic replicas are cache and do not count toward a committee's R target.
Until the cap ships, operators must size disk for worst-case growth on hot nodes.

**Admission control and priority.** ExpertResponse carries a `busy` status, but v1
has no rate limiting, per-peer quota, or bandwidth reservation. The intended
priority order (v1 gap) is token-loop fetch above eager repair above bulk sync,
with per-peer request quotas, so a light node or attacker cannot soak
expert-serving capacity and starve inference. Not implemented; documented as
required.

**Frame and announce bounds.** A frame body is read in bounded chunks, so
resident memory tracks bytes actually delivered rather than the length a peer
claims; the client-side reader now enforces the same 64 MiB ceiling as the
server (it previously allowed 512 MiB in up to 64 concurrent prefetch threads).
An announce whose holdings bitmap is not exactly one bit per shard is rejected
by the peer table, and an oversized frame payload returns an error rather than
overflowing the u32 length field.

**Connection deadlines.** Every accepted connection and every outbound peer
dial carries a deadline enforced by a watchdog that shuts the socket down when
it expires (30 s serve, 10 s peer). This closes the gap where the connection
semaphore below was the only bound: without a deadline, idle connections held
every handler slot permanently. Two platform limits are documented in
`src/core/sockopt.zig`: `SO_RCVTIMEO` is unusable (the Threaded Io backend
panics on `EAGAIN`), and `ConnectOptions.timeout` is unimplemented, so the TCP
handshake itself still falls back to the OS timeout. The eager-repair loop no
longer holds the engine mutex across peer I/O, so a silent peer cannot stall
inference.

**Peer-table bounds and mesh authenticity.** The peer table is now bounded
(`MAX_PEERS`, coldest-`last_seen` eviction on overflow) and holdings updates carry
a monotonic sequence so a stale or replayed bitmap cannot clobber a fresh one
(implemented). Frame decode is snappy-capped (`decodeWithMax`, `MAX_BODY_BYTES`)
against decompression bombs, and connection handlers are semaphore-bounded against
thread-per-accept floods (implemented). Still a v1 gap: authenticated announces
(fake-holder mesh poisoning is possible on an untrusted network) and exponential
backoff or probation on repeatedly-dead peers. v1 runs LAN-scale and
operator-trusted.

**Transport security.** The RPC and P2P transports are plaintext TCP with no TLS
and no authentication. This is acceptable for a lab/LAN operator-run swarm but
not for a public service economy; TLS plus peer authentication is a prerequisite
for any untrusted deployment.

**Failure-mode catalog (v1 behavior):**

| Failure | v1 behavior |
|---|---|
| Bootnode down (post-join) | No effect on joined nodes; gossip mesh plus repair continue; new joins blocked until it returns or another registry exists |
| Network partition | Each side serves from its own holdings plus reachable mesh; repair reconverges on heal |
| Peer lies about holdings | Caught on fetch (`not_held`/wrong bytes, digest reject, next peer); availability dips, integrity preserved |
| Disk full | Writes fail; the shard stays unheld and is retried; no cap/eviction yet (see above) |
| Version skew mid-fetch | ExpertRequest pins version; a re-versioned peer returns `version_mismatch` and the requester falls through |
| Committee hole (member died) | Token fetch falls to mesh; errors only if no holder anywhere; repair restores R eventually |
| RPC/expert flooding | No admission control in v1; capacity can be soaked (documented gap) |

## Architecture diagrams

### Full node

```
                                +-----------------------------------------------------+
                                |                      FULL NODE                      |
                                |                                                     |
   apps / light nodes ------->  |  +---------------+      +----------------------+    |
   {"prompt",...,"client":ID}   |  |  RPC server   |----->|   Metering ledger    |    |
   <-- {..,"cost","balance"} -- |  |  (JSON, TCP)  | gate | quota+credits-usage  |    |
                                |  +------+--------+      | payment_required     |    |
                                |         | generate      | credit / tab ops     |    |
                                |         v               +----------------------+    |
                                |  +--------------------------------+                 |
                                |  |        Inference engine        |                 |
                                |  |  dense path (mmap resident):   |                 |
                                |  |  MLA attn / router / shared    |                 |
                                |  |  KV cache (compressed, local)  |                 |
                                |  +-------+------------------------+                 |
                                |          | routed-expert reads (per MoE layer)      |
                                |          v                                          |
                                |  +--------------------------------+                 |
                                |  |  Expert access (tiered)        |                 |
                                |  |  1. held shard -> pread        |                 |
                                |  |     (page cache = RAM tier)    |                 |
                                |  |  2. miss -> parallel prefetch -+----------+      |
                                |  +-------+------------------------+          |      |
                                |          | verified writes                   |      |
                                |          v                                   v      |
                                |  +--------------------------------+   +-----------+ |
                                |  |  Weight store (sparse GGUF)    |   |  Expert   | |
                                |  |  manifest (digests+extents,    |   |  fetch    | |
                                |  |  Merkle version id)            |   | committee | |
                                |  |  holdings / wanted bitmaps     |   | first,    | |
                                |  +---------------^----------------+   | then mesh | |
                                |                  | serve / repair     +-----+-----+ |
                                |  +---------------+----------------+          |      |
   peers ---------------------> |  |  P2P server (TCP, frames+text) |          |      |
   GETR/ExpertRequest, GOSSIP,  |  |  shard serving / gossip /      |          |      |
   Heartbeat, JOIN (bootnode)   |  |  heartbeat answers / JOIN      |          |      |
                                |  +---------------+----------------+          |      |
                                |                  |                           |      |
                                |  +---------------v----------------+          |      |
                                |  |  Peer table (mesh)             |<---------+      |
                                |  |  addr/version/holdings/        |  holder lookup  |
                                |  |  committee id                  |                 |
                                |  +------+---------+--------+-------+                 |
                                |         |         |        |                        |
                                |   +-----+---+ +---+----+ +-+--------+               |
                                |   | gossip  | | heart- | |  eager   |               |
                                |   | loop 3s | | beat 5s| | repair 2s|               |
                                |   +---------+ +--------+ +----------+               |
                                +-----------------------------------------------------+
```

### Light node

```
                       +-------------------------------------------+
                       |                LIGHT NODE                 |
                       |   (no weights / no store / no engine)     |
                       |                                           |
  local apps ------->  |  +-----------------+   +---------------+  |
  {"prompt",...}       |  | Local RPC       |-->|  Delegator    |  |
  <-- full response -- |  | (same protocol) |   | stamp client  |  |
      + cost/balance   |  +-----------------+   | id / round-   |  |
                       |                        | robin +       |  |
                       |  +-----------------+   | failover      |  |
                       |  |  Spend tally    |<--+               |  |
                       |  | (from "cost")   |   +-------+-------+  |
                       |  +-----------------+           |          |
                       +--------------------------------+----------+
                                                        | metered JSON RPC
                                     +------------------+------------------+
                                     v                  v                  v
                               +-----------+      +-----------+      +-----------+
                               | FULL NODE |      | FULL NODE |      | FULL NODE |
                               | (ledger A)|      | (ledger B)|      | (ledger C)|
                               +-----------+      +-----------+      +-----------+
                                 per-provider ledgers: each full node meters
                                 the service it renders
```

### Swarm topology

```
                       +------------+  JOIN gives committee id,
        new node ----> |  BOOTNODE  |  members, assigned want-set
                       | (registry, |  (least-covered-first)
                       |  full GGUF)|  never in the inference path
                       +-----+------+
                             | manifest + initial shards
        +--------------------+---------------------+
        v                    v                     v
  +-----------+  heartbeat +-----------+         +-----------+
  | FULL NODE |<--- 5s --->| FULL NODE |   ...   | FULL NODE |   committee 0
  |  shards   |  shard     |  shards   |         |  shards   |   (complete: every
  |  A..M     |  fetch     |  N..Z     |         |  A..K     |    shard has >=1
  +-----^-----+            +-----^-----+         +-----^-----+    holder)
        |                        |                     |
        +------------------------+---------------------+
                                 | global gossip mesh (3s announces:
                                 | addr / version / committee / holdings bitmap)
        +------------------------+---------------------+
        v                        v                     v
  +-----------+            +-----------+         +-----------+
  | FULL NODE |            | FULL NODE |   ...   | light     |   committee 1 +
  |           |            |           |         | nodes     |   light clients
  +-----------+            +-----------+         +-----------+
```

## Shard committees

A **committee** is a group of nodes that collectively holds the complete shard
set: a self-sufficient serving cell for one model version.

- **Completeness invariant:** each committee must have at least one node holding
  every shard of the GGUF set. Until a forming committee reaches completeness, the
  mesh (and the bootnode, which holds everything) covers the gaps.
- **Intra-committee redundancy:** more than one node in a committee may hold the
  same shard. The bootnode fills a committee toward a redundancy target `R`
  (**R = 3 target, R = 2 floor; `--r-target` default 2**): joiners are assigned
  the currently least-covered shards. Completeness (every shard held by at least 1
  member, R >= 1) and redundancy (R >= 2) are distinct thresholds. Only assigned
  holdings count toward R; opportunistic organic replicas (see Resource
  governance) are cache, not coverage.
- **Committee lifecycle:** joiners go to the first committee whose minimum shard
  coverage is below `R`. When every shard in every existing committee has coverage
  at or above `R` (saturated), the next joiner opens a new committee.
- **Heartbeats:** every node maintains a regular heartbeat (wire Heartbeat frames,
  5 s interval) with its committee members. The member set is the join-time seed
  plus the gossip-derived committee view, so membership stays current as the
  committee grows. A member that stops answering is marked dead locally, and its
  shards become candidates for re-replication via the eager repair loop.

## Global gossip network

Independent of committees, every node regularly announces the weights it holds
on the global gossip topic (addr plus manifest version plus holdings bitmap,
epidemic exchange, 3 s interval). Receiving peers merge these announcements into
their peer table, the **mesh**.

The mesh serves two purposes:

1. **Discovery:** transitive, so a node learns holders it was never configured
   with.
2. **Fallback routing:** see query path below.

ENR (planned) carries the compact summary (holdings-bitmap hash plus seq); the
gossip topic carries the full bitmap.

## Query path for a weight / expert

When a node needs a shard it does not hold (boot-time sync, eager repair, or an
expert fetch inside the token loop):

1. **Committee first:** ask the allotted peer(s) within its own committee that
   hold the shard (round-robin across replicas, client-side spreading).
2. **Mesh fallback:** if no committee holder responds, query any peer from the
   gossip-built mesh whose announced holdings cover the shard.
3. Every response is digest-verified against the manifest before use. A shard
   obtained at token time is persisted and announced on the next gossip round
   (fetch-on-demand is organic heat replication).

A request fails only when neither the committee nor the mesh has a reachable
holder; the eager repair loop works to make that state transient.

## Wire messages v1 (binary frames, snappy)

The three structured message families (heartbeat, gossip announce, and the expert
request/response) are binary frames. All integers little-endian. Text ops remain
for debugging; frames ride the same TCP connection behind a `FRAME <len>` header
line.

**Frame envelope** (8-byte header plus body):

```
magic:    "LM"          (2 bytes)
type:     u8            (message type, below)
flags:    u8            bit0 = body is snappy-compressed
body_len: u32           length of the (possibly compressed) body
body:     [body_len]u8
```

Compression policy: the encoder compresses the body with snappy and keeps the
compressed form only if it is smaller (adaptive). In practice holdings bitmaps and
manifests compress well, while quantized expert payloads are high-entropy and
usually ship raw. Decoders must handle both.

**Heartbeat, type 0x01 (request), 0x02 (response).** Committee-internal, every
5 s. Carries liveness plus exactly the state a committee member needs to act on:
version drift detection, holdings freshness, and a load hint for client-side
spreading. One exchange refreshes both sides (the response is the same container).

```
proto:            u8       (= 2)
network_id:       u64      (the LLM-network identity; see below)
committee_id:     u32      (0xFFFFFFFF = not in a committee)
manifest_version: [32]u8   (Merkle root; zeros = no store)
holdings_seq:     u64      (monotonic; bumps on every holdings change)
holdings_digest:  [32]u8   (sha256 of the holdings bitmap)
load:             u16      (in-flight expert requests being served)
sent_at_ns:       i64      (sender clock; RTT/skew measurement)
addr_len:         u8
addr:             [addr_len]u8   (sender's advertised host:port)
```

Rationale: the full bitmap does not ride the heartbeat. `holdings_seq` plus
`holdings_digest` let the receiver detect staleness cheaply; when they change, the
fresh bitmap arrives via the next gossip announce (or a pull). A committee member
is marked dead after missed heartbeats, and its shards become re-replication
candidates.

**Announce, type 0x03.** The global gossip record (every 3 s): the weights a
node holds. Receivers merge this record into their mesh table.

```
proto:            u8
network_id:       u64
committee_id:     u32
manifest_version: [32]u8
holdings_seq:     u64
addr_len:         u8
addr:             [addr_len]u8
bitmap_len:       u32
holdings_bitmap:  [bitmap_len]u8   (1 bit per shard; frame-level snappy
                                    typically compresses this well)
```

Receivers keep the entry with the highest `holdings_seq` per addr. ENR (planned)
carries `network_id + manifest_version + holdings_seq + holdings_digest` only
(fits the 300-byte limit); the gossip announce carries the full bitmap.

**network_id (proto v2).** One loom network serves one model, and `network_id`
is that network's identity (the analogue of Ethereum's chainId). A node refuses
to peer across networks: an inbound Heartbeat or Announce whose `network_id`
differs is answered `ERR wrong_network` and never merged into the mesh table,
and gossip-learned records from other networks are dropped at merge. The id is
configured with `--network-id N`; the default (`auto`) derives it from the
weight manifest's leading eight bytes, so nodes sharding the same model agree
without configuration, while an explicit id keeps one network together across
model hardforks. `network_id` gates membership; `manifest_version` continues
to gate content (expert fetch refuses other versions within a network).
Reserved ids: 1 = mainnet (GLM 5.2), 2 = testnet (GLM-4.6), 1337 = devnet
(Qwen3-30B-A3B); see docs/NETWORKS.md. These are protocol constants.

**AnnounceBatch, type 0x04.** The gossip exchange response: the responder's own
Announce followed by its whole table, so one round trip both announces and syncs
the mesh view. Frame-level snappy compresses across the similar bitmaps of many
peers.

```
count:   u32
entries: count x { len: u32, body: [len]u8 }   (each an Announce body)
```

Because announces carry `committee_id`, the mesh table doubles as the
**gossip-derived committee view**: committee membership is not static-at-join.
Earlier members learn later joiners from their announces, and the heartbeat loop
targets the seed set plus every table entry carrying my committee_id. Members
rely on this view rather than the bootnode after joining, so a dead bootnode
strands nothing.

**RAG gossip, types 0x20/0x21/0x22 (optional, --rag).** A network may share
retrieval chunks. Only text travels: a loom network serves one model
(network_id), so every node recomputes the same embedding (mean-pooled
`token_embd` rows, L2-normalized) from the same text. Vectors are never
accepted from the wire, which removes the vector-poisoning surface and any
dimension mismatch by construction.

```
RagInv    (0x20): count u16 (<=512), then count x [32]u8 chunk hashes
                  (sha256 of chunk text; a rotating window, newest first)
RagWant   (0x21): same shape (<=32): the hashes the sender lacks
RagPush   (0x22): count u16 (<=32), then count x { len u32 (<=8192),
                  text [len]u8 }
RagInvReq (0x23): empty body -- "send me your inventory window"
```

Both directions converge on one outbound dial (NAT-friendly: a dial-out-only
node still receives): push first (my Inv -> your Want -> my Push), then pull
(my InvReq -> your Inv -> my Want -> your Push). All handlers are stateless.

The exchange piggybacks on the gossip round: Announce/AnnounceBatch first,
then Inv -> Want -> Push on the same stream. Receivers insert by recomputing
the embedding locally; duplicates dedup by text hash. Caps bound a round.
When a store exceeds one round's inventory cap, the advertised window
rotates round to round, so every hash is offered within ceil(count/cap)
rounds; the exchange with every known peer plus transitive re-gossip gives
global-topic semantics with eventual convergence, including for peers that
rejoin after long absence. Compression: the wire applies the frame encoder's
adaptive snappy (kept only when smaller) to every RAG frame; at-rest chunk
text may additionally be brotli-compressed locally (dlopen, optional), a
per-node choice invisible to the protocol, since hashes are of raw text and
Push always carries raw text.

**ExpertRequest, type 0x10.** Ask a remote peer (committee member first, then
mesh) for one expert shard the requester does not hold.

```
proto:            u8
request_id:       u64      (echoed in the response; enables pipelining)
manifest_version: [32]u8   (hard guard: server refuses other versions)
shard_id:         u32      (manifest shard index; the expert block)
```

**ExpertResponse, type 0x11.**

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
`shard_id` before use: the response carries no digest by design, because the
manifest is the only trust root. `busy` tells the client to spread to another
replica (it pairs with the heartbeat `load` hint).

## Wire protocol (legacy / debug line form)

> **Normative transport is the binary frame protocol above.** This line-delimited
> form is transitional: it still serves `JOIN`, `GETR`, `GOSSIP`/`TABLE`, `PING`,
> and the `MANIFEST*` ops today (migration in progress), and is convenient for
> `nc`-driven debugging. New message types are defined as frames; do not add line
> ops.

| Op | Response | Purpose |
|---|---|---|
| `JOIN addr=<h:p> fraction=<f>` | `COMMITTEE id=<n> members=<a,b,...> assign=<hex bitmap>` | bootnode assigns committee plus want-set (capacity ~ f x expert shards) |
| `COMMITTEES` | summary lines | bootnode debug: per-committee members, min/max coverage, complete/saturated |
| `MANIFESTFILE` | serialized manifest | digests plus extent lists, root-verified by the client |
| `GETR <i>` | shard bytes | fetch one shard (committee or mesh) |
| `GOSSIP addr=.. version=.. holdings=..` | peer table | the global announce plus mesh exchange |
| `PING` | `PONG` | heartbeat |

The same ops move to gossipsub/discv5 transports later; the semantics above are
the spec.

## Sizing (GLM 5.2 target)

Derivation and the full deployment plan live in
[../docs/ROADMAP.md](../docs/ROADMAP.md).

19,200 expert shards (~19 MB each) plus a resident bundle. Committee sizing at
50 GB/node: completeness (R >= 1) needs at least 8 nodes (~2,600 expert shards
each), R = 2 needs at least 15, R = 3 needs at least 22. Holdings bitmap
2.4 KB; manifest ~1 MB.

The resident bundle is the floor, and the 16 GB minimum is conditional on it.
Every node holds it in full, in RAM (mmap'd). DeepSeek/GLM shared experts and
embeddings dominate it, and MLA KV grows with context. A per-tensor-class
breakdown for the target GGUF (embeddings, attention projections including
q/kv-LoRA, shared-expert FFNs, router, norms) and a KV-bytes-vs-context table
must be published from the real converted GLM 5.2 GGUF before the 16 GB row in
the deployment table is asserted as a default rather than a conditional
target. For DeepSeek-V2-Lite the measured resident bundle is 0.78 GB (73 chunks);
the GLM 5.2 ~10 GB figure is an int4 estimate pending the real file.
