# Alpha testing guide

Thanks for trying loom. This page tells you how to join the devnet, what is
worth exercising, what telemetry the node reports (and how to turn it off),
and how to file useful reports.

## Join

One command. It installs loom if missing, joins the devnet (Qwen3-30B-A3B,
30B), downloads your share of the model from peers, and serves a local
OpenAI-compatible API plus a chat UI:

```sh
curl -fsSL https://raw.githubusercontent.com/ch4r10t33r/loom/main/scripts/join-devnet.sh | sh
```

You need about 4 GB of free disk and 8 GB of RAM. The initial sync is
roughly 3 GB (the ~1 GB resident bundle every node holds, plus your expert
share). Ports 8770-8772 and 8555 are used locally; nothing needs to be
reachable from the internet (NAT is fine).

To leave: stop the process and delete `~/.cache/loom`.

## What to expect, honestly

- Sync speed is your downlink. The devnet's origin serves from Germany.
- Generation speed depends on how much of the model you hold and how far
  you are from other holders. At the default hold fraction (0.7), most
  tokens run from local shards; the remainder are fetched mid-generation
  from peers. On a fast link near peers a miss costs seconds; far away on
  a thin link it is minutes. That boundary is by design and documented in
  the [whitepaper](../whitepaper/WHITEPAPER.md); loom distributes for
  capacity first. More holders make everyone faster.
- `LOOM_HOLD` trades disk for speed: 1.0 (~11 GB) generates fully locally;
  below ~0.5 expect origin-bound tokens while the devnet has few holders.
  The default is 0.7 so each joiner meaningfully raises network replication
  instead of leaning on the origin.
- The chat UI is at `http://127.0.0.1:8555`.

## Things worth trying

1. Plain chat through the UI or `curl` against `http://127.0.0.1:8772/v1`.
2. Ingest a fact into the shared RAG store and see whether another tester's
   node answers with it:
   `curl -X POST 127.0.0.1:8772/v1/rag/chunks -d '{"text":"..."}'`
3. Kill your node mid-generation and restart it; it should resync only what
   changed.
4. Run two nodes in one house and watch the second sync from the first.
5. Anything that breaks. Especially that.

## Telemetry (opt-in, numeric only)

The join script enables a once-a-minute report to the devnet bootnode so we
can see what works and what doesn't across the fleet. Disable it with
`LOOM_NO_METRICS=1`, or by omitting `--report-metrics` when running
`loom node` directly.

The report is one JSON line containing exactly these fields and nothing
else — never prompts, never generated text, never RAG content, never your
address or hostname:

| field | meaning |
|---|---|
| `id` | random id generated at each boot (no stable identity) |
| `v`, `os`, `arch` | loom version, OS tag, CPU architecture |
| `up_s` | seconds since the node started |
| `net` | network id (1337 on devnet) |
| `hold_target`, `held`, `total` | your configured hold fraction and actual shard holdings |
| `peers` | size of your peer table |
| `gens`, `tok_s_avg`, `hit_avg` | generations served, average tokens/sec, average expert-cache hit rate |
| `draft_rounds`, `draft_drafted`, `draft_accepted` | DSD draft-verify: verification round trips, tokens drafted locally, drafts the verifying peer accepted |
| `draft_gamma`, `draft_bails` | DSD: last speculation-window size, and how many requests fell back to wholesale delegation |

The wire message is `METRICS <json>` to the bootstrap peer; a node that was
not started with `--alpha-ingest` refuses it, so reports only ever land on
the operator's collector. Adding any field to this report requires updating
this table in the same change.

## Filing reports

Use the issue templates:
[bug report](https://github.com/ch4r10t33r/loom/issues/new?template=bug_report.yml)
for anything broken,
[alpha feedback](https://github.com/ch4r10t33r/loom/issues/new?template=alpha_feedback.yml)
for experience notes and numbers. `loom version` output plus the last ~20
lines of node output make almost any report actionable.
