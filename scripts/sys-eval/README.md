# sys-eval — the loom system paper's measurement batteries

Turnkey protocol scripts for the evaluation of the loom system paper
(MLSys-class target). Everything here encodes the snapshot-controlled A/B
protocol from the pre-gating paper's batteries (whitepaper decision log,
2026-08-12/13 rows), now runnable at real low holdings after the issue #252
fix (v0.41.1+).

## Boxes

- **Bootnode / origin**: already live (the devnet bootnode). Scripts never
  touch it beyond normal p2p traffic.
- **Measurement box(es)**: Hetzner CPX31-class or better, x86-64-v3 (AVX2),
  >= 8 GB RAM, >= 25 GB free disk per joiner (store at 0.5 hold ≈ 6.5 GB,
  plus snapshot copy). For the E1 capacity run (Qwen3-235B, 86 GB total),
  4-6 boxes with >= 40 GB disk each.

Every script is self-contained: copy it to the box with scp FIRST, then run
it there over ssh. Do NOT paste scripts through remote heredocs and do not
scp from inside one (the 2026-08-12 lesson: an scp inside a remote heredoc
runs on the remote and fails silently).

## Run order (one measurement box)

```sh
scp scripts/sys-eval/*.sh root@BOX:
ssh root@BOX 'bash 00-setup-box.sh'                 # install loom + pregate head
ssh root@BOX 'bash 10-join-sync.sh 0.2'             # join devnet at 20% hold, snapshot
ssh root@BOX 'bash 20-ab-battery.sh 3'              # 3 A/B pairs -> ab-results.csv
ssh root@BOX 'bash 10-join-sync.sh 0.5 && bash 20-ab-battery.sh 3'
scp root@BOX:sys-eval-out/ab-results.csv results-BOX/
```

The pair (hold 0.2, hold 0.5) at the measurement box's natural link speed is
the paper's missing bandwidth-curve point. Record the link speed with the
probe printed during sync (or `curl -o /dev/null` against the bootnode).

## Batteries and what they feed

| script | experiment | paper section |
|---|---|---|
| `10-join-sync.sh` + `20-ab-battery.sh` | low-holdings pre-gate A/B | Eval: bandwidth curve |
| `30-churn-battery.sh` | kill a holder mid-generation, time recovery; striped-repair engagement | Eval: fault tolerance |
| `40-baseline-llamacpp.sh` | llama.cpp single-node paging + RPC split | Eval: baselines |
| E1 capacity (manual, see below) | 235B pooled across N boxes, no box holds it all | Eval: the capacity headline |

## E1 — the capacity demonstration (recipe, not yet scripted)

Goal: serve Qwen3-235B-A22B Q2_K (~86 GB) from pooled RAM/disk of N boxes
where NO single box could hold it. Steps: bootnode gets the GGUF (fits the
150 GB disk after the GLM deletion); each of N=4..6 joiners joins with
`--hold-fraction` ≈ 1.2/N (overlap for R=2); a client (loom light or RPC)
drives generation; record tok/s, per-box RSS/disk, and that every box's
holdings < model size. Script lands once the join flow is rehearsed on the
30B model via `10-join-sync.sh`.

## Churn battery notes

`30-churn-battery.sh` runs on an OBSERVER box and needs at least two other
holders (bootnode + one joiner) so that killing one strands nothing. It
kills the joiner's loom process by exact binary path (never pkill by
pattern — the self-match trap), times the observer's in-flight generation,
and greps the observer's log for repair and striped-repair lines.

## Baseline notes

`40-baseline-llamacpp.sh` measures (a) single-node llama.cpp with the model
paging from disk — the "one box, no network" floor — and (b) llama.cpp's
RPC mode with layers split across two boxes, the pipeline-parallel
alternative the whitepaper argues against. Same GGUF, same greedy 96-token
prompt. llama.cpp is built from source pinned to a tag for reproducibility.

## Outputs

Every battery appends to `sys-eval-out/*.csv` with a `run_id` and wall-clock
seconds; `20-ab-battery.sh` also records tok/s and the store-restore hash so
a warmed store can never masquerade as a cold run. Keep the raw CSVs — they
are the paper's artifact.
