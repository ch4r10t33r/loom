# Walkthrough: sharding a real MoE model across multiple nodes

End-to-end on real weights: download a model, shard it by expert, run a swarm,
and watch one node serve inference using experts it does not hold.

Everything here was run and verified. Where a number is quoted, it came from an
actual run.

## Which models work

Two hard constraints, and both bite:

**1. Architecture must be `deepseek2`.** That is the only engine wired into the
distributed serving path. Loom also runs `llama`, but only single-node — a
llama model will shard and distribute perfectly, yet a partial node cannot
*serve* it (it silently falls back to the loom-format `--model` engine).

This rules out most MoE GGUFs you will find:

| Model | GGUF arch | Works? |
|---|---|---|
| DeepSeek-V2 / V2-Lite / Coder-V2-Lite | `deepseek2` | **yes** |
| Mixtral 8x7B | `llama` | shards, but no distributed serving |
| Qwen1.5/3 MoE | `qwen2moe` / `qwen3moe` | not supported |
| GLM-4.5 MoE | `glm4moe` | not supported |

**2. Quantization must be one Loom implements:** `F32`, `F16`, `Q4_0`, `Q5_0`,
`Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`. **IQ-quants are not supported** (`IQ2_M`,
`IQ3_XXS`, …), and most modern GGUF repos publish mostly IQ variants — check
before downloading.

### Recommended model

**DeepSeek-Coder-V2-Lite-Instruct**, Q4_K_S, **9.53 GB** — the smallest
Loom-compatible real MoE:

```
repo: bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF
file: DeepSeek-Coder-V2-Lite-Instruct-Q4_K_S.gguf
```

Verified from its GGUF header: `general.architecture = deepseek2`,
`block_count = 27`, `expert_count = 64`, `expert_used_count = 6`,
`expert_shared_count = 2`. Roughly 1,600+ expert shards.

Larger compatible quants from the same repo, if you want them: Q4_K_M (10.4 GB),
Q5_K_M (11.9 GB), Q6_K (14.1 GB), Q8_0 (16.7 GB).

### What you need on disk

- **Origin:** the model file itself. It is opened **in place** and never
  copied; its store directory holds only sidecars (~KB).
- **Each joining node:** a **sparse** copy sized to the full file, where only
  held shards occupy real blocks. Measured on a 19 MB model at
  `--hold-fraction 0.4`: 19,077,344 bytes logical, **6,496,256 bytes actually on
  disk**. So a node at 30% of a 9.5 GB model uses roughly 3 GB, not 9.5 GB.

RAM: `--ram-gb 4` is plenty for this model. The resident bundle (attention,
shared experts, embeddings) is mmap'd and every node holds it in full.

---

## Step 0 — build

```sh
zig build -Doptimize=ReleaseFast
```

Use ReleaseFast. The debug build is roughly 10x slower and this model is large
enough that you will notice.

> If you edit code between steps, re-run `zig build`. `zig build test` builds
> only the test binary and leaves `zig-out/bin/loom` stale — that cost me a
> confusing debugging session.

## Step 1 — download the model

```sh
curl -L -o /tmp/dsc-v2-lite.gguf \
  https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_S.gguf
```

## Step 2 — confirm Loom can read it

```sh
./zig-out/bin/loom gguf info /tmp/dsc-v2-lite.gguf | head -20
```

Check `general.architecture = "deepseek2"`. If it says anything else, stop —
the rest of this walkthrough will not work.

## Step 3 — look at the expert-aligned shard plan

```sh
./zig-out/bin/loom gguf shard /tmp/dsc-v2-lite.gguf
```

This prints the manifest without writing anything: one shard per
(layer, expert), plus the resident bundle chunked at 16 MB, each with a SHA-256,
and the Merkle root that becomes the **model version id**.

If you see `no 3D expert tensors found — not a MoE GGUF`, the model is dense and
only fixed-size ranges apply.

## Step 4 — sanity-check the model single-node

Before involving the network, prove the model runs at all:

```sh
./zig-out/bin/loom gguf run /tmp/dsc-v2-lite.gguf \
  --prompt "def fibonacci(n):" --max-tokens 40
```

Expect coherent code. This is also your reference output for step 8.

---

## Step 5 — start the origin (also the bootnode)

The origin holds every shard and assigns committees to joiners.

```sh
./zig-out/bin/loom node \
  --gguf /tmp/dsc-v2-lite.gguf \
  --rpc-port 8770 --openai-port 8772 \
  --p2p-port 8771 --advertise 127.0.0.1:8771 \
  --ram-gb 4
```

In the banner, confirm:

```
bootnode   committee registry active (R target 2)
shards     mode=expert total=… (resident=…, expert=…) held=… (100.0%)
serving    distributed GGUF (deepseek2): … chat=deepseek
```

`mode=expert` is the one to check — that is expert-aligned sharding. If it says
`mode=fixed`, the model is not MoE.

## Step 6 — start a partial node

**In a second terminal.** Three things must differ from the origin: the ports,
the advertised address, and `HOME` (which selects the store directory, so each
node gets its own).

```sh
mkdir -p /tmp/node2
HOME=/tmp/node2 ./zig-out/bin/loom node \
  --bootstrap 127.0.0.1:8771 \
  --hold-fraction 0.3 \
  --rpc-port 8780 --openai-port 8782 \
  --p2p-port 8781 --advertise 127.0.0.1:8781 \
  --ram-gb 4
```

It will join a committee, receive a shard assignment, and sync ~30% of the
experts (plus the resident bundle, which is always held in full). Expect:

```
joined committee 0 (…)
  synced N/N assigned shards, … MB, verified against manifest root
shards     mode=expert total=… held=… (…%)
serving    distributed GGUF (deepseek2): …
```

`--hold-fraction` picks an **exact** count (`round(fraction x expert_shards)`);
only *which* shards is random, seeded by `--seed` so a restart re-picks the same
set.

## Step 7 — prove it serves using experts it does not hold

Query **node 2** (port 8780), the one holding only 30%:

```sh
printf '{"prompt":"def fibonacci(n):","max_tokens":30,"seed":1}\n' | nc -w 120 127.0.0.1 8780
```

The response carries `hit_rate`. **A value below 1.0 is the whole point** — it
means some experts were not held locally and were fetched from the origin
*inside the token loop*. On a 33% store of DeepSeek-V2-Lite an earlier run
measured `hit_rate` around 0.90, streaming 641 experts (3.5 GB) from one peer
with zero failures.

The first request on a cold node is slow: every miss is a synchronous fetch.
It speeds up as fetched shards are persisted — fetch-on-demand doubles as
organic replication, so `held=` grows as you use it.

Same thing over the OpenAI surface:

```sh
curl -s http://127.0.0.1:8782/v1/completions \
  -d '{"prompt":"def fibonacci(n):","max_tokens":30}'
```

## Step 8 — verify correctness, not just liveness

Same prompt and seed against the full-copy origin (8770) and the partial node
(8780):

```sh
printf '{"prompt":"def fibonacci(n):","max_tokens":30,"seed":1}\n' | nc -w 120 127.0.0.1 8770
printf '{"prompt":"def fibonacci(n):","max_tokens":30,"seed":1}\n' | nc -w 120 127.0.0.1 8780
```

The `text` fields must be **identical**. A partial node streaming experts must
produce exactly what a full copy produces — anything else means the fetch path
is corrupting weights. (Every fetched shard is digest-verified against the
manifest before it touches disk, so this should hold by construction.)

## Step 9 — add a third node and watch the committee fill

```sh
mkdir -p /tmp/node3
HOME=/tmp/node3 ./zig-out/bin/loom node \
  --bootstrap 127.0.0.1:8771 --hold-fraction 0.3 \
  --rpc-port 8790 --openai-port 8792 \
  --p2p-port 8791 --advertise 127.0.0.1:8791 \
  --ram-gb 4
```

The bootnode assigns it the **least-covered** shards first, so coverage is
achieved by construction rather than by luck. Watch for gossip and heartbeat
lines on the existing nodes as they discover it. Peer tables:

```sh
printf 'TABLE\n'       | nc -w 5 127.0.0.1 8771
printf 'COMMITTEES\n'  | nc -w 5 127.0.0.1 8771
```

## Step 10 — kill a node and watch repair

Stop node 3 (Ctrl-C). Within one heartbeat interval (5 s) the survivors mark it
dead and adopt its advertised shards into their want-set; the eager repair loop
(2 s) then re-fetches them from another holder. Watch node 2:

```
heartbeat: committee member 127.0.0.1:8791 DEAD
heartbeat: adopted N shard(s) from dead 127.0.0.1:8791 into wanted
repair: recovered N ranges, held …
```

---

## Verifying integrity directly

Two checks that do not involve inference at all.

**Reassembly is byte-exact.** Give a node `--hold-fraction 1.0`, let it sync,
then compare its store copy with the original:

```sh
shasum -a 256 /tmp/dsc-v2-lite.gguf
shasum -a 256 /tmp/node2/.cache/loom/models/gguf-synced/model.gguf
```

Verified on the 19 MB llama model: identical size and identical SHA-256.

**Corruption is caught and healed.** Corrupt a held shard, then restart the
node:

```sh
python3 -c "f=open('/tmp/node2/.cache/loom/models/gguf-synced/model.gguf','r+b'); f.seek(9000000); f.write(b'\xde\xad\xbe\xef'*16); f.close()"
```

On open, the store re-audits every held shard, clears the bit for the one that
fails its digest, and repair re-fetches it. Verified: the file returns to
byte-identical with the original.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `p2p server failed: AddressInUse` then exit(1) | Another node already has that p2p port. Every node needs its own `--p2p-port`. This is fatal on purpose — a node that cannot serve peers should not pretend to run. |
| Nodes never see each other | `--advertise` wrong. It is what peers dial you on; the default only works on one box. |
| Both nodes share one store | `HOME` not set per node. The store lives under `$HOME/.cache/loom/models`. |
| `held=0 (0.0%)` | Older builds had a `--hold-fraction` bug that could select nothing on small shard counts. Fixed — pull latest. |
| `mode=fixed`, not `expert` | The model is dense, not MoE. |
| `gguf serve disabled: …` in the banner | The store is not expert-sharded, or its resident bundle is incomplete. The node falls back to the loom-format engine and serves that instead. |
| First request takes minutes | Expected on a cold partial node: every miss is a synchronous peer fetch. It warms as shards persist. |
| `UnsupportedTensorType` | An IQ-quant. Use Q4_K/Q5_K/Q6_K/Q8_0. |

## Doing it with containers instead

[`docker-compose.yml`](../docker-compose.yml) runs the same two-node topology
with a generated synthetic model, no download:

```sh
docker compose up --build
curl -s localhost:8782/v1/completions -d '{"prompt":"the","max_tokens":8}'
```

To use a real model there, mount it and point `--gguf` at the mounted path.

## See also

- [docs/CLI.md](CLI.md) — every flag in detail
- [spec/SPEC.md](../spec/SPEC.md) — committees, wire protocol, trust model
- [README](../README.md) — overview
