# Running Loom across two physical machines

The single-host walkthrough ([MULTI-NODE-WALKTHROUGH.md](MULTI-NODE-WALKTHROUGH.md))
proves the mechanics. Two real machines prove the thing that actually matters:
whether fetching experts across *your* network is viable, on *your* fabric.

> **What is verified here and what is not.** The model facts below were read
> out of the published GGUF headers over HTTP range requests, and the shard
> arithmetic is computed from them. The Loom behaviour was verified on
> multi-node runs on one host. The cross-machine numbers are predictions from
> measured bandwidth, not results — producing them is the point of the
> exercise.

## Step 0 — measure the link before you download anything

This is the whitepaper's central caveat and it decides whether the rest is
worth doing. Do it first; it takes a minute and can save an 18 GB download.

On machine A:

```bash
iperf3 -s
```

On machine B:

```bash
iperf3 -c <A-LAN-IP> -t 10
```

No `iperf3`? A crude but adequate substitute, on B:

```bash
ssh <A-LAN-IP> 'dd if=/dev/zero bs=1M count=2000' | dd of=/dev/null bs=1M
```

Take the number and read it off this table. `f` is the fraction of experts the
partial node holds; only the misses cross the wire.

| measured | 50% held | 60% held | 75% held | verdict |
|---|---|---|---|---|
| ~1 Gb/s | ~4.6 s/tok | ~3.7 s/tok | ~2.3 s/tok | works, slow — a capacity demo, not a speed one |
| ~2.5 Gb/s | ~1.8 s/tok | ~1.5 s/tok | ~0.9 s/tok | usable |
| ~10 Gb/s | ~0.46 s/tok | ~0.37 s/tok | ~0.23 s/tok | genuinely good |

Gigabit is fine for *this* test — you are measuring the fetch path, not
chasing tokens per second. Just expect seconds per token and size your
patience accordingly. If you want it faster, raise `--hold-fraction`: it moves
the miss rate directly.

## The model

**Qwen3-30B-A3B**, `Q4_K_M`, **18.56 GB**.

```
repo: Qwen/Qwen3-30B-A3B-GGUF
file: Qwen3-30B-A3B-Q4_K_M.gguf
```

Verified from its GGUF header, not assumed: `general.architecture = qwen3moe`,
48 blocks, **128 experts, 8 used**, no shared expert, gpt2-style BPE tokenizer,
a chat template, and tensor types `Q4_K` / `Q6_K` / `F32` — all of which Loom
reads.

Why this one for a two-machine test:

- **6,144 expert shards** (48 layers x 128 experts, ~2.65 MB each). Fine-grained
  placement, so a hold fraction means what it says instead of quantizing into a
  handful of coarse blocks.
- **The resident bundle is only ~2.3 GB.** Every node must hold that in full,
  so it sets the floor on what a machine needs. 2.3 GB is a low floor for a
  30B model.
- **~1.02 GB of expert reads per token** (384 fetches). Small enough that even
  a gigabit link produces a working, if slow, system.
- It is a current, genuinely useful model rather than a test article.

### Alternatives

| Model | File | Size | Arch | Why |
|---|---|---|---|---|
| Qwen3-30B-A3B `IQ4_XS` | `unsloth/Qwen3-30B-A3B-GGUF` | 16.38 GB | `qwen3moe` | 2 GB smaller, and exercises the codebook-quant decoders |
| DeepSeek-Coder-V2-Lite `Q4_K_S` | `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF` | 9.53 GB | `deepseek2` | Half the download; the MLA engine instead of GQA |
| Mixtral 8x7B `Q4_K_M` | `TheBloke`/others | ~26 GB | `llama` | Only 256 shards of ~99 MB, and **~6.3 GB per token** — a poor fit for anything under 10 GbE |

Avoid `Q2_K`, `Q3_K`, `IQ2_M`, `IQ3_M` and similar mixes: llama.cpp's quant
*mixes* combine several tensor types, and those particular ones fold in `Q2_K`
or `Q3_K` tensors that Loom does not implement. Step 2 below checks this
directly rather than guessing from the filename.

### What each machine needs

|  | Machine A (origin) | Machine B (partial) |
|---|---|---|
| Download from HuggingFace | the 18.56 GB file | **nothing** |
| Disk | 18.56 GB (opened in place, never copied) | resident 2.3 GB + `f` x 16.3 GB, in a sparse file |
| Disk at `--hold-fraction 0.5` | 18.6 GB | ~10.5 GB |
| RAM | `--ram-gb` sets the cache budget; 8 GB is comfortable | same |

**Machine B never downloads the model.** It syncs the resident bundle plus its
assigned shards from A over your LAN, digest-verified against the manifest
root. That is the interesting part of the boot path, and on a gigabit link
expect a few minutes for ~10 GB.

---

## Step 1 — build on both machines

```bash
git clone https://github.com/ch4r10t33r/loom && cd loom
zig build -Doptimize=ReleaseSafe
```

An optimized build matters here; the debug build is roughly 10x slower.

Note both LAN addresses now — every step below needs them:

```bash
ip -4 addr show | grep inet    # Linux
ipconfig getifaddr en0         # macOS
```

This guide writes them as `A_IP` and `B_IP`.

## Step 2 — download and check the model (machine A only)

```bash
curl -L -o ~/qwen3-30b-a3b.gguf \
  https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf
```

Then confirm Loom can actually read it, before committing to anything:

```bash
./zig-out/bin/loom gguf info ~/qwen3-30b-a3b.gguf | head -30
```

Check `general.architecture = "qwen3moe"` and that no tensor type is one Loom
lacks. If a type is unsupported, the shard step below fails with
`UnsupportedTensorType` — deliberately, at plan time, rather than mid-token.

## Step 3 — look at the shard plan (machine A)

```bash
./zig-out/bin/loom gguf shard ~/qwen3-30b-a3b.gguf
```

Writes nothing. Expect roughly:

```
shards         6289 total = 145 resident + 6144 expert
resident       2.3 GB in 145 chunks (held by every node)
```

The Merkle root printed here is the **model version id**. Both machines must
agree on it; a mismatch means they are not running the same weights.

## Step 4 — sanity-check single-node (machine A)

Prove the model runs at all before involving the network:

```bash
./zig-out/bin/loom gguf run ~/qwen3-30b-a3b.gguf \
  --prompt "Write a haiku about distributed systems." --max-tokens 40
```

Keep this output. It is the reference for Step 8.

## Step 5 — open the firewall

Machine A must accept the p2p port (8771) from B, and vice versa (8781).

```bash
# Linux / ufw
sudo ufw allow from <B_IP> to any port 8771 proto tcp   # on A
sudo ufw allow from <A_IP> to any port 8781 proto tcp   # on B
```

macOS usually prompts on first bind — allow it.

> **Security.** The RPC and OpenAI surfaces have **no TLS and no
> authentication** (see [spec/SPEC.md](../spec/SPEC.md), "Transport security").
> The commands below keep them on `127.0.0.1`, which is the default, and reach
> them over SSH. Do not set `--rpc-addr 0.0.0.0` on a network you do not fully
> control — that publishes an unauthenticated inference endpoint. Only the
> **p2p** port needs to be reachable between the two machines.

## Step 6 — start the origin (machine A)

A holds every shard and acts as the bootnode that assigns committees.

```bash
./zig-out/bin/loom node \
  --gguf ~/qwen3-30b-a3b.gguf \
  --rpc-port 8770 --openai-port 8772 \
  --p2p-addr 0.0.0.0 --p2p-port 8771 \
  --advertise <A_IP>:8771 \
  --ram-gb 8
```

`--advertise` must be A's **LAN address**, not `127.0.0.1`. It is the address B
will dial; the default only works on one host and is the single most common
reason two machines never see each other.

Confirm in the banner:

```
bootnode   committee registry active (R target 2)
shards     mode=expert total=6289 (resident=145, expert=6144) held=6289 (100.0%)
serving    distributed GGUF (qwen3moe): ctx=... chat=chatml
```

`mode=expert` is the one to check — that is expert-aligned sharding. The
architecture on the `serving` line is read from the file, not assumed.

## Step 7 — start the partial node (machine B)

```bash
./zig-out/bin/loom node \
  --bootstrap <A_IP>:8771 \
  --hold-fraction 0.5 \
  --rpc-port 8780 --openai-port 8782 \
  --p2p-addr 0.0.0.0 --p2p-port 8781 \
  --advertise <B_IP>:8781 \
  --ram-gb 8
```

No `--gguf`: B has no model file and does not need one. It will join a
committee, receive an assignment, and sync ~10.5 GB from A. Expect:

```
joined committee 0 (...)
  synced N/N assigned shards, ... MB, verified against manifest root
shards     mode=expert total=6289 held=... (~51%)
serving    distributed GGUF (qwen3moe): ...
```

On gigabit this takes a few minutes. `--hold-fraction` selects an **exact**
count — `round(fraction x expert_shards)` — and only *which* shards is random,
seeded by `--seed`, so a restart re-picks the same set.

## Step 8 — the measurement that matters

Query **B** — the machine holding half the model:

```bash
printf '{"prompt":"Write a haiku about distributed systems.","max_tokens":40,"seed":1}\n' \
  | nc -w 600 127.0.0.1 8780
```

The response carries **`hit_rate`**. A value below 1.0 is the entire point: it
means experts B does not hold were fetched from A **inside the token loop**,
across your network, mid-inference. At `--hold-fraction 0.5` expect roughly
0.5 to 0.6, climbing as fetched shards are persisted — fetch-on-demand doubles
as organic replication, so `held=` grows as you use it.

The first request on a cold node is slow: every miss is a synchronous fetch.

Then the correctness check. Same prompt and seed, against A:

```bash
ssh <A_IP> "printf '{\"prompt\":\"Write a haiku about distributed systems.\",\"max_tokens\":40,\"seed\":1}\n' | nc -w 600 127.0.0.1 8770"
```

**The two `text` fields must be identical.** A node streaming half its weights
over a network must produce exactly what a full copy produces; anything else
means the fetch path is corrupting weights. Every fetched shard is
digest-verified against the manifest before it touches disk, so this should
hold by construction — which is exactly why it is worth confirming.

## Step 9 — get real throughput numbers

Now that B is warm, measure it against A, same prompt:

```bash
# on B
time (printf '{"prompt":"Explain consistent hashing.","max_tokens":100,"seed":1}\n' | nc -w 900 127.0.0.1 8780)
# on A
time (printf '{"prompt":"Explain consistent hashing.","max_tokens":100,"seed":1}\n' | nc -w 900 127.0.0.1 8770)
```

The gap between them is the cost of the network tier on your fabric. Sweep
`--hold-fraction` (0.3, 0.5, 0.75) and plot it — that curve is the honest
answer for your hardware, and it is the number worth having.

## Step 10 — churn

Stop A (Ctrl-C) while B is running. Within one heartbeat interval (5 s) B marks
it dead and adopts its shards into the want-set; the repair loop (2 s) then
looks for another holder:

```
heartbeat: committee member <A_IP>:8771 DEAD
heartbeat: adopted N shard(s) from dead <A_IP>:8771 into wanted
```

With only two machines there is no other holder, so repair will keep trying —
which is the correct behaviour and a useful thing to see. Requests to B for
experts it does not hold will now fail loudly rather than return quietly wrong
output. Restart A and watch it recover.

---

## Two machines and the `R target 2` caveat

Loom targets **at least two sources per expert**. With two nodes, one of which
is the origin holding everything, a shard B does not hold has exactly one
source. So this topology demonstrates the *fetch path* honestly, but not the
*fault tolerance* the design aims at — for that you need a third machine, or
two partial nodes plus a seed.

If you want to see genuine mutual dependence on two boxes, run **two nodes on
each machine**: one origin plus one partial per host, with the partials at
`--hold-fraction 0.4`. Then no single node holds the model, the union covers
it, and pulling either machine exercises real repair.

## Troubleshooting

| Symptom | Cause |
|---|---|
| B never finds A | `--advertise` is `127.0.0.1` instead of the LAN IP. This is the most common failure. |
| Connection refused on the p2p port | Firewall, or A bound p2p to loopback. A needs `--p2p-addr 0.0.0.0`. |
| `p2p server failed: AddressInUse` then exit(1) | Port already taken. Fatal on purpose: a node that cannot serve peers should not pretend to run. |
| `mode=fixed`, not `expert` | The model is dense, not MoE — nothing to shard by expert. |
| `UnsupportedTensorType` at shard time | A quantization Loom does not implement. Check with `gguf info` and pick another quant. |
| `gguf serve disabled: ...` in the banner | The store is not expert-sharded, or its resident bundle is incomplete; the node falls back to the loom-format engine. |
| Sync is slower than the link | Expected on a cold start: B is writing ~10 GB into a sparse file while verifying every shard's digest. |
| First request takes minutes | Expected on a cold partial node: every miss is a synchronous peer fetch. It warms as shards persist. |
| Outputs differ between A and B | Should not happen. Confirm both report the same Merkle version id in `gguf shard` / the banner — different quant files are different models. |

## See also

- [MULTI-NODE-WALKTHROUGH.md](MULTI-NODE-WALKTHROUGH.md) — the same flow on one host, plus integrity and corruption-healing checks
- [CLI.md](CLI.md) — every flag in detail
- [spec/SPEC.md](../spec/SPEC.md) — committees, wire protocol, trust model


## Mixed Mac + Linux

Yes, and for the capacity experiment it is arguably better than two Macs.

**What is platform-neutral.** The distribution plane is: GGUF byte ranges,
SHA-256 leaf digests, a Merkle-rooted manifest, and binary wire frames written
little-endian. None of that depends on the host — a shard hashed on Linux
verifies on macOS and vice versa. Compute is node-local by design, so a peer
never sees another node's activations, only its expert bytes.

**What is not.** The two nodes will not produce *identical* text. A Mac
defaults to the recorded Metal path (f32 activations) while a Linux node runs
the CPU kernels (int8 activations, approximate by ~0.4% per element). Both are
valid forward passes and each node answers its own requests, so this is fine
for serving and for the capacity measurement. It is not fine for anything that
assumes bit-identical output across nodes — the earlier three-node run asserted
identical text, and that held only because every node was on the same CPU path.
Redundant-recompute verification (v2) has to account for this before it can
compare two nodes' answers.

**What was broken until now.** Loom did not build for Linux at all: a profiler
added during Metal work called `std.c.getenv` and `std.c.clock_gettime`, and
libc was only linked for the Metal build. macOS links libc regardless, so it
compiled there and failed everywhere else with "dependency on libc must be
explicitly specified". Since the release workflow ships
`x86_64-linux-musl` and `aarch64-linux-musl`, that broke the Linux artifacts
too. libc is now linked for every target; all four release targets verified.

**Why a Linux box helps the experiment.** The point is pooling RAM across nodes
that individually cannot hold the model. A second machine with *different* RAM
makes the result stronger, not weaker: it shows the capacity argument does not
depend on two identical boxes. And a Linux box with a discrete GPU is the only
hardware that can eventually test the Vulkan backend, where discrete VRAM
removes the shared-bus ceiling that caps the Metal win at ~1.9x.

**One caveat to plan around.** Loom has no Vulkan backend yet, so a Linux node
runs on CPU. If that machine is also the one holding most of the shards, its
serving throughput will be the CPU number, not the Metal number. For the
capacity experiment that is fine — what is being measured is cold-miss-to-disk
rate and whether the working set stays resident somewhere — but do not read a
mixed-cluster tok/s figure as a per-node performance result.
