# GLM 5.2 readiness audit (2026-08-02)

Run with `loom gguf check` against live checkpoints plus the llama.cpp
support trail. Verdict: **GO, with one bounded engine variant** — and a
zero-work bridge model available immediately.

## GLM 5.2 (the thesis model)

- **Checkpoints exist.** Full 744B GGUFs (Unsloth UD quants): ~223 GB at
  1-bit, ~239 GB at 2-bit (fits a 256 GB box), 372–475 GB at 4-bit.
  Community REAP-pruned 504B variants halve the expert count (128 experts).
- **`loom gguf check` on a real shard:** arch `glm-dsa`, 78 blocks,
  expert-aligned — **loom's sharder already distributes it**. Two blockers:
  no engine for `glm-dsa`, and Q3_K decode for that particular quant
  (avoided by choosing a Q4_K-family quant; the V3 check below shows the
  existing decoder coverage suffices there).
- **llama.cpp state:** mainline runs GLM 5.2 with a **dense-attention
  fallback** — the DSA sparse-indexer path lives only in forks. The indexer
  tensors ship on ~1 layer in 4; community GGUFs patch the rest by
  duplication.
- **loom engine delta (dense fallback, llama.cpp's own approach):**
  `glm-dsa` as a deepseek2-family variant — MLA attention (have it),
  noaux_tc sigmoid routing with `exp_probs_b` (have it), shared expert
  (have it). New work: tensor/metadata mapping, and skipping the DSA
  indexer + MTP tensors at load. DSA sparse attention itself is a later,
  separate feature (it is a speed/long-context optimization, not a
  correctness requirement).

## Bridge model: DeepSeek-V3-0324 (671B)

`loom gguf check` verdict: **"loads and runs."** deepseek2 arch, 61 blocks,
256 experts / 8 active, MLA, Q4_K/Q6_K/F32 only, expert-aligned. Q4_K_M is
~404 GB across 9 shards; 2-bit cuts to ~230 GB. Zero engine work: this is
the thesis-scale deployment loom can run **today**, exercising the same
sharding, gossip, network_id and churn machinery as GLM 5.2 will.

## Hardware gate (the open decision)

Per-token expert working set at this scale makes WAN fetch useless (see the
fabric table in CLAUDE.md, re-confirmed by the three-machine test). Options:

| shape | fits | cost class |
|---|---|---|
| one 512 GB RAM dedicated box (origin, v0 disk-streaming + pinning) | V3 Q4_K_M, GLM 5.2 4-bit | ~EUR 150-250/mo |
| one 256 GB box | V3 2-bit, GLM 5.2 2-bit | ~EUR 80-140/mo |
| LAN cluster, >=25 GbE, ~400 GB aggregate RAM | full v1 distribution | project-sized |

## Recommended order

1. Deploy DeepSeek-V3 on approved hardware (bridge; zero engine risk).
2. Build the `glm-dsa` engine variant against a small fixture + the REAP
   checkpoint's header shapes; validate vs llama.cpp's dense fallback.
3. Swap the network to GLM 5.2 once 1 and 2 are green.
