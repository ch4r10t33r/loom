# Kimi K3: what supporting it would take

`moonshotai/Kimi-K3`, 1.56 TB of open weights. This note is a gap analysis, not
a plan. It separates what Loom already handles from what it does not, and sizes
each piece.

Every figure below comes from the published `config.json`, the safetensors
index, and a range request against one weight shard. None of it is estimated.

## Why this model matters to Loom

K3 fits the architecture better than any previous target, and it also tests
the architecture's stated limits harder.

**92.7% of the file is routed experts.** 1.446 TB of the 1.561 TB total is
`block_sparse_moe.experts.*`: immutable, content-addressed, and sparsely read.
The non-expert remainder is 114 GB. That ratio is more extreme than GLM 5.2's, and
it is exactly the shape the design assumes.

**It is sparser than anything Loom has targeted.** 16 of 896 experts activate
per layer per token: 1.79%, against GLM 5.2's 8-of-256 (3.12%). Sparser
routing means a pinned hot set covers proportionally more activations, and
cluster-wide deduplication has more to work with.

**No commodity box holds it.** At 1.56 TB, the capacity argument stops being a
nice-to-have and becomes the only way to run the model outside a datacenter.
For GLM 5.2 a single well-provisioned machine was a real alternative; here it
is not.

**The bandwidth math gets worse.** Per token a forward pass
reads 92 MoE layers x 16 experts x 17.55 MB = 25.8 GB, against GLM 5.2's
11.4 GB. That is 1,472 separate ~17.5 MB fetches per token. Against the
whitepaper's bandwidth table:

| source | effective bw | K3 s/token | GLM 5.2 s/token |
|---|---|---|---|
| 1 GbE | ~0.11 GB/s | ~235 | ~104 |
| 25 GbE | ~3.1 GB/s | ~8.3 | ~3.7 |
| 100 GbE / RDMA | ~12 GB/s | ~2.2 | ~1.0 |
| local PCIe5 NVMe | ~5 GB/s | ~5.2 | ~2.3 |

The existing warning that a slow fabric buys capacity rather than speed applies
about 2.3x harder. K3 on anything below 100 GbE is a throughput-and-capacity
play only.

## What already works unchanged

**Expert-aligned sharding.** `buildExpertManifest` keys on the GGUF naming
convention for stacked 3D expert tensors, not on an architecture name. Given a
GGUF conversion, K3 shards into 92 x 896 = 82,432 expert shards of ~17.5 MB.
The bookkeeping is affordable: a 2.64 MB manifest of SHA-256 digests and a
10.1 KB holdings bitmap.

**The router.** K3 uses sigmoid scoring with a `noaux_tc` selection bias
(`e_score_correction_bias`), renormalized top-k, and a routed scaling factor of
1.0. Its `num_expert_group`/`topk_group` are both 1, so the grouped top-k
degenerates to plain top-k. That is exactly what `src/gguf/moe.zig` already
implements; the routing needs nothing new.

**MXFP4.** The checkpoint is `mxfp4-pack-quantized` (4-bit, group size 32,
uint8 scales) on expert weights, with attention, shared experts, dense MLP,
`lm_head` and the vision tower left unquantized. Loom implements MXFP4 as of
the codebook-quant work, so the expert format is already readable.

**Fetch, verification, repair, metering, the control plane.** All of it is
byte-oriented and architecture-blind.

## What does not work, in order of difficulty

### 1. There is no GGUF (blocking)

K3 ships as safetensors with `custom_code` and a `kimi_k3` model type that
llama.cpp does not convert. Loom reads GGUF exclusively. Nothing downstream can
start until either llama.cpp adds the architecture or Loom grows a
safetensors path. Everything below assumes this is solved.

One wrinkle: the per-expert weights are separate tensors
(`experts.<e>.w1.weight_packed`, 497,220 tensors in total), not the stacked 3D
tensors llama.cpp conversions normally produce. A conversion that preserves the
per-expert layout would need `buildExpertManifest` taught a second naming
pattern; a conversion that stacks them works today.

### 2. Kimi Delta Attention — 69 of 93 layers (the big one)

K3 is a hybrid. Only 24 layers (indices 3, 7, 11, ... 91, plus 92) use MLA. The
other 69 use KDA, a gated linear attention with a short causal convolution:
`q/k/v_proj` into 96 heads x 128, a depthwise `conv1d` of kernel width 4 on
each of q/k/v, a low-rank forget gate (`f_a_proj` 7168->128, `f_b_proj`
128->12288), Mamba-style `A_log` and `dt_bias` decay parameters, a per-head
output norm, and an output gate (`g_proj`) that both attention types use
(`mla_use_output_gate: true`).

Loom has no linear-attention machinery at all. This is a new engine, not a
variant of the GQA one, and it changes the memory model:

- A recurrent state replaces the KV cache for those layers: 96 x 128 x 128
  floats = 6.29 MB per layer, 434 MB per sequence across 69 layers, and
  constant in sequence length rather than growing with it.
- The whitepaper's "KV cache is a non-issue, do not build KV distribution"
  conclusion was derived for MLA. It needs redoing for a hybrid: the 24 MLA
  layers still accumulate per-token KV, and the model advertises a 1M context.

### 3. Latent MoE

Experts do not operate on the 7168-wide residual. Each MoE layer has a shared
`routed_expert_up_proj` (7168->3584), a `routed_expert_norm`, and a
`routed_expert_down_proj` (3584->7168); the 896 experts live in that 3584-wide
latent space. Loom's model of "an expert is a self-contained FFN" still holds:
the projections are per-layer, not per-expert, so they belong in the resident
bundle. But the forward pass and the resident/expert split both need to know
about them. At BF16 they add ~9.4 GB to the resident set.

### 4. The resident bundle is 114 GB (the hardest constraint)

Loom requires every node to hold the resident bundle in full, and fails
closed otherwise. This is a deliberate safety property: an mmap hole reads as
zeros and would silently corrupt inference rather than erroring.

114 GB per node is not commodity hardware. There are two ways out, and
choosing between them is a real design decision:

- **Requantize the dense part.** It is BF16 today because the MXFP4 recipe
  explicitly excludes attention, shared experts and `lm_head`. At 4 bits the
  resident set drops to roughly 29 GB, which is mmap-able on a 32-64 GB box.
  This costs quality on the tensors the model authors chose not to
  quantize.
- **Relax "resident is mandatory."** This is load-bearing for the current
  safety argument and should not be weakened casually.

### 5. Smaller, but still required

- **The `situ` activation** (`activation_situ_beta: 4.0`,
  `activation_situ_linear_beta: 25.0`) replaces SwiGLU. `denseFFN` hardcodes
  SwiGLU.
- **Cross-block residual machinery**: `self_attention_res_norm`/`_proj`,
  `mlp_res_norm`/`_proj`, `output_attn_res_norm`/`_proj`, all `[1, 7168]`
  per-channel gates, with `attn_res_block_size: 12` suggesting connections
  every 12 layers. Small tensors, new forward-pass structure.
- **MLA variations**: `mla_use_nope: true` and an output gate on the MLA
  layers, neither of which the current MLA engine implements.
- **Tokenizer**: tiktoken-based, 163,840 entries. Probably lands on the
  existing byte-level BPE path if the conversion emits merges, but unverified.
- **Vision tower** (27 blocks) and `mm_projector`. Ignorable for text-only
  serving, though they occupy the file and would be sharded as resident bytes
  unless excluded.

## Summary

| Area | Status |
|---|---|
| Sharding, placement, sync, repair, verification | works unchanged |
| MoE routing (sigmoid, noaux_tc bias, renorm) | works unchanged |
| MXFP4 expert weights | works unchanged |
| GGUF conversion | **blocking**: does not exist |
| KDA linear attention, 69/93 layers | new engine, new state model |
| Latent MoE projections | new forward-pass structure |
| 114 GB resident bundle | needs requantization or a policy change |
| `situ`, residual gates, MLA nope/output-gate | incremental forward-pass work |
| Vision tower, tiktoken | out of scope / probably free |

The distribution plane (the part Loom actually is) needs no changes. The work
is all in the inference engine, and most of it is one item: KDA.
