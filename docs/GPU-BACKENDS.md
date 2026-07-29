# GPU backends: Metal on Apple, Vulkan on Linux and Windows

Plan for issues [#10](https://github.com/ch4r10t33r/loom/issues/10),
[#12](https://github.com/ch4r10t33r/loom/issues/12) and
[#13](https://github.com/ch4r10t33r/loom/issues/13), following the structure
ZINC ([zolotukhin/zinc](https://github.com/zolotukhin/zinc)) used to bring up
Apple Silicon. ZINC is the closest prior art: a Zig inference engine that beats
llama.cpp on AMD RDNA4 with no ROCm, and ships native MSL shaders on macOS.

Its approach is worth copying, and its scale is worth knowing before starting.

## What ZINC actually did

Read out of its own enablement notes and source tree:

| piece | size | what it is |
|---|---|---|
| `src/gpu/interface.zig` | 68 lines | the entire backend switch |
| `src/metal/` | ~2,150 lines | device, buffer, pipeline, command, plus one ObjC shim |
| `src/vulkan/` | ~2,190 lines | the Linux equivalent |
| `src/shaders/metal/` | **23,014 lines, 185 files** | the actual work |

The ratio is the lesson. The abstraction is trivial; the shaders are the
project. ZINC does not have one generic `matmul` kernel — it has
`dmmv_q4k_lmhead_argmax.metal`, `dmmv_q4k_dense_gate_up_swiglu.metal`,
`dmmv_mxfp4_moe_sg.metal`. Kernels are specialized per quantization, per fusion
opportunity, and per shape, because that is where the performance is.

Five design decisions worth adopting wholesale:

1. **Backend selection is comptime, not runtime.** macOS compiles Metal, Linux
   compiles Vulkan, and the inactive backend is not in the binary. This keeps
   the seam cheap and stops Vulkan headers leaking into the macOS build.
2. **One Objective-C file.** `shim.m` is the only non-Zig source, built by
   `build.zig` with `-fobjc-arc -fmodules`, linking `Metal` and `Foundation`.
   No Xcode project.
3. **Shaders compile at runtime from source** into pipeline state objects,
   rather than being precompiled into a metallib. Simpler build, and kernels
   can be specialized on the model's actual shapes at load.
4. **Zero-copy weight wrapping.** GGUF tensor regions are page-aligned and
   handed to `newBufferWithBytesNoCopy`. Weights are never staged or uploaded;
   the mmap *is* the GPU buffer. ZINC calls this "the single biggest
   architectural shift" in its Apple bring-up.
5. **Only the forward runtime is backend-specific.** Tokenizer, GGUF parsing,
   HTTP routes, scheduler and catalog stay shared.

## Where loom differs, and why it matters

Loom is not a single-machine engine, and that changes one thing fundamentally.

**A fetched expert has to reach the GPU before the matmul can run.** In loom a
routed expert may arrive from a peer *during* the token loop. On the CPU path
that block is simply memory. On a GPU it is not:

- **Apple, unified memory.** The fetched bytes are already in the address space
  the GPU reads. `newBufferWithBytesNoCopy` wraps them and the cost is a page
  table entry. This is free, and it is why Metal comes first.
- **Discrete GPU over Vulkan.** The block must be copied host→VRAM before use.
  At ~5 MB per expert and 6 experts per layer, that is a PCIe transfer sitting
  *inside* the critical path of every token — a new tier the whitepaper's
  analysis does not currently account for.

That second case is in direct tension with the whitepaper's principle that
nothing requiring a transform-before-use belongs in the token loop, which is
exactly why erasure coding was banished to the propagation plane. A VRAM upload
is a smaller version of the same mistake.

Consequences for the design:

- **Metal first**, not because macOS matters more, but because unified memory
  is the case where the distributed thesis and GPU execution compose without a
  new bottleneck.
- **On discrete GPUs, the resident set should live in VRAM and the hot expert
  set should be pinned there**, with peer-fetched experts treated as a slow
  tier that is uploaded once and cached — the same tiering logic loom already
  applies to disk versus peers, one level down. A node whose VRAM cannot hold
  the resident bundle should say so and fall back, the way the resident-gate
  check already does.
- **The measured-tier-order principle extends to VRAM.** Which is faster for a
  missing expert — a peer fetch plus upload, or a local disk read plus upload —
  is again a per-deployment question, not a constant.

## Proposed shape

```
src/compute/backend.zig     comptime backend selection (the 68-line file)
src/compute/cpu.zig         current ggml.zig kernels, behind the interface
src/metal/{shim.h,shim.m}   the one Objective-C file
src/metal/{device,buffer,pipeline,command}.zig
src/vulkan/…                same, for Linux and Windows
src/shaders/metal/*.metal
src/shaders/vulkan/*.comp   compiled to SPIR-V
```

The seam is already in the right place. Everything performance-critical now
goes through two functions in `ggml.zig` — `matvec` and `matmul` — because the
SIMD, threading and batching work concentrated it there. A backend interface is
essentially those two entry points plus buffer lifetime, which is why #10 is
small and #12/#13 are not.

## Order of work

1. **#10 — backend seam. Done.** `src/compute/backend.zig` selects the backend
   at comptime from `-Dgpu=`; `src/compute/cpu.zig` binds the existing kernels
   to it. Fourteen operations, and every engine now calls through them. No
   behaviour change: the bit-identical, golden-vector and architecture tests
   all pass unchanged, which is the proof it was a pure refactor.
2. **#12 — Metal. Substrate and kernels done. On Apple Silicon it cannot win
   much at decode, and the reason is architectural rather than fixable.**

   Effective bandwidth against tensor size, one dispatch per command buffer.
   Best-of-20 at cols=2048 on an M5 (10 GPU cores, 16 GB), against 8 CPU
   threads:

   | rows | bytes read | GPU before | GPU after | CPU, 8 threads |
   |---|---|---|---|---|
   | 2,048 | 2.4 MB | 0.146 ms | 0.139 ms | 0.05 ms |
   | 5,632 | 6.5 MB | 0.238 | 0.186 | 0.13 |
   | 16,384 | 18.9 MB | 0.416 | 0.419 | 0.28 |
   | 32,000 | 36.9 MB | 0.639 | 0.612 | 0.62 |
   | 65,536 | 75.5 MB | 1.212 | **0.919** | 1.30 |
   | 131,072 | 151.0 MB | 2.381 | **1.609** | 2.57 |

   At 131,072 rows that is 63 GB/s before and **94 GB/s** after.

   **The earlier "bandwidth wall" reading was wrong, and the way it was wrong
   is worth recording.** The first version of this table topped out at 61 GB/s
   and was read as the machine's ceiling, on the reasoning that unified memory
   makes the CPU and GPU share one bus. But 61 GB/s was *our kernel's*
   achieved bandwidth, and nothing had measured the hardware's. A kernel doing
   nothing but streaming reads — sum a large f32 buffer, one float4 per lane —
   reaches **~110 GB/s** on the same M5 (`LOOM_BW_PROBE=1 zig build test
   -Dgpu=metal`). The ceiling was ours, not the machine's, and 61 was 55% of
   it.

   Two changes closed most of the gap:

   1. **Four bytes per lane instead of one.** The original kernel was
      perfectly coalesced but issued a quarter-width load, so the inner loop
      ran four times as often, with four times the address arithmetic and four
      times the scale unpacking, and never had enough loads in flight to cover
      their own latency. Each lane now takes four consecutive quant bytes; the
      32 lanes still cover a super-block's 128 bytes, in one step rather than
      four.
   2. **Branchless 6-bit scale unpack.** This was not optional. With one byte
      per lane every lane had the same sub-block index and the `j < 4` test
      was uniform across the SIMD group. Widening the load makes the index
      vary with the lane, so the branch diverges and the group executes both
      sides. Measured on its own, the widened kernel *without* this change was
      **slower than the original at every size below 131,072 rows** — the win
      only appeared once the branch was gone.

   Two things were tried and are not kept, both measured: unrolling the block
   loop by two on independent accumulators (+6% at 131k rows, −30% between 5k
   and 32k), and hoisting the loop body into a `static inline` helper (−30% at
   16k).

   **A note on measurement.** These numbers are best-of-20, not means. The
   benchmark previously averaged, and on a machine whose GPU is shared with
   the window server that measures whatever else was drawing: repeated runs of
   an unchanged kernel varied by 50%, wider than most of the effects under
   test. Averages briefly showed the widened-but-branchy kernel as a 1.4x win
   when it was in fact a regression.

   `MIN_GPU_ROWS` drops from 100,000 to 65,536 accordingly — the crossover
   moved, but not far enough to reach a 1.1B model, whose largest tensor is
   the 32,000-row output head. **Decode on this hardware is still CPU
   territory**, and the reason is unchanged even though the number was wrong:
   the CPU sits at ~53 GB/s flat, and even a GPU at 94 GB/s cannot overcome a
   ~262 us fixed cost per command buffer when a tensor's worth of work is tens
   of microseconds.

   **Where a GPU still earns its place**, and neither is decode on this
   hardware:

   - **Prefill.** A batch of N tokens reuses each weight N times, which turns
     a bandwidth-bound problem into a compute-bound one. That is the regime a
     GPU is built for, and the one place a large win should exist here.
   - **Discrete GPUs.** Dedicated VRAM has its own bus, several times wider
     than host DRAM, so the shared-bandwidth argument above does not apply —
     though the host-to-VRAM upload tier does (see below).

   Done: the substrate, `dmmv_q4k` (coalesced, one SIMD group per row), the
   elementwise and norm kernels, device-resident activations, a whole FFN
   block in one command buffer, one encoder with barriers only between
   dependent dispatches. All validated against exact CPU references. Metal
   declines any matvec it would lose on, so the build matches the CPU rather
   than losing to it.

3. **Distributed integration.** Peer-fetched experts wrapped zero-copy on
   Apple; a VRAM tier with pinning on discrete GPUs.

4. **#13 — Vulkan. Not yet, and deliberately.** Porting a design that does not
   beat the CPU would reproduce the same result against a second API, with
   more work: a discrete GPU adds the host-to-VRAM upload tier that unified
   memory avoids, so Vulkan starts strictly harder. Metal should first
   demonstrate a win on some real workload — most likely batched prefill or a
   larger model — and only the structure that produced that win is worth
   porting.

## The kernel is at the memory ceiling; submission is the gap

Measured on an M5, Q4_K matvec at cols=2048, best-of, with and without the
per-dispatch command buffer:

| rows | bytes | gpu, 1 cb each | gpu, amortized | GB/s amortized | cpu, 8t |
|---|---|---|---|---|---|
| 2,048 | 2.4 MB | 0.152 ms | 0.035 ms | 67.4 | 0.049 ms |
| 16,384 | 18.9 MB | 0.373 | 0.171 | 110.1 | 0.354 |
| 32,000 | 36.9 MB | 0.504 | 0.323 | **114.2** | 0.662 |
| 65,536 | 75.5 MB | 0.825 | 0.651 | 115.9 | 1.405 |
| 131,072 | 151 MB | 1.465 | 1.306 | 115.6 | 2.440 |

Two things follow, and the first corrects an earlier conclusion in this
document.

**The kernel is not the problem.** At 115 GB/s it is at the DRAM ceiling a pure
streaming-read probe measures on this machine (~110 GB/s), and level with
ZINC's own microbenchmark, which reports 109 GB/s at 27 MB. An earlier
comparison here concluded loom's kernel was ~3x slower; that was an artifact of
comparing loom's per-dispatch timings against ZINC's amortized ones. ZINC
reports 259.81 us for a 27 MB shape, which is *less* than a single command
buffer costs here, so its numbers cannot include one. Measurements taken under
different conditions are not a comparison.

**Submission is the gap.** At 32,000 rows the same kernel takes 0.504 ms with a
command buffer per dispatch and 0.323 ms without — ~180 us of pure overhead,
which is most of the difference between losing to the CPU and beating it 1.3x.
Amortized, the GPU wins from 32,000 rows up; per-dispatch it does not win until
131,072.

**Where ZINC is genuinely ahead is small shapes.** It reports 239 GB/s at
2.25 MB where loom manages 67 GB/s at 2.4 MB. Cache-resident matvecs are
latency- and occupancy-bound rather than bandwidth-bound, and that is what
ZINC's register-level work buys: scales held in `ushort4` registers rather than
a thread-private array, fully unrolled inner loops, the 16-byte block header
read as one `packed_uint4`, nibbles extracted four at a time from a `ushort`.
Porting the first three of those gave loom nothing at DRAM-bound sizes, which
is consistent — there is no headroom left there — and they are the right thing
to revisit for models whose tensors sit in cache.

## Against ZINC, measured

TinyLlama-1.1B Q4_K_M, 128 tokens, same prompt, M5, decode only (both engines
report decode separately from prefill):

| engine | backend actually used | tok/s |
|---|---|---|
| loom | **CPU** (Metal built in, declined by calibration) | 55.7, 56.0, 56.2 |
| ZINC | **Metal** | 52.9, 53.0, 56.1 |

Loom matches ZINC on this model **without using the GPU at all**.

That settles a question worth being precise about: matching tok/s against a
Metal engine does *not* demonstrate that loom's Metal optimisations are
complete. Here it demonstrates the opposite — loom's CPU path is already
competitive with ZINC's Metal path, and loom's GPU path is not being used
because calibration measured it and found it slower at every shape this model
issues.

It also bounds what the remaining GPU work can win. On unified memory the CPU
and GPU share one bus; measured on this machine the GPU beats the CPU by ~1.5x
only above 65k rows, and TinyLlama's largest tensor is the 32k-row output head.
The prize for finishing the GPU path is real but it is not a multiple, and it
is largest for big models and for prefill, not for small-model decode.

## What this means for the target, honestly

For **decode** on unified memory, the ceiling is the memory bus and the CPU is
already near it. Measured on this machine, for a 1.1B Q4_K_M model reading
0.64 GB per token:

| | bandwidth | tokens/sec |
|---|---|---|
| the real forward pass today | 36 GB/s | 56 |
| the CPU kernels in isolation | 53 GB/s | 83 |
| the GPU, on tensors large enough | 61 GB/s | 96 |

So the next win for decode is not a GPU. It is closing the gap between the 36
GB/s a full forward pass achieves and the 53 GB/s the kernels already reach on
their own — the difference is attention, the elementwise ops and the types
still on the slow dequantize path, not the matmuls.

## Scale, honestly

ZINC's 23,000 lines of shaders are one person's sustained effort, and it is
still adding scenarios. This is a multi-week program per backend, not a
sitting. The order above is chosen so each step is independently useful: the
seam alone makes the CPU path testable in isolation, and Q4_K `dmmv` alone
covers most of a real model before a single other kernel exists.

The CPU path is not exhausted either. It is currently around 5.8 GB/s on Q4_K
against roughly 70 GB/s of achievable bandwidth, so there is still an order of
magnitude available before the GPU is the only remaining lever.

## References

- [ZINC Apple Silicon enablement notes](https://github.com/zolotukhin/zinc/blob/main/docs/APPLE_SILICON_METAL_ENABLEMENT.md)
- [ZINC Metal backend reference](https://github.com/zolotukhin/zinc/blob/main/docs/APPLE_METAL_REFERENCE.md)
- [whitepaper](../whitepaper/WHITEPAPER.md), principle 7 and the bandwidth table
