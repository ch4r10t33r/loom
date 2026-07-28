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

   Effective bandwidth against tensor size, one dispatch per command buffer:

   | rows | bytes read | GPU | CPU, 8 threads |
   |---|---|---|---|
   | 2,048 | 2.4 MB | 7.9 GB/s | 51.3 GB/s |
   | 5,632 | 6.5 MB | 13.8 | 46.0 |
   | 16,384 | 18.9 MB | 19.8 | 51.2 |
   | 32,000 | 36.9 MB | 24.9 | 53.0 |
   | 65,536 | 75.5 MB | 38.0 | 53.7 |
   | 131,072 | 151.0 MB | **61.4** | 53.5 |

   The CPU is flat at roughly 50 GB/s across four orders of magnitude, which
   is the signature of a memory-bandwidth-bound workload: it is already
   extracting close to what the memory system will give. The GPU climbs from
   7.9 to 61.4 GB/s as its fixed dispatch cost amortizes, and tops out only
   **15% above the CPU** — on tensors four times larger than anything in a
   1.1B model.

   That is the whole story, and it is a property of the machine. Apple Silicon
   is unified memory: the CPU cores and the GPU read the *same* DRAM over the
   *same* bus. A quantized matvec moves far more bytes than it does
   arithmetic, so both processors run into the same wall, and the GPU's
   advantage shrinks to how much more of the bus it can saturate.

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
