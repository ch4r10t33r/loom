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
2. **#12 — Metal. Substrate and kernels done; dispatch structure is the
   remaining gap.**

   The decisive measurement, on an M5:

   | | time |
   |---|---|
   | empty command buffer, commit + wait | 15.4 us |
   | + one trivial dispatch | 263.4 us |
   | ten dispatches, one buffer | 262.3 us (26 us each) |

   The cost is neither per dispatch nor the commit floor. It is a **fixed
   ~262 us paid once per command buffer that contains any GPU work**. Ten
   dispatches cost the same as one. So the unit of work that matters is not
   the kernel, it is the buffer, and anything that forces the host to read an
   intermediate splits one buffer in two and adds 262 us.

   With a matvec kernel measured at 18 us, that arithmetic decides everything:

   | dispatches per command buffer | predicted |
   |---|---|
   | one per matvec (~110 per token) | 33 tok/s |
   | one per layer (22) | 134 tok/s |
   | one per token | 504 tok/s |

   The first row matches what was actually observed, which is the check that
   the model is right rather than a convenient story.

   Done so far: the Metal substrate, `dmmv_q4k` (8x faster per row than eight
   CPU threads), the elementwise and norm kernels, device-resident activation
   buffers, and a whole FFN block encoded into a single command buffer.

   Still slower than the CPU, and the reason is now specific: the shim opens a
   **new compute encoder per dispatch**, so a six-dispatch FFN pays six encoder
   boundaries. The FFN block measures 1.04 ms against the CPU's 0.47 ms, well
   above the ~310 us the buffer arithmetic predicts. One encoder per command
   buffer, with barriers only between genuinely dependent dispatches, is the
   next change.

   After that, the remaining kernels — Q6_K `dmmv`, RoPE, attention — so a
   whole layer, and then a whole token, fits in one buffer.

3. **Distributed integration.** Peer-fetched experts wrapped zero-copy on
   Apple; a VRAM tier with pinning on discrete GPUs.

4. **#13 — Vulkan.** Same kernels as compute shaders, plus the upload tier
   that Apple does not need.

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
