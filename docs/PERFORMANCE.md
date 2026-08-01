# Performance

Kernels are SIMD (`@Vector`, so portable rather than per-architecture
intrinsics) and matvecs run row-parallel across a worker pool. Measured on a
10-core Apple M5, TinyLlama 1.1B Q4_K_M, median of three 64-token runs:

| | tok/s | vs baseline |
|---|---|---|
| scalar, single-threaded (original) | 1.1 | 1x |
| SIMD, single-threaded | 6.0 | 5.5x |
| SIMD + 8 threads | 26.0 | 23.6x |
| + int8 activations, 1 thread | 15.0 | 13.6x |
| + int8 activations, 8 threads | 51.9 | 47x |
| **+ vectorized attention** | **57.8** | **53x** |

Prefill is batched separately, because the prompt is known up front: the same
unpacked weight serves several tokens instead of being unpacked once per
token. A 145-token prefill on the same model, best of three:

| `--batch` | time |
|---|---|
| 1 (off) | 2.65 s |
| 4 | 2.01 s |
| **8** | **1.87 s** |

That is 1.4x rather than the 2.4x the kernel microbenchmark shows in
isolation, because attention is quadratic in prompt length and is not batched
— it is bound by the KV cache rather than by weight reads, so there is nothing
to amortize there.

The last step is the one worth understanding. Profiling the kernels showed
**76-92% of a quantized matvec was the dequantize, not the dot** — each block
was expanded into a 256-float scratch buffer and pushed through L1 only to be
read straight back. Quantizing the *activation* vector to int8 once per matvec
and dotting it against the packed weights as integers removes that buffer
entirely and uses lanes four times as wide.

That step is an approximation the earlier ones were not: the activation
carries roughly 0.4% per-element error, so results are no longer bit-identical
to the dequantize path. Over a long dot those errors are independent and
largely cancel, and it is what every production CPU inference engine does.

Row-splitting is exact, not approximate: each output row is one independent
dot product, so there is no reassociation and results are bit-identical to the
serial path at any thread count. The tests assert that rather than assuming it.

Thread count defaults to `cpu_count - 2` — a node keeps p2p, gossip and repair
threads running during a generation, and past that point oversubscription
costs more than the extra cores return. Override with `--threads N`;
`--threads 1` disables the pool.

The CPU path is not exhausted — Q4_K runs at about 5.8 GB/s against roughly
70 GB/s of achievable bandwidth — but the GPU backends below have overtaken
it on every machine that has one.

## GPU: Metal (Apple Silicon)

DeepSeek-Coder-V2-Lite-Instruct Q4_K_M (16B MoE, ~10 GB), single-stream
decode on an Apple M5, `--gpu-ops`. The reference bar was llama.cpp's
reported 38 tok/s on an M3. The chain that got there — each step merged only
after a differential test against the exact reference:

| step | tok/s |
|---|---|
| CPU baseline (int8, 8 threads) | ~3.7 |
| device-resident weights + per-op kernels | ~12 |
| device-side routing + fused MoE layer | ~25 |
| whole-layer submission (`mlaLayerTail`) | 34.7 |
| whole-token command buffer, 1.8 cb/token | 42 |
| **+ attention re-grids, f16 W_k** | **39.7–44.4** |

The recurring lesson: submission count is the number that never lies. The
same work in fewer `commitAndWait`s is where every large jump came from.

## GPU: Vulkan (NVIDIA / anything with a driver)

Same model, RTX 3060 12 GB (driver 580.173, Ubuntu 22.04), one afternoon of
measured steps from bring-up to 6.5x the CPU. End-to-end numbers include the
CPU prompt prefill; marginal is the steady-state decode rate:

| step | end-to-end tok/s |
|---|---|
| CPU baseline (9 cores) | 3.1 |
| bring-up (host-memory weights, ~500 submits/token) | 0.6 |
| VRAM weights + one-submission MoE chain | 1.5 |
| + measured matvec size cutover (small ops stay on CPU) | 4.4 |
| + MLA attention and whole layer tail on device | 7.1 |
| + whole-token frame, **1.0 command buffers/token** | 9.7 |
| + vectorized dmmv kernels (u32 + vec4 loads) | 15.6 |
| + coalesced attention grids | 19.8 |
| + four-rows-per-workgroup dmmv | 20.3 (27 tok/s marginal) |
| + working set to VRAM (activations, slots, cache) | 32.1 (63.7 tok/s marginal) |
| **+ kernel/dispatch consolidation series (17 PRs)** | **~32** (**~70 tok/s marginal**) |

Every kernel is pinned by an f64 dequantize-everything differential, and the
whole-token frame carries a same-inputs determinism probe — both ran as
30-run soaks on real hardware before merging. `LOOM_FRAME_DEBUG=1` splits a
token into per-phase timings; `LOOM_VK_MIN_BYTES` moves the CPU/GPU matvec
cutover for re-measurement on other hardware.

The decisive fix was memory placement, not kernel shape: ten kernel-level
experiments measured neutral before the profiler's uniform per-kernel deficit
identified host-visible intermediates -- every dispatch paying PCIe
first-touch latency -- as the real cost. With the working set in VRAM the
3060 passes the M5's Metal path (44 tok/s) at 63.7 tok/s marginal decode.
Backend design notes are in [`docs/GPU-BACKENDS.md`](GPU-BACKENDS.md).

## Benchmarking

```sh
loom bench          # kernel timings plus invariant checks
loom bench --check  # non-zero exit if an invariant fails; CI runs this
```

CI gates on **invariants** ("batching beats unbatched", "threads beat one
thread", "the selected path beats the alternative") rather than on wall-clock,
because a shared runner's absolute timings are noise while those relationships
are properties of the code. See [`docs/BENCHMARKING.md`](BENCHMARKING.md).

Continuous batching — where a MoE layer costs the *union* of a batch's experts
rather than the sum — is partly built: the batched decode step is in and
verified, the scheduler that would use it is not. Design, remaining work and the
measurement rules are in
[`docs/CONTINUOUS-BATCHING.md`](CONTINUOUS-BATCHING.md).
