# Benchmarking

```sh
loom bench            # table + invariants
loom bench --json     # machine-readable
loom bench --check    # non-zero exit if an invariant fails (this is what CI runs)
```

## Two kinds of number, and why only one is safe to gate on

**Absolute timings** are what you quote in a release note. They are as much a
property of the machine as of the code, so comparing them across hosts means
nothing, and comparing them across commits is only meaningful on the same
host, idle, on mains power. A shared CI runner has none of those properties.

**Invariants** are relationships that hold whatever the machine's speed:

| invariant | why it holds |
|---|---|
| batching beats unbatched | one unpacked weight serves several tokens |
| threads beat one thread | rows split with no shared state |
| the selected path beats the alternative | a backend that loses should not be chosen |

Those survive a noisy runner, so CI gates on them and merely reports the
timings.

They still need a threshold chosen for the worst environment they run in,
not the best. "Batching beats unbatched" is a ratio of 0.70 on eight cores
here and 0.96 on a single-core CI runner, where the batched path's larger
working set costs back most of what the shared weight unpack saves. A gate set
from the local number failed on the runner.

The threshold that works is 0.98, and it still catches what the invariant
exists for: when batching is not wired into a path, `matmul` falls through to
calling `matvec` n times, which is literally the other side of the
comparison — so the ratio is 1.00, not 0.96. The gate separates "batching is
less effective on this machine" from "batching is not happening".

This is not theoretical. The two performance bugs found in this codebase were
both batched prefill wired into only one of the two prefill paths: the
first measurement round showed no improvement at all, twice, and the cause was
`runEngine` still on the per-token loop while `runStoreWith` had the new one.
An absolute-time threshold on a shared runner would not reliably have caught
that. `batching beats unbatched` catches it every time.

## Reading the table

```
  kernel                                 ms
  matvec/q4_k                         0.126
  matvec/q5_k                         0.639     <- still on the dequant path
  matvec/q8_0                         0.098
  matmul/q4_k x8                      0.709
  matvec/q4_k x8 (unbatched)          1.032
```

The gap between `q4_k` and `q5_k` above is not noise: `q4_k`, `q6_k` and
`q8_0` have int8-activation kernels and the rest still dequantize to f32
first. The benchmark is how that became visible.

## What this deliberately does not do

It does not compare against llama.cpp. ZINC, which is the closest prior art,
publishes headline numbers from a strict server-vs-server contract — same
GGUF, same prompts, same warmup and run counts, provenance stamped from `git
describe --dirty` — because anything looser produces numbers that cannot be
reproduced or defended. That bar is worth meeting before making a comparative
claim, and it needs dedicated idle hardware rather than a laptop that is also
running a browser.

Until then this measures loom against itself, which is enough to catch
regressions and to know where the time goes.

## Adding a benchmark

Add the timing to `src/bench.zig` and, where there is a relationship that must
hold, a `check(...)` for it. Prefer the invariant: a number that only ever
gets printed will be ignored, while one that fails a build gets fixed.

## The expert FFN runs 4.4x slower in the engine than in isolation

Before building an MLA + MoE GPU path on the assumption that `expert ffn` is
compute-bound, the assumption was checked. It does not hold, and the GPU is not
the lever.

`loom bench --ffn` runs exactly the shape DeepSeek-V2-Lite uses — dim 2048,
moe_ffn 1408, gate/up Q4_K, down Q5_1, 5.16 MB per expert — over a ring of 64
distinct experts so nothing sits in cache:

| | ms per expert FFN | implied |
|---|---|---|
| bench, 8 threads | 0.254 | 21.3 GB/s |
| bench, 1 thread | 1.150 | 4.7 GB/s |
| **in-engine, 8 threads** | **1.08** | **4.8 GB/s** |

The in-engine figure is a whisker from the bench's single-threaded number.
Threading in the engine is not absent (forcing `--threads 1` moves `expert ffn`
from 168.3 to 276.2 ms), but that is 1.64x where the same kernels on the same
shapes get 4.52x.

So roughly 4x of the expert FFN is being lost to something that is neither the
kernel nor the quantization nor the thread count. Candidates not yet
distinguished:

- **TLB pressure.** The bench indexes a 330 MB slab (~20k pages); the engine
  indexes an 8 GB LRU slab at random slot offsets (~500k pages). Random access
  across that many pages defeats the page-walk caches in a way the bench cannot
  reproduce.
- **Contention with prefetch threads.** `Source.prefetch` spawns a thread per
  missing shard for each MoE layer, and with ~24 disk misses per token those
  are live for much of the forward pass, competing with the kernel pool for
  four cores.
- **Residency.** With `--mmap-weights` the costs move between buckets rather
  than shrinking (`get` 219 -> 316 ms and `ffn` 208 -> 107 ms), which says the
  page-touch cost is real and merely attributed to whoever reads the bytes
  first, not that either arrangement avoids it.

This matters more than the GPU work it displaced. `expert ffn` is ~40% of a
token, so recovering the isolation number would be worth about 2x overall —
against an estimated 1.3x for moving the same work to Metal, which would read
the same pages from the same unified memory and inherit whichever of the above
is responsible.
