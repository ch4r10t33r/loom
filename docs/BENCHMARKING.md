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

This is not theoretical. The two performance bugs found in this codebase were
both *batched prefill wired into only one of the two prefill paths* — the
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
