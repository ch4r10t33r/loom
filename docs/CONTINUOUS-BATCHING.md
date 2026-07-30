# Continuous batching

Status: increment 1 merged (`decodeBatch`), increments 2 and 3 open.

## Why

A MoE layer under a batch of `n` sequences needs the *union* of their routed
experts, not the sum. One 6 MB expert read then serves every sequence that
routed to it. This is the whitepaper's "throughput hides latency" argument, and
it is the only lever left that does not depend on kernel work: measured on
DeepSeek-V2-Lite, the expert FFN is ~45% of a token and its weights are read
once per sequence today.

It buys throughput, not latency. A single stream gets no faster, and may get
slightly slower. Size expectations accordingly, and measure aggregate tokens
per second across concurrent requests — a per-request figure will read as a
regression.

## What exists

`deepseek.decodeBatch(m, states, tokens, positions)` — merged, verified, and
not yet called by anything.

- Attention stays per sequence. Each has its own KV cache and its own position,
  so there is nothing to share.
- `expertFfnBatched` runs one expert's FFN over `n` activation vectors in a
  single pass over its weights, via `backend.matmul`. That pass is where the
  saving is; calling `denseFFN` per sequence walks the weights again each time.
- Routing is inverted per layer: expert → the sequences that chose it. Each
  unique expert runs once. The shared expert takes the whole batch.
- Falls back to `step` at `n == 1`, which keeps the recorded whole-layer GPU
  path in use where it wins.

Its test (`decodeBatch agrees with sequential step, token for token`) builds a
small MLA model in memory with `buildFixture` and requires logits and argmax to
match sequential decoding. The two sequences carry *different* tokens
deliberately: identical ones pass even if the routing inversion collapses the
batch to one sequence.

## What is left

### Increment 2 — batched `q5_0`

`backend.matmul`'s batched set is `q4_k, q6_k, q8_0`. `ffn_down_exps` is `q5_0`
in half of DeepSeek-V2-Lite's layers, and those fall back to one matvec per row
— no amortization at all for roughly 38% of an expert's bytes.

A batched `q5_0` kernel was written and rejected: every kernel in that set
quantizes activations to int8, while `matvecQ50` dequantizes exactly in f32.
`matmul` must equal `n` matvecs *exactly* — `matmul is bit-identical to repeated
matvec` enforces that with no tolerance, and caught it immediately.

Two ways to close it, neither small:

- an f32-exact batched path, which needs the raw activations `matmulRows` is not
  currently given; or
- move single-vector `q5_0` onto int8 activations, which changes decode
  numerics for every model that uses the type.

The constraint is recorded at the batched set in `ggml.zig`.

### Increment 3 — the scheduler

Today `genGgufInner` owns its `State`, runs prefill and the whole decode loop,
and `openai.zig` holds `engine_lock` for the request's lifetime. Requests
queue; they never share a forward pass.

```
Batcher owns the model and N sequence slots (N <= MAX_DECODE_BATCH)
  submit(prompt_toks, max_tokens, sink) -> Handle
    prefill runs per sequence -- different prompts cannot share a pass
    the slot then joins the decode set
  worker loop:
    collect active slots -> decodeBatch(m, states, tokens, positions)
    sample per slot, emit to that slot's sink, advance its position
    retire on EOS, max_tokens, or client disconnect
openai.zig: replace lock-for-whole-request with submit + await tokens
```

Care is needed in three places that the current code does not have to handle: a
failed generation must not strand a slot, a disconnecting client must retire
one promptly rather than at EOS, and `backend.parallelBegin`/`parallelEnd` is
currently scoped to a request and would become scoped to the batcher.

## Measuring it

Two rules, both learned the hard way on the development machine:

- **A/B two binaries alternating within one session.** Cross-run comparison is
  worthless here: the same binary moved the expert-FFN bucket from 35.6 to 68.5
  ms/token with no code change.
- **Quote the bucket, not just the total.** Attention drifts run to run and
  swamps the total; `LOOM_PROFILE=1` with `--status-secs 20` reports a rolling
  window, and the profile prints how many MoE layers took the backend block
  against the host loop — check that before believing any number.

Prefer a machine with nothing else on the GPU. The development Mac shares it
with the window server, which is most of the drift above.
