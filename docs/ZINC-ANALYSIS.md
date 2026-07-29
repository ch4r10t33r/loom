# What ZINC actually does, and what loom has to change

Analysis of [zolotukhin/zinc](https://github.com/zolotukhin/zinc) as a Metal
reference, written before porting anything, because the first attempt at
porting went the other way round — a kernel was read, techniques were copied,
none of them moved the number, and the reason only surfaced afterwards.

## The headline: it is not the kernels

Loom's Q4_K matvec, measured with submission amortized, reaches 114–116 GB/s
from 32k rows up. ZINC's own microbenchmark reports 109 GB/s at 27 MB. A
streaming-read probe puts this machine's DRAM ceiling at ~110 GB/s. **The two
kernels are at parity and both are at the memory ceiling**, so there is nothing
to win by copying kernel technique at DRAM-bound sizes.

The one place ZINC is genuinely faster is cache-resident shapes: 239 GB/s at
2.25 MB against loom's 67 GB/s at 2.4 MB. There a matvec is latency- and
occupancy-bound, and their register discipline pays — see "Kernel technique"
below. That matters for small models (TinyLlama's tensors are 6.5 MB) and not
for large ones (Mistral-7B's are 33 MB).

## The actual difference: command buffers per token

ZINC ships an environment flag, `ZINC_METAL_KERNEL_TIMING`, that wraps every
dispatch in commit+wait so it can attribute time per kernel. Running the same
model with and without it is a controlled experiment on exactly the variable
loom gets wrong:

| | ms/token | tok/s |
|---|---|---|
| ZINC, normal | 18.9 | 53.0 |
| ZINC, commit+wait per dispatch | 84.8 | 11.8 |

**4.5x.** Loom's Metal path today issues one command buffer per operation —
roughly 150 per token for a 22-layer model — which is the 84.8 ms/token regime.

Their code says the same thing explicitly. `dense_cmd_group_layers = 60` puts
up to sixty layers in a single command buffer, and the comment records the
measurements behind it:

- llama.cpp's guidance that the optimal number of command buffers per token is
  "1 or 2", and that more "can degrade the performance", because each extra one
  adds queue-scheduling latency and a GPU clock-ramp window between
  submissions.
- Raising the per-chunk layer count took cmds/step from 2.89 to 1.89 with
  byte-identical output.
- Going all the way to one chunk measured 43.87 vs 44.32 tok/s — **flat within
  noise**.

So the target is ~2 command buffers per token, and the last step from 2 to 1 is
worth nothing. The entire prize is in getting from *per-operation* down to
*per-token*.

## How they hold that shape

Four things, all of which loom currently violates:

1. **One command buffer spans the whole decode body**, with `cmd.barrier()`
   between dependent dispatches rather than a commit. The encoder is opened in
   concurrent mode so independent dispatches overlap; barriers are inserted
   only where a dispatch reads what a previous one wrote.

2. **Nothing is read back mid-token.** Weights are wired into an
   `MTLResidencySet` at load (zero-copy mmap), the KV cache lives on device,
   and activations stay in device buffers between dispatches. Any host read
   would force a commit and split the buffer.

3. **Kernel fusion cuts the dispatch count further.** 46 of their 185 shaders
   fuse two to four operations:
   `dmmv_q4k_dense_gate_up_swiglu` (two matvecs + SwiGLU in one kernel),
   `dmmv_q4k_lmhead_argmax` (output projection + argmax),
   `add_rms_norm` (residual add + norm),
   `gemma_moe_weighted_post_norm_residual`.
   Fusion is a second-order win *after* batching: it reduces dispatches within
   an already-single command buffer.

4. **Command-buffer count is a first-class metric.** Their profile logs
   `cmds=` and `commits=` per step alongside tok/s, which is why they can tell
   a 2.89 → 1.89 change from noise.

## Kernel technique, for later

Worth copying only for cache-resident shapes, where loom is 3.5x behind:

- Two rows per SIMD group (`NR0=2`), sharing one activation load.
- The Q4_K `d*sc*q - dmin*m` split: accumulate `sum(q*y)` and `sum(y)`
  separately so the min term is applied once per block, not per element.
- The 16-byte block header (`d`, `dmin`, 12 scale bytes) read as one
  `packed_uint4`.
- Scales kept in `ushort4` **registers**. Their comments are explicit that a
  local `uchar[12]` forces the array into thread-private memory and costs more
  than the technique gains — loom reproduced exactly that regression when
  porting this naively.
- Nibbles extracted four at a time from a `ushort` with one vector AND against
  `ushort4(0x000F, 0x0F00, 0x00F0, 0xF000)`.
- Full unrolling of the inner loop (`#pragma clang loop unroll(full)`).

Loom has the first three and they changed nothing at DRAM-bound sizes, which is
consistent with there being no headroom there.

## What loom has to change

The seam is the problem. `src/compute/backend.zig` exposes one-shot calls —
`matvec`, `ffnBlock`, `attnHeads` — each of which internally creates a command
buffer, dispatches, commits, waits, and copies the result back to the host.
That shape *cannot* express a per-token command buffer no matter how good the
kernels behind it are.

What is needed, in order:

1. **A frame abstraction on the seam**: `beginFrame` / `record*` / `endFrame`,
   where the recording calls encode into a shared command buffer and only
   `endFrame` commits. The engine's token loop opens one frame per token.

2. **Device-resident activations for a whole token.** Loom already has
   `act[4]` device buffers but copies host↔device around every operation.
   Those copies are what force the commits.

3. **Barriers instead of commits** between dependent dispatches, which
   `ffnBlock` already does correctly *within* a block — the pattern exists and
   needs extending to the token.

4. **A kernel for every operation on the path.** Any CPU fallback mid-token
   splits the buffer and gives back the win. This is what makes the remaining
   quant kernels (Q5_K, Q5_1) matter — not their own speed, but that their
   absence forces a split.

5. Only then, fusion.

The measured prize: 4.5x on the GPU path, against a kernel already at parity
with ZINC's and at the machine's memory ceiling.
