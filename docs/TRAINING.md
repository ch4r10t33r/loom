# Training — getting started with `loomtrain`

Loom has two executables. `loom` (Zig, static binary) serves models;
**`loomtrain`** (Python, [`train/`](../train)) produces the bolt-on
artifacts the engine loads beside a frozen model:

| Artifact | Trained by | Loaded with | What it does |
|---|---|---|---|
| Pre-gate head (`.lpg`) | `loomtrain pregate-probe` + `pregate-export` | `loom node --pregate-head f.lpg` | predicts every layer's experts from layer-0 state; prefetch with lead time (measured: −16.1% wall time at 2.5 MB/s) |
| Recover-LoRA adapters (`.lra`) | `loomtrain recover-lora train` + `export` | `loom node --recover-lora f.lra` | per-expert low-rank deltas that recover aggressive-quantization loss while staying fully resident |
| Retrained routers (research) | `loomtrain router-retrain` | (experiment artifacts) | the cacheability-vs-perplexity exchange rate study |

No artifact retrains the base model, and every flag off means byte-identical
behavior — they are files beside the model, not forks of it.

## Install

**Rented GPU box with Docker** (Vast/RunPod take the image as the template;
all dependency pins are baked in):

```sh
docker run --gpus all -v $PWD:/work ghcr.io/ch4r10t33r/loom-train --help
```

**pip, from a release tag** (Colab, or any box with Python ≥ 3.10):

```sh
pip install "loomtrain @ git+https://github.com/ch4r10t33r/loom@v0.45.0#subdirectory=train"
loomtrain --help
```

Every subcommand also runs module-direct with zero install from a checkout:
`python3 -u -m loomtrain.recover_lora ...` (identical behavior; `-u`
because buffered training logs have hidden progress before).

## First run: Recover-LoRA on a small MoE (fits Colab)

Train adapters against the *exact deployed quantization* — the script
dequantizes the GGUF's expert tensors into the HF model before freezing,
because adapters must learn the deployed rounding, not a proxy:

```sh
huggingface-cli download bartowski/OLMoE-1B-7B-0924-GGUF --include "*Q4_K_M*" --local-dir .
loomtrain recover-lora train --model allenai/OLMoE-1B-7B-0924 \
    --gguf OLMoE-1B-7B-0924-Q4_K_M.gguf --rank 4 --tokens 2000000 \
    --batch 2 --seq 512 --out olmoe-rlora.pt
loomtrain recover-lora export --ckpt olmoe-rlora.pt --out olmoe.lra
```

Watch for the `first-step grad sum` line: it must be nonzero, or the
freeze mask or checkpointing ate the backward pass and the run aborts
loudly rather than training a ghost. (OLMoE validates the training path;
loom's engine serves qwen-family archs, so production `.lra` targets are
the devnet model — see the hardware note below.)

## The production run: Qwen3-30B against the devnet's Q2_K

```sh
loomtrain recover-lora train --model Qwen/Qwen3-30B-A3B \
    --gguf qwen3-30b-a3b-q2k.gguf --rank 8 --tokens 50000000 --out rlora-ckpt.pt
loomtrain recover-lora export --ckpt rlora-ckpt.pt --out qwen3-30b-q2k.lra
loom node --network devnet --recover-lora qwen3-30b-q2k.lra ...
```

Honest hardware bar: the base must be bf16 GGUF-dequant weights (~60 GB
for the 30B), so this is an 80 GB A100/H100 rental, not a 24 GB card.
The engine validates the `.lra` header against the loaded model and
refuses shape mismatches — a stale or foreign file downgrades to a loud
log line, never wrong output.

## Testing, versioning, house rules

- The test ladder (CPU byte-identity check → CI image gate → Colab
  functional run → production) is in [`train/README.md`](../train/README.md).
- One release tag versions both executables; install `loomtrain` from the
  same tag your nodes run.
- torch/transformers pins live in the container image and each module's
  docstring, not pyproject — the box's CUDA build dictates the wheel.
  The known-good transformers is `4.55.4`.
- Training-side decisions and measured results live in the
  [whitepaper decision log](../whitepaper/WHITEPAPER.md), same as
  everything else in this project.
