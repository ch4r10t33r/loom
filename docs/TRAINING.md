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

## First run: Recover-LoRA, end to end on a loom-served model

One-click on Colab (select the **A100** runtime, then Run all):
[train/notebooks/recover_lora_a27b.ipynb](https://colab.research.google.com/github/ch4r10t33r/loom/blob/main/train/notebooks/recover_lora_a27b.ipynb)
— it performs every step below and saves `a27b.lra` to your Drive.

Train adapters against the *exact deployed quantization* — the script
dequantizes the GGUF's expert tensors into the HF model before freezing,
because adapters must learn the deployed rounding, not a proxy:

```sh
huggingface-cli download Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF --include "*q4_k_m*" --local-dir .
loomtrain recover-lora train --model Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --gguf qwen1_5-moe-a2_7b-chat-q4_k_m.gguf --rank 4 --tokens 2000000 \
    --batch 2 --seq 512 --out a27b-rlora.pt
loomtrain recover-lora export --ckpt a27b-rlora.pt --out a27b.lra
```

The base is chosen so the loop CLOSES: Qwen1.5-MoE-A2.7B is the
smallest MoE loom's engine serves (qwen2moe arch), so the exported
`.lra` loads straight into `loom gguf run --recover-lora a27b.lra` and
the whole pipeline — dequant install, adapter training, export, engine
attach — is exercised on one model. ~29 GB bf16 for training: a Colab
Pro A100 (40 GB) or a cheap 48 GB rental; free-tier Colab cannot hold
it, and a smaller non-servable model would test only half the point.

Watch for the `first-step grad sum` line: it must be nonzero, or the
freeze mask or checkpointing ate the backward pass and the run aborts
loudly rather than training a ghost.

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
