# loomtrain — loom's training plane

Python package producing the bolt-on artifacts the `loom` engine serves:
pre-gate heads (LPG1), Recover-LoRA adapters (LRA1), retrained routers,
and (next) BTX expert branches. Training is Python/PyTorch by decision
(whitepaper decision log, 2026-08); inference is the Zig `loom` binary;
the two meet only through content-addressed artifact files.

```sh
# from a rented GPU box, pinned to a loom release tag:
pip install "loomtrain @ git+https://github.com/ch4r10t33r/loom@v0.45.1#subdirectory=train"
loomtrain recover-lora train --model Qwen/Qwen3-30B-A3B --gguf model-q2k.gguf ...
# or, zero-install, module-direct (identical behavior):
python3 -u -m loomtrain.recover_lora train ...
```

On boxes with Docker (Vast/RunPod templates take the image directly):

```sh
docker run --gpus all -v $PWD:/work ghcr.io/ch4r10t33r/loom-train \
  recover-lora train --model Qwen/Qwen3-30B-A3B --gguf model-q2k.gguf ...
```

The image (train/Dockerfile, published by CI on main) bakes in the house
pins -- transformers 4.55.4 and friends -- so a fresh rental skips the
dependency archaeology entirely.

torch/transformers/datasets/gguf are deliberately unpinned in
pyproject.toml: the box's CUDA build dictates the torch wheel; each
module's docstring records the per-run pins that worked (the house
recipe). Ops shell scripts and measurement batteries stay in
`../scripts/` — this package is only for code that trains.

## Testing the package

Four tiers, cheapest first:

1. **CPU, anywhere** (no GPU, ~2 min): install and re-export the released
   pre-gate checkpoint; the output must be byte-identical to the deployed
   artifact (both are release assets on v0.40.3):
   ```sh
   loomtrain pregate-export pregate-head-qwen3.pt check.lpg --n-expert 128
   cmp pregate-qwen3.lpg check.lpg   # byte-identical or the package is broken
   ```
2. **CI, automatic**: every change under train/ rebuilds the container
   image and imports every module under the exact pinned stack.
3. **Colab Pro / small rental, end-to-end test**: the pip install path is
   exactly what Colab exercises -- no Docker needed. Base model:
   **Qwen1.5-MoE-A2.7B** -- deliberately the smallest MoE loom's engine
   SERVES (qwen2moe arch), so the exported .lra closes the loop through
   `loom gguf run --recover-lora`. ~29 GB bf16: an A100-40GB runtime
   (Colab Pro) or a 48 GB rental; free-tier T4 cannot hold it.
   ```
   !pip install "loomtrain @ git+https://github.com/ch4r10t33r/loom@v0.45.1#subdirectory=train"
   !huggingface-cli download RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf --include "*Q4_K_M*" --local-dir .
   !python -u -m loomtrain.recover_lora train --model Qwen/Qwen1.5-MoE-A2.7B-Chat        --gguf Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf --rank 4 --tokens 2000000 --batch 2 --seq 512        --out a27b-rlora.pt
   !python -m loomtrain.recover_lora export --ckpt a27b-rlora.pt --out a27b.lra
   ```
   This validates the full path: GGUF dequant install, adapter
   gradients, LRA1 export, and the engine attach on a model loom
   actually serves.
4. **The real run** (rented GPU): Qwen3-30B-A3B against the devnet's Q2_K
   GGUF needs the base in bf16 (~60 GB) -- an 80 GB A100/H100 rental, not
   a 24 GB card. Rank {4,8,16} curve, then the devnet A/B with
   `loom node --recover-lora qwen3-30b-q2k.lra`.
