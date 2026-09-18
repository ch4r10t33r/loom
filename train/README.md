# loomtrain — loom's training plane

Python package producing the bolt-on artifacts the `loom` engine serves:
pre-gate heads (LPG1), Recover-LoRA adapters (LRA1), retrained routers,
and (next) BTX expert branches. Training is Python/PyTorch by decision
(whitepaper decision log, 2026-08); inference is the Zig `loom` binary;
the two meet only through content-addressed artifact files.

```sh
# from a rented GPU box, pinned to a loom release tag:
pip install "loomtrain @ git+https://github.com/ch4r10t33r/loom@v0.44.0#subdirectory=train"
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
