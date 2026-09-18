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

torch/transformers/datasets/gguf are deliberately unpinned in
pyproject.toml: the box's CUDA build dictates the torch wheel; each
module's docstring records the per-run pins that worked (the house
recipe). Ops shell scripts and measurement batteries stay in
`../scripts/` — this package is only for code that trains.
