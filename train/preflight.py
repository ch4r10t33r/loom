#!/usr/bin/env python3
"""Pre-flight: execute the ENTIRE recover-lora pipeline on a tiny random
qwen2moe model on CPU, before any notebook or GPU run is handed to a
human. Catches device-placement bugs, HF-module-shape drift, GGUF dequant
issues, and export-format breakage in ~1 minute with no GPU and no
dataset download (synthetic text).

Born from the 2026-09-18 Colab session: two first-contact failures
(a nonexistent GGUF repo id, LoRA params created on cpu against a cuda
model) that this script would have caught for $0. Rule: run this after
ANY change to recover_lora.py; hand nothing to a GPU that failed here.

  python3 -u train/preflight.py   # needs torch+transformers+gguf (cpu ok)
"""
import itertools
import os
import sys
import tempfile

import numpy as np
import torch
import datasets as real_datasets
from transformers import AutoTokenizer, Qwen2MoeConfig, Qwen2MoeForCausalLM

import gguf as gg
from loomtrain import recover_lora as rl


def main():
    work = tempfile.mkdtemp(prefix="loomtrain-preflight-")
    os.chdir(work)
    print("preflight workdir:", work)

    cfg = Qwen2MoeConfig(hidden_size=64, intermediate_size=128,
                         moe_intermediate_size=48,
                         shared_expert_intermediate_size=96,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, num_experts=8,
                         num_experts_per_tok=2, vocab_size=151936,
                         decoder_sparse_step=1, max_position_embeddings=512)
    torch.manual_seed(0)
    m = Qwen2MoeForCausalLM(cfg)
    m.save_pretrained("tinymoe")
    AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat").save_pretrained("tinymoe")

    w = gg.GGUFWriter("tinymoe.gguf", "qwen2moe")
    w.add_block_count(2)
    for li, layer in enumerate(m.model.layers):
        for proj, name in (("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"), ("down_proj", "ffn_down_exps")):
            stack = torch.stack([getattr(e, proj).weight.data for e in layer.mlp.experts]).float().numpy()
            w.add_tensor(f"blk.{li}.{name}.weight", stack, raw_dtype=gg.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    # synthetic stream: the dry-run must not depend on dataset connectivity
    def fake_load_dataset(*a, **k):
        def gen():
            for i in itertools.count():
                yield {"text": (f"the loom weaves expert {i} ") * 40}
        return gen()
    real_datasets.load_dataset = fake_load_dataset

    class A:
        pass
    a = A()
    a.model = "tinymoe"
    a.gguf = "tinymoe.gguf"
    a.rank = 2
    a.alpha = 4.0
    a.tokens = 3000
    a.lr = 2e-4
    a.seq = 128
    a.batch = 2
    a.seed = 0
    a.device = "cpu"
    a.out = "tiny-rlora.pt"
    rl.train(a)
    a.ckpt = "tiny-rlora.pt"
    a.out = "tiny.lra"
    rl.export(a)

    # the exported size must match the LRA1 formula exactly
    nl, ne, r, dim, ffn = 2, 8, 2, 64, 48
    want = 24 + nl * ne * 3 * r * (dim + ffn) * 2
    got = os.path.getsize("tiny.lra")
    assert got == want, f"LRA1 size {got} != formula {want}"
    print(f"PREFLIGHT OK: trained, exported, size matches formula ({got} bytes)")


if __name__ == "__main__":
    main()
