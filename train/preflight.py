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
        # distinct content per dataset name: identical streams train
        # identical branches, whose mixture is routing-invariant -- which
        # made the router gradient legitimately zero and masked nothing
        tag = abs(hash(a[0] if a else "x")) % 997
        def gen():
            for i in itertools.count():
                yield {"text": (f"stream {tag} weaves thread {i * (tag + 1)} ") * 40}
        return gen()
    real_datasets.load_dataset = fake_load_dataset  # covers recover_lora AND btx streams

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
    print(f"recover-lora preflight OK ({got} bytes)")

    # ---- BTX: tiny dense seed -> 2 branches -> exact merge -> router train
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from loomtrain import btx

    scfg = Qwen3Config(hidden_size=64, intermediate_size=128,
                       num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=16, vocab_size=151936,
                       max_position_embeddings=512, tie_word_embeddings=True)
    torch.manual_seed(1)
    Qwen3ForCausalLM(scfg).save_pretrained("tinyseed")
    AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat").save_pretrained("tinyseed")

    class B:
        pass
    def branch_args(i, out, ternary=False):
        b = B()
        b.seed = "tinyseed"
        b.dataset = f"synthetic-{i}"
        b.subset = None
        b.tokens = 2000
        b.lr = 5e-5
        b.seq = 128
        b.batch = 2
        b.device = "cpu"
        b.seed_val = i
        b.ternary = ternary
        b.out = out
        return b

    for i, out in enumerate(("tb0", "tb1")):
        btx.branch(branch_args(i, out))

    m = B()
    m.seed = "tinyseed"
    m.branches = ["tb0", "tb1"]
    m.datasets = ["synthetic-a", "synthetic-b"]
    m.router_tokens = 1500
    m.router_lr = 1e-3
    m.seq = 128
    m.batch = 2
    m.device = "cpu"
    m.out = "tinymerged"
    btx.merge(m)

    # the merged model must load as a qwen3moe and run a forward
    from transformers import AutoModelForCausalLM as AM
    mm = AM.from_pretrained("tinymerged")
    assert mm.config.model_type == "qwen3_moe" and mm.config.num_experts == 2
    ids = torch.randint(0, 1000, (1, 16))
    mm(input_ids=ids)
    # exact-merge invariant: the merged trunk equals the seed's trunk
    seed_m = AM.from_pretrained("tinyseed")
    a = mm.model.layers[0].self_attn.q_proj.weight
    bq = seed_m.model.layers[0].self_attn.q_proj.weight
    assert torch.equal(a, bq), "trunk drifted -- merge is not exact"

    # ---- BTX 3-branch merge: the first TRULY sparse routing (top-2-of-3
    # selects, unlike top-2-of-2 which always uses both experts)
    btx.branch(branch_args(2, "tb2"))
    m3 = B()
    m3.seed = "tinyseed"
    m3.branches = ["tb0", "tb1", "tb2"]
    m3.datasets = ["synthetic-a", "synthetic-b", "synthetic-c"]
    m3.router_tokens = 1500
    m3.router_lr = 1e-3
    m3.seq = 128
    m3.batch = 2
    m3.device = "cpu"
    m3.out = "tinymerged3"
    btx.merge(m3)
    mm3 = AM.from_pretrained("tinymerged3")
    assert mm3.config.num_experts == 3 and mm3.config.num_experts_per_tok == 2
    mm3(input_ids=ids)

    # ---- BTX ternary arm: QAT branches -> saved weights ternary-valued ->
    # merge still composes and forward-runs
    for i, out in enumerate(("tt0", "tt1")):
        btx.branch(branch_args(i, out, ternary=True))
    tb = AM.from_pretrained("tt0")
    w = tb.model.layers[0].mlp.gate_proj.weight
    uniq = max(int(torch.unique(row).numel()) for row in w)
    assert uniq <= 3, f"ternary save not snapped: {uniq} distinct values in a row"
    assert not torch.equal(w, seed_m.model.layers[0].mlp.gate_proj.weight), \
        "ternary branch FFN never moved off the seed"
    mt = m
    mt.branches = ["tt0", "tt1"]
    mt.out = "tinymerged-t"
    btx.merge(mt)
    mmt = AM.from_pretrained("tinymerged-t")
    ew = mmt.model.layers[0].mlp.experts[0].gate_proj.weight
    uniq = max(int(torch.unique(row).numel()) for row in ew)
    assert uniq <= 3, "merged expert lost ternary values"
    mmt(input_ids=ids)
    print("PREFLIGHT OK: recover-lora + btx pipelines both pass")


if __name__ == "__main__":
    main()
