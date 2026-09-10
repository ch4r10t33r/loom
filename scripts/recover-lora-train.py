#!/usr/bin/env python3
"""Recover-LoRA training: per-expert low-rank adapters that recover the
quality lost to aggressive weight quantization (recipe generalized from
Edge0, github.com/Edge0-AI/Edge0), for loom's --recover-lora flag.

The honest core of the recipe: the adapters must be trained against the
EXACT quantized weights the devnet serves, not a proxy. So this script
reads the deployed GGUF, dequantizes each expert's Q2_K/Q3_K/... tensors
with gguf-py, and installs those bytes into the HF model's expert Linears
before freezing them. bnb-4bit is NOT a stand-in -- a different rounding
gives the adapters a different error to learn.

Stages:
  train:   python3 -u recover-lora-train.py train \
             --model Qwen/Qwen3-30B-A3B --gguf qwen3-30b-a3b-q2k.gguf \
             --rank 8 --tokens 50000000 --out rlora-ckpt.pt
  export:  python3 recover-lora-train.py export \
             --ckpt rlora-ckpt.pt --out qwen3-30b-q2k.lra

LRA1 format (matches src/gguf/recover_lora.zig): "LRA1", u32 n_layers,
n_expert, rank, dim, ffn; then f16 per (layer, expert):
  Ag[r*dim] Bg[ffn*r] Au[r*dim] Bu[ffn*r] Ad[r*ffn] Bd[dim*r]
PyTorch Linear layout (row-major [out, in]); the alpha/r scale is baked
into B at export so the engine applies plain B(Ax).

House recipe (learned the hard way, see decision log 2026-08-12):
non-reentrant gradient checkpointing + first-step grad sanity print;
timestamps on every log line; run with python3 -u; eval/export only from
the FINAL checkpoint.
"""

import argparse
import struct
import sys
import time


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- GGUF expert dequant ----------------------------------------------------

def load_gguf_experts(gguf_path):
    """-> {(layer, proj): np.ndarray [n_expert, out, in] fp32} for
    proj in gate/up/down, dequantized from whatever quant the file uses."""
    import numpy as np
    from gguf import GGUFReader
    from gguf.quants import dequantize

    r = GGUFReader(gguf_path)
    out = {}
    for t in r.tensors:
        name = t.name  # e.g. blk.7.ffn_gate_exps.weight
        if ".ffn_" not in name or "_exps.weight" not in name:
            continue
        layer = int(name.split(".")[1])
        proj = name.split(".ffn_")[1].split("_exps")[0]  # gate|up|down
        # gguf-py returns data in ggml layout; dequantize -> fp32 with shape
        # reversed vs ne: [n_expert, out, in] for a 3D expert tensor
        w = dequantize(t.data, t.tensor_type).reshape(
            [int(d) for d in reversed(t.shape)]
        )
        out[(layer, proj)] = np.ascontiguousarray(w, dtype=np.float32)
    if not out:
        raise SystemExit(f"no expert tensors found in {gguf_path}")
    log(f"dequantized {len(out)} expert tensor groups from {gguf_path}")
    return out


# ---- LoRA wrapper -----------------------------------------------------------

def build(args):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    log(f"loading {args.model} (bf16, low_cpu_mem_usage)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    tok = AutoTokenizer.from_pretrained(args.model)

    experts = load_gguf_experts(args.gguf)

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int, alpha: float):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False
            in_f, out_f = base.in_features, base.out_features
            self.A = nn.Parameter(torch.randn(rank, in_f, dtype=torch.float32) * 0.02)
            self.B = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.float32))
            self.scale = alpha / rank

        def forward(self, x):
            y = self.base(x)
            d = (x.to(torch.float32) @ self.A.T) @ self.B.T
            return y + (self.scale * d).to(y.dtype)

    # install dequantized-GGUF weights, then wrap with LoRA
    n_wrapped = 0
    for li, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "experts"):
            continue
        for ei, ex in enumerate(mlp.experts):
            for proj, attr in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
                lin = getattr(ex, attr)
                w = experts[(li, proj)][ei]
                assert w.shape == tuple(lin.weight.shape), (
                    f"layer {li} expert {ei} {proj}: gguf {w.shape} vs hf {tuple(lin.weight.shape)}"
                )
                with torch.no_grad():
                    lin.weight.copy_(torch.from_numpy(w))
                setattr(ex, attr, LoRALinear(lin, args.rank, args.alpha))
                n_wrapped += 1
    for name, p in model.named_parameters():
        p.requires_grad = (".A" in name or ".B" in name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"wrapped {n_wrapped} expert projections; trainable {trainable/1e6:.1f}M params")
    return model, tok


def train(args):
    import torch
    from datasets import load_dataset

    model, tok = build(args)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    seen, step_i, buf = 0, 0, []
    t0 = time.time()
    for row in ds:
        ids = tok(row["text"], truncation=True, max_length=args.seq)["input_ids"]
        if len(ids) < 32:
            continue
        buf.append(ids)
        if len(buf) < args.batch:
            continue
        maxlen = max(len(x) for x in buf)
        input_ids = torch.full((len(buf), maxlen), tok.pad_token_id or 0,
                               dtype=torch.long, device="cuda")
        for i, x in enumerate(buf):
            input_ids[i, : len(x)] = torch.tensor(x)
        labels = input_ids.clone()
        labels[input_ids == (tok.pad_token_id or 0)] = -100
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        if step_i == 0:
            # gradient-flow sanity: silence here means checkpointing or the
            # freeze mask ate the backward -- abort loudly, don't train a ghost
            gn = sum(
                p.grad.abs().sum().item()
                for p in model.parameters() if p.requires_grad and p.grad is not None
            )
            log(f"first-step grad sum {gn:.3e}")
            assert gn > 0, "no gradient reached the adapters"
        opt.step()
        opt.zero_grad(set_to_none=True)
        seen += int((labels != -100).sum())
        step_i += 1
        buf = []
        if step_i % 20 == 0:
            log(f"step {step_i} loss {out.loss.item():.4f} tokens {seen/1e6:.1f}M "
                f"({seen/max(time.time()-t0,1):.0f} tok/s)")
        if step_i % 500 == 0 or seen >= args.tokens:
            save(model, args)
        if seen >= args.tokens:
            break
    save(model, args)
    log("train done")


def save(model, args):
    import torch
    sd = {k: v for k, v in model.state_dict().items() if ".A" in k or ".B" in k}
    torch.save({"lora": sd, "rank": args.rank, "alpha": args.alpha}, args.out)
    log(f"checkpoint -> {args.out} ({len(sd)} tensors)")


def export(args):
    """checkpoint -> LRA1, alpha/rank baked into B."""
    import numpy as np
    import torch

    ck = torch.load(args.ckpt, map_location="cpu")
    sd, rank, alpha = ck["lora"], ck["rank"], ck["alpha"]
    scale = alpha / rank

    # keys look like model.layers.{L}.mlp.experts.{E}.{proj}_proj.A
    layers, ex_max = set(), 0
    for k in sd:
        parts = k.split(".")
        layers.add(int(parts[2]))
        ex_max = max(ex_max, int(parts[5]))
    n_layers, n_expert = max(layers) + 1, ex_max + 1
    a_g = sd[f"model.layers.{min(layers)}.mlp.experts.0.gate_proj.A"]
    dim = a_g.shape[1]
    ffn = sd[f"model.layers.{min(layers)}.mlp.experts.0.gate_proj.B"].shape[0]
    log(f"export: {n_layers} layers x {n_expert} experts, rank {rank}, dim {dim}, ffn {ffn}")

    with open(args.out, "wb") as f:
        f.write(b"LRA1")
        f.write(struct.pack("<5I", n_layers, n_expert, rank, dim, ffn))
        for li in range(n_layers):
            for ei in range(n_expert):
                p = f"model.layers.{li}.mlp.experts.{ei}"
                for proj, indim in (("gate", dim), ("up", dim), ("down", ffn)):
                    A = sd[f"{p}.{proj}_proj.A"].float().numpy()
                    B = (sd[f"{p}.{proj}_proj.B"].float() * scale).numpy()
                    assert A.shape == (rank, indim)
                    f.write(np.ascontiguousarray(A, dtype=np.float16).tobytes())
                    f.write(np.ascontiguousarray(B, dtype=np.float16).tobytes())
    log(f"wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tp = sub.add_parser("train")
    tp.add_argument("--model", required=True)
    tp.add_argument("--gguf", required=True)
    tp.add_argument("--rank", type=int, default=8)
    tp.add_argument("--alpha", type=float, default=16.0)
    tp.add_argument("--tokens", type=int, default=50_000_000)
    tp.add_argument("--lr", type=float, default=2e-4)
    tp.add_argument("--seq", type=int, default=1024)
    tp.add_argument("--batch", type=int, default=4)
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--out", default="rlora-ckpt.pt")
    ep = sub.add_parser("export")
    ep.add_argument("--ckpt", required=True)
    ep.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    else:
        export(a)
