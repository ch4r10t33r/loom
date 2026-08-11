#!/usr/bin/env python3
"""Export a pre-gate head checkpoint (pregate-probe.py --head-out) to the
LPG1 format src/gguf/pregate.zig loads: "LPG1", u32 hid/width/n_pred/
n_expert, then f32 w1, b1, w2, b2 in PyTorch Linear layout.

Usage: pregate-export.py pregate-head-qwen3.pt qwen3.lpg --n-expert 128
"""
import argparse
import struct

import torch

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("out")
ap.add_argument("--n-expert", type=int, required=True,
                help="experts per layer (128 for Qwen3-30B-A3B, 64 for OLMoE)")
a = ap.parse_args()

sd = torch.load(a.checkpoint, weights_only=True, map_location="cpu")
w1, b1, w2, b2 = sd["0.weight"], sd["0.bias"], sd["2.weight"], sd["2.bias"]
width, hid = w1.shape
out_dim = w2.shape[0]
if out_dim % a.n_expert:
    raise SystemExit(f"head output {out_dim} not divisible by n_expert {a.n_expert}")
n_pred = out_dim // a.n_expert

with open(a.out, "wb") as f:
    f.write(b"LPG1")
    f.write(struct.pack("<4I", hid, width, n_pred, a.n_expert))
    for t in (w1, b1, w2, b2):
        f.write(t.float().contiguous().numpy().tobytes())

print(f"wrote {a.out}: hid={hid} width={width} n_pred={n_pred} n_expert={a.n_expert}")
