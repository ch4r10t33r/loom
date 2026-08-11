#!/usr/bin/env python3
"""Tier-3 feasibility probe (research lever 9): are a trained MoE's experts
low-rank around a shared core?

The tier-3 architecture stores a resident shared backbone plus routed
experts as low-rank deltas, shrinking a fetch from ~1.5 MB to tens of KB.
Whether an EXISTING model can be converted to that shape (factorize, then
heal by distillation) hinges on one measurable property: how much of each
expert's energy lies in the shared mean plus a low-rank residual. This
probe answers it with linear algebra only -- no training, no model
instantiation, just safetensors reads and SVDs.

Reports, per sampled layer and projection: the fraction of total expert
energy carried by the cross-expert mean, and the cumulative residual
energy captured at rank r for r in RANKS, averaged over sampled experts.

Usage: expert-spectrum.py SNAPSHOT_DIR [--layers 0,12,24,36,47]
       [--experts 16] [--device cuda]
"""
import argparse
import json
import os
from collections import defaultdict

import torch
from safetensors import safe_open

RANKS = (4, 8, 16, 32, 64)
PROJS = ("gate_proj", "up_proj", "down_proj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--layers", default="0,12,24,36,47")
    ap.add_argument("--experts", type=int, default=16,
                    help="experts to SVD per (layer, proj); mean uses all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    idx = json.load(open(os.path.join(a.snapshot, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    handles = {}

    def get(name):
        shard = wmap[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(a.snapshot, shard),
                                       framework="pt", device="cpu")
        return handles[shard].get_tensor(name)

    n_expert = 0
    while f"model.layers.{layers[0]}.mlp.experts.{n_expert}.gate_proj.weight" in wmap:
        n_expert += 1
    print(f"{n_expert} experts/layer; layers {layers}; "
          f"SVD sample {a.experts}/layer/proj on {a.device}")

    agg = defaultdict(list)  # (proj, rank) -> energy fractions
    shared = defaultdict(list)  # proj -> mean-energy fraction
    for li in layers:
        for proj in PROJS:
            ws = [get(f"model.layers.{li}.mlp.experts.{e}.{proj}.weight").float()
                  for e in range(n_expert)]
            stack = torch.stack(ws)
            mean = stack.mean(0)
            total = (stack ** 2).sum().item()
            resid = stack - mean
            resid_total = (resid ** 2).sum().item()
            shared[proj].append(1.0 - resid_total / total)
            step = max(1, n_expert // a.experts)
            for e in range(0, n_expert, step):
                r = resid[e].to(a.device)
                sv = torch.linalg.svdvals(r).cpu()
                e2 = sv ** 2
                cum = torch.cumsum(e2, 0) / e2.sum()
                for rank in RANKS:
                    agg[(proj, rank)].append(cum[min(rank, len(cum)) - 1].item())
            del stack, resid, ws
        print(f"layer {li} done")

    print(f"\n== shared core: fraction of expert energy in the cross-expert mean ==")
    for proj in PROJS:
        v = shared[proj]
        print(f"  {proj}: {sum(v)/len(v)*100:5.1f}%")
    print(f"\n== residual energy captured at rank r (mean over sampled experts) ==")
    print(f"{'proj':>10} " + " ".join(f"r={r:>3}" for r in RANKS))
    for proj in PROJS:
        row = " ".join(f"{sum(agg[(proj, r)])/len(agg[(proj, r)])*100:4.1f}%" for r in RANKS)
        print(f"{proj:>10} {row}")
    full_rank = min(get(f"model.layers.{layers[0]}.mlp.experts.0.gate_proj.weight").shape)
    print(f"\n(full rank {full_rank}; a viable tier-3 conversion wants high shared "
          f"fraction and steep residual capture at small r)")


if __name__ == "__main__":
    main()
