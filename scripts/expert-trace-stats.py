#!/usr/bin/env python3
"""Activation-shape statistics from a LOOM_EXPERT_TRACE file (research
lever 9, stage 0).

The trace is one line per routed MoE layer, "<layer> <e0> <e1> ...";
token boundaries are recovered from the layer index wrapping. Reports the
four properties the lever cares about, each mapped to the loom cost it
drives:

  coverage    -> how well a static pinned set can work (Zipf skew)
  stickiness  -> how well an LRU cache can work (temporal locality)
  window union-> what a batch/draft window pays (DSD & batching bill)
  prediction  -> how well a cross-layer prefetcher can work
  replay      -> the bottom line: fetch-MB/token under LRU or pinned
                 caches of swept sizes

Usage: expert-trace-stats.py TRACE [--expert-mb MB]
"""
import argparse
import sys
from collections import Counter, defaultdict


def parse(path):
    """-> list of tokens, each {layer: [expert,...]}."""
    tokens, cur, prev_li = [], {}, None
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            li, experts = int(parts[0]), [int(x) for x in parts[1:]]
            if prev_li is not None and li <= prev_li and cur:
                tokens.append(cur)
                cur = {}
            cur[li] = experts
            prev_li = li
    if cur:
        tokens.append(cur)
    return tokens


def coverage(tokens, layers, n_expert):
    """Fraction of all activations covered by the hottest X% of experts."""
    fracs = [0.10, 0.25, 0.50]
    out = {f: [] for f in fracs}
    zipf_ratio = []
    for li in layers:
        c = Counter()
        for t in tokens:
            c.update(t.get(li, []))
        total = sum(c.values())
        if not total:
            continue
        by_freq = [n for _, n in c.most_common()]
        for f in fracs:
            k = max(1, int(n_expert * f))
            out[f].append(sum(by_freq[:k]) / total)
        # top-decile mass over uniform mass: 1.0 == flat, higher == skewed
        k10 = max(1, n_expert // 10)
        zipf_ratio.append((sum(by_freq[:k10]) / total) / (k10 / n_expert))
    return {f: sum(v) / len(v) for f, v in out.items()}, sum(zipf_ratio) / len(zipf_ratio)


def stickiness(tokens, layers):
    """Mean |sel_t ∩ sel_{t+1}| / n_used, per layer, averaged."""
    per_layer = []
    for li in layers:
        overlaps = []
        for a, b in zip(tokens, tokens[1:]):
            sa, sb = set(a.get(li, [])), set(b.get(li, []))
            if sa and sb:
                overlaps.append(len(sa & sb) / len(sa))
        if overlaps:
            per_layer.append(sum(overlaps) / len(overlaps))
    return sum(per_layer) / len(per_layer), min(per_layer), max(per_layer)


def window_union(tokens, layers, n_used):
    """Mean expert-union size over a trailing window of W tokens."""
    out = {}
    for w in (1, 2, 4, 8, 16, 32):
        sizes = []
        for li in layers:
            for i in range(0, len(tokens) - w, w):
                u = set()
                for t in tokens[i : i + w]:
                    u.update(t.get(li, []))
                sizes.append(len(u))
        out[w] = (sum(sizes) / len(sizes), w * n_used)
    return out


def predictability(tokens, layers, n_expert, n_used):
    """Predict layer l+1's selection from layer l's, co-occurrence counts
    trained on the first half of the trace, evaluated on the second.
    Baseline: always predict the training half's globally hottest n_used."""
    half = len(tokens) // 2
    train, test = tokens[:half], tokens[half:]
    hits = base_hits = total = 0
    for li in layers[:-1]:
        lj = li + 1
        if lj not in layers:
            continue
        co = defaultdict(Counter)
        marg = Counter()
        for t in train:
            for e in t.get(li, []):
                co[e].update(t.get(lj, []))
            marg.update(t.get(lj, []))
        base_pred = set(e for e, _ in marg.most_common(n_used))
        for t in test:
            actual = set(t.get(lj, []))
            if not actual:
                continue
            score = Counter()
            for e in t.get(li, []):
                score.update(co[e])
            pred = set(e for e, _ in score.most_common(n_used))
            hits += len(pred & actual)
            base_hits += len(base_pred & actual)
            total += len(actual)
    return hits / total, base_hits / total


def replay(tokens, layers, n_expert, expert_mb, budgets):
    """Fetch-MB/token under per-layer caches of the swept sizes.
    LRU: least-recently-used eviction. PIN: static hottest-from-first-half."""
    half = len(tokens) // 2
    out = {}
    for frac in budgets:
        cap = max(1, int(n_expert * frac))
        lru_miss = pin_miss = acts = 0
        for li in layers:
            # LRU over the whole trace
            order = []  # most recent last
            for t in tokens:
                for e in t.get(li, []):
                    acts += 1
                    if e in order:
                        order.remove(e)
                    else:
                        lru_miss += 1
                        if len(order) >= cap:
                            order.pop(0)
                    order.append(e)
            # static pin: hottest in first half, replay second half
            c = Counter()
            for t in tokens[:half]:
                c.update(t.get(li, []))
            pinned = set(e for e, _ in c.most_common(cap))
            for t in tokens[half:]:
                for e in t.get(li, []):
                    if e not in pinned:
                        pin_miss += 1
        n_tok = len(tokens)
        out[frac] = (
            lru_miss / n_tok * expert_mb,
            pin_miss / (n_tok - half) * expert_mb,
            lru_miss / acts,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--expert-mb", type=float, default=1.5,
                    help="bytes fetched per expert miss, in MB (default 1.5 ~ Qwen3-30B Q2_K)")
    args = ap.parse_args()

    tokens = parse(args.trace)
    # The batched prefill path routes layer-major (all tokens through layer L
    # before layer L+1), which the boundary heuristic shreds into one-layer
    # fragments; keep only complete decode tokens, where every property below
    # is well-defined. Prefill's cost is the batch union, measured separately.
    all_layers = sorted({li for t in tokens for li in t})
    n_frag = sum(1 for t in tokens if len(t) < len(all_layers))
    tokens = [t for t in tokens if len(t) == len(all_layers)]
    if n_frag:
        print(f"(dropped {n_frag} prefill fragments; {len(tokens)} complete decode tokens kept)")
    if len(tokens) < 16:
        sys.exit(f"only {len(tokens)} tokens in trace; need more to say anything")
    layers = sorted({li for t in tokens for li in t})
    n_used = max(len(v) for t in tokens for v in t.values())
    n_expert = max(e for t in tokens for v in t.values() for e in v) + 1

    print(f"trace: {len(tokens)} tokens, {len(layers)} MoE layers, "
          f"{n_expert} experts inferred, top-{n_used}, {args.expert_mb} MB/expert")

    cov, zr = coverage(tokens, layers, n_expert)
    print("\ncoverage (pinning potential; flat routing would equal the fraction):")
    for f, v in cov.items():
        print(f"  hottest {int(f*100):>2}% of experts serve {v*100:5.1f}% of activations")
    print(f"  top-decile skew ratio {zr:.2f}x over uniform")

    mean_s, min_s, max_s = stickiness(tokens, layers)
    print(f"\nstickiness (LRU potential): adjacent tokens reuse "
          f"{mean_s*100:.1f}% of experts (layer range {min_s*100:.0f}-{max_s*100:.0f}%)")

    print("\nwindow union (batch/DSD bill; union vs worst-case W*top-k):")
    for w, (u, worst) in window_union(tokens, layers, n_used).items():
        print(f"  W={w:>2}: union {u:5.1f} of {worst} worst-case ({u/worst*100:4.1f}%)")

    hit, base = predictability(tokens, layers, n_expert, n_used)
    print(f"\ncross-layer prediction (prefetch potential): layer l predicts "
          f"{hit*100:.1f}% of layer l+1's experts (hot-set baseline {base*100:.1f}%)")

    print("\nreplay (fetch-MB/token at cache budget, per-layer caches):")
    for frac, (lru_mb, pin_mb, miss) in replay(
            tokens, layers, n_expert, args.expert_mb, (0.125, 0.25, 0.5)).items():
        print(f"  budget {int(frac*100):>2}% of experts: "
              f"LRU {lru_mb:7.2f} MB/tok ({miss*100:4.1f}% miss)   "
              f"pinned {pin_mb:7.2f} MB/tok")


if __name__ == "__main__":
    main()
