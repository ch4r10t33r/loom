#!/usr/bin/env python3
"""Stage 1 of research lever 9: the systems-side ceiling, by replay.

Takes a LOOM_EXPERT_TRACE file and answers two questions the stage-0
shape statistics cannot: which CACHE policy minimizes fetch traffic on
the current model, and how much of the remaining fetch latency a
one-layer-lookahead PREFETCHER can hide at a given bandwidth. Together
they bound what systems work alone extracts -- the bar any retrained
router (stage 2) has to beat.

Method notes:
  - The first half of the trace is warmup/training (predictors and
    frequency tables learn there; caches warm through it); every number
    reported is measured on the second half only.
  - Stall model, per layer transition: predicted misses were issued one
    layer-compute early, so they stall max(0, bytes/B - layer_compute);
    unpredicted misses stall their full bytes/B. Bandwidth is shared, so
    bytes aggregate per layer. Wrong predictions waste traffic but not
    stall.

Usage: expert-replay-sim.py TRACE [--expert-mb 1.5] [--layer-ms 1.0]
"""
import argparse
import sys
from collections import Counter, defaultdict


def parse(path):
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
    all_layers = sorted({li for t in tokens for li in t})
    tokens = [t for t in tokens if len(t) == len(all_layers)]
    return tokens, all_layers


# ---------------- cache policies ----------------
# Each cache maps layer -> resident set, with its own eviction rule.
# `global_pool=True` shares one slot budget across layers (hot layers
# earn more slots); otherwise each layer gets an equal slice.

class LRU:
    name = "LRU"

    def __init__(self, layers, cap, global_pool=False):
        # dicts preserve insertion order: first key is the eviction victim.
        self.global_pool = global_pool
        if global_pool:
            self.d = {}  # (layer, expert) -> None
            self.cap = cap * len(layers)
        else:
            self.d = {li: {} for li in layers}  # layer -> {expert: None}
            self.cap = cap

    def access(self, li, e):
        """-> True on hit."""
        m = self.d if self.global_pool else self.d[li]
        key = (li, e) if self.global_pool else e
        hit = key in m
        if hit:
            del m[key]  # reinsert below to refresh recency
        elif len(m) >= self.cap:
            del m[next(iter(m))]
        m[key] = None
        return hit

    def resident(self, li, e):
        return ((li, e) in self.d) if self.global_pool else (e in self.d[li])


class DecayLFU:
    name = "decayed-LFU"
    HALF_LIFE = 64 * 384  # in accesses (~64 tokens of 48 layers x 8)

    def __init__(self, layers, cap, global_pool=False):
        self.score = {}  # (layer,expert) -> (count, last_access_step)
        self.global_pool = global_pool
        self.res = set() if global_pool else {li: set() for li in layers}
        self.cap = cap * (len(layers) if global_pool else 1)
        self.decay = 0.5 ** (1.0 / self.HALF_LIFE)
        self.step = 0

    def _eff(self, key):
        c, last = self.score.get(key, (0.0, self.step))
        return c * (self.decay ** (self.step - last))

    def access(self, li, e):
        self.step += 1
        key = (li, e)
        self.score[key] = (self._eff(key) + 1.0, self.step)
        pool = self.res if self.global_pool else self.res[li]
        hit = key in pool
        if not hit:
            if len(pool) >= self.cap:
                pool.remove(min(pool, key=self._eff))
            pool.add(key)
        return hit

    def resident(self, li, e):
        pool = self.res if self.global_pool else self.res[li]
        return (li, e) in pool


def cache_sweep(tokens, layers, expert_mb, budgets):
    half = len(tokens) // 2
    n_expert = max(e for t in tokens for v in t.values() for e in v) + 1
    rows = []
    for frac in budgets:
        cap = max(1, int(n_expert * frac))
        for cls in (LRU, DecayLFU):
            for global_pool in (False, True):
                c = cls(layers, cap, global_pool)
                miss = acts = 0
                for i, t in enumerate(tokens):
                    for li in layers:
                        for e in t.get(li, []):
                            hit = c.access(li, e)
                            if i >= half:
                                acts += 1
                                miss += 0 if hit else 1
                mb = miss / (len(tokens) - half) * expert_mb
                rows.append((frac, cls.name, "global" if global_pool else "per-layer",
                             mb, miss / acts))
    return rows


# ---------------- prefetch simulation ----------------

def train_cooc(tokens, layers, depth):
    """Co-occurrence of layer li's selection with layer li+depth's."""
    co = {li: defaultdict(Counter) for li in layers if li + depth <= layers[-1]}
    for t in tokens:
        for li in co:
            tgt = t.get(li + depth, [])
            for e in t.get(li, []):
                co[li][e].update(tgt)
    return co


def prefetch_sim(tokens, layers, expert_mb, layer_ms, bandwidths, cache_frac, depths):
    """Replay the second half with an LRU cache plus a DEPTH-layer-lookahead
    prefetcher: predictions for layer li+D are issued at layer li, so a
    predicted miss has D*layer_ms of fetch lead. A note on two deliberate
    simplifications: prefetched experts do not enter the cache (no pollution
    from wrong guesses, no credit for later reuse -- roughly cancels for a
    ceiling estimate), and recency-style predictors are omitted because the
    LRU already holds exactly what recency would predict (measured: zero
    effect). The oracle row is the shape's ceiling at that depth; the cooc
    row is what a table learned from the trace achieves; a trained predictor
    lands between them."""
    half = len(tokens) // 2
    n_expert = max(e for t in tokens for v in t.values() for e in v) + 1
    cap = max(1, int(n_expert * cache_frac))
    def predict_for(kind, co, t, li, tgt):
        if kind.endswith("oracle"):
            return t[tgt]
        score = Counter()
        for e in t.get(li, []):
            score.update(co.get((li, tgt), {}).get(e, {}))
        return [e for e, _ in score.most_common(8)]

    results = []
    # pregate-*: layer 0's routing issues predictions for EVERY later layer
    # at once, so layer lj's fetches get lj*layer_ms of lead -- the schedule
    # a pre-gated/trained-predictable model (the lever's stage-2 target)
    # would make real. Depth-D kinds model a conventional pipelined
    # predictor issuing one layer at a time.
    for kind in ("none", "cooc", "oracle", "pregate-cooc", "pregate-oracle"):
        for depth in ([0] if kind == "none" else [0] if kind.startswith("pregate") else depths):
            co = None
            if kind == "cooc":
                co = {(li, li + depth): tbl
                      for li, tbl in train_cooc(tokens[:half], layers, depth).items()}
            elif kind == "pregate-cooc":
                co = {}
                for d in range(1, len(layers)):
                    co.update({(0, d): train_cooc(tokens[:half], layers, d).get(0, {})})
            for B in bandwidths:  # MB/s
                cache = LRU(layers, cap)
                stall_ms = demand_mb = waste_mb = 0.0
                for i, t in enumerate(tokens):
                    measured = i >= half
                    pending = {}  # target layer -> (experts in flight, lead ms)
                    for li in layers:
                        unpred_bytes = pred_bytes = 0.0
                        inflight, lead = pending.get(li, ((), 0.0))
                        for e in t.get(li, []):
                            hit = cache.access(li, e)
                            if not hit:
                                if e in inflight:
                                    pred_bytes += expert_mb
                                else:
                                    unpred_bytes += expert_mb
                        if measured:
                            demand_mb += pred_bytes + unpred_bytes
                            stall_ms += unpred_bytes / B * 1000
                            stall_ms += max(0.0, pred_bytes / B * 1000 - lead)
                        # issue predictions
                        targets = ()
                        if kind in ("cooc", "oracle"):
                            targets = (li + depth,)
                        elif kind.startswith("pregate") and li == 0:
                            targets = layers[1:]
                        for tgt in targets:
                            if tgt > layers[-1] or tgt not in t:
                                continue
                            pred = predict_for(kind, co, t, li, tgt)
                            fresh = {e for e in pred if not cache.resident(tgt, e)}
                            old, _ = pending.get(tgt, (set(), 0.0))
                            pending[tgt] = (set(old) | fresh, (tgt - li) * layer_ms)
                            if measured:
                                waste_mb += sum(expert_mb for e in fresh
                                                if e not in set(t[tgt]))
                n = len(tokens) - half
                results.append((kind, depth, B, stall_ms / n, demand_mb / n, waste_mb / n))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--expert-mb", type=float, default=1.5)
    ap.add_argument("--layer-ms", type=float, default=1.0,
                    help="compute time per MoE layer (lead time a 1-layer prefetch gets)")
    args = ap.parse_args()

    tokens, layers = parse(args.trace)
    if len(tokens) < 64:
        sys.exit(f"only {len(tokens)} complete tokens; need more")
    print(f"trace: {len(tokens)} decode tokens, {len(layers)} layers, "
          f"{args.expert_mb} MB/expert, {args.layer_ms} ms/layer "
          f"(measured on the last {len(tokens) - len(tokens)//2} tokens)")

    print("\n== cache policy sweep (fetch-MB/token) ==")
    print(f"{'budget':>7} {'policy':>12} {'slots':>9} {'MB/tok':>8} {'miss':>6}")
    for frac, name, mode, mb, miss in cache_sweep(
            tokens, layers, args.expert_mb, (0.125, 0.25, 0.5)):
        print(f"{int(frac*100):>6}% {name:>12} {mode:>9} {mb:>8.1f} {miss*100:>5.1f}%")

    compute_ms = len(layers) * args.layer_ms
    print(f"\n== depth-D lookahead prefetch on LRU@25% "
          f"(compute floor {compute_ms:.0f} ms/token) ==")
    print(f"{'predictor':>9} {'depth':>5} {'MB/s':>6} {'stall':>9} {'tok/s':>6} {'waste':>7}")
    for kind, depth, B, stall, demand, waste in prefetch_sim(
            tokens, layers, args.expert_mb, args.layer_ms,
            (125.0, 375.0, 1250.0), 0.25, (1, 4, 8, 16, 47)):
        toks = 1000.0 / (compute_ms + stall)
        print(f"{kind:>9} {depth:>5} {B:>6.0f} {stall:>8.1f}ms {toks:>6.1f} {waste:>6.1f}MB")


if __name__ == "__main__":
    main()
