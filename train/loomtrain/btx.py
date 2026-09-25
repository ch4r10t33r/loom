#!/usr/bin/env python3
"""BTX (Branch-Train-MiX) stage 0: dense branches -> loom-servable MoE.

The distributed-training inversion loom can honestly claim (whitepaper
decision log, 2026-08): branches train INDEPENDENTLY -- separate machines
or separate GPUs, zero communication -- and compose by merge into one
qwen3moe-shaped model, which loom's engine serves natively. Gradients
never cross a wire; only finished artifacts do.

The merge is EXACT by construction: branches train only their FFNs
(mlp.gate_proj/up_proj/down_proj); attention, norms, embeddings stay
frozen at the seed's values, so the merged trunk is the seed verbatim and
each branch's FFN drops in as expert e with zero averaging error. A fresh
per-layer router is then trained alone on mixed data (the stage-2a
frozen-backbone recipe).

Stages:
  branch:  python3 -u -m loomtrain.btx branch --seed Qwen/Qwen3-0.6B-Base \
             --dataset HuggingFaceFW/fineweb-edu --tokens 30000000 \
             --device cuda:0 --out branch-web
  merge:   python3 -u -m loomtrain.btx merge --seed Qwen/Qwen3-0.6B-Base \
             --branches branch-web branch-math --router-tokens 5000000 \
             --datasets HuggingFaceFW/fineweb-edu open-web-math/open-web-math \
             --out btx-moe
           (one "repo[:subset]" spec per branch; router trains on the
            round-robin interleave of all of them)
  eval:    python3 -u -m loomtrain.btx eval --model btx-moe \
             --dataset open-web-math/open-web-math --tokens 200000

House recipe: non-reentrant checkpointing, first-step grad sanity,
timestamps, python3 -u, eval only from final weights.
"""

import argparse
import math
import time


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_device(name):
    import torch
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def stream_batches(tok, dataset, seq, batch, skip=0, subset=None):
    """Padded (input_ids, labels) batches from a streaming HF dataset.
    `dataset` may carry its config inline as "repo[:subset]"."""
    import torch
    from datasets import load_dataset
    if subset is None and ":" in dataset:
        dataset, subset = dataset.split(":", 1)
    kwargs = {"streaming": True, "split": "train"}
    if subset:
        kwargs["name"] = subset
    ds = load_dataset(dataset, **kwargs)
    if skip:
        ds = ds.skip(skip)
    buf = []
    for row in ds:
        ids = tok(row["text"], truncation=True, max_length=seq)["input_ids"]
        if len(ids) < 32:
            continue
        buf.append(ids)
        if len(buf) < batch:
            continue
        maxlen = max(len(x) for x in buf)
        pad = tok.pad_token_id or 0
        input_ids = torch.full((len(buf), maxlen), pad, dtype=torch.long)
        for i, x in enumerate(buf):
            input_ids[i, : len(x)] = torch.tensor(x)
        labels = input_ids.clone()
        labels[input_ids == pad] = -100
        buf = []
        yield input_ids, labels


def train_loop(model, tok, dataset, tokens, seq, batch, lr, device, save_fn, only=None, skip=0, subset=None, stream=None):
    """Shared trainer: freeze everything except params matching `only`,
    fp32 masters + bf16 autocast, grad-sanity gate on step one."""
    import torch
    for name, p in model.named_parameters():
        p.requires_grad = only(name) if only else True
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable {sum(p.numel() for p in trainable)/1e6:.1f}M params")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = torch.optim.AdamW(trainable, lr=lr)
    # Warmup + cosine decay to zero over the planned token budget. Constant
    # LR leaves the weights "hot" at save time: stage-0 branches trained at
    # constant 5e-5 ended WORSE than the seed on their own domain, and an
    # unannealed endpoint is the standard cause.
    total_steps = max(math.ceil(tokens / (batch * seq)), 1)
    warmup = min(100, max(total_steps // 20, 1))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * min((s - warmup) / max(total_steps - warmup, 1), 1.0)))))
    use_amp = device.startswith("cuda")

    seen, step_i, t0 = 0, 0, time.time()
    batches = stream if stream is not None else stream_batches(tok, dataset, seq, batch, skip=skip, subset=subset)
    for input_ids, labels in batches:
        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        if step_i == 0:
            gn = sum(p.grad.abs().sum().item() for p in trainable if p.grad is not None)
            log(f"first-step grad sum {gn:.3e}")
            assert gn > 0, "no gradient reached the trainable params"
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        seen += int((labels != -100).sum())
        step_i += 1
        if step_i % 20 == 0:
            log(f"step {step_i} loss {out.loss.item():.4f} tokens {seen/1e6:.1f}M "
                f"({seen/max(time.time()-t0,1):.0f} tok/s)")
        if step_i % 1000 == 0:
            save_fn()
        if seen >= tokens:
            break
    save_fn()
    log(f"loop done: {seen/1e6:.1f}M tokens")


def ternary_quantize(w):
    """BitNet-b1.58-style ternary snap with a per-output-row scale:
    gamma = mean|w| per row, w_q = clip(round(w/gamma), -1, 1) * gamma.
    Every row's values land in {-gamma, 0, +gamma}."""
    import torch
    gamma = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return (w / gamma).round().clamp(-1, 1) * gamma


def apply_ternary_qat(lin):
    """QAT forward on one Linear: ternary weights in the forward pass,
    straight-through estimator so gradients reach the fp32 latents."""
    import torch

    def fwd(x):
        w = lin.weight
        w_ste = w + (ternary_quantize(w) - w).detach()
        return torch.nn.functional.linear(x, w_ste, lin.bias)

    lin.forward = fwd


def branch(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed_val)
    dev = resolve_device(args.device)
    log(f"branch: seed {args.seed} on {args.dataset} -> {args.out} (device {dev})"
        + (" [ternary QAT]" if args.ternary else ""))
    # fp32 masters (bf16 compute via autocast): full-FFN training in bf16
    # params is where silent quality loss hides
    model = AutoModelForCausalLM.from_pretrained(
        args.seed, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        device_map={"": dev})
    tok = AutoTokenizer.from_pretrained(args.seed)

    def is_ffn(name):
        return any(k in name for k in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))

    q_lins = []
    if args.ternary:
        # Ternary is a TRAINING-time property (QAT), not a conversion: the
        # latents stay fp32, the forward sees ternary weights via STE, and
        # the saved artifact snaps to the ternary values. Quality arm only:
        # weights are ternary-VALUED but stored f16/f32 -- the fetch-byte
        # win needs a ternary storage type + loom kernel, gated on this
        # arm's eval.
        for name, mod in model.named_modules():
            if is_ffn(name) and isinstance(mod, torch.nn.Linear):
                apply_ternary_qat(mod)
                q_lins.append(mod)
        assert q_lins, "no FFN Linear modules matched for ternary QAT"
        z = sum((ternary_quantize(m.weight) == 0).sum().item() for m in q_lins)
        n = sum(m.weight.numel() for m in q_lins)
        log(f"ternary QAT on {len(q_lins)} FFN projections ({z/n:.1%} zeros at init)")

    def save():
        if q_lins:
            # export the SNAPPED weights (the model the merge composes must
            # be the model QAT optimized), keeping latents intact to train on
            with torch.no_grad():
                latents = [m.weight.detach().cpu().clone() for m in q_lins]
                for m in q_lins:
                    m.weight.copy_(ternary_quantize(m.weight))
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        if q_lins:
            with torch.no_grad():
                for m, w in zip(q_lins, latents):
                    m.weight.copy_(w.to(m.weight.device))
        log(f"saved -> {args.out}")

    train_loop(model, tok, args.dataset, args.tokens, args.seq, args.batch,
               args.lr, dev, save, only=is_ffn, subset=args.subset)


def build_moe(seed_path, branch_paths):
    """Seed trunk + branch FFNs as experts + fresh router. Exact merge."""
    import torch
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
    seed = AutoModelForCausalLM.from_pretrained(seed_path, torch_dtype=torch.float32)
    sc = seed.config
    k = len(branch_paths)
    cfg = Qwen3MoeConfig(
        vocab_size=sc.vocab_size, hidden_size=sc.hidden_size,
        intermediate_size=sc.intermediate_size,
        moe_intermediate_size=sc.intermediate_size,
        num_hidden_layers=sc.num_hidden_layers,
        num_attention_heads=sc.num_attention_heads,
        num_key_value_heads=sc.num_key_value_heads,
        head_dim=getattr(sc, "head_dim", sc.hidden_size // sc.num_attention_heads),
        rms_norm_eps=sc.rms_norm_eps, rope_theta=sc.rope_theta,
        max_position_embeddings=sc.max_position_embeddings,
        tie_word_embeddings=sc.tie_word_embeddings,
        # top-2 of K: with top-1 + normalized probs the selected weight is
        # identically 1.0 and the router receives ZERO gradient (caught by
        # pre-flight: grad sum 7e-9). Routing min(2, k) keeps the weights on
        # the softmax and the gradient alive; at k=2 there is no sparsity to
        # lose anyway.
        num_experts=k, num_experts_per_tok=min(2, k), decoder_sparse_step=1,
        mlp_only_layers=[], norm_topk_prob=True,
    )
    moe = Qwen3MoeForCausalLM(cfg)
    # trunk: every seed weight whose name exists identically in the MoE
    ssd = seed.state_dict()
    msd = moe.state_dict()
    copied = 0
    for name, w in ssd.items():
        if name in msd and msd[name].shape == w.shape:
            msd[name].copy_(w)
            copied += 1
    log(f"trunk: {copied} tensors copied from seed")
    # experts: branch e's FFN -> experts[e]
    for e, bp in enumerate(branch_paths):
        b = AutoModelForCausalLM.from_pretrained(bp, torch_dtype=torch.float32)
        bsd = b.state_dict()
        n = 0
        for li in range(cfg.num_hidden_layers):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                src = bsd[f"model.layers.{li}.mlp.{proj}.weight"]
                msd[f"model.layers.{li}.mlp.experts.{e}.{proj}.weight"].copy_(src)
                n += 1
        del b, bsd
        log(f"expert {e} <- {bp} ({n} tensors)")
    # fresh small-random routers (trained next)
    torch.manual_seed(0)
    for li in range(cfg.num_hidden_layers):
        msd[f"model.layers.{li}.mlp.gate.weight"].normal_(0, 0.02)
    moe.load_state_dict(msd)
    return moe


def merge(args):
    import itertools
    import torch
    from transformers import AutoTokenizer
    dev = resolve_device(args.device)
    log(f"merge: {len(args.branches)} branches -> {args.out} (device {dev})")
    moe = build_moe(args.seed, args.branches).to(dev)
    tok = AutoTokenizer.from_pretrained(args.seed)

    def save():
        moe.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        log(f"saved -> {args.out}")

    if args.router_tokens > 0:
        # router-only training on a round-robin interleave of every branch's
        # domain (one dataset spec per branch, "repo[:subset]")
        streams = [stream_batches(tok, ds, args.seq, args.batch)
                   for ds in args.datasets]
        mixed = itertools.chain.from_iterable(zip(*streams))
        train_loop(moe, tok, "mixed", args.router_tokens, args.seq,
                   args.batch, args.router_lr, dev, save,
                   only=lambda n: "mlp.gate." in n, stream=mixed)
    else:
        save()


def evaluate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = resolve_device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map={"": dev})
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    losses, seen = [], 0
    with torch.no_grad():
        for input_ids, labels in stream_batches(tok, args.dataset, args.seq,
                                                args.batch, skip=args.skip, subset=args.subset):
            out = model(input_ids=input_ids.to(model.device), labels=labels.to(model.device))
            losses.append(out.loss.item())
            seen += int((labels != -100).sum())
            if seen >= args.tokens:
                break
    import math
    mean = sum(losses) / len(losses)
    log(f"eval {args.model} on {args.dataset}: loss {mean:.4f} ppl {math.exp(mean):.2f} ({seen/1e3:.0f}k tokens)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("branch")
    bp.add_argument("--seed", required=True)
    bp.add_argument("--dataset", required=True)
    bp.add_argument("--subset", default=None)
    bp.add_argument("--tokens", type=int, default=30_000_000)
    bp.add_argument("--lr", type=float, default=5e-5)
    bp.add_argument("--seq", type=int, default=1024)
    bp.add_argument("--batch", type=int, default=8)
    bp.add_argument("--device", default="auto")
    bp.add_argument("--seed-val", type=int, default=0)
    bp.add_argument("--ternary", action="store_true")
    bp.add_argument("--out", required=True)

    mp = sub.add_parser("merge")
    mp.add_argument("--seed", required=True)
    mp.add_argument("--branches", nargs="+", required=True)
    mp.add_argument("--datasets", nargs="+", required=True,
                    help='one "repo[:subset]" spec per branch domain')
    mp.add_argument("--router-tokens", type=int, default=5_000_000)
    mp.add_argument("--router-lr", type=float, default=1e-3)
    mp.add_argument("--seq", type=int, default=1024)
    mp.add_argument("--batch", type=int, default=8)
    mp.add_argument("--device", default="auto")
    mp.add_argument("--out", required=True)

    ep = sub.add_parser("eval")
    ep.add_argument("--model", required=True)
    ep.add_argument("--dataset", required=True)
    ep.add_argument("--subset", default=None)
    ep.add_argument("--tokens", type=int, default=200_000)
    ep.add_argument("--skip", type=int, default=20000)
    ep.add_argument("--seq", type=int, default=1024)
    ep.add_argument("--batch", type=int, default=4)
    ep.add_argument("--device", default="auto")

    a = ap.parse_args()
    {"branch": branch, "merge": merge, "eval": evaluate}[a.cmd](a)
