#!/usr/bin/env python3
"""Stage 2a of research lever 9: retrain ONLY the routers of a frozen MoE
for cacheability, and measure the exchange rate.

The stage-0/1 evidence: load-balance losses flatten expert usage (OLMoE's
LRU miss at a 25% budget is 48.5% vs Qwen3's 19.6%), and temporal locality
is what caches actually exploit. This experiment asks what routing looks
like when the balance pressure is removed and a consistency pressure is
added -- with the backbone frozen so language modeling can only degrade via
routing, making Delta-ppl the honest price tag.

Trainable: the 16 router matrices (mlp.gate, ~2M params). Loss:
LM cross-entropy + lambda * consistency, where consistency is the mean
symmetric KL between adjacent tokens' router distributions per layer.
router_aux_loss_coef is zeroed (the balance loss is the anti-cache force).

Per run it reports held-out ppl before/after and dumps a decode-shape
trace (LOOM_EXPERT_TRACE format) from the retrained model, so
expert-trace-stats.py / expert-replay-sim.py score the shape change
against the frozen baseline directly.

Usage: train_2a.py --lam 0.5 [--tokens 8000000] [--out routers-l0.5.pt]
"""
import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0924"


def batches(tok, seq_len, batch_size, skip_docs=0):
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    buf, seen = [], 0
    for ex in ds:
        seen += 1
        if seen <= skip_docs:
            continue
        buf.extend(tok(ex["text"], add_special_tokens=False)["input_ids"])
        buf.append(tok.eos_token_id)
        while len(buf) >= seq_len * batch_size:
            chunk = buf[: seq_len * batch_size]
            buf = buf[seq_len * batch_size:]
            yield torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len)


def consistency_loss(router_logits, bsz, seqlen):
    """Mean symmetric KL between adjacent tokens' routing, over layers."""
    total = 0.0
    for rl in router_logits:
        lp = F.log_softmax(rl.float().view(bsz, seqlen, -1), dim=-1)
        p, q = lp[:, 1:], lp[:, :-1]
        kl_pq = (p.exp() * (p - q)).sum(-1)
        kl_qp = (q.exp() * (q - p)).sum(-1)
        total = total + (kl_pq + kl_qp).mean() / 2
    return total / len(router_logits)


@torch.no_grad()
def evaluate(model, tok, seq_len, n_batches, trace_path=None):
    """Held-out CE (and optionally a LOOM-format trace) on a fixed slice
    far past the training stream."""
    ce = n = 0.0
    lines = []
    top_k = model.config.num_experts_per_tok
    for bi, ids in enumerate(batches(tok, seq_len, 2, skip_docs=200_000)):
        if bi >= n_batches:
            break
        ids = ids.cuda()
        out = model(ids, labels=ids, output_router_logits=trace_path is not None)
        ce += out.loss.item() * ids.numel()
        n += ids.numel()
        if trace_path:
            sels = [rl.topk(top_k, dim=-1).indices for rl in out.router_logits]
            for pos in range(sels[0].shape[0]):
                for li in range(len(sels)):
                    es = " ".join(str(int(e)) for e in sels[li][pos])
                    lines.append(f"{li} {es}\n")
    if trace_path:
        open(trace_path, "w").writelines(lines)
    return ce / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--tokens", type=int, default=8_000_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-batches", type=int, default=48)
    ap.add_argument("--skip-baseline-eval", action="store_true")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.router_aux_loss_coef = 0.0  # the anti-cache force, off
    # non-reentrant: with a frozen backbone no checkpoint segment INPUT
    # requires grad, and reentrant checkpointing would silently skip those
    # segments' backward, starving the router params of gradients.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():
        p.requires_grad_(False)

    class FP32Gate(torch.nn.Module):
        """Router in fp32 inside a bf16 model: cast the input up, compute,
        cast the logits back, so neighbors see the dtype they expect."""
        def __init__(self, lin):
            super().__init__()
            self.weight = torch.nn.Parameter(lin.weight.data.float())
        def forward(self, x):
            return F.linear(x.float(), self.weight).to(x.dtype)

    routers = []
    for layer in model.model.layers:
        layer.mlp.gate = FP32Gate(layer.mlp.gate).cuda()
        routers.append(layer.mlp.gate.weight)
    n_train = sum(p.numel() for p in routers)
    print(f"trainable router params: {n_train} across {len(routers)} layers; "
          f"lambda={a.lam} tokens={a.tokens}")

    if not a.skip_baseline_eval:
        model.eval()
        base_ce = evaluate(model, tok, a.seq_len, a.eval_batches)
        print(f"baseline held-out CE {base_ce:.4f} (ppl {torch.tensor(base_ce).exp():.2f})",
              flush=True)

    opt = torch.optim.AdamW(routers, lr=a.lr)
    model.train()
    seen = 0
    for bi, ids in enumerate(batches(tok, a.seq_len, a.batch_size)):
        if seen >= a.tokens:
            break
        ids = ids.cuda()
        out = model(ids, labels=ids, output_router_logits=True)
        cons = consistency_loss(out.router_logits, ids.shape[0], ids.shape[1])
        loss = out.loss + a.lam * cons
        opt.zero_grad()
        loss.backward()
        opt.step()
        seen += ids.numel()
        if seen <= a.batch_size * a.seq_len:
            gn = routers[0].grad
            print(f"sanity: router grad {'MISSING' if gn is None else 'flowing'}", flush=True)
        if bi % 50 == 0:
            print(f"tokens {seen}: lm {out.loss.item():.4f} cons {cons.item():.4f}",
                  flush=True)

    model.eval()
    trace = a.out.replace(".pt", "-trace.txt")
    ce = evaluate(model, tok, a.seq_len, a.eval_batches, trace_path=trace)
    print(f"post held-out CE {ce:.4f} (ppl {torch.tensor(ce).exp():.2f}); trace -> {trace}")
    torch.save({f"layer{i}": r.detach().cpu() for i, r in enumerate(routers)}, a.out)
    print(f"routers -> {a.out}")


if __name__ == "__main__":
    main()
