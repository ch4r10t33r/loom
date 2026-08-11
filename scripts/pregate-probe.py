#!/usr/bin/env python3
"""Stage 2b of research lever 9: can a small learned head predict EVERY
layer's expert selection from layer 0's hidden state?

Stage 1 established (replay, Qwen3-30B traces) that pre-gating is the
regime change for a fetch-bound partial holder -- oracle pre-gating
nearly reaches the compute floor -- and that a trace-learned
co-occurrence table captures only ~30% of the oracle gap. This probe
measures where a *trained* predictor lands on an open MoE (OLMoE-1B-7B:
16 layers, 64 experts, top-8), with zero risk to model quality (the
backbone is frozen; the head is a side artifact).

Method: stream FineWeb-Edu through the frozen model, collect layer-0
hidden states and every layer's router top-k; train an MLP head
h1 -> all layers' expert multi-hots; report per-layer top-8 overlap on
held-out data against the hot-set baseline (the same metric
scripts/expert-trace-stats.py reports for the co-occurrence table).
Also dumps a decode-shape trace of the frozen model in
LOOM_EXPERT_TRACE format so the stage-0/1 scripts can score OLMoE's
native activation shape.
"""
import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0924"


def batches(tok, seq_len, batch_size):
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    buf = []
    for ex in ds:
        buf.extend(tok(ex["text"], add_special_tokens=False)["input_ids"])
        buf.append(tok.eos_token_id)
        while len(buf) >= seq_len * batch_size:
            chunk = buf[: seq_len * batch_size]
            buf = buf[seq_len * batch_size:]
            yield torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len)


@torch.no_grad()
def collect(model, ids, top_k):
    """-> h1 [B*T, H], targets multi-hot [B*T, L-1, E], sel [L, B*T, top_k]"""
    out = model(ids.cuda(), output_hidden_states=True, output_router_logits=True)
    h1 = out.hidden_states[1].reshape(-1, model.config.hidden_size).float()
    sels = []
    for rl in out.router_logits:  # per layer [B*T, E]
        sels.append(rl.topk(top_k, dim=-1).indices)
    n_e = model.config.num_experts
    tgt = torch.zeros(h1.shape[0], len(sels) - 1, n_e, device=h1.device)
    for i, sel in enumerate(sels[1:]):
        tgt[:, i].scatter_(1, sel, 1.0)
    return h1, tgt, sels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-batches", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--trace-out", default="olmoe-trace.txt")
    ap.add_argument("--head-out", default="pregate-head.pt")
    ap.add_argument("--pred-out", default="")
    ap.add_argument("--head-width", type=int, default=1024)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    top_k = cfg.num_experts_per_tok
    n_layers, n_e, hid = cfg.num_hidden_layers, cfg.num_experts, cfg.hidden_size
    print(f"{MODEL}: {n_layers} layers, {n_e} experts, top-{top_k}, hidden {hid}")

    head = nn.Sequential(
        nn.Linear(hid, args.head_width), nn.GELU(),
        nn.Linear(args.head_width, (n_layers - 1) * n_e)).cuda().float()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    # hot-set baseline accumulates over the SAME training stream the head sees
    hot = torch.zeros(n_layers - 1, n_e, device="cuda")

    eval_hits = eval_base = eval_total = 0
    per_layer_hits = torch.zeros(n_layers - 1)
    per_layer_total = torch.zeros(n_layers - 1)
    trace_lines = []
    pred_lines = []
    seen = 0

    for bi, ids in enumerate(batches(tok, args.seq_len, args.batch_size)):
        if bi >= args.train_batches:
            break
        h1, tgt, sels = collect(model, ids, top_k)
        is_eval = (bi % args.eval_every) == args.eval_every - 1
        if not is_eval:
            for i in range(n_layers - 1):
                hot[i] += tgt[:, i].sum(0)
            logits = head(h1).view(-1, n_layers - 1, n_e)
            loss = bce(logits, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            seen += h1.shape[0]
            if bi % 50 == 0:
                print(f"batch {bi}: loss {loss.item():.4f} ({seen} positions)")
        else:
            with torch.no_grad():
                logits = head(h1).view(-1, n_layers - 1, n_e)
                pred = logits.topk(top_k, dim=-1).indices  # [N, L-1, k]
                base = hot.topk(top_k, dim=-1).indices  # [L-1, k]
                for i in range(n_layers - 1):
                    actual = sels[i + 1]  # [N, k]
                    p = pred[:, i]
                    hits = (p.unsqueeze(2) == actual.unsqueeze(1)).any(2).sum().item()
                    bhits = (base[i].view(1, -1, 1) == actual.unsqueeze(1)).any(1).sum().item()
                    eval_hits += hits
                    eval_base += bhits
                    eval_total += actual.numel()
                    per_layer_hits[i] += hits
                    per_layer_total[i] += actual.numel()
            # decode-shape trace of the frozen model, LOOM_EXPERT_TRACE format,
            # plus the head's aligned predictions for layers 1.. so the replay
            # sim can score the predictor as a pregate prefetcher.
            if len(trace_lines) < 400_000:
                for pos in range(sels[0].shape[0]):
                    for li in range(n_layers):
                        es = " ".join(str(int(e)) for e in sels[li][pos])
                        trace_lines.append(f"{li} {es}\n")
                    for i in range(n_layers - 1):
                        ps = " ".join(str(int(e)) for e in pred[pos, i])
                        pred_lines.append(f"{i+1} {ps}\n")

    print(f"\n== pregate head vs hot-set baseline (held-out) ==")
    print(f"head:     {eval_hits / eval_total * 100:.1f}% of experts predicted")
    print(f"hot-set:  {eval_base / eval_total * 100:.1f}%")
    print("per-layer (head):")
    for i in range(n_layers - 1):
        if per_layer_total[i]:
            print(f"  layer {i+1:>2}: {per_layer_hits[i] / per_layer_total[i] * 100:.1f}%")

    with open(args.trace_out, "w") as f:
        f.writelines(trace_lines)
    if args.pred_out:
        with open(args.pred_out, "w") as f:
            f.writelines(pred_lines)
    torch.save(head.state_dict(), args.head_out)
    print(f"\ntrace -> {args.trace_out} ({len(trace_lines)} lines), head -> {args.head_out}")


if __name__ == "__main__":
    main()
