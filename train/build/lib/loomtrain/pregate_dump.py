#!/usr/bin/env python3
"""Dump aligned trace + predictions from a TRAINED pregate head on fresh
held-out data (the in-training dump captured the head's infancy)."""
import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/OLMoE-1B-7B-0924"


def batches(tok, seq_len, batch_size, skip_tokens):
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    buf, skipped = [], 0
    for ex in ds:
        ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
        if skipped < skip_tokens:  # skip past the training stream
            skipped += len(ids)
            continue
        buf.extend(ids)
        buf.append(tok.eos_token_id)
        while len(buf) >= seq_len * batch_size:
            chunk = buf[: seq_len * batch_size]
            buf = buf[seq_len * batch_size:]
            yield torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="pregate-head-xl.pt")
    ap.add_argument("--head-width", type=int, default=2048)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--skip-tokens", type=int, default=20_000_000)
    ap.add_argument("--top-pred", type=int, default=0,
                    help="dump this many ranked predictions per layer (0 = top_k)")
    ap.add_argument("--trace-out", default="olmoe-trace-final.txt")
    ap.add_argument("--pred-out", default="olmoe-pred-final.txt")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    cfg = model.config
    top_k, n_layers, n_e, hid = (cfg.num_experts_per_tok, cfg.num_hidden_layers,
                                 cfg.num_experts, cfg.hidden_size)
    head = nn.Sequential(
        nn.Linear(hid, args.head_width), nn.GELU(),
        nn.Linear(args.head_width, (n_layers - 1) * n_e)).cuda().float()
    head.load_state_dict(torch.load(args.head, weights_only=True))
    head.eval()

    hits = total = 0
    tl, pl = [], []
    with torch.no_grad():
        for bi, ids in enumerate(batches(tok, args.seq_len, args.batch_size,
                                         args.skip_tokens)):
            if bi >= args.batches:
                break
            out = model(ids.cuda(), output_hidden_states=True, output_router_logits=True)
            h1 = out.hidden_states[1].reshape(-1, hid).float()
            sels = [rl.topk(top_k, dim=-1).indices for rl in out.router_logits]
            n_dump = args.top_pred or top_k
            logits = head(h1).view(-1, n_layers - 1, n_e)
            pred = logits.topk(n_dump, dim=-1).indices
            pred8 = pred[:, :, :top_k]
            for i in range(n_layers - 1):
                hits += (pred8[:, i].unsqueeze(2) == sels[i + 1].unsqueeze(1)).any(2).sum().item()
                total += sels[i + 1].numel()
            for pos in range(sels[0].shape[0]):
                for li in range(n_layers):
                    tl.append(f"{li} " + " ".join(str(int(e)) for e in sels[li][pos]) + "\n")
                for i in range(n_layers - 1):
                    pl.append(f"{i+1} " + " ".join(str(int(e)) for e in pred[pos, i]) + "\n")

    print(f"trained-head accuracy on this dump: {hits / total * 100:.1f}%")
    open(args.trace_out, "w").writelines(tl)
    open(args.pred_out, "w").writelines(pl)
    print(f"{len(tl)} trace lines, {len(pl)} pred lines")


if __name__ == "__main__":
    main()
