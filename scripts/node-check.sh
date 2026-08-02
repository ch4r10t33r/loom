#!/bin/sh
# Health-check a running loom node: wait for it to serve, run a few small
# prompts, and report whether generation works and at what speed.
#
#   ./scripts/node-check.sh                 # checks 127.0.0.1:8770 (RPC)
#   ./scripts/node-check.sh 127.0.0.1 9870  # explicit host/port
#
# Environment:
#   LOOM_CHECK_PROMPTS   number of prompts to run          (default 3)
#   LOOM_CHECK_TOKENS    max_tokens per prompt             (default 8)
#   LOOM_CHECK_TIMEOUT   seconds allowed per prompt        (default 3600)
#   LOOM_CHECK_WAIT      seconds to wait for the port      (default 900)
#
# On a fresh devnet node the first prompt is the slow one: missing experts
# stream from peers and persist, so the hit rate climbs and later prompts
# speed up. The report shows that curve, which is the number that tells you
# whether the node is warming or genuinely stuck.
set -eu
HOST="${1:-127.0.0.1}"
PORT="${2:-8770}"

python3 - "$HOST" "$PORT" <<'PY'
import json, socket, sys, time, os

host, port = sys.argv[1], int(sys.argv[2])
n_prompts = int(os.environ.get("LOOM_CHECK_PROMPTS", 3))
max_tokens = int(os.environ.get("LOOM_CHECK_TOKENS", 8))
timeout = int(os.environ.get("LOOM_CHECK_TIMEOUT", 3600))
wait = int(os.environ.get("LOOM_CHECK_WAIT", 900))

PROMPTS = [
    "The capital of France is",
    "Water freezes at a temperature of",
    "The first person to walk on the moon was",
    "Two plus two equals",
    "The chemical symbol for gold is",
]

print(f"waiting for {host}:{port} (up to {wait}s; a big model maps and loads first)...")
t0 = time.time()
while True:
    try:
        socket.create_connection((host, port), timeout=5).close()
        break
    except OSError:
        if time.time() - t0 > wait:
            print("FAIL: node never started serving RPC")
            sys.exit(1)
        time.sleep(5)
print(f"node is serving (after {time.time()-t0:.0f}s)\n")

results = []
for i in range(n_prompts):
    prompt = PROMPTS[i % len(PROMPTS)]
    req = json.dumps({"prompt": prompt, "max_tokens": max_tokens}) + "\n"
    t0 = time.time()
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.settimeout(timeout)
        s.sendall(req.encode())
        buf = b""
        while b"\n" not in buf:
            d = s.recv(65536)
            if not d:
                break
            buf += d
        s.close()
        el = time.time() - t0
        r = json.loads(buf.decode())
        if not r.get("ok"):
            print(f"[{i+1}/{n_prompts}] FAIL after {el:.0f}s: {r.get('error','no ok field')}")
            results.append(None)
            continue
        results.append((el, r.get("tok_per_s", 0.0), r.get("hit_rate", 0.0), r.get("generated", 0)))
        print(f"[{i+1}/{n_prompts}] ok  {el:7.1f}s  {r.get('tok_per_s',0):6.2f} tok/s  "
              f"hit {r.get('hit_rate',0):.3f}  ({r.get('generated',0)} tokens)  \"{prompt}\"")
    except (OSError, json.JSONDecodeError) as e:
        el = time.time() - t0
        print(f"[{i+1}/{n_prompts}] FAIL after {el:.0f}s: {type(e).__name__}: {e}")
        results.append(None)

ok = [r for r in results if r]
print()
if not ok:
    print("VERDICT: NOT WORKING -- no prompt completed. Check the node console;")
    print("if it is mid-sync, wait for the sync lines to finish and rerun.")
    sys.exit(1)

last_el, last_tok, last_hit, _ = ok[-1]
print(f"VERDICT: WORKING -- {len(ok)}/{n_prompts} prompts completed")
print(f"current speed : {last_tok:.2f} tok/s (latest prompt, the number that reflects the warmed store)")
if len(ok) > 1:
    first_hit = ok[0][2]
    print(f"warming       : hit rate {first_hit:.3f} -> {last_hit:.3f} across the run"
          + ("  (rising: fetched experts are persisting)" if last_hit > first_hit + 0.005 else ""))
else:
    print(f"hit rate      : {last_hit:.3f}")
if last_hit < 0.95:
    print("note          : below ~0.95 hit rate, speed is your network, not your CPU/GPU.")
    print("                More prompts warm it further; a higher --hold-fraction starts warmer.")
sys.exit(0)
PY
