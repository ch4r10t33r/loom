#!/bin/sh
# Summarize the alpha telemetry a bootnode collected with --alpha-ingest.
#
#   ./scripts/alpha-report.sh /path/to/metrics.jsonl
#
# Reads the JSONL produced by the METRICS wire command and prints a per-node
# and fleet summary: versions, platforms, hold fractions, generation rates,
# and hit rates. Pure read-only; python3 only.
set -eu
FILE="${1:?usage: alpha-report.sh <metrics.jsonl>}"

python3 - "$FILE" <<'PY'
import json, sys, statistics as st
from collections import defaultdict

nodes = {}          # id -> last report
first_seen = {}     # id -> first up_s report count
rows = 0
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    rows += 1
    nodes[d.get("id", "?")] = d

if not nodes:
    print("no valid reports")
    sys.exit(0)

vals = list(nodes.values())
def col(k, default=0):
    return [v.get(k, default) for v in vals]

print(f"reports: {rows} lines, {len(nodes)} distinct nodes\n")
print(f"{'id':<18} {'ver':<9} {'os/arch':<15} {'up':>7} {'held':>12} {'peers':>5} {'gens':>5} {'tok/s':>7} {'hit':>6}")
for id_, v in sorted(nodes.items(), key=lambda kv: -kv[1].get("up_s", 0)):
    held = f"{v.get('held',0)}/{v.get('total',0)}"
    up_h = v.get("up_s", 0) / 3600
    print(f"{id_:<18} {v.get('v','?'):<9} {v.get('os','?')+'/'+v.get('arch','?'):<15} "
          f"{up_h:>6.1f}h {held:>12} {v.get('peers',0):>5} {v.get('gens',0):>5} "
          f"{v.get('tok_s_avg',0):>7.2f} {v.get('hit_avg',0):>6.3f}")

gens_total = sum(col("gens"))
active = [v for v in vals if v.get("gens", 0) > 0]
print(f"\nfleet: {gens_total} generations across {len(active)} generating nodes")
if active:
    ts = [v["tok_s_avg"] for v in active]
    hr = [v["hit_avg"] for v in active]
    print(f"tok/s   median {st.median(ts):.2f}  min {min(ts):.2f}  max {max(ts):.2f}")
    print(f"hit     median {st.median(hr):.3f}  min {min(hr):.3f}  max {max(hr):.3f}")
holds = [v.get("held", 0) / max(v.get("total", 1), 1) for v in vals]
print(f"held    median {st.median(holds)*100:.1f}%  sum-of-fractions {sum(holds):.2f} copies")
by_ver = defaultdict(int)
for v in vals:
    by_ver[v.get("v", "?")] += 1
print("versions " + ", ".join(f"{k}:{n}" for k, n in sorted(by_ver.items())))
PY
