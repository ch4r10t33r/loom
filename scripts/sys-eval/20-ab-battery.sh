#!/usr/bin/env bash
# Snapshot-controlled pre-gate A/B battery. Run ON the box after
# 10-join-sync.sh. Each run restores the store from the snapshot first
# (fetches persist and would otherwise warm later runs), then generates the
# identical greedy 96 tokens against the bootnode peer.
#
#   HOLD=0.2 bash 20-ab-battery.sh 3     # 3 pairs at the 0.2 snapshot
#
# Appends to sys-eval-out/ab-results.csv:
#   ts,hold,pair,mode,secs,tok_s,text_sha256
# Output text must be byte-identical across every run of a battery; the
# sha column is the check.
set -euo pipefail

PAIRS="${1:-3}"
HOLD="${HOLD:?set HOLD to the snapshot fraction, e.g. HOLD=0.2}"
BOOT="${BOOT:-168.119.167.242:8771}"
MAXTOK="${MAXTOK:-96}"
PROMPT="${PROMPT:-Explain how a mixture-of-experts language model routes tokens to experts, and why caching the hot experts matters for serving throughput.}"
STORE=~/.cache/loom/models/gguf-synced
SNAP=~/store-snap-"$HOLD"
HEAD=~/pregate-qwen3.lpg
OUT=~/sys-eval-out
CSV="$OUT/ab-results.csv"
mkdir -p "$OUT"
[ -d "$SNAP" ] || { echo "no snapshot $SNAP -- run 10-join-sync.sh $HOLD first" >&2; exit 1; }
[ -f "$HEAD" ] || { echo "no pregate head $HEAD -- run 00-setup-box.sh first" >&2; exit 1; }
[ -f "$CSV" ] || echo "ts,hold,pair,mode,secs,tok_s,text_sha256" >"$CSV"

one_run() { # mode: base | pregate
    local mode="$1" pair="$2"
    rm -rf "$STORE"
    cp -a "$SNAP" "$STORE"
    sync
    echo 3 >/proc/sys/vm/drop_caches 2>/dev/null || true

    local extra=()
    [ "$mode" = pregate ] && extra=(--pregate-head "$HEAD")
    local log="$OUT/run-$HOLD-$pair-$mode.log"
    local t0=$SECONDS
    /usr/local/bin/loom gguf run "$STORE" --peers "$BOOT" \
        --prompt "$PROMPT" --max-tokens "$MAXTOK" --temp 0 "${extra[@]}" >"$log" 2>&1
    local secs=$((SECONDS - t0))
    local toks
    toks=$(grep -oE "[0-9]+\.[0-9]+ tok/s" "$log" | tail -1 | cut -d' ' -f1 || true)
    # the generated text is everything after the last header line; hash the
    # whole log minus timing lines as a stable byte-identity proxy
    local sha
    sha=$(grep -v "tok/s\|prefill\|sync\|fetch" "$log" | sha256sum | cut -d' ' -f1)
    echo "$(date -u +%FT%TZ),$HOLD,$pair,$mode,$secs,${toks:-NA},$sha" >>"$CSV"
    echo "  pair $pair $mode: ${secs}s ${toks:-?} tok/s"
}

echo "battery: hold=$HOLD pairs=$PAIRS peer=$BOOT maxtok=$MAXTOK"
for pair in $(seq 1 "$PAIRS"); do
    # alternate order so peer-side page-cache warmth cannot favor one mode
    if [ $((pair % 2)) -eq 1 ]; then
        one_run base "$pair"
        one_run pregate "$pair"
    else
        one_run pregate "$pair"
        one_run base "$pair"
    fi
done

echo "--- $CSV ---"
tail -n $((PAIRS * 2)) "$CSV"
echo "check: every text_sha256 above must be identical; if not, the runs"
echo "were not byte-identical and the battery is invalid."
