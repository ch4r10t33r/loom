#!/usr/bin/env bash
# Churn battery: kill a holder mid-generation, measure the observer's
# recovery. Run ON the observer box (a synced joiner). Needs a second
# joiner (the victim) that this box can ssh to, and the bootnode alive so
# nothing is stranded (>= 2 sources per range, the availability invariant).
#
#   HOLD=0.5 VICTIM=root@1.2.3.4 bash 30-churn-battery.sh 3
#
# Per trial: one control run (no kill) and one churn run (victim killed
# KILL_AFTER seconds in). Appends to sys-eval-out/churn-results.csv:
#   ts,trial,kind,secs,repair_lines,striped_lines
set -euo pipefail

TRIALS="${1:-3}"
HOLD="${HOLD:?set HOLD to the snapshot fraction}"
VICTIM="${VICTIM:?set VICTIM to the ssh target of the holder to kill}"
VICTIM_RESTART="${VICTIM_RESTART:-1}" # restart victim's node after each trial
BOOT="${BOOT:-168.119.167.242:8771}"
VICTIM_P2P="${VICTIM_P2P:-}" # host:port of victim's p2p, appended to --peers
KILL_AFTER="${KILL_AFTER:-20}"
MAXTOK="${MAXTOK:-256}"
PROMPT="${PROMPT:-Explain how a mixture-of-experts language model routes tokens to experts, and why caching the hot experts matters for serving throughput.}"
STORE=~/.cache/loom/models/gguf-synced
SNAP=~/store-snap-"$HOLD"
OUT=~/sys-eval-out
CSV="$OUT/churn-results.csv"
mkdir -p "$OUT"
[ -f "$CSV" ] || echo "ts,trial,kind,secs,repair_lines,striped_lines" >"$CSV"

PEERS="$BOOT"
[ -n "$VICTIM_P2P" ] && PEERS="$VICTIM_P2P,$BOOT"

gen() { # kind: control | churn
    local kind="$1" trial="$2"
    rm -rf "$STORE"
    cp -a "$SNAP" "$STORE"
    local log="$OUT/churn-$trial-$kind.log"
    local t0=$SECONDS
    /usr/local/bin/loom gguf run "$STORE" --peers "$PEERS" \
        --prompt "$PROMPT" --max-tokens "$MAXTOK" --temp 0 >"$log" 2>&1 &
    local genpid=$!
    if [ "$kind" = churn ]; then
        sleep "$KILL_AFTER"
        # kill the victim's node by exact binary path, never by pattern
        ssh -o BatchMode=yes "$VICTIM" \
            'kill $(ps aux | awk "\$11 == \"/usr/local/bin/loom\" {print \$2}" | head -1)' || true
    fi
    wait "$genpid"
    local secs=$((SECONDS - t0))
    local repairs striped
    repairs=$(grep -c "repair" "$log" || true)
    striped=$(grep -c "striped" "$log" || true)
    echo "$(date -u +%FT%TZ),$trial,$kind,$secs,$repairs,$striped" >>"$CSV"
    echo "  trial $trial $kind: ${secs}s (repair=$repairs striped=$striped)"
    if [ "$kind" = churn ] && [ "$VICTIM_RESTART" = 1 ]; then
        ssh -o BatchMode=yes "$VICTIM" \
            'nohup /usr/local/bin/loom node --network devnet --hold-fraction '"$HOLD"' --ram-gb 4 >> ~/node.log 2>&1 & disown' || true
        sleep 30 # let it come back before the next trial
    fi
}

echo "churn battery: trials=$TRIALS victim=$VICTIM kill_after=${KILL_AFTER}s"
for t in $(seq 1 "$TRIALS"); do
    gen control "$t"
    gen churn "$t"
done
tail -n $((TRIALS * 2)) "$CSV"
