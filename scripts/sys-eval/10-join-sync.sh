#!/usr/bin/env bash
# Join the devnet at a given hold-fraction, wait for the bootstrap sync to
# finish, then snapshot the store for the A/B battery. Run ON the box.
#
#   bash 10-join-sync.sh 0.2
#
# Produces ~/store-snap-<F> and a meta line in sys-eval-out/join-meta.csv.
# Requires 00-setup-box.sh first. The sync pulls from the HF mirror by
# default (fast, digest-verified); the battery itself always fetches from
# the bootnode peer, so the mirror does not contaminate the measurement.
set -euo pipefail

F="${1:?usage: 10-join-sync.sh <hold-fraction>}"
STORE=~/.cache/loom/models/gguf-synced
SNAP=~/store-snap-"$F"
OUT=~/sys-eval-out
LOG="$OUT/join-$F.log"
mkdir -p "$OUT"

# fresh store: the point of #252's fix is that a rejoin at a new fraction
# re-assigns, but a clean slate keeps every battery independent
rm -rf "$STORE" "$SNAP"

nohup /usr/local/bin/loom node --network devnet --hold-fraction "$F" --ram-gb 4 >"$LOG" 2>&1 &
disown

echo "syncing at hold-fraction $F (log: $LOG)..."
deadline=$((SECONDS + 5400))
until grep -q "serving\.\.\." "$LOG" 2>/dev/null; do
    if [ $SECONDS -gt $deadline ]; then
        echo "sync did not finish in 90 min; tail of log:" >&2
        tail -5 "$LOG" >&2
        exit 1
    fi
    sleep 15
done

# stop by exact binary path -- never pkill by pattern
PID=$(ps aux | awk '$11 == "/usr/local/bin/loom" {print $2}' | head -1)
kill "$PID" 2>/dev/null || true
sleep 3
kill -9 "$PID" 2>/dev/null || true

held=$(grep -o "held=[0-9]* ([0-9.]*%)" "$LOG" | tail -1)
echo "sync done: $held"
grep -E "joined committee|synced .* shards" "$LOG" | tail -3

cp -a "$STORE" "$SNAP"
echo "$(date -u +%FT%TZ),$F,$held,$(du -sh "$SNAP" | cut -f1)" >>"$OUT/join-meta.csv"
echo "snapshot at $SNAP"
