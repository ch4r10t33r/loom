#!/bin/sh
# One command to join the loom devnet (GLM-4.5-Air, network id 1337):
#
#   curl -fsSL https://raw.githubusercontent.com/ch4r10t33r/loom/main/scripts/join-devnet.sh | sh
#
# Installs loom if it is not on PATH (via the release installer), then
# starts a node that bootstraps from the devnet's registry bootnode, syncs
# its share of the expert shards from peers (no model download needed), and
# serves an OpenAI-compatible API locally.
#
# Tunables (environment):
#   LOOM_HOLD       fraction of expert shards to hold      (default 0.2)
#   LOOM_RAM_GB     RAM budget for the expert cache        (default 4)
#   LOOM_P2P_PORT   p2p port                               (default 8771)
#   LOOM_RPC_PORT   native RPC port                        (default 8770)
#   LOOM_API_PORT   OpenAI-compatible API port             (default 8772)
#   LOOM_HTTP_MIRROR URL of a static mirror of the devnet model file;
#                   initial sync pulls shard ranges from it in parallel
#                   (digest-verified) instead of loading the bootnode
#   LOOM_ADVERTISE  host:port peers can dial you on        (default unset:
#                   fine behind NAT; set it if you are publicly reachable
#                   so you can serve shards to others)
#   LOOM_NO_METRICS set to 1 to disable the alpha telemetry report
#                   (numeric operational stats only -- never prompts or
#                   text; the full field list is in docs/ALPHA.md)
set -eu

# Where the binary actually lives. `command -v` alone is not enough: when the
# installer cannot write /usr/local/bin and cannot sudo, it falls back to
# ~/.local/bin, which is typically NOT on the PATH of the shell running this
# script -- so a fresh install would end with "exec: loom: not found" (hit by
# an alpha tester). Check the installer's fallback locations explicitly.
find_loom() {
    if command -v loom >/dev/null 2>&1; then
        command -v loom
        return 0
    fi
    for p in "$HOME/.local/bin/loom" /usr/local/bin/loom; do
        if [ -x "$p" ]; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

LOOM="$(find_loom || true)"
if [ -z "$LOOM" ]; then
    echo "loom not found -- installing latest release..."
    curl -fsSL https://raw.githubusercontent.com/ch4r10t33r/loom/main/install.sh | sh
    LOOM="$(find_loom || true)"
    if [ -z "$LOOM" ]; then
        echo "install finished but no loom binary found (looked on PATH, ~/.local/bin, /usr/local/bin)" >&2
        exit 1
    fi
else
    # Upgrade in place when a newer release exists. Best-effort: if the
    # GitHub API is unreachable or rate-limited, run what is installed.
    latest=$(curl -fsSL --max-time 10 https://api.github.com/repos/ch4r10t33r/loom/releases/latest 2>/dev/null \
        | sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p')
    have=$("$LOOM" version 2>/dev/null | sed -n 's/^loom \([0-9][0-9.]*\).*/\1/p' | head -1)
    if [ -n "$latest" ] && [ -n "$have" ] && [ "$latest" != "$have" ]; then
        echo "upgrading loom v$have -> v$latest..."
        curl -fsSL https://raw.githubusercontent.com/ch4r10t33r/loom/main/install.sh | sh
        LOOM="$(find_loom)"
    fi
fi

ADV=""
[ -n "${LOOM_ADVERTISE:-}" ] && ADV="--advertise ${LOOM_ADVERTISE}"

MIRROR=""
[ -n "${LOOM_HTTP_MIRROR:-}" ] && MIRROR="--bootstrap-http ${LOOM_HTTP_MIRROR}"

METRICS="--report-metrics"
if [ "${LOOM_NO_METRICS:-}" = "1" ]; then
    METRICS=""
else
    echo "alpha metrics: on (numeric stats only; opt out with LOOM_NO_METRICS=1; docs/ALPHA.md)"
fi

echo "joining devnet (GLM-4.5-Air, network id 1337)..."
exec "$LOOM" node \
    --network devnet \
    --hold-fraction "${LOOM_HOLD:-0.2}" \
    --ram-gb "${LOOM_RAM_GB:-4}" \
    --p2p-port "${LOOM_P2P_PORT:-8771}" \
    --rpc-port "${LOOM_RPC_PORT:-8770}" \
    --openai-port "${LOOM_API_PORT:-8772}" \
    --rag \
    $METRICS \
    $MIRROR \
    $ADV
