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
#   LOOM_ADVERTISE  host:port peers can dial you on        (default unset:
#                   fine behind NAT; set it if you are publicly reachable
#                   so you can serve shards to others)
set -eu

if ! command -v loom >/dev/null 2>&1; then
    echo "loom not found -- installing latest release..."
    curl -fsSL https://raw.githubusercontent.com/ch4r10t33r/loom/main/install.sh | sh
fi

ADV=""
[ -n "${LOOM_ADVERTISE:-}" ] && ADV="--advertise ${LOOM_ADVERTISE}"

echo "joining devnet (GLM-4.5-Air, network id 1337)..."
exec loom node \
    --network devnet \
    --hold-fraction "${LOOM_HOLD:-0.2}" \
    --ram-gb "${LOOM_RAM_GB:-4}" \
    --p2p-port "${LOOM_P2P_PORT:-8771}" \
    --rpc-port "${LOOM_RPC_PORT:-8770}" \
    --openai-port "${LOOM_API_PORT:-8772}" \
    --rag \
    $ADV
