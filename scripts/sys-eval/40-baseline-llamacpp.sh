#!/usr/bin/env bash
# Baseline: llama.cpp on the same GGUF, same greedy 96-token prompt.
# (a) single-node with the model paging from disk -- the no-network floor;
# (b) optionally, llama.cpp RPC mode split across boxes -- the
#     pipeline-parallel alternative the whitepaper argues against.
# Run ON the box.
#
#   bash 40-baseline-llamacpp.sh                # build + single-node run
#   RPC_SERVERS=10.0.0.2:50052 bash 40-baseline-llamacpp.sh   # add RPC run
#     (start on each worker first:  ./rpc-server -H 0.0.0.0 -p 50052)
#
# Appends to sys-eval-out/baseline-results.csv: ts,kind,secs,tok_s
set -euo pipefail

LLAMA_TAG="${LLAMA_TAG:-b6100}" # pin for reproducibility; bump deliberately
MODEL="${MODEL:-$HOME/models/qwen3-30b-a3b-q2k.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q2_K.gguf}"
MAXTOK="${MAXTOK:-96}"
PROMPT="${PROMPT:-Explain how a mixture-of-experts language model routes tokens to experts, and why caching the hot experts matters for serving throughput.}"
OUT=~/sys-eval-out
CSV="$OUT/baseline-results.csv"
mkdir -p "$OUT"
[ -f "$CSV" ] || echo "ts,kind,secs,tok_s" >"$CSV"

if [ ! -d ~/llama.cpp ]; then
    apt-get update -qq && apt-get install -y -qq git cmake build-essential curl
    git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp ~/llama.cpp
    # RPC backend must be enabled at build time for run (b)
    cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build ~/llama.cpp/build --config Release -j"$(nproc)" -t llama-cli rpc-server
fi
CLI=~/llama.cpp/build/bin/llama-cli

if [ ! -f "$MODEL" ]; then
    mkdir -p "$(dirname "$MODEL")"
    echo "downloading model (~11 GB)..."
    curl -fSL -o "$MODEL" "$MODEL_URL"
fi

run_one() { # kind, extra args...
    local kind="$1"
    shift
    sync
    echo 3 >/proc/sys/vm/drop_caches 2>/dev/null || true
    local log="$OUT/baseline-$kind.log"
    local t0=$SECONDS
    "$CLI" -m "$MODEL" -p "$PROMPT" -n "$MAXTOK" --temp 0 -no-cnv "$@" >"$log" 2>&1
    local secs=$((SECONDS - t0))
    local toks
    toks=$(grep -oE "[0-9]+\.[0-9]+ tokens per second" "$log" | tail -1 | cut -d' ' -f1 || true)
    echo "$(date -u +%FT%TZ),$kind,$secs,${toks:-NA}" >>"$CSV"
    echo "  $kind: ${secs}s ${toks:-?} tok/s"
}

run_one single-node
if [ -n "${RPC_SERVERS:-}" ]; then
    run_one "rpc-$RPC_SERVERS" --rpc "$RPC_SERVERS"
fi
tail -3 "$CSV"
