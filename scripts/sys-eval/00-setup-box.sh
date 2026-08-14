#!/usr/bin/env bash
# Prepare a fresh measurement box: install the latest loom release binary
# and the pre-gate head asset. Idempotent. Run ON the box.
set -euo pipefail

LOOM_VERSION="${LOOM_VERSION:-latest}"
PREGATE_TAG="${PREGATE_TAG:-v0.40.3}" # release that carries pregate-qwen3.lpg
OUT=~/sys-eval-out
mkdir -p "$OUT"

arch=$(uname -m)
case "$arch" in
x86_64)
    # prefer the AVX2 build; every Hetzner CPX qualifies
    asset="x86_64-v3-linux"
    grep -q avx2 /proc/cpuinfo || asset="x86_64-linux"
    ;;
aarch64) asset="aarch64-linux" ;;
*)
    echo "unsupported arch $arch" >&2
    exit 1
    ;;
esac

if [ "$LOOM_VERSION" = "latest" ]; then
    LOOM_VERSION=$(curl -fsSL https://api.github.com/repos/ch4r10t33r/loom/releases/latest | grep -om1 '"tag_name": *"[^"]*"' | cut -d'"' -f4)
fi
echo "installing loom $LOOM_VERSION ($asset)"

tmp=$(mktemp -d)
curl -fsSL -o "$tmp/loom.tar.gz" \
    "https://github.com/ch4r10t33r/loom/releases/download/$LOOM_VERSION/loom-$LOOM_VERSION-$asset.tar.gz"
tar xzf "$tmp/loom.tar.gz" -C "$tmp"
install -m 755 "$(find "$tmp" -name loom -type f | head -1)" /usr/local/bin/loom
rm -rf "$tmp"
loom version | head -1

if [ ! -f ~/pregate-qwen3.lpg ]; then
    echo "fetching pre-gate head ($PREGATE_TAG asset)"
    curl -fsSL -o ~/pregate-qwen3.lpg \
        "https://github.com/ch4r10t33r/loom/releases/download/$PREGATE_TAG/pregate-qwen3.lpg"
fi
ls -la ~/pregate-qwen3.lpg

# link-speed probe against the bootnode's release mirror path; the battery
# CSV wants this number attached to every run
BOOT="${BOOT:-168.119.167.242}"
echo "probing link to bootnode $BOOT ..."
ping -c 3 -q "$BOOT" | tail -1 | tee "$OUT/link-probe.txt" || true
echo "setup done"
