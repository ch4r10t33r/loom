# syntax=docker/dockerfile:1
#
# Multi-stage build for the `loom` node. The build toolchain is Zig 0.16.0,
# fetched on demand by anyzig (matching the project's local workflow); the
# runtime image carries only the compiled binary.

# ---- builder: fetch Zig 0.16.0 via anyzig, compile a ReleaseFast binary ----
FROM debian:bookworm-slim AS builder

# set by BuildKit: "amd64" | "arm64"
ARG TARGETARCH
ARG ANYZIG_VERSION=v2026_03_26

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git xz-utils \
    && rm -rf /var/lib/apt/lists/*

# anyzig provides `zig <version>` on demand and downloads the compiler itself.
RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) azarch=x86_64 ;; \
        arm64) azarch=aarch64 ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/marler8997/anyzig/releases/download/${ANYZIG_VERSION}/anyzig-${azarch}-linux.tar.gz" \
        | tar -xz -C /usr/local/bin; \
    chmod +x /usr/local/bin/zig

WORKDIR /src
# only the inputs the build needs (see build.zig.zon .paths)
COPY build.zig build.zig.zon ./
COPY src ./src

# `zig 0.16.0` triggers anyzig to fetch Zig 0.16.0; the build then fetches the
# snappy dependency (git+https) and compiles.
RUN zig 0.16.0 build -Doptimize=ReleaseFast

# ---- runtime: minimal image with just the binary ----
FROM debian:bookworm-slim AS runtime

# ca-certificates: the node downloads models from Hugging Face over HTTPS.
# Pre-create the cache dir owned by `loom` so the anonymous volume inherits its
# ownership (a VOLUME mount point is otherwise created root-owned, which a
# non-root user cannot write).
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 loom \
    && mkdir -p /home/loom/.cache/loom \
    && chown -R loom:loom /home/loom

COPY --from=builder /src/zig-out/bin/loom /usr/local/bin/loom

ENV HOME=/home/loom
USER loom
WORKDIR /home/loom

# model cache: HF downloads, synthetic models, and GGUF stores live here
VOLUME ["/home/loom/.cache/loom"]

# rpc (8770), p2p (8771), openai (8772)
EXPOSE 8770 8771 8772

ENTRYPOINT ["loom"]
# default: serve the built-in synthetic model over RPC on all interfaces
CMD ["node", "--rpc-addr", "0.0.0.0"]
