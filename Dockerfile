# syntax=docker/dockerfile:1
#
# Multi-stage build for the `loom` node. The build toolchain is Zig 0.16.0,
# fetched on demand by anyzig (matching the project's local workflow); the
# runtime image carries only the compiled binary.

# ---- builder: fetch Zig 0.16.0 via anyzig, compile a ReleaseSafe binary ----
# Base pinned by digest, not tag (security issue #33): a tag is mutable, so
# an unpinned base silently changes what is built and shipped.
# debian:bookworm-slim as of 2026-07-27. Re-pin when refreshing for CVEs.
FROM debian@sha256:7b140f374b289a7c2befc338f42ebe6441b7ea838a042bbd5acbfca6ec875818 AS builder

# set by BuildKit: "amd64" | "arm64"
ARG TARGETARCH
ARG ANYZIG_VERSION=v2026_03_26
# Checksums of the release assets (security issue #33). ANYZIG_VERSION is a git
# tag, which is mutable: without verification a moved tag or a replaced asset
# silently swaps the compiler that builds the shipped binary.
ARG ANYZIG_SHA256_x86_64=f9d5a09fbd7c019eecef1a397613ce5baec22872a1c3eb5ab4b1132e917c3d71
ARG ANYZIG_SHA256_aarch64=1963afb44ca0705768cba7346fc649b5b56879c5c6ab91303bc8808604ab3a3c

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
    case "${azarch}" in \
        x86_64)  want="${ANYZIG_SHA256_x86_64}" ;; \
        aarch64) want="${ANYZIG_SHA256_aarch64}" ;; \
    esac; \
    curl -fsSL -o /tmp/anyzig.tar.gz \
        "https://github.com/marler8997/anyzig/releases/download/${ANYZIG_VERSION}/anyzig-${azarch}-linux.tar.gz"; \
    echo "${want}  /tmp/anyzig.tar.gz" | sha256sum -c -; \
    tar -xz -C /usr/local/bin -f /tmp/anyzig.tar.gz; \
    rm -f /tmp/anyzig.tar.gz; \
    chmod +x /usr/local/bin/zig

WORKDIR /src
# only the inputs the build needs (see build.zig.zon .paths)
COPY build.zig build.zig.zon ./
COPY src ./src

# `zig 0.16.0` triggers anyzig to fetch Zig 0.16.0; the build then fetches the
# snappy dependency (git+https) and compiles.
ARG COMMIT=unknown
RUN zig 0.16.0 build -Doptimize=ReleaseSafe -Dcommit="$COMMIT"

# ---- runtime: minimal image with just the binary ----
FROM debian@sha256:7b140f374b289a7c2befc338f42ebe6441b7ea838a042bbd5acbfca6ec875818 AS runtime

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
