# Install

Prebuilt binaries are attached to each
[release](https://github.com/ch4r10t33r/loom/releases). No runtime
dependencies; the Linux builds are statically linked against musl, so they run
on any distro.

| platform | asset suffix |
|---|---|
| macOS, Apple silicon | `aarch64-macos` |
| macOS, Intel | `x86_64-macos` |
| Linux, x86-64 | `x86_64-linux` |
| Linux, arm64 | `aarch64-linux` |

Windows is not built yet.

One line, which detects your platform, verifies the download against the
release checksums, and installs to `/usr/local/bin`:

```sh
curl -fsSL https://raw.githubusercontent.com/ch4r10t33r/loom/main/install.sh | sh
```

It takes `--version vX.Y.Z` and `--dir PATH`, or the same as `LOOM_VERSION` and
`LOOM_INSTALL_DIR`. On Windows it stops with an explanation and points at WSL2,
where the Linux build works. A `GITHUB_TOKEN` (or signed-in `gh`) is
optional and only raises the GitHub API rate limit.

By hand instead:

```sh
VER=v0.1.0; PLAT=aarch64-macos       # pick yours from the table
curl -fsSL -O https://github.com/ch4r10t33r/loom/releases/download/$VER/loom-$VER-$PLAT.tar.gz
curl -fsSL -O https://github.com/ch4r10t33r/loom/releases/download/$VER/SHA256SUMS
shasum -a 256 -c SHA256SUMS --ignore-missing
tar -xzf loom-$VER-$PLAT.tar.gz
sudo mv loom-$VER-$PLAT/loom /usr/local/bin/
loom version
```

`loom version` reports the version, the commit it was built from, and its
target triple. Include that output in any bug report.

On macOS, Gatekeeper quarantines unsigned downloads. Clear it with
`xattr -d com.apple.quarantine /usr/local/bin/loom`, or right-click and Open
once. The binaries are not code-signed or notarized.

Docker images are published to `ghcr.io/ch4r10t33r/loom:latest` on every push
to `main`.
