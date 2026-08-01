#!/bin/sh
# Install loom: detect the platform, download the matching release binary,
# verify it against the release checksums, and put it on PATH.
#
#   curl -fsSL https://raw.githubusercontent.com/ch4r10t33r/loom/main/install.sh | sh
#
# Options (flags or environment):
#   --version vX.Y.Z   LOOM_VERSION       release to install (default: latest)
#   --dir PATH         LOOM_INSTALL_DIR   install location (default: /usr/local/bin)
#                      GITHUB_TOKEN       optional; raises GitHub API rate limits
#
# POSIX sh on purpose: this runs through `sh` from a pipe, on machines that may
# have no bash, no jq and no gh.

set -eu

REPO="ch4r10t33r/loom"
BIN="loom"
API="https://api.github.com/repos/${REPO}"

VERSION="${LOOM_VERSION:-}"
INSTALL_DIR="${LOOM_INSTALL_DIR:-/usr/local/bin}"

# ---- output -----------------------------------------------------------------
# Colour only when stdout is a terminal; piped installs stay plain text.
if [ -t 1 ]; then
    B=$(printf '\033[1m'); R=$(printf '\033[31m'); Y=$(printf '\033[33m'); D=$(printf '\033[2m'); Z=$(printf '\033[0m')
else
    B=''; R=''; Y=''; D=''; Z=''
fi
say()  { printf '%s\n' "$*"; }
step() { printf '%s==>%s %s\n' "$B" "$Z" "$*"; }
warn() { printf '%swarning:%s %s\n' "$Y" "$Z" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$R" "$Z" "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Install the loom binary for this machine.

usage: install.sh [--version vX.Y.Z] [--dir PATH]

  --version vX.Y.Z   release to install (default: the latest release)
  --dir PATH         where to install (default: /usr/local/bin)
  --help             this message

environment:
  LOOM_VERSION, LOOM_INSTALL_DIR   same as the flags above
  GITHUB_TOKEN or GH_TOKEN         optional; raises GitHub API rate limits
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --version) [ $# -ge 2 ] || die "--version needs a value"; VERSION="$2"; shift 2 ;;
        --version=*) VERSION="${1#*=}"; shift ;;
        --dir) [ $# -ge 2 ] || die "--dir needs a value"; INSTALL_DIR="$2"; shift 2 ;;
        --dir=*) INSTALL_DIR="${1#*=}"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
done

# ---- platform ---------------------------------------------------------------
# Note WSL reports Linux and is genuinely supported; a native Windows shell
# (Git Bash, MSYS, Cygwin) is pointed at the prebuilt zip instead.
uname_s=$(uname -s 2>/dev/null || echo unknown)
case "$uname_s" in
    Darwin) os="macos" ;;
    Linux)  os="linux" ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        printf '%serror:%s this installer is for macOS and Linux shells.\n' "$R" "$Z" >&2
        cat >&2 <<EOF

On Windows, download the prebuilt zip instead:

  https://github.com/${REPO}/releases/latest
  (asset: loom-<version>-x86_64-windows.zip — unzip and run loom.exe)

Or run it under WSL2, where the Linux build works and this script will
install it (open a WSL shell and re-run this command).

EOF
        exit 1 ;;
    *) die "unsupported operating system: ${uname_s} (loom ships macOS and Linux builds)" ;;
esac

uname_m=$(uname -m 2>/dev/null || echo unknown)
case "$uname_m" in
    x86_64|amd64)  arch="x86_64" ;;
    arm64|aarch64) arch="aarch64" ;;
    *) die "unsupported architecture: ${uname_m} (loom ships x86_64 and aarch64 builds)" ;;
esac

platform="${arch}-${os}"
step "Platform: ${platform} ${D}(${uname_s} ${uname_m})${Z}"

# ---- prerequisites ----------------------------------------------------------
command -v curl >/dev/null 2>&1 || die "curl is required but not installed"
command -v tar  >/dev/null 2>&1 || die "tar is required but not installed"

# Checksum verification is not optional: this script is meant to be piped into
# a shell, so the download must be provably the released artifact.
if command -v sha256sum >/dev/null 2>&1;  then sha_cmd="sha256sum"
elif command -v shasum >/dev/null 2>&1;   then sha_cmd="shasum -a 256"
elif command -v openssl >/dev/null 2>&1;  then sha_cmd="openssl-dgst"
else die "need one of sha256sum, shasum or openssl to verify the download"
fi
sha256_of() {
    if [ "$sha_cmd" = "openssl-dgst" ]; then
        openssl dgst -sha256 "$1" | sed 's/.*= *//'
    else
        # shellcheck disable=SC2086
        $sha_cmd "$1" | cut -d' ' -f1
    fi
}

# The repository is public; a token is optional but raises the GitHub API
# rate limit. Fall back to the gh CLI's token when one is already signed in.
TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
if [ -z "$TOKEN" ] && command -v gh >/dev/null 2>&1; then
    TOKEN=$(gh auth token 2>/dev/null || true)
fi

api_get() { # url -> stdout
    if [ -n "$TOKEN" ]; then
        curl -fsSL -H "Authorization: Bearer ${TOKEN}" -H "Accept: application/vnd.github+json" "$1"
    else
        curl -fsSL -H "Accept: application/vnd.github+json" "$1"
    fi
}

no_access() {
    if [ -n "$TOKEN" ]; then
        cat >&2 <<EOF
${R}error:${Z} no release ${VERSION:-(latest)} found in ${REPO}.

A token was used, so this is most likely a version that does not exist.
Available releases: https://github.com/${REPO}/releases
EOF
    else
        cat >&2 <<EOF
${R}error:${Z} could not read releases for ${REPO}.

The requested version may not exist, or the anonymous GitHub API rate
limit may be exhausted. Set a token and retry:

    export GITHUB_TOKEN=<any GitHub token>

or sign in with the GitHub CLI (\`gh auth login\`) and re-run this script.
Available releases:
    https://github.com/${REPO}/releases
EOF
    fi
    exit 1
}

# ---- resolve the release ----------------------------------------------------
if [ -z "$VERSION" ]; then
    step "Resolving latest release"
    release_json=$(api_get "${API}/releases/latest" 2>/dev/null) || no_access
    VERSION=$(printf '%s' "$release_json" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
    [ -n "$VERSION" ] || die "could not determine the latest release tag"
else
    release_json=$(api_get "${API}/releases/tags/${VERSION}" 2>/dev/null) || no_access
fi

asset="${BIN}-${VERSION}-${platform}.tar.gz"
step "Installing ${BIN} ${VERSION} ${D}(${asset})${Z}"

# Resolve an asset's API download URL from the release JSON, without jq.
#
# Two things make the naive approach wrong. The release *body* quotes every
# asset filename in its own install instructions, so a document-wide grep for
# the filename matches prose. And the API returns pretty-printed JSON to curl
# but compact JSON to some clients, so neither line-oriented nor
# single-line parsing works on its own.
#
# Normalizing on commas gives one field per line in both shapes. From there:
# remember the most recent asset URL seen (the /releases/assets/N form appears
# only inside the assets array, and is an asset's first field), and print it
# when the matching name field turns up. Stop at the first match, which is
# always inside the assets array and never in the trailing body.
#
# The asset API URL is used rather than browser_download_url; it works for
# public and private repositories alike -- one code path instead of two.
asset_url_of() { # filename -> url
    printf '%s' "$release_json" | tr ',' '\n' | awk -v want="$1" '
        /"url"[[:space:]]*:[[:space:]]*"[^"]*\/releases\/assets\/[0-9]+"/ {
            u = $0
            sub(/^[^:]*:[[:space:]]*"/, "", u)
            sub(/".*$/, "", u)
            last = u
        }
        index($0, "\"name\"") && index($0, "\"" want "\"") {
            if (last != "") { print last; exit }
        }
    '
}

tmp=$(mktemp -d 2>/dev/null || mktemp -d -t loom)
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT INT TERM

fetch_asset() { # filename destination
    _url=$(asset_url_of "$1")
    [ -n "$_url" ] || die "release ${VERSION} has no asset named $1
Available builds are listed at https://github.com/${REPO}/releases/tag/${VERSION}"
    if [ -n "$TOKEN" ]; then
        curl -fsSL -H "Authorization: Bearer ${TOKEN}" -H "Accept: application/octet-stream" -o "$2" "$_url"
    else
        curl -fsSL -H "Accept: application/octet-stream" -o "$2" "$_url"
    fi
}

step "Downloading"
fetch_asset "$asset" "${tmp}/${asset}" || die "download failed"
fetch_asset "SHA256SUMS" "${tmp}/SHA256SUMS" || die "could not download SHA256SUMS (refusing to install unverified)"

step "Verifying checksum"
want=$(grep "  ${asset}\$" "${tmp}/SHA256SUMS" | cut -d' ' -f1 | head -1)
[ -n "$want" ] || die "${asset} is not listed in SHA256SUMS"
got=$(sha256_of "${tmp}/${asset}")
if [ "$want" != "$got" ]; then
    die "checksum mismatch for ${asset}
  expected ${want}
  actual   ${got}
Refusing to install. Re-run; if it persists, report it."
fi
say "    ${D}sha256 ${got}${Z}"

step "Extracting"
tar -xzf "${tmp}/${asset}" -C "$tmp"
src="${tmp}/${BIN}-${VERSION}-${platform}/${BIN}"
[ -f "$src" ] || die "archive did not contain ${BIN}"
chmod +x "$src"

# ---- install ----------------------------------------------------------------
# Prefer the requested directory, escalate with sudo if needed, and fall back
# to a user-owned directory rather than failing outright.
dest="${INSTALL_DIR}/${BIN}"
step "Installing to ${dest}"
if mkdir -p "$INSTALL_DIR" 2>/dev/null && [ -w "$INSTALL_DIR" ]; then
    mv -f "$src" "$dest"
elif command -v sudo >/dev/null 2>&1 && [ -t 0 ]; then
    say "    ${D}${INSTALL_DIR} is not writable; using sudo${Z}"
    sudo mkdir -p "$INSTALL_DIR"
    sudo mv -f "$src" "$dest"
    sudo chmod +x "$dest"
else
    INSTALL_DIR="${HOME}/.local/bin"
    dest="${INSTALL_DIR}/${BIN}"
    warn "cannot write to the install directory and cannot sudo; using ${INSTALL_DIR}"
    mkdir -p "$INSTALL_DIR"
    mv -f "$src" "$dest"
fi

# Unsigned downloads are quarantined by Gatekeeper; clear it so the binary runs
# without a right-click-Open dance. Harmless if xattr is absent.
if [ "$os" = "macos" ]; then
    xattr -d com.apple.quarantine "$dest" >/dev/null 2>&1 || true
fi

# ---- verify -----------------------------------------------------------------
step "Verifying install"
"$dest" version || die "installed binary would not run"

case ":${PATH}:" in
    *":${INSTALL_DIR}:"*) ;;
    *)
        warn "${INSTALL_DIR} is not on your PATH. Add it:"
        say  "    export PATH=\"${INSTALL_DIR}:\$PATH\""
        ;;
esac

cat <<EOF

${B}Installed.${Z} Try:

    ${BIN} --help
    ${BIN} node --gguf <model.gguf>      ${D}# serve a model${Z}

Docs:
    https://github.com/${REPO}/blob/main/docs/CLI.md                  ${D}# every flag${Z}
    https://github.com/${REPO}/blob/main/docs/TWO-MACHINE-TEST.md     ${D}# run a real swarm${Z}
EOF
