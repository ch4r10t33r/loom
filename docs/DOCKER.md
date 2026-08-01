# Docker

A multi-stage [`Dockerfile`](Dockerfile) builds a ReleaseSafe binary (Zig 0.16.0
is fetched by anyzig at build time) into a ~120 MB Debian-slim runtime image.

A prebuilt image is published to the GitHub Container Registry on every push to
`main`, tagged `latest` and with the commit SHA (pin the SHA for anything you
need to reproduce or roll back):

```sh
docker pull ghcr.io/ch4r10t33r/loom:latest
```

Only images that pass the CI smoke test are published, and the image pushed is
the one that was tested, not a rebuild. To build locally instead:

```sh
docker build -t loom:dev .

# run a single node (serves the built-in synthetic model).
# Ports are published to 127.0.0.1: the RPC and OpenAI surfaces have no TLS and
# no authentication, so do not expose them on a routable interface without an
# authenticating proxy in front.
docker run --rm -p 127.0.0.1:8770:8770 -p 127.0.0.1:8772:8772 loom:dev \
  node --rpc-addr 0.0.0.0 --openai-port 8772
curl -s localhost:8772/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

The image exposes `8770` (RPC), `8771` (P2P), `8772` (OpenAI), runs as a non-root
user, and persists the model cache in the `/home/loom/.cache/loom` volume. The
base image is pinned by digest and the build verifies the compiler tarball's
SHA-256, so neither a moved tag nor a swapped release asset can change what is
built. The compose demo additionally drops all capabilities, sets
`no-new-privileges`, and caps memory/PIDs.

**Distributed swarm** — [`docker-compose.yml`](docker-compose.yml) brings up a
two-node swarm (an origin that generates + serves a synthetic deepseek2 GGUF, and
a partial node that holds ~30% of the experts and fetches the rest from the
origin at token time), no external model needed:

```sh
docker compose up --build
# node2's OpenAI port; a hit_rate < 1 on the RPC response = token-loop peer fetch
curl -s localhost:8782/v1/completions -d '{"prompt":"the","max_tokens":8}'
```

Peers can be addressed by IP **or hostname** — the p2p layer resolves names via a
minimal DNS client ([`src/p2p/dns.zig`](src/p2p/dns.zig): IP literal, then
`/etc/hosts`, then a UDP A-query to `/etc/resolv.conf`'s first nameserver), so the
compose demo addresses peers by Compose service name (`origin`), and Kubernetes
Service DNS works the same way.
