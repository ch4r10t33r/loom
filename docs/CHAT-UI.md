# Chat UI

`loom node` serves a small chat app at **`http://127.0.0.1:8555`** (change with
`--ui-port`, disable with `--ui-port 0`). It is a single page compiled into the
binary, so there is nothing to install and nothing to configure:

```sh
loom node --gguf model.gguf
# then open http://127.0.0.1:8555
```

Streaming replies, temperature and max-token controls, and per-response
timings that separate the two rates worth knowing apart — **decode speed** and
**time to first token**. On a partial node the first token also pays for every
expert fetch the prefill needed, so folding them together understates
steady-state throughput badly.

The header shows the live peer count and the local-hit rate, polled every few
seconds, so you can watch a node discover peers while you use it.

It runs on its own listener but shares the HTTP implementation and the
generator with the OpenAI API, so the page is same-origin with the endpoint it
calls: no CORS, and no host to configure in the page. Like the API it has **no
TLS and no authentication**, so it binds to loopback by default; do not widen
`--ui-addr` without an authenticating proxy in front.

If the node is serving one of loom's synthetic fixtures, the UI says so. Those
have random weights and answer with meaningless text by construction, which
otherwise reads as a broken model rather than one that was never trained.

## Node console

A running node prints a status line at a fixed interval (`--status-secs`,
default 30, `0` to disable), because the interesting facts change without any
request arriving:

```
status  peers 3  committee 2  shards 4021/6289 (63.9%)  local-hit 87.4%  up 12m
```

Stable field order, so `grep` and `awk` work on it.
