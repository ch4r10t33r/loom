# Repository layout

```
whitepaper/   the living whitepaper (updated with every design decision)
spec/         p2p-layer specification (SPEC.md)
docs/         roadmap and planning documents
src/main.zig  CLI entry point
src/core/     primitives: hashing/Merkle, tensor math, int4 quant, stats, iobench
src/engine/   the loom-format MoE engine (MLA, router, expert cache, checkpoints)
src/gguf/     GGUF plane: parser, GGML + IQ kernels, MLA + GQA engines, shared MoE routing, BPE
src/p2p/      distribution: wire frames, gossip, committees, sync, token-loop fetch
src/node/     the daemon: node orchestration, RPC server, model resolver
```

## Source map

| Module | Role |
|---|---|
| `engine/model.zig` | `ModelConfig`; GLM-5.2 shape + runnable `tiny` shape; expert/working-set sizing |
| `core/quant.zig` | loom int4 expert format (f32 scale/32 weights) quantize + fused matvec |
| `core/tensor.zig` | RMSNorm, softmax, SwiGLU, dense matvec, partial RoPE |
| `engine/attention.zig` | MLA: q/kv-LoRA, partial RoPE, compressed-latent KV cache |
| `engine/moe.zig` | DeepSeek-V3 sigmoid router (top-k + shared expert); streamed int4 expert FFN |
| `engine/checkpoint.zig` | loom on-disk format: content-addressed, Merkle-rooted, deduped |
| `engine/expert_cache.zig` | tiered expert cache: pinned hot-set → LRU → pread, usage stats, digest verify |
| `engine/forward.zig` / `engine/engine.zig` | forward step wiring; engine lifecycle, RAM-budget → cache sizing |
| `node/node.zig` | `loom node` orchestration: model → engine → RPC/P2P/gossip/repair |
| `node/hf.zig` | model resolver: local dir / synthetic / Hugging Face download (local-first) |
| `node/generator.zig` | generation abstraction over the loom-format engine and the distributed GGUF engines (MLA and GQA); both serve paths call it |
| `gguf/chat_template.zig` | per-model chat-template detection + rendering for OpenAI `messages[]` (deepseek/chatml/llama2/llama3/gemma/mistral/generic) |
| `gguf/special.zig` | special-token matcher: splices control / user-defined tokens (chat markers) to atomic ids during BPE + SPM encoding |
| `node/rpc.zig` | JSON-over-TCP inference server (concurrent connections, serialized generate) |
| `node/openai.zig` | OpenAI-compatible HTTP API: `/v1/chat/completions`, `/v1/completions`, `/v1/models` (shares the engine + meter with `rpc.zig`) |
| `node/light.zig` | light-node native-RPC delegator (forces client id, round-robin failover) |
| `node/light_openai.zig` | light-node OpenAI delegator: metered reverse proxy to full-node OpenAI endpoints |
| `p2p/p2p.zig` | P2P line protocol: expert directory, weight ranges, gossip |
| `p2p/weights.zig` | range-sharded GGUF store: manifest, version id, holdings/wanted bitmaps, verified IO |
| `p2p/sync.zig` | peer sync client: manifest adoption, root verification, multi-peer range fetch |
| `p2p/peers.zig` | dynamic peer table shared by gossip/repair/P2P threads |
| `p2p/gossip.zig` | 3 s gossip loop: announce self, merge peers-of-peers |
| `p2p/dns.zig` | minimal DNS resolver for peer hostnames (IP literal / /etc/hosts / UDP A-query); Zig std has none |
| `gguf/gguf.zig` | GGUF v2/v3 parser (metadata incl. tokenizer arrays, tensor table) + fixture writer |
| `gguf/ggml.zig` | GGML kernels: F32/F16/Q4_0/Q8_0 fused matvec + row dequant |
| `gguf/llama.zig` | llama-arch engine over mmap'd GGUF: GQA, NORM RoPE, SwiGLU, SPM tokenizer |
| `gguf/deepseek.zig` | deepseek2-arch engine (Kimi/DeepSeek/GLM): MLA + MoE routing over mmap'd GGUF |
| `core/stats.zig` | RSS, usage histograms, STATS→PIN hot-set selection |
| `core/iobench.zig` | parallel random-read disk profiler |
| `engine/gen_checkpoint.zig` | deterministic synthetic checkpoint generator |
| `core/hash.zig` | SHA-256 content addressing + Merkle root |
| `engine/sampler.zig` / `tokenizer.zig` | greedy/temperature sampling; byte-level tokenizer (synthetic model) |
