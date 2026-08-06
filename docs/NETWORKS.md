# Pre-configured networks

Loom ships three named networks, in the style of Ethereum's chain registry.
A name resolves to a stable network id (a protocol constant; changing it
partitions the network) and the canonical model that network serves. Every
checkpoint listed here passed `loom gguf check` against this codebase.

| network | id | model | arch | phase |
|---|---|---|---|---|
| `devnet` | 1337 | [Qwen3-30B-A3B](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF) Q2_K (~11 GB) | qwen3moe | PoC validation; runs distributed on commodity boxes today |
| `testnet` | 2 | [GLM-4.6](https://huggingface.co/unsloth/GLM-4.6-GGUF) Q4_K_M (~200 GB) | glm4moe | performance tier; one 256 GB box or a small cluster |
| `mainnet` | 1 | [GLM 5.2](https://huggingface.co/unsloth/GLM-5.2-GGUF) UD-Q4_K_XL (~372-475 GB) | glm-dsa | production; needs the glm-dsa engine (see docs/GLM52-READINESS.md); Kimi K3 is the recorded alternative |

```sh
loom node --network devnet --gguf Qwen3-30B-A3B-Q2_K.gguf ...
```

`--network` sets the id; peers on a different id are refused at the p2p
layer (`ERR wrong_network`). **Arch policy:** one network serves one model.
On `testnet` and `mainnet` a node refuses to start with a model whose
architecture differs from the network's. `devnet` only warns, since a PoC
network is where mismatches get discovered on purpose. The rest of the
system depends on the one-network-one-model invariant: RAG gossips text
only because every node embeds identically, and expert fetch assumes a
shared manifest lineage.

An explicit `--network-id N` still works for private networks; combining it
with a conflicting `--network` name is an error.

## Phases

1. **devnet / Qwen3-30B-A3B**: validate every claim end to end (sharded
   serving, membership gating, RAG, churn) at 30B on existing hardware.
2. **testnet / GLM-4.6**: the same matrix at 357B under real load, plus
   performance measurement, on 256 GB-class hardware.
3. **mainnet / GLM 5.2**: the thesis model, once the glm-dsa engine
   variant lands and hardware is provisioned. Kimi K3 remains a candidate
   if its checkpoint/engine economics win.
