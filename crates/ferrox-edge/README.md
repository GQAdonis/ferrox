# ferrox-edge

Edge-native MoE serving policy for Ferrox.

A Rust port of the host-side decision logic in
[FreeToken](https://github.com/FlashML-org/FreeToken) (Apache-2.0), the
edge-native MoE serving engine described in
[*FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive
Execution*](https://arxiv.org/abs/2608.16157). What lives here is the
part of that engine that is pure policy — no tensors, no CUDA, no
allocator: the arithmetic that decides *what* to compute where, and the
state machines that turn a token stream back into an agent-shaped
response.

- `qstar` — the bandwidth-adaptive `q*` split: how many of a step's
  expert-cache misses to fetch over PCIe versus run on the CPU
- `expert_cache` — the global LRU expert cache and its admission rules
- `radix` — the prefix-reuse radix cache (plus SWA / hybrid variants)
- `pool` — page pools and elastic re-sizing between expert cache and KV
- `scheduler` — admission, chunked prefill, and retraction policy
- `parser` — reasoning-content and tool-call parsers, streaming-safe
- `effort`, `detokenize`, `cache_report` — the serving-edge details

Nothing here allocates device memory or touches a model. Each module
takes measured numbers (bandwidths, byte costs, pool sizes) and returns
a decision, so all of it is testable on any host.

See [`docs/THIRD_PARTY_NOTICES.md`](../../docs/THIRD_PARTY_NOTICES.md)
for the upstream attribution.
