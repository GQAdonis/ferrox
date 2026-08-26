# ferrox-edge

Edge-native MoE serving policy for Ferrox.

A Rust port of the host-side decision logic in
[FreeToken](https://github.com/FlashML-org/FreeToken) (Apache-2.0), the
edge-native MoE serving engine described in *FreeToken: Efficient
Edge-Native MoE Serving with Bandwidth-Adaptive Execution*
([arXiv:2608.16157](https://arxiv.org/abs/2608.16157)). What lives here
is the part of that engine that is pure policy, with no tensors, no CUDA
and no allocator: the arithmetic that decides *what* to compute where,
and the state machines that turn a token stream back into an
agent-shaped response.

Modules that drive something in `ferrox-server` or `ferrox-cli` today:

| Module | Decides |
|---|---|
| `parser` | where reasoning ends and the answer begins, and which tool was called in which format |
| `detokenize` | what text is safe to stream after one more token |
| `radix` | which prefix of a prompt is already computed (`plain`, `swa`, `hybrid`) |
| `scheduler` | admission, chunked-prefill sizing, and what a chunk reserves |
| `effort` | which reasoning-effort dialect a checkpoint speaks |
| `stats` | what a server may honestly claim about its own throughput and latency |
| `maintenance` · `rebuild` · `outbox` · `footprint` | whether a rebuild or a stop may proceed, whether it rolls back, what the receipt is worth, and what this process really occupies |
| `pool` | how VRAM splits between the expert cache and KV, and how it is re-split live |
| `dsv4` | per-layer KV tier sizing, and which compressor each layer runs |
| `bench_profile` · `bench_client` | when a measured bandwidth profile may be trusted, and what a serving benchmark may report |

Complete and tested with no consumer yet: `qstar` (the bandwidth split),
`expert_cache`, `placement`, `residency`, `cache_manager`,
`cache_report`, `anchor`, `window_pool`, `state_pool` and `supervisor`.
`expert_slots` sits between the two, with a host implementation of its
`SlotDevice` trait and a compile-verified CUDA one.

Every module takes measured numbers (bandwidths, byte costs, pool
sizes, token counts) and returns a decision, so all of it runs in a unit
test on any host, with no GPU and no model.

Where ferrox already had a mechanism this port would have duplicated,
such as content-addressed KV blocks (`ferrox-core::kv_block`), the SSD
expert tier (`ferrox-core::expert_store`) and continuous batching
(`ferrox-server::batch_scheduler`), these modules plug into it instead
of shadowing it. Each module's docs say which.

See [`docs/THIRD_PARTY_NOTICES.md`](../../docs/THIRD_PARTY_NOTICES.md)
for the upstream attribution and the file-by-file provenance, and
[`docs/ROADMAP.md`](../../docs/ROADMAP.md) for what is wired in and what
is still groundwork.
