# ferrox-edge

Edge-native MoE serving policy for Ferrox.

A Rust port of the host-side decision logic in
[FreeToken](https://github.com/FlashML-org/FreeToken) (Apache-2.0), the
edge-native MoE serving engine described in *FreeToken: Efficient
Edge-Native MoE Serving with Bandwidth-Adaptive Execution*
([arXiv:2608.16157](https://arxiv.org/abs/2608.16157)). What lives here
is the part of that engine that is pure policy — no tensors, no CUDA, no
allocator: the arithmetic that decides *what* to compute where, and the
state machines that turn a token stream back into an agent-shaped
response.

| Module | Decides |
|---|---|
| `qstar` | how many of a step's expert-cache misses to fetch over PCIe vs. run on the CPU |
| `expert_cache` | which experts stay resident, as one global LRU over a flat `(layer, expert)` id space |
| `radix` | which prefix of a prompt is already computed (`plain`, `swa`, `hybrid`) |
| `pool` | how VRAM splits between the expert cache and KV, and how it is re-split live |
| `placement` | which layers decode on the CPU when the expert banks exceed the host's pinning budget |
| `scheduler` | admission, chunked-prefill sizing, and what a chunk reserves |
| `parser` | where reasoning ends and the answer begins; which tool was called, in which format |
| `effort` | which reasoning-effort dialect a checkpoint speaks |
| `detokenize` | what text is safe to stream after one more token |
| `cache_report` | what to show a human about all of the above |

Every module takes measured numbers (bandwidths, byte costs, pool
sizes, token counts) and returns a decision, so all of it runs in a unit
test on any host, with no GPU and no model.

Where ferrox already had a mechanism this port would have duplicated —
content-addressed KV blocks (`ferrox-core::kv_block`), the SSD expert
tier (`ferrox-core::expert_store`), continuous batching
(`ferrox-server::batch_scheduler`) — these modules plug into it instead
of shadowing it. Each module's docs say which.

See [`docs/THIRD_PARTY_NOTICES.md`](../../docs/THIRD_PARTY_NOTICES.md)
for the upstream attribution and the file-by-file provenance, and
[`docs/ROADMAP.md`](../../docs/ROADMAP.md) for what is wired in and what
is still groundwork.
