# Third-party notices

Thanks to the projects whose public designs, file formats, and CLI
conventions shaped ferrox's architecture: llama.cpp, candle, and
others credited below. This file records that influence, plus the
license and copyright notices required wherever real upstream source
was read or adapted.

## GGUF file format

The GGUF binary format (magic, version, key-value metadata section,
tensor descriptor table, alignment padding, block-quantized tensor
data) originates from the ggml / llama.cpp project and is publicly
documented at:

  https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

`ferrox-gguf` is an independent implementation written against that
public specification. ggml and llama.cpp are MIT licensed:

  Copyright (c) 2023-2026 The ggml authors / Georgi Gerganov and
  contributors (llama.cpp)

## safetensors file format

The safetensors binary format (8-byte little-endian header length, UTF-8
JSON header describing each tensor's dtype/shape/byte range, followed by
raw tensor data) is a public format originated and maintained by Hugging
Face, documented and reference-implemented at:

  https://github.com/huggingface/safetensors

`ferrox-safetensors` is an independent implementation written against
that public format. The header-parsing and byte-layout logic is original,
but the specific validation rules it enforces (tensor offsets must be
re-sorted by `data_offsets` before checking they're contiguous starting
at 0, non-overlapping, size-consistent with the declared dtype+shape, and
must exactly cover the rest of the file) were checked directly against
that reference implementation's `Metadata::validate` to make sure a
real, non-obvious edge case (JSON key order not matching offset order)
was handled correctly, not guessed. `huggingface/safetensors` is Apache-2.0
licensed:

  Copyright (c) 2023-2026 Hugging Face and contributors

## Q4_0 / Q8_0 / Q6_K block quantization layouts

The block-quantization conventions dequantized by `ferrox-quant`
(fixed-size blocks of packed low-bit values sharing one f16 scale) are
the public layout used throughout the ggml/llama.cpp ecosystem. The
dequantization routines here are written independently against that
public layout description.

## IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S codebook tables

The low-bit IQ quantization formats are defined not only by a block
layout but by fixed numeric codebook grids (`iq1s_grid`, `iq2xxs_grid`,
`iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `ksigns_iq2xs`,
`kmask_iq2xs`). These tables are part of the format itself -- a file
quantized with them cannot be decoded without these exact values, and
they cannot be independently re-derived.
`crates/ferrox-quant/src/iq_tables.rs` reproduces them from the
MIT-licensed ggml project's `ggml-common.h`:

  Copyright (c) 2023-2026 The ggml authors / Georgi Gerganov and
  contributors (llama.cpp)

The dequantization *code* using those tables is written independently
against ggml's published dequant semantics, and is cross-validated
against the real compiled ggml implementation -- for IQ1_S/IQ2_XXS/
IQ3_XXS through an independent Python reference, and for IQ2_XS/IQ2_S/
IQ3_S/IQ1_M by linking `ggml-quants.c` and asserting bit-exact equality
with its own output (`crates/ferrox-quant/src/iq_tier_goldens.rs`).

## Speculative decoding design

`ferrox-models::speculative`'s prompt-lookup speculative decoding
(propose candidate continuation tokens by finding a repeat of the
current context elsewhere in token history, verify them in a batched
forward pass) follows the same idea popularized by vLLM's "prompt
lookup decoding" feature -- no code is copied; the n-gram matching,
batched verification, and accept/reject accounting in ferrox's
implementation are original.

## MoE expert-offload design

The CPU/GPU expert-placement model in `ferrox-moe` (`PlacementPlan`,
default-CPU-with-GPU-overrides) is a Rust-native design inspired by the
tensor-name-regex offload controls popularized by ik_llama.cpp
(`--cpu-moe`, `-ncmoe`, `--override-tensor`):

  https://github.com/ikawrakow/ik_llama.cpp

ik_llama.cpp is a fork of llama.cpp and inherits its MIT license.

## Metal Concurrent MoE hazard tracking

`ferrox-metal`'s `MemRanges` (and the one-Concurrent-command-buffer MoE
encode path that uses it) follows the same hazard-tracking idea as
llama.cpp / ggml `ggml_mem_ranges` under `MTLDispatchTypeConcurrent`:
overlapping dispatches may share sources when destinations do not alias.
The Rust implementation in `crates/ferrox-metal/src/mem_ranges.rs` is
written independently against that public design; no ggml source lines
are copied.

## CUDA execution path and hardware capability detection

`ferrox-cuda`'s `HardwareProfile`/`SimdCaps` detection (probe once,
report a plain struct, let performance decisions derive from the
detected machine) is an original Rust-native design; no code is copied.

`ferrox-cuda`'s optional `cuda` feature uses `cudarc`
(https://github.com/coreylowman/cudarc), an MIT-licensed Rust CUDA
binding crate, configured for dynamic loading (no CUDA toolkit
required to build). `cudarc` is pinned to
version 0.11.9 in `ferrox-cuda/Cargo.toml`; see that file's comment for
why (newer releases' transitive dependencies require a newer rustc
than this project's development environment has had available).

The CUDA C kernel sources in `ferrox-cuda/src/gpu.rs` are original code
written to mirror `ferrox-quant`'s CPU fused-dequant math; see that
module's doc comments for their verification status.

## Tokenizer pre-tokenization pattern

The GPT2 byte-to-unicode remap table and the pre-tokenization regex
pattern used by `ferrox-models::tokenizer::GgufBpeTokenizer` reproduce
the publicly documented algorithm from OpenAI's GPT-2 `encoder.py`
(the `bytes_to_unicode()` function and the pre-tokenization regex),
reimplemented independently in Rust using the MIT/Apache-2.0-licensed
`regex` crate (https://github.com/rust-lang/regex). One deliberate,
documented deviation: the real pattern's negative-lookahead whitespace
clause is dropped, since the `regex` crate does not support lookaround
assertions by design (it stays linear-time). See the tokenizer
module's doc comments for the exact practical effect of that
difference.

## SentencePiece-BPE tokenizer algorithm and test data

`ferrox-models::tokenizer::GgufSpmTokenizer` implements the
SentencePiece-BPE encoding algorithm (score-prioritized pairwise
symbol merging via a priority queue, plus UTF-8 byte-fallback), which
is publicly documented behavior of the SentencePiece library
(https://github.com/google/sentencepiece) and of llama.cpp's own
`llm_tokenizer_spm` for GGUF files reporting `tokenizer.ggml.model =
"llama"`. Ferrox's implementation is written independently against
that public description; no source code is copied from either project.

The test fixtures `tests/fixtures/llama-spm-vocab.gguf` and its
accompanying `.gguf.inp`/`.gguf.out` reference test cases are
downloaded directly, unmodified, from `ggml-org/llama.cpp`'s own
repository (`models/ggml-vocab-llama-spm.gguf` and its `.inp`/`.out`
companions), used here under the same terms as the rest of that MIT-
licensed project, purely as a correctness test oracle -- verifying
ferrox's independent implementation produces the same output as
llama.cpp's own tokenizer for the same real vocabulary and real test
inputs.

## OpenAI-compatible HTTP surface

The `/v1/chat/completions`, `/v1/models`, and `/health` endpoint shapes
in `ferrox-server` follow the now-industry-standard OpenAI Chat
Completions API convention used by llama.cpp's server, vLLM, SGLang,
mistral.rs, and this project's own sibling,
antonellof/cognitora-inference. No server code is copied from any of
these; the wire format is a public API contract, not source code.

## Response caching design

`ferrox-server::cache::ResponseCache` (a whole-response LRU+TTL cache
for exact-repeat requests) is a from-scratch Rust implementation
inspired by the general design of Shimmy's
`src/cache/response_cache.rs` (keying a cache by prompt + model +
generation parameters, with TTL expiry). No code is copied; the
eviction bookkeeping, key digest, and hit/miss accounting in ferrox's
version are original.

## Architecture vocabulary

RMSNorm, RoPE, grouped-query attention (GQA), and SwiGLU-gated MoE
feed-forward blocks are standard, widely published transformer
building blocks (see the LLaMA, Mixtral, and DeepSeek-V3 technical
reports). Ferrox's implementations in `ferrox-core` and `ferrox-moe`
are original Rust code written against those public descriptions.

## Sliding-window attention and Qwen2-MoE shared-expert gating

`ferrox_core::attention::causal_gqa_attention_windowed` (sliding-window
GQA attention) and `MoeWeights::shared_expert_gate`'s sigmoid-gated
shared-expert combine (`ferrox-models::decoder`) were identified as real
capability gaps by reading `huggingface/candle`'s real, Apache-2.0-licensed
source (`candle-transformers/src/models/mixtral.rs` and `qwen2_moe.rs`),
which mask attention scores where `key_pos + sliding_window < query_pos`
and scale the shared-expert output by `sigmoid(shared_expert_gate(x))`
respectively. The exact formulas were independently re-confirmed against
the real `transformers` library source
(`transformers/models/qwen2_moe/modeling_qwen2_moe.py`) and llama.cpp's
real `src/models/qwen2moe.cpp` (confirming the exact real GGUF tensor
name, `blk.N.ffn_gate_inp_shexp.weight`) before being written into
ferrox. Ferrox's implementation is written independently (candle uses a
different tensor/backend abstraction throughout), not copied
line-for-line from candle's source, but the two algorithms and their
real GGUF tensor mapping were confirmed via candle as the reference,
under the same Apache-2.0 license as ferrox itself.

## Ferrox Studio frontend (`ui/`)

The web UI is a separate React application under [`ui/`](../ui). It is
distributed as built JavaScript, so every dependency's licence travels
with the bundle rather than staying a lockfile detail. **All of them are
permissive: MIT, Apache-2.0 or ISC.** No AGPL or GPL dependency is
permitted, and `npm run licenses` (`ui/scripts/check-licenses.mjs`)
walks the whole installed tree on every CI run and fails the build on
anything that is not.

Runtime dependencies, which is to say the code that ends up in the
shipped bundle:

| Package | Licence |
|---|---|
| `react`, `react-dom` | MIT |
| `react-router` | MIT |
| `@assistant-ui/react`, `@assistant-ui/react-markdown`, `assistant-stream` | MIT |
| `remark-gfm` (and the `react-markdown` / unified tree under it) | MIT |
| `@radix-ui/react-*` (dialog, label, popover, separator, slider, slot, tabs, tooltip) | MIT |
| `@tanstack/react-table` | MIT |
| `tailwindcss`, `@tailwindcss/vite`, `tw-animate-css` | MIT |
| `clsx`, `tailwind-merge` | MIT |
| `class-variance-authority` | Apache-2.0 |
| `lucide-react` | ISC |

Build-time only (Vite, TypeScript, ESLint and their trees) is MIT and
Apache-2.0, with two exceptions that contribute no code to the bundle
and are allow-listed by name in the licence checker: `lightningcss`
(MPL-2.0), Tailwind's CSS transformer, and `caniuse-lite` (CC-BY-4.0),
a browser-support data table read during the build and discarded.

No component source, stylesheet, class-name string or design-token value
was copied from any of the products studied while designing this UI. In
particular **Unsloth Studio is AGPL-3.0** and ferrox is Apache-2.0: it
was read for information architecture and for which libraries a shipped
product of this kind chooses, and nothing else. Camelid (MIT) was read
the same way, for feature and layout ideas only.

## FreeToken — edge-native MoE serving policy (`ferrox-edge`)

`crates/ferrox-edge` is a Rust port of the host-side decision logic in
[FreeToken](https://github.com/FlashML-org/FreeToken), the edge-native
MoE serving engine described in *FreeToken: Efficient Edge-Native MoE
Serving with Bandwidth-Adaptive Execution*
([arXiv:2608.16157](https://arxiv.org/abs/2608.16157), Yang, Fan, Pan,
Xi, Wang, Sun, Keutzer, Han, Zaharia, Xu and Stoica, 2026). This is a
real port, not independent design work: the algorithms, the constants,
the tie-breaking rules and the module boundaries follow FreeToken's
Python source directly, and each Rust module names the file it came
from. FreeToken is Apache-2.0 licensed, the same license as ferrox:

  Copyright (c) 2026 FlashML and the FreeToken contributors

What was ported, and from where:

| ferrox-edge module | FreeToken source |
|---|---|
| `radix::{tree,plain}` | `python/freetoken/kvcache/radix_cache.py` |
| `radix::swa` | `python/freetoken/kvcache/swa_radix_cache.py` |
| `radix::hybrid` | `python/freetoken/kvcache/hybrid_radix_cache.py` |
| `qstar` | `python/freetoken/moe/bench_profile.py`, the split in `moe/offload_kernels.py` |
| `expert_cache` | `python/freetoken/moe/offload_cache.py`, `moe/offload_kernels.py` |
| `pool` | `python/freetoken/engine/cache_budget.py`, `kvcache/hybrid_swa_pool.py`, `kvcache/base.py` |
| `placement` | `python/freetoken/engine/engine.py` (CPU-layer selection) |
| `scheduler` | `python/freetoken/scheduler/{prefill,decode,table,status}.py` |
| `parser::reasoning` | `python/freetoken/server/reasoning_parser.py` |
| `parser::tool_call` | `python/freetoken/server/function_call_parser.py` |
| `effort` | `python/freetoken/tokenizer/effort.py`, part of `tokenizer/tokenize.py` |
| `detokenize` | `python/freetoken/tokenizer/detokenize.py` |
| `cache_report` | `python/freetoken/cache_report.py` |

FreeToken itself credits SGLang, vLLM, FlashInfer,
flash-linear-attention, LightLLM and llama.cpp; in particular its radix
prefix cache follows SGLang's `RadixCache` / `SWARadixCache` design and
its incremental detokenization borrows the printable-text heuristic from
SGLang and `transformers`' `TextStreamer`. Those lineages carry through
this port.

What was **not** ported, and why: everything in FreeToken that computes
rather than decides. The Triton and CUDA kernels, the C++ CPU MoE
executor, the FTW weight loader's O_DIRECT/mmap machinery, the pinned
host-bank allocator and the CUDA-graph capture ordering are all
torch/CUDA-bound, and ferrox has its own equivalents or its own plans
for them. `ferrox-edge` is deliberately tensor-free: every module takes
measured numbers and returns a decision.

Two *decision*-side pieces are also absent, and are absent because they
are single-family grammars for checkpoints ferrox does not run, not
because they were hard:

- **MiniMax-M3's tool-call grammar.** Its reasoning markers and its
  adaptive leading-closer are ported; its namespaced, recursively
  nested `]<]minimax[>[<k>…` element grammar for *arguments* is not, so
  `ToolCallFormat` has no M3 arm. Its reasoning parser is unaffected.
- **muse-glimmer's ATEM channel format**, on both the reasoning and the
  tool-call side. It is one vendor's channel protocol with its own
  header-span and synthetic-terminator rules, and nothing in
  `docs/MODELS.md` runs it.

Adding either is mechanical: both slot into the same `ReasoningFormat` /
`ToolCallFormat` tables as the nine families that are here.

Where ferrox already had a mechanism the port would have duplicated, the
port plugs into it rather than shadowing it — `ferrox-core::kv_block`
(content-addressed KV blocks), `ferrox-core::expert_store` (the SSD
expert tier), `ferrox-server::batch_scheduler` (continuous batching),
`ferrox-server::stop` (which now delegates its withhold rule to
`ferrox-edge` so there is one implementation of it in the workspace).

## Design inspiration, not code reuse

The following projects informed ferrox's architecture and are credited
here for that influence. Aside from the section immediately above, none
of their source code appears in this repository:

- **llama.cpp** (ggml-org) -- GGUF, mmap-based weight loading, CPU/GPU
  hybrid execution model.
- **ik_llama.cpp** (ikawrakow) -- MoE-aware CPU/GPU tensor placement,
  fused MoE operators, SOTA quantization types.
- **Candle** and **mistral.rs** (Hugging Face / EricLBuehler) -- pure-Rust
  tensor/model execution patterns, GGUF-in-Rust precedent (see also the
  section above for the one case where candle's real source was read
  directly to close a confirmed capability gap).
- **antonellof/cognitora-inference** -- the author's own orchestration
  layer above vLLM/SGLang/llama.cpp; ferrox is designed to be pluggable
  into cognitora's `cgn-agent` as an additional engine backend.

If ferrox ever vendors or adapts actual source lines from an MIT or
Apache-2.0 licensed project, the applicable upstream copyright and
license notice will be added to this file alongside that code, per the
terms of those licenses.
