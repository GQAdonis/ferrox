# Roadmap

Goal: match or beat [llama.cpp](https://github.com/ggerganov/llama.cpp)
tok/s on the same host, backend, and GGUF. Current speed numbers:
[`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

What ships today: [`FEATURES.md`](FEATURES.md) · [`MODELS.md`](MODELS.md).

Priority 1 is prefill (`pp512`), which was the largest deficit in the
project. The work below is ordered by
`(fraction of prefill time removed) × (rows affected) / effort`, and
every item names the llama.cpp function it ports, so the plan can be
argued with rather than taken on faith.

## Landed

Ported from `.scratch/llama.cpp`, measured on Host B (Apple M2 Pro):

- **`kernel_mul_mm` for every quant kind.** The simdgroup GEMM existed
  only for Q4_K and Q6_K; Q8_0, Q4_0, Q5_K and IQ4_XS decomposed a
  batched matmul into N matvecs and re-read the whole weight matrix once
  per token. The shared body now takes its block geometry from the
  dequant functor (`NL`, `BLOCK_BYTES`) the way llama's `nl` template
  argument does — 16 for 256-element super-blocks, 2 for the 32-element
  legacy formats.
- **FA-vec causal prefill at head_dim 64 and 96.** Only 128 and 256
  existed, so every d=64 model fell back to `gqa_prefill`, whose
  per-thread threadgroup accumulator (`tg * head_dim` floats, ~27 KB)
  caps occupancy at roughly one threadgroup per core. That kernel was
  62% of SmolLM2's Metal prefill.
- **`IQ4_XS` reaching Metal at all.** `metal_kind_supported` and
  `apply_gpu`'s kind table disagreed, so batched IQ4_XS silently ran on
  the CPU — Q/K/V and O projections included.
- **`lm_head` only for positions whose logits are used.** `forward_batch`
  projected all 512 prompt positions through the vocabulary and every
  prefill caller then dropped all but the last. llama does not do the
  work at all (`llama_batch_get_one` leaves `logits` unset, `inp_out_ids`
  selects one row). That projection is `V·H / (V·H + L·P_layer)` of
  prefill matmul work: 30% on Gemma-3-1B, 23% on Gemma-2-2B, 21% on
  Llama-3.2-1B.
- **Batched dense FFN in one command buffer.** gate and up each
  round-tripped a `[batch × ffn_dim]` result to the host (29 MB apiece at
  batch 512 on an 8B), the activation ran on the CPU, and the result went
  back up for the down projection.
- **llama's vectorized B-tile store** in `mul_mm` (one `half2x4` op
  instead of 8 scalar loads + 8 scalar stores per thread per K step),
  plus its `float4` bulk copy on the partial-tile output path.

Two bugs fell out of this work rather than being looked for:

- The resident weight-buffer cache was keyed by `(pointer, length)`,
  which is not an identity — free a tensor and the allocator can hand the
  same address and length to a different one, after which the cache
  serves stale weights. Entries now carry a sampled content fingerprint.
- `ferrox-cli`'s `cuda` feature never forwarded to `ferrox-models`, so
  `cargo build -p ferrox-cli --features cuda` did not compile. CI has no
  GPU and never built that combination.

## Planned

### 1. Keep prefill activations resident on the GPU

The remaining small-model Metal gap is fixed cost, not kernel time. Every
matmul still allocates its input buffer with `newBufferWithBytes` (a full
copy in), commits its own command buffer, waits, and copies the result
back out; between them, RMSNorm, residual add, RoPE and softcap run as
host loops in `decoder.rs`. That is several commit/wait pairs and a dozen
host↔device copies of a `[batch × hidden]` block per layer.

llama executes SmolLM2 `pp512` in ~69 ms. At that scale the entire budget
is a few hundred kernel launches with no host involvement.

In cost order: (a) launcher variants taking an existing `MTLBuffer` for
input and output, so consecutive projections chain without readback — the
resident-buffer machinery already exists for KV and for weights; (b)
Metal kernels for RMSNorm, residual add and softcap; (c) one command
buffer per layer instead of one per operation.

Measure with `FERROX_METAL_MM_TIMING=1` and watch `setup + readback` fall
toward zero. Affects all 15 Metal `pp512` rows; the six sub-1.5B rows
decide the outcome.

### 2. Parallelize CPU prefill attention, RoPE and RMSNorm

`ferrox-core/src/attention.rs` has zero rayon call sites —
`causal_gqa_attention_softcap` is a plain `for h in 0..n_heads`, driven
from a serial `for b in 0..batch_size` in `forward_hidden_batch`. So the
one term that grows as O(batch²) runs single-threaded while the matmuls
around it use every core. The RoPE loop, the `cache.push` loop and all
three `rms_norm` passes are serial too.

Every `(b, h)` pair is independent once the push loop has populated
`cache.k`/`cache.v`, so this is a rayon region over pairs writing
disjoint slices. Quant-kind-independent, so it caps whatever item 3 can
recover. Affects all 11 CPU `pp512` rows.

### 3. CPU batch GEMM for Q4_K, Q5_K, Q6_K, Q4_0

Exactly one kind has a real CPU batch GEMM: Q8_0 on aarch64+dotprod
(`gemm_q8_0x4_group`). Q4_K runs a GEMV per position; Q5_K, Q6_K and the
fallback run a scalar row-dot per (row, position). llama ships 15
`ggml_gemm_*` kernels on ARM covering Q4_0, Q4_K, Q5_K, Q6_K, Q2_K,
Q8_0, IQ4_NL and MXFP4.

Add `gemm_q4_kx8_group` mirroring `gemm_q8_0x4_group` — hoist the
per-block weight unpack out of the activation loop and accumulate 4 Q8_K
activations against the 8-row group, llama's `q8_k_blocklen = 4 ×
ncols_interleaved = 8` shape. Q5_K and Q6_K need a repack first.

### 4. Remove the `apply_batch` transpose; dedupe activation quantization

Six CPU kind branches end with the same serial transpose, and the store
stride is `rows * 4` bytes, so every store misses a cache line and a TLB
entry — roughly 705M strided stores per Mistral-7B prefill. Activation
quantization is serial and repeated three times over an identical
`normed_batch` for q/k/v. llama parallelizes that step and never
transposes, because `forward_mul_mat_one_chunk` writes straight into
`dst` at the right stride.

### 5. MoE: `mul_mat_id` on both backends

`try_cpu_moe_mul_mat_id` is named after llama's kernel but takes one
position at a time, so expert weights stream once per token instead of
once per expert. llama's `ggml_mul_mat_id` builds a token→expert map
(`mul_mm_id_map0` on Metal), runs one batched GEMM per expert over the
tokens routed to it, and scatters the results back. Porting that map is
what makes MoE prefill batch at all.

### 6. CUDA

Now buildable and measured on a rented RTX 4090 — see the CUDA table in
[`RESULTS.md`](../benchmarks/RESULTS.md). Everything above is Metal/CPU
work; the CUDA kernels have had no tuning pass at all.

## Correctness (blocking — these rows are timings of the wrong computation)

Neither benchmark harness inspects generated text, so a model can report
healthy tok/s while producing garbage. **Adding an output-sanity check to
the suite is a prerequisite for calling any of these rows closed.**

- **SmolLM2-135M Q8_0 on Metal produces garbage.** CPU gives "the city of
  Paris, a vibrant metropolis…"; Metal gives word salad. Predates all the
  prefill work above (reproduces at `1e6eded`). Still wrong with
  `FERROX_METAL_ATTN=0`, `FERROX_METAL_FA_VEC=0`, `FERROX_METAL_MUL_MM=0`
  and `FERROX_METAL_DENSE=0`, so it is not the dense matmul or the
  attention kernel. TinyLlama Q8_0 on Metal is correct, so it is specific
  to this model's shape (d=64, 9 heads / 3 KV heads, tied embeddings).
- **Gemma-2-2B Metal decode** diverges after a few tokens;
  `FERROX_METAL_ATTN=0` is correct, so it is in the Metal attention path.
- **Gemma-3-1B Metal** is wrong from the first token and stays wrong with
  `FERROX_METAL_ATTN=0` — a separate bug in the dense path.
- **Qwen1.5-MoE runs with its shared expert missing.** The GGUF has
  `blk.N.ffn_{gate,up,down}_shexp.weight` and `shared_expert_gate` is
  loaded, but `n_shared_experts` is derived only from
  `{arch}.expert_shared_count`, a key this file does not have. Roughly
  40% of the active FFN is silently skipped. Fixing it **adds** work, so
  expect that row to get worse before items 3 and 5 pull it back. Do not
  report a gap for this model until the shared expert runs.

## Where ≤ 1.0× is not honestly reachable

Stated up front rather than discovered later.

- **Sub-1B Metal rows land near 1.5–2×, not 1.0×, without a graph
  executor.** llama does SmolLM2 `pp512` in 69 ms. Even after item 1
  removes readbacks, ferrox's per-layer Rust driver re-plans and
  re-encodes every layer from host code. The last 1.5× needs a compiled
  graph with pre-encoded command buffers — a larger architectural change
  than anything on this list.
- **Every CPU row is capped around 1.5–2× without i8mm.** llama selects
  `ggml_gemm_q8_0_4x8_q8_0` and `ggml_gemm_q4_K_8x8_q8_K` on M2+, both
  built on `vmmlaq_s32`. ferrox's Q8_0 GEMM is the older dotprod 4×4
  tier. Item 3 closes the GEMV-vs-GEMM gap, not the dotprod-vs-i8mm gap.
- **x86 has no batch GEMM for any kind.** `gemm_q8_0x4_group`'s
  non-aarch64 path is the GEMV once per activation. No x86 host is
  pinned, so nothing regresses — but no CPU claim here transfers to x86.

## Deliberately not proposed

A persistent spin-barrier thread pool to replace rayon fork-join. On the
*batched* path the fork-join count is per-matmul, not per-position:
roughly 224 real fork-joins for a 512-token 32-layer prefill, ~0.44 per
token. The argument holds for decode; it does not carry to prefill, and
effort spent there will not move a `pp512` row.

## Measurement rules

Learned the hard way; see [`.scratch/NOTES_LLAMA_CPU.md`](../.scratch/NOTES_LLAMA_CPU.md).

- **Never force a thread count on either engine.** llama.cpp defaults to
  performance cores and loses 2–4× above them; pinning both to one count
  flatters ferrox rather than making the comparison fair.
- **Sequential runs spread ±20% on Host B.** Anything claiming less than
  that must be measured by interleaved A/B. Two batches of runs is not a
  measurement.
- **Never benchmark a loaded box.** Check `uptime` first; a load average
  above ~2 reads known-good rows 25–45% low.
- **`FERROX_METAL_MM_TIMING=1`** splits per-matmul setup / GPU / readback.
- **The serving suite's prompt is ~30 tokens and cannot see prefill.**
  Read prefill off the engine table only.
- **Discard a receipt whose `llama_tps` is implausible** rather than
  rendering it.

## Direction

Where the project should go beyond closing the measured gaps.

1. **Run bigger models on the same hardware.** Make Qwen3 35B-A3B Q5
   usable on a box that today only sensibly runs Q4, or an 8B. Most items
   below serve this.
2. **RAM / VRAM optimization.** Residency planning exists
   (`ferrox inspect-plan`); what is missing is acting on it hard enough to
   change which models fit — tighter KV (`turbo3`, quantized CTK),
   streaming expert residency, not materializing activations we do not
   need.
3. **Hybrid CPU/GPU, especially MoE.** Routed experts are the natural
   split: hot experts resident on GPU, cold ones streamed or run on CPU.
   `PlacementPlan` and `ExpertStore` are the groundwork.
4. **CUDA performance.** The kernels now build and run on real hardware
   but have had no tuning pass.
5. **Tool calling** and **full OpenAI API compatibility** — see
   [`API.md`](API.md). Grammar/JSON-schema constrained decoding and MCP
   invocation are the gaps.
6. **Docker images**, so none of the above requires a Rust toolchain to
   evaluate.

**Models**

- Gemma-4 tokenizer (`gemma4`) + fair-chat pin (engine loads today)
- HybridEngine + Qwen3.5
- Llama 4 / MiniMax engines
- Vision (projector + generate)
- Real GLM-5.2 / DeepSeek V4 / full Kimi e2e
- MTP draft heads
- Qwen2-MoE / Mixtral pins when GGUF fits Host B

**Serving**

- Tool calling: OpenAI `tools` / `tool_choice` request + response shape
- Full grammar / JSON schema constrained decoding
- MCP tool invocation; Anthropic streaming + tools
- Full OpenAI API surface (see [`API.md`](API.md))
- Docker images (CPU / Metal / CUDA variants)
- Continuous-batching multi-request throughput pin
- Full KV layer offload; multi-GPU / tensor parallel / PD disaggregation
- Serving pins: re-measure now that the harness no longer forces `-t`

**KV cache / memory**

- `turbo3` dtype; Metal WHT on the CTK path
- Act on `inspect-plan`'s residency plan: stream cold experts, bound the
  KV budget, report what a host can actually fit
- Hybrid CPU/GPU expert placement for MoE — the main lever for running a
  larger model or a higher quant on unchanged hardware

**Engineering practice worth taking from llama.cpp**

- A `test-backend-ops` equivalent: every kernel checked against a CPU
  reference across shapes and quant kinds, so a backend can never be
  merged that merely runs fast.
- Build the CUDA feature combination in CI, even without a GPU to run it
  on. The break above was a compile error, not a runtime one.
