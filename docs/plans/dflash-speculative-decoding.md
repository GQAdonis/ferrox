---
name: dFlash / dFlash2, block-diffusion speculative decoding
overview: "GOAL: give ferrox a real speculative drafter. Today `ferrox-models::speculative` is prompt-lookup only (an n-gram match over the history, no model), and `--mtp` errors. dFlash is a block-diffusion drafter: it predicts a whole K-token block in ONE forward pass instead of autoregressively, conditioned on the target model's own hidden states, sharing the target's embedding and LM head, and the target verifies the block in one batched pass. Published numbers: mean acceptance length 4.80 tok and 2.7-3.4x throughput on the drafter's own benchmark table, 3.3x vs EAGLE-3's 2.1x on GSM8K at equal acceptance length. THE HONEST CONSTRAINT, stated up front: dFlash is a TRAINED drafter and ferrox does not train. Everything here is worthless without a published drafter checkpoint in a format ferrox can load, so `dflash-checkpoint-reality` is the first todo and it is a GO/NO-GO, if no loadable checkpoint exists, the useful subset of this plan is the engine-side work (lossless verification, the drafter trait, cache resume, acceptance metrics), which is worth doing on its own and is written to stand alone."
todos:
  - id: dflash-checkpoint-reality
    content: "GO/NO-GO, do this before writing any kernel. Establish whether a dFlash drafter can be loaded by ferrox AT ALL. Answer three questions with evidence, not inference: (1) does a published checkpoint exist and under what licence (search names the HF repo `incoai/Qwen3.8-27B-DFlash2`, and claims 3.5M downloads and integrations in SGLang / vLLM / TensorRT-LLM / llama.cpp / Ollama / oMLX); (2) since llama.cpp is claimed as an integration, is there a GGUF convert path and what tensor names + metadata keys does it emit, read llama.cpp's converter and its dFlash graph, that is this repo's default method and it settles the format question by reading rather than guessing; (3) does the checkpoint ship the drafter ALONE (sharing the target's embedding + LM head) or a standalone model. If the answer to (1) is no, STOP and record that; the engine-side todos below still stand"
    status: pending
  - id: spec-lossless-verify
    content: "BLOCKING CORRECTNESS, and worth landing whatever happens to dFlash. `speculative_decode` (ferrox-models/src/speculative.rs:123) accepts a draft token only when `argmax(batch_logits[i]) == guess`, greedy only. Every published dFlash speedup is stated as LOSSLESS, which at temp > 0 means the speculative-sampling rejection rule (accept with p = min(1, p_target(x)/p_draft(x)), else resample from the normalised residual max(0, p_target - p_draft)), not argmax matching. Without it, ferrox can either speculate at temp=0 or be silently non-lossless at temp>0, and the second is the `coverage-fail-closed` bug class. Needs the DRAFT distribution, not just draft tokens, so it constrains the drafter trait below. Test against the invariant that matters: over many seeds the speculative sampler's output distribution must match plain sampling from the target"
    status: pending
  - id: spec-drafter-trait
    content: "Generalise `PromptLookupSpeculator` (speculative.rs:41) into a `Drafter` trait so the verify loop stops being welded to n-gram lookup. Required shape, derived from what dFlash actually needs: propose(history, target_hidden: &[f32], k) -> DraftBlock { tokens, per-position draft probabilities }. Two things the current signature cannot express and both are load-bearing: the draft PROBABILITIES (needed by spec-lossless-verify) and the target model's last-layer hidden state (dFlash conditions on it). ferrox already exposes the second, `Decoder::forward_hidden_batch` exists and `forward_batch` is a thin wrapper over it plus `output_head` (decoder.rs:3282), so the conditioning tensor is already computed and thrown away. Keep prompt-lookup as an impl of the trait: it is the zero-dependency fallback and it is the only drafter that works with no checkpoint"
    status: pending
  - id: spec-cache-resume
    content: "`speculative_decode`'s doc comment states `kv_caches` must be freshly initialised and that continuing an already-populated cache across calls is not supported (speculative.rs:120-122). That makes it a demo, not a serving path: ferrox-server hands a decode loop a warm cache from the prefix cache. Take a start position, and make the rollback arithmetic (`cache.truncate(committed_len)`, speculative.rs, `KvCache::truncate` at ferrox-core/src/cache.rs:283) correct against a non-zero base. Prerequisite for spec-server-integration"
    status: pending
  - id: spec-acceptance-metrics
    content: "`SpeculativeDecodeResult` already carries `forward_calls` / `tokens_generated` / `tokens_per_call` (speculative.rs:88-113), which is the right instinct and is exactly the published metric (`acceptance length` = completion tokens / verification steps). Surface it: through `usage` on the wire beside the timings that landed in 233328f, in `/admin/stats`, and per-position accept rate so SUFFIX DECAY is observable, dFlash2 exists because per-position accuracy falls from 99.5% at the first drafted position to 87.8% at the last, and a drafter that is fast but decays is indistinguishable from a good one on a single mean"
    status: pending
  - id: dflash-drafter-forward
    content: "The drafter itself: a small transformer (published as ~5 layers, hidden dim below the target's, embedding and LM head SHARED with the target so the parameter count is 1-5% of the target). Load it as a second GGUF whose embedding/output tensors are absent and resolved against the already-loaded target, that sharing is the whole reason the drafter is cheap, so a loader that duplicates them defeats it. Runs on the CPU path first; Metal is dflash-metal below"
    status: pending
  - id: dflash-block-diffusion
    content: "The drafting procedure. Start the K-token block as mask tokens, run a small number of denoising steps (sources say typically 4-8, configurable) that refine all positions JOINTLY, and emit the block plus per-position distributions. This is the part with no precedent in ferrox, there is no diffusion or masked-denoising machinery anywhere in the tree. Block size K and step count are the two knobs (SGLang exposes `--speculative-dflash-block-size`, e.g. 8 or 16). NOTE A CONTRADICTION IN THE SOURCES, do not resolve it by picking the convenient one: the arXiv summary describes TREE verification of multiple continuations, while the LMSYS integration write-up states explicitly that dFlash is LINEAR block diffusion and not tree-based. ferrox's verify loop is linear today. Read the reference implementation before building either"
    status: pending
  - id: dflash-kv-injection
    content: "The trick that makes the drafter cheap on long context, and the one most likely to be got wrong. Rather than have the drafter re-encode the prompt, extract the target model's hidden representations for the context tokens and INJECT them into the drafter's KV cache, so the drafter skips modelling the full context from scratch. The reference integration materialises the injected KV immediately (layer-batched linear projection, fused norm+RoPE post-processing) rather than lazily, specifically so the drafter's KV does not balloon and so prefix sharing still works. ferrox's prefix cache now has block identity and payload-derived signatures (ferrox-core/src/kv_block.rs, kv_signature.rs), injected drafter KV is DERIVED state and must not be confused with target KV under the same block hash. Salt it, or the persistent cache will hand a drafter block to a target model. `kv-cache-signature` in docs/plans/serving-and-tiered-kv.md exists for exactly this class of mistake"
    status: pending
  - id: dflash2-path-selector
    content: "dFlash2 innovation 1, and the cheapest real win in the whole plan. Independent per-position argmax gives a block where each token is plausible alone and the sequence is incoherent. Keep the top 16 candidates per position and score adjacent pairs with a bilinear form: S_t(a,b) = U_t(b) + <A(a) (*) H(h_t), B(b)>, where U_t(b) is the drafter's own logit for candidate b, A and B are 256-dim predecessor/candidate embeddings, H(h_t) is a context gate over which components matter, and (*) is elementwise product. Then Viterbi one coherent path through the block. Published cost/benefit: +2.0M parameters and +0.6% latency for +0.34-0.47 tokens of acceptance length, against DSpark's +77.8M and +9.6% for less. Pure inference-side arithmetic given the weights"
    status: pending
  - id: dflash2-two-tap-conv
    content: "dFlash2 innovation 2, the fix for suffix decay (99.5% accuracy at the first drafted position falling to 87.8% at the last). Two-tap dynamic depthwise convolutions before and after each attention and feed-forward sublayer: Conv_k(x)_t = k_t,0 (*) x_t + k_t,1 (*) x_{t-1}, where each coefficient is a learned base kernel plus a content-dependent correction and every 16 channels share one correction vector. Position 0 reads the last VERIFIED token's representation; later positions read their immediate predecessor; all positions still compute in parallel, which is the property the whole method rests on. Published cost: +16.5M params (3%) and +0.7% latency, letting a 5-layer drafter approach a 15-layer one. Diagnosis behind it, worth keeping because it is testable: within-block attention mass collapses from 30% at layer 1 to 8% at layer 5"
    status: pending
  - id: dflash-metal
    content: "Metal path for the drafter block forward + denoising steps. Do NOT start until the CPU drafter agrees with a reference: this repo's rule is that a GPU path which cannot match CPU is refused rather than admitted (see coverage-phi-metal-rope, where Phi-4-mini is CPU-only on exactly those grounds). The drafter is small and runs K positions per step, so it is a batched-GEMM shape the existing `mul_mm_sg` covers; the denoising loop is the new dispatch pattern. Guard with `ferrox verify --backend metal` before any timing"
    status: pending
  - id: spec-server-integration
    content: "Wire speculation into ferrox-server: `batch_scheduler.rs` decode loop, per-request enable/disable, and interaction with the chunked prefill state machine that just landed. Two hazards specific to serving, neither present in the CLI demo: a rejected block must not leave the request's KV cache longer than its accepted prefix under CONCURRENCY (the rollback is per-request, the pool is shared), and a drafter is a second set of weights competing for the same device budget, `mem-preload-kv-budget` in the serving plan must account for it or admission will overcommit"
    status: pending
  - id: spec-bench-harness
    content: "`ferrox bench` measures pp512/tg128 with no speculation, so none of this is visible in benchmarks/RESULTS.md. Add acceptance length and speculative tok/s as first-class bench outputs, on the SAME quiet-host contract as every other row (the 2.0 load bar is now enforced in ferrox-cli/src/host_state.rs). Speculation speedup is workload-dependent by construction, code and math accept long, open chat accepts short, so a single number is a lie; report per-workload, mirroring how the sources report GSM8K / HumanEval / MT-Bench separately"
    status: pending
isProject: false
---

# dFlash / dFlash2, block-diffusion speculative decoding

> Plan for implementing [dFlash 2](https://inco.ai/blog/dflash2/) in ferrox.
> Sources read for this plan: the dFlash2 post above, the
> [arXiv paper](https://arxiv.org/pdf/2602.06036) (*DFlash: Block Diffusion
> for Flash Speculative Decoding*), and the
> [LMSYS integration write-up](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/).
> Where they disagree, the disagreement is recorded rather than resolved.

## What dFlash actually is

It is **not** a flash-attention variant. The name is misleading if you come
to it from FlashAttention: dFlash is a **speculative decoding drafter**, and
the "d" is *diffusion*.

Ordinary speculative decoding runs a small draft model autoregressively for
K steps, then has the target verify all K at once. The draft phase is
therefore K sequential forward passes of a small model, cheap per step, but
still serial, and that serial chain is what caps the speedup.

dFlash replaces it with **block diffusion**: the drafter emits the whole
K-token block in **one** forward pass, starting from mask tokens and running
a handful of joint denoising steps over all positions at once. It is
conditioned on the target model's own hidden states and shares the target's
embedding and LM head, so it is 1–5% of the target's parameters.

The verification half is unchanged and is the part ferrox already has.

## What ferrox already has, and what it does not

Already here, this is why the plan is tractable at all:

| Piece | Where |
|---|---|
| Batched verify of a draft block in one call | `Decoder::forward_batch`, `decoder.rs:3282` |
| Target hidden states, already computed and discarded | `Decoder::forward_hidden_batch`, same site |
| Accept-longest-prefix + KV rollback | `speculative.rs:123`, `KvCache::truncate` at `cache.rs:283` |
| Acceptance-length counters | `SpeculativeDecodeResult`, `speculative.rs:88` |
| A CLI entry point | `ferrox speculative` |

Not here:

- **Any trained drafter.** `ferrox speculative` is prompt-lookup: an n-gram
  match over the history with no model at all. `--mtp` errors by design and
  says so in `docs/CLI.md:83`.
- **Lossless verification.** The accept test is `argmax(logits[i]) == guess`.
  That is correct at temp 0 and quietly wrong above it.
- **Any diffusion / masked-denoising machinery.** Nothing in the tree
  resembles it.
- **Serving integration.** `speculative_decode` demands a *fresh* KV cache
  (`speculative.rs:120`), which a server never has.

## The constraint that decides whether this plan is real

**ferrox does not train models.** dFlash's whole value is in learned weights:
the 5-layer drafter, the 2.0M-parameter path selector, the 16.5M-parameter
two-tap convolutions. Implementing the forward pass without a checkpoint
produces a drafter that proposes noise, and a drafter that proposes noise is
strictly slower than no drafter, every rejected block costs a target forward
pass that would otherwise have produced a token.

So `dflash-checkpoint-reality` is first and is a **go/no-go**. Search
indicates a Hugging Face repo (`incoai/Qwen3.8-27B-DFlash2`) and claims
integrations in SGLang, vLLM, TensorRT-LLM, llama.cpp, Ollama and oMLX. The
llama.cpp claim is the load-bearing one for ferrox: if llama.cpp converts and
runs these drafters, then a GGUF representation exists, and reading its
converter and graph settles the tensor names, the metadata keys and the
embedding-sharing question by reading rather than by guessing. That is this
repo's default method and it applies here.

**If no loadable checkpoint exists, stop at that todo.** The four engine-side
items, `spec-lossless-verify`, `spec-drafter-trait`, `spec-cache-resume`,
`spec-acceptance-metrics`, are worth landing regardless. They are what makes
*any* drafter usable in serving, they close a real correctness hole at
temp > 0, and they are the prerequisites for MTP heads, EAGLE, or dFlash
whenever a checkpoint does appear.

## Published numbers, and how much to trust them

All of these are the sources' own, measured on the sources' own hardware and
harness. None has been reproduced here. They are recorded so the plan has a
target to fail against, not as claims ferrox can make.

dFlash2 post, mean acceptance length and throughput vs autoregressive:

| Model (as named by the source) | Acceptance length | Throughput |
|---|---|---|
| Qwen3.8-27B | 4.80 (vs MTP 4.28, DSpark 3.62) | 2.7–3.4× |
| Muse Glimmer | 5.70 | 3.1–4.6× |

LMSYS, dFlash vs EAGLE-3 (acceptance length / speedup), note the middle
column: on GSM8K the acceptance lengths are **identical** at 4.2 and the
speedup still goes 2.1× → 3.3×, because the win is in the *draft* phase
being one pass instead of K:

| Task | EAGLE-3 | dFlash |
|---|---|---|
| GSM8K | 4.2 / 2.1× | 4.2 / 3.3× |
| HumanEval | 4.3 / 2.2× | 4.0 / 3.2× |
| MT-Bench | 3.1 / 1.4× | 3.0 / 2.2× |

That table is the single most useful thing in the sources for ferrox, because
it says the mechanism being ported is *drafting latency*, not draft *quality*.
Drafting latency is something a CPU/Metal engine can actually move.

## dFlash2's two additions

Both are inference-side arithmetic given weights, and both come with a stated
diagnosis that ferrox can independently test once `spec-acceptance-metrics`
reports per-position accept rates.

**1. Path selector.** Predicting every position independently gives a block
where each token is individually plausible and the sequence does not cohere.
Keep the top 16 candidates per position, score adjacent pairs

```
S_t(a, b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>
```

(`U_t(b)` the drafter's own logit, `A`/`B` 256-dim predecessor/candidate
embeddings, `H(h_t)` a context gate, `⊙` elementwise), then take the best
path. Cost +2.0M params and +0.6% latency for +0.34–0.47 acceptance tokens,
against DSpark's +77.8M and +9.6%.

**2. Two-tap dynamic convolution.** Fixes *suffix decay*, per-position
accuracy falling from 99.5% at the first drafted position to 87.8% at the
last, because within-block attention mass collapses from 30% at layer 1 to 8%
at layer 5.

```
Conv_k(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t-1}
```

inserted before and after each attention and FFN sublayer. Each coefficient is
a learned base kernel plus a content-dependent correction, with one correction
vector per 16 channels. Position 0 reads the last *verified* token; later
positions read their immediate predecessor; everything still computes in
parallel. +16.5M params (3%), +0.7% latency, and a 5-layer drafter reaches
roughly 15-layer quality.

## Order of work

1. `dflash-checkpoint-reality`, go/no-go, and it gates everything below it.
2. `spec-lossless-verify`, correctness, independent of dFlash, do it anyway.
3. `spec-drafter-trait` + `spec-cache-resume` + `spec-acceptance-metrics`,    the engine-side foundation, all three independent of dFlash.
4. `dflash-drafter-forward` → `dflash-block-diffusion` → `dflash-kv-injection`.
5. `dflash2-path-selector` → `dflash2-two-tap-conv`.
6. `spec-server-integration`, then `dflash-metal`, then `spec-bench-harness`.

## Measurement contract

Same as `docs/plans/llama-cpp-parity-push.md`: quiet host (1-minute load
average below 2.0, now enforced by `ferrox-cli/src/host_state.rs`), both
engines measured in the same session, interleaved A/B, receipts checked in.

One addition specific to speculation: **report per workload.** Acceptance
length is a property of the *text*, not just the model, code and math accept
long blocks, open-ended chat accepts short ones. A single averaged speedup
number for speculative decoding is not a measurement, and the sources
themselves report GSM8K, HumanEval and MT-Bench separately for that reason.
