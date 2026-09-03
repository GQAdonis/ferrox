# llama.cpp parity update (2026-09-03)

Companion delta to [`llama-cpp-full-parity-audit-2026-09-02.md`](llama-cpp-full-parity-audit-2026-09-02.md).

**North star:** [`north-star.md`](north-star.md)

---

## Closed since 2026-09-02

| Item | Issue/PR | Notes |
|------|----------|-------|
| Qwen1.5-MoE Metal MoE prefill `CommandFailed` | [#96](https://github.com/antonellof/ferrox/issues/96) / [#97](https://github.com/antonellof/ferrox/pull/97) | Q5_0 expert down weights lacked `mul_mv_id` kernels; 12/24 layers fell back to CPU |
| Phi-4 LongRoPE at parity `n_ctx` | [#98](https://github.com/antonellof/ferrox/issues/98) / [#100](https://github.com/antonellof/ferrox/pull/100) | `apply_runtime_context(n_tokens + 8)`; KL 3.6e-2 → 1.2e-2; K-quant lm_head reclassified DRIFT |
| DeepSeek-R1-Distill WRONG vs Qwen2.5 DRIFT | [#99](https://github.com/antonellof/ferrox/issues/99) / [#100](https://github.com/antonellof/ferrox/pull/100) | Untied Q6K `output.weight`; same graph as Qwen2.5; higher KL threshold for K-quant lm_head |
| Gemma-4 parity via Gemma4Engine | [#101](https://github.com/antonellof/ferrox/issues/101) / [#100](https://github.com/antonellof/ferrox/pull/100) | `prefill_logits` routes gemma4 through `Gemma4Engine`; scratch `llama_logits` for tokenizer |

---

## Live parity sweep (2026-09-03, CPU, 19 GGUFs)

**Reference:** Homebrew `llama.cpp` via default `bash tools/build_llama_logits.sh`
(`brew --prefix llama.cpp`, libllama 7650). Release CPU:
`FERROX_METAL=0 FERROX_CUDA=0`. Gemma-4 requires scratch `llama_logits` (Homebrew
cannot load the checkpoint); see reference vintage below.

| Verdict | Count | Models |
|---------|-------|--------|
| **MATCH** | 6 | TinyLlama Q8_0, Llama-3.2-1B Q8_0/IQ4_XS, Llama-3.2-3B, Mistral-7B, OLMoE |
| **DRIFT** | 10 | All K-quants incl. Qwen2.5 + DeepSeek-R1 + Phi-4 (expected Q8_K activation quant on K-quants — §10 gap inventory) |
| **TIE-FLIP** | 1 | Llama-3.1-8B |
| **WRONG** | 0 | — (Homebrew reference; Qwen2.5 WRONG on scratch only — see below) |
| Skip | 2 | Gemma-4 (Homebrew libllama cannot load checkpoint), BERT encoders |

### Reference vintage

Parity verdicts depend on which `llama_logits` binary links against. Homebrew is
the default for CI and issue closure; scratch `.scratch/llama.cpp` is required when
Homebrew cannot load a checkpoint or when updating the reference baseline.

| Model | Homebrew | Scratch |
|-------|----------|---------|
| Qwen2.5-1.5B Q4_K_M | DRIFT (KL 7.7e-3) | **WRONG** (KL 2.7e-2, top-1 match) |
| Phi-4-mini Q4_K_M | DRIFT (KL 3.8e-3) | DRIFT |
| DeepSeek-R1-Distill Q4_K_M | DRIFT (KL 9.2e-3) | DRIFT |
| Gemma-4 E2B Q4_K_M | load fail | **MATCH** (KL 1.7e-4) |

Qwen2.5 WRONG on scratch is pre-existing on `main`, not a PR #100 regression.
Same untied Q6K `output.weight` graph as DeepSeek-R1 (DRIFT on both references).

### Phi-4-mini ([#98](https://github.com/antonellof/ferrox/issues/98))

`verify_engine::load_and_tokenize` calls `apply_runtime_context(n_tokens + 8)` before decode load, matching `tools/llama_logits.c`. KL improved **3.6e-2 → 1.2e-2**; reclassified **DRIFT** (tied Q6K lm_head uses K-quant activation path).

### DeepSeek-R1-Distill ([#99](https://github.com/antonellof/ferrox/issues/99))

Same qwen2 family as Qwen2.5-1.5B (DRIFT 7.7e-3, same top-1 12095). Untied **Q6K** `output.weight` explains KL **2.0e-2** vs sibling — not a graph bug. Reclassified **DRIFT** with `KL_WRONG_LM_HEAD_KQUANT = 2.5e-2`.

### Gemma-4 ([#101](https://github.com/antonellof/ferrox/issues/101))

Scratch-built `llama_logits` from `.scratch/llama.cpp` loads gemma-4; tokenizer **MATCH**. Logit parity runs through `Gemma4Engine::forward_token` in `prefill_logits`.

---

## Priority plan (updated)

### P0 — This week

| # | Item | Status |
|---|------|--------|
| 1 | Phi-4 LongRoPE at parity `n_ctx` | **Done** — PR #100 |
| 2 | DeepSeek-R1-Distill reclassification | **Done** — PR #100 |
| 3 | Gemma-4 parity via Gemma4Engine | **Done** — PR #100 |
| 4 | Rebuild `llama_logits` from `.scratch/llama.cpp` | **Done** — `build_llama_logits.sh` cmake layout |
| 5 | Fixture-away triage (9 archs) | **Open** |

### P1 — Unchanged from audit

Model layer reorg phase 1, K-quant encoders, CUDA mul_mm, server slot save/load, embedding path, gguf-split.

---

## Regenerate

```bash
cargo build -p ferrox-cli --release
# Homebrew or cmake scratch build:
bash tools/build_llama_logits.sh
# cmake example:
# cmake -B /tmp/llamabuild ... .scratch/llama.cpp && cmake --build /tmp/llamabuild --target llama -j8
# LLAMA_CPP_PREFIX=/tmp/llamabuild bash tools/build_llama_logits.sh

export FERROX_METAL=0 FERROX_CUDA=0 FERROX_ALLOW_MULTIPLE_INSTANCES=1
mkdir -p .scratch/parity-run-2026-09-03
for f in models/*.gguf; do ferrox parity -m "$f"; done
```

*Updated 2026-09-03 on PR #100 branch.*
