# llama.cpp parity update (2026-09-03)

Companion delta to [`llama-cpp-full-parity-audit-2026-09-02.md`](llama-cpp-full-parity-audit-2026-09-02.md).

**North star:** [`north-star.md`](north-star.md)

---

## Closed since 2026-09-02

| Item | Issue/PR | Notes |
|------|----------|-------|
| Qwen1.5-MoE Metal MoE prefill `CommandFailed` | [#96](https://github.com/antonellof/ferrox/issues/96) / [#97](https://github.com/antonellof/ferrox/pull/97) | Q5_0 expert down weights lacked `mul_mv_id` kernels; 12/24 layers fell back to CPU |

---

## Live parity sweep (2026-09-03, CPU, 19 GGUFs)

Logs: `.scratch/parity-run-2026-09-03/`

| Verdict | Count | Models |
|---------|-------|--------|
| **MATCH** | 6 | TinyLlama Q8_0, Llama-3.2-1B Q8_0/IQ4_XS, Llama-3.2-3B, Mistral-7B, OLMoE |
| **DRIFT** | 8 | K-quants (expected Q8_K activation quant — §10 gap inventory) |
| **TIE-FLIP** | 1 | Llama-3.1-8B |
| **WRONG** | 2 | DeepSeek-R1-Distill (KL 2.0e-2), Phi-4-mini (KL **1.2e-2** after LongRoPE fix, was 3.6e-2) |
| Skip | 2 | Gemma-4 (stale Homebrew libllama), BERT encoders |

### Phi-4-mini progress ([#98](https://github.com/antonellof/ferrox/issues/98))

`verify_engine::load_and_tokenize` now calls `apply_runtime_context(n_tokens + 8)` before decode load, matching `tools/llama_logits.c`. KL improved **3.6e-2 → 1.2e-2**; still borderline WRONG (threshold 1e-2).

### DeepSeek-R1-Distill ([#99](https://github.com/antonellof/ferrox/issues/99) when filed)

Same qwen2 family as Qwen2.5-1.5B (DRIFT 7.7e-3) but KL **2.0e-2** — investigate hparam / quant mix, not K-quant alone.

---

## Priority plan (updated)

### P0 — This week

| # | Item | Status |
|---|------|--------|
| 1 | Phi-4 LongRoPE at parity `n_ctx` | **In PR** — KL 3.6e-2 → 1.2e-2 |
| 2 | DeepSeek-R1-Distill WRONG investigation | **Open issue** |
| 3 | Rebuild `llama_logits` from `.scratch/llama.cpp` | **Open** — unblocks gemma-4 parity |
| 4 | Fixture-away triage (9 archs) | **Open** |

### P1 — Unchanged from audit

Model layer reorg phase 1, K-quant encoders, CUDA mul_mm, server slot save/load, embedding path, gguf-split.

---

## Regenerate

```bash
cargo build -p ferrox-cli --release
bash tools/build_llama_logits.sh
export FERROX_METAL=0 FERROX_CUDA=0 FERROX_ALLOW_MULTIPLE_INSTANCES=1
mkdir -p .scratch/parity-run-2026-09-03
for f in models/*.gguf; do ferrox parity -m "$f"; done
```

*Generated 2026-09-03 on main.*
