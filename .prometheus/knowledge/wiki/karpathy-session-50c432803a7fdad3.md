---
type: SessionRecord
id: karpathy-session-50c432803a7fdad3
title: Karpathy session 50c432803a7f
tags:
- karpathy
- session-learning
sources:
- session:473cad21-bddf-4449-8516-e75e1b8704c0
timestamp: 2026-09-01T14:37:52.184339+00:00
created_at: 2026-09-01T14:37:52.184339+00:00
updated_at: 2026-09-01T14:37:52.184339+00:00
revision: 0
---

## Delta

## Recommendation: Ferrox, with a named fallback trigger

I ran this against primary sources — the Ferrox repo on disk, its generated benchmark receipts, its source tree, and KnowMe's actual `Cargo.toml`. Two of my earlier statements were wrong and I've corrected them below.

### First, the correction that matters

**I misread the benchmark.** I told you "~0.70× llama.cpp (30% slower)." The table states `Gap = llama / ferrox`, **<1 means Ferrox is faster**. So 0.70× was Ferrox *winning*. On Metal, at the sizes you'd run:

| Model | Ferrox | llama.cpp | |
|---|---|---|---|
| Llama-3.1-8B Q4_K_M `tg128` | 30.34 tok/s | 30.05 | parity |
| Mistral-7B Q4_K_M `tg128` | 32.29 | 32.11 | parity |
| Dense `pp512`, all rows | — | — | 0.98–1.10× |

Performance is **not** a reason to prefer llama.cpp on Apple Silicon. CPU-only is a different story (1.41×–5.06× slower) — that's the disqualifier if you ever ship a Windows/Linux CPU target.

### Why Ferrox wins here

1. **Integration cost is the real differentiator, and you have hard evidence.** KnowMe had to *vendor and patch* `llama-cpp-2` because `llama-cpp-sys-2` force-enables `common`, dragging in llama.cpp's HTTP tooling and breaking iOS with unresolved `httplib::Client`. That's a maintained fork you own forever. Ferrox is a pure-Rust cargo dependency with no C++ toolchain, no ggml, no sys-crate feature-unification trap.
2. **Tool calling is unusually well-built for your exact need.** The "eleven wire formats" claim is backed by code (Hermes, Granite, Nemotron, functionary, deepseek_r1 all present in source). More importantly, **today's commit added lazy grammars** — a port of llama.cpp's own trigger mechanism, whose doc comment states the problem precisely: an eager grammar forbids the model from saying "let me look that up" before emitting a tool call. That is exactly your code-map-query flow.
3. **There's an `mcp.rs` in the server** — relevant since your code map is reached via MCP/CLI.
4. **`ferrox download`** exists (HF Hub syntax), so the "download a model on first run" requirement is a built-in, not something you write.
5. **The docs are honest to the point of self-harm** — they publish their own red rows, mark things "compiles, never measured," and refuse to load models they only partly implement rather than returning fluent garbage. That's the strongest credibility signal I found.

### The uncomfortable part

**Ferrox is v0.14.0 with a single-digit-days-old tool-calling feature, and the largest model in its benchmark suite is 8B.** Qwen2.5-Coder is `qwen2` architecture — supported and suite-tested, but only at **0.5B**. Nothing at 7B/14B/32B coder scale has a published receipt. llama.cpp has years of production hardening and every model on day one.

**My honest read:** the architecture support is real (dense `qwen2` is the best-tested path), so the risk is *bugs at scale*, not *won't run*. Given that you own the Ferrox codebase's reference copy and it's under active daily development, that risk is manageable — but it should be a **tested assumption, not a bet**: run Qwen2.5-Coder-7B through Ferrox before committing the integration. If it fails, `llama-cpp-2` (vendored, as KnowMe proved) is the fallback, and the abstraction seam makes that swap cheap.

### Model: **Qwen2.5-Coder-7B-Instruct Q5_K_M**

- **Why 7B not 14B/32B:** you need headroom for the compass map in context *and* the app itself. 7B at Q5_K_M lands ~5.5 GB — comfortable on 32 GB, leaves room for a big context window. 14B Q4_K_M (~9 GB) is the upgrade if quality disappoints.
- **Why Q5_K_M not Q4_K_M:** tool calling is format-sensitive; Q4 degrades structured output noticeably more than prose. The extra ~1.3 GB is cheap insurance for JSON that must parse.
- **Why Qwen2.5-Coder over alternatives:** it's `qwen2` arch (Ferrox's best-tested dense family), has native Hermes-style tool-call templates, and is strong at code comprehension — your actual task.
- **GGUF:** `bartowski/Qwen2.5-Coder-7B-Instruct-GGUF` is the co

## Root Cause

No explicit root-cause section was captured; preserve this as a session record, not an inferred diagnosis.

## Corrective Actions

Review and promote only reusable findings.

## Session Metadata

- Harness: claude-code
- Session: 473cad21-bddf-4449-8516-e75e1b8704c0
- Captured: 2026-09-01T14:37:51.098544Z
- Project: /Users/gqadonis/Projects/references/ferrox

## Changed Paths

- No changed paths detected.
