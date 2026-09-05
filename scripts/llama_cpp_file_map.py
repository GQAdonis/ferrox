#!/usr/bin/env python3
"""Map llama.cpp source files to ferrox Rust equivalents.

Reads .scratch/llama.cpp and crates/ to produce a structured inventory
for parity planning. Run from repo root:

    python3 scripts/llama_cpp_file_map.py > .scratch/parity-run-2026-09-02/file_map.json
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
LLAMA = ROOT / ".scratch" / "llama.cpp"
CRATES = ROOT / "crates"

Status = Literal[
    "ported",       # ferrox has equivalent functionality
    "partial",      # some of the file's scope exists
    "missing",      # no ferrox counterpart
    "out_of_scope", # training, RPC, etc.
    "n/a",          # llama.cpp-specific infra (cmake, CI)
]

# Hand-curated mapping: llama.cpp path pattern -> ferrox location + status.
# Patterns are matched with startswith or exact match.
MAPPING: list[tuple[str, str, Status, str]] = [
    # --- Core library ---
    ("src/llama.cpp", "crates/ferrox-models/src/lib.rs + loader.rs", "partial", "Public API surface; ferrox has no libllama-style C API"),
    ("src/llama-model.cpp", "crates/ferrox-models/src/loader.rs", "partial", "Model load + tensor wiring; one loader vs per-arch graphs"),
    ("src/llama-model-loader.cpp", "crates/ferrox-gguf/src/lib.rs", "ported", "GGUF mmap read"),
    ("src/llama-model-saver.cpp", "crates/ferrox-gguf/src/writer.rs", "partial", "GGUF write; no full model export"),
    ("src/llama-arch.cpp", "crates/ferrox-models/src/capability.rs", "partial", "150 catalog rows vs 140 graphs; 16 audited"),
    ("src/llama-hparams.cpp", "crates/ferrox-models/src/config.rs", "partial", "Hyperparameter parsing"),
    ("src/llama-vocab.cpp", "crates/ferrox-models/src/tokenizer.rs", "partial", "Tokenizer; emoji edge cases diverge on BERT"),
    ("src/llama-context.cpp", "crates/ferrox-models/src/decoder.rs", "partial", "Decode context; no llama_context C struct"),
    ("src/llama-batch.cpp", "crates/ferrox-server/src/serving/batch/", "partial", "Continuous batching; incremental stream gap"),
    ("src/llama-graph.cpp", "crates/ferrox-models/src/decoder.rs", "partial", "Hand-written decode graph, not ggml graph"),
    ("src/llama-sampler.cpp", "crates/ferrox-models/src/sampling.rs", "partial", "834 lines vs 4106; missing dry/xtc/typ_p/top_n_sigma"),
    ("src/llama-grammar.cpp", "crates/ferrox-models/src/grammar/", "partial", "GBNF yes; some grammar features missing"),
    ("src/llama-chat.cpp", "crates/ferrox-server/src/completion/", "partial", "Chat templates via tokenizer config"),
    ("src/llama-kv-cache", "crates/ferrox-core/src/cache.rs + ferrox-models/src/kv_budget.rs", "partial", "Standard GQA KV; no DSA/ISWA/MSA variants"),
    ("src/llama-memory", "crates/ferrox-core/src/expert_cache.rs + residency", "partial", "MoE residency policy wired; not executed"),
    ("src/llama-mmap.cpp", "crates/ferrox-gguf/src/lib.rs", "ported", "mmap GGUF"),
    ("src/llama-quant.cpp", "crates/ferrox-quant/src/", "partial", "Read all quants; write Q8_0 only; K-quant encoders in progress"),
    ("src/llama-adapter.cpp", "—", "missing", "LoRA adapters"),
    ("src/unicode", "crates/ferrox-models/src/tokenizer/unicode.rs", "partial", "Unicode normalization"),
    # --- Per-architecture graphs (140 files) ---
    ("src/models/", "crates/ferrox-models/src/decoder.rs + engine_factory.rs", "partial", "One 6700-line decoder + 4 dedicated engines vs 140 files"),
    # --- ggml core ---
    ("ggml/src/ggml.c", "crates/ferrox-core/src/lib.rs", "partial", "No tensor graph IR; direct matmul/attn calls"),
    ("ggml/src/ggml-quants.c", "crates/ferrox-quant/src/lib.rs", "partial", "21/26 quant types on CPU"),
    ("ggml/src/ggml-backend.c", "crates/ferrox-core/src/kernel_registry.rs", "partial", "Backend dispatch + seal; no ggml op enum"),
    ("ggml/src/ggml-alloc.c", "—", "missing", "Graph allocator"),
    ("ggml/src/ggml-opt.c", "—", "out_of_scope", "Training optimizers"),
    # --- ggml backends ---
    ("ggml/src/ggml-cpu/", "crates/ferrox-quant/ + ferrox-core/", "partial", "AVX2/NEON/i8mm; no AVX512/SVE/AMX"),
    ("ggml/src/ggml-cuda/", "crates/ferrox-cuda/src/", "partial", "8 kernels vs 65 op families; no GEMM/MoE FA"),
    ("ggml/src/ggml-metal/", "crates/ferrox-metal/src/", "partial", "62 kernels vs ~70 ops; competitive MoE stack"),
    ("ggml/src/ggml-vulkan/", "crates/ferrox-vulkan/src/", "partial", "Q8_0 beachhead only; verdict GO"),
    ("ggml/src/ggml-sycl/", "—", "missing", "Intel SYCL backend"),
    ("ggml/src/ggml-hip/", "—", "missing", "AMD HIP (falls from CUDA port)"),
    ("ggml/src/ggml-opencl/", "—", "missing", "Mobile GPU OpenCL"),
    ("ggml/src/ggml-blas/", "—", "missing", "BLAS bridge"),
    ("ggml/src/ggml-rpc/", "—", "missing", "Remote RPC backend"),
    # --- common/ (shared CLI helpers) ---
    ("common/arg.cpp", "crates/ferrox-cli/src/main.rs + run.rs", "partial", "CLI flags; gaps in -ngl partial, -b/-ub"),
    ("common/sampling.cpp", "crates/ferrox-models/src/sampling.rs + sampler_order.rs", "partial", "Sampler chain ordering landed"),
    ("common/common.cpp", "crates/ferrox-cli/src/", "partial", "Shared CLI utilities"),
    ("common/json-schema-to-grammar.cpp", "crates/ferrox-models/src/grammar/json_schema/", "partial", "Converter exists; some schema edges"),
    ("common/ngram-cache.cpp", "crates/ferrox-models/src/speculative.rs", "partial", "Prompt-lookup speculative only"),
    # --- tools ---
    ("tools/cli/", "crates/ferrox-cli/src/run.rs", "partial", "ferrox run; flag parity mostly done"),
    ("tools/server/", "crates/ferrox-server/src/", "partial", "OpenAI API; slot save/load missing"),
    ("tools/quantize/", "crates/ferrox-cli/src/quantize.rs", "partial", "Q8_0 write; K-quant encoders missing"),
    ("tools/perplexity/", "crates/ferrox-cli/src/perplexity.rs", "partial", "Corpus ppl; no HellaSwag sub-tools"),
    ("tools/llama-bench/", "crates/ferrox-cli/src/bench_model.rs", "ported", "ferrox bench mirrors llama-bench"),
    ("tools/gguf-split/", "—", "missing", "Merge/split utility"),
    ("tools/imatrix/", "—", "missing", "Importance matrix for quant"),
    ("tools/tokenize/", "crates/ferrox-cli/src/parity/tokenize.rs", "partial", "Via parity tokenizer sweep"),
    ("tools/batched-bench/", "—", "missing", "Batched throughput bench"),
    ("tools/mtmd/", "—", "missing", "Multimodal (vision)"),
    ("tools/tts/", "—", "missing", "Text-to-speech"),
    ("tools/rpc/", "—", "missing", "RPC server"),
    ("tools/export-lora/", "—", "missing", "LoRA export"),
    ("tools/cvector-generator/", "—", "missing", "Control vector generation"),
    ("tools/fit-params/", "—", "missing", "Parameter fitting"),
]


@dataclass
class FileEntry:
    llama_path: str
    ferrox_path: str
    status: Status
    note: str
    category: str


def categorize(path: str) -> str:
    if path.startswith("src/models/"):
        return "architecture_graph"
    if path.startswith("src/"):
        return "core"
    if path.startswith("ggml/src/ggml-cpu"):
        return "backend_cpu"
    if path.startswith("ggml/src/ggml-cuda"):
        return "backend_cuda"
    if path.startswith("ggml/src/ggml-metal"):
        return "backend_metal"
    if path.startswith("ggml/src/ggml-vulkan"):
        return "backend_vulkan"
    if path.startswith("ggml/src/"):
        return "ggml"
    if path.startswith("common/"):
        return "common"
    if path.startswith("tools/"):
        return "tool"
    return "other"


def match_mapping(path: str) -> tuple[str, Status, str]:
    for pattern, ferrox, status, note in MAPPING:
        if path == pattern or path.startswith(pattern):
            return ferrox, status, note
    return "—", "missing", "No mapped ferrox counterpart"


def collect_llama_files() -> list[str]:
    files: list[str] = []
    for base in ["src", "ggml/src", "common", "tools"]:
        root = LLAMA / base
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix in {".cpp", ".cu", ".metal", ".h", ".cuh", ".c"} and p.is_file():
                files.append(str(p.relative_to(LLAMA)))
    return sorted(files)


def main() -> None:
    entries: list[FileEntry] = []
    status_counts: dict[str, int] = {}
    category_counts: dict[str, dict[str, int]] = {}

    for path in collect_llama_files():
        ferrox, status, note = match_mapping(path)
        cat = categorize(path)
        entry = FileEntry(path, ferrox, status, note, cat)
        entries.append(entry)
        status_counts[status] = status_counts.get(status, 0) + 1
        category_counts.setdefault(cat, {})
        category_counts[cat][status] = category_counts[cat].get(status, 0) + 1

    # Per-architecture model files
    model_files = [e for e in entries if e.category == "architecture_graph"]
    audited = [
        "llama", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "olmoe", "gemma2", "gemma3",
        "phi3", "gpt-oss", "dots1", "bailingmoe", "deepseek", "maincoder", "hunyuan-moe", "seed_oss",
    ]
    model_detail = []
    for e in model_files:
        arch = Path(e.llama_path).stem
        if arch in audited:
            st = "ported"
        elif arch in {"deepseek2", "deepseek32", "mistral4", "glm-dsa", "glm4", "kimi-linear", "gemma4"}:
            st = "partial"
        else:
            st = "missing"
        model_detail.append({"arch": arch, "llama": e.llama_path, "status": st})

    # Ferrox crate inventory
    crate_lines: dict[str, int] = {}
    for crate in sorted(CRATES.iterdir()):
        if not crate.is_dir():
            continue
        total = 0
        for rs in crate.rglob("*.rs"):
            total += len(rs.read_text(errors="replace").splitlines())
        crate_lines[crate.name] = total

    out = {
        "generated": "2026-09-02",
        "llama_root": str(LLAMA),
        "total_llama_files": len(entries),
        "status_counts": status_counts,
        "category_counts": category_counts,
        "architecture_graphs": {
            "total": len(model_files),
            "audited_ferrox": len(audited),
            "detail": model_detail,
        },
        "ferrox_crates_lines": crate_lines,
        "entries": [asdict(e) for e in entries],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
