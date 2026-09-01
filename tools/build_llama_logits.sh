#!/usr/bin/env bash
# Builds the `ferrox parity` reference dumper against an installed
# libllama. The dumper is C, not Rust, and is not part of the cargo
# workspace on purpose: it exists to be llama.cpp's own answer, so it
# links llama.cpp's own library rather than reimplementing anything.
#
# Override the llama.cpp prefix with LLAMA_CPP_PREFIX when it is not a
# Homebrew install (a source build works too — point at the directory
# holding include/llama.h and lib/libllama.*).
set -euo pipefail

PREFIX="${LLAMA_CPP_PREFIX:-}"
if [[ -z "$PREFIX" ]]; then
# THE REFERENCE'S VINTAGE IS PART OF THE RESULT.
#
# Homebrew's `llama.cpp` bottle can be months behind `.scratch/llama.cpp`,
# and a stale reference does not fail loudly -- it disagrees, and the
# disagreement reads as a ferrox bug. Both known cases were the bottle:
#
#   * bottle 7650 predates `BertNormalizer`'s `strip_accents` switch, so
#     it welds a standalone combining acute into the word and yields
#     `[UNK]` where current llama.cpp, HuggingFace and ferrox all drop
#     it. That showed up as a WordPiece "divergence" that was not one.
#   * the same bottle cannot LOAD a gemma-4 checkpoint at all, so
#     `ferrox parity` skipped that model entirely and its tokenizer went
#     unchecked.
#
# Building llama.cpp from `.scratch/llama.cpp` and pointing this script
# at it fixed both: BGE went DIVERGES to MATCH, and gemma-4 joined the
# sweep. If a single model disagrees and the rest match, SUSPECT THE
# REFERENCE before the engine:
#
#   cmake -B /tmp/llamabuild -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
#     -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
#   cmake --build /tmp/llamabuild --target llama -j8
#   cc -std=c11 -O2 -I.scratch/llama.cpp/include \
#     -I.scratch/llama.cpp/ggml/include tools/llama_logits.c \
#     -L/tmp/llamabuild/bin -lllama -Wl,-rpath,/tmp/llamabuild/bin \
#     -o target/llama_logits

  if command -v brew >/dev/null 2>&1 && brew --prefix llama.cpp >/dev/null 2>&1; then
    PREFIX="$(brew --prefix llama.cpp)"
  else
    echo "cannot find llama.cpp. Install it (brew install llama.cpp) or set" >&2
    echo "LLAMA_CPP_PREFIX to a directory holding include/llama.h and lib/libllama.*" >&2
    exit 1
  fi
fi

if [[ ! -f "$PREFIX/include/llama.h" ]]; then
  echo "no llama.h under $PREFIX/include — wrong LLAMA_CPP_PREFIX?" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/target"
OUT="$ROOT/target/llama_logits"

# -Wall -Wextra because this binary is a REFERENCE: a silently truncated
# int or a misread length here would be reported as a ferrox defect.
cc -std=c11 -O2 -Wall -Wextra \
  -I"$PREFIX/include" \
  -L"$PREFIX/lib" -lllama -Wl,-rpath,"$PREFIX/lib" \
  "$ROOT/tools/llama_logits.c" -o "$OUT"

echo "built $OUT (llama.cpp at $PREFIX)"
echo
echo "It writes into target/, so 'cargo clean' removes it — rebuild with this"
echo "script rather than assuming ferrox parity lost its reference."
