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

cc -std=c11 -O2 \
  -I"$PREFIX/include" \
  -L"$PREFIX/lib" -lllama -Wl,-rpath,"$PREFIX/lib" \
  "$ROOT/tools/llama_logits.c" -o "$OUT"

echo "built $OUT (llama.cpp at $PREFIX)"
