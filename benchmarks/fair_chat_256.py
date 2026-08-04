#!/usr/bin/env python3
"""Compat shim — prefer benchmarks/run_suite.py.

Maps legacy --label/--model to the suite runner when possible.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent

# Legacy labels → suite id
LABEL_MAP = {
    "llama31_8b_q4km": "llama31_8b_q4km",
    "llama31_8b_q4km_f16kv_fa_nsg8_pin": "llama31_8b_q4km",
    "llama32_1b_q4km": "llama32_1b_q4km",
    "tinyllama_q8": "tinyllama_q8",
    "tinyllama_q8_metal": "tinyllama_q8",
    "tinyllama_q8_cpu": "tinyllama_q8",
    "review_tinyllama_q8_metal": "tinyllama_q8",
    "review_tinyllama_q8_cpu": "tinyllama_q8",
    "review_llama32_1b_iq4xs": "iq4_xs",
    "iq4_xs": "iq4_xs",
    "gemma2_2b_q4km": "gemma2_2b_q4km",
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Deprecated shim → run_suite.py. Use: python3 benchmarks/run_suite.py --id … --backend …"
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ferrox-bin", default=None)
    ap.add_argument("--llama-bin", default="llama-server")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--ferrox-port", type=int, default=8383)
    ap.add_argument("--llama-port", type=int, default=8384)
    ap.add_argument("--skip-llama", action="store_true")
    ap.add_argument("--skip-ferrox", action="store_true")
    ap.add_argument(
        "--backend",
        choices=["metal", "cpu"],
        default=None,
        help="Default: cpu if FERROX_METAL=0 else metal",
    )
    args = ap.parse_args()

    suite_id = LABEL_MAP.get(args.label)
    if not suite_id:
        # Heuristic: strip date-like prefixes / use label as id
        suite_id = args.label
        print(
            f"warning: unknown legacy label {args.label!r}; "
            f"passing --id {suite_id!r} to run_suite.py",
            file=sys.stderr,
        )

    backend = args.backend
    if backend is None:
        backend = "cpu" if os.environ.get("FERROX_METAL", "1") in ("0", "false", "off") else "metal"

    cmd = [
        sys.executable,
        str(BENCH / "run_suite.py"),
        "--id",
        suite_id,
        "--backend",
        backend,
        "--gguf-override",
        args.model,
        "--llama-bin",
        args.llama_bin,
        "--max-tokens",
        str(args.max_tokens),
        "--ferrox-port",
        str(args.ferrox_port),
        "--llama-port",
        str(args.llama_port),
    ]
    if args.ferrox_bin:
        cmd += ["--ferrox-bin", args.ferrox_bin]
    if args.skip_llama:
        cmd.append("--skip-llama")
    if args.skip_ferrox:
        cmd.append("--skip-ferrox")

    print(
        "note: fair_chat_256.py is a shim; prefer "
        f"`python3 benchmarks/run_suite.py --id {suite_id} --backend {backend}`",
        file=sys.stderr,
    )
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
