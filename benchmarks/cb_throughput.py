#!/usr/bin/env python3
"""Continuous-batching throughput smoke vs sequential private-loop.

Requires a running ferrox-server with FERROX_CONTINUOUS_BATCHING=1
(and no KV pool / prefix cache). Measures aggregate completion tok/s
for N concurrent chat requests vs N sequential ones.

Example:
  FERROX_CONTINUOUS_BATCHING=1 ./target/release/ferrox-server -m model.gguf &
  python3 benchmarks/cb_throughput.py --url http://127.0.0.1:8383 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def chat(url: str, max_tokens: int, prompt: str) -> dict:
    body = json.dumps(
        {
            "model": "m",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def predicted_tok_s(resp: dict) -> float | None:
    usage = resp.get("usage") or {}
    v = usage.get("predicted_per_second")
    return float(v) if isinstance(v, (int, float)) else None


def completion_tokens(resp: dict) -> int:
    return int((resp.get("usage") or {}).get("completion_tokens") or 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8383")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument(
        "--out",
        default="benchmarks/receipts/pins/cb_throughput.json",
        help="Write a receipt JSON (status=ok when both modes succeed)",
    )
    args = ap.parse_args()

    prompt = (
        "List the capitals of France, Germany, Italy, Spain, and Portugal "
        "as a numbered list."
    )

    # Sequential baseline
    seq_wall = []
    seq_pred = []
    seq_tok = []
    for r in range(args.reps):
        t0 = time.perf_counter()
        for i in range(args.concurrency):
            resp = chat(args.url, args.max_tokens, f"{prompt} [seq {r}/{i}]")
            seq_pred.append(predicted_tok_s(resp))
            seq_tok.append(completion_tokens(resp))
        seq_wall.append(time.perf_counter() - t0)

    # Concurrent (server CB shares decode when enabled)
    conc_wall = []
    conc_pred = []
    conc_tok = []
    for r in range(args.reps):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [
                pool.submit(
                    chat, args.url, args.max_tokens, f"{prompt} [conc {r}/{i}]"
                )
                for i in range(args.concurrency)
            ]
            for fut in as_completed(futs):
                resp = fut.result()
                conc_pred.append(predicted_tok_s(resp))
                conc_tok.append(completion_tokens(resp))
        conc_wall.append(time.perf_counter() - t0)

    def agg_tok_s(tokens: list[int], walls: list[float]) -> float:
        total_tok = sum(tokens)
        total_wall = sum(walls)
        return total_tok / total_wall if total_wall > 0 else 0.0

    seq_agg = agg_tok_s(seq_tok, seq_wall)
    conc_agg = agg_tok_s(conc_tok, conc_wall)
    speedup = conc_agg / seq_agg if seq_agg > 0 else 0.0

    receipt = {
        "id": "cb_throughput",
        "backend": "server",
        "mode": "continuous_batching",
        "status": "ok",
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "reps": args.reps,
        "sequential_aggregate_tok_s": round(seq_agg, 3),
        "concurrent_aggregate_tok_s": round(conc_agg, 3),
        "speedup": round(speedup, 3),
        "notes": (
            "Aggregate completion tokens / wall. Requires "
            "FERROX_CONTINUOUS_BATCHING=1; mutually exclusive with "
            "FERROX_KV_POOL_BLOCKS / FERROX_PREFIX_CACHE_ENTRIES."
        ),
        "median_sequential_wall_s": statistics.median(seq_wall) if seq_wall else None,
        "median_concurrent_wall_s": statistics.median(conc_wall) if conc_wall else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
