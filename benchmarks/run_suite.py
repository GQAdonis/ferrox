#!/usr/bin/env python3
"""Fair-chat suite: ferrox-server vs llama-server for models in suite.json.

Stable pins at receipts/pins/{id}_{backend}.json (overwrite on re-run).
Regenerates RESULTS.md after each pin write.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
SUITE_PATH = BENCH / "suite.json"
PINS = BENCH / "receipts" / "pins"
RECEIPTS = BENCH / "receipts"

HOST_DEFAULT = "Host B (Apple M2 Pro, 10 logical cores)"
PROMPT = (
    "Write a detailed numbered list of 80 European capital cities with one "
    "interesting historical fact each. Continue until you reach item 80. unique={uid}"
)


def load_suite() -> dict:
    return json.loads(SUITE_PATH.read_text())


def redact_home(path: str) -> str:
    home = str(Path.home())
    p = str(Path(path).resolve()) if Path(path).exists() else path
    if p == home or p.startswith(home + os.sep):
        return "~" + p[len(home) :]
    # Prefer repo-relative models/ when under ROOT
    try:
        rel = Path(p).resolve().relative_to(ROOT)
        return str(rel)
    except ValueError:
        pass
    return path


def resolve_gguf(entry: dict, override: str | None = None) -> Path | None:
    raw = override or entry["gguf"]
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = ROOT / p
    return p if p.is_file() else None


def kill_port(port: int) -> None:
    try:
        out = subprocess.check_output(
            ["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"], text=True
        ).strip()
        for pid in out.split():
            os.kill(int(pid), signal.SIGTERM)
        time.sleep(2)
    except Exception:
        pass


def wait_health(port: int, timeout: float = 300.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=1
            ) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def chat(port: int, n: int, content: str) -> dict:
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": n,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        resp = json.loads(r.read().decode())
    wall = time.time() - t0
    usage = resp.get("usage") or {}
    timings = resp.get("timings") or {}
    pred = usage.get("predicted_per_second") or timings.get("predicted_per_second")
    prompt = usage.get("prompt_per_second") or timings.get("prompt_per_second")
    ct = usage.get("completion_tokens") or timings.get("predicted_n") or n
    prefix = ""
    try:
        prefix = resp["choices"][0]["message"]["content"][:60]
    except Exception:
        pass
    return {
        "wall_s": round(wall, 3),
        "predicted_per_second": pred,
        "prompt_per_second": prompt,
        "completion_tokens": ct,
        "wall_tok_s": round((ct or 0) / wall, 3) if wall else 0,
        "prefix": prefix,
    }


def aggregate(runs: list[dict], reps: int) -> dict:
    """Median across repetitions for the headline metrics (llama-bench
    style: N repetitions, report central tendency + spread, never a
    single sample). Per-rep rows are kept under `runs`."""
    ok = [r for r in runs if not r.get("error")]
    if not ok:
        return runs[0] if runs else {"error": "no runs"}

    def series(key):
        vals = [r.get(key) for r in ok]
        return [v for v in vals if isinstance(v, (int, float))]

    agg = dict(ok[-1])  # keep prefix/completion_tokens shape from a real run
    spread_name = {
        "predicted_per_second": "predicted_stddev",
        "prompt_per_second": "prompt_stddev",
        "wall_tok_s": "wall_tok_s_stddev",
    }
    for key in ("predicted_per_second", "prompt_per_second", "wall_tok_s", "wall_s"):
        vals = series(key)
        if vals:
            agg[key] = round(statistics.median(vals), 3)
            if len(vals) >= 2 and key in spread_name:
                agg[spread_name[key]] = round(statistics.stdev(vals), 3)
    agg["reps"] = reps
    agg["runs"] = [
        {
            k: r.get(k)
            for k in (
                "predicted_per_second",
                "prompt_per_second",
                "wall_tok_s",
                "wall_s",
                "completion_tokens",
                "error",
            )
            if k in r
        }
        for r in runs
    ]
    return agg


def run_ferrox(
    bin_path: str,
    model: str,
    port: int,
    n: int,
    backend: str,
    threads: int,
    reps: int = 1,
) -> dict:
    if backend == "metal":
        # A plain `cargo build --release` (no --features metal) produces a
        # binary that silently ignores FERROX_METAL=1 and decodes on CPU —
        # refuse to pin garbage numbers against it.
        if b"FERROX_METAL_FA_VEC" not in Path(bin_path).read_bytes():
            return {
                "error": "ferrox binary lacks metal feature "
                "(rebuild with --features metal)",
                "engine": "ferrox",
            }
    kill_port(port)
    env = os.environ.copy()
    env["FERROX_MODEL_PATH"] = model
    env["FERROX_ADDR"] = f"127.0.0.1:{port}"
    env["RAYON_NUM_THREADS"] = str(threads)
    if backend == "metal":
        env["FERROX_METAL"] = "1"
        env["FERROX_METAL_ATTN"] = "1"
    else:
        env["FERROX_METAL"] = "0"
        env["FERROX_CPU_INT_DOT"] = env.get("FERROX_CPU_INT_DOT", "1")
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    log_path = RECEIPTS / f"_tmp_ferrox_{port}.log"
    log = open(log_path, "w")
    proc = subprocess.Popen([bin_path], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_health(port):
            return {
                "error": "ferrox health fail",
                "log_tail": log_path.read_text()[-2000:],
                "engine": "ferrox",
            }
        chat(port, 4, f"Hi {uuid.uuid4().hex[:8]}-w")
        runs = [
            chat(port, n, PROMPT.format(uid=uuid.uuid4().hex[:8]))
            for _ in range(max(reps, 1))
        ]
        row = aggregate(runs, max(reps, 1))
        txt = log_path.read_text()
        row["metal_fails"] = sum(
            txt.count(s)
            for s in (
                "Metal attn block failed",
                "Metal dense layer failed",
                "Metal dense stack failed",
            )
        )
        row["engine"] = "ferrox"
        return row
    except Exception as e:
        return {
            "error": str(e),
            "engine": "ferrox",
            "log_tail": log_path.read_text()[-2000:] if log_path.exists() else "",
        }
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        log.close()


def run_llama(
    bin_path: str,
    model: str,
    port: int,
    n: int,
    backend: str,
    threads: int,
    reps: int = 1,
) -> dict:
    kill_port(port)
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    log_path = RECEIPTS / f"_tmp_llama_{port}.log"
    log = open(log_path, "w")
    ngl = "0" if backend == "cpu" else "99"
    cmd = [
        bin_path,
        "-m",
        model,
        "-ngl",
        ngl,
        "-t",
        str(threads),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "-c",
        "4096",
        "--jinja",
    ]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_health(port):
            return {
                "error": "llama health fail",
                "log_tail": log_path.read_text()[-2000:],
                "engine": "llama",
            }
        chat(port, 4, f"Hi {uuid.uuid4().hex[:8]}-w")
        runs = [
            chat(port, n, PROMPT.format(uid=uuid.uuid4().hex[:8]))
            for _ in range(max(reps, 1))
        ]
        row = aggregate(runs, max(reps, 1))
        row["engine"] = "llama"
        return row
    except Exception as e:
        return {
            "error": str(e),
            "engine": "llama",
            "log_tail": log_path.read_text()[-2000:] if log_path.exists() else "",
        }
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=25)
        except Exception:
            proc.kill()
        log.close()


LLAMA_PROMPT_RE = re.compile(
    r"prompt eval time.*?\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second"
)
LLAMA_EVAL_RE = re.compile(
    r"(?<!prompt )eval time.*?\(\s*[\d.]+ ms per token,\s*([\d.]+) tokens per second"
)
LLAMA_EVAL_N_RE = re.compile(r"(?<!prompt )eval time\s*=\s*[\d.]+ ms /\s*(\d+) runs")
# llama.cpp b7650 llama-cli prints one summary line to stdout instead:
#   [ Prompt: 28099.2 t/s | Generation: 244.2 t/s ]
LLAMA_SUMMARY_RE = re.compile(
    r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]"
)
FERROX_CLI_RE = re.compile(
    r"ferrox: prompt (\d+) tokens, ([\d.]+) t/s;\s*predict (\d+) tokens, ([\d.]+) t/s"
)


def run_cli_once(
    cmd: list[str], engine: str, env: dict | None = None, timeout: float = 1800.0
) -> dict:
    """One CLI completion run (fresh process — llama.cpp-style `-p ... -n N`).
    Parses the tool's own stderr timings; decode t/s excludes model load."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired:
        return {"error": "cli timeout", "engine": engine}
    wall = time.time() - t0
    err = proc.stderr or ""
    row: dict = {"wall_s": round(wall, 3), "engine": engine}
    if engine == "ferrox":
        m = FERROX_CLI_RE.search(err)
        if not m:
            row["error"] = f"no ferrox timing line (exit {proc.returncode})"
            row["log_tail"] = err[-1500:]
            return row
        row["prompt_per_second"] = float(m.group(2))
        row["completion_tokens"] = int(m.group(3))
        row["predicted_per_second"] = float(m.group(4))
    else:
        out = proc.stdout or ""
        mp = LLAMA_PROMPT_RE.search(err)
        me = LLAMA_EVAL_RE.search(err)
        mn = LLAMA_EVAL_N_RE.search(err)
        ms = LLAMA_SUMMARY_RE.search(out) or LLAMA_SUMMARY_RE.search(err)
        if me:
            row["prompt_per_second"] = float(mp.group(1)) if mp else None
            row["predicted_per_second"] = float(me.group(1))
            row["completion_tokens"] = int(mn.group(1)) if mn else None
        elif ms:
            row["prompt_per_second"] = float(ms.group(1))
            row["predicted_per_second"] = float(ms.group(2))
        else:
            row["error"] = f"no llama perf line (exit {proc.returncode})"
            row["log_tail"] = (err + out)[-1500:]
            return row
    row["prefix"] = (proc.stdout or "")[:60]
    return row


def run_cli_engine(
    engine: str,
    bin_path: str,
    model: str,
    n: int,
    backend: str,
    threads: int,
    reps: int,
) -> dict:
    """Reps × one-shot CLI completion for one engine. Strictly serial:
    the process must exit before the next rep (and before the other
    engine runs at all — never two engines resident at once)."""
    if engine == "ferrox" and backend == "metal":
        # Same guard as the server path: a binary built without
        # --features metal silently decodes on CPU.
        if b"FERROX_METAL_FA_VEC" not in Path(bin_path).read_bytes():
            return {
                "error": "ferrox binary lacks metal feature "
                "(rebuild with --features metal)",
                "engine": engine,
                "mode": "cli",
            }
    ngl = "0" if backend == "cpu" else "99"
    # Explicit env: a stray `export FERROX_METAL=1` in the calling shell
    # must not silently turn a "cpu" pin into a Metal run.
    env = {k: v for k, v in os.environ.items() if not k.startswith("FERROX_")}
    if backend == "metal":
        env["FERROX_METAL"] = "1"
        env["FERROX_METAL_ATTN"] = "1"
    else:
        env["FERROX_METAL"] = "0"
        env["FERROX_CPU_INT_DOT"] = os.environ.get("FERROX_CPU_INT_DOT", "1")
    runs = []
    for _ in range(max(reps, 1)):
        prompt = PROMPT.format(uid=uuid.uuid4().hex[:8])
        if engine == "ferrox":
            cmd = [
                bin_path, "-m", model, "-p", prompt,
                "-n", str(n), "-t", str(threads), "--ngl", ngl,
                "--temp", "0", "--no-cnv", "--ignore-eos",
            ]
        else:
            cmd = [
                bin_path, "-m", model, "-p", prompt,
                "-n", str(n), "-t", str(threads), "-ngl", ngl,
                "--temp", "0", "-no-cnv", "-st", "--ignore-eos",
                "--no-display-prompt",
            ]
        runs.append(run_cli_once(cmd, engine, env=env))
    row = aggregate(runs, max(reps, 1))
    row["engine"] = engine
    row["mode"] = "cli"
    return row


def pin_status(expect: str, ferrox: dict | None, llama: dict | None) -> str:
    if expect == "refuse":
        if ferrox and ferrox.get("error"):
            return "refuse"
    if not ferrox or ferrox.get("error"):
        return "error" if ferrox else "error"
    if not llama or llama.get("error"):
        return "error"
    if expect == "weak":
        return "weak"
    return "ok"


def write_pin(pin: dict) -> Path:
    PINS.mkdir(parents=True, exist_ok=True)
    suffix = "_cli" if pin.get("mode") == "cli" else ""
    path = PINS / f"{pin['id']}_{pin['backend']}{suffix}.json"
    path.write_text(json.dumps(pin, indent=2) + "\n")
    return path


def render_results() -> None:
    subprocess.check_call([sys.executable, str(BENCH / "render_results.py")])


def run_one(
    entry: dict,
    backend: str,
    *,
    ferrox_bin: str,
    llama_bin: str,
    max_tokens: int,
    threads: int,
    ferrox_port: int,
    llama_port: int,
    skip_llama: bool,
    skip_ferrox: bool,
    gguf_override: str | None,
    host: str,
    reps: int = 1,
    mode: str = "server",
) -> dict:
    model_path = resolve_gguf(entry, gguf_override)
    pin: dict = {
        "schema": 1,
        "id": entry["id"],
        "name": entry["name"],
        "backend": backend,
        "expect": entry.get("expect", "ok"),
        "date": date.today().isoformat(),
        "host": host,
        "model": entry["gguf"] if not model_path else redact_home(str(model_path)),
        "max_tokens": max_tokens,
        "reps": reps,
        "mode": mode,
        "workload": PROMPT.split("unique=")[0].strip() + " (unique suffix)",
        "notes": entry.get("notes"),
        "ferrox": None,
        "llama": None,
    }
    # Prefer suite workload string
    suite = load_suite()
    pin["workload"] = suite.get("workload", pin["workload"])

    if model_path is None:
        pin["status"] = "missing"
        write_pin(pin)
        print(f"missing {entry['id']} ({entry['gguf']})", flush=True)
        return pin

    if entry.get("expect") == "refuse" and not skip_ferrox:
        # Still attempt both so refuse is evidenced; ferrox may fail health.
        pass

    print(f"=== {entry['id']} / {backend} / {mode} ===", flush=True)
    if mode == "cli":
        # Strictly serial: llama-cli runs to completion and exits before
        # ferrox starts — never two engines resident at once.
        if not skip_llama:
            print("--- llama (cli) ---", flush=True)
            pin["llama"] = run_cli_engine(
                "llama", llama_bin, str(model_path), max_tokens, backend, threads, reps
            )
            print(json.dumps(pin["llama"], indent=2), flush=True)
        if not skip_ferrox:
            print("--- ferrox (cli) ---", flush=True)
            pin["ferrox"] = run_cli_engine(
                "ferrox", ferrox_bin, str(model_path), max_tokens, backend, threads, reps
            )
            print(json.dumps(pin["ferrox"], indent=2), flush=True)
    else:
        if not skip_ferrox:
            print("--- ferrox ---", flush=True)
            pin["ferrox"] = run_ferrox(
                ferrox_bin, str(model_path), ferrox_port, max_tokens, backend, threads, reps
            )
            print(json.dumps(pin["ferrox"], indent=2), flush=True)
        if not skip_llama:
            print("--- llama ---", flush=True)
            pin["llama"] = run_llama(
                llama_bin, str(model_path), llama_port, max_tokens, backend, threads, reps
            )
            print(json.dumps(pin["llama"], indent=2), flush=True)

    pin["status"] = pin_status(entry.get("expect", "ok"), pin["ferrox"], pin["llama"])
    # Prefer refuse when expect=refuse and ferrox errored
    if entry.get("expect") == "refuse" and pin.get("ferrox") and pin["ferrox"].get("error"):
        pin["status"] = "refuse"

    path = write_pin(pin)
    print(f"wrote {path.relative_to(ROOT)} status={pin['status']}", flush=True)
    fp = (pin.get("ferrox") or {}).get("predicted_per_second")
    lp = (pin.get("llama") or {}).get("predicted_per_second")
    if (
        isinstance(fp, (int, float))
        and isinstance(lp, (int, float))
        and fp > 0
        and not (pin.get("ferrox") or {}).get("error")
        and not (pin.get("llama") or {}).get("error")
    ):
        fs = (pin.get("ferrox") or {}).get("predicted_stddev")
        ls = (pin.get("llama") or {}).get("predicted_stddev")
        fs_str = f"±{fs:.2f}" if isinstance(fs, (int, float)) else ""
        ls_str = f"±{ls:.2f}" if isinstance(ls, (int, float)) else ""
        print(
            f"summary predicted tok/s (median of {reps}): "
            f"ferrox={fp:.3f}{fs_str} llama={lp:.3f}{ls_str} gap={lp/fp:.2f}x"
        )
    return pin


def main() -> None:
    suite = load_suite()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", help="Suite model id (default: all matching --backend)")
    ap.add_argument(
        "--backend",
        choices=["metal", "cpu"],
        help="Backend to run (required unless --list)",
    )
    ap.add_argument("--list", action="store_true", help="List suite entries and exit")
    ap.add_argument("--gguf-override", help="Override GGUF path for --id")
    ap.add_argument(
        "--mode",
        choices=["server", "cli"],
        default="server",
        help="server = fair-chat HTTP servers; cli = one-shot completion "
        "(llama-cli vs `ferrox run`, strictly sequential)",
    )
    ap.add_argument("--ferrox-bin", default=None, help="Default depends on --mode")
    ap.add_argument("--llama-bin", default=None, help="Default depends on --mode")
    ap.add_argument("--max-tokens", type=int, default=suite.get("max_tokens", 512))
    ap.add_argument(
        "--reps",
        type=int,
        default=suite.get("reps", 3),
        help="Measured repetitions per engine (median ± stddev; llama-bench style)",
    )
    ap.add_argument("--threads", type=int, default=suite.get("threads", 10))
    ap.add_argument("--ferrox-port", type=int, default=8383)
    ap.add_argument("--llama-port", type=int, default=8384)
    ap.add_argument("--skip-llama", action="store_true")
    ap.add_argument("--skip-ferrox", action="store_true")
    ap.add_argument("--skip-missing", action="store_true", help="Do not write missing pins")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    if args.list:
        for m in suite["models"]:
            present = "yes" if resolve_gguf(m) else "no"
            print(f"{m['id']:20} backends={','.join(m['backends']):12} gguf={present}  {m['name']}")
        return

    if not args.backend:
        raise SystemExit("--backend metal|cpu required (or --list)")

    # Resolve binaries per mode. Prefer the pinned bench copies (immune
    # to concurrent `cargo build` runs replacing target/release binaries,
    # possibly without --features metal — which silently decodes on CPU).
    if args.ferrox_bin is None:
        if args.mode == "cli":
            bench = ROOT / "target/bench/ferrox-cli-metal"
            args.ferrox_bin = str(bench if bench.is_file() else ROOT / "target/release/ferrox")
        else:
            bench = ROOT / "target/bench/ferrox-server-metal"
            args.ferrox_bin = str(
                bench if bench.is_file() else ROOT / "target/release/ferrox-server"
            )
    if args.llama_bin is None:
        args.llama_bin = "llama-cli" if args.mode == "cli" else "llama-server"

    host = suite.get("host_default", HOST_DEFAULT)
    entries = suite["models"]
    if args.id:
        entries = [m for m in entries if m["id"] == args.id]
        if not entries:
            raise SystemExit(f"unknown suite id: {args.id}")
        if args.backend not in entries[0]["backends"]:
            raise SystemExit(
                f"id {args.id} does not list backend {args.backend} "
                f"(has {entries[0]['backends']})"
            )
    else:
        entries = [m for m in entries if args.backend in m["backends"]]

    for entry in entries:
        if resolve_gguf(entry, args.gguf_override if args.id else None) is None:
            if args.skip_missing:
                print(f"skip missing {entry['id']}", flush=True)
                continue
        run_one(
            entry,
            args.backend,
            ferrox_bin=args.ferrox_bin,
            llama_bin=args.llama_bin,
            max_tokens=args.max_tokens,
            threads=args.threads,
            ferrox_port=args.ferrox_port,
            llama_port=args.llama_port,
            skip_llama=args.skip_llama,
            skip_ferrox=args.skip_ferrox,
            gguf_override=args.gguf_override if args.id else None,
            host=host,
            reps=max(args.reps, 1),
            mode=args.mode,
        )

    if not args.no_render:
        render_results()
        print("updated RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
