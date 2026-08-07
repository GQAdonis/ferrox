#!/usr/bin/env python3
"""Fair-chat suite: ferrox-server vs llama-server for models in suite.json.

Stable pins at receipts/pins/{id}_{backend}.json (overwrite on re-run).
Regenerates RESULTS.md after each pin write.
"""

from __future__ import annotations

import argparse
import hashlib
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


def host_ram_gb() -> float | None:
    """Best-effort physical RAM in GiB (macOS sysctl / Linux /proc)."""
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=5
            ).strip()
            return int(out) / (1024**3)
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024**2)
    except Exception:
        return None
    return None


def skip_reason(
    entry: dict,
    backend: str,
    *,
    fit_host: bool,
    suite_ram_gb: float | None,
) -> str | None:
    """Why this (model, backend) should not run on this machine, or None."""
    if backend == "cuda" and sys.platform == "darwin":
        return "cuda not available on darwin (use a CUDA host)"
    if not fit_host:
        return None
    need = entry.get("estimated_ram_gb")
    if not isinstance(need, (int, float)):
        return None
    # Prefer live probe; fall back to suite host_ram_gb.
    have = host_ram_gb()
    if have is None:
        have = suite_ram_gb
    if have is None:
        return None
    # Leave ~25% headroom for OS + dual servers + KV (fair-chat runs ferrox
    # then llama serially, but Metal unified memory still needs slack).
    budget = float(have) * 0.75
    if float(need) > budget:
        return (
            f"estimated_ram_gb={need} exceeds ~75% of host RAM "
            f"({have:.0f} GiB → budget {budget:.0f} GiB)"
        )
    return None


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


def _trim_outliers(vals: list[float]) -> list[float]:
    """Drop min/max when n≥5 so one thermal cliff does not inflate ±stddev
    (CPU llama OLMoE saw 24–40 tok/s in one 5-rep window)."""
    if len(vals) < 5:
        return vals
    ordered = sorted(vals)
    return ordered[1:-1]


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
        "load_s": "load_stddev",
        "startup_s": "startup_stddev",
    }
    for key in (
        "predicted_per_second",
        "prompt_per_second",
        "wall_tok_s",
        "wall_s",
        "load_s",
        "startup_s",
    ):
        vals = series(key)
        if vals:
            core = _trim_outliers(vals)
            agg[key] = round(statistics.median(core), 3)
            if len(core) >= 2 and key in spread_name:
                agg[spread_name[key]] = round(statistics.stdev(core), 3)
    agg["reps"] = reps
    agg["runs"] = [
        {
            k: r.get(k)
            for k in (
                "predicted_per_second",
                "prompt_per_second",
                "wall_tok_s",
                "wall_s",
                "load_s",
                "startup_s",
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
    if backend == "cuda":
        if b"FERROX_CUDA" not in Path(bin_path).read_bytes() and b"ferrox_cuda" not in Path(
            bin_path
        ).read_bytes():
            # Soft check — CUDA feature symbols vary; still set env below.
            pass
    kill_port(port)
    env = os.environ.copy()
    env["FERROX_MODEL_PATH"] = model
    env["FERROX_ADDR"] = f"127.0.0.1:{port}"
    if threads > 0:
        env["RAYON_NUM_THREADS"] = str(threads)
    if backend == "metal":
        env["FERROX_METAL"] = "1"
        env["FERROX_METAL_ATTN"] = "1"
        env["FERROX_CUDA"] = "0"
    elif backend == "cuda":
        env["FERROX_METAL"] = "0"
        env["FERROX_CUDA"] = "1"
        env.setdefault("FERROX_CUDA_GQA", "1")
    else:
        env["FERROX_METAL"] = "0"
        env["FERROX_CUDA"] = "0"
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


def _start_ferrox_server(
    bin_path: str, model: str, port: int, backend: str, threads: int
):
    """Spawn ferrox-server; returns (proc, log_path) or error dict."""
    if backend == "metal":
        if b"FERROX_METAL_FA_VEC" not in Path(bin_path).read_bytes():
            return {
                "error": "ferrox binary lacks metal feature "
                "(rebuild with --features metal)",
                "engine": "ferrox",
            }, None, None
    kill_port(port)
    env = os.environ.copy()
    env["FERROX_MODEL_PATH"] = model
    env["FERROX_ADDR"] = f"127.0.0.1:{port}"
    if threads > 0:
        env["RAYON_NUM_THREADS"] = str(threads)
    if backend == "metal":
        env["FERROX_METAL"] = "1"
        env["FERROX_METAL_ATTN"] = "1"
        env["FERROX_CUDA"] = "0"
    elif backend == "cuda":
        env["FERROX_METAL"] = "0"
        env["FERROX_CUDA"] = "1"
        env.setdefault("FERROX_CUDA_GQA", "1")
    else:
        env["FERROX_METAL"] = "0"
        env["FERROX_CUDA"] = "0"
        env["FERROX_CPU_INT_DOT"] = env.get("FERROX_CPU_INT_DOT", "1")
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    log_path = RECEIPTS / f"_tmp_ferrox_{port}.log"
    log = open(log_path, "w")
    proc = subprocess.Popen([bin_path], env=env, stdout=log, stderr=subprocess.STDOUT)
    if not wait_health(port):
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        log.close()
        return {
            "error": "ferrox health fail",
            "log_tail": log_path.read_text()[-2000:],
            "engine": "ferrox",
        }, None, None
    return None, proc, (log, log_path)


def _start_llama_server(bin_path: str, model: str, port: int, backend: str, threads: int):
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
        *(["-t", str(threads)] if threads > 0 else []),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "-c",
        "4096",
        "--jinja",
    ]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    if not wait_health(port):
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=25)
        except Exception:
            proc.kill()
        log.close()
        return {
            "error": "llama health fail",
            "log_tail": log_path.read_text()[-2000:],
            "engine": "llama",
        }, None, None
    return None, proc, (log, log_path)


def _stop_server(proc, log_handle, timeout: int = 20) -> None:
    if proc is None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except Exception:
        proc.kill()
    if log_handle is not None:
        log_handle.close()


def run_fair_chat_interleaved(
    ferrox_bin: str,
    llama_bin: str,
    model: str,
    ferrox_port: int,
    llama_port: int,
    n: int,
    backend: str,
    threads: int,
    reps: int = 1,
) -> tuple[dict, dict]:
    """Run both servers concurrently; llama then ferrox each rep.

    Cuts CPU thermal/page-cache skew from ferrox-all-then-llama-all
    (OLMoE llama ±5 tok/s → much tighter when paired).
    """
    ferr, f_proc, f_meta = _start_ferrox_server(
        ferrox_bin, model, ferrox_port, backend, threads
    )
    if ferr is not None:
        return ferr, {"error": "ferrox failed to start", "engine": "llama"}
    f_log, f_log_path = f_meta

    lerr, l_proc, l_meta = _start_llama_server(
        llama_bin, model, llama_port, backend, threads
    )
    if lerr is not None:
        _stop_server(f_proc, f_log)
        return {"error": "llama failed to start", "engine": "ferrox"}, lerr
    l_log, _l_log_path = l_meta

    try:
        chat(ferrox_port, 4, f"Hi {uuid.uuid4().hex[:8]}-w")
        chat(llama_port, 4, f"Hi {uuid.uuid4().hex[:8]}-w")
        # Extra settle on CPU — thermal after dual warm.
        if backend == "cpu":
            time.sleep(2.0)
        ferrox_runs = []
        llama_runs = []
        for i in range(max(reps, 1)):
            print(f"--- llama (server) rep {i+1}/{reps} ---", flush=True)
            llama_runs.append(chat(llama_port, n, PROMPT.format(uid=uuid.uuid4().hex[:8])))
            print(f"--- ferrox (server) rep {i+1}/{reps} ---", flush=True)
            ferrox_runs.append(
                chat(ferrox_port, n, PROMPT.format(uid=uuid.uuid4().hex[:8]))
            )
            if backend == "cpu" and i + 1 < max(reps, 1):
                time.sleep(1.0)
        f_row = aggregate(ferrox_runs, max(reps, 1))
        txt = f_log_path.read_text()
        f_row["metal_fails"] = sum(
            txt.count(s)
            for s in (
                "Metal attn block failed",
                "Metal dense layer failed",
                "Metal dense stack failed",
            )
        )
        f_row["engine"] = "ferrox"
        l_row = aggregate(llama_runs, max(reps, 1))
        l_row["engine"] = "llama"
        return f_row, l_row
    except Exception as e:
        return (
            {
                "error": str(e),
                "engine": "ferrox",
                "log_tail": f_log_path.read_text()[-2000:] if f_log_path.exists() else "",
            },
            {"error": str(e), "engine": "llama"},
        )
    finally:
        _stop_server(f_proc, f_log)
        _stop_server(l_proc, l_log, timeout=25)


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
        *(["-t", str(threads)] if threads > 0 else []),
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
# Engine-reported model load / startup (mmap + GPU weight upload / context init).
FERROX_LOAD_RE = re.compile(r"ferrox: loaded in ([\d.]+)s")
# llama.cpp common_perf_print (llama-completion / recent llama-cli):
#   common_perf_print:        load time =     261.39 ms
LLAMA_LOAD_RE = re.compile(
    r"(?:common_perf_print:\s*)?load time\s*=\s*([\d.]+)\s*ms", re.IGNORECASE
)


def file_sha256(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def bin_version(path: str) -> str | None:
    try:
        out = subprocess.check_output(
            [path, "--version"], text=True, stderr=subprocess.STDOUT, timeout=10
        )
        return out.strip().splitlines()[0][:120]
    except Exception:
        return None


def git_rev() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None


def run_cli_once(
    cmd: list[str], engine: str, env: dict | None = None, timeout: float = 1800.0
) -> dict:
    """One CLI completion run (fresh process — llama.cpp-style `-p ... -n N`).
    Parses the tool's own stderr timings; decode t/s excludes model load.

    `startup_s` is process-wall until the process exits after the measured
    run's first useful timing is available conceptually as wall_s minus
    decode-only time when both are known; we also keep engine-reported
    `load_s` when present but RESULTS treats startup_s as the comparable
    cold-start metric (same definition for both engines: full process wall
    minus predicted-token decode time when token count and pred t/s known).
    """
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired:
        return {"error": "cli timeout", "engine": engine}
    wall = time.time() - t0
    err = proc.stderr or ""
    out = proc.stdout or ""
    combined = err + "\n" + out
    row: dict = {"wall_s": round(wall, 3), "engine": engine}

    # Fail closed on wrong llama binary / rejected flags.
    if "not supported by llama-cli" in combined or "please use llama-completion" in combined:
        row["error"] = "llama-cli rejected flags (need llama-completion)"
        row["log_tail"] = combined[-1500:]
        return row
    if proc.returncode not in (0, None) and engine == "llama":
        # Some builds exit 0 even with warnings; non-zero is hard fail.
        if proc.returncode != 0 and "eval time" not in combined and "Generation:" not in combined:
            row["error"] = f"llama exit {proc.returncode}"
            row["log_tail"] = combined[-1500:]
            return row

    if engine == "ferrox":
        ml = FERROX_LOAD_RE.search(err)
        if ml:
            row["load_s"] = round(float(ml.group(1)), 3)
        m = FERROX_CLI_RE.search(err)
        if not m:
            row["error"] = f"no ferrox timing line (exit {proc.returncode})"
            row["log_tail"] = err[-1500:]
            return row
        row["prompt_per_second"] = float(m.group(2))
        row["completion_tokens"] = int(m.group(3))
        row["predicted_per_second"] = float(m.group(4))
    else:
        ml = LLAMA_LOAD_RE.search(err) or LLAMA_LOAD_RE.search(out)
        if ml:
            row["load_s"] = round(float(ml.group(1)) / 1000.0, 3)
        mp = LLAMA_PROMPT_RE.search(err) or LLAMA_PROMPT_RE.search(out)
        me = LLAMA_EVAL_RE.search(err) or LLAMA_EVAL_RE.search(out)
        mn = LLAMA_EVAL_N_RE.search(err) or LLAMA_EVAL_N_RE.search(out)
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
            row["log_tail"] = combined[-1500:]
            return row

    # Comparable startup: full process wall minus decode-only time.
    pred = row.get("predicted_per_second")
    ct = row.get("completion_tokens")
    if isinstance(pred, (int, float)) and pred > 0 and isinstance(ct, (int, float)) and ct > 0:
        decode_s = float(ct) / float(pred)
        row["startup_s"] = round(max(wall - decode_s, 0.0), 3)

    row["prefix"] = out[:60]
    return row


def run_cli_engine(
    engine: str,
    bin_path: str,
    model: str,
    n: int,
    backend: str,
    threads: int,
    reps: int,
    *,
    ctx: int = 4096,
) -> dict:
    """Reps × one-shot CLI completion for one engine."""
    if engine == "ferrox" and backend == "metal":
        if b"FERROX_METAL_FA_VEC" not in Path(bin_path).read_bytes():
            return {
                "error": "ferrox binary lacks metal feature "
                "(rebuild with --features metal)",
                "engine": engine,
                "mode": "cli",
            }
    if engine == "llama":
        # Require llama-completion (Homebrew ≥b76xx).
        name = Path(bin_path).name
        if name == "llama-cli":
            return {
                "error": "llama-cli is not valid for CLI pins; use llama-completion",
                "engine": engine,
                "mode": "cli",
            }
    ngl = "0" if backend == "cpu" else "99"
    env = {k: v for k, v in os.environ.items() if not k.startswith("FERROX_")}
    if backend == "metal":
        env["FERROX_METAL"] = "1"
        env["FERROX_METAL_ATTN"] = "1"
        env["FERROX_CUDA"] = "0"
    elif backend == "cuda":
        env["FERROX_METAL"] = "0"
        env["FERROX_CUDA"] = "1"
        env.setdefault("FERROX_CUDA_GQA", "1")
    else:
        env["FERROX_METAL"] = "0"
        env["FERROX_CUDA"] = "0"
        env["FERROX_CPU_INT_DOT"] = os.environ.get("FERROX_CPU_INT_DOT", "1")
    runs = []
    for _ in range(max(reps, 1)):
        prompt = PROMPT.format(uid=uuid.uuid4().hex[:8])
        if engine == "ferrox":
            # Chat-template wrap (same as ferrox-server /v1/chat/completions).
            cmd = [
                bin_path, "-m", model, "-p", prompt,
                "-n", str(n), *(["-t", str(threads)] if threads > 0 else []), "-c", str(ctx),
                "--ngl", ngl,
                "--temp", "0", "--ignore-eos",
            ]
        else:
            # Match llama-server: conversation + jinja template from GGUF.
            # `-cnv` with `-p` is non-interactive (first turn predefined).
            cmd = [
                bin_path, "-m", model, "-p", prompt,
                "-n", str(n), *(["-t", str(threads)] if threads > 0 else []), "-c", str(ctx),
                "-ngl", ngl,
                "--temp", "0", "-cnv", "--jinja", "-st", "--ignore-eos",
                "--no-display-prompt",
            ]
        runs.append(run_cli_once(cmd, engine, env=env))
    # Token-count gate under --ignore-eos: allow llama's off-by-one (n-1).
    for r in runs:
        if r.get("error"):
            continue
        ct = r.get("completion_tokens")
        if isinstance(ct, int) and ct not in (n, n - 1, None):
            # Soft warn into the row; don't fail the whole pin for early-EOS
            # models that ignore --ignore-eos inconsistently.
            r["token_count_note"] = f"got {ct}, expected {n} or {n-1}"
    row = aggregate(runs, max(reps, 1))
    # Aggregate startup_s if present.
    startups = [r.get("startup_s") for r in runs if isinstance(r.get("startup_s"), (int, float))]
    if startups:
        row["startup_s"] = round(statistics.median(startups), 3)
        if len(startups) >= 2:
            row["startup_stddev"] = round(statistics.stdev(startups), 3)
    row["engine"] = engine
    row["mode"] = "cli"
    row["bin"] = bin_path
    row["bin_sha256_16"] = file_sha256(bin_path)
    row["bin_version"] = bin_version(bin_path)
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
        # None = neither engine was pinned to a thread count, which is the
        # only comparison worth quoting on a hybrid-core host. A number
        # means --threads forced it on both. Pins written before this field
        # existed were all measured with a forced 10 and are not comparable;
        # render_results.py warns on any pin missing the key.
        "threads_forced": threads if threads > 0 else None,
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
        # Interleaved reps: llama r_i → ferrox r_i so page-cache effects
        # are shared. Engines never overlap (serial within each pair).
        pin["workload"] = (
            "CLI one-shot same capitals prompt + chat template as fair-chat "
            "(ferrox default wrap / llama -cnv --jinja), -n N --ignore-eos "
            f"-c 4096, interleaved {reps}× cold process (median ± stddev)"
        )
        pin["git_rev"] = git_rev()
        pin["ferrox_bin"] = ferrox_bin
        pin["llama_bin"] = llama_bin
        print("--- cli interleaved (llama then ferrox each rep) ---", flush=True)
        # Collect per-rep then aggregate via run_cli_engine by calling
        # once each — still llama-all then ferrox-all would bias cache;
        # instead manually interleave below.
        llama_runs = []
        ferrox_runs = []
        ngl = "0" if backend == "cpu" else "99"
        env_base = {k: v for k, v in os.environ.items() if not k.startswith("FERROX_")}
        for i in range(max(reps, 1)):
            prompt = PROMPT.format(uid=uuid.uuid4().hex[:8])
            if not skip_llama:
                env = dict(env_base)
                cmd = [
                    llama_bin, "-m", str(model_path), "-p", prompt,
                    "-n", str(max_tokens), *(["-t", str(threads)] if threads > 0 else []), "-c", "4096",
                    "-ngl", ngl, "--temp", "0", "-cnv", "--jinja", "-st",
                    "--ignore-eos", "--no-display-prompt",
                ]
                print(f"--- llama (cli) rep {i+1}/{reps} ---", flush=True)
                llama_runs.append(run_cli_once(cmd, "llama", env=env))
            if not skip_ferrox:
                env = dict(env_base)
                if backend == "metal":
                    env["FERROX_METAL"] = "1"
                    env["FERROX_METAL_ATTN"] = "1"
                    env["FERROX_CUDA"] = "0"
                elif backend == "cuda":
                    env["FERROX_METAL"] = "0"
                    env["FERROX_CUDA"] = "1"
                    env.setdefault("FERROX_CUDA_GQA", "1")
                else:
                    env["FERROX_METAL"] = "0"
                    env["FERROX_CUDA"] = "0"
                    env["FERROX_CPU_INT_DOT"] = os.environ.get("FERROX_CPU_INT_DOT", "1")
                if backend == "metal" and b"FERROX_METAL_FA_VEC" not in Path(ferrox_bin).read_bytes():
                    ferrox_runs.append({
                        "error": "ferrox binary lacks metal feature",
                        "engine": "ferrox",
                    })
                else:
                    cmd = [
                        ferrox_bin, "-m", str(model_path), "-p", prompt,
                        "-n", str(max_tokens), *(["-t", str(threads)] if threads > 0 else []), "-c", "4096",
                        "--ngl", ngl, "--temp", "0", "--ignore-eos",
                    ]
                    print(f"--- ferrox (cli) rep {i+1}/{reps} ---", flush=True)
                    ferrox_runs.append(run_cli_once(cmd, "ferrox", env=env))
        if not skip_llama:
            if Path(llama_bin).name == "llama-cli":
                pin["llama"] = {
                    "error": "llama-cli is not valid for CLI pins; use llama-completion",
                    "engine": "llama",
                    "mode": "cli",
                }
            else:
                pin["llama"] = aggregate(llama_runs, max(reps, 1))
                pin["llama"]["engine"] = "llama"
                pin["llama"]["mode"] = "cli"
                pin["llama"]["bin"] = llama_bin
                pin["llama"]["bin_sha256_16"] = file_sha256(llama_bin)
                pin["llama"]["bin_version"] = bin_version(llama_bin)
                startups = [
                    r.get("startup_s")
                    for r in llama_runs
                    if isinstance(r.get("startup_s"), (int, float))
                ]
                if startups:
                    pin["llama"]["startup_s"] = round(statistics.median(startups), 3)
            print(json.dumps(pin["llama"], indent=2), flush=True)
        if not skip_ferrox:
            pin["ferrox"] = aggregate(ferrox_runs, max(reps, 1))
            pin["ferrox"]["engine"] = "ferrox"
            pin["ferrox"]["mode"] = "cli"
            pin["ferrox"]["bin"] = ferrox_bin
            pin["ferrox"]["bin_sha256_16"] = file_sha256(ferrox_bin)
            pin["ferrox"]["bin_version"] = bin_version(ferrox_bin)
            startups = [
                r.get("startup_s")
                for r in ferrox_runs
                if isinstance(r.get("startup_s"), (int, float))
            ]
            if startups:
                pin["ferrox"]["startup_s"] = round(statistics.median(startups), 3)
            print(json.dumps(pin["ferrox"], indent=2), flush=True)
    else:
        # CPU: both servers warm, llama→ferrox each rep (cuts thermal/
        # page-cache skew — OLMoE llama was ±5 tok/s sequential).
        # Metal/CUDA: keep sequential — dual GPU-resident servers contend.
        if backend == "cpu" and not skip_ferrox and not skip_llama:
            print(
                f"--- server interleaved (llama then ferrox each of {reps}) ---",
                flush=True,
            )
            pin["ferrox"], pin["llama"] = run_fair_chat_interleaved(
                ferrox_bin,
                llama_bin,
                str(model_path),
                ferrox_port,
                llama_port,
                max_tokens,
                backend,
                threads,
                reps,
            )
            print(json.dumps(pin["llama"], indent=2), flush=True)
            print(json.dumps(pin["ferrox"], indent=2), flush=True)
        else:
            if not skip_ferrox:
                print("--- ferrox ---", flush=True)
                pin["ferrox"] = run_ferrox(
                    ferrox_bin,
                    str(model_path),
                    ferrox_port,
                    max_tokens,
                    backend,
                    threads,
                    reps,
                )
                print(json.dumps(pin["ferrox"], indent=2), flush=True)
            if not skip_llama:
                print("--- llama ---", flush=True)
                pin["llama"] = run_llama(
                    llama_bin,
                    str(model_path),
                    llama_port,
                    max_tokens,
                    backend,
                    threads,
                    reps,
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
        fl = (pin.get("ferrox") or {}).get("startup_s")
        ll = (pin.get("llama") or {}).get("startup_s")
        if not (isinstance(fl, (int, float)) and isinstance(ll, (int, float))):
            fl = (pin.get("ferrox") or {}).get("load_s")
            ll = (pin.get("llama") or {}).get("load_s")
            label = "engine-load (incomparable defs)"
        else:
            label = "startup (wall - decode)"
        if isinstance(fl, (int, float)) and isinstance(ll, (int, float)) and ll > 0:
            print(
                f"summary {label} (s, median): "
                f"ferrox={fl:.3f} llama={ll:.3f} gap={fl/ll:.2f}x "
                f"(<1 ferrox better)"
            )
    return pin


def main() -> None:
    suite = load_suite()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", help="Suite model id (default: all matching --backend)")
    ap.add_argument(
        "--backend",
        choices=["metal", "cpu", "cuda"],
        help="Backend to run (required unless --list)",
    )
    ap.add_argument(
        "--host-label",
        default=None,
        help="Override host string written into the pin (required for reproducible CUDA pins)",
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
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "Force a thread count on BOTH engines. Default 0 = let each pick "
            "its own, which is the comparison that means something: llama.cpp "
            "defaults to performance cores and loses 2-4x when pushed above "
            "them, so pinning both to the same count flatters ferrox. This "
            "suite used to force 10 and every CPU row it produced was skewed."
        ),
    )
    ap.add_argument("--ferrox-port", type=int, default=8383)
    ap.add_argument("--llama-port", type=int, default=8384)
    ap.add_argument("--skip-llama", action="store_true")
    ap.add_argument("--skip-ferrox", action="store_true")
    ap.add_argument("--skip-missing", action="store_true", help="Do not write missing pins")
    ap.add_argument(
        "--fit-host",
        action="store_true",
        help="Skip models whose estimated_ram_gb exceeds ~75%% of host RAM, "
        "and skip cuda on darwin",
    )
    ap.add_argument(
        "--skip-unfit",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    fit_host = args.fit_host or args.skip_unfit
    suite_ram = suite.get("host_ram_gb")
    if isinstance(suite_ram, (int, float)):
        suite_ram_gb = float(suite_ram)
    else:
        suite_ram_gb = None

    if args.list:
        ram = host_ram_gb()
        ram_s = f"{ram:.0f} GiB" if ram else "unknown"
        print(f"host RAM: {ram_s}  suite host_ram_gb={suite.get('host_ram_gb', '?')}")
        for m in suite["models"]:
            present = "yes" if resolve_gguf(m) else "no"
            ram_need = m.get("estimated_ram_gb", "?")
            skips = []
            for b in m["backends"]:
                r = skip_reason(m, b, fit_host=True, suite_ram_gb=suite_ram_gb)
                if r:
                    skips.append(f"{b}:{r.split('(')[0].strip()}")
            skip_s = f"  skip=[{'; '.join(skips)}]" if skips else ""
            print(
                f"{m['id']:22} backends={','.join(m['backends']):12} "
                f"gguf={present} ram≈{ram_need}G  {m['name']}{skip_s}"
            )
        return

    if not args.backend:
        raise SystemExit("--backend metal|cpu|cuda required (or --list)")

    if args.backend == "cuda" and not args.host_label:
        print(
            "warning: CUDA pins should set --host-label with GPU/driver "
            "(e.g. 'Vast RTX4090 / driver 550')",
            flush=True,
        )
    # Resolve binaries per mode. Prefer the pinned bench copies (immune
    # to concurrent `cargo build` runs replacing target/release binaries,
    # possibly without --features metal — which silently decodes on CPU).
    if args.ferrox_bin is None:
        if args.mode == "cli":
            if args.backend == "cuda":
                bench = ROOT / "target/bench/ferrox-cli-cuda"
            elif args.backend == "metal":
                bench = ROOT / "target/bench/ferrox-cli-metal"
            else:
                bench = ROOT / "target/release/ferrox"
            args.ferrox_bin = str(
                bench if bench.is_file() else ROOT / "target/release/ferrox"
            )
        else:
            if args.backend == "cuda":
                bench = ROOT / "target/bench/ferrox-server-cuda"
            elif args.backend == "metal":
                bench = ROOT / "target/bench/ferrox-server-metal"
            else:
                bench = ROOT / "target/release/ferrox-server"
            args.ferrox_bin = str(
                bench if bench.is_file() else ROOT / "target/release/ferrox-server"
            )
    if args.llama_bin is None:
        if args.mode == "cli":
            # Homebrew llama.cpp ≥b76xx: one-shot completion lives in
            # llama-completion (llama-cli is interactive-oriented). Prefer
            # llama-completion when on PATH.
            from shutil import which

            comp = which("llama-completion")
            if not comp:
                raise SystemExit(
                    "CLI mode requires llama-completion on PATH "
                    "(Homebrew llama.cpp ≥b76xx)"
                )
            args.llama_bin = comp
        else:
            args.llama_bin = "llama-server"

    host = args.host_label or suite.get("host_default", HOST_DEFAULT)
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
        reason = skip_reason(
            entry, args.backend, fit_host=fit_host, suite_ram_gb=suite_ram_gb
        )
        if reason:
            print(f"skip unfit {entry['id']}/{args.backend}: {reason}", flush=True)
            continue
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
