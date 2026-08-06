#!/usr/bin/env python3
"""Generate benchmarks/RESULTS.md from suite.json + receipts/pins/*.json.

Do not hand-edit RESULTS.md headlines — re-run this script (or run_suite.py).
"""

from __future__ import annotations

import json
from pathlib import Path

BENCH = Path(__file__).resolve().parent
SUITE_PATH = BENCH / "suite.json"
PINS = BENCH / "receipts" / "pins"
OUT = BENCH / "RESULTS.md"


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def fmt_num(x, sd=None) -> str:
    if x is None:
        return "—"
    if isinstance(sd, (int, float)):
        return f"**{x:.2f}** ±{sd:.1f}"
    return f"**{x:.2f}**"


def stddev(row: dict | None):
    if not row or row.get("error"):
        return None
    v = row.get("predicted_stddev")
    return v if isinstance(v, (int, float)) else None


def pred(row: dict | None):
    if not row or row.get("error"):
        return None
    v = row.get("predicted_per_second")
    return v if isinstance(v, (int, float)) else None


def load_s(row: dict | None):
    """Prefer comparable `startup_s` (wall − decode); fall back to engine load."""
    if not row or row.get("error"):
        return None
    for key in ("startup_s", "load_s"):
        v = row.get(key)
        if isinstance(v, (int, float)):
            return v
    return None


def load_sd(row: dict | None):
    if not row or row.get("error"):
        return None
    for key in ("startup_stddev", "load_stddev"):
        v = row.get(key)
        if isinstance(v, (int, float)):
            return v
    return None


def cli_pin_trustworthy(pin: dict) -> bool:
    """Reject CLI pins known to have used llama-cli or mismatched templates."""
    llama = pin.get("llama") or {}
    if llama.get("error"):
        return False
    prefix = str(llama.get("prefix") or "")
    log = str(llama.get("log_tail") or "")
    if "not supported by llama-cli" in prefix or "not supported by llama-cli" in log:
        return False
    if "please use llama-completion" in prefix or "please use llama-completion" in log:
        return False
    # Stale 256-tok / 1-rep server-shaped CLI leftovers
    if pin.get("max_tokens") not in (None, 512) and pin.get("max_tokens", 512) < 512:
        return False
    return True


def fmt_load(x, sd=None) -> str:
    """Format load/startup seconds (keep ms precision for sub-second loads)."""
    if x is None:
        return "—"
    if isinstance(sd, (int, float)):
        return f"**{x:.3f}** ±{sd:.3f}"
    return f"**{x:.3f}**"


def gap_ratio(fp, lp) -> float | None:
    if fp and lp and fp > 0:
        return lp / fp
    return None


# Near-parity band: tok/s jitter of a few percent is not a meaningful win/loss.
# ~1.03–1.04× is treated as parity (⚪), not 🔴 llama.
PARITY_BAND = 0.05


def color_gap(g: float | None, *, lower_is_better: bool = False) -> str:
    """Gap cell for GitHub markdown (no inline CSS — GH strips it).

    Convention: gap < 1 → ferrox better. Uses emoji + bold text so the
    signal survives GitHub Flavored Markdown. Within ``PARITY_BAND`` of
    1.0 → ⚪ near-parity (not a loss).
    """
    del lower_is_better  # ratios are pre-normalized to "<1 = ferrox better"
    if g is None:
        return "—"
    if abs(g - 1.0) < PARITY_BAND:
        if abs(g - 1.0) < 0.015:
            return "⚪ **1.00×**"
        return f"⚪ **~{g:.2f}×**"
    if g < 1.0:
        return f"🟢 **~{g:.2f}×**"
    return f"🔴 **~{g:.2f}×**"


def gap_str(fp, lp) -> str:
    return color_gap(gap_ratio(fp, lp))


def winner_str(fp, lp) -> str:
    """Who is faster on predicted tok/s (same near-parity band as gap_str)."""
    g = gap_ratio(fp, lp)
    if g is None:
        return "—"
    if abs(g - 1.0) < PARITY_BAND:
        return "⚪ parity"
    if g < 1.0:
        return "🟢 **ferrox**"
    return "🔴 **llama**"


def status_cell(pin: dict | None, expect: str) -> str:
    if pin is None:
        return "no pin"
    st = pin.get("status", "ok")
    if st in ("missing", "refuse", "error", "weak"):
        return st
    if expect == "weak":
        return "weak"
    return "ok"


def main() -> None:
    suite = json.loads(SUITE_PATH.read_text())
    pins: dict[tuple[str, str], dict] = {}
    cli_pins: dict[tuple[str, str], dict] = {}
    orphan_pin_files: list[str] = []
    if PINS.is_dir():
        for path in PINS.glob("*.json"):
            data = load_json(path)
            if not data:
                orphan_pin_files.append(path.name)
                continue
            key = (data.get("id"), data.get("backend"))
            if key[0] and key[1]:
                if data.get("mode") == "cli":
                    cli_pins[key] = data
                else:
                    pins[key] = data
            else:
                orphan_pin_files.append(path.name)

    # Fail closed on unreadable / schema-invalid pin JSON. Suite entries may
    # lack pin files (shown as "no pin"); never emit markdown links to missing
    # files. Extra on-disk pins (e.g. future CUDA) are fine if they parse.
    if orphan_pin_files:
        raise SystemExit(
            "render_results.py validation failed — unreadable or schema-invalid "
            f"pins: {', '.join(orphan_pin_files)}"
        )

    metal8 = pins.get(("llama31_8b_q4km", "metal"))
    if metal8 and pred(metal8.get("ferrox")) and pred(metal8.get("llama")):
        fp = pred(metal8["ferrox"])
        lp = pred(metal8["llama"])
        north = (
            f"**8B Metal pin:** {fmt_num(fp)} vs llama {fmt_num(lp)} pred "
            f"({gap_str(fp, lp)}) — "
            f"[`pins/llama31_8b_q4km_metal.json`](receipts/pins/llama31_8b_q4km_metal.json).\n"
        )
    else:
        north = "**8B Metal pin:** not yet paired in `receipts/pins/`.\n"

    lines: list[str] = [
        "# Results vs llama.cpp\n",
        "\n",
        f"Host B = Apple M2 Pro (10 cores). Greedy chat, warm, then "
        f"`max_tokens={suite.get('max_tokens', 512)}` × {suite.get('reps', 3)} reps "
        f"(median ± stddev) unless noted. Prefer **predicted** tok/s.\n",
        "\n",
        "Suite: [`suite.json`](suite.json). Runner: [`run_suite.py`](run_suite.py). "
        "Pins: [`receipts/pins/`](receipts/pins/). "
        "**This file is generated** by [`render_results.py`](render_results.py) — "
        "do not hand-edit headlines.\n",
        "\n",
        "**Gap** = `llama_pred / ferrox_pred` (&lt;1 ferrox faster; &gt;1 ferrox slower). "
        "**Winner** = faster engine on predicted tok/s "
        f"(near-parity within ~{PARITY_BAND * 100:.0f}%).\n",
        "\n",
        "**North star:** ≥ llama.cpp same host/GGUF/backend.\n",
        north,
        "\n",
        "One pin per `(model_id, backend)`. Re-run overwrites the pin. "
        "Gap only when both engines succeed.\n",
        "\n",
        "**Gap colors (GitHub-safe):** 🟢 ferrox better; "
        f"⚪ near-parity (within ~{PARITY_BAND * 100:.0f}%); "
        "🔴 ferrox meaningfully slower.\n",
        "\n",
        "Keep off (regressions): legacy GQA NSG=4, sequential GREEDY argmax, "
        "float4 elem, early Multi-CB. `FERROX_METAL_FA_VEC=0` → ~25.5 pred.\n",
        "\n",
        "## Headlines\n",
        "\n",
        "| Model | Backend | ferrox pred (tok/s) | llama pred (tok/s) | Gap | Winner | Status | Pin |\n",
        "|---|---|---|---|---|---|---|---|\n",
    ]

    for entry in suite["models"]:
        mid = entry["id"]
        name = entry["name"]
        expect = entry.get("expect", "ok")
        for backend in entry["backends"]:
            pin = pins.get((mid, backend))
            fp = pred(pin.get("ferrox") if pin else None)
            lp = pred(pin.get("llama") if pin else None)
            fsd = stddev(pin.get("ferrox") if pin else None)
            lsd = stddev(pin.get("llama") if pin else None)
            st = status_cell(pin, expect)
            if st == "refuse":
                fcell = "refused"
            elif st == "missing":
                fcell = "—"
                lp = None
            elif fp is not None:
                fcell = fmt_num(fp, fsd)
            else:
                fcell = "—"
            lcell = fmt_num(lp, lsd) if lp is not None else "—"
            if backend == "cpu" and lp is not None:
                lcell = f"{lcell} (−ngl 0)"
            # Only link pins that exist on disk — never invent broken markdown links.
            pin_name = f"{mid}_{backend}.json"
            link = (
                f"[`{mid}_{backend}`](receipts/pins/{pin_name})"
                if pin and (PINS / pin_name).is_file()
                else "—"
            )
            star = "\\*" if expect == "weak" else ""
            lines.append(
                f"| {name}{star} | {backend} | {fcell} | {lcell} | "
                f"{gap_str(fp, lp)} | {winner_str(fp, lp)} | {st} | {link} |\n"
            )

    lines.append("\n")
    for e in suite["models"]:
        if e.get("expect") == "weak":
            lines.append(f"\\*weak — {e['name']}: {e.get('notes', 'see suite.json')}\n")
    miss_ids = sorted(
        {
            e["id"]
            for e in suite["models"]
            for b in e["backends"]
            if (pins.get((e["id"], b)) or {}).get("status") == "missing"
        }
    )
    if miss_ids:
        lines.append(
            f"Missing GGUF (no tok/s): {', '.join(f'`{i}`' for i in miss_ids)}. "
            "Place the files at their configured `models/` paths.\n"
        )

    if cli_pins:
        lines.append(
            "\n## CLI completion (`llama-completion` vs `ferrox run`)\n\n"
            "One-shot `-p … -n N --ignore-eos -c 4096` with the **same capitals "
            "prompt + chat template** as fair-chat server (ferrox wraps via GGUF "
            "template; llama `-cnv --jinja`). Fresh process per rep, interleaved "
            "(llama then ferrox each rep). Requires `llama-completion` (not "
            "`llama-cli`). Engines' own stderr timings; **pred** tok/s excludes "
            "model load. **startup** = wall − decode (comparable process overhead); "
            "falls back to engine-reported load if startup missing. "
            "**Startup gap** = `ferrox / llama` (&lt;1 ferrox better).\n"
            "Pins that used `llama-cli` or rejected options are omitted.\n\n"
            "| Model | Backend | ferrox pred | llama pred | Gap | "
            "ferrox startup (s) | llama startup (s) | Startup gap | Pin |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        for entry in suite["models"]:
            mid = entry["id"]
            for backend in entry["backends"]:
                pin = cli_pins.get((mid, backend))
                if not pin or pin.get("status") == "missing":
                    continue
                if not cli_pin_trustworthy(pin):
                    continue
                fp = pred(pin.get("ferrox"))
                lp = pred(pin.get("llama"))
                fsd = stddev(pin.get("ferrox"))
                lsd = stddev(pin.get("llama"))
                fl = load_s(pin.get("ferrox"))
                ll = load_s(pin.get("llama"))
                flsd = load_sd(pin.get("ferrox"))
                llsd = load_sd(pin.get("llama"))
                ferr = (pin.get("ferrox") or {}).get("error")
                fcell = "error" if ferr else fmt_num(fp, fsd)
                lcell = fmt_num(lp, lsd) if lp is not None else "—"
                load_gap = "—"
                if fl and ll and ll > 0:
                    load_gap = color_gap(fl / ll)
                lines.append(
                    f"| {entry['name']} | {backend} | {fcell} | {lcell} | "
                    f"{gap_str(fp, lp)} | {fmt_load(fl, flsd)} | {fmt_load(ll, llsd)} | "
                    f"{load_gap} | "
                    f"[`{mid}_{backend}_cli`](receipts/pins/{mid}_{backend}_cli.json) |\n"
                )

    lines.append("\n## Open\n\n")
    lines.append(
        "1. Metal fair-chat 8B is ahead (~0.92×); 3B ~parity (~0.97×). Keep "
        "watching `prompt_per_second` vs llama after FA-vec prefill.\n"
        "2. OLMoE Metal ~1.59× / CPU ~1.40× after concurrent encode + shared "
        "Q8 act — next: expert residency hoist (`docs/ROADMAP.md`).\n"
        "3. CUDA — re-measure on comparable CUDA hardware (no in-tree CUDA pin; "
        "skipped on darwin via `--fit-host`).\n"
        "4. Gemma-4-E2B: both ferrox and Homebrew llama refuse (`gemma4` arch / "
        "per-layer+shared-KV+SWA split); suite `expect=refuse`.\n"
        "5. CB multi-request tok/s receipt.\n"
        "6. DS4 / GLM / MLA MoE real-checkpoint e2e when feasible.\n"
        "7. Qwen2-MoE / Mixtral: missing GGUF or `--fit-host` RAM skip on Host B.\n"
        "8. Suite contention: re-pin outliers alone if full-suite medians disagree with CLI.\n"
    )
    lines.append("\nDo not invent numbers without a pin.\n")

    OUT.write_text("".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
