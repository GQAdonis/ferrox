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


def gap_str(fp, lp) -> str:
    if fp and lp and fp > 0:
        g = lp / fp
        if abs(g - 1.0) < 0.015:
            return "**1.00×**"
        return f"~{g:.2f}×"
    return "—"


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
    if PINS.is_dir():
        for path in PINS.glob("*.json"):
            data = load_json(path)
            if not data:
                continue
            key = (data.get("id"), data.get("backend"))
            if key[0] and key[1]:
                if data.get("mode") == "cli":
                    cli_pins[key] = data
                else:
                    pins[key] = data

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
        "**North star:** ≥ llama.cpp same host/GGUF/backend.\n",
        north,
        "\n",
        "One pin per `(model_id, backend)`. Re-run overwrites the pin. "
        "Gap only when both engines succeed.\n",
        "\n",
        "Keep off (regressions): legacy GQA NSG=4, sequential GREEDY argmax, "
        "float4 elem, early Multi-CB. `FERROX_METAL_FA_VEC=0` → ~25.5 pred.\n",
        "\n",
        "## Headlines\n",
        "\n",
        "| Model | Backend | ferrox pred | llama pred | Gap | Status | Pin |\n",
        "|---|---|---|---|---|---|---|\n",
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
            link = (
                f"[`{mid}_{backend}`](receipts/pins/{mid}_{backend}.json)"
                if pin
                else "—"
            )
            star = "\\*" if expect == "weak" else ""
            lines.append(
                f"| {name}{star} | {backend} | {fcell} | {lcell} | "
                f"{gap_str(fp, lp)} | {st} | {link} |\n"
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
            "\n## CLI completion (llama-cli vs `ferrox run`)\n\n"
            "One-shot `-p … -n N --no-cnv --ignore-eos`, fresh process per rep, "
            "strictly sequential (llama exits before ferrox starts). Engines' own "
            "stderr timings; decode excludes model load.\n\n"
            "| Model | Backend | ferrox pred | llama pred | Gap | Pin |\n"
            "|---|---|---|---|---|---|\n"
        )
        for entry in suite["models"]:
            mid = entry["id"]
            for backend in entry["backends"]:
                pin = cli_pins.get((mid, backend))
                if not pin or pin.get("status") == "missing":
                    continue
                fp = pred(pin.get("ferrox"))
                lp = pred(pin.get("llama"))
                fsd = stddev(pin.get("ferrox"))
                lsd = stddev(pin.get("llama"))
                ferr = (pin.get("ferrox") or {}).get("error")
                fcell = "error" if ferr else fmt_num(fp, fsd)
                lcell = fmt_num(lp, lsd) if lp is not None else "—"
                lines.append(
                    f"| {entry['name']} | {backend} | {fcell} | {lcell} | "
                    f"{gap_str(fp, lp)} | "
                    f"[`{mid}_{backend}_cli`](receipts/pins/{mid}_{backend}_cli.json) |\n"
                )

    lines.append("\n## Open\n\n")
    lines.append(
        "1. Metal prefill ≪ llama on large models; FA-vec covers d=128/64 (other dims → legacy GQA).\n"
        "2. CUDA — re-measure on comparable CUDA hardware.\n"
        "3. Gemma-2 arch support (pin refuses).\n"
        "4. CB multi-request tok/s receipt.\n"
        "5. DS4 / GLM real e2e when feasible.\n"
    )
    lines.append("\nDo not invent numbers without a pin.\n")

    OUT.write_text("".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
