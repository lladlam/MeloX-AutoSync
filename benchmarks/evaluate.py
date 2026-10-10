#!/usr/bin/env python
"""Evaluate alignment output against reference boundaries.

Inputs
------
* ``--result``  : a MeloX-AutoSync result JSON (from ``autosync align``).
* ``--reference``: a reference JSON ``{"words": [{"text","start","end"}, ...]}``.

Matching
--------
We match predicted units to reference words **in order**, normalising text to
lowercase alphanumeric. We assume lyrics and reference are in the same order
(always true for this engine, since alignment preserves order). Mismatched
counts are reported rather than silently dropped.

Metrics (seconds)
-----------------
start MAE / median / p90, end MAE / median / p90, count matched, and (for
reporting) the fraction of lines that failed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# Keep latin alphanumerics and CJK/kana; drop punctuation and spaces.
_NORM = re.compile(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]+")


def norm(text: str) -> str:
    return _NORM.sub("", text.lower())


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def evaluate(result_path: Path, reference_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    pred_units = []
    n_lines = 0
    n_failed = 0
    for line in result.get("lines", []):
        n_lines += 1
        if line.get("source") == "failed":
            n_failed += 1
        pred_units.extend(line.get("units", []))

    ref_words = reference["words"]

    # Align predictions to references in order by normalised text.
    matched: list[tuple[dict, dict]] = []
    i = 0
    for ref in ref_words:
        rn = norm(ref["text"])
        while i < len(pred_units) and norm(pred_units[i]["text"]) != rn:
            i += 1
        if i < len(pred_units):
            matched.append((pred_units[i], ref))
            i += 1

    start_err = [abs(p["start"] - r["start"]) for p, r in matched]
    end_err = [abs(p["end"] - r["end"]) for p, r in matched]

    return {
        "n_lines": n_lines,
        "n_lines_failed": n_failed,
        "line_failure_rate": (n_failed / n_lines) if n_lines else 0.0,
        "n_reference_words": len(ref_words),
        "n_predicted_units": len(pred_units),
        "n_matched": len(matched),
        "start_mae": float(np.mean(start_err)) if start_err else float("nan"),
        "start_median": _percentile(start_err, 50),
        "start_p90": _percentile(start_err, 90),
        "start_max": float(np.max(start_err)) if start_err else float("nan"),
        "end_mae": float(np.mean(end_err)) if end_err else float("nan"),
        "end_median": _percentile(end_err, 50),
        "end_p90": _percentile(end_err, 90),
        "end_max": float(np.max(end_err)) if end_err else float("nan"),
        "engine": result.get("engine"),
        "model": (result.get("model") or {}).get("name"),
        "pairs": [
            {
                "text": r["text"],
                "ref_start": r["start"],
                "pred_start": p["start"],
                "start_err": round(abs(p["start"] - r["start"]), 4),
                "ref_end": r["end"],
                "pred_end": p["end"],
                "end_err": round(abs(p["end"] - r["end"]), 4),
            }
            for p, r in matched
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    report = evaluate(Path(args.result), Path(args.reference))
    print(f"engine={report['engine']} model={report['model']}")
    print(
        f"lines={report['n_lines']} failed={report['n_lines_failed']} "
        f"ref_words={report['n_reference_words']} pred_units={report['n_predicted_units']} "
        f"matched={report['n_matched']}"
    )
    for k in ("start_mae", "start_median", "start_p90", "start_max", "end_mae", "end_median", "end_p90", "end_max"):
        print(f"  {k:14s} = {report[k]:.4f} s")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
