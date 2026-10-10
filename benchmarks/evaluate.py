#!/usr/bin/env python
"""Evaluate alignment output against reference boundaries.

Inputs
------
* ``--result``  : a MeloX-AutoSync result JSON (from ``autosync align``).
* ``--reference``: a reference JSON ``{"words": [{"text","start","end"}, ...]}``.

Reporting philosophy
--------------------
We do **not** compute error statistics only over the matches that happened to
succeed. The report always includes:

* coverage / miss-rate (how many reference words were actually matched),
* extra (spurious) predicted units,
* boundary error statistics (MAE / median / p90 / max) for start and end,
* the fraction of matched boundaries worse than 0.1 s / 0.3 s / 0.5 s,
* the concrete **failure samples** (missed references, extra units).

See ``benchmarks/metrics.py`` for the matching/grouping logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import metrics


def evaluate(result_path: Path, reference_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    pred_units: list[dict] = []
    n_lines = 0
    n_failed = 0
    for line in result.get("lines", []):
        n_lines += 1
        if line.get("source") == "failed":
            n_failed += 1
        pred_units.extend(line.get("units", []))

    report = metrics.score(pred_units, reference["words"])
    report.update(
        {
            "n_lines": n_lines,
            "n_lines_failed": n_failed,
            "line_failure_rate": (n_failed / n_lines) if n_lines else 0.0,
            "engine": result.get("engine"),
            "model": (result.get("model") or {}).get("name"),
        }
    )
    return report


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
        f"matched={report['n_matched_words']} coverage={report['coverage']:.3f} "
        f"miss_rate={report['miss_rate']:.3f} extra={report['n_extra_units']}"
    )
    for k in (
        "start_mae",
        "start_median",
        "start_p90",
        "start_max",
        "end_mae",
        "end_median",
        "end_p90",
        "end_max",
        "duration_mae",
        "start_err_rate_gt_0p1",
        "start_err_rate_gt_0p3",
        "start_err_rate_gt_0p5",
        "end_err_rate_gt_0p1",
        "end_err_rate_gt_0p3",
        "end_err_rate_gt_0p5",
    ):
        print(f"  {k:26s} = {report[k]:.4f}")
    if report["missed"]:
        print(f"  MISSED ({len(report['missed'])}): {[m['text'] for m in report['missed'][:10]]}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
