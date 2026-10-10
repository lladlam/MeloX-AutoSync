#!/usr/bin/env python
"""Human-annotation template + scorer for real-song accuracy (V1.0 P4).

We do **not** invent accuracy numbers: without human-confirmed reference
boundaries there is no accuracy to report. This tool

1. ``template`` – picks a representative sample of lyric units from a result
   JSON (first/last lines, longest and shortest units, repeated lines, plus a
   seeded random sample) and writes a CSV whose ``ref_start_s``/``ref_end_s``
   columns are **empty** for a human to fill by listening;
2. ``score`` – reads a filled CSV and reports start/end MAE, P90 and the
   fraction of errors above 300/500/1000 ms, overall and per language.

Units without human reference times are skipped and counted as "unannotated",
never silently treated as correct.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

COLS = [
    "song", "lang", "line_index", "line_text", "unit_index", "unit_text",
    "pred_start_s", "pred_end_s", "ref_start_s", "ref_end_s", "notes",
]


def build_rows(result: dict, song: str, max_units: int, seed: int = 7) -> list[dict]:
    lang = result.get("language", "unknown")
    lines = [ln for ln in result.get("lines", []) if ln.get("units") and ln.get("kind", "singing") == "singing"]
    rows: list[dict] = []

    def add(ln, ui, note=""):
        u = ln["units"][ui]
        rows.append(
            {
                "song": song, "lang": lang, "line_index": ln["index"], "line_text": ln["text"],
                "unit_index": ui, "unit_text": u["text"],
                "pred_start_s": round(u["start"], 3), "pred_end_s": round(u["end"], 3),
                "ref_start_s": "", "ref_end_s": "", "notes": note,
            }
        )

    if not lines:
        return rows
    for ln in lines[:2] + lines[-2:]:
        for ui in range(min(3, len(ln["units"]))):
            add(ln, ui, "edge line")
    flat = [(ln, ui) for ln in lines for ui in range(len(ln["units"]))]
    durs = [(ln["units"][ui]["end"] - ln["units"][ui]["start"], ln, ui) for ln, ui in flat]
    durs.sort(key=lambda x: -x[0])
    for _, ln, ui in durs[:5]:
        add(ln, ui, "longest")
    for _, ln, ui in durs[-5:]:
        add(ln, ui, "shortest")
    counts = defaultdict(int)
    for ln in lines:
        counts[ln["text"]] += 1
    for ln in lines:
        if counts[ln["text"]] > 1:
            add(ln, 0, "repeated lyric")
    rng = random.Random(seed)
    pool = flat[:]
    rng.shuffle(pool)
    for ln, ui in pool:
        if len(rows) >= max_units:
            break
        add(ln, ui, "random")
    # de-duplicate, keep order
    seen, out = set(), []
    for r in rows:
        key = (r["line_index"], r["unit_index"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out[:max_units]


def write_template(results: dict[str, dict], out: Path, max_units: int, seed: int) -> Path:
    rows: list[dict] = []
    for song, result in results.items():
        rows += build_rows(result, song, max_units, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return out


def score_csv(path: Path) -> dict:
    per_lang_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_song: dict[str, list[tuple[float, float]]] = defaultdict(list)
    unannotated = invalid = 0
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("ref_start_s", "").strip() or not row.get("ref_end_s", "").strip():
                unannotated += 1
                continue
            if not row.get("pred_start_s", "").strip():
                invalid += 1
                continue
            pred_s, pred_e = float(row["pred_start_s"]), float(row["pred_end_s"])
            ref_s, ref_e = float(row["ref_start_s"]), float(row["ref_end_s"])
            pair = (abs(pred_s - ref_s), abs(pred_e - ref_e))
            per_lang_pairs[row.get("lang", "unknown")].append(pair)
            per_song[row.get("song", "")].append(pair)

    def stats(pairs: list[tuple[float, float]]) -> dict:
        if not pairs:
            return {"n": 0}
        se = [p[0] for p in pairs]
        ee = [p[1] for p in pairs]
        return {
            "n": len(pairs),
            "start_mae": float(np.mean(se)),
            "start_p90": float(np.percentile(se, 90)),
            "end_mae": float(np.mean(ee)),
            "end_p90": float(np.percentile(ee, 90)),
            "start_gt_0.3": float(np.mean([x > 0.3 for x in se])),
            "start_gt_0.5": float(np.mean([x > 0.5 for x in se])),
            "start_gt_1.0": float(np.mean([x > 1.0 for x in se])),
        }

    return {
        "unannotated_rows": unannotated,
        "rows_without_predictions": invalid,
        "by_language": {k: stats(v) for k, v in per_lang_pairs.items()},
        "by_song": {k: stats(v) for k, v in per_song.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--results", nargs="+", required=True, help="result JSON files")
    t.add_argument("--out", default="experiments/annotation/annotation_template.csv")
    t.add_argument("--max-units", type=int, default=30, help="per song")
    t.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score")
    s.add_argument("--csv", required=True)
    s.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.cmd == "template":
        results = {}
        for p in args.results:
            path = Path(p)
            data = json.loads(path.read_text(encoding="utf-8"))
            results[path.stem] = data
        out = write_template(results, Path(args.out), args.max_units, args.seed)
        with out.open(encoding="utf-8") as fh:
            n = sum(1 for _ in fh) - 1
        print(f"wrote {out} with {n} rows to annotate")
        return 0

    report = score_csv(Path(args.csv))
    print(f"unannotated rows: {report['unannotated_rows']} (never counted as correct)")
    for lang, st in report["by_language"].items():
        if st.get("n"):
            print(f"  {lang}: n={st['n']} startMAE={st['start_mae']:.3f} startP90={st['start_p90']:.3f} "
                  f"endMAE={st['end_mae']:.3f} >0.3s={st['start_gt_0.3']:.2f} >1.0s={st['start_gt_1.0']:.2f}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
