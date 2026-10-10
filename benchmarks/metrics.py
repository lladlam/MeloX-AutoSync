"""Shared scoring for lyric alignment.

Key principles
--------------
* **Never evaluate only successful matches.** Coverage, miss-rate and extra
  (spurious) units are always reported alongside error statistics.
* Predictions at a finer granularity than the reference (e.g. Chinese
  character units vs multi-character reference words) are grouped onto the
  reference by walking both normalised character streams **in order**.
* Time inside a multi-character predicted token is split uniformly, and every
  such atom is marked ``uncertain`` so downstream analysis can exclude or
  discount it.

The reference is ``{"words": [{"text", "start", "end"}, ...]}`` in seconds.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# Keep latin alphanumerics and CJK/kana; drop punctuation and spaces.
_NORM = re.compile(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]+")

ERROR_THRESHOLDS = (0.1, 0.3, 0.5)


def norm(text: str) -> str:
    return _NORM.sub("", text.lower())


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def char_timeline(pred_units: list[dict]) -> list[dict]:
    """Flatten predicted units into a per-character timeline.

    A unit covering ``k`` characters is split into ``k`` equal time slices;
    each slice is flagged ``uncertain`` (the internal boundary is not
    model-derived).
    """
    atoms: list[dict] = []
    for ui, unit in enumerate(pred_units):
        chars = list(norm(unit.get("text", "")))
        if not chars:
            continue
        k = len(chars)
        span = float(unit["end"]) - float(unit["start"])
        for ci, ch in enumerate(chars):
            atoms.append(
                {
                    "ch": ch,
                    "start": float(unit["start"]) + span * ci / k,
                    "end": float(unit["start"]) + span * (ci + 1) / k,
                    "unit": ui,
                    "uncertain": bool(unit.get("uncertain", False)) or k > 1,
                }
            )
    return atoms


def score(pred_units: list[dict], ref_words: list[dict]) -> dict[str, Any]:
    """Score predicted units against reference words with full coverage stats."""
    atoms = char_timeline(pred_units)
    ref_chars: list[tuple[str, int]] = []
    for wi, w in enumerate(ref_words):
        for ch in norm(w.get("text", "")):
            ref_chars.append((ch, wi))

    # Longest-common-subsequence alignment between the predicted character
    # stream and the reference character stream. LCS (rather than a greedy
    # scan) is required so that a *missing* reference character does not cause
    # the following predicted character to be dropped as an "extra".
    A, R = len(atoms), len(ref_chars)
    dp = [[0] * (R + 1) for _ in range(A + 1)]
    for i in range(A - 1, -1, -1):
        ai_ch = atoms[i]["ch"]
        row, nxt = dp[i], dp[i + 1]
        for j in range(R - 1, -1, -1):
            if ai_ch == ref_chars[j][0]:
                row[j] = 1 + nxt[j + 1]
            else:
                row[j] = max(nxt[j], row[j + 1])

    matched_atoms: dict[int, list[dict]] = {}
    used_units: set[int] = set()
    i = j = 0
    while i < A and j < R:
        if atoms[i]["ch"] == ref_chars[j][0]:
            matched_atoms.setdefault(ref_chars[j][1], []).append(atoms[i])
            used_units.add(atoms[i]["unit"])
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1

    pairs: list[dict] = []
    missed: list[dict] = []
    for wi, w in enumerate(ref_words):
        group = matched_atoms.get(wi)
        if not group:
            missed.append({"text": w.get("text"), "start": w["start"], "end": w["end"]})
            continue
        p_start = group[0]["start"]
        p_end = group[-1]["end"]
        pairs.append(
            {
                "text": w.get("text"),
                "ref_start": float(w["start"]),
                "pred_start": p_start,
                "start_err": abs(p_start - float(w["start"])),
                "ref_end": float(w["end"]),
                "pred_end": p_end,
                "end_err": abs(p_end - float(w["end"])),
                "uncertain": any(a["uncertain"] for a in group),
            }
        )

    extra = [
        {"text": u.get("text"), "start": u["start"], "end": u["end"]}
        for ui, u in enumerate(pred_units)
        if ui not in used_units
    ]

    start_err = [p["start_err"] for p in pairs]
    end_err = [p["end_err"] for p in pairs]
    duration_err = [abs((p["pred_end"] - p["pred_start"]) - (p["ref_end"] - p["ref_start"])) for p in pairs]

    n_ref = len(ref_words)
    n_matched = len(pairs)
    report: dict[str, Any] = {
        "n_reference_words": n_ref,
        "n_predicted_units": len(pred_units),
        "n_matched_words": n_matched,
        "coverage": (n_matched / n_ref) if n_ref else float("nan"),
        "miss_rate": ((n_ref - n_matched) / n_ref) if n_ref else float("nan"),
        "n_extra_units": len(extra),
        "extra_rate": (len(extra) / len(pred_units)) if pred_units else float("nan"),
        "start_mae": _mean(start_err),
        "start_median": _pct(start_err, 50),
        "start_p90": _pct(start_err, 90),
        "start_max": (max(start_err) if start_err else float("nan")),
        "end_mae": _mean(end_err),
        "end_median": _pct(end_err, 50),
        "end_p90": _pct(end_err, 90),
        "end_max": (max(end_err) if end_err else float("nan")),
        "duration_mae": _mean(duration_err),
        "uncertain_matched": sum(1 for p in pairs if p["uncertain"]),
        "pairs": pairs,
        "missed": missed,
        "extra": extra,
    }
    for thr in ERROR_THRESHOLDS:
        key = str(thr).replace(".", "p")
        report[f"start_err_rate_gt_{key}"] = (
            sum(1 for e in start_err if e > thr) / len(start_err) if start_err else float("nan")
        )
        report[f"end_err_rate_gt_{key}"] = (
            sum(1 for e in end_err if e > thr) / len(end_err) if end_err else float("nan")
        )
    return report


def merge_reports(reports: list[dict], keys: list[str] | None = None) -> dict[str, Any]:
    """Aggregate several per-segment reports into a micro-average."""
    if not reports:
        return {}
    keys = keys or [
        "coverage",
        "miss_rate",
        "start_mae",
        "start_median",
        "start_p90",
        "end_mae",
        "end_median",
        "end_p90",
        "duration_mae",
        "start_err_rate_gt_0p1",
        "start_err_rate_gt_0p3",
        "start_err_rate_gt_0p5",
        "end_err_rate_gt_0p1",
        "end_err_rate_gt_0p3",
        "end_err_rate_gt_0p5",
    ]
    out: dict[str, Any] = {
        "n_segments": len(reports),
        "n_reference_words": sum(r["n_reference_words"] for r in reports),
        "n_matched_words": sum(r["n_matched_words"] for r in reports),
        "n_extra_units": sum(r["n_extra_units"] for r in reports),
        "uncertain_matched": sum(r.get("uncertain_matched", 0) for r in reports),
    }
    # Pool raw errors for micro-averaged MAE/median/p90.
    all_start = [p["start_err"] for r in reports for p in r["pairs"]]
    all_end = [p["end_err"] for r in reports for p in r["pairs"]]
    all_dur = [abs((p["pred_end"] - p["pred_start"]) - (p["ref_end"] - p["ref_start"])) for r in reports for p in r["pairs"]]
    n_ref = out["n_reference_words"]
    n_matched = out["n_matched_words"]
    out["coverage"] = (n_matched / n_ref) if n_ref else float("nan")
    out["miss_rate"] = ((n_ref - n_matched) / n_ref) if n_ref else float("nan")
    out["start_mae"] = _mean(all_start)
    out["start_median"] = _pct(all_start, 50)
    out["start_p90"] = _pct(all_start, 90)
    out["end_mae"] = _mean(all_end)
    out["end_median"] = _pct(all_end, 50)
    out["end_p90"] = _pct(all_end, 90)
    out["duration_mae"] = _mean(all_dur)
    for thr in ERROR_THRESHOLDS:
        key = str(thr).replace(".", "p")
        out[f"start_err_rate_gt_{key}"] = (
            sum(1 for e in all_start if e > thr) / len(all_start) if all_start else float("nan")
        )
        out[f"end_err_rate_gt_{key}"] = (
            sum(1 for e in all_end if e > thr) / len(all_end) if all_end else float("nan")
        )
    return out
