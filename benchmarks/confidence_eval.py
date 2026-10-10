#!/usr/bin/env python
"""Confidence calibration and selective-alignment evaluation (V0.7 §2, §5).

Pipeline separation (strict):

* **inference** – HuBERT V0.5 forced alignment + :func:`extract_features`, using
  only model outputs and the lyric text;
* **scoring** – reference character times (GTSinger) are used afterwards to label
  *error events* and to evaluate detection.

Group-aware validation: thresholds are chosen on calibration folds only, using
**leave-one-singer-out** and **leave-one-song-out**; the held-out fold never
participates in threshold selection. With 30 lines / 2 singers this is
**exploratory** and is reported as such.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import metrics
import numpy as np
from singing_eval import discover, line_text, load_reference

from autosync.alignment.ctc import forced_align
from autosync.audio.io import load_audio
from autosync.lyrics.ja_pronunciation import locate_inner_offset
from autosync.models.phoneme_ctc_aligner import PhonemeCTCAligner, map_spans_to_char_units
from autosync.quality import extract_features

# higher = more reliable for these features
CANDIDATE_FEATURES = [
    "score_margin_per_token",
    "score_margin_per_frame",
    "mean_token_logp",
    "mean_peak_advantage",
    "free_decode_agreement",
    "blank_dominant_ratio",  # lower is better -> negated before use
    "max_intra_char_span_s",  # lower is better
    "char_duration_ratio",  # lower is better
]
LOWER_IS_BETTER = {"blank_dominant_ratio", "max_intra_char_span_s", "char_duration_ratio"}


def _char_of_kept(readings, phonemes, keep: list[int]) -> list[int]:
    inner = [p for r in readings for p in r.phonemes]
    offset = locate_inner_offset(phonemes, inner)
    char_of = [-1] * len(phonemes)
    cursor = offset
    for ci, r in enumerate(readings):
        for _ in range(len(r.phonemes)):
            if cursor < len(char_of):
                char_of[cursor] = ci
            cursor += 1
    return [char_of[k] if k < len(char_of) else -1 for k in keep]


def _orient(features: dict, name: str) -> float:
    v = features.get(name, float("nan"))
    return -v if name in LOWER_IS_BETTER else v


def _auc(scores: list[float], labels: list[int]) -> float:
    # probability that a random positive (severe) has a lower reliability score
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 for a in pos for b in neg if a < b) + 0.5 * sum(1.0 for a in pos for b in neg if a == b)
    return wins / (len(pos) * len(neg))


def _best_threshold(records: list[dict], name: str, labels: list[int]) -> float | None:
    scores = [_orient(r["features"], name) for r in records]
    finite = [s for s in scores if np.isfinite(s)]
    if not finite:
        return None
    best_t, best_f1 = None, -1.0
    for t in np.unique(np.percentile(finite, np.linspace(0, 100, 41))):
        tp = fp = fn = tn = 0
        for s, y in zip(scores, labels):
            reject = s < t  # low reliability -> flag as severe
            if y == 1 and reject:
                tp += 1
            elif y == 1 and not reject:
                fn += 1
            elif y == 0 and reject:
                fp += 1
            else:
                tn += 1
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="experiments/singing/gtsinger")
    ap.add_argument("--severe-threshold", type=float, default=1.0, help="line is 'severe' if max start err exceeds this")
    ap.add_argument("--out", default="experiments/reports/confidence_eval.json")
    args = ap.parse_args()

    segs = [s for s in discover(Path(args.root)) if s["language"] == "ja"]
    aligner = PhonemeCTCAligner()
    aligner.load()
    blank = aligner._blank_id()
    fs = aligner.frame_seconds
    mapper = aligner._mapper

    records = []
    for seg in segs:
        audio, sr = load_audio(seg["wav"], 16000)
        ref_words = load_reference(seg["json"])
        text = line_text("ja", ref_words)
        readings, phonemes = mapper.map(text)
        ids, keep = aligner._phoneme_ids(phonemes)
        if not ids:
            continue
        logp = aligner._log_probs(audio, sr)
        al = forced_align(logp, ids, blank=blank)
        char_of = _char_of_kept(readings, phonemes, keep)
        feats = extract_features(logp, ids, blank, al, char_of, fs)

        spans_by_phoneme = {k: s for k, s in zip(keep, al.spans) if s is not None}
        units = map_spans_to_char_units(readings, phonemes, spans_by_phoneme, fs)
        pred = [{"text": u.text, "start": u.start, "end": u.end, "uncertain": u.uncertain} for u in units]
        sc = metrics.score(pred, ref_words)
        pairs = sc["pairs"]
        se = [p["start_err"] for p in pairs]
        ee = [p["end_err"] for p in pairs]
        records.append(
            {
                "id": seg["id"],
                "singer": seg["singer"],
                "song": seg["song"],
                "technique": seg["technique"],
                "coverage": sc["coverage"],
                "n": len(pairs),
                "start_mae": float(np.mean(se)) if se else float("nan"),
                "start_p90": float(np.percentile(se, 90)) if se else float("nan"),
                "start_max": float(np.max(se)) if se else float("nan"),
                "end_max": float(np.max(ee)) if ee else float("nan"),
                "features": feats.to_dict(),
                "pairs": pairs,
            }
        )

    labels = [1 if r["start_max"] > args.severe_threshold else 0 for r in records]
    severe05 = sum(1 for r in records if r["start_max"] > 0.5)
    severe10 = sum(1 for r in records if r["start_max"] > 1.0)
    print(f"segments: {len(records)}  severe(max start>0.5s): {severe05}  severe(>1.0s): {severe10}  "
          f"primary threshold={args.severe_threshold}s -> {sum(labels)}")

    # ---- feature screening (correlation with max start error) --------------
    print("\nfeature screening (Spearman vs max start error):")
    screening = {}
    for name in CANDIDATE_FEATURES:
        vals = [_orient(r["features"], name) for r in records]
        errs = [r["start_max"] for r in records]
        finite = [(v, e) for v, e in zip(vals, errs) if np.isfinite(v) and np.isfinite(e)]
        if len(finite) < 5:
            continue
        va, ea = zip(*finite)
        # rank correlation
        rv = np.argsort(np.argsort(va))
        re = np.argsort(np.argsort(ea))
        rho = float(np.corrcoef(rv, re)[0, 1])
        auc = _auc(vals, labels)
        screening[name] = {"spearman": rho, "auc_detect_severe": auc}
        print(f"  {name:26s} spearman={rho:+.3f}  auc={auc:.3f}")

    # primary feature = best AUC among the candidates (selected on all data for
    # *reporting* only; the CV below re-selects on calibration folds).
    primary = max(
        (n for n in screening),
        key=lambda n: screening[n]["auc_detect_severe"] if np.isfinite(screening[n]["auc_detect_severe"]) else -1,
        default="score_margin_per_token",
    )
    print(f"\nprimary feature: {primary}")

    # ---- group-aware calibration ------------------------------------------
    groups = {"leave_one_singer_out": "singer", "leave_one_song_out": "song"}
    cv = {}
    for cv_name, key in groups.items():
        levels = sorted({r[key] for r in records})
        folds = []
        for lv in levels:
            test = [r for r in records if r[key] == lv]
            train = [r for r in records if r[key] != lv]
            if not test or not train:
                continue
            tr_labels = [1 if r["start_max"] > args.severe_threshold else 0 for r in train]
            best = None
            for name in {primary, "score_margin_per_token", "free_decode_agreement"}:
                t = _best_threshold(train, name, tr_labels)
                if t is None:
                    continue
                scores = [_orient(r["features"], name) for r in train]
                # rough F1 on train to choose the feature
                tp = sum(1 for s, y in zip(scores, tr_labels) if y == 1 and s < t)
                fp = sum(1 for s, y in zip(scores, tr_labels) if y == 0 and s < t)
                fn = sum(1 for s, y in zip(scores, tr_labels) if y == 1 and s >= t)
                f1 = _prf(tp, fp, fn)[2]
                if best is None or f1 > best[2]:
                    best = (name, t, f1)
            if best is None:
                continue
            name, t, _ = best
            tp = fp = fn = tn = 0
            for r in test:
                y = 1 if r["start_max"] > args.severe_threshold else 0
                s = _orient(r["features"], name)
                reject = (not np.isfinite(s)) or s < t
                if y == 1 and reject:
                    tp += 1
                elif y == 1:
                    fn += 1
                elif y == 0 and reject:
                    fp += 1
                else:
                    tn += 1
            folds.append({"held_out": lv, "feature": name, "threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                          "n_test": len(test), "n_severe": sum(1 for r in test if r["start_max"] > args.severe_threshold)})
        tp = sum(f["tp"] for f in folds)
        fp = sum(f["fp"] for f in folds)
        fn = sum(f["fn"] for f in folds)
        p, r_, f1 = _prf(tp, fp, fn)
        cv[cv_name] = {
            "folds": folds,
            "pooled": {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r_, "f1": f1},
        }
        print(f"\n{cv_name}: pooled precision={p:.2f} recall={r_:.2f} f1={f1:.2f} "
              f"(severe {sum(f['n_severe'] for f in folds)}/{sum(f['n_test'] for f in folds)})")
        for f in folds:
            print(f"  hold-out {f['held_out']:14s} feature={f['feature']:20s} thr={f['threshold']:+.2f} "
                  f"TP={f['tp']} FP={f['fp']} FN={f['fn']} TN={f['tn']} (severe {f['n_severe']}/{f['n_test']})")

    # ---- risk-coverage curve (accept most-reliable fraction by `primary`) --
    name = primary
    order = sorted(records, key=lambda r: _orient(r["features"], name), reverse=True)  # most reliable first
    curve = []
    for k in range(len(order) + 1):
        acc = order[:k]
        pairs = [p for r in acc for p in r["pairs"]]
        se = [p["start_err"] for p in pairs]
        curve.append(
            {
                "coverage_lines": (k / len(order)) if order else 0.0,
                "coverage_units": (sum(r["n"] for r in acc) / sum(r["n"] for r in order)) if order else 0.0,
                "start_mae": float(np.mean(se)) if se else float("nan"),
                "start_p90": float(np.percentile(se, 90)) if se else float("nan"),
                "severe_line_rate": float(np.mean([r["start_max"] > args.severe_threshold for r in acc])) if acc else float("nan"),
                "severe_unit_rate": float(np.mean([x > 0.5 for x in se])) if se else float("nan"),
            }
        )
    print(f"\nrisk-coverage (accept most-reliable proportion by {name}):")
    for c in curve:
        if abs(c["coverage_lines"] * 4 - round(c["coverage_lines"] * 4)) < 1e-9:
            print(f"  cov_lines={c['coverage_lines']:.2f} cov_units={c['coverage_units']:.2f} "
                  f"sMAE={c['start_mae']:.3f} sP90={c['start_p90']:.3f} "
                  f"sev_line={c['severe_line_rate']:.2f} sev_unit={c['severe_unit_rate']:.2f}")

    by_singer = {}
    for r in records:
        by_singer.setdefault(r["singer"], []).append(r)
    singer_summary = {
        s: {"n": len(v), "start_mae": float(np.mean([x["start_mae"] for x in v])),
            "start_max_mean": float(np.mean([x["start_max"] for x in v])),
            "severe": int(sum(x["start_max"] > args.severe_threshold for x in v))}
        for s, v in by_singer.items()
    }

    payload = {
        "generated": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "note": "exploratory: 30 lines / 2 singers; thresholds from grouped CV only",
        "severe_threshold_s": args.severe_threshold,
        "n_severe": int(sum(labels)),
        "screening": screening,
        "cv": cv,
        "risk_coverage": curve,
        "by_singer": singer_summary,
        "records": records,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
