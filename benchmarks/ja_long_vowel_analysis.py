#!/usr/bin/env python
"""Japanese long-vowel diagnosis and cross-check (V0.5 §2).

Runs pyshiro (v2) and the corrected HuBERT phoneme-CTC on the Japanese
GTSinger set under the same ``lrc_only`` protocol, and reports:

* overall metrics for each engine and for an **ensemble** (per-character mean of
  the two independent model predictions — a rule fixed in advance, never chosen
  using ground truth);
* **long-vowel vs normal** reference units separately (long = reference duration
  >= 0.5 s);
* a raw per-character dump (original char, kana, phonemes, reference interval,
  both predictions) for the problem song(s).

Ground truth is used **only** for scoring and for the offline long/normal split,
never to pick a model or adjust a boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import metrics
import numpy as np
from singing_eval import discover, line_text, load_reference

from autosync.audio.io import load_audio
from autosync.lyrics.ja_pronunciation import JapanesePronunciationMapper
from autosync.models.phoneme_ctc_aligner import PhonemeCTCAligner
from autosync.models.pyshiro_aligner import PySHIROAligner

LONG_SECONDS = 0.5


def _units(aligner, audio, sr, text):
    out = aligner.align_text(audio, sr, text)
    return [
        {"text": u.text, "start": u.start, "end": u.end, "uncertain": getattr(u, "uncertain", False)}
        for u in out
    ]


def _ensemble(a: list[dict], b: list[dict]) -> list[dict]:
    """Per-character mean of two predictions (only where texts agree in order)."""
    out = []
    for i in range(min(len(a), len(b))):
        if metrics.norm(a[i]["text"]) != metrics.norm(b[i]["text"]):
            # texts disagree -> keep the first engine's unit untouched
            out.append(a[i])
            continue
        out.append(
            {
                "text": a[i]["text"],
                "start": (a[i]["start"] + b[i]["start"]) / 2,
                "end": (a[i]["end"] + b[i]["end"]) / 2,
                "uncertain": a[i].get("uncertain", False) and b[i].get("uncertain", False),
            }
        )
    # keep any tail from the longer prediction
    longer = a if len(a) >= len(b) else b
    out.extend(longer[min(len(a), len(b)) :])
    return out


def _split_metrics(reports: list[dict]) -> dict:
    pairs = [p for r in reports for p in r["pairs"]]
    long_pairs = [p for p in pairs if p["ref_end"] - p["ref_start"] >= LONG_SECONDS]
    norm_pairs = [p for p in pairs if p["ref_end"] - p["ref_start"] < LONG_SECONDS]

    def stat(ps):
        se = [p["start_err"] for p in ps]
        ee = [p["end_err"] for p in ps]
        return {
            "n": len(ps),
            "start_median": float(np.median(se)) if se else float("nan"),
            "start_p90": float(np.percentile(se, 90)) if se else float("nan"),
            "start_mae": float(np.mean(se)) if se else float("nan"),
            "end_mae": float(np.mean(ee)) if ee else float("nan"),
        }

    return {"long": stat(long_pairs), "normal": stat(norm_pairs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="experiments/singing/gtsinger")
    ap.add_argument("--dump-song", default="さんぽ")
    ap.add_argument("--max-dump", type=int, default=1)
    ap.add_argument("--out", default="experiments/reports/ja_long_vowel.json")
    args = ap.parse_args()

    segs = [s for s in discover(Path(args.root)) if s["language"] == "ja"]
    shiro = PySHIROAligner("v2")
    shiro.load()
    hubert = PhonemeCTCAligner()
    hubert.load()
    mapper = JapanesePronunciationMapper()

    engines = {"pyshiro_v2": shiro, "phoneme_ctc": hubert}
    reports: dict[str, list[dict]] = {k: [] for k in engines}
    reports["ensemble"] = []
    dumps: list[dict] = []

    for seg in segs:
        audio, sr = load_audio(seg["wav"], 16000)
        ref_words = load_reference(seg["json"])
        text = line_text("ja", ref_words)
        preds = {name: _units(al, audio, sr, text) for name, al in engines.items()}
        for name, pred in preds.items():
            reports[name].append(metrics.score(pred, ref_words))
        reports["ensemble"].append(metrics.score(_ensemble(preds["pyshiro_v2"], preds["phoneme_ctc"]), ref_words))

        if args.dump_song in seg["song"] and len(dumps) < args.max_dump:
            readings, _full = mapper.map(text)
            dumps.append(
                {
                    "id": seg["id"],
                    "text": text,
                    "chars": [
                        {
                            "char": r.char if r else "",
                            "kana": r.kana,
                            "phonemes": r.phonemes,
                            "ref_start": ref_words[i]["start"] if i < len(ref_words) else None,
                            "ref_end": ref_words[i]["end"] if i < len(ref_words) else None,
                            "pyshiro_start": preds["pyshiro_v2"][i]["start"] if i < len(preds["pyshiro_v2"]) else None,
                            "pyshiro_end": preds["pyshiro_v2"][i]["end"] if i < len(preds["pyshiro_v2"]) else None,
                            "hubert_start": preds["phoneme_ctc"][i]["start"] if i < len(preds["phoneme_ctc"]) else None,
                            "hubert_end": preds["phoneme_ctc"][i]["end"] if i < len(preds["phoneme_ctc"]) else None,
                        }
                        for i, r in enumerate(readings)
                    ],
                }
            )

    summary = {name: metrics.merge_reports(rep) for name, rep in reports.items()}
    split = {name: _split_metrics(rep) for name, rep in reports.items()}

    print(f"{'engine':12s} {'cov':>5s} {'sMAE':>6s} {'sMed':>6s} {'sP90':>6s} {'eMAE':>6s} {'eP90':>6s}")
    for name, a in summary.items():
        print(
            f"{name:12s} {a['coverage']:5.3f} {a['start_mae']:6.3f} {a['start_median']:6.3f} "
            f"{a['start_p90']:6.3f} {a['end_mae']:6.3f} {a['end_p90']:6.3f}"
        )
    print("\nlong (>=0.5s) vs normal:")
    for name, sp in split.items():
        lo, no = sp["long"], sp["normal"]
        print(
            f"  {name:12s} long n={lo['n']:3d} sMed={lo['start_median']:.3f} sP90={lo['start_p90']:.3f} eMAE={lo['end_mae']:.3f} | "
            f"normal n={no['n']:3d} sMed={no['start_median']:.3f} sP90={no['start_p90']:.3f} eMAE={no['end_mae']:.3f}"
        )

    for dump in dumps:
        print(f"\n=== dump {dump['id']} ===\n{dump['text']}")
        for c in dump["chars"]:
            print(
                f"  {c['char']!s:2s} {c['kana']:>5s} {''.join(c['phonemes']):>6s} "
                f"ref[{c['ref_start']},{c['ref_end']}] "
                f"shiro[{c['pyshiro_start']}..{c['pyshiro_end']}] "
                f"hub[{c['hubert_start']}..{c['hubert_end']}]"
            )

    payload = {"summary": summary, "long_vs_normal": split, "dumps": dumps}
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
