#!/usr/bin/env python
"""Accompaniment robustness: align a-cappella vs the same vocal + a music bed.

Why a *synthesised* bed
-----------------------
The GTSinger segments used here are a-cappella, so they carry exact ground
truth. To measure how much a musical accompaniment degrades alignment we mix
the clean vocal with an accompaniment bed **rendered from the segment's own
score** (``note`` / ``note_start`` / ``note_end`` in the GTSinger sidecar JSON).
This bed contains no third-party audio, so it is fully licence-safe, and every
error increase is attributable to the mixture.

This is a *controlled* experiment: it is **not** a real commercial mix. A real
vocal+accompaniment recording (e.g. MIR-1K) could not be obtained here — the
public download link is dead and its licence is research-only. See
docs/SINGING_EVAL.md and docs/EXPERIMENT_RESULTS.md.

For each segment we align:
* the clean a-cappella, and
* the mixture at several vocal-to-accompaniment ratios,
and report the error against the GTSinger reference for both.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import metrics
import numpy as np
from singing_eval import (
    discover,
    line_text,
    load_reference,
    run_model,
)

from autosync.audio.io import load_audio
from autosync.models.registry import get_aligner


def render_bed(ref_json: Path, sr: int, duration: float) -> np.ndarray:
    """Render a simple harmonic accompaniment bed from the segment's score."""
    data = json.loads(ref_json.read_text(encoding="utf-8"))
    n = int(duration * sr)
    bed = np.zeros(n, dtype=np.float64)
    for entry in data:
        notes = entry.get("note") or []
        starts = entry.get("note_start") or []
        ends = entry.get("note_end") or []
        for midi, s, e in zip(notes, starts, ends):
            if not midi or midi <= 0:
                continue
            f0 = 440.0 * 2 ** ((midi - 69) / 12.0)
            i0 = max(0, int(s * sr))
            i1 = min(n, int(e * sr))
            if i1 <= i0:
                continue
            tt = np.arange(i1 - i0) / sr
            env = np.exp(-1.2 * tt)
            for octave, oamp in ((-1, 0.6), (-2, 0.3)):
                for harm, hamp in ((1, 1.0), (2, 0.4), (3, 0.2)):
                    bed[i0:i1] += oamp * hamp * np.sin(2 * np.pi * f0 * 2**octave * harm * tt) * env
    return bed


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def mix(vocal: np.ndarray, bed: np.ndarray, ratio_db: float) -> np.ndarray:
    bed = bed[: len(vocal)]
    if bed.size < len(vocal):
        bed = np.pad(bed, (0, len(vocal) - bed.size))
    vocal_rms = _rms(vocal)
    bed_rms = _rms(bed)
    if bed_rms > 0:
        bed = bed * (vocal_rms * 10 ** (ratio_db / 20.0) / bed_rms)
    out = vocal + bed
    peak = np.max(np.abs(out)) or 1.0
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="experiments/singing/gtsinger")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--ratios", default="0,-6", help="vocal:accomp dB (0 = equal RMS)")
    ap.add_argument("--max-segments", type=int, default=8)
    ap.add_argument("--pad", type=float, default=0.30)
    ap.add_argument("--out", default="experiments/reports/accompaniment_report.json")
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    segs = [s for s in discover(Path(args.root)) if s["language"] == args.language][: args.max_segments]
    aligner = get_aligner(args.language)
    aligner.load()
    print(f"model: {aligner.meta.name}; {len(segs)} segments")

    results: dict[str, dict] = {"clean": {}}
    for r in ratios:
        results[f"mix_{r:+.0f}dB"] = {}
    per_engine: dict[str, list[dict]] = {k: [] for k in results}

    for seg in segs:
        audio, sr = load_audio(seg["wav"], 16000)
        ref_words = load_reference(seg["json"])
        text = line_text(seg["language"], ref_words)
        start, end = ref_words[0]["start"], ref_words[-1]["end"]
        bed = render_bed(seg["json"], sr, len(audio) / sr)

        variants = {"clean": audio}
        for r in ratios:
            variants[f"mix_{r:+.0f}dB"] = mix(audio, bed, r)

        for key, sig in variants.items():
            for engine, refine in (("model", False), ("model+energy", True)):
                try:
                    pred = run_model(aligner, sig, sr, text, start, end, args.pad, refine=refine)
                    rep = metrics.score(pred, ref_words)
                    rep["error"] = None
                except Exception as exc:  # noqa: BLE001
                    rep = metrics.score([], ref_words)
                    rep["error"] = f"{type(exc).__name__}: {exc}"
                rep["segment"] = seg["id"]
                rep["engine"] = engine
                per_engine[key].append(rep)

    summary = {k: metrics.merge_reports(v) for k, v in per_engine.items()}
    print(f"\nlanguage={args.language}  model={aligner.meta.name}")
    print(f"{'variant':12s} {'startMAE':>9s} {'startP90':>9s} {'endMAE':>8s} {'endP90':>8s} {'cover':>6s} {'fail':>5s}")
    for k in results:
        a = summary[k]
        fails = sum(1 for r in per_engine[k] if r.get("error"))
        print(
            f"{k:12s} {a['start_mae']:9.3f} {a['start_p90']:9.3f} {a['end_mae']:8.3f} "
            f"{a['end_p90']:8.3f} {a['coverage']:6.3f} {fails:5d}"
        )

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "language": args.language,
        "model": aligner.meta.name,
        "ratios_db": ratios,
        "note": "synthesised accompaniment bed from the segment score; not a real commercial mix",
        "summary": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
