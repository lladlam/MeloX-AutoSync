#!/usr/bin/env python
"""Singing-lyric alignment evaluation on GTSinger (CC BY-NC-SA 4.0).

This is the V0.2 "does it actually work on *singing*" benchmark. It compares
three engines on real singing segments that carry per-word (English) or
per-character (Chinese/Japanese) reference boundaries:

* ``heuristic``   – uniform distribution across the known line span (no acoustics)
* ``model``       – V0.1 wav2vec2 / FunASR forced alignment
* ``model+energy``– V0.2 model + energy-aware boundary refinement

It reports start/end MAE, median, P90, coverage, miss-rate, extra units, error
rates, wall time and peak RSS, and keeps the failure samples.

DATA / LICENSE
--------------
The GTSinger audio and annotations are downloaded locally by
``tools/fetch_gtsinger.py`` into ``experiments/singing/`` (gitignored). They are
**not** redistributed. Run this on a machine where you have accepted the
GTSinger licence. See docs/SINGING_EVAL.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import metrics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autosync.alignment.heuristic import distribute
from autosync.alignment.refine import refine_units_energy
from autosync.alignment.segment import _sanitize, check_reconstruction
from autosync.audio.io import crop, load_audio
from autosync.lyrics.normalize import display_units
from autosync.models.registry import get_aligner

EXCLUDE = {"<SP>", "<AP>"}
DEFAULT_ROOT = Path("experiments/singing/gtsinger")


def load_reference(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out = []
    for e in data:
        if e.get("word") in EXCLUDE:
            continue
        out.append(
            {
                "text": e["word"],
                "start": float(e["start_time"]),
                "end": float(e["end_time"]),
            }
        )
    return out


def line_text(language: str, words: list[dict]) -> str:
    return " ".join(w["text"] for w in words) if language == "en" else "".join(w["text"] for w in words)


def discover(root: Path) -> list[dict]:
    segments = []
    for json_path in sorted(root.glob("**/*.json")):
        if json_path.name == "manifest.json":
            continue
        wav = json_path.with_suffix(".wav")
        if not wav.exists():
            continue
        parts = json_path.relative_to(root).parts
        language = {"English": "en", "Japanese": "ja", "Chinese": "zh"}.get(parts[0], "?")
        segments.append(
            {
                "language": language,
                "singer": parts[1] if len(parts) > 1 else "",
                "technique": parts[2] if len(parts) > 2 else "",
                "group": parts[-2] if len(parts) > 1 else "",
                "song": parts[3] if len(parts) > 3 else "",
                "stem": json_path.stem,
                "json": json_path,
                "wav": wav,
                "id": str(json_path.relative_to(root).with_suffix("")),
            }
        )
    return segments


def run_heuristic(language: str, text: str, start: float, end: float) -> list[dict]:
    units = display_units(language, text)
    dist = distribute(start, end, units)
    return [
        {"text": u.text, "start": u.start, "end": u.end, "uncertain": False} for u in dist
    ]


def run_model(
    aligner,
    audio,
    sr: int,
    text: str,
    start: float,
    end: float,
    pad: float,
    refine: bool,
) -> list[dict]:
    w0 = max(0.0, start - pad)
    w1 = min(len(audio) / sr, end + pad)
    window = crop(audio, sr, w0, w1)
    units = aligner.align_text(window, sr, text)
    units, _ = _sanitize(units, len(window) / sr)
    if refine and units:
        units = refine_units_energy(
            units,
            window,
            sr,
            frame_seconds=getattr(aligner, "frame_seconds", 0.02) or 0.02,
        )
    return [
        {"text": u.text, "start": w0 + u.start, "end": w0 + u.end, "uncertain": u.uncertain}
        for u in units
    ]


def evaluate_segment(seg: dict, engines: list[str], aligner, pad: float) -> dict:
    audio, sr = load_audio(seg["wav"], 16000)
    ref_words = load_reference(seg["json"])
    if not ref_words:
        return {"id": seg["id"], "skipped": "no reference words"}
    text = line_text(seg["language"], ref_words)
    start = ref_words[0]["start"]
    end = ref_words[-1]["end"]

    out: dict = {"id": seg["id"], "language": seg["language"], "group": seg["group"], "engines": {}}
    for engine in engines:
        t0 = time.time()
        error = None
        pred: list[dict] = []
        recon = None
        try:
            if engine == "heuristic":
                pred = run_heuristic(seg["language"], text, start, end)
            elif engine == "model":
                if aligner is None:
                    raise RuntimeError("model unavailable")
                pred = run_model(aligner, audio, sr, text, start, end, pad, refine=False)
                recon = check_reconstruction(seg["language"], text, _as_units(pred))
            elif engine == "model+energy":
                if aligner is None:
                    raise RuntimeError("model unavailable")
                pred = run_model(aligner, audio, sr, text, start, end, pad, refine=True)
                recon = check_reconstruction(seg["language"], text, _as_units(pred))
        except Exception as exc:  # noqa: BLE001 - record failure, do not hide it
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - t0

        report = metrics.score(pred, ref_words)
        report["seconds"] = elapsed
        report["error"] = error
        report["reconstruction"] = recon
        out["engines"][engine] = report
    return out


def _as_units(pred: list[dict]):
    from autosync.types import AlignUnit

    return [
        AlignUnit(
            text=p["text"],
            start=p["start"],
            end=p["end"],
            uncertain=p.get("uncertain", False),
        )
        for p in pred
    ]


def aggregate(segment_reports: list[dict], engine: str) -> dict:
    reports = [s["engines"][engine] for s in segment_reports if engine in s["engines"]]
    agg = metrics.merge_reports(reports)
    agg["total_seconds"] = sum(r["seconds"] for r in reports)
    agg["n_failed_segments"] = sum(1 for r in reports if r.get("error"))
    agg["n_reconstruction_mismatch"] = sum(1 for r in reports if r.get("reconstruction"))
    return agg


def print_table(results: dict, engines: list[str]) -> None:
    for language in sorted(results):
        print(f"\n=== language: {language} ===")
        header = f"{'engine':14s} {'n':>3s} {'cover':>6s} {'startMAE':>9s} {'startP90':>9s} {'endMAE':>8s} {'endP90':>8s} {'fail':>5s} {'sec':>7s}"
        print(header)
        for engine in engines:
            a = results[language].get(engine)
            if not a:
                continue
            print(
                f"{engine:14s} {a['n_matched_words']:3d} {a['coverage']:6.3f} "
                f"{a['start_mae']:9.3f} {a['start_p90']:9.3f} {a['end_mae']:8.3f} {a['end_p90']:8.3f} "
                f"{a.get('n_failed_segments', 0):5d} {a.get('total_seconds', 0):7.1f}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--languages", default="en,ja,zh")
    ap.add_argument("--engines", default="heuristic,model,model+energy")
    ap.add_argument("--pad", type=float, default=0.30)
    ap.add_argument("--out", default="experiments/reports/singing_report.json")
    args = ap.parse_args()

    langs = [s for s in args.languages.split(",") if s]
    engines = [s for s in args.engines.split(",") if s]
    segments = discover(Path(args.root))
    print(f"discovered {len(segments)} segments under {args.root}")

    all_reports: list[dict] = []
    results: dict[str, dict] = {}
    for language in langs:
        segs = [s for s in segments if s["language"] == language]
        aligner = None
        if any(e.startswith("model") for e in engines) and segs:
            try:
                aligner = get_aligner(language)
                aligner.load()
                print(f"[{language}] model loaded: {aligner.meta.name}")
            except Exception as exc:  # noqa: BLE001
                print(f"[{language}] MODEL UNAVAILABLE: {type(exc).__name__}: {exc}")
                aligner = None
        lang_reports = []
        for seg in segs:
            rep = evaluate_segment(seg, engines, aligner, args.pad)
            rep["language"] = language
            lang_reports.append(rep)
        all_reports.extend(lang_reports)
        results[language] = {e: aggregate(lang_reports, e) for e in engines}

    print_table(results, engines)

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "root": args.root,
        "engines": engines,
        "peak_rss_mb": rss_mb,
        "summary": results,
        "segments": all_reports,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\npeak RSS: {rss_mb:.0f} MB")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
