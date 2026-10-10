#!/usr/bin/env python
"""Singing-lyric alignment evaluation on GTSinger (CC BY-NC-SA 4.0).

Two evaluation modes (P0, V0.3)
-------------------------------
The V0.1/V0.2 harness leaked ground truth: it used the first/last annotated word
boundaries as the line window for *both* the heuristic and the model. V0.3 fixes
this by separating two clearly-labelled modes:

* ``lrc_only`` (the **realistic** mode) - the engine may use only the audio, the
  lyric *text* and an LRC line start time. There is no real LRC for GTSinger
  clips, so the LRC line start is simulated as ``0.0`` (the clip start); an
  audio-only energy onset is available via ``--lrc-onset energy``. Ground-truth
  boundaries are used **only** by the scorer, never to crop/align/segment.
* ``oracle`` - the line window is the annotated first/last word span. This is a
  *cheating* upper bound kept **only** for acoustic-algorithm research; it is
  always reported separately and never mixed with ``lrc_only``.

Engines: ``heuristic`` (uniform within the line window), ``model`` (V0.1) and
``model+energy`` (V0.2). Reports coverage, miss-rate, MAE/median/P90 for start
and end, error rates, uncertain-boundary ratio, failure samples, and the
per-singer / per-song / per-technique breakdowns plus the oracle-vs-lrc gap.

DATA / LICENSE
--------------
GTSinger audio + annotations are downloaded locally by
``tools/fetch_gtsinger.py`` into ``experiments/singing/`` (gitignored) and are
**not** redistributed. See docs/SINGING_EVAL.md.
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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autosync.alignment.heuristic import distribute
from autosync.alignment.refine import _frame_db, refine_units_energy
from autosync.alignment.resegment import resegment_align
from autosync.alignment.segment import _sanitize, check_reconstruction
from autosync.audio.io import crop, load_audio
from autosync.lyrics.normalize import display_units
from autosync.models.registry import get_aligner

EXCLUDE = {"<SP>", "<AP>"}
DEFAULT_ROOT = Path("experiments/singing/gtsinger")
MODES = ("lrc_only", "oracle")


# --------------------------------------------------------------------------- data
def load_reference(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out = []
    for e in data:
        if e.get("word") in EXCLUDE:
            continue
        out.append({"text": e["word"], "start": float(e["start_time"]), "end": float(e["end_time"])})
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
        group = parts[-2] if len(parts) > 1 else ""
        segments.append(
            {
                "language": language,
                "singer": parts[1] if len(parts) > 1 else "",
                "technique": group.replace("_Group", "") or "unknown",
                "song": parts[3] if len(parts) > 3 else "",
                "stem": json_path.stem,
                "json": json_path,
                "wav": wav,
                "id": str(json_path.relative_to(root).with_suffix("")),
            }
        )
    return segments


# ------------------------------------------------------------------- blind window
def energy_onset(audio: np.ndarray, sr: int, frame_seconds: float = 0.02) -> float:
    """First voiced frame from audio energy only (no annotations)."""
    db, fs = _frame_db(audio, sr, frame_seconds)
    if db.size == 0:
        return 0.0
    threshold = max(float(np.percentile(db, 20)) + 8.0, -50.0)
    voiced = np.nonzero(db > threshold)[0]
    return float(voiced[0] * fs) if voiced.size else 0.0


def line_window(
    mode: str,
    lrc_onset: str,
    audio: np.ndarray,
    sr: int,
    ref_start: float,
    ref_end: float,
) -> tuple[float, float]:
    """Return the line window used for cropping/alignment/heuristic.

    ``lrc_only`` never touches ``ref_start``/``ref_end`` (ground truth).
    """
    duration = len(audio) / sr
    if mode == "oracle":
        return ref_start, ref_end
    start = 0.0
    if lrc_onset == "energy":
        start = energy_onset(audio, sr)
    return start, duration


# ------------------------------------------------------------------ engines
def run_heuristic(language: str, text: str, start: float, end: float) -> list[dict]:
    units = display_units(language, text)
    dist = distribute(start, end, units)
    return [{"text": u.text, "start": u.start, "end": u.end, "uncertain": False} for u in dist]


def run_model(
    aligner, audio, sr: int, text: str, start: float, end: float, pad: float, refine: bool, start_mode: str = "none"
) -> list[dict]:
    w0 = max(0.0, start - pad)
    w1 = min(len(audio) / sr, end + pad)
    window = crop(audio, sr, w0, w1)
    units = aligner.align_text(window, sr, text)
    units, _ = _sanitize(units, len(window) / sr)
    if refine and units:
        units = refine_units_energy(
            units, window, sr, frame_seconds=getattr(aligner, "frame_seconds", 0.02) or 0.02,
            start_mode=start_mode,
        )
    return [
        {"text": u.text, "start": w0 + u.start, "end": w0 + u.end, "uncertain": u.uncertain}
        for u in units
    ]


def run_reseg(aligner, audio, sr: int, text: str, start: float, end: float, pad: float, language: str) -> list[dict]:
    w0 = max(0.0, start - pad)
    w1 = min(len(audio) / sr, end + pad)
    window = crop(audio, sr, w0, w1)
    units = resegment_align(aligner, window, sr, text, language=language)
    units, _ = _sanitize(units, len(window) / sr)
    return [
        {"text": u.text, "start": w0 + u.start, "end": w0 + u.end, "uncertain": u.uncertain}
        for u in units
    ]


def _as_units(pred: list[dict]):
    from autosync.types import AlignUnit

    return [
        AlignUnit(text=p["text"], start=p["start"], end=p["end"], uncertain=p.get("uncertain", False))
        for p in pred
    ]


# engine name -> (kind, aligner_key, refine, start_mode)
ENGINE_SPEC: dict[str, tuple[str, str | None, bool, str]] = {
    "model": ("window", "ctc", False, "none"),
    "model+energy": ("window", "ctc", True, "none"),
    "model+onset": ("window", "ctc", True, "backward"),
    "model+reseg": ("reseg", "ctc", False, "none"),
    "pyshiro": ("window", "shiro_v2", False, "none"),
    "pyshiro+energy": ("window", "shiro_v2", True, "none"),
    "pyshiro_v1": ("window", "shiro_v1", False, "none"),
    "pyshiro_v1+energy": ("window", "shiro_v1", True, "none"),
    "phoneme_ctc": ("window", "phoneme_ctc", False, "none"),
}


def evaluate_segment(seg: dict, engines: list[str], modes: list[str], aligners: dict, pad: float, lrc_onset: str) -> dict:
    audio, sr = load_audio(seg["wav"], 16000)
    ref_words = load_reference(seg["json"])
    if not ref_words:
        return {"id": seg["id"], "skipped": "no reference words"}
    text = line_text(seg["language"], ref_words)

    out: dict = {
        "id": seg["id"],
        "language": seg["language"],
        "singer": seg["singer"],
        "song": seg["song"],
        "technique": seg["technique"],
        "windows": {},
        "engines": {},
    }
    for mode in modes:
        start, end = line_window(mode, lrc_onset, audio, sr, ref_words[0]["start"], ref_words[-1]["end"])
        out["windows"][mode] = {"start": round(start, 4), "end": round(end, 4)}
        out["engines"][mode] = {}
        for engine in engines:
            t0 = time.time()
            error = None
            recon = None
            pred: list[dict] = []
            try:
                if engine == "heuristic":
                    pred = run_heuristic(seg["language"], text, start, end)
                else:
                    kind, akey, refine, smode = ENGINE_SPEC[engine]
                    al = aligners.get(akey)
                    if al is None:
                        raise RuntimeError(f"aligner '{akey}' unavailable")
                    if kind == "reseg":
                        pred = run_reseg(al, audio, sr, text, start, end, pad, seg["language"])
                    else:
                        pred = run_model(al, audio, sr, text, start, end, pad, refine=refine, start_mode=smode)
                    recon = check_reconstruction(seg["language"], text, _as_units(pred))
            except Exception as exc:  # noqa: BLE001 - record failure, never hide it
                error = f"{type(exc).__name__}: {exc}"
            report = metrics.score(pred, ref_words)
            report["seconds"] = time.time() - t0
            report["error"] = error
            report["reconstruction"] = recon
            out["engines"][mode][engine] = report
    return out


# ------------------------------------------------------------------ aggregation
def agg(reports: list[dict]) -> dict:
    if not reports:
        return {}
    a = metrics.merge_reports(reports)
    a["total_seconds"] = sum(r["seconds"] for r in reports)
    a["n_failed_segments"] = sum(1 for r in reports if r.get("error"))
    a["segment_failure_rate"] = a["n_failed_segments"] / len(reports)
    a["n_reconstruction_mismatch"] = sum(1 for r in reports if r.get("reconstruction"))
    return a


def _group_reports(segment_reports: list[dict], mode: str, engine: str, key: str) -> dict:
    buckets: dict[str, list[dict]] = {}
    for s in segment_reports:
        rep = s.get("engines", {}).get(mode, {}).get(engine)
        if rep is None:
            continue
        buckets.setdefault(str(s.get(key, "")), []).append(rep)
    return {k: agg(v) for k, v in buckets.items()}


def build_payload(segments: list[dict], langs, engines, modes, lrc_onset, pad) -> dict:
    all_reports: list[dict] = []
    summary: dict[str, dict] = {}
    by_singer: dict[str, dict] = {}
    by_song: dict[str, dict] = {}
    by_technique: dict[str, dict] = {}
    gap: dict[str, dict] = {}

    for language in langs:
        segs = [s for s in segments if s["language"] == language]
        aligners: dict[str, object] = {}
        needed = {ENGINE_SPEC[e][1] for e in engines if e in ENGINE_SPEC}
        if "ctc" in needed and segs:
            try:
                # Pin the ja CTC baseline to ReazonSpeech so the "model" engine keeps
                # its V0.5 meaning (the ja default backend is now HuBERT).
                ctc_backend = "reazonspeech-wav2vec2-base" if language == "ja" else None
                ctc = get_aligner(language, backend=ctc_backend)
                ctc.load()
                aligners["ctc"] = ctc
                print(f"[{language}] ctc model loaded: {ctc.meta.name}")
            except Exception as exc:  # noqa: BLE001
                print(f"[{language}] CTC MODEL UNAVAILABLE: {type(exc).__name__}: {exc}")
                aligners["ctc"] = None
        if language == "ja" and segs and ({"shiro_v1", "shiro_v2"} & needed):
            from autosync.models.pyshiro_aligner import PySHIROAligner

            for key, version in (("shiro_v2", "v2"), ("shiro_v1", "v1")):
                if key not in needed:
                    continue
                try:
                    shiro = PySHIROAligner(version)
                    shiro.load()
                    aligners[key] = shiro
                    print(f"[{language}] {key} loaded: {shiro.meta.name}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[{language}] {key} UNAVAILABLE: {type(exc).__name__}: {exc}")
                    aligners[key] = None
        if language == "ja" and segs and "phoneme_ctc" in needed:
            from autosync.models.phoneme_ctc_aligner import PhonemeCTCAligner

            try:
                pc = PhonemeCTCAligner()
                pc.load()
                aligners["phoneme_ctc"] = pc
                print(f"[{language}] phoneme_ctc loaded: {pc.meta.name}")
            except Exception as exc:  # noqa: BLE001
                print(f"[{language}] phoneme_ctc UNAVAILABLE: {type(exc).__name__}: {exc}")
                aligners["phoneme_ctc"] = None
        lang_reports = []
        for seg in segs:
            rep = evaluate_segment(seg, engines, modes, aligners, pad, lrc_onset)
            lang_reports.append(rep)
        all_reports.extend(lang_reports)

        summary[language] = {m: {e: agg([s["engines"][m][e] for s in lang_reports if "engines" in s and m in s["engines"]]) for e in engines} for m in modes}
        for m in modes:
            by_singer.setdefault(language, {})[m] = {e: _group_reports(lang_reports, m, e, "singer") for e in engines}
            by_song.setdefault(language, {})[m] = {e: _group_reports(lang_reports, m, e, "song") for e in engines}
            by_technique.setdefault(language, {})[m] = {e: _group_reports(lang_reports, m, e, "technique") for e in engines}
        # oracle vs lrc_only gap (lrc - oracle; positive = lrc worse)
        gap[language] = {}
        for e in engines:
            o = summary[language]["oracle"].get(e) or {}
            l = summary[language]["lrc_only"].get(e) or {}
            if not o or not l:
                gap[language][e] = None
                continue
            gap[language][e] = {
                "coverage_delta": l["coverage"] - o["coverage"],
                "start_mae_delta": l["start_mae"] - o["start_mae"],
                "end_mae_delta": l["end_mae"] - o["end_mae"],
            }

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "modes": modes,
        "engines": engines,
        "lrc_onset": lrc_onset,
        "peak_rss_mb": rss_mb,
        "summary": summary,
        "oracle_vs_lrc_gap": gap,
        "by_singer": by_singer,
        "by_song": by_song,
        "by_technique": by_technique,
        "segments": all_reports,
    }


def print_table(payload: dict, langs, engines, modes) -> None:
    for language in langs:
        for mode in modes:
            print(f"\n=== {language} / {mode} ===")
            print(f"{'engine':14s} {'n':>4s} {'cover':>6s} {'sMAE':>7s} {'sP90':>7s} {'eMAE':>7s} {'eP90':>7s} {'unc':>5s} {'fail%':>6s}")
            for engine in engines:
                a = payload["summary"][language][mode].get(engine)
                if not a:
                    continue
                print(
                    f"{engine:14s} {a['n_matched_words']:4d} {a['coverage']:6.3f} "
                    f"{a['start_mae']:7.3f} {a['start_p90']:7.3f} {a['end_mae']:7.3f} {a['end_p90']:7.3f} "
                    f"{a.get('uncertain_ratio', 0):5.2f} {100 * a.get('segment_failure_rate', 0):6.1f}"
                )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--languages", default="en,ja,zh")
    ap.add_argument("--engines", default="heuristic,model,model+energy")
    ap.add_argument("--modes", default="lrc_only,oracle")
    ap.add_argument("--lrc-onset", choices=["zero", "energy"], default="zero",
                    help="how the simulated LRC line start is chosen in lrc_only mode")
    ap.add_argument("--pad", type=float, default=0.30)
    ap.add_argument("--out", default="experiments/reports/singing_report.json")
    args = ap.parse_args()

    langs = [s for s in args.languages.split(",") if s]
    engines = [s for s in args.engines.split(",") if s]
    modes = [s for s in args.modes.split(",") if s]
    segments = discover(Path(args.root))
    print(f"discovered {len(segments)} segments under {args.root}")

    payload = build_payload(segments, langs, engines, modes, args.lrc_onset, args.pad)
    print_table(payload, langs, engines, modes)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\npeak RSS: {payload['peak_rss_mb']:.0f} MB")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
