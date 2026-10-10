#!/usr/bin/env python
"""Ablation of pause-aware alignment variants on Japanese singing (V0.6 P2).

Compares, under the same ``lrc_only`` protocol and the same audio/text:

1. ``none``            – V0.5 HuBERT forced alignment
2. ``blank_aware``     – trim token spans to the frames where they beat blank
3. ``optional_pause``  – allow optional ``pau``/``sil`` (expected no-op)
4. ``gap_penalized``   – duration-regularised Viterbi (non-blank penalty)
5. combinations

Two independent measurements are reported:

* **non-lyric detection** – frame-level Precision/Recall/F1 of the gaps the
  method leaves (complement of its lyric token spans) against the reference
  ``<AP>``/``<SP>`` intervals (reference used for scoring only);
* **lyric boundary accuracy** – per-character start/end MAE, median, P90, P95 and
  the fraction of boundaries worse than 0.5 s / 1.0 s, plus coverage and
  by-singer breakdown.

No reference annotation is ever used to place a boundary.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import metrics
import numpy as np
from singing_eval import discover, line_text, load_reference

from autosync.alignment.pause_aware import PauseConfig, resolve_spans
from autosync.audio.io import load_audio
from autosync.lyrics.ja_pronunciation import locate_inner_offset
from autosync.models.phoneme_ctc_aligner import PhonemeCTCAligner, map_spans_to_char_units

METHODS: dict[str, PauseConfig] = {
    "v05_none": PauseConfig(mode="none"),
    "blank_aware": PauseConfig(mode="blank_aware", trim_margin=0.8, trim_min_frames=1),
    "optional_pause": PauseConfig(mode="optional_pause", pause_gain_margin=1.0),
    "gap_pen0.5": PauseConfig(mode="gap_penalized", non_blank_penalty=0.5),
    "gap_state": PauseConfig(mode="gap_state", sm_dmin=2, sm_dmax=25, sm_gmax_intra=2, sm_gmax_inter=250),
    "gap_state_loose": PauseConfig(mode="gap_state", sm_dmin=1, sm_dmax=40, sm_gmax_intra=4, sm_gmax_inter=400),
}


def _intra_mask(readings, phonemes) -> list[bool]:
    """True for the transition between two phonemes of the *same* character."""
    inner = [p for r in readings for p in r.phonemes]
    offset = locate_inner_offset(phonemes, inner)
    char_of = [-1] * len(phonemes)
    cursor = offset
    for ci, r in enumerate(readings):
        for _ in range(len(r.phonemes)):
            if cursor < len(char_of):
                char_of[cursor] = ci
            cursor += 1
    return [char_of[i] == char_of[i + 1] and char_of[i] >= 0 for i in range(len(phonemes) - 1)]


def _nonlyric_frames(ref_json: Path, n_frames: int, fs: float) -> np.ndarray:
    data = json.loads(ref_json.read_text(encoding="utf-8"))
    mask = np.zeros(n_frames, dtype=bool)
    for e in data:
        if e["word"] in ("<AP>", "<SP>"):
            i0, i1 = int(e["start_time"] / fs), int(e["end_time"] / fs)
            mask[max(0, i0) : min(n_frames, i1)] = True
    return mask


def _detected_gap_frames(units, n_frames: int, fs: float) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for u in units:
        i0, i1 = int(u.start / fs), int(np.ceil(u.end / fs))
        mask[max(0, i0) : min(n_frames, i1)] = True
    return ~mask


def _prf(detected: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(detected & gt))
    p = tp / int(detected.sum()) if detected.sum() else 0.0
    r = tp / int(gt.sum()) if gt.sum() else float("nan")
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def _stats(pairs: list[dict]) -> dict:
    se = [p["start_err"] for p in pairs]
    ee = [p["end_err"] for p in pairs]
    n = len(se)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "start_mae": float(np.mean(se)),
        "start_median": float(np.median(se)),
        "start_p90": float(np.percentile(se, 90)),
        "start_p95": float(np.percentile(se, 95)),
        "end_mae": float(np.mean(ee)),
        "end_median": float(np.median(ee)),
        "end_p90": float(np.percentile(ee, 90)),
        "end_p95": float(np.percentile(ee, 95)),
        "start_gt0.5": float(np.mean([x > 0.5 for x in se])),
        "start_gt1.0": float(np.mean([x > 1.0 for x in se])),
        "end_gt0.5": float(np.mean([x > 0.5 for x in ee])),
        "end_gt1.0": float(np.mean([x > 1.0 for x in ee])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="experiments/singing/gtsinger")
    ap.add_argument("--out", default="experiments/reports/pause_eval.json")
    args = ap.parse_args()

    segs = [s for s in discover(Path(args.root)) if s["language"] == "ja"]
    aligner = PhonemeCTCAligner()
    aligner.load()
    blank = aligner._blank_id()
    vocab = aligner._vocab
    pause_ids = tuple(i for i in (vocab.get("pau"), vocab.get("sil")) if i is not None)
    fs = aligner.frame_seconds
    mapper = aligner._mapper

    # set optional-pause ids
    for cfg in METHODS.values():
        if cfg.mode == "optional_pause":
            cfg.pause_ids = pause_ids

    per_method: dict[str, dict] = {
        m: {"pairs": [], "by_singer": {}, "prf": [], "gap_s": [], "seconds": 0.0} for m in METHODS
    }
    t_start = time.time()

    for seg in segs:
        audio, sr = load_audio(seg["wav"], 16000)
        ref_words = load_reference(seg["json"])
        text = line_text("ja", ref_words)
        readings, phonemes = mapper.map(text)
        ids, keep = aligner._phoneme_ids(phonemes)
        intra = _intra_mask(readings, phonemes)
        logp = aligner._log_probs(audio, sr)
        n_frames = logp.shape[0]
        gt = _nonlyric_frames(seg["json"], n_frames, fs)

        for name, cfg in METHODS.items():
            t0 = time.time()
            al = resolve_spans(logp, ids, blank, cfg, intra=intra)
            spans_by_phoneme = {k: s for k, s in zip(keep, al.spans) if s is not None}
            units = map_spans_to_char_units(readings, phonemes, spans_by_phoneme, fs)
            pred = [{"text": u.text, "start": u.start, "end": u.end, "uncertain": u.uncertain} for u in units]
            rep = metrics.score(pred, ref_words)
            dt = time.time() - t0

            per_method[name]["seconds"] += dt
            per_method[name]["prf"].append(_prf(_detected_gap_frames(units, n_frames, fs), gt))
            per_method[name]["pairs"].extend(rep["pairs"])
            per_method[name]["by_singer"].setdefault(seg["singer"], []).extend(rep["pairs"])

    print(f"{'method':16s} {'cov':>5s} {'sMAE':>6s} {'sMed':>6s} {'sP90':>6s} {'sP95':>6s} {'>1.0s':>6s} {'eMAE':>6s} {'eP90':>6s} {'gapF1':>6s} {'sec':>7s}")
    total_ref = sum(len(load_reference(s["json"])) for s in segs)
    summary = {}
    for name in METHODS:
        pairs = per_method[name]["pairs"]
        st = _stats(pairs)
        cov = st["n"] / total_ref if total_ref else float("nan")
        f1s = [f for _, _, f in per_method[name]["prf"]]
        gap_f1 = float(np.mean(f1s)) if f1s else float("nan")
        summary[name] = {**st, "coverage": cov, "gap_f1": gap_f1, "seconds": per_method[name]["seconds"]}
        print(
            f"{name:16s} {cov:5.3f} {st.get('start_mae', float('nan')):6.3f} {st.get('start_median', float('nan')):6.3f} "
            f"{st.get('start_p90', float('nan')):6.3f} {st.get('start_p95', float('nan')):6.3f} {st.get('start_gt1.0', float('nan')):6.3f} "
            f"{st.get('end_mae', float('nan')):6.3f} {st.get('end_p90', float('nan')):6.3f} {gap_f1:6.3f} {per_method[name]['seconds']:7.1f}"
        )

    by_singer = {
        name: {s: _stats(p) for s, p in per_method[name]["by_singer"].items()} for name in METHODS
    }

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "peak_rss_mb": rss_mb,
        "total_seconds": time.time() - t_start,
        "summary": summary,
        "by_singer": by_singer,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\npeak RSS {rss_mb:.0f} MB, total {payload['total_seconds']:.1f}s")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
