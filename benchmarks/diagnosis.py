#!/usr/bin/env python
"""Per-segment acoustic diagnosis for the wav2vec2 (en/ja) engines.

Answers, with evidence, the V0.3 P1/P2 questions:

* Is a failure the *model* (CTC transcript far from the reference) or the
  *window / algorithm*?
* How do CTC **blank** and token frames distribute, and how large is the
  dominant token (collapse signal)?
* Does the **window** matter (full clip / oracle / energy-trimmed)?
* How many multi-character (uncertain) tokens are there?

It does not change any thresholds; it only measures. Output is a JSON report
plus a printed summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics
from singing_eval import discover, energy_onset, line_text, load_reference

from autosync.alignment.ctc import forced_align
from autosync.audio.io import crop, load_audio
from autosync.lyrics.normalize import normalize_cjk, normalize_en
from autosync.models.registry import get_aligner


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize_text(language: str, text: str) -> str:
    return normalize_en(text) if language == "en" else normalize_cjk(text)


def window_variants(language, audio, sr, ref_words, onset_energy: bool) -> dict[str, tuple[float, float]]:
    dur = len(audio) / sr
    variants = {
        "lrc": (0.0, dur),
        "oracle": (ref_words[0]["start"], ref_words[-1]["end"]),
    }
    if onset_energy:
        o = energy_onset(audio, sr)
        variants["energy_trim"] = (o, dur)
    return variants


def inspect_window(aligner, language, window, sr, text) -> dict:
    window = np.ascontiguousarray(window, dtype=np.float32)
    logp = aligner._log_probs(window, sr)  # (T, C)
    normalized = normalize_text(language, text)
    tokens = aligner._tokenize(normalized)
    chars = aligner._ids_to_chars(tokens)
    blank = aligner._blank_id()
    al = forced_align(logp, tokens, blank=blank)

    # CTC transcript (argmax) for model-vs-reference comparison.
    ids = np.argmax(logp, axis=-1)
    try:
        transcript = aligner._processor.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
    except Exception:  # noqa: BLE001
        transcript = ""
    transcript_n = normalize_text(language, transcript)

    valid = [s for s in al.spans if s is not None]
    token_frames = sum(s.num_frames for s in valid)
    T = logp.shape[0]
    max_frac = (max((s.num_frames for s in valid), default=0) / T) if T else 0.0
    multi_char = sum(1 for c in chars if len(c.replace("\u2581", "").strip()) > 1)
    return {
        "frames": int(T),
        "n_tokens": len(tokens),
        "n_multi_char_tokens": int(multi_char),
        "feasible": al.feasible,
        "reasons": al.reasons,
        "max_span_fraction": round(float(max_frac), 3),
        "blank_fraction": round(1.0 - token_frames / T, 3) if T else None,
        "mean_token_score": round(float(np.mean([s.score for s in valid])), 2) if valid else None,
        "transcript": transcript_n,
        "transcript_cer": round(levenshtein(transcript_n, normalized) / max(1, len(normalized)), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--language", default="ja")
    ap.add_argument("--root", default="experiments/singing/gtsinger")
    ap.add_argument("--out", default="experiments/reports/diagnosis_ja.json")
    args = ap.parse_args()

    segs = [s for s in discover(Path(args.root)) if s["language"] == args.language]
    aligner = get_aligner(args.language)
    aligner.load()
    print(f"{args.language}: {len(segs)} segments, model={aligner.meta.name}")

    records = []
    for seg in segs:
        audio, sr = load_audio(seg["wav"], 16000)
        ref_words = load_reference(seg["json"])
        text = line_text(args.language, ref_words)
        variants = window_variants(args.language, audio, sr, ref_words, onset_energy=True)
        rec = {"id": seg["id"], "singer": seg["singer"], "song": seg["song"], "technique": seg["technique"]}
        for name, (s, e) in variants.items():
            w = crop(audio, sr, s, e)
            if w.size < int(0.05 * sr):
                rec[name] = {"error": "empty window"}
                continue
            try:
                rec[name] = inspect_window(aligner, args.language, w, sr, text)
            except Exception as exc:  # noqa: BLE001
                rec[name] = {"error": f"{type(exc).__name__}: {exc}"}
        # Word-level signed bias using the real engine output on the lrc window.
        try:
            w = crop(audio, sr, 0.0, len(audio) / sr)
            units = aligner.align_text(w, sr, text)
            pred = [
                {"text": u.text, "start": u.start, "end": u.end, "uncertain": getattr(u, "uncertain", False)}
                for u in units
            ]
            sc = metrics.score(pred, ref_words)
            signed_start = [p["pred_start"] - p["ref_start"] for p in sc["pairs"]]
            signed_end = [p["pred_end"] - p["ref_end"] for p in sc["pairs"]]
            rec["word_level_lrc"] = {
                "coverage": sc["coverage"],
                "start_mae": sc["start_mae"],
                "end_mae": sc["end_mae"],
                "signed_start_mean": float(np.mean(signed_start)) if signed_start else None,
                "signed_end_mean": float(np.mean(signed_end)) if signed_end else None,
            }
        except Exception as exc:  # noqa: BLE001
            rec["word_level_lrc"] = {"error": f"{type(exc).__name__}: {exc}"}
        records.append(rec)

    n = len(records)
    collapse = {name: sum(1 for r in records if not r.get(name, {}).get("feasible", False)) for name in ("lrc", "oracle", "energy_trim")}
    cer = [r["lrc"].get("transcript_cer") for r in records if r.get("lrc", {}).get("transcript_cer") is not None]
    print(f"\ncollapse (infeasible) by window over {n} segments:")
    for k, v in collapse.items():
        print(f"  {k:12s}: {v}/{n}")
    if cer:
        print(f"\nCTC transcript CER (lrc window): mean={np.mean(cer):.3f} median={np.median(cer):.3f}")
    print("\nper-segment (lrc / oracle): feasible, max-span-frac, CER")
    for r in records[:40]:
        l, o = r.get("lrc", {}), r.get("oracle", {})
        print(
            f"  {r['song'][:14]:14s} {r['technique'][:9]:9s} "
            f"lrc={l.get('feasible')} max={l.get('max_span_fraction')} cer={l.get('transcript_cer')} | "
            f"orc={o.get('feasible')} max={o.get('max_span_fraction')}"
        )

    payload = {"language": args.language, "model": aligner.meta.name, "n_segments": n, "collapse": collapse, "segments": records}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
