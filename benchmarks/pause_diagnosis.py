#!/usr/bin/env python
"""Non-lyric / pause diagnosis for the Japanese HuBERT phoneme CTC (V0.6 P0/P3).

Ground truth (``<AP>``/``<SP>`` intervals, per-character times) is used **only**
for offline diagnosis and scoring here — never inside an alignment algorithm.

For every Japanese segment it records, per frame:
* blank (``PAD``), ``pau``, ``sil`` and max-lyric posterior,
* audio energy,
and, split by reference non-lyric vs lyric intervals, plus the V0.5 alignment
error. It also writes the raw posterior arrays for the requested segments so a
diagnostic figure can be reproduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import metrics
import numpy as np
from singing_eval import discover, line_text, load_reference

from autosync.audio.io import load_audio
from autosync.models.phoneme_ctc_aligner import PhonemeCTCAligner


def _reference_intervals(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [
        {"word": e["word"], "start": e["start_time"], "end": e["end_time"]}
        for e in data
    ]


def _energy(db: np.ndarray, t0: float, t1: float, fs: float) -> float | None:
    i0, i1 = int(t0 / fs), min(len(db), int(t1 / fs))
    return float(np.mean(db[i0:i1])) if i1 > i0 else None


def _posterior_stats(p: np.ndarray, blank: int, pau: int, sil: int, t0: float, t1: float, fs: float) -> dict:
    i0, i1 = int(t0 / fs), int(t1 / fs)
    i0, i1 = max(0, i0), min(p.shape[0], i1)
    if i1 <= i0:
        return {}
    seg = p[i0:i1]
    lyric = np.delete(seg, [blank, pau, sil], axis=1)
    return {
        "blank": float(seg[:, blank].mean()),
        "pau": float(seg[:, pau].mean()) if pau >= 0 else None,
        "sil": float(seg[:, sil].mean()) if sil >= 0 else None,
        "max_lyric": float(lyric.max(axis=1).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="experiments/singing/gtsinger")
    ap.add_argument("--language", default="ja")
    ap.add_argument("--dump-segments", default="さんぽ", help="substring; raw arrays saved")
    ap.add_argument("--max-dump", type=int, default=1)
    ap.add_argument("--out", default="experiments/reports/pause_diagnosis.json")
    args = ap.parse_args()

    segs = [s for s in discover(Path(args.root)) if s["language"] == args.language]
    aligner = PhonemeCTCAligner()
    aligner.load()
    blank = aligner._blank_id()
    vocab = aligner._vocab
    pau = vocab.get("pau", -1)
    sil = vocab.get("sil", -1)
    fs = aligner.frame_seconds

    records = []
    dumps = []
    for seg in segs:
        audio, sr = load_audio(seg["wav"], 16000)
        ref_words = load_reference(seg["json"])
        text = line_text(args.language, ref_words)
        logp = aligner._log_probs(audio, sr)
        p = np.exp(logp)

        intervals = _reference_intervals(seg["json"])
        ap = [e for e in intervals if e["word"] in ("<AP>", "<SP>")]
        interior_ap = [e for e in ap if e["start"] > 0 and e["end"] < intervals[-1]["end"]]
        lyric = [e for e in intervals if e["word"] not in ("<AP>", "<SP>")]

        lyric_stats = [_posterior_stats(p, blank, pau, sil, e["start"], e["end"], fs) for e in lyric]
        ap_stats = [_posterior_stats(p, blank, pau, sil, e["start"], e["end"], fs) for e in ap]

        def agg(stats, key):
            vals = [s[key] for s in stats if s and s.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        # energy
        fl = int(fs * sr)
        n = len(audio) // fl
        rms = np.sqrt(np.mean(audio[: n * fl].reshape(n, fl) ** 2, axis=1))
        db = 20 * np.log10(rms + 1e-9)

        ap_energy = [x for x in (_energy(db, e["start"], e["end"], fs) for e in ap) if x is not None]
        lyric_energy = [x for x in (_energy(db, e["start"], e["end"], fs) for e in lyric) if x is not None]

        pred = aligner.align_text(audio, sr, text)
        pred_units = [{"text": u.text, "start": u.start, "end": u.end, "uncertain": getattr(u, "uncertain", False)} for u in pred]
        report = metrics.score(pred_units, ref_words)

        rec = {
            "id": seg["id"],
            "singer": seg["singer"],
            "song": seg["song"],
            "technique": seg["technique"],
            "n_ref": len(ref_words),
            "start_mae": report["start_mae"],
            "start_p90": report["start_p90"],
            "end_mae": report["end_mae"],
            "internal_pause_s": sum(e["end"] - e["start"] for e in interior_ap),
            "n_ap": len(ap),
            "blank_all": float(p[:, blank].mean()),
            "max_lyric_all": float(np.delete(p, [blank, pau, sil], axis=1).max(axis=1).mean()),
            "blank_ap": agg(ap_stats, "blank"),
            "max_lyric_ap": agg(ap_stats, "max_lyric"),
            "blank_lyric": agg(lyric_stats, "blank"),
            "max_lyric_lyric": agg(lyric_stats, "max_lyric"),
            "pau_all": float(p[:, pau].mean()) if pau >= 0 else None,
            "sil_all": float(p[:, sil].mean()) if sil >= 0 else None,
            "energy_ap_db": float(np.mean(ap_energy)) if ap_energy else None,
            "energy_lyric_db": float(np.mean(lyric_energy)) if lyric_energy else None,
        }
        records.append(rec)

        if args.dump_segments in seg["song"] and len(dumps) < args.max_dump:
            np.savez(
                str(Path(args.out).with_suffix("").with_name(Path(args.out).stem + "_arrays.npz")),
                probabilities=p.astype(np.float32),
                frame_seconds=fs,
                blank=blank,
                pau=pau,
                sil=sil,
            )
            dumps.append({"id": seg["id"], "text": text, "ref": intervals})

    # correlations
    mae = np.array([r["start_mae"] for r in records])
    lyric_post = np.array([r["max_lyric_all"] for r in records])
    blank_post = np.array([r["blank_all"] for r in records])
    pause = np.array([r["internal_pause_s"] for r in records])

    print(f"segments: {len(records)}")
    print(f"corr(start_MAE, max_lyric_all) = {float(np.corrcoef(mae, lyric_post)[0, 1]):.3f}")
    print(f"corr(start_MAE, blank_all)     = {float(np.corrcoef(mae, blank_post)[0, 1]):.3f}")
    print(f"corr(start_MAE, internal_pause)= {float(np.corrcoef(mae, pause)[0, 1]):.3f}")
    print("\nworst 6 by start_MAE:")
    for r in sorted(records, key=lambda r: -r["start_mae"])[:6]:
        eap = None if r["energy_ap_db"] is None else round(r["energy_ap_db"], 1)
        ely = None if r["energy_lyric_db"] is None else round(r["energy_lyric_db"], 1)
        print(
            f"  {r['song'][:20]:22s} sMAE={r['start_mae']:.2f} lyricP={r['max_lyric_all']:.3f} "
            f"blankP={r['blank_all']:.3f} pause={r['internal_pause_s']:.1f}s Eap={eap} Elyr={ely}"
        )

    payload = {"records": records, "dumps": [d["id"] for d in dumps]}
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
