#!/usr/bin/env python
"""Build a synthetic speech clip with *exact* word boundaries (espeak-ng).

Why this exists
---------------
To measure forced-alignment error quantitatively we need reference word
boundaries. Real sung audio with human-annotated boundaries is not available in
this environment (and we must not fabricate results). Synthesising each word
separately and concatenating them with known gaps gives us a clip whose word
onsets/offsets we know *exactly*, so we can honestly report absolute timing
error of the alignment engine.

This is a **synthetic voice** test (espeak-ng). It validates the algorithm and
the plumbing; it is NOT evidence about singing. See docs/EXPERIMENT_RESULTS.md.

It also writes a matching single-line LRC so the whole CLI can be exercised.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SR = 16_000


def _synth_word(word: str, voice: str, sr: int = SR) -> np.ndarray:
    if not shutil.which("espeak-ng"):
        raise SystemExit("espeak-ng not found; install it or provide real audio")
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "w.wav"
        subprocess.run(
            ["espeak-ng", "-v", voice, "-w", str(wav), word], check=True, capture_output=True
        )
        raw = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(wav),
                "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def _silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(round(seconds * sr), dtype=np.float32)


def _fmt(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def build(text: str, out: Path, gap: float, lead: float, voice: str, split: str) -> tuple[Path, Path, Path]:
    if split == "char":
        words = [c for c in text if not c.isspace()]
    else:
        words = text.split()
    if not words:
        raise SystemExit("empty text")
    lead_sil = _silence(lead)
    gap_sil = _silence(gap)

    chunks: list[np.ndarray] = [lead_sil]
    refs: list[dict] = []
    t = len(lead_sil) / SR
    for i, w in enumerate(words):
        s = _synth_word(w, voice)
        start, end = t, t + len(s) / SR
        refs.append({"text": w, "start": round(start, 4), "end": round(end, 4)})
        chunks.append(s)
        t = end
        if i < len(words) - 1:
            chunks.append(gap_sil)
            t += len(gap_sil) / SR

    audio = np.concatenate(chunks)
    import soundfile as sf

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, SR)

    ref_path = out.with_suffix(".ref.json")
    ref_path.write_text(
        json.dumps({"text": text, "sr": SR, "split": split, "voice": voice, "words": refs}, indent=2),
        encoding="utf-8",
    )

    lrc_path = out.with_suffix(".lrc")
    lrc_path.write_text(f"[00:00.00]{text}\n", encoding="utf-8")

    return out, ref_path, lrc_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", default="hello world this is a test")
    ap.add_argument("--out", required=True, help="output wav path")
    ap.add_argument("--gap", type=float, default=0.45, help="silence between words (s)")
    ap.add_argument("--lead", type=float, default=0.50, help="leading silence (s)")
    ap.add_argument("--voice", default="en", help="espeak-ng voice, e.g. en / cmn / ja")
    ap.add_argument("--split", choices=["word", "char"], default="word")
    args = ap.parse_args()
    out, ref, lrc = build(args.text, Path(args.out), args.gap, args.lead, args.voice, args.split)
    print(f"wrote {out}")
    print(f"wrote {ref}")
    print(f"wrote {lrc}")
    print(f"(_fmt check: {_fmt(1.5)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
