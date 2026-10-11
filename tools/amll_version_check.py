#!/usr/bin/env python
"""Check whether an AMLL TTML annotation matches a local recording before use.

The AMLL TTML database and a local MP3 are not guaranteed to be the same edit.
Rather than assuming they match (or silently reusing an unconfirmed reference),
this tool estimates the **global time offset** between the TTML line starts and
the audio's own vocal onsets and reports how well they line up *after* that
single offset.  A high onset-hit rate at a small offset is evidence that the
annotations describe this recording; a low rate is evidence they do not.

It never modifies the audio or the reference, and it prints no lyric text.

Usage::

    .venv/bin/python tools/amll_version_check.py \
        --ttml experiments/amll_reference/2755332551.ttml \
        --audio "Test/DAMIDAMI.mp3" --out /tmp/amll_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _clock(value: str) -> float | None:
    """Parse AMLL clocks: ``HH:MM:SS.mmm``, ``MM:SS.mmm`` or plain seconds."""
    v = value.strip()
    if not v:
        return None
    parts = v.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return None
    return None


def ttml_line_starts(path: Path) -> list[float]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(path.read_text(encoding="utf-8"))
    starts: list[float] = []
    # match by local name: AMLL files often reset xmlns="" on inner elements
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] != "p":
            continue
        t = _clock(el.get("begin", ""))
        if t is not None:
            starts.append(t)
    return sorted(starts)


def onset_envelope(audio, sr: int, hop: float = 0.02):
    import numpy as np

    n = int(sr * hop)
    frames = len(audio) // n
    if frames < 4:
        return np.zeros(0, dtype=np.float32), hop
    x = audio[: frames * n].reshape(frames, n)
    energy = np.log1p((x ** 2).sum(axis=1))
    flux = np.diff(energy, prepend=energy[:1])
    flux = np.maximum(flux, 0.0)
    if flux.max() > 0:
        flux = flux / flux.max()
    return flux.astype(np.float32), hop


def check(ttml: Path, audio_path: Path, max_offset: float = 30.0) -> dict:
    from math import gcd

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if sr != 16000:
        g = gcd(16000, sr)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        sr = 16000
    env, hop = onset_envelope(x, sr)
    starts = ttml_line_starts(ttml)
    if env.size == 0 or not starts:
        return {"error": "no envelope or no ttml starts"}

    # impulse train of ttml starts, then scan a single global offset
    imp = np.zeros(env.size, dtype=np.float32)
    for s in starts:
        i = round(s / hop)
        if 0 <= i < imp.size:
            imp[i] = 1.0
    best = (None, -1.0)
    rng = round(max_offset / hop)
    for lag in range(-rng, rng + 1):
        shifted = np.roll(imp, lag)
        score = float((shifted * env).sum() / max(1.0, shifted.sum()))
        if score > best[1]:
            best = (lag * hop, score)

    offset, score = best
    # onset hit rate: for each ttml start, is there an onset within +-0.4 s after offset?
    hits = 0
    troughs = 0
    for s in starts:
        t = s + offset
        i = round(t / hop)
        lo, hi = max(0, i - 2), min(env.size, i + 20)
        if hi > lo and env[lo:hi].max() >= 0.15:
            hits += 1
        else:
            troughs += 1
    return {
        "ttml": ttml.name, "audio": audio_path.name,
        "n_ttml_lines": len(starts),
        "estimated_offset_s": round(offset, 3),
        "onset_score": round(score, 4),
        "onset_hit_rate": round(hits / max(1, len(starts)), 3),
        "misses": troughs,
        "verdict": ("CONSISTENT" if hits / max(1, len(starts)) >= 0.75 and abs(offset) <= max_offset
                    else "INCONSISTENT_OR_VERSION_MISMATCH"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ttml", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    res = check(args.ttml, args.audio)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
