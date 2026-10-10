#!/usr/bin/env python
"""Measure model load time, inference time and peak RSS for one language.

Example:
    uv run --extra funasr python benchmarks/measure_resources.py \
        --language zh --audio experiments/audio/synthetic_zh.wav \
        --lyrics experiments/audio/synthetic_zh.lrc
"""

from __future__ import annotations

import argparse
import resource
import time

from autosync.audio.io import load_audio
from autosync.lyrics.lrc import parse_lrc
from autosync.models.registry import get_aligner


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--language", required=True, choices=["en", "ja", "zh"])
    ap.add_argument("--audio", required=True)
    ap.add_argument("--lyrics", required=True)
    args = ap.parse_args()

    lines, _ = parse_lrc(args.lyrics)
    audio, sr = load_audio(args.audio, 16000)

    tier = get_aligner(args.language)
    t0 = time.time()
    tier.load()
    load_s = time.time() - t0

    t1 = time.time()
    n_units = 0
    for ln in lines:
        n_units += len(tier.align_text(audio, sr, ln.text))
    infer_s = time.time() - t1

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"language={args.language} model={tier.meta.name}")
    print(f"  audio_seconds   = {len(audio) / sr:.2f}")
    print(f"  load_seconds    = {load_s:.3f}")
    print(f"  infer_seconds   = {infer_s:.3f}")
    print(f"  units           = {n_units}")
    print(f"  peak_rss_mb     = {rss_mb:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
