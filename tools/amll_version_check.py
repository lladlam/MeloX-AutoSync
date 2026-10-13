#!/usr/bin/env python
"""Audit whether an AMLL TTML annotation matches a local recording.

V1.2 estimated a single global offset from broadband energy and printed
"CONSISTENT". That is unsafe: drum transients are not vocal onsets, ``np.roll``
could wrap around and produce phantom matches, and a high onset-hit rate alone
does not prove the same recording. This version addresses each flaw and, most
importantly, **never auto-declares a match**:

* onsets are measured on a **vocal band** (200-4000 Hz) spectral flux, and a
  harmonic/percussive proxy is reported so a purely percussive offset is visible;
* the offset scan uses a **valid overlap only** (no circular wraparound);
* three **negative controls** are run: a different song, random offsets, and a
  time-shuffled copy of the annotation. The best real score must beat all of
  them, otherwise the verdict is ``MISMATCH``;
* the offset is estimated **per song third** and its spread is reported, so a
  single lucky global peak cannot masquerade as a match;
* the verdict is ``UNVERIFIED`` unless a human explicitly confirms it with
  ``--human-verified`` (statuses: ``VERIFIED`` / ``UNVERIFIED`` / ``MISMATCH``).

The original TTML and audio are never modified. No lyric text is printed.

Usage::

    .venv/bin/python tools/amll_version_check.py \
        --ttml experiments/amll_reference/2750140001.ttml \
        --audio "Test/提瓦特民谣.mp3" \
        --control-audio "Test/红透晚烟青.mp3" \
        --out /tmp/amll_audit.json
    # a human who listened and agrees adds:
    ... --human-verified --human-note "confirmed lead-in 5.3s by ear"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SR = 16000
HOP = 0.02
VOCAL_LO = 200.0
VOCAL_HI = 4000.0


# --------------------------------------------------------------------------- ttml
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


# --------------------------------------------------------------------------- audio
def load_mono16k(path: Path):
    from math import gcd

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def _bandpass(x, lo: float, hi: float):
    """Zero-phase band-pass using an FFT mask (no transient smearing)."""
    import numpy as np

    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1 / SR)
    spec[(freqs < lo) | (freqs > hi)] = 0
    return np.fft.irfft(spec, n=x.size).astype(np.float32)


def vocal_onset_envelope(x, hop_seconds: float = HOP):
    """Half-wave rectified spectral flux of the vocal band, in [0, 1]."""
    import numpy as np

    band = _bandpass(x, VOCAL_LO, VOCAL_HI)
    n = int(SR * hop_seconds)
    frames = band.size // n
    if frames < 4:
        return np.zeros(0, dtype=np.float32), hop_seconds
    fr = band[: frames * n].reshape(frames, n)
    # spectral flux (magnitude spectra), which is more onset-like than raw energy
    mag = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1))
    flux = np.diff(mag, axis=0)
    flux = np.maximum(flux, 0.0).sum(axis=1)
    flux = np.concatenate([[0.0], flux]).astype(np.float32)
    if flux.max() > 0:
        flux = flux / flux.max()
    return flux, hop_seconds


def _percussive_ratio(x) -> float:
    """Rough percussive share: HF (4-8k) energy vs vocal band (0.2-4k).

    Purely percussive material scores high; a vocal-only take scores low. Used
    only to *explain* an offset, never to accept a match.
    """
    import numpy as np

    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, 1 / SR)
    low = spec[(freqs >= VOCAL_LO) & (freqs <= VOCAL_HI)].sum()
    high = spec[(freqs > VOCAL_HI) & (freqs <= 8000)].sum()
    return float(high / (low + high)) if (low + high) > 0 else 0.0


# --------------------------------------------------------------------------- scan
def best_offset(env, starts, max_offset: float, hop: float, jitter: float = 0.05):
    """Best global lag using a *valid overlap* only (no wraparound).

    Returns ``(offset_s, score)`` where score is the mean envelope value at the
    TTML starts shifted by the offset, plus a small tolerance window.
    """
    import numpy as np

    if env.size == 0 or not starts:
        return None, 0.0
    idx = np.array([round(s / hop) for s in starts], dtype=int)
    jit = max(1, round(jitter / hop))
    rng = round(max_offset / hop)
    n_env = env.size
    best = (0.0, -1.0)
    for lag in range(-rng, rng + 1):
        s = idx + lag
        s = s[(s >= 0) & (s < n_env)]  # valid overlap only
        if s.size < max(3, len(idx) // 2):
            continue
        win = np.arange(-jit, jit + 1)
        vals = env[np.clip(s[:, None] + win[None, :], 0, n_env - 1)].max(axis=1)
        score = float(vals.mean())
        if score > best[1]:
            best = (lag * hop, score)
    return best


def _random_offset_scores(env, starts, max_offset: float, hop: float, n: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    n_env = env.size
    jit = max(1, round(0.05 / hop))
    scores = []
    for _ in range(n):
        lag = int(rng.integers(-round(max_offset / hop), round(max_offset / hop) + 1))
        idx = np.array([round(s / hop) for s in starts], dtype=int) + lag
        idx = idx[(idx >= 0) & (idx < n_env)]
        if idx.size < 3:
            continue
        win = np.arange(-jit, jit + 1)
        vals = env[np.clip(idx[:, None] + win[None, :], 0, n_env - 1)].max(axis=1)
        scores.append(float(vals.mean()))
    return scores


def _shuffled_score(env, starts, max_offset: float, hop: float, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    dur = env.size * hop
    shuffled = np.sort(rng.uniform(0.0, dur, size=len(starts)))
    _, score = best_offset(env, list(shuffled), max_offset, hop)
    return score


def offset_stability(env, starts, max_offset: float, hop: float, parts: int = 3) -> dict:
    """Estimate the offset independently on each song third."""
    import numpy as np

    if len(starts) < parts * 3:
        return {"n_parts": 0, "offsets": [], "spread_s": None}
    chunks = np.array_split(np.array(starts), parts)
    offsets = []
    for ch in chunks:
        off, score = best_offset(env, list(ch), max_offset, hop)
        if off is not None and score > 0:
            offsets.append(round(float(off), 3))
    spread = (max(offsets) - min(offsets)) if len(offsets) >= 2 else None
    return {"n_parts": len(offsets), "offsets": offsets, "spread_s": spread}


# --------------------------------------------------------------------------- main
def audit(ttml: Path, audio: Path, control_audio: Path | None = None,
          max_offset: float = 15.0, seed: int = 0,
          human_verified: bool = False, human_note: str = "") -> dict:
    import numpy as np

    starts = ttml_line_starts(ttml)
    x = load_mono16k(audio)
    env, hop = vocal_onset_envelope(x)
    offset, score = best_offset(env, starts, max_offset, hop)

    rand = _random_offset_scores(env, starts, max_offset, hop, 200, seed)
    rand_p95 = float(np.percentile(rand, 95)) if rand else 0.0
    shuf = _shuffled_score(env, starts, max_offset, hop, seed + 1)

    wrong = None
    if control_audio is not None and Path(control_audio).exists():
        cx = load_mono16k(Path(control_audio))
        cenv, ch = vocal_onset_envelope(cx, hop)
        _, wrong = best_offset(cenv, starts, max_offset, ch)

    stab = offset_stability(env, starts, max_offset, hop)

    # ---- verdict: never auto-VERIFIED --------------------------------
    beats_controls = (
        offset is not None
        and score > rand_p95
        and score > shuf
        and (wrong is None or score > wrong)
    )
    stable = stab["spread_s"] is not None and stab["spread_s"] <= 0.5
    if not beats_controls:
        status = "MISMATCH"
        why = "best offset does not beat the random / shuffled / wrong-song controls"
    elif not stable:
        status = "UNVERIFIED"
        why = f"offset is not stable across song thirds (spread {stab['spread_s']}s)"
    elif human_verified:
        status = "VERIFIED"
        why = "statistics pass and a human confirmed the same recording"
    else:
        status = "UNVERIFIED"
        why = "statistics pass but no human confirmation yet (listening still required)"

    return {
        "ttml": ttml.name,
        "audio": audio.name,
        "n_ttml_lines": len(starts),
        "max_offset_s": max_offset,
        "best_offset_s": offset,
        "best_onset_score": round(score, 4),
        "control_random_p95": round(rand_p95, 4),
        "control_shuffled_score": round(shuf, 4),
        "control_wrong_song_score": (round(wrong, 4) if wrong is not None else None),
        "beats_all_controls": bool(beats_controls),
        "offset_stability": stab,
        "percussive_ratio": round(_percussive_ratio(x), 3),
        "status": status,
        "reason": why,
        "human_verified": bool(human_verified),
        "human_note": human_note,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ttml", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--control-audio", type=Path, default=None,
                    help="a *different* song, used as the wrong-song negative control")
    ap.add_argument("--max-offset", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--human-verified", action="store_true",
                    help="set only after a human listened and confirmed same version")
    ap.add_argument("--human-note", default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    res = audit(args.ttml, args.audio, args.control_audio, args.max_offset,
                args.seed, args.human_verified, args.human_note)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
