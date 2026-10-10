"""Audio loading, cropping and (real) resampling.

Design notes
------------
* Resampling is always *real* resampling. We never just relabel a sample rate.
* When ``ffmpeg`` is available we let it decode and resample (swresample /
  soxr quality). Otherwise we fall back to ``soundfile`` + polyphase resampling.
* Everything is mono float32 in memory.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

DEFAULT_SAMPLE_RATE = 16_000


class AudioLoadError(RuntimeError):
    """Raised when audio cannot be decoded."""


def _load_with_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - env dependent
        stderr = exc.stderr.decode("utf-8", "replace")[-2000:]
        raise AudioLoadError(f"ffmpeg failed to decode {path}: {stderr}") from exc
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise AudioLoadError(f"ffmpeg produced empty output for {path}")
    return np.ascontiguousarray(audio)


def _load_with_soundfile(path: Path, sample_rate: int) -> np.ndarray:
    import soundfile as sf  # local import: optional fallback path

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1)  # downmix to mono
    if sr != sample_rate:
        audio = resample_poly(audio, sample_rate, sr)
    return np.ascontiguousarray(audio.astype(np.float32))


def resample_poly(audio: np.ndarray, new_rate: int, old_rate: int) -> np.ndarray:
    """Polyphase resampling. Uses scipy when available, otherwise a small
    windowed-sinc implementation. Never merely relabels the sample rate."""
    if new_rate == old_rate:
        return audio
    try:
        from math import gcd

        from scipy.signal import resample_poly as sp_resample_poly

        g = gcd(int(new_rate), int(old_rate))
        return sp_resample_poly(audio, new_rate // g, old_rate // g).astype(np.float32)
    except Exception:  # noqa: BLE001 - scipy is optional; fall back to linear
        return _resample_linear(audio, new_rate, old_rate)


def _resample_linear(audio: np.ndarray, new_rate: int, old_rate: int) -> np.ndarray:
    """Fallback linear-interpolation resampler.

    Lower quality than polyphase; only used when scipy is unavailable. It still
    changes the number of samples, i.e. it performs a real resample.
    """
    n_out = round(len(audio) * new_rate / old_rate)
    if n_out <= 0:
        return audio[:0]
    old_idx = np.linspace(0.0, len(audio) - 1, num=n_out, dtype=np.float64)
    return np.interp(old_idx, np.arange(len(audio)), audio).astype(np.float32)


def load_audio(
    path: str | Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 at ``target_sr``.

    Returns ``(samples, sample_rate)``.
    """
    path = Path(path)
    if not path.exists():
        raise AudioLoadError(f"audio file not found: {path}")

    if shutil.which("ffmpeg"):
        return _load_with_ffmpeg(path, target_sr), target_sr
    return _load_with_soundfile(path, target_sr), target_sr


def crop(
    audio: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    pad: float = 0.0,
) -> np.ndarray:
    """Return the sub-signal ``[start-pad, end+pad]`` clamped to the array.

    Times are seconds. ``pad`` expands the window symmetrically (useful margin
    around a lyric line so the model sees natural onsets/offsets).
    """
    if end < start:
        start, end = end, start
    left = max(0.0, start - pad)
    right = max(0.0, end + pad)
    i0 = round(left * sample_rate)
    i1 = round(right * sample_rate)
    i0 = max(0, min(i0, len(audio)))
    i1 = max(i0, min(i1, len(audio)))
    return audio[i0:i1]


def duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    return len(audio) / float(sample_rate)
