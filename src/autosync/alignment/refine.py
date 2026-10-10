"""V0.2 boundary refinement: energy-aware adjustment of CTC unit boundaries.

Motivation
----------
CTC frame alignment has two characteristic errors on singing:

* the **end** of a unit stops early, because the trailing vowel / held note is
  absorbed by the CTC blank label;
* the **start** can land inside leading silence, because the acoustic model
  spreads probability before the sung onset.

This module corrects both using the raw waveform energy:

* **start snapping** - if a unit starts in an unvoiced frame, move its start
  forward to the next voiced frame (bounded by ``max_start_shift`` and by the
  unit's own end).
* **end extension** - extend a unit's end over contiguous voiced frames up to
  the start of the next unit (bounded by ``max_extend``).

It never invents time beyond real audio energy and never reorders units. It is
opt-in (``--refine energy``); whether it helps is decided by the singing
evaluation, not by assumption.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..types import AlignUnit

_MIN_DURATION = 0.02


def _frame_db(audio: np.ndarray, sample_rate: int, frame_seconds: float) -> tuple[np.ndarray, float]:
    frame = max(1, round(frame_seconds * sample_rate))
    n = len(audio) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float64), frame / sample_rate
    block = audio[: n * frame].astype(np.float64).reshape(n, frame)
    rms = np.sqrt(np.mean(block**2, axis=1))
    db = 20.0 * np.log10(rms + 1e-9)
    return db, frame / sample_rate


def refine_units_energy(
    units: list[AlignUnit],
    audio: np.ndarray,
    sample_rate: int,
    frame_seconds: float = 0.02,
    max_extend: float = 0.6,
    floor_percentile: float = 20.0,
    rel_db: float = 8.0,
    abs_floor_db: float = -50.0,
    max_gap_frames: int = 2,
    snap_starts: bool = False,
    max_start_shift: float = 0.6,
) -> list[AlignUnit]:
    """Return a copy of ``units`` with energy-refined boundaries (seconds).

    ``snap_starts`` defaults to ``False``: on the GTSinger evaluation start
    snapping *increased* start error for both English and Chinese (see
    docs/EXPERIMENT_RESULTS.md), so only end-extension is enabled by default.
    The capability is kept for reproducible experiments.
    """
    if not units:
        return units
    db, fs = _frame_db(audio, sample_rate, frame_seconds)
    if db.size == 0:
        return units
    floor = float(np.percentile(db, floor_percentile))
    threshold = max(floor + rel_db, abs_floor_db)
    voiced = db > threshold
    total = len(audio) / sample_rate

    starts = [u.start for u in units]

    if snap_starts:
        max_steps = max(1, round(max_start_shift / fs))
        for i, unit in enumerate(units):
            f0 = round(unit.start / fs)
            f0 = max(0, min(f0, len(voiced) - 1))
            if not voiced[f0]:
                j, steps = f0, 0
                while j < len(voiced) and steps < max_steps and not voiced[j]:
                    j += 1
                    steps += 1
                if j < len(voiced) and voiced[j]:
                    candidate = j * fs
                    upper = max(starts[i], unit.end - _MIN_DURATION)
                    starts[i] = min(max(candidate, starts[i]), upper)
            if i and starts[i] < starts[i - 1]:
                starts[i] = starts[i - 1]

    max_frames = max(1, round(max_extend / fs))
    out: list[AlignUnit] = []
    for i, unit in enumerate(units):
        limit = starts[i + 1] if i + 1 < len(units) else total
        f_start = round(unit.end / fs)
        last_voiced = f_start
        gap = 0
        j = f_start
        steps = 0
        while j < len(voiced) and steps < max_frames and (j * fs) < limit:
            if voiced[j]:
                last_voiced = j + 1
                gap = 0
            else:
                gap += 1
                if gap > max_gap_frames:
                    break
            j += 1
            steps += 1
        new_end = max(unit.end, min(limit, last_voiced * fs))
        new_end = max(new_end, starts[i] + _MIN_DURATION)
        new_end = min(new_end, limit) if limit > starts[i] + _MIN_DURATION else new_end
        out.append(replace(unit, start=round(starts[i], 6), end=round(new_end, 6)))
    return out
