"""Energy-based local re-segmentation (V0.3 P1 mitigation).

Hypothesis (to be tested, not assumed): a long a-cappella window with leading /
trailing accompaniment and internal breath silences makes the CTC aligner spread
one token over a large region ("collapse"). Splitting the window at silence
valleys and aligning each *part* of the lyric to its region shortens the DP
problem and may avoid collapse.

The text split between regions is **proportional to character count**, which is a
heuristic: it can be wrong, and it is reported as such. Regions are found from
audio energy only (no ground truth).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..lyrics.normalize import display_units
from ..types import AlignUnit
from .refine import _frame_db


def voiced_regions(
    audio: np.ndarray,
    sample_rate: int,
    frame_seconds: float = 0.02,
    floor_percentile: float = 20.0,
    rel_db: float = 8.0,
    abs_floor_db: float = -50.0,
    min_speech: float = 0.15,
    min_gap: float = 0.12,
) -> list[tuple[float, float]]:
    """Contiguous voiced regions ``(start, end)`` in seconds (audio-only)."""
    db, fs = _frame_db(audio, sample_rate, frame_seconds)
    if db.size == 0:
        return [(0.0, len(audio) / sample_rate)]
    threshold = max(float(np.percentile(db, floor_percentile)) + rel_db, abs_floor_db)
    voiced = db > threshold

    raw: list[list[int]] = []
    i, n = 0, len(voiced)
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            raw.append([i, j])
            i = j
        else:
            i += 1
    if not raw:
        return [(0.0, len(audio) / sample_rate)]

    merged: list[list[int]] = []
    for r in raw:
        if merged and (r[0] - merged[-1][1]) * fs < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r[:])
    regions = [(r[0] * fs, r[1] * fs) for r in merged if (r[1] - r[0]) * fs >= min_speech]
    return regions or [(float(merged[0][0]) * fs, float(merged[-1][1]) * fs)]


def split_units_evenly(units: list[str], k: int) -> list[list[str]]:
    """Split ``units`` into ``k`` contiguous groups with balanced counts."""
    if k <= 0:
        return []
    if k >= len(units):
        return [[u] for u in units]
    # Balance by cumulative length so groups are similar in size.
    total = len(units)
    groups: list[list[str]] = []
    start = 0
    for g in range(k):
        end = round(total * (g + 1) / k)
        groups.append(units[start:end])
        start = end
    return [g for g in groups if g]


def resegment_align(
    aligner,
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    language: str,
    pad: float = 0.15,
) -> list[AlignUnit]:
    """Align ``text`` to ``audio`` by splitting at energy valleys.

    Falls back to a single alignment when fewer than two regions are found.
    Times are returned relative to the start of ``audio``.
    """
    from ..audio.io import crop

    regions = voiced_regions(audio, sample_rate)
    units_text = display_units(language, text)
    if len(regions) < 2 or not units_text:
        return aligner.align_text(audio, sample_rate, text)

    groups = split_units_evenly(units_text, len(regions))
    if len(groups) != len(regions):
        # Cannot split text one-to-one with regions; do not drop text.
        return aligner.align_text(audio, sample_rate, text)

    out: list[AlignUnit] = []
    for region, group in zip(regions, groups):
        if not group:
            continue
        part = " ".join(group) if language == "en" else "".join(group)
        w0 = max(0.0, region[0] - pad)
        w1 = min(len(audio) / sample_rate, region[1] + pad)
        window = crop(audio, sample_rate, w0, w1)
        if window.size < int(0.05 * sample_rate):
            continue
        units = aligner.align_text(window, sample_rate, part)
        for u in units:
            out.append(replace(u, start=u.start + w0, end=u.end + w0))
    return out
