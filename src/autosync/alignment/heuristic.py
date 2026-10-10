"""Heuristic baselines.

These do **not** use any acoustic model. Results produced here are always
labelled ``source="heuristic"`` and must never be presented as model output.
The main use is (a) a sanity baseline for error comparison and (b) graceful
degradation when a model is unavailable.
"""

from __future__ import annotations

from ..types import AlignUnit


def distribute(start: float, end: float, units: list[str]) -> list[AlignUnit]:
    """Split ``[start, end]`` evenly across ``units`` by weight.

    Weight is proportional to unit length (number of characters), so long words
    get proportionally more time. A minimum positive duration is enforced so
    the output is monotonically increasing.
    """
    if not units:
        return []
    if end < start:
        start, end = end, start
    weights = [max(1, len(u)) for u in units]
    total_w = sum(weights)
    span = end - start
    out: list[AlignUnit] = []
    cursor = start
    for unit, w in zip(units, weights):
        dur = span * (w / total_w)
        out.append(
            AlignUnit(text=unit, start=cursor, end=cursor + dur, source="heuristic", score=None)
        )
        cursor += dur
    return out
