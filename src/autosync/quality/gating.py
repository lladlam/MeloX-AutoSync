"""Player-facing fallback policy for the three quality states (V0.7 §4).

The gate never overwrites, alters or discards the original lyrics. Given the
per-line quality states it produces a `player_timeline`:

* ``RELIABLE``  -> emit the word-level units;
* ``UNCERTAIN`` -> keep the diagnostic timeline but the player should use the
  original **line-level LRC**;
* ``FAILED``    -> structured failure; player uses the original line-level LRC.

Line-level fallback is used (not per-word) to avoid visual flicker between
word and line display within a line. Per-language policies are independent; only
Japanese uses the calibrated gate by default.
"""

from __future__ import annotations

from typing import Any

from ..types import AlignedLine, AlignmentResult
from .decision import QualityPolicy, QualityState

# Default policy is intentionally conservative and language-scoped. English and
# Chinese are NOT gated (their calibration is out of scope for V0.7).
DEFAULT_POLICIES: dict[str, QualityPolicy | None] = {
    "ja": QualityPolicy(name="ja-hubert-conservative-v07", max_intra_char_span_s=1.5, max_char_duration_ratio=10.0),
    "en": None,
    "zh": None,
}


def summarize(lines: list[AlignedLine]) -> dict[str, int]:
    counts = {s.value: 0 for s in QualityState}
    counts["not_assessed"] = 0
    for line in lines:
        if line.quality in counts:
            counts[line.quality] += 1
        else:
            counts["not_assessed"] += 1
    return counts


def player_timeline(result: AlignmentResult) -> list[dict[str, Any]]:
    """Build the player timeline, falling back to line LRC for non-reliable lines."""
    out: list[dict[str, Any]] = []
    for line in result.lines:
        # ``None`` means "not assessed" (e.g. en/zh are not gated): keep word mode.
        reliable = line.quality in (None, QualityState.RELIABLE.value)
        entry: dict[str, Any] = {
            "index": line.index,
            # original lyrics are always preserved
            "text": line.text,
            "line_start": line.start,
            "line_end": line.end,
            "mode": "word" if reliable and line.units else "line_lrc",
            "quality": line.quality or "not_assessed",
            "reasons": line.quality_reasons,
        }
        if entry["mode"] == "word":
            entry["units"] = [
                {"text": u.text, "start": u.start, "end": u.end, "source": u.source} for u in line.units
            ]
        out.append(entry)
    return out


def gate_result(result: AlignmentResult, policy: QualityPolicy | None = None) -> AlignmentResult:
    """Attach the quality summary (and, if given, apply a policy to None-states)."""
    result.quality_summary = summarize(result.lines)
    if policy is not None:
        for line in result.lines:
            if line.quality is None:
                # policy-gated but not assessed -> be conservative
                line.quality = QualityState.UNCERTAIN.value
                line.quality_reasons = ["not assessed; defaulted to uncertain"]
    return result
