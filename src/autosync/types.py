"""Core data types shared across MeloX-AutoSync.

These types are intentionally plain dataclasses so that they serialise cleanly
and do not depend on any model framework.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# Where a timestamp came from. We never disguise a heuristic estimate as a
# model result.
AlignmentSource = Literal["model", "heuristic", "failed"]


@dataclass
class LyricLine:
    """One line of an LRC file (times in seconds, absolute song time)."""

    index: int
    start: float
    text: str
    # Explicit end time from *enhanced* LRC (e.g. ``[00:12.00]<00:15.2>end``).
    # ``None`` means the source LRC only carries line-start times, which is the
    # common case. We must NOT silently substitute the next line's start here;
    # the real end is decided by acoustic alignment.
    end: float | None = None
    # Optional word/segment-level timings parsed from enhanced LRC.
    segments: list[LyricSegment] | None = None


@dataclass
class LyricSegment:
    """A ``<mm:ss.xx>`` segment inside an enhanced LRC line."""

    start: float
    text: str


@dataclass
class AlignUnit:
    """A smallest aligned unit (a word for English, a character for zh/ja)."""

    text: str
    start: float
    end: float
    source: AlignmentSource = "model"
    # Mean token log-probability over the frames assigned to this unit. A rough
    # confidence proxy, not a calibrated probability.
    score: float | None = None
    # Surface form actually displayed to the user, when ``text`` is a reading /
    # normalised form (e.g. Japanese kana reading of a kanji).
    raw: str | None = None
    # ``True`` when the acoustic model produced this unit from a token that
    # covers several original characters (e.g. the Japanese token ``世界``), so
    # the boundary *inside* the unit cannot be recovered exactly. The outer
    # boundaries are still model-derived; downstream consumers should treat the
    # internal split as unknown rather than assuming one char == one syllable.
    uncertain: bool = False


@dataclass
class AlignedLine:
    index: int
    text: str
    start: float
    end: float
    units: list[AlignUnit] = field(default_factory=list)
    source: AlignmentSource = "model"
    error: str | None = None
    # V0.7 quality gate: "reliable" | "uncertain" | "failed" | None (not assessed)
    quality: str | None = None
    quality_reasons: list[str] = field(default_factory=list)
    # V0.8: the ORIGINAL LRC line times, kept separate from the model prediction
    # (``start``/``end``). ``lrc_end`` stays ``None`` for plain LRC without an
    # explicit end time; it must never be fabricated.
    lrc_start: float | None = None
    lrc_end: float | None = None
    # V0.9: non-singing line classification ("singing" | "metadata" |
    # "instrumental" | "unknown" | None for older results).
    kind: str | None = None
    # Why a non-singing line was not aligned (empty for singing lines).
    skip_reason: str | None = None
    # V1.0: seconds of audio actually sent to the model for this line.
    window_seconds: float | None = None
    # V1.0: set when the result must never reach the player word timeline
    # (e.g. the reconstructed text does not match the lyric line).
    force_fallback: bool = False
    # V1.0 mixed-language routing audit trail.
    line_language: str | None = None
    route_model: str | None = None
    route_reason: str | None = None


@dataclass
class ModelInfo:
    name: str
    version: str
    language: str
    license: str = "unknown"
    device: str = "cpu"
    load_seconds: float | None = None
    infer_seconds: float | None = None
    # Duration of one acoustic frame in seconds (the alignment time quantum).
    frame_seconds: float | None = None
    # V0.8: lifecycle status of the backend (stable | experimental | research)
    status: str = "stable"


@dataclass
class AlignmentResult:
    language: str
    engine: str
    lines: list[AlignedLine]
    model: ModelInfo | None = None
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # V0.7: counts of quality states across lines, e.g. {"reliable": 3, ...}
    quality_summary: dict[str, Any] = field(default_factory=dict)
    # V0.9: counts by line kind, e.g. {"singing": 8, "metadata": 3, ...}
    line_kind_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data
