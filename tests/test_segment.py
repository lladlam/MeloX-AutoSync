from __future__ import annotations

import numpy as np

from autosync.alignment.segment import SegmentOptions, align_lines
from autosync.types import AlignUnit, LyricLine


class FakeAligner:
    """Deterministic test double (NOT a product implementation)."""

    frame_seconds = 0.02

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()
        self.calls = 0
        self.infer_seconds = 0.0

    def align_text(self, audio, sample_rate, text):
        self.calls += 1
        if text in self.fail_on:
            raise RuntimeError("synthetic failure")
        dur = len(audio) / sample_rate
        # One unit occupying the first half of the window.
        return [AlignUnit(text=text, start=0.0, end=dur * 0.5, source="model", score=-0.2)]


def test_absolute_time_mapping() -> None:
    sr = 16_000
    audio = np.zeros(sr * 10, dtype=np.float32)
    lines = [LyricLine(index=0, start=2.0, text="hello"), LyricLine(index=1, start=6.0, text="world")]
    aligned, warnings = align_lines(
        audio, sr, lines, FakeAligner(), SegmentOptions(padding=0.3), progress=False
    )
    # Line 0 window is [1.7, 6.3]; unit is [0, half] within it.
    u0 = aligned[0].units[0]
    assert abs(u0.start - 1.7) < 1e-6
    assert aligned[0].source == "model"
    # Line 1 (last line) window starts at 6.0 - padding = 5.7 (correct behaviour).
    assert abs(aligned[1].units[0].start - 5.7) < 1e-6
    assert warnings == []


def test_failure_is_reported_not_faked() -> None:
    sr = 16_000
    audio = np.zeros(sr * 5, dtype=np.float32)
    lines = [LyricLine(index=0, start=0.5, text="boom")]
    aligned, _ = align_lines(
        audio, sr, lines, FakeAligner(fail_on={"boom"}), SegmentOptions(), progress=False
    )
    assert aligned[0].source == "failed"
    assert aligned[0].units == []
    assert aligned[0].error and "synthetic failure" in aligned[0].error


def test_fallback_heuristic_marks_source() -> None:
    sr = 16_000
    audio = np.zeros(sr * 5, dtype=np.float32)
    lines = [LyricLine(index=0, start=0.5, text="boom")]
    aligned, _ = align_lines(
        audio,
        sr,
        lines,
        FakeAligner(fail_on={"boom"}),
        SegmentOptions(fallback="heuristic"),
        progress=False,
    )
    assert aligned[0].source == "heuristic"
    assert aligned[0].units
