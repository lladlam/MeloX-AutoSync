"""V1.0 P1 tests: window bounding, heuristic absolute times, reconstruction fallback."""

from __future__ import annotations

import numpy as np

from autosync.alignment.segment import SegmentOptions, align_lines
from autosync.lyrics.lrc import parse_lrc_text
from autosync.types import AlignUnit

SR = 16_000


class WindowRecorder:
    frame_seconds = 0.02

    def __init__(self, text_transform=None, boom: bool = False) -> None:
        self.windows: list[float] = []
        self.text_transform = text_transform
        self.boom = boom
        self.last_quality = None

    def align_text(self, audio, sample_rate, text):
        self.windows.append(len(audio) / sample_rate)
        if self.boom:
            raise RuntimeError("synthetic failure")
        out = text if self.text_transform is None else self.text_transform(text)
        dur = len(audio) / sample_rate
        return [AlignUnit(text=out, start=0.0, end=min(dur, 0.4), source="model")]


def _tone(seconds: float, start: float = 0.0, end: float | None = None) -> np.ndarray:
    a = np.zeros(int(SR * seconds), dtype=np.float32)
    i0, i1 = int(start * SR), int((end if end is not None else seconds) * SR)
    t = np.arange(i1 - i0) / SR
    a[i0:i1] = 0.3 * np.sin(2 * np.pi * 220 * t)
    return a


def test_long_interlude_is_capped() -> None:
    # line at 0, next line 45s later, audio silent in between
    lines, _ = parse_lrc_text("[00:00.00]first\n[00:45.00]second\n")
    audio = _tone(60, 0, 2)  # only 2s of singing, then silence
    rec = WindowRecorder()
    aligned, _ = align_lines(audio, SR, lines, rec, SegmentOptions(), progress=False)
    assert rec.windows[0] <= 12.5, rec.windows  # capped at max_line_seconds
    assert aligned[0].window_seconds is not None and aligned[0].window_seconds <= 12.5


def test_long_tail_after_last_line_is_capped() -> None:
    lines, _ = parse_lrc_text("[00:00.00]only line\n")
    audio = _tone(120, 0, 2)
    rec = WindowRecorder()
    align_lines(audio, SR, lines, rec, SegmentOptions(), progress=False)
    assert rec.windows[0] <= 12.5


def test_long_held_note_extends_beyond_cap() -> None:
    # a held note runs from 1s to 20s; the window must extend past the 12s cap
    lines, _ = parse_lrc_text("[00:00.00]held\n[00:40.00]next\n")
    audio = _tone(60, 1, 20)
    rec = WindowRecorder()
    align_lines(audio, SR, lines, rec, SegmentOptions(), progress=False)
    assert 12.4 < rec.windows[0] <= 30.5, rec.windows


def test_heuristic_absolute_times_are_not_shifted() -> None:
    lines, _ = parse_lrc_text("[00:10.00]failed line\n[00:20.00]next\n")
    audio = np.zeros(SR * 30, dtype=np.float32)  # silence -> model returns nothing
    rec = WindowRecorder(boom=False)
    rec.align_text = lambda audio, sr, text: []  # force heuristic fallback path
    aligned, _ = align_lines(
        audio, SR, lines, rec, SegmentOptions(fallback="heuristic"), progress=False
    )
    units = aligned[0].units
    assert units and units[0].source == "heuristic"
    # must start at the original LRC time (10.0), not 10.0 - padding
    assert abs(units[0].start - 10.0) < 0.02, units[0].start
    assert units[-1].end <= 20.0 + 1e-6


def test_reconstruction_failure_forces_line_fallback() -> None:
    lines, _ = parse_lrc_text("[00:01.00]hello world\n")
    audio = _tone(5, 0, 4)
    # model returns a unit whose text does not reconstruct the lyric
    rec = WindowRecorder(text_transform=lambda _t: "WRONG")
    aligned, warnings = align_lines(
        audio, SR, lines, rec, SegmentOptions(), progress=False, language="en"
    )
    assert aligned[0].force_fallback is True
    assert any("reconstruction" in w for w in warnings)

    from autosync.quality.gating import player_timeline
    from autosync.types import AlignmentResult

    result = AlignmentResult(language="en", engine="model", lines=aligned)
    tl = player_timeline(result)
    assert tl[0]["mode"] == "line_lrc"  # never reaches the word timeline


def test_no_reconstruction_issue_keeps_word_mode() -> None:
    lines, _ = parse_lrc_text("[00:01.00]hello world\n")
    audio = _tone(5, 0, 4)
    rec = WindowRecorder()
    aligned, _ = align_lines(
        audio, SR, lines, rec, SegmentOptions(), progress=False, language="en"
    )
    assert aligned[0].force_fallback is False
