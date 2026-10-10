"""Synthetic multi-line LRC chain tests (V0.7 P2 §6).

Verifies the pipeline-level guarantees a whole song needs, without real audio:
per-line isolation, monotonic non-crossing timelines, recoverable original
lyrics, and that one failing line does not affect the others.
"""

from __future__ import annotations

import numpy as np

from autosync.alignment.segment import SegmentOptions, align_lines
from autosync.lyrics.lrc import parse_lrc_text
from autosync.types import AlignUnit

SR = 16_000


class FakeAligner:
    """Deterministic test double that optionally fails on given lines."""

    frame_seconds = 0.02

    def __init__(self, fail_on: set[str] | None = None, quality: str | None = None) -> None:
        self.fail_on = fail_on or set()
        self.quality = quality
        self.last_quality = None

    def align_text(self, audio, sample_rate, text):
        self.last_quality = (
            {"state": self.quality, "reasons": ["test"]} if self.quality is not None else None
        )
        if text in self.fail_on:
            raise RuntimeError("synthetic failure")
        dur = len(audio) / sample_rate
        return [AlignUnit(text=text, start=0.0, end=dur * 0.4, source="model")]


LRC = """[00:02.00]first line
[00:06.00]second line
[00:11.00]third line
[00:16.00]fourth line
"""


def _align(fail_on=None, quality=None):
    lines, _ = parse_lrc_text(LRC)
    audio = np.zeros(SR * 20, dtype=np.float32)
    aligned, warnings = align_lines(
        audio, SR, lines, FakeAligner(fail_on=fail_on, quality=quality), SegmentOptions(padding=0.3), progress=False
    )
    return lines, aligned, warnings


def test_multiline_text_is_preserved() -> None:
    lines, aligned, _ = _align()
    assert [ln.text for ln in aligned] == [ln.text for ln in lines]
    assert [ln.index for ln in aligned] == [0, 1, 2, 3]


def test_lines_are_monotonic_and_do_not_cross() -> None:
    _, aligned, _ = _align()
    # line starts are ordered
    assert aligned[0].start <= aligned[1].start <= aligned[2].start <= aligned[3].start
    # units of one line stay inside its own window (start-pad .. next.start+pad)
    for i, ln in enumerate(aligned[:-1]):
        next_start = aligned[i + 1].start
        for u in ln.units:
            assert u.start >= ln.start - 0.31
            assert u.end <= next_start + 0.31


def test_one_failing_line_does_not_affect_others() -> None:
    _, aligned, _ = _align(fail_on={"second line"})
    assert aligned[1].source == "failed" and aligned[1].units == []
    for i in (0, 2, 3):
        assert aligned[i].source == "model" and aligned[i].units


def test_quality_state_attached_and_summarized() -> None:
    _, aligned, _ = _align(quality="uncertain")
    assert all(ln.quality == "uncertain" for ln in aligned)


def test_lrc_timestamp_offset_is_honoured() -> None:
    lines, _ = parse_lrc_text("[offset:+500]\n[00:10.00]x")
    audio = np.zeros(SR * 20, dtype=np.float32)
    aligned, _ = align_lines(audio, SR, lines, FakeAligner(), SegmentOptions(), progress=False)
    assert abs(aligned[0].start - 10.5) < 0.31


def _align_text(lrc: str, seconds: float, fail_on=None):
    lines, _ = parse_lrc_text(lrc)
    audio = np.zeros(int(SR * seconds), dtype=np.float32)
    aligned, _ = align_lines(
        audio, SR, lines, FakeAligner(fail_on=fail_on), SegmentOptions(padding=0.3), progress=False
    )
    return lines, aligned


def test_intro_interlude_outro_gaps() -> None:
    lrc = "[00:10.00]intro line\n[00:40.00]verse line\n[00:70.00]outro line\n"
    _lines, aligned = _align_text(lrc, 90)
    starts = [ln.start for ln in aligned]
    assert starts == sorted(starts)
    # each line's units stay in its own window, not the neighbour's
    for i, ln in enumerate(aligned[:-1]):
        nxt = aligned[i + 1].start
        assert all(u.end <= nxt + 0.31 for u in ln.units)


def test_repeated_lyrics_are_each_aligned() -> None:
    lrc = "[00:02.00]same words\n[00:07.00]same words\n[00:12.00]same words\n"
    _, aligned = _align_text(lrc, 20)
    assert [ln.text for ln in aligned] == ["same words"] * 3
    assert all(ln.source == "model" for ln in aligned)


def test_units_are_monotonic_within_each_line() -> None:
    lrc = "[00:01.00]aaa\n[00:05.00]bbb\n[00:09.00]ccc\n"
    _, aligned = _align_text(lrc, 15)
    for ln in aligned:
        last = -1e9
        for u in ln.units:
            assert u.end >= u.start
            assert u.start >= last - 1e-6
            last = u.end


def test_empty_lyric_line_is_safe() -> None:
    lrc = "[00:01.00]good\n[00:04.00]\n[00:07.00]after\n"
    lines, aligned = _align_text(lrc, 12)
    # the pipeline must not crash and must keep every parsed line ordered
    starts = [ln.start for ln in aligned]
    assert starts == sorted(starts)
    assert len(aligned) == len(lines)
