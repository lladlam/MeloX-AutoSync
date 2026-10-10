"""Fallback / timestamp-contract tests (V0.8 P0.2, P0.3, P1)."""

from __future__ import annotations

import numpy as np

from autosync.alignment.segment import SegmentOptions, align_lines
from autosync.lyrics.lrc import parse_lrc_text
from autosync.quality.gating import DEFAULT_POLICIES, gate_result, player_timeline
from autosync.types import AlignUnit

SR = 16_000
LRC = "[00:02.00]line one\n[00:06.00]line two\n"


class FakeAligner:
    """Model double whose predicted start differs from the LRC start."""

    frame_seconds = 0.02

    def __init__(self, quality: str | None) -> None:
        self._q = quality
        self.last_quality = None

    def align_text(self, audio, sample_rate, text):
        self.last_quality = {"state": self._q, "reasons": ["test"]} if self._q else None
        dur = len(audio) / sample_rate
        # relative to the (padded) window; the caller adds the window start
        return [AlignUnit(text=text, start=0.25, end=0.25 + dur * 0.3, source="model")]


def _run(quality: str | None, language: str = "ja", mode: str = "production"):
    lines, _ = parse_lrc_text(LRC)
    audio = np.zeros(SR * 10, dtype=np.float32)
    aligned, _ = align_lines(
        audio, SR, lines, FakeAligner(quality), SegmentOptions(padding=0.3), progress=False
    )
    from autosync.types import AlignmentResult

    result = AlignmentResult(language=language, engine="model", lines=aligned)
    gate_result(result, DEFAULT_POLICIES[language], mode=mode)
    return result, player_timeline(result)


def test_lrc_times_preserved_independently_of_model_prediction() -> None:
    _result, tl = _run("reliable")
    # original LRC start preserved exactly
    assert tl[0]["lrc_start"] == 2.0
    assert tl[1]["lrc_start"] == 6.0
    # plain LRC has no explicit end -> null, never fabricated
    assert tl[0]["lrc_end"] is None
    # model prediction is kept separately and differs from the LRC start
    assert tl[0]["model_start"] is not None
    assert abs(tl[0]["model_start"] - 2.0) > 0.01


def test_derived_display_end_is_marked() -> None:
    _, tl = _run("reliable")
    assert tl[0]["display_end"] == 6.0
    assert tl[0]["display_end_source"] == "derived_from_next_lrc_start"
    assert tl[1]["display_end"] is None  # last line: nothing to derive


def test_uncertain_falls_back_to_original_lrc() -> None:
    _result, tl = _run("uncertain")
    assert tl[0]["mode"] == "line_lrc" and "units" not in tl[0]
    assert tl[0]["text"] == "line one"
    assert tl[0]["lrc_start"] == 2.0  # original time, not the model time


def test_failed_falls_back_and_is_counted() -> None:
    result, tl = _run("failed")
    assert tl[0]["mode"] == "line_lrc"
    assert result.quality_summary["failed"] == 2


def test_not_assessed_is_safe_fallback_when_gated() -> None:
    # quality analysis returned None but the language is gated -> must not pass
    result, tl = _run(None, language="ja")
    assert all(ln.quality == "uncertain" for ln in result.lines)
    assert all(e["mode"] == "line_lrc" for e in tl)
    assert result.quality_summary["not_assessed"] == 0


def test_not_gated_language_keeps_word_mode_when_not_assessed() -> None:
    result, tl = _run(None, language="en")
    assert all(e["mode"] == "word" for e in tl)
    assert result.quality_summary["not_assessed"] == 2


def test_experimental_mode_allows_diagnostic_uncertain() -> None:
    _, tl = _run("uncertain", language="ja", mode="experimental")
    assert tl[0]["mode"] == "word"
    assert tl[0]["quality"] == "uncertain"


def test_original_text_recoverable_from_timeline() -> None:
    _, tl = _run("uncertain")
    assert [e["text"] for e in tl] == ["line one", "line two"]
