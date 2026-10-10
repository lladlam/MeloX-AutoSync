"""Tests for energy-based local re-segmentation (V0.3 P1)."""

from __future__ import annotations

import numpy as np

from autosync.alignment.resegment import (
    resegment_align,
    split_units_evenly,
    voiced_regions,
)
from autosync.types import AlignUnit

SR = 16_000


class _StubAligner:
    frame_seconds = 0.02

    def __init__(self) -> None:
        self.calls: list[str] = []

    def align_text(self, audio, sample_rate, text):
        self.calls.append(text)
        dur = len(audio) / sample_rate
        return [AlignUnit(text=text, start=0.0, end=dur, source="model")]


def _two_tone_audio() -> np.ndarray:
    t = np.arange(int(3.0 * SR)) / SR
    audio = np.zeros_like(t, dtype=np.float32)
    seg1 = slice(int(0.2 * SR), int(1.0 * SR))
    seg2 = slice(int(1.6 * SR), int(2.6 * SR))
    audio[seg1] = 0.3 * np.sin(2 * np.pi * 220 * t[seg1])
    audio[seg2] = 0.3 * np.sin(2 * np.pi * 220 * t[seg2])
    return audio.astype(np.float32)


def test_voiced_regions_finds_two() -> None:
    regions = voiced_regions(_two_tone_audio(), SR)
    assert len(regions) == 2
    assert regions[0][0] < 0.4 and regions[1][0] > 1.4


def test_voiced_regions_silence_single() -> None:
    regions = voiced_regions(np.zeros(int(2.0 * SR), dtype=np.float32), SR)
    assert len(regions) == 1


def test_split_units_evenly() -> None:
    assert split_units_evenly(list("abcd"), 2) == [["a", "b"], ["c", "d"]]
    assert split_units_evenly(["a"], 3) == [["a"]]


def test_resegment_splits_and_preserves_text() -> None:
    stub = _StubAligner()
    units = resegment_align(stub, _two_tone_audio(), SR, "你好世界", "zh")
    assert len(stub.calls) >= 2
    assert "".join(u.text for u in units) == "你好世界"


def test_resegment_falls_back_on_silence() -> None:
    stub = _StubAligner()
    units = resegment_align(stub, np.zeros(int(2.0 * SR), dtype=np.float32), SR, "你好", "zh")
    assert len(stub.calls) == 1
    assert units[0].text == "你好"
