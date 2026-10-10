"""Regression tests for the V0.2 energy refinement."""

from __future__ import annotations

import numpy as np

from autosync.alignment.refine import refine_units_energy
from autosync.types import AlignUnit

SR = 16_000


def _tone_then_silence() -> np.ndarray:
    t = np.arange(int(2.0 * SR)) / SR
    audio = np.zeros_like(t, dtype=np.float32)
    seg = slice(int(0.5 * SR), int(1.4 * SR))
    audio[seg] = 0.3 * np.sin(2 * np.pi * 220 * t[seg])
    return audio.astype(np.float32)


def test_refine_extends_end_over_voiced_region() -> None:
    audio = _tone_then_silence()
    units = [AlignUnit(text="a", start=0.0, end=0.6), AlignUnit(text="b", start=1.6, end=1.9)]
    out = refine_units_energy(units, audio, SR, max_extend=1.2)
    assert out[0].end > 0.6
    # must never cross the next unit's start
    assert out[0].end <= units[1].start + 1e-6


def test_refine_does_not_extend_into_silence() -> None:
    audio = np.zeros(int(2.0 * SR), dtype=np.float32)
    units = [AlignUnit(text="a", start=0.0, end=0.5), AlignUnit(text="b", start=1.0, end=1.5)]
    out = refine_units_energy(units, audio, SR)
    assert abs(out[0].end - 0.5) < 1e-6


def test_refine_preserves_order_and_starts() -> None:
    audio = _tone_then_silence()
    units = [AlignUnit(text="a", start=0.1, end=0.3), AlignUnit(text="b", start=0.4, end=0.5)]
    out = refine_units_energy(units, audio, SR)
    assert [u.start for u in out] == [0.1, 0.4]
    assert out[0].end <= out[1].start + 1e-6
    assert out[1].end <= out[0].start + 2.0


def test_start_mode_forward_is_off_by_default() -> None:
    # A unit starting inside leading silence is moved to the voiced onset only
    # when explicitly requested; the default leaves it untouched.
    audio = np.zeros(int(2.0 * SR), dtype=np.float32)
    seg = slice(int(0.5 * SR), int(1.5 * SR))
    t = np.arange(int(2.0 * SR)) / SR
    audio[seg] = 0.3 * np.sin(2 * np.pi * 220 * t[seg])
    units = [AlignUnit(text="a", start=0.1, end=0.9)]
    default = refine_units_energy(units, audio.astype(np.float32), SR)
    assert abs(default[0].start - 0.1) < 1e-6
    forward = refine_units_energy(units, audio.astype(np.float32), SR, start_mode="forward")
    assert forward[0].start > 0.1


def test_start_mode_backward_snaps_to_rising_edge() -> None:
    audio = np.zeros(int(2.0 * SR), dtype=np.float32)
    seg = slice(int(0.5 * SR), int(1.5 * SR))
    t = np.arange(int(2.0 * SR)) / SR
    audio[seg] = 0.3 * np.sin(2 * np.pi * 220 * t[seg])
    units = [AlignUnit(text="a", start=0.9, end=1.4)]
    out = refine_units_energy(units, audio.astype(np.float32), SR, start_mode="backward")
    assert abs(out[0].start - 0.5) < 0.06
