"""Regression tests for the V0.3 evaluation-methodology fixes (P0).

The key invariant: in ``lrc_only`` mode the line window must NOT depend on the
ground-truth word boundaries.
"""

from __future__ import annotations

import metrics
import numpy as np
import singing_eval as se


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def test_lrc_only_ignores_ground_truth_boundaries() -> None:
    audio = _audio(3.0)
    # Absurd reference boundaries must not influence the window at all.
    s, e = se.line_window("lrc_only", "zero", audio, 16000, 1.234, 2.345)
    assert s == 0.0
    assert abs(e - 3.0) < 1e-6


def test_lrc_only_window_is_full_clip_regardless_of_reference() -> None:
    audio = _audio(5.0)
    a = se.line_window("lrc_only", "zero", audio, 16000, 0.1, 0.2)
    b = se.line_window("lrc_only", "zero", audio, 16000, 4.0, 4.9)
    assert a == b == (0.0, 5.0)


def test_oracle_uses_ground_truth_boundaries() -> None:
    audio = _audio(3.0)
    s, e = se.line_window("oracle", "zero", audio, 16000, 1.0, 2.5)
    assert (s, e) == (1.0, 2.5)


def test_energy_onset_is_audio_only() -> None:
    audio = _audio(2.0)
    t = np.arange(int(1.0 * 16000)) / 16000
    audio[16000:] = 0.3 * np.sin(2 * np.pi * 220 * t)
    onset = se.energy_onset(audio, 16000)
    assert 0.9 < onset < 1.2


def test_metrics_reports_uncertain_ratio() -> None:
    ref = [{"text": "世界", "start": 0.0, "end": 2.0}]
    # One 2-char predicted token -> both atoms uncertain.
    pred = [{"text": "世界", "start": 0.0, "end": 2.0, "uncertain": True}]
    r = metrics.score(pred, ref)
    # The single matched word is flagged uncertain (it came from a 2-char token).
    assert r["uncertain_matched"] == 1
    assert r["uncertain_ratio"] == 1.0
