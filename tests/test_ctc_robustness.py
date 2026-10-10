"""Regression tests for CTC Viterbi robustness (V0.2 item 1)."""

from __future__ import annotations

import numpy as np
import pytest

from autosync.alignment.ctc import (
    InfeasibleAlignmentError,
    forced_align,
    log_softmax,
    minimum_frames,
)


def _lp(path: list[int], num_classes: int, peak: float = 8.0) -> np.ndarray:
    logits = np.zeros((len(path), num_classes), dtype=np.float64)
    for t, c in enumerate(path):
        logits[t, c] = peak
    return log_softmax(logits)


def test_minimum_frames_counts_identical_pairs() -> None:
    assert minimum_frames([1, 2, 3]) == 3
    assert minimum_frames([1, 1]) == 3  # identical pair needs an extra blank frame
    assert minimum_frames([1, 1, 1]) == 5
    assert minimum_frames([]) == 0


def test_too_few_frames_is_infeasible() -> None:
    lp = _lp([1, 0], num_classes=3)  # T=2
    al = forced_align(lp, [1, 1], blank=0)  # requires 3 frames
    assert not al.feasible
    assert al.required_frames == 3
    assert any("too few frames" in r for r in al.reasons)
    assert al.spans == [None, None]


def test_collapsed_alignment_is_detected() -> None:
    T, C = 100, 5
    logits = np.zeros((T, C), dtype=np.float64)
    logits[:, 1] = 10.0  # token 1 dominates almost the whole audio
    logits[-3, 2] = 10.0
    logits[-2, 3] = 10.0
    logits[-1, 4] = 10.0
    lp = log_softmax(logits)
    al = forced_align(lp, [1, 2, 3, 4], blank=0)
    assert not al.feasible
    assert any("collapsed" in r for r in al.reasons)


def test_normal_alignment_is_feasible() -> None:
    lp = _lp([1, 1, 0, 2, 2, 0, 3, 3], num_classes=4)
    al = forced_align(lp, [1, 2, 3], blank=0)
    assert al.feasible
    assert all(s is not None for s in al.spans)
    assert [s.token for s in al.spans if s] == [1, 2, 3]


def test_many_single_frame_tokens_are_not_collapse() -> None:
    # Character CTC on sustained notes emits brief spikes separated by blanks.
    # That must NOT be flagged as a collapse.
    T = 40
    path = [1, 0, 2, 0, 3, 0, 4, 0]
    logits = np.zeros((T, 5), dtype=np.float64)
    for i, c in enumerate(path):
        logits[i * 5, c] = 10.0
        logits[i * 5 + 1, 0] = 10.0
    lp = log_softmax(logits)
    al = forced_align(lp, [1, 2, 3, 4], blank=0)
    assert al.feasible


def test_strict_mode_raises() -> None:
    lp = _lp([1, 0], num_classes=3)
    with pytest.raises(InfeasibleAlignmentError):
        forced_align(lp, [1, 1], blank=0, strict=True)


def test_empty_targets_feasible() -> None:
    lp = _lp([0, 0], num_classes=2)
    al = forced_align(lp, [], blank=0)
    assert al.feasible
    assert al.spans == []
