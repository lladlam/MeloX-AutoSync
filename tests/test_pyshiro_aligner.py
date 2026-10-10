"""Tests for the pyshiro adapter's pure internals (V0.4 §1)."""

from __future__ import annotations

from autosync.models.pyshiro_aligner import _state_count, phoneme_intervals

_PM = {
    "pau": {"states": [0, 0, 0]},
    "a": {"states": [1, 1, 1]},
    "o": {"states": [2, 2, 2]},
    "O": {"states": [3, 3, 3], "alias_of": "o"},
    "skipme": {"pskip": 1, "states": []},
}


def test_state_count_defaults_and_pskip() -> None:
    assert _state_count(_PM, "a") == 3
    assert _state_count(_PM, "unknown") == 3
    assert _state_count(_PM, "skipme") == 0


def test_phoneme_intervals_keep_consecutive_identical_phonemes() -> None:
    # 6 states: o (3), o (3) must NOT be merged into one interval.
    segments = [(0, 1), (1, 2), (2, 5), (5, 6), (6, 7), (7, 9)]
    iv = phoneme_intervals(_PM, ["o", "o"], segments)
    assert len(iv) == 2
    assert iv[0] == (0, 5, "o")
    assert iv[1] == (5, 9, "o")


def test_phoneme_intervals_handle_pskip_zero_width() -> None:
    segments = [(0, 1), (1, 2), (2, 3)]  # one phoneme with states, one skipped
    iv = phoneme_intervals(_PM, ["a", "skipme", "a"], segments + segments)
    assert iv[1][0] == iv[1][1]  # zero-width skipped phoneme
    assert len(iv) == 3
