"""Tests for the pyshiro adapter's per-phoneme state mapping (V0.5 P1)."""

from __future__ import annotations

import itertools

from autosync.models.pyshiro_aligner import phoneme_intervals_from_starts


def test_repeated_identical_phonemes_stay_separate() -> None:
    # o (3 states) followed by another o (3 states) must not be merged.
    starts = [0, 3, 6]
    segments = [(0, 1), (1, 2), (2, 5), (5, 6), (6, 7), (7, 9)]
    iv = phoneme_intervals_from_starts(["o", "o"], starts, segments)
    assert len(iv) == 2
    assert iv[0] == (0, 5, "o")
    assert iv[1] == (5, 9, "o")


def test_skipped_phoneme_is_zero_width_not_dropped() -> None:
    # A pskip phoneme still owns states; if the path skips them they are
    # zero-width but present and in order.
    phonemes = ["a", "skipme", "a"]
    starts = [0, 3, 6, 9]
    segments = [(0, 2), (2, 3), (3, 4), (4, 4), (4, 4), (4, 4), (4, 6), (6, 7), (7, 8)]
    iv = phoneme_intervals_from_starts(phonemes, starts, segments)
    assert len(iv) == 3
    assert iv[1][0] == iv[1][1] == 4  # zero-width, at its own state position
    assert iv[2][0] == 4


def test_interval_order_is_monotonic() -> None:
    phonemes = list("aiaio")
    starts = [0, 3, 6, 9, 12, 15]
    segments = [(i, i + 1) for i in range(15)]
    iv = phoneme_intervals_from_starts(phonemes, starts, segments)
    for (_, e1, _), (s2, _, _) in itertools.pairwise(iv):
        assert e1 <= s2
