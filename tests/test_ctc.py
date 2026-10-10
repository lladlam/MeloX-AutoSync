from __future__ import annotations

import itertools

import numpy as np

from autosync.alignment.ctc import forced_align, log_softmax


def make_log_probs(path: list[int], num_classes: int, peak: float = 8.0) -> np.ndarray:
    T = len(path)
    logits = np.zeros((T, num_classes), dtype=np.float64)
    for t, c in enumerate(path):
        logits[t, c] = peak
    return log_softmax(logits)


def test_single_token_span() -> None:
    # frames 0,1 -> token 1 ; frames 2,3 -> blank
    lp = make_log_probs([1, 1, 0, 0], num_classes=3)
    al = forced_align(lp, [1], blank=0)
    assert len(al.spans) == 1
    span = al.spans[0]
    assert span is not None
    assert (span.start_frame, span.end_frame) == (0, 1)


def test_two_tokens_with_blank() -> None:
    lp = make_log_probs([1, 0, 2], num_classes=3)
    al = forced_align(lp, [1, 2], blank=0)
    s0, s1 = al.spans
    assert s0 is not None and s1 is not None
    assert (s0.start_frame, s0.end_frame) == (0, 0)
    assert (s1.start_frame, s1.end_frame) == (2, 2)


def test_repeated_token_needs_blank() -> None:
    # token 1 twice, separated by a blank frame
    lp = make_log_probs([1, 0, 1], num_classes=3)
    al = forced_align(lp, [1, 1], blank=0)
    s0, s1 = al.spans
    assert s0 is not None and s1 is not None
    assert (s0.start_frame, s0.end_frame) == (0, 0)
    assert (s1.start_frame, s1.end_frame) == (2, 2)


def test_empty_targets() -> None:
    lp = make_log_probs([0, 0], num_classes=2)
    al = forced_align(lp, [], blank=0)
    assert al.spans == []


def test_spans_are_monotonic() -> None:
    # A longer, noisier sequence: spans must be non-overlapping and ordered.
    rng = np.random.default_rng(0)
    T, C = 40, 5
    logits = rng.normal(0, 1, size=(T, C))
    path = [1, 1, 2, 3, 3, 4]
    for t, c in enumerate(path):
        logits[t * 6, c] += 6.0
    lp = log_softmax(logits)
    al = forced_align(lp, [1, 2, 3, 4], blank=0)
    spans = [s for s in al.spans if s is not None]
    assert len(spans) == 4
    for a, b in itertools.pairwise(spans):
        assert a.end_frame < b.start_frame


def test_out_of_range_token_raises() -> None:
    lp = make_log_probs([1, 0], num_classes=3)
    try:
        forced_align(lp, [7], blank=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for out-of-range token")
