"""Tests for the V0.6 pause-aware alignment variants."""

from __future__ import annotations

import numpy as np
import pytest

from autosync.alignment.ctc import TokenSpan, forced_align, log_softmax
from autosync.alignment.pause_aware import (
    PauseConfig,
    _best_voiced_run,
    blank_aware_spans,
    gap_penalized_alignment,
    resolve_spans,
    semi_markov_align,
)

BLANK = 0


def _lp(rows: list[list[float]]) -> np.ndarray:
    return log_softmax(np.asarray(rows, dtype=np.float64))


def test_best_voiced_run_picks_max_sum_window() -> None:
    adv = np.array([1.0, 1.0, 1.0, -5.0, -5.0, 1.0, 1.0])
    assert _best_voiced_run(adv, 1) == (0, 3)
    adv2 = np.array([-1.0, -1.0, 3.0, 3.0, 3.0, -1.0])
    assert _best_voiced_run(adv2, 2) == (2, 5)


def test_blank_aware_trims_token_span_to_voiced_run() -> None:
    # token 1 beats blank only on frames 2..4; base span wrongly covers 0..6.
    rows = [
        [0.0, -1.0],  # blank higher
        [0.0, -1.0],
        [-1.0, 0.0],  # token higher
        [-1.0, 0.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [0.0, -1.0],
    ]
    lp = _lp(rows)
    base = forced_align(lp, [1], blank=BLANK)
    # force a stretched base span to test the trimming itself
    base.spans = [TokenSpan(token=1, start_frame=0, end_frame=6, score=0.0)]
    spans = blank_aware_spans(lp, [1], BLANK, base, margin=0.8, min_frames=1)
    assert spans[0] is not None
    assert (spans[0].start_frame, spans[0].end_frame) == (2, 4)


def test_blank_aware_preserves_order_and_nonempty() -> None:
    rng = np.random.default_rng(0)
    lp = log_softmax(rng.normal(0, 1, size=(30, 4)))
    base = forced_align(lp, [1, 2, 3], blank=BLANK)
    spans = blank_aware_spans(lp, [1, 2, 3], BLANK, base, margin=0.0, min_frames=1)
    prev = -1
    for s in spans:
        if s is None:
            continue
        assert s.end_frame >= s.start_frame
        assert s.start_frame > prev
        prev = s.end_frame


def test_gap_penalty_shrinks_a_flat_span() -> None:
    # token 1 is (slightly) higher than blank on every frame; without a penalty
    # the span covers everything, with a penalty it collapses to few frames.
    rows = [[0.0, 0.1] for _ in range(10)]
    lp = _lp(rows)
    base = forced_align(lp, [1], blank=BLANK)
    assert base.spans[0] is not None
    long_len = base.spans[0].num_frames

    penalized = gap_penalized_alignment(lp, [1], BLANK, non_blank_penalty=1.0)
    assert penalized.spans[0] is not None
    assert penalized.spans[0].num_frames < long_len


def test_gap_penalty_zero_equals_base() -> None:
    lp = _lp([[0.0, 1.0], [0.0, -1.0], [0.0, 0.0], [0.0, -1.0]])
    a = forced_align(lp, [1], blank=BLANK)
    b = gap_penalized_alignment(lp, [1], BLANK, non_blank_penalty=0.0)
    assert a.spans[0].start_frame == b.spans[0].start_frame
    assert a.spans[0].end_frame == b.spans[0].end_frame


def test_resolve_spans_dispatch_and_unknown_mode() -> None:
    lp = _lp([[0.0, 1.0], [0.0, -1.0], [0.0, 1.0]])
    a = resolve_spans(lp, [1], BLANK, PauseConfig(mode="none"))
    assert a.spans[0] is not None
    b = resolve_spans(lp, [1], BLANK, PauseConfig(mode="blank_aware"))
    assert b.spans[0] is not None
    c = resolve_spans(lp, [1], BLANK, PauseConfig(mode="gap_penalized", non_blank_penalty=0.5))
    assert c.spans[0] is not None
    with pytest.raises(ValueError):
        resolve_spans(lp, [1], BLANK, PauseConfig(mode="bogus"))


def test_optional_pause_without_ids_is_noop() -> None:
    lp = _lp([[0.0, 1.0], [0.0, -1.0], [0.0, 1.0]])
    cfg = PauseConfig(mode="optional_pause", pause_ids=())
    out = resolve_spans(lp, [1], BLANK, cfg)
    base = forced_align(lp, [1], blank=BLANK)
    assert out.spans[0].start_frame == base.spans[0].start_frame
    assert out.spans[0].end_frame == base.spans[0].end_frame


def test_semi_markov_recovers_separated_tokens() -> None:
    T, C = 20, 3
    logits = np.full((T, C), -5.0)
    for f in range(1, 4):
        logits[f, 1] = 5.0
    for f in range(10, 13):
        logits[f, 2] = 5.0
    lp = log_softmax(logits)
    spans = semi_markov_align(
        lp, [1, 2], BLANK, intra=[False], dmin=1, dmax=6, gmin=0, gmax_intra=2, gmax_inter=10
    )
    assert spans is not None
    assert spans[0].start_frame <= 1 and spans[0].end_frame >= 3
    assert spans[1].start_frame <= 10 and spans[1].end_frame >= 12


def test_semi_markov_bounds_intra_character_gap() -> None:
    # token 2's only peak is far away, but the two tokens share a character, so
    # the small intra-character gap bound must keep them close together.
    T, C = 30, 3
    logits = np.full((T, C), -5.0)
    for f in range(2, 5):
        logits[f, 1] = 5.0
    logits[25, 2] = 4.0
    lp = log_softmax(logits)
    spans = semi_markov_align(
        lp, [1, 2], BLANK, intra=[True], dmin=1, dmax=8, gmin=0, gmax_intra=2, gmax_inter=40
    )
    assert spans is not None
    assert spans[1].start_frame - spans[0].end_frame <= 4
