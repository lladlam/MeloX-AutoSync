from __future__ import annotations

import itertools

from autosync.alignment.heuristic import distribute


def test_distribute_covers_span_monotonically() -> None:
    units = distribute(2.0, 8.0, ["a", "bb", "ccc"])
    assert units[0].start == 2.0
    assert abs(units[-1].end - 8.0) < 1e-9
    for a, b in itertools.pairwise(units):
        assert a.end <= b.start + 1e-9
    assert all(u.source == "heuristic" for u in units)


def test_distribute_weights_by_length() -> None:
    units = distribute(0.0, 10.0, ["a", "aaaa"])
    # "aaaa" should get roughly 4x the time of "a"
    dur_a = units[0].end - units[0].start
    dur_aaaa = units[1].end - units[1].start
    assert dur_aaaa > dur_a * 2


def test_distribute_empty() -> None:
    assert distribute(0.0, 1.0, []) == []
