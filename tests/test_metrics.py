"""Regression tests for evaluation metrics (coverage / miss / extras)."""

from __future__ import annotations

import metrics


def test_coverage_and_misses_are_reported() -> None:
    ref = [
        {"text": "a", "start": 0.0, "end": 1.0},
        {"text": "b", "start": 1.0, "end": 2.0},
        {"text": "c", "start": 2.0, "end": 3.0},
    ]
    pred = [
        {"text": "a", "start": 0.1, "end": 1.0},
        {"text": "c", "start": 2.1, "end": 3.0},
    ]
    r = metrics.score(pred, ref)
    assert r["n_matched_words"] == 2
    assert abs(r["coverage"] - 2 / 3) < 1e-9
    assert abs(r["miss_rate"] - 1 / 3) < 1e-9
    assert [m["text"] for m in r["missed"]] == ["b"]


def test_extra_units_reported() -> None:
    ref = [{"text": "a", "start": 0.0, "end": 1.0}]
    pred = [
        {"text": "a", "start": 0.0, "end": 1.0},
        {"text": "zzz", "start": 1.0, "end": 2.0},
    ]
    r = metrics.score(pred, ref)
    assert r["n_extra_units"] >= 1
    assert r["coverage"] == 1.0


def test_multi_char_token_is_split_as_uncertain() -> None:
    atoms = metrics.char_timeline([{"text": "世界", "start": 0.0, "end": 2.0, "uncertain": False}])
    assert len(atoms) == 2
    assert all(a["uncertain"] for a in atoms)
    assert atoms[0]["start"] == 0.0 and atoms[1]["end"] == 2.0


def test_error_rates_present() -> None:
    ref = [{"text": "a", "start": 0.0, "end": 1.0}]
    pred = [{"text": "a", "start": 0.5, "end": 1.0}]
    r = metrics.score(pred, ref)
    assert r["start_err_rate_gt_0p1"] == 1.0
    assert r["start_err_rate_gt_0p5"] == 0.0


def test_merge_reports_micro_averages() -> None:
    ref = [{"text": "a", "start": 0.0, "end": 1.0}]
    r1 = metrics.score([{"text": "a", "start": 0.1, "end": 1.0}], ref)
    r2 = metrics.score([], ref)
    merged = metrics.merge_reports([r1, r2])
    assert merged["n_reference_words"] == 2
    assert merged["n_matched_words"] == 1
    assert abs(merged["coverage"] - 0.5) < 1e-9
