"""Tests for the V0.7 alignment quality / confidence system."""

from __future__ import annotations

import numpy as np

from autosync.alignment.ctc import forced_align, log_softmax
from autosync.export.json_export import result_to_dict
from autosync.quality import (
    QualityPolicy,
    QualityState,
    decide,
    extract_features,
    player_timeline,
    summarize,
)
from autosync.quality.features import QualityFeatures
from autosync.types import AlignedLine, AlignmentResult, AlignUnit


def _lp(path: list[int], C: int, peak: float = 6.0) -> np.ndarray:
    logits = np.zeros((len(path), C), dtype=np.float64)
    for t, c in enumerate(path):
        logits[t, c] = peak
    return log_softmax(logits)


def test_feature_extraction_basic() -> None:
    lp = _lp([1, 1, 0, 2, 2, 0, 3, 3], C=4)
    al = forced_align(lp, [1, 2, 3], blank=0)
    feats = extract_features(lp, [1, 2, 3], 0, al, char_of_token=[0, 1, 2], frame_seconds=0.02)
    assert feats.n_tokens == 3
    assert feats.missing_ratio == 0.0
    assert np.isfinite(feats.score_margin_per_token)
    assert 0.0 <= feats.blank_frame_ratio <= 1.0


def test_feature_missing_tokens_reported() -> None:
    lp = _lp([1, 0], C=3)
    al = forced_align(lp, [1, 1], blank=0)  # infeasible -> missing spans
    feats = extract_features(lp, [1, 1], 0, al, char_of_token=[0, 1], frame_seconds=0.02)
    assert feats.missing_ratio == 1.0


def test_decision_failed_on_missing() -> None:
    f = QualityFeatures(n_tokens=3, missing_ratio=0.7)
    d = decide(f)
    assert d.state is QualityState.FAILED
    assert d.allow_word_timeline is False
    assert d.fallback == "line_lrc_failed"


def test_decision_uncertain_on_timing_outlier() -> None:
    f = QualityFeatures(n_tokens=10, missing_ratio=0.0, max_intra_char_span_s=3.0)
    policy = QualityPolicy(max_intra_char_span_s=1.5)
    d = decide(f, policy)
    assert d.state is QualityState.UNCERTAIN
    assert d.fallback == "line_lrc"
    assert any("max_intra_char_span_s" in r for r in d.reasons)


def test_decision_reliable_when_clean() -> None:
    f = QualityFeatures(n_tokens=10, missing_ratio=0.0, max_intra_char_span_s=0.3, char_duration_ratio=2.0)
    policy = QualityPolicy(max_intra_char_span_s=1.5, max_char_duration_ratio=10.0)
    d = decide(f, policy)
    assert d.state is QualityState.RELIABLE
    assert d.allow_word_timeline is True
    assert d.fallback == "none"


def _result() -> AlignmentResult:
    return AlignmentResult(
        language="ja",
        engine="model",
        lines=[
            AlignedLine(
                index=0,
                text="きょうはいい天気",
                start=1.0,
                end=2.0,
                units=[AlignUnit(text="き", start=1.0, end=1.5), AlignUnit(text="ょ", start=1.5, end=2.0)],
                source="model",
                quality="reliable",
            ),
            AlignedLine(
                index=1,
                text="あしたはあめ",
                start=3.0,
                end=4.0,
                units=[AlignUnit(text="あ", start=3.0, end=4.0)],
                source="model",
                quality="uncertain",
                quality_reasons=["low_free_decode_agreement"],
            ),
        ],
    )


def test_summarize_counts_states() -> None:
    s = summarize(_result().lines)
    assert s["reliable"] == 1 and s["uncertain"] == 1 and s["failed"] == 0


def test_player_timeline_fallback_preserves_lyrics() -> None:
    tl = player_timeline(_result())
    assert tl[0]["mode"] == "word" and "units" in tl[0]
    assert tl[1]["mode"] == "line_lrc" and "units" not in tl[1]
    # original lyrics always preserved
    assert tl[1]["text"] == "あしたはあめ"
    # original LRC start is kept independently of the model prediction
    assert tl[1]["lrc_start"] == 3.0
    # a plain LRC line has no explicit end -> null, never fabricated
    assert tl[1]["lrc_end"] is None


def test_json_export_has_quality_and_timeline() -> None:
    data = result_to_dict(_result())
    assert "quality_summary" in data and "player_timeline" in data
    assert data["quality_summary"]["reliable"] == 1
    assert data["schema_version"] == "0.2"
