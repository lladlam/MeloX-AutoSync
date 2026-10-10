"""Three-state alignment quality decision (V0.7 P1).

* ``RELIABLE``  – acoustic evidence is sufficient; word-level timings may be used.
* ``UNCERTAIN`` – evidence is insufficient; keep the diagnostic timeline but the
  player-facing output should fall back to the original line-level LRC.
* ``FAILED``    – structural failure (missing spans); return a structured failure
  and keep the original LRC.

Thresholds live in :class:`QualityPolicy` and are **calibrated on a training /
calibration split only** (see ``benchmarks/confidence_eval.py``); this module
contains no dataset-specific constants beyond conservative defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .features import QualityFeatures


class QualityState(str, Enum):
    RELIABLE = "reliable"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


@dataclass
class QualityPolicy:
    name: str = "default-conservative"
    # --- structural failure ---
    fail_missing_ratio: float = 0.5
    # --- insufficient evidence (any violation -> UNCERTAIN) ---
    min_free_decode_agreement: float = 0.0
    min_score_margin_per_token: float = -1e9
    min_mean_peak_advantage: float = -1e9
    max_blank_dominant_ratio: float = 1.0
    max_intra_char_span_s: float = 1e9
    max_char_duration_ratio: float = 1e9
    max_inter_token_gap_s: float = 1e9

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class QualityDecision:
    state: QualityState
    reasons: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def allow_word_timeline(self) -> bool:
        return self.state is QualityState.RELIABLE

    @property
    def fallback(self) -> str:
        """What the player-facing output should do."""
        if self.state is QualityState.RELIABLE:
            return "none"
        if self.state is QualityState.UNCERTAIN:
            return "line_lrc"
        return "line_lrc_failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reasons": self.reasons,
            "allow_word_timeline": self.allow_word_timeline,
            "fallback": self.fallback,
            "features": self.features,
        }


def decide(features: QualityFeatures, policy: QualityPolicy | None = None) -> QualityDecision:
    """Map features to a quality state using the (calibrated) policy."""
    policy = policy or QualityPolicy()
    reasons: list[str] = []

    if features.n_tokens == 0:
        return QualityDecision(
            state=QualityState.FAILED,
            reasons=["no target tokens"],
            features=features.to_dict(),
        )
    if features.missing_ratio >= policy.fail_missing_ratio:
        return QualityDecision(
            state=QualityState.FAILED,
            reasons=[f"missing_ratio={features.missing_ratio:.2f} >= {policy.fail_missing_ratio}"],
            features=features.to_dict(),
        )

    if features.free_decode_agreement < policy.min_free_decode_agreement:
        reasons.append(
            f"free_decode_agreement={features.free_decode_agreement:.2f} < {policy.min_free_decode_agreement}"
        )
    if features.score_margin_per_token < policy.min_score_margin_per_token:
        reasons.append(
            f"score_margin_per_token={features.score_margin_per_token:.2f} < {policy.min_score_margin_per_token}"
        )
    if features.mean_peak_advantage < policy.min_mean_peak_advantage:
        reasons.append(
            f"mean_peak_advantage={features.mean_peak_advantage:.2f} < {policy.min_mean_peak_advantage}"
        )
    if features.blank_dominant_ratio > policy.max_blank_dominant_ratio:
        reasons.append(
            f"blank_dominant_ratio={features.blank_dominant_ratio:.2f} > {policy.max_blank_dominant_ratio}"
        )
    if features.max_intra_char_span_s > policy.max_intra_char_span_s:
        reasons.append(
            f"max_intra_char_span_s={features.max_intra_char_span_s:.2f} > {policy.max_intra_char_span_s}"
        )
    if features.char_duration_ratio > policy.max_char_duration_ratio:
        reasons.append(
            f"char_duration_ratio={features.char_duration_ratio:.2f} > {policy.max_char_duration_ratio}"
        )
    if features.max_inter_token_gap_s > policy.max_inter_token_gap_s:
        reasons.append(
            f"max_inter_token_gap_s={features.max_inter_token_gap_s:.2f} > {policy.max_inter_token_gap_s}"
        )

    if reasons:
        return QualityDecision(state=QualityState.UNCERTAIN, reasons=reasons, features=features.to_dict())
    return QualityDecision(state=QualityState.RELIABLE, reasons=["ok"], features=features.to_dict())
