"""Acoustic confidence features for forced alignment (V0.7 P0).

Everything here is computed from the model's own outputs — frame log-probabilities,
the target token sequence, the forced-alignment spans and the (inference-time)
phoneme→character mapping. **No reference annotation is ever used.**

The features deliberately go beyond "mean blank probability" (which the task
forbids as a sole quality score) and are individually validated against real
singing errors in ``benchmarks/confidence_eval.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..alignment.ctc import Alignment


@dataclass
class QualityFeatures:
    # --- path scores -------------------------------------------------------
    n_frames: int = 0
    n_tokens: int = 0
    score_margin_total: float = 0.0        # forced path minus all-blank path
    score_margin_per_token: float = 0.0
    score_margin_per_frame: float = 0.0
    mean_token_logp: float = 0.0
    mean_peak_advantage: float = 0.0       # token prob minus best competitor at its peak
    # --- path composition --------------------------------------------------
    blank_frame_ratio: float = 0.0         # frames the path assigned to blank
    blank_dominant_ratio: float = 0.0      # frames where blank argmax / prob > 0.5
    longest_blank_run_s: float = 0.0
    # --- target integrity --------------------------------------------------
    missing_ratio: float = 0.0             # tokens with no span
    sparse_ratio: float = 0.0              # tokens with <= 1 frame
    free_decode_agreement: float = 0.0     # 1 - normalised edit distance(model free path, target)
    # --- timing plausibility ----------------------------------------------
    max_inter_token_gap_s: float = 0.0
    mean_intra_char_span_s: float = 0.0
    max_intra_char_span_s: float = 0.0
    char_duration_max_s: float = 0.0
    char_duration_median_s: float = 0.0
    char_duration_ratio: float = 0.0       # max / median
    token_duration_cv: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _collapsed(ids: list[int], blank: int) -> list[int]:
    out: list[int] = []
    prev = None
    for i in ids:
        if i != blank and i != prev:
            out.append(int(i))
        prev = i
    return out


def _edit_distance(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def extract_features(
    log_probs: np.ndarray,
    targets: list[int],
    blank: int,
    alignment: Alignment,
    char_of_token: list[int] | None = None,
    frame_seconds: float = 0.02,
    blank_prob_threshold: float = 0.5,
) -> QualityFeatures:
    """Compute :class:`QualityFeatures` for one aligned line.

    ``char_of_token[k]`` is the character index that phoneme ``k`` belongs to
    (``-1`` for non-character tokens such as the leading/trailing ``pau``); it is
    known at inference time from the lyric→phoneme mapping.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    T = lp.shape[0]
    n_tok = len(targets)
    f = QualityFeatures(n_frames=T, n_tokens=n_tok)
    if T == 0 or n_tok == 0:
        f.score_margin_total = 0.0
        return f

    p = np.exp(lp)
    blank_total = float(np.sum(lp[:, blank]))
    covered = np.zeros(T, dtype=bool)
    token_logps: list[float] = []
    peak_advantages: list[float] = []
    token_durs: list[int] = []
    spans = alignment.spans
    missing = 0
    sparse = 0
    for tok, span in zip(targets, spans):
        if span is None:
            missing += 1
            continue
        s, e = span.start_frame, span.end_frame
        e = max(e, s)
        covered[s : e + 1] = True
        seg = lp[s : e + 1, tok]
        token_logps.append(float(np.mean(seg)))
        token_durs.append(e - s + 1)
        if e - s + 1 <= 1:
            sparse += 1
        # peak advantage at the token's best frame
        sub = lp[s : e + 1]
        peak = int(np.argmax(sub[:, tok]))
        others = np.delete(sub[peak], tok) if sub.shape[1] > 1 else np.array([-np.inf])
        peak_advantages.append(float(sub[peak, tok] - np.max(others)))

    # path log-probability: token frames use their token logp, others blank
    forced_total = 0.0
    for tok, span in zip(targets, spans):
        if span is None:
            continue
        s, e = span.start_frame, span.end_frame
        forced_total += float(np.sum(lp[s : e + 1, tok]))
    forced_total += float(np.sum(lp[~covered, blank]))

    f.score_margin_total = forced_total - blank_total
    f.score_margin_per_token = f.score_margin_total / n_tok
    f.score_margin_per_frame = f.score_margin_total / T
    f.mean_token_logp = float(np.mean(token_logps)) if token_logps else -np.inf
    f.mean_peak_advantage = float(np.mean(peak_advantages)) if peak_advantages else -np.inf
    f.missing_ratio = missing / n_tok
    f.sparse_ratio = sparse / n_tok
    f.blank_frame_ratio = float(np.mean(~covered))
    f.blank_dominant_ratio = float(np.mean(p[:, blank] > blank_prob_threshold))
    f.longest_blank_run_s = _longest_run(p[:, blank] > blank_prob_threshold) * frame_seconds

    # free CTC decode (greedy, collapsed) vs target
    free = _collapsed(list(np.argmax(lp, axis=1)), blank)
    target_noblank = [t for t in targets if t != blank]
    denom = max(len(free), len(target_noblank), 1)
    f.free_decode_agreement = 1.0 - _edit_distance(free, target_noblank) / denom

    # timing plausibility
    ordered = [(s.start_frame, s.end_frame) for s in spans if s is not None]
    ordered.sort()
    if len(ordered) >= 2:
        gaps = [ordered[i + 1][0] - ordered[i][1] - 1 for i in range(len(ordered) - 1)]
        f.max_inter_token_gap_s = max(0, max(gaps)) * frame_seconds
    f.token_duration_cv = float(np.std(token_durs) / np.mean(token_durs)) if token_durs and np.mean(token_durs) > 0 else 0.0

    if char_of_token is not None:
        by_char: dict[int, list[tuple[int, int]]] = {}
        for tok_idx, span in zip(range(n_tok), spans):
            if span is None:
                continue
            ch = char_of_token[tok_idx] if tok_idx < len(char_of_token) else -1
            if ch < 0:
                continue
            by_char.setdefault(ch, []).append((span.start_frame, span.end_frame))
        spans_s = [(max(e for _, e in v) - min(s for s, _ in v) + 1) * frame_seconds for v in by_char.values() if v]
        if spans_s:
            f.mean_intra_char_span_s = float(np.mean(spans_s))
            f.max_intra_char_span_s = float(np.max(spans_s))
            f.char_duration_max_s = f.max_intra_char_span_s
            f.char_duration_median_s = float(np.median(spans_s))
            med = f.char_duration_median_s
            f.char_duration_ratio = (f.char_duration_max_s / med) if med > 0 else 0.0
    return f


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best
