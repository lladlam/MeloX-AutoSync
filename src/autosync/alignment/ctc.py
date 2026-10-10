"""CTC forced alignment via Viterbi dynamic programming.

We implement the standard CTC alignment algorithm ourselves (rather than
depending on ``torchaudio.functional.forced_align``) so that the behaviour is
fully controlled, dependency-light and unit-testable. This is the heart of the
engine.

Given frame-level log-probabilities ``log_probs`` of shape ``(T, C)`` and a
target token sequence, we find the single most probable *frame alignment* that
emits exactly the target tokens (with optional CTC blanks between them),
i.e. the Viterbi path through the extended sequence:

    [blank, t0, blank, t1, blank, ..., t_{N-1}, blank]

Robustness (V0.2)
-----------------
The aligner explicitly detects and reports *infeasible* requests instead of
silently returning a degenerate path:

* **too few frames** – emitting ``N`` tokens requires at least
  ``N + (#adjacent identical token pairs)`` frames, because two identical
  tokens must be separated by a blank frame. If ``T`` is smaller the request is
  impossible and ``Alignment.feasible`` is ``False``.
* **impossible path / all-missing spans** – if the Viterbi path does not cover
  every token state (or is all ``-inf``), the alignment is flagged infeasible.
* **non-monotonic spans** – a structural sanity check rejects overlapping or
  out-of-order spans.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

NEG_INF = -1e30


class InfeasibleAlignmentError(ValueError):
    """Raised when a forced alignment request cannot be satisfied."""


@dataclass
class TokenSpan:
    token: int
    # Inclusive frame indices into the log-probability time axis.
    start_frame: int
    end_frame: int
    # Mean token log-probability over the assigned frames (confidence proxy).
    score: float

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class Alignment:
    spans: list[TokenSpan | None]
    total_log_prob: float
    path: np.ndarray  # state index per frame
    feasible: bool = True
    # Minimum number of frames this target sequence can possibly occupy.
    required_frames: int = 0
    reasons: list[str] = field(default_factory=list)
    # Mean token log-probability over all assigned frames (confidence proxy).
    mean_score: float = 0.0


# A single token claiming more than this fraction of the frames is treated as a
# collapsed/degenerate alignment: the acoustic model failed to distinguish the
# tokens and the DP dumped the whole audio into one state.
#
# NOTE: many single-frame tokens is NOT a collapse signal on its own — a
# character-level CTC model on sustained singing naturally emits each character
# as a short spike with blanks (and silence/held notes) in between.
_MAX_SPAN_FRACTION = 0.5


def _degeneracy_reasons(spans: list[TokenSpan | None], T: int) -> list[str]:
    valid = [s for s in spans if s is not None]
    n = len(valid)
    if n == 0 or T <= 0:
        return ["no usable spans"]
    reasons: list[str] = []
    top = max(s.num_frames for s in valid)
    frac = top / T
    if n >= 4 and frac > _MAX_SPAN_FRACTION:
        reasons.append(f"collapsed: one token spans {frac:.0%} of the audio")
    return reasons


def minimum_frames(targets: Sequence[int]) -> int:
    """Minimum frames needed to emit ``targets`` under CTC rules.

    Each token needs at least one frame; every pair of *identical adjacent*
    tokens additionally needs at least one blank frame between them.
    """
    n = len(targets)
    if n == 0:
        return 0
    extra = sum(1 for k in range(n - 1) if targets[k] == targets[k + 1])
    return n + extra


def _spans_are_monotonic(spans: list[TokenSpan | None]) -> bool:
    previous_end = -1
    for span in spans:
        if span is None:
            continue
        if span.start_frame <= previous_end:
            return False
        if span.end_frame < span.start_frame:
            return False
        previous_end = span.end_frame
    return True


def forced_align(
    log_probs: np.ndarray,
    targets: Sequence[int],
    blank: int = 0,
    strict: bool = False,
) -> Alignment:
    """CTC forced alignment.

    Parameters
    ----------
    log_probs:
        ``(T, C)`` array of log-probabilities (e.g. ``log_softmax`` of logits).
    targets:
        Target token ids, blanks excluded.
    blank:
        Index of the CTC blank symbol.
    strict:
        When ``True``, raise :class:`InfeasibleAlignmentError` instead of
        returning an ``Alignment`` with ``feasible=False``.

    Returns
    -------
    :class:`Alignment`. ``spans[k]`` is ``None`` only for a token that received
    zero frames; that never happens for a feasible alignment.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    if lp.ndim != 2:
        raise ValueError(f"log_probs must be 2-D, got shape {lp.shape}")
    T, C = lp.shape
    targets = [int(t) for t in targets]
    N = len(targets)
    required = minimum_frames(targets)

    def _result(spans, total, path, feasible, reasons, mean_score=0.0):
        if strict and not feasible:
            raise InfeasibleAlignmentError("; ".join(reasons) or "infeasible alignment")
        return Alignment(
            spans=spans,
            total_log_prob=total,
            path=path,
            feasible=feasible,
            required_frames=required,
            reasons=reasons,
            mean_score=mean_score,
        )

    if N == 0:
        return _result([], 0.0, np.zeros(T, dtype=np.int64), True, [])

    for tok in targets:
        if not 0 <= tok < C:
            raise ValueError(f"target token {tok} out of range [0, {C})")

    if T == 0:
        return _result(
            [None] * N,
            NEG_INF,
            np.zeros(0, dtype=np.int64),
            False,
            ["no frames (T=0)"],
        )
    if T < required:
        return _result(
            [None] * N,
            NEG_INF,
            np.zeros(0, dtype=np.int64),
            False,
            [f"too few frames: T={T} < required={required} for {N} tokens"],
        )

    S = 2 * N + 1
    ext = np.full(S, blank, dtype=np.int64)
    for k, tok in enumerate(targets):
        ext[2 * k + 1] = tok

    dp = np.full(S, NEG_INF, dtype=np.float64)
    dp[0] = lp[0, ext[0]]
    if S > 1:
        dp[1] = lp[0, ext[1]]

    # backpointers[t-1][s] in {0: stay, 1: from s-1, 2: from s-2}
    backptr = np.zeros((max(T - 1, 0), S), dtype=np.int8)

    for t in range(1, T):
        prev = dp
        c1 = prev  # stay on same state (repeat token / blank)
        c2 = np.full(S, NEG_INF, dtype=np.float64)
        c2[1:] = prev[:-1]
        c3 = np.full(S, NEG_INF, dtype=np.float64)
        if S > 2:
            # Allow skipping the blank between two *different* tokens.
            allow = ext[2:] != ext[:-2]
            tmp = np.full(S - 2, NEG_INF, dtype=np.float64)
            tmp[allow] = prev[:-2][allow]
            c3[2:] = tmp
        stack = np.stack([c1, c2, c3], axis=0)  # (3, S)
        idx = np.argmax(stack, axis=0).astype(np.int8)
        best = stack[idx, np.arange(S)]
        dp = best + lp[t, ext]
        backptr[t - 1] = idx

    total = float(np.max(dp))

    if S == 1:
        end_state = 0
    else:
        end_state = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2

    path = np.empty(T, dtype=np.int64)
    s = int(end_state)
    for t in range(T - 1, -1, -1):
        path[t] = s
        if t > 0:
            s = s - int(backptr[t - 1][s])

    spans: list[TokenSpan | None] = []
    missing: list[int] = []
    for k in range(N):
        state = 2 * k + 1
        frames = np.nonzero(path == state)[0]
        if frames.size == 0:
            spans.append(None)
            missing.append(k)
            continue
        start_f = int(frames[0])
        end_f = int(frames[-1])
        tok = targets[k]
        score = float(np.mean(lp[start_f : end_f + 1, tok]))
        spans.append(TokenSpan(token=tok, start_frame=start_f, end_frame=end_f, score=score))

    reasons: list[str] = []
    if not np.isfinite(total) or total <= NEG_INF / 2:
        reasons.append("all paths have -inf probability")
    if missing:
        reasons.append(f"{len(missing)} token(s) received zero frames: {missing[:8]}")
    if not _spans_are_monotonic(spans):
        reasons.append("spans are not monotonic")
    reasons.extend(_degeneracy_reasons(spans, T))
    feasible = not reasons
    valid_scores = [s.score for s in spans if s is not None]
    mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0
    return _result(spans, total, path, feasible, reasons, mean_score)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    return (x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))).astype(np.float32)
