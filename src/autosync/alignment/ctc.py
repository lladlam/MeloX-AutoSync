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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

NEG_INF = -1e30


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


def forced_align(
    log_probs: np.ndarray,
    targets: Sequence[int],
    blank: int = 0,
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

    Returns
    -------
    :class:`Alignment` with one :class:`TokenSpan` per target token. A span is
    ``None`` only when a target token got zero frames, which indicates the
    alignment is degenerate for that token.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    if lp.ndim != 2:
        raise ValueError(f"log_probs must be 2-D, got shape {lp.shape}")
    T, C = lp.shape
    targets = [int(t) for t in targets]
    N = len(targets)
    if N == 0:
        return Alignment(spans=[], total_log_prob=0.0, path=np.zeros(T, dtype=np.int64))
    if T == 0:
        return Alignment(
            spans=[None] * N, total_log_prob=NEG_INF, path=np.zeros(0, dtype=np.int64)
        )

    S = 2 * N + 1
    ext = np.full(S, blank, dtype=np.int64)
    for k, tok in enumerate(targets):
        if not 0 <= tok < C:
            raise ValueError(f"target token {tok} out of range [0, {C})")
        ext[2 * k + 1] = tok
    # Reject impossible transitions (adjacent identical tokens need a blank).
    for k in range(N - 1):
        if targets[k] == targets[k + 1]:
            pass  # handled automatically by the skip-mask below

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
    for k in range(N):
        state = 2 * k + 1
        frames = np.nonzero(path == state)[0]
        if frames.size == 0:
            spans.append(None)
            continue
        start_f = int(frames[0])
        end_f = int(frames[-1])
        tok = targets[k]
        score = float(np.mean(lp[start_f : end_f + 1, tok]))
        spans.append(TokenSpan(token=tok, start_frame=start_f, end_frame=end_f, score=score))

    return Alignment(spans=spans, total_log_prob=total, path=path)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    return (x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))).astype(np.float32)
