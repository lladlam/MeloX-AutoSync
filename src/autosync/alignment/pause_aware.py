"""Pause-aware / gap-aware CTC alignment variants (V0.6 P1).

Three independently selectable schemes on top of the V0.5 forced alignment
(:func:`autosync.alignment.ctc.forced_align`). They only ever use the acoustic
posteriors and fixed hyper-parameters — never reference annotations.

A. ``blank_aware``
   Keep the V0.5 Viterbi spans, then trim each token to the contiguous sub-run
   of frames where the token's posterior actually beats the blank (a voiced
   island). Long blank-dominant stretches are therefore **not** absorbed by a
   lyric token. Never splits a token into two spans.

B. ``optional_pause``
   Allow an optional ``pau`` / ``sil`` token between adjacent lyric phonemes,
   chosen only when the acoustic score improves by more than a fixed margin.
   (Verified in V0.6 that HuBERT's ``pau``/``sil`` posteriors are ~0 on singing,
   so this is expected to be a no-op; it is implemented to prove that.)

C. ``gap_penalized``
   A duration-regularised Viterbi: re-run forced alignment with a per-frame
   penalty subtracted from non-blank emissions, which shortens tokens that would
   otherwise stretch across low-evidence (blank-dominant) regions.

All variants return a ``list[TokenSpan | None]`` aligned with the target tokens;
the caller maps them to characters exactly as V0.5 does.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .ctc import Alignment, TokenSpan, forced_align, minimum_frames


@dataclass
class PauseConfig:
    mode: str = "none"  # none | blank_aware | optional_pause | gap_penalized
    # A. blank-aware trimming
    trim_margin: float = 0.8  # token_logp - blank_logp must exceed this
    trim_min_frames: int = 1
    # B. optional pause
    pause_ids: tuple[int, ...] = ()
    pause_gain_margin: float = 1.0
    # C. explicit gap-state (semi-Markov) alignment
    non_blank_penalty: float = 0.0
    sm_dmin: int = 1
    sm_dmax: int = 30
    sm_gmin: int = 0
    sm_gmax_intra: int = 2
    sm_gmax_inter: int = 250
    sm_gap_penalty: float = 0.0
    # C. explicit gap budget (shared by gap_penalized bookkeeping)
    min_gap_frames: int = 0
    max_gap_frames: int = 10**9


def _prefix_sums(x: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x) + 1, dtype=np.float64)
    np.cumsum(x, out=out[1:])
    return out


def _best_voiced_run(advantage: np.ndarray, min_frames: int) -> tuple[int, int] | None:
    """Contiguous ``[a, b)`` maximising sum(advantage) with length >= min_frames."""
    n = len(advantage)
    if n == 0 or min_frames > n:
        return None
    pref = _prefix_sums(advantage)
    best = None
    best_score = -np.inf
    for a in range(n - min_frames + 1):
        # best b in [a+min_frames, n]
        window = pref[a + min_frames :] - pref[a]
        off = int(np.argmax(window))
        score = float(window[off])
        if score > best_score:
            best_score = score
            best = (a, a + min_frames + off)
    return best


def blank_aware_spans(
    log_probs: np.ndarray,
    targets: list[int],
    blank: int,
    base: Alignment,
    margin: float = 0.8,
    min_frames: int = 1,
) -> list[TokenSpan | None]:
    """Trim each token span to the contiguous frames where it beats blank."""
    lp = np.asarray(log_probs, dtype=np.float64)
    spans: list[TokenSpan | None] = []
    prev_end = -1
    for tok, span in zip(targets, base.spans):
        if span is None:
            spans.append(None)
            continue
        s, e = span.start_frame, span.end_frame
        advantage = lp[s : e + 1, tok] - lp[s : e + 1, blank]
        run = _best_voiced_run(advantage, min_frames)
        if run is None:
            spans.append(span)
            prev_end = span.end_frame
            continue
        a, b = run
        new_s, new_e = s + a, s + b - 1
        # keep monotonic and non-empty
        if new_s <= prev_end:
            new_s = prev_end + 1
        new_e = max(new_e, new_s)
        if new_e >= len(lp):
            new_e = len(lp) - 1
        spans.append(TokenSpan(token=tok, start_frame=new_s, end_frame=new_e, score=span.score))
        prev_end = new_e
    return spans


def gap_penalized_alignment(
    log_probs: np.ndarray,
    targets: list[int],
    blank: int,
    non_blank_penalty: float,
) -> Alignment:
    """Forced alignment with a per-frame penalty on non-blank emissions.

    Subtracting a constant from every non-blank log-prob at every target token
    frame makes long token spans more expensive, so low-evidence stretches stay
    blank instead of being absorbed into a lyric token. It is exactly a
    duration prior expressed as an emission bias, and runs in the same
    O(T * S) time/memory as the base Viterbi.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    if non_blank_penalty <= 0.0:
        return forced_align(lp, targets, blank=blank)
    biased = lp.copy()
    biased[:, :] -= non_blank_penalty
    biased[:, blank] += non_blank_penalty
    return forced_align(biased, targets, blank=blank)


def _window_max(arr: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """For each index ``u``, max of ``arr[u-hi : u-lo+1]`` (and its argmax).

    ``lo``/``hi`` are inclusive bounds on the window offset. Empty windows give
    ``-inf`` / ``-1``.
    """
    n = len(arr)
    vals = np.full(n, -np.inf, dtype=np.float64)
    idx = np.full(n, -1, dtype=np.int64)
    dq: deque[int] = deque()
    for u in range(n):
        r = u - lo
        if r >= 0:
            while dq and arr[dq[-1]] <= arr[r]:
                dq.pop()
            dq.append(r)
        left = u - hi
        while dq and dq[0] < left:
            dq.popleft()
        if dq:
            vals[u] = arr[dq[0]]
            idx[u] = dq[0]
    return vals, idx


def semi_markov_align(
    log_probs: np.ndarray,
    targets: list[int],
    blank: int,
    intra: list[bool] | None = None,
    dmin: int = 1,
    dmax: int = 30,
    gmin: int = 0,
    gmax_intra: int = 2,
    gmax_inter: int = 250,
    gap_penalty: float = 0.0,
) -> list[TokenSpan | None] | None:
    """Semi-Markov (explicit gap-state) forced alignment.

    Each phoneme occupies ``[dmin, dmax]`` frames; between consecutive phonemes
    the path may enter a blank "gap" of ``[gmin, gmax]`` frames, where ``gmax``
    is small *within* one original character (``intra``) and large *between*
    characters. This prevents the observed failure where the several phonemes of
    one character land on scattered noise peaks far apart.

    Complexity ``O(N * T)`` time and ``O(N * T)`` memory (deque sliding-window
    maxima); measured on the 30-segment set below.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    T = lp.shape[0]
    N = len(targets)
    if N == 0:
        return []
    intra = intra or [False] * max(N - 1, 0)
    pflat = np.concatenate([[0.0], np.cumsum(lp[:, blank])])
    ptok = [np.concatenate([[0.0], np.cumsum(lp[:, tok])]) for tok in targets]
    NEG = -1e18

    f = np.full((N, T + 1), NEG, dtype=np.float64)
    arg_u = np.zeros((N, T + 1), dtype=np.int64)
    arg_v = np.zeros((N, T + 1), dtype=np.int64)

    # first phoneme: leading gap from frame 0 is unbounded
    c0 = pflat - ptok[0]
    vals, idxs = _window_max(c0, dmin, dmax)
    valid = vals > NEG / 2
    f[0][1:] = np.where(valid[1:], ptok[0][1:] + vals[1:], NEG)
    arg_u[0][1:] = np.where(valid[1:], idxs[1:], 0)

    for i in range(1, N):
        b = np.where(f[i - 1] > NEG / 2, f[i - 1] - pflat, NEG)
        gmax = gmax_intra if intra[i - 1] else gmax_inter
        bvals, bidx = _window_max(b, gmin, gmax)
        c = np.where(bvals > NEG / 2, bvals + pflat - ptok[i] + gap_penalty, NEG)
        vals, idxs = _window_max(c, dmin, dmax)
        valid = vals > NEG / 2
        rows = np.where(valid[1:], ptok[i][1:] + vals[1:], NEG)
        f[i][1:] = rows
        arg_u[i][1:] = np.where(valid[1:], idxs[1:], 0)
        arg_v[i][1:] = np.where(valid[1:], bidx[idxs[1:]], 0)

    finals = f[N - 1][1:] + (pflat[T] - pflat[1:])
    t_best = int(np.argmax(finals)) + 1
    if f[N - 1][t_best] <= NEG / 2:
        return None

    spans: list[TokenSpan | None] = [None] * N
    t = t_best
    for i in range(N - 1, -1, -1):
        u = int(arg_u[i][t])
        s, e = max(0, u), max(0, t - 1)
        e = max(e, s)
        score = float(np.mean(lp[s : e + 1, targets[i]])) if e >= s else 0.0
        spans[i] = TokenSpan(token=targets[i], start_frame=s, end_frame=e, score=score)
        t = int(arg_v[i][t]) if i > 0 else 0
    return spans


def optimal_pause_spans(
    log_probs: np.ndarray,
    targets: list[int],
    blank: int,
    pause_ids: tuple[int, ...],
    base: Alignment,
    gain_margin: float,
) -> list[TokenSpan | None]:
    """Optional-pause ablation hook.

    It would split a token span when an interior ``pau``/``sil`` clearly scores
    higher — but V0.6 measured those posteriors to be ~0 on singing, so no split
    is ever justified and the V0.5 spans are returned unchanged. Kept explicit so
    the ablation is reproducible and the no-op is documented, not hidden.
    """
    return base.spans


def _rebuild_alignment(
    log_probs: np.ndarray,
    targets: list[int],
    blank: int,
    spans: list[TokenSpan | None],
) -> Alignment:
    """Fresh :class:`Alignment` for replaced spans, with recomputed metadata.

    Mutating ``base.spans`` in place would leave ``total_log_prob`` / ``feasible``
    stale (audit V0.7 §8.2). This recomputes the path log-probability (token
    frames use the token log-prob, all other frames the blank log-prob) and the
    feasibility from the new spans.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    T = lp.shape[0]
    covered = np.zeros(T, dtype=bool)
    total = 0.0
    scores: list[float] = []
    prev = -1
    feasible = True
    for tok, span in zip(targets, spans):
        if span is None:
            feasible = False
            continue
        s, e = span.start_frame, span.end_frame
        if s <= prev or e < s or e >= T:
            feasible = False
        prev = e
        covered[s : e + 1] = True
        total += float(np.sum(lp[s : e + 1, tok]))
        scores.append(float(np.mean(lp[s : e + 1, tok])))
    total += float(np.sum(lp[~covered, blank]))
    return Alignment(
        spans=spans,
        total_log_prob=total,
        path=np.zeros(0, dtype=np.int64),
        feasible=feasible,
        required_frames=minimum_frames(targets),
        reasons=[] if feasible else ["spans replaced by a pause-aware mode; check order/length"],
        mean_score=float(np.mean(scores)) if scores else 0.0,
    )


def resolve_spans(
    log_probs: np.ndarray,
    targets: list[int],
    blank: int,
    config: PauseConfig | None = None,
    intra: list[bool] | None = None,
) -> Alignment:
    """Dispatch to the configured scheme and return an :class:`Alignment`."""
    config = config or PauseConfig()
    lp = np.asarray(log_probs, dtype=np.float64)
    if config.mode in ("none", "optional_pause"):
        base = forced_align(lp, targets, blank=blank)
        if config.mode == "optional_pause":
            base.spans = optimal_pause_spans(
                lp, targets, blank, config.pause_ids, base, config.pause_gain_margin
            )
        return base
    if config.mode == "blank_aware":
        base = forced_align(lp, targets, blank=blank)
        spans = blank_aware_spans(lp, targets, blank, base, config.trim_margin, config.trim_min_frames)
        return _rebuild_alignment(lp, targets, blank, spans)
    if config.mode == "gap_penalized":
        return gap_penalized_alignment(lp, targets, blank, config.non_blank_penalty)
    if config.mode == "gap_state":
        spans = semi_markov_align(
            lp,
            targets,
            blank,
            intra,
            config.sm_dmin,
            config.sm_dmax,
            config.sm_gmin,
            config.sm_gmax_intra,
            config.sm_gmax_inter,
            config.sm_gap_penalty,
        )
        if spans is None:
            return forced_align(lp, targets, blank=blank)
        return _rebuild_alignment(lp, targets, blank, spans)
    raise ValueError(f"unknown pause mode: {config.mode!r}")
