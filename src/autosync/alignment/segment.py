"""Per-line alignment pipeline: crop audio, run a model, map back to song time.

This is where the engine logic lives. It is model-agnostic and calls only the
:class:`~autosync.models.base.BaseAligner` interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..audio.io import crop, duration_seconds
from ..lyrics.normalize import normalize_cjk, normalize_en, normalize_for
from ..types import AlignedLine, AlignUnit, LyricLine
from .heuristic import distribute
from .refine import refine_units_energy

log = logging.getLogger("autosync.segment")

# Default maximum line duration when the next line is far away (seconds).
MAX_LINE_SECONDS = 12.0


@dataclass
class SegmentOptions:
    padding: float = 0.30
    fallback: str = "none"  # "none" | "heuristic"
    max_line_seconds: float = MAX_LINE_SECONDS
    # V0.2: post-alignment boundary refinement ("none" | "energy").
    refine: str = "none"
    refine_max_extend: float = 0.6


def _window_for_line(
    line: LyricLine,
    next_line: LyricLine | None,
    total_duration: float,
    padding: float,
    max_line_seconds: float,
) -> tuple[float, float]:
    """Return the audio window ``(start, end)`` for one lyric line.

    The window starts at the line's timestamp and extends to the next line's
    timestamp (or a bounded default). Padding is added on both sides but never
    beyond the neighbours, to avoid bleeding adjacent lines into this one.
    """
    if next_line is not None and next_line.start > line.start:
        end = next_line.start
    else:
        # Last line: extend to the end of the audio. The CTC aligner assigns
        # trailing silence/instrumental to the blank label, so this is safe and
        # avoids truncating a long final line.
        end = total_duration
    end = min(end, total_duration)

    start = max(0.0, line.start - padding)
    end = min(total_duration, end + padding)
    if end <= start:
        end = min(total_duration, start + max(0.2, max_line_seconds))
    return start, end


def _sanitize(unz: list[AlignUnit], window_len: float) -> tuple[list[AlignUnit], str | None]:
    """Clamp units into the window and enforce monotonic, positive durations."""
    msg: str | None = None
    out: list[AlignUnit] = []
    last_end = 0.0
    for u in unz:
        s = float(max(0.0, min(u.start, window_len)))
        e = float(max(0.0, min(u.end, window_len)))
        if e < s:
            s, e = e, s
            msg = "swapped inverted unit boundary"
        if s < last_end:
            # Overlap with previous unit: clamp and note it.
            s = last_end
            msg = "clamped overlapping unit boundary"
        if e <= s:
            e = min(window_len, s + 0.01)
        out.append(
            AlignUnit(
                text=u.text,
                start=s,
                end=e,
                source=u.source,
                score=u.score,
                raw=u.raw,
                uncertain=u.uncertain,
            )
        )
        last_end = e
    return out, msg


def check_reconstruction(language: str, text: str, units: list[AlignUnit]) -> str | None:
    """Verify that the aligned units reproduce the original lyric text.

    Returns ``None`` when the units reconstruct the normalised line, otherwise a
    human-readable mismatch description. This makes "the original text can be
    recovered" a checked property rather than an assumption.
    """
    if not units:
        return None
    try:
        expected = normalize_for(language, text)
    except ValueError:
        return None
    joined = " ".join(u.text for u in units) if language == "en" else "".join(u.text for u in units)
    got = normalize_en(joined) if language == "en" else normalize_cjk(joined)
    if got != expected:
        return (
            f"text reconstruction mismatch: reconstructed {got!r} "
            f"but expected {expected!r}"
        )
    return None


def align_lines(
    audio: np.ndarray,
    sample_rate: int,
    lines: list[LyricLine],
    aligner,
    options: SegmentOptions | None = None,
    progress: bool = True,
    language: str | None = None,
) -> tuple[list[AlignedLine], list[str]]:
    """Align every lyric line against ``audio``.

    Returns ``(aligned_lines, warnings)``. Absolute song time is used in the
    output; units returned by the aligner are window-relative and mapped back.
    """
    options = options or SegmentOptions()
    total = duration_seconds(audio, sample_rate)
    warnings: list[str] = []
    results: list[AlignedLine] = []
    infer_seconds = 0.0

    iterator = lines
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(lines, desc="aligning lines", unit="line")
        except ImportError:
            iterator = lines

    for i, line in enumerate(iterator):
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        w_start, w_end = _window_for_line(
            line, next_line, total, options.padding, options.max_line_seconds
        )
        window = crop(audio, sample_rate, w_start, w_end, pad=0.0)
        win_len = len(window) / sample_rate

        units: list[AlignUnit] = []
        err: str | None = None
        if window.size == 0 or win_len < 0.05:
            err = "empty or too-short audio window"
        else:
            import time

            t0 = time.time()
            if hasattr(aligner, "last_quality"):
                aligner.last_quality = None
            try:
                units = aligner.align_text(window, sample_rate, line.text)
            except Exception as exc:  # noqa: BLE001 - record any model failure honestly
                err = f"{type(exc).__name__}: {exc}"
                log.warning("line %d alignment failed: %s", line.index, err)
            infer_seconds += time.time() - t0

        units, san_msg = _sanitize(units, win_len)
        if san_msg:
            warnings.append(f"line {line.index}: {san_msg}")

        # V0.2: extend unit ends over voiced frames (opt-in).
        if units and options.refine == "energy":
            units = refine_units_energy(
                units,
                window,
                sample_rate,
                frame_seconds=getattr(aligner, "frame_seconds", 0.02) or 0.02,
                max_extend=options.refine_max_extend,
            )

        # Make "original text is recoverable" a verified property.
        if units and language is not None:
            rec = check_reconstruction(language, line.text, units)
            if rec:
                warnings.append(f"line {line.index}: {rec}")

        if not units:
            if options.fallback == "heuristic":
                disp = [line.text] if line.text.strip() else []
                if disp:
                    end_ref = next_line.start if next_line and next_line.start > line.start else (
                        min(total, line.start + options.max_line_seconds)
                    )
                    units = distribute(0.0, max(0.05, end_ref - line.start), disp)
                    src = "heuristic"
                else:
                    src = "failed"
            else:
                src = "failed"
            if err is None and src == "failed":
                err = err or "no aligned units produced"
        else:
            src = "model"

        abs_units = [
            AlignUnit(
                text=u.text,
                start=w_start + u.start,
                end=w_start + u.end,
                source=u.source,
                score=u.score,
                raw=u.raw,
                uncertain=u.uncertain,
            )
            for u in units
        ]
        line_start = abs_units[0].start if abs_units else line.start
        line_end = abs_units[-1].end if abs_units else line_start

        # V0.7: attach the per-line acoustic quality assessment, if the aligner
        # produced one. This never changes the timings themselves.
        quality = getattr(aligner, "last_quality", None)
        q_state = quality.get("state") if isinstance(quality, dict) else None
        q_reasons = list(quality.get("reasons", [])) if isinstance(quality, dict) else []

        results.append(
            AlignedLine(
                index=line.index,
                text=line.text,
                start=line_start,
                end=line_end,
                units=abs_units,
                source=src,
                error=err,
                quality=q_state,
                quality_reasons=q_reasons,
                lrc_start=line.start,
                lrc_end=line.end,
            )
        )

    try:
        aligner.infer_seconds = infer_seconds
    except AttributeError:  # pragma: no cover - all shipped adapters expose it
        log.debug("aligner does not expose infer_seconds")
    return results, warnings
