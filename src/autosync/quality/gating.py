"""Player-facing fallback policy for the three quality states (V0.7/V0.8).

The gate never overwrites, alters or discards the original lyrics or the original
LRC timestamps.  :func:`player_timeline` keeps **four independent times** per
line:

* ``lrc_start`` / ``lrc_end`` – the original LRC times (``lrc_end`` is ``null``
  when the LRC has no explicit end; it is never fabricated);
* ``model_start`` / ``model_end`` – the model prediction (``null`` when the line
  did not align);
* ``display_end`` – a *derived* hint for the player (the next line's original LRC
  start), explicitly marked via ``display_end_source``.

Modes
-----
* ``production`` (default) – word timeline only for ``RELIABLE`` lines; when a
  language gate is enabled, an unassessed line is treated as ``UNCERTAIN`` and
  safely falls back. Non-gated languages (en/zh) keep word mode.
* ``experimental`` – additionally allow the diagnostic word timeline for
  ``UNCERTAIN`` lines (never for ``FAILED``).

FAILED/UNCERTAIN fall back to the **original LRC times and text** (line level).
"""

from __future__ import annotations

from typing import Any

from ..types import AlignedLine, AlignmentResult
from .decision import QualityPolicy, QualityState

# Default policy is intentionally conservative and language-scoped. English and
# Chinese are NOT gated (their calibration is out of scope).
DEFAULT_POLICIES: dict[str, QualityPolicy | None] = {
    "ja": QualityPolicy(name="ja-hubert-conservative-v07", max_intra_char_span_s=1.5, max_char_duration_ratio=10.0),
    "en": None,
    "zh": None,
}


def summarize(lines: list[AlignedLine]) -> dict[str, int]:
    counts = {s.value: 0 for s in QualityState}
    counts["not_assessed"] = 0
    for line in lines:
        if line.source == "skipped":
            continue  # non-singing lines have no quality decision
        if line.quality in counts:
            counts[line.quality] += 1
        else:
            counts["not_assessed"] += 1
    return counts


def summarize_kinds(lines: list[AlignedLine]) -> dict[str, int]:
    """Count lines by classification kind (V0.9)."""
    counts = {"singing": 0, "metadata": 0, "instrumental": 0, "unknown": 0, "skipped": 0}
    for line in lines:
        if line.source == "skipped":
            counts["skipped"] += 1
        key = line.kind or "singing"
        counts[key] = counts.get(key, 0) + 1
    return counts


def gate_result(
    result: AlignmentResult,
    policy: QualityPolicy | None = None,
    mode: str = "production",
) -> AlignmentResult:
    """Resolve not-assessed lines and record the gate configuration.

    When a policy is active, any **model** line that was not assessed (quality
    analysis missing or failed) is conservatively marked ``UNCERTAIN`` so it
    falls back rather than being treated as reliable.
    """
    enabled = policy is not None
    if enabled:
        for line in result.lines:
            if line.quality is None and line.source == "model":
                line.quality = QualityState.UNCERTAIN.value
                line.quality_reasons = ["not assessed by the quality analyzer; safe fallback"]
    result.quality_summary = summarize(result.lines)
    result.diagnostics["quality_gate"] = {
        "enabled": enabled,
        "policy": policy.name if policy else None,
        "mode": mode,
    }
    return result


def player_timeline(result: AlignmentResult) -> list[dict[str, Any]]:
    """Build the player timeline with independent LRC / model / derived times."""
    gate = result.diagnostics.get("quality_gate", {}) if isinstance(result.diagnostics, dict) else {}
    enabled = bool(gate.get("enabled", False))
    mode = gate.get("mode", "production")

    lines = result.lines
    out: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        reliable = line.quality == QualityState.RELIABLE.value
        is_model = line.source in ("model", "heuristic") and bool(line.units)
        if line.quality == QualityState.FAILED.value:
            word_mode = False
        elif line.quality == QualityState.UNCERTAIN.value:
            # diagnostic timings only in experimental mode
            word_mode = is_model and mode == "experimental"
        elif reliable:
            word_mode = is_model
        else:  # not assessed
            word_mode = is_model and not enabled

        next_lrc = None
        if i + 1 < len(lines) and lines[i + 1].lrc_start is not None:
            next_lrc = lines[i + 1].lrc_start

        entry: dict[str, Any] = {
            "index": line.index,
            # original text is always preserved
            "text": line.text,
            # --- original LRC times (never fabricated) ---
            "lrc_start": line.lrc_start if line.lrc_start is not None else line.start,
            "lrc_end": line.lrc_end,  # null when the LRC has no explicit end
            # --- model prediction (null when not aligned) ---
            "model_start": line.start if line.source in ("model", "heuristic") and line.units else None,
            "model_end": line.end if line.source in ("model", "heuristic") and line.units else None,
            # --- derived display hint (explicitly not an original LRC end) ---
            "display_end": next_lrc,
            "display_end_source": "derived_from_next_lrc_start" if next_lrc is not None else None,
            "mode": "word" if word_mode else "line_lrc",
            "source": line.source,
            "quality": line.quality or "not_assessed",
            "kind": line.kind or "singing",
            "skip_reason": line.skip_reason,
            "reasons": line.quality_reasons,
        }
        if word_mode:
            entry["units"] = [
                {"text": u.text, "start": u.start, "end": u.end, "source": u.source} for u in line.units
            ]
        out.append(entry)
    return out
