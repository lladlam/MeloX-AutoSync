"""JSON timeline export.

We emit a self-describing document: the original lyric text is preserved, each
line and each alignment unit carries start/end times, and every timestamp is
tagged with its ``source`` (``model`` / ``heuristic`` / ``failed``) so a
downstream consumer can never mistake a guess for an acoustic measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..quality.gating import player_timeline, summarize
from ..types import AlignmentResult

SCHEMA_VERSION = "0.2"


def result_to_dict(result: AlignmentResult) -> dict:
    data = result.to_dict()
    data["schema_version"] = SCHEMA_VERSION
    data["generator"] = "MeloX-AutoSync"
    # V0.7: quality states and the conservative player timeline (LRC fallback)
    data["quality_summary"] = summarize(result.lines)
    data["player_timeline"] = player_timeline(result)
    return data


def write_json(result: AlignmentResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result_to_dict(result)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
