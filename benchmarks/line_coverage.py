#!/usr/bin/env python
"""Per-line coverage accounting: every line gets an explicit outcome.

V1.2 found that lyrics like 提瓦特民谣's Latin-only backing lines were dropped by
``normalize_cjk`` with no record. This tool enumerates **every** line of a song
and records its outcome so nothing can vanish:

* ``metadata`` / ``instrumental`` / ``unknown`` -> not singable (by classification)
* ``singing`` -> routed to a model (zh / ja / en) or explicitly ``not_routable``
  with a reason. It is never removed from the coverage denominator.

It prints no lyric text (only counts, kinds, languages and reasons).

Usage::

    .venv/bin/python benchmarks/line_coverage.py --song "提瓦特民谣" --primary zh --out cov.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autosync.lyrics.classify import classify_lines
from autosync.lyrics.language import detect_line_language
from autosync.lyrics.lrc import parse_lrc


def find_audio(root: Path, song: str) -> Path | None:
    for ext in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
        p = root / f"{song}{ext}"
        if p.exists():
            return p
    for p in root.glob(f"{song}*"):
        if p.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a", ".ogg"}:
            return p
    return None


def coverage(song: str, primary: str, root: Path) -> dict:
    lrc = next((p for p in root.glob(f"{song}*.lrc")), None)
    if lrc is None:
        raise FileNotFoundError(f"no lrc for {song!r} under {root}")
    lines, _meta = parse_lrc(lrc)
    kinds = classify_lines(lines)

    kinds_count: Counter = Counter()
    singing_routed: Counter = Counter()
    not_routable: list[dict] = []
    singing_total = 0
    for i, line in enumerate(lines):
        kind = kinds[i].kind.value
        kinds_count[kind] += 1
        if not kinds[i].kind.alignable:
            continue
        singing_total += 1
        d = detect_line_language(line.text, primary)
        if d.routable:
            singing_routed[d.language] += 1
        else:
            not_routable.append({"line_index": line.index, "lang": d.language,
                                 "reason": d.reason, "script": d.script_summary})
    covered = sum(singing_routed.values()) + len(not_routable)
    return {
        "song": song,
        "primary": primary,
        "lrc": lrc.name,
        "audio_present": find_audio(root, song) is not None,
        "total_lines": len(lines),
        "by_kind": dict(kinds_count),
        "singing_total": singing_total,
        "singing_routed": dict(singing_routed),
        "singing_not_routable": len(not_routable),
        "not_routable_examples": not_routable[:5],
        # accounting invariant: every singing line has exactly one outcome
        "denominator": singing_total,
        "accounted_for": covered,
        "unaccounted": singing_total - covered,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--song", required=True)
    ap.add_argument("--primary", default="zh", choices=["zh", "ja", "en"])
    ap.add_argument("--root", type=Path, default=ROOT / "Test")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    res = coverage(args.song, args.primary, args.root)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert res["unaccounted"] == 0, "accounting invariant violated"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
