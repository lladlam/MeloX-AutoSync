#!/usr/bin/env python
"""Redact third-party lyric text from committed evaluation reports (V1.0 policy).

Public Git must contain **only aggregate metrics**. Evaluation reports keep all
numeric fields, structure and sample counts, but lyric-bearing string fields are
replaced with a fixed placeholder so that no third-party text is redistributed.

Usage:
    uv run python tools/redact_reports.py experiments/reports/*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PLACEHOLDER = "<redacted>"
REDACT_KEYS = {"text", "transcript", "char", "kana", "line_text", "unit_text", "raw"}


def redact(node):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in REDACT_KEYS and isinstance(v, str):
                out[k] = PLACEHOLDER if v else v
            else:
                out[k] = redact(v)
        return out
    if isinstance(node, list):
        return [redact(v) for v in node]
    return node


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for p in args.paths:
        path = Path(p)
        data = json.loads(path.read_text(encoding="utf-8"))
        redacted = redact(data)
        text = json.dumps(redacted, indent=2, ensure_ascii=False)
        if args.dry_run:
            print(f"would redact {path}")
            continue
        path.write_text(text, encoding="utf-8")
        print(f"redacted {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
