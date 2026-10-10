#!/usr/bin/env python
"""Fetch a small, curated subset of GTSinger (CC BY-NC-SA 4.0) for local testing.

GTSinger is a *free-to-use, non-commercial* singing corpus with per-segment
Praat-style annotations (word + phoneme boundaries, technique flags).

IMPORTANT: this tool downloads raw audio **locally only** (into a gitignored
directory). Do NOT re-upload GTSinger audio to any public repository. See
``docs/SINGING_EVAL.md`` for the license discussion.

Usage:
    uv run python tools/fetch_gtsinger.py --out experiments/singing/gtsinger
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

BASE = "https://huggingface.co/datasets/GTSinger/GTSinger/resolve/main"

# (language, singer, technique_dir, song)
DEFAULT_SPECS: list[tuple[str, str, str, str]] = [
    # English (alto)
    ("English", "EN-Alto-1", "Vibrato", "edelweiss"),
    ("English", "EN-Alto-1", "Glissando", "edelweiss"),
    ("English", "EN-Alto-1", "Breathy", "edelweiss"),
    # Japanese (soprano)
    ("Japanese", "JA-Soprano-1", "Vibrato", "heartful song"),
    ("Japanese", "JA-Soprano-1", "Vibrato", "时雨"),
    ("Japanese", "JA-Soprano-1", "Glissando", "heartful song"),
    # Chinese (alto)
    ("Chinese", "ZH-Alto-1", "Vibrato", "大鱼"),
    ("Chinese", "ZH-Alto-1", "Vibrato", "南山南"),
    ("Chinese", "ZH-Alto-1", "Glissando", "十年"),
]


def _proxies() -> dict[str, str] | None:
    hp = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if hp:
        return {"http": hp, "https": hp}
    return None


def _get(url: str, timeout: int = 120) -> requests.Response | None:
    r = requests.get(url, timeout=timeout, proxies=_proxies())
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r


def _seg_url(lang: str, singer: str, tech: str, song: str, group: str, name: str) -> str:
    from urllib.parse import quote

    path = f"{lang}/{singer}/{tech}/{song}/{group}/{name}"
    return f"{BASE}/{quote(path)}"


def fetch_spec(
    out_root: Path,
    lang: str,
    singer: str,
    tech: str,
    song: str,
    max_segments: int,
) -> list[dict]:
    groups = [f"{tech}_Group"]
    if tech != "Control":
        groups.append("Control_Group")

    records: list[dict] = []
    for group in groups:
        for idx in range(max_segments):
            stem = f"{idx:04d}"
            rel_dir = Path(lang) / singer / tech / song / group
            ref_local = out_root / rel_dir / f"{stem}.json"
            wav_local = out_root / rel_dir / f"{stem}.wav"

            ref_url = _seg_url(lang, singer, tech, song, group, f"{stem}.json")
            if not ref_local.exists():
                resp = _get(ref_url)
                if resp is None:
                    break  # no more segments in this group
                ref_local.parent.mkdir(parents=True, exist_ok=True)
                ref_local.write_bytes(resp.content)

            if not wav_local.exists():
                wresp = _get(_seg_url(lang, singer, tech, song, group, f"{stem}.wav"))
                if wresp is None:
                    break
                wav_local.write_bytes(wresp.content)

            sha = hashlib.sha256(wav_local.read_bytes()).hexdigest()[:16]
            records.append(
                {
                    "language": lang,
                    "singer": singer,
                    "technique": tech,
                    "song": song,
                    "group": group,
                    "stem": stem,
                    "wav": str(wav_local.relative_to(out_root)),
                    "json": str(ref_local.relative_to(out_root)),
                    "wav_sha256_16": sha,
                    "wav_bytes": wav_local.stat().st_size,
                }
            )
            print(f"  ok {rel_dir}/{stem}.wav")
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="experiments/singing/gtsinger")
    ap.add_argument("--max-per-group", type=int, default=3)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    for lang, singer, tech, song in DEFAULT_SPECS:
        print(f"[{lang}] {singer} / {tech} / {song}")
        try:
            all_records += fetch_spec(out_root, lang, singer, tech, song, args.max_per_group)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  !! failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    manifest = out_root / "manifest.json"
    manifest.write_text(json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(all_records)} segments; manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
