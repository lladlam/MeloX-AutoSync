#!/usr/bin/env python
"""Download a curated subset of GTSinger (CC BY-NC-SA 4.0) to an arbitrary dir.

Unlike ``tools/fetch_gtsinger.py`` (which has a fixed spec list and writes into
``experiments/singing/``), this tool takes an explicit list of
``language/singer/technique/song`` specs and an output directory, so a caller
can build disjoint calibration / validation pools and keep the data off the
repository disk.

It downloads ``.wav`` + ``.json`` (+ ``.TextGrid`` when present) per segment,
stops at the first missing index, verifies that annotation end times stay inside
the audio duration, and writes a ``manifest.json`` with per-file SHA-256.

Usage::

    python tools/fetch_gtsinger_subset.py --out /path/to/pool --max-segments 3 \
        --spec English/EN-Alto-2/Vibrato/A\\ Thousand\\ Years \
        --spec Japanese/JA-Tenor-1/Breathy/さんぽ

Data is **never** committed; the caller is responsible for accepting the licence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import wave
from pathlib import Path

import requests

BASE = "https://huggingface.co/datasets/AaronZ345/GTSinger/resolve/main/"
LICENCE = "CC BY-NC-SA 4.0"
SOURCE = "https://huggingface.co/datasets/AaronZ345/GTSinger"
LANGS = {"English": "en", "Japanese": "ja", "Chinese": "zh", "French": "fr",
         "German": "de", "Italian": "it", "Korean": "ko", "Russian": "ru", "Spanish": "es"}


def _get(session: requests.Session, rel: str) -> bytes | None:
    url = BASE + urllib.parse.quote(rel, safe="/")
    r = session.get(url, timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def fetch(out: Path, spec: str, max_segments: int, groups: list[str]) -> list[dict]:
    lang, singer, tech, song = spec.split("/", 3)
    records: list[dict] = []
    session = requests.Session()
    for group in groups:
        for i in range(max_segments):
            rel = f"{lang}/{singer}/{tech}/{song}/{group}/{i:04d}"
            wav = _get(session, f"{rel}.wav")
            if wav is None:
                break
            ann = _get(session, f"{rel}.json")
            if ann is None:
                raise RuntimeError(f"{rel}.wav present but .json missing")
            tg = _get(session, f"{rel}.TextGrid")
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            (dest.with_suffix(".wav")).write_bytes(wav)
            (dest.with_suffix(".json")).write_bytes(ann)
            if tg is not None:
                (dest.with_suffix(".TextGrid")).write_bytes(tg)
            with wave.open(str(dest.with_suffix(".wav")), "rb") as fh:
                dur = fh.getnframes() / fh.getframerate()
            words = json.loads(ann.decode("utf-8"))
            end = max((float(w.get("end_time", 0.0)) for w in words if isinstance(w, dict)), default=0.0)
            if end > dur + 0.1:
                raise RuntimeError(f"{rel}: annotation end {end:.2f}s > audio {dur:.2f}s")
            records.append({
                "language": LANGS.get(lang, lang), "singer": singer, "technique": tech,
                "song": song, "group": group, "stem": f"{i:04d}", "rel": rel,
                "duration_s": dur, "n_units": len(words),
                "wav_sha256": hashlib.sha256(wav).hexdigest(),
                "wav_bytes": len(wav), "json_bytes": len(ann),
            })
            print(f"  ok {rel} ({dur:.2f}s, {len(words)} units)", flush=True)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", action="append", required=True,
                    help="language/singer/technique/song (repeatable)")
    ap.add_argument("--group", action="append", default=None,
                    help="group dir name(s), e.g. Vibrato_Group / Control_Group (default: both)")
    ap.add_argument("--max-segments", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = args.group or None
    all_records: list[dict] = []
    for spec in args.spec:
        tech = spec.split("/")[2]
        use_groups = groups or ([f"{tech}_Group"] + ([] if tech == "Control" else ["Control_Group"]))
        print(f"[{spec}] groups={use_groups}", flush=True)
        try:
            all_records += fetch(out, spec, args.max_segments, use_groups)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  !! {type(exc).__name__}: {exc}", flush=True)

    manifest = out / "manifest.json"
    payload = {"source": SOURCE, "licence": LICENCE,
               "note": "local research only; never redistribute", "records": all_records}
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mb = sum(r["wav_bytes"] + r["json_bytes"] for r in all_records) / 1e6
    print(f"\n{len(all_records)} segments ({mb:.1f} MB) -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
