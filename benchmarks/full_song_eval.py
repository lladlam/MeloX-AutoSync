#!/usr/bin/env python
"""Full-song end-to-end evaluation over the local ``Test/`` folder (V1.0 P0/P3).

Runs the **real CLI** (same entry point users call) per song, then validates the
exported timeline: LRC/text preservation, monotonicity, cross-line bleeding,
out-of-range units, and the line-kind / quality / player-mode breakdown.

The third-party audio and lyrics under ``Test/`` are **never** committed; this
script only writes aggregate stats to ``experiments/reports/``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus")
# Song language is declared explicitly: 不安灵魂收容所 / 提瓦特民谣 are Chinese
# songs with English sections, so naive script counting would mis-classify them.
SONG_LANGUAGE = {
    "不安灵魂收容所": "zh",
    "提瓦特民谣": "zh",
    "SAN値直葬": "ja",
    "歌姫失格": "ja",
    "预言 Prophecy": "en",
    "DAMIDAMI": "en",
}

_WRAPPER = (
    "import sys, json, resource;"
    "from autosync.cli import main;"
    "rc = main(sys.argv[1:]);"
    "print('__STATS__' + json.dumps({'rc': rc,"
    " 'maxrss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}))"
)


def discover(test_dir: Path) -> list[dict]:
    audios = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    lrcs = sorted(test_dir.glob("*.lrc"))
    songs = []
    for audio in audios:
        stem = audio.stem
        exact = [l for l in lrcs if l.stem == stem]
        loose = [l for l in lrcs if l.stem.startswith((stem + "-", stem + " ", stem + "("))]
        cands = exact or loose
        if len(cands) != 1:
            raise SystemExit(f"ambiguous LRC match for {audio.name}: {[c.name for c in cands]}")
        songs.append({"song": stem, "audio": audio, "lrc": cands[0], "language": SONG_LANGUAGE.get(stem)})
    return songs


def run_cli(audio: Path, lrc: Path, language: str, out: Path, *, mode: str, routing: bool, extra: list[str]) -> dict:
    args = [
        "align", "--audio", str(audio), "--lyrics", str(lrc), "--language", language,
        "--output", str(out), "--mode", mode, "-q",
    ]
    if routing:
        args.append("--route-lines")
    args += extra
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-c", _WRAPPER, *args], capture_output=True, text=True, check=False)
    wall = time.time() - t0
    stats = {"rc": proc.returncode, "maxrss_mb": None}
    for line in proc.stdout.splitlines():
        if line.startswith("__STATS__"):
            stats.update(json.loads(line[len("__STATS__") :]))
    return {"wall_seconds": round(wall, 2), **stats, "stderr_tail": proc.stderr[-400:] if proc.returncode else ""}


def validate(data: dict, dur: float) -> dict:
    lines = data.get("lines", [])
    kinds = Counter(l.get("kind") or "singing" for l in lines)
    quality = Counter((l.get("quality") or "not_assessed") for l in lines if l.get("source") != "skipped")
    modes = Counter(e["mode"] for e in data.get("player_timeline", []))

    overlap = nonmono = out_of_range = 0
    max_end = 0.0
    for l in lines:
        last = -1.0
        for u in l.get("units", []):
            if u["end"] < u["start"] - 1e-9:
                out_of_range += 1
            if u["start"] < last - 1e-6:
                overlap += 1
            if u["start"] < last - 1e-6:
                nonmono += 1
            last = u["end"]
            max_end = max(max_end, u["end"])
    out_of_range += sum(1 for l in lines for u in l.get("units", []) if u["end"] > dur + 0.5)

    sing = [l for l in lines if (l.get("kind") or "singing") == "singing"]
    crossings = 0
    for a, b in itertools.pairwise(sing):
        if a.get("units") and a["units"][-1]["end"] > b["lrc_start"] + 0.31:
            crossings += 1

    latin_lines = [l for l in sing if any(c.isascii() and c.isalpha() for c in l["text"])]
    failed_latin = [l for l in latin_lines if l["source"] == "failed"]
    repeated = len(sing) - len({l["text"].strip() for l in sing if l["text"].strip()})

    windows = [l.get("window_seconds") for l in lines if l.get("window_seconds") is not None]

    return {
        "lines": len(lines),
        "kinds": dict(kinds),
        "quality": dict(quality),
        "player_modes": dict(modes),
        "model_lines": sum(1 for l in lines if l["source"] == "model"),
        "failed_lines": sum(1 for l in lines if l["source"] == "failed"),
        "skipped_lines": sum(1 for l in lines if l["source"] == "skipped"),
        "cross_line_violations": crossings,
        "non_monotonic_units": nonmono,
        "overlapping_units": overlap,
        "out_of_range_units": out_of_range,
        "max_unit_end": round(max_end, 2),
        "duration": round(dur, 2),
        "latin_lines": len(latin_lines),
        "latin_failed": len(failed_latin),
        "repeated_lines": repeated,
        "window_seconds_min": round(min(windows), 2) if windows else None,
        "window_seconds_max": round(max(windows), 2) if windows else None,
        "line_language": dict(Counter(l.get("line_language") or "?" for l in lines)),
        "routing_reasons": dict(Counter(l.get("route_reason") or "?" for l in lines if l.get("route_reason"))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-dir", default="Test")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--modes", default="production", help="comma-separated CLI modes to run")
    ap.add_argument("--routing", action="store_true", help="enable per-line language routing")
    ap.add_argument("--only", default=None, help="comma-separated song stems to run")
    ap.add_argument("--out-dir", default="experiments/reports")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    songs = discover(test_dir)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        songs = [s for s in songs if s["song"] in wanted]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for song in songs:
        if song["language"] is None:
            print(f"!! no language mapping for {song['song']}; skipping")
            continue
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(song["audio"])],
            capture_output=True, text=True, check=False,
        ).stdout)
        for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
            out = out_dir / f"fullsong_{args.tag}_{song['song'].replace(' ', '_')}_{mode}.json"
            run = run_cli(song["audio"], song["lrc"], song["language"], out, mode=mode,
                          routing=args.routing, extra=[])
            row = {"language": song["language"], "mode": mode, **run}
            if out.exists():
                data = json.loads(out.read_text(encoding="utf-8"))
                row.update(validate(data, dur))
            reports.setdefault(song["song"], {})[mode] = row
            print(f"{song['song'][:14]:16s} {mode:12s} rc={run['rc']} wall={run['wall_seconds']:6.1f}s "
                  f"rss={run['maxrss_mb'] and round(run['maxrss_mb'])}MB "
                  f"model={row.get('model_lines')} failed={row.get('failed_lines')} "
                  f"word={row.get('player_modes', {}).get('word')} line_lrc={row.get('player_modes', {}).get('line_lrc')} "
                  f"cross={row.get('cross_line_violations')} oob={row.get('out_of_range_units')}")

    out = out_dir / f"full_song_eval_{args.tag}.json"
    out.write_text(json.dumps({"tag": args.tag, "routing": args.routing, "songs": reports}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
