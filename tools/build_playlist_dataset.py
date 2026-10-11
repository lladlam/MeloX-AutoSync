#!/usr/bin/env python
"""Build a real-pop-music evaluation set from ``Test/`` with scene labels and splits.

The repository's ``Test/`` folder holds third-party audio + lyrics (never
committed).  This tool turns it into a reproducible, machine-readable dataset:

1. recursively discovers audio (``mp3/wav/flac/m4a/ogg``) and lyrics
   (``lrc/ttml/txt``) and matches them by filename;
2. detects the lyric language (zh / ja / en / mixed) from the *singing* text;
3. validates each pair (audio duration, lyric time range, non-empty text) and
   records problems instead of hiding them;
4. derives observable scene tags (lyric density, singing rate, repeated lines,
   sung-span ratio, a crude vocal-band ratio) and marks anything it cannot
   measure as ``unknown`` — it never invents a label;
5. groups by song family and assigns deterministic tune / val / blind splits so
   the same recording never appears in both selection and blind sets.

Audio and lyrics are **never** copied or written into git-tracked paths; only the
manifest (paths are relative, text is not included) is written under
``experiments/playlist/`` (gitignored) plus a small aggregate summary.

Usage::

    .venv/bin/python tools/build_playlist_dataset.py --root Test \
        --out experiments/playlist --report docs/PLAYLIST_DATASET.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}
LYRIC_EXT = {".lrc", ".ttml", ".txt"}
SR = 16000

# From lyrics/classify: credit lines carry timestamps but are not sung.
_CREDIT_HINTS = (
    "作词", "作詞", "作曲", "编曲", "編曲", "制作", "製作", "混音", "母带", "母帶",
    "录音", "錄音", "和声", "和聲", "合声", "合聲", "吉他", "贝斯", "貝斯", "钢琴", "鋼琴",
    "鼓", "弦乐", "弦樂", "管弦", "人声", "人聲", "监制", "監製", "出品", "发行", "發行",
    "lyrics", "music", "composed", "arranged", "produced", "producer", "mixed", "master",
    "recording", "vocals", "guitar", "bass", "drums", "piano", "strings",
    "ミックス", "マスタリング", "アレンジ", "コーラス", "ギター", "ベース", "ドラム",
)
_TIME_RE = re.compile(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
_META_RE = re.compile(r"^\[(ar|ti|al|by|offset|length|language|kana):(.*?)\]$", re.IGNORECASE)


def _norm_stem(name: str) -> str:
    s = unicodedata.normalize("NFKC", name).lower()
    return re.sub(r"[^0-9a-z\u3040-\u30ff\u4e00-\u9fff]+", "", s)


def _frac(mm: str, ss: str, frac: str | None) -> float:
    v = int(mm) * 60 + int(ss)
    if frac:
        v += int(frac) / (10 ** len(frac))
    return v


def parse_lrc(path: Path) -> tuple[list[tuple[float, str]], dict[str, str]]:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "shift_jis", "latin-1"):
        try:
            text = path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        text = path.read_bytes().decode("utf-8", "replace")
    entries: list[tuple[float, str]] = []
    meta: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _META_RE.match(line)
        if m:
            meta[m.group(1).lower()] = m.group(2).strip()
            continue
        times = [_frac(*g) for g in _TIME_RE.findall(line)]
        if not times:
            continue
        content = _TIME_RE.sub("", line).strip()
        for t in times:
            entries.append((t, content))
    entries.sort(key=lambda x: x[0])
    return entries, meta


def parse_ttml(path: Path) -> tuple[list[tuple[float, str]], dict[str, str]]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(path.read_text(encoding="utf-8"))
    entries: list[tuple[float, str]] = []
    for p in root.iter("{http://www.w3.org/ns/ttml}p"):
        begin = p.get("begin")
        if not begin:
            continue
        txt = "".join(p.itertext()).strip()
        ms = re.match(r"(\d+):(\d+):(\d+)(?:\.(\d+))?", begin)
        if ms and txt:
            h, mi, se, fr = ms.groups()
            t = int(h) * 3600 + int(mi) * 60 + int(se) + (float(f"0.{fr}") if fr else 0.0)
            entries.append((t, txt))
    return entries, {}


def is_credit(text: str) -> bool:
    low = text.lower()
    if any(h in text or h in low for h in _CREDIT_HINTS):
        return True
    return bool(re.search(r"[:：]\s*\S+", text)) and len(text) <= 40


def detect_language(texts: list[str]) -> tuple[str, dict[str, int]]:
    kana = han = latin = 0
    for t in texts:
        for ch in t:
            o = ord(ch)
            if 0x3040 <= o <= 0x30FF:
                kana += 1
            elif 0x4E00 <= o <= 0x9FFF:
                han += 1
            elif ch.isascii() and ch.isalpha():
                latin += 1
    counts = {"kana": kana, "han": han, "latin": latin}
    total = max(1, kana + han + latin)
    if kana / total > 0.10:
        return "ja", counts
    if han / total >= 0.30 and latin / total >= 0.30:
        return "mixed", counts
    if han / total > 0.20:
        return "zh", counts
    return "en", counts


def audio_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60, check=False)
        return float(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return None


def clip_stats(path: Path) -> dict:
    """Crude, cheap scene proxies on a single mid-song window (streamed)."""
    try:
        import numpy as np
        import soundfile as sf

        info = sf.info(str(path))
        dur = info.frames / info.samplerate
        start = int(max(0.0, dur * 0.35) * info.samplerate)
        n = int(min(20.0, dur * 0.3) * info.samplerate)
        data, sr = sf.read(str(path), dtype="float32", always_2d=True,
                           start=start, frames=max(n, 1))
        x = data.mean(axis=1)
        if sr != SR:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(SR, sr)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        if x.size < 1024:
            return {"vocal_band_ratio": None}
        # crude: energy share in the 200-4000 Hz vocal band vs 60 Hz-8 kHz
        spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
        freqs = np.fft.rfftfreq(x.size, 1 / SR)
        vocal = spec[(freqs >= 200) & (freqs <= 4000)].sum()
        full = spec[(freqs >= 60) & (freqs <= 8000)].sum()
        return {"vocal_band_ratio": float(vocal / full) if full > 0 else None,
                "clip_duration_s": float(x.size / SR)}
    except Exception:  # noqa: BLE001 - scene stats are best-effort
        return {"vocal_band_ratio": None}


def match_lyrics(audio: Path, lyrics: list[Path]) -> Path | None:
    a = _norm_stem(audio.stem)
    best, best_score = None, 0.0
    for ly in lyrics:
        l = _norm_stem(ly.stem)
        if not l or not a:
            continue
        if l == a:
            return ly
        # the audio file is usually the short name and the lyric file carries
        # the artist list, e.g. `DAMIDAMI.mp3` vs `DAMIDAMI-Sihan,...lrc`.
        if len(a) >= 2 and l.startswith(a):
            return ly
        if len(l) >= 2 and a.startswith(l):
            return ly
        # tolerant match for near-identical stems (e.g. 游记 vs 游计)
        common = 0
        for ca, cl in zip(a, l, strict=False):
            if ca != cl:
                break
            common += 1
        if common >= max(2, len(a) - 1):
            return ly
        score = common / max(len(a), len(l))
        if score > best_score:
            best, best_score = ly, score
    return best if best_score >= 0.4 else None


def build(root: Path) -> dict:
    audio = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    lyrics = sorted(p for p in root.rglob("*") if p.suffix.lower() in LYRIC_EXT)
    records = []
    for a in audio:
        ly = match_lyrics(a, lyrics)
        rec: dict = {
            "song_id": a.stem,
            "audio": str(a.relative_to(root)),
            "audio_bytes": a.stat().st_size,
            "lyrics": str(ly.relative_to(root)) if ly else None,
            "lyrics_kind": ly.suffix.lower().lstrip(".") if ly else None,
            "problems": [],
        }
        dur = audio_duration(a)
        rec["duration_s"] = round(dur, 3) if dur else None
        if ly is None:
            rec["problems"].append("no_matching_lyrics")
            records.append(rec)
            continue
        try:
            entries, meta = (parse_ttml(ly) if ly.suffix.lower() == ".ttml" else parse_lrc(ly))
        except Exception as exc:  # noqa: BLE001
            rec["problems"].append(f"lyrics_parse_error:{type(exc).__name__}")
            records.append(rec)
            continue
        singing = [(t, txt) for t, txt in entries if txt and not is_credit(txt)]
        rec["n_timed_lines"] = len(entries)
        rec["n_singing_lines"] = len(singing)
        rec["lyrics_meta"] = {k: v for k, v in meta.items() if k in ("ti", "ar", "al", "language")}
        if not singing:
            rec["problems"].append("no_singing_lines")
        texts = [txt for _, txt in singing]
        rec["n_chars"] = sum(len(t) for t in texts)
        lang, counts = detect_language(texts)
        rec["language"] = lang
        rec["language_counts"] = counts
        if dur and entries:
            last = max(t for t, _ in entries)
            rec["lyric_last_time_s"] = round(last, 3)
            rec["lyric_span_ratio"] = round(last / dur, 3) if dur > 0 else None
            if last > dur + 2.0:
                rec["problems"].append(f"lyrics_exceed_audio:{last:.1f}s>{dur:.1f}s")
            elif last < dur * 0.5:
                rec["problems"].append("lyrics_cover_less_than_half")
        # scene tags (observable only)
        if len(singing) >= 4:
            starts = [t for t, _ in singing]
            gaps = [b - a2 for a2, b in itertools.pairwise(starts) if b > a2]
            span = starts[-1] - starts[0] if starts[-1] > starts[0] else None
            dense = len(singing) / (span / 60.0) if span and span > 5 else None
            cps = None
            if gaps:
                import statistics

                med_gap = statistics.median(gaps)
                cps = round(statistics.median([len(t) for _, t in singing]) / med_gap, 2) if med_gap > 0 else None
            dup = Counter(t for _, t in singing)
            rep = sum(c - 1 for c in dup.values() if c > 1) / max(1, len(singing))
            rec["scene"] = {
                "lines_per_min": round(dense, 2) if dense else None,
                "median_chars_per_s": cps,
                "repeated_line_ratio": round(rep, 3),
                "sung_span_s": round(span, 1) if span else None,
            }
            rec["scene"]["fast_singing"] = bool(cps and cps >= 6.0)
            rec["scene"]["dense_lyrics"] = bool(dense and dense >= 12.0)
            rec["scene"]["repeated_chorus"] = bool(rep >= 0.25)
        else:
            rec["scene"] = {"unknown": "too_few_singing_lines"}
        rec["scene"]["vocal_band_ratio"] = clip_stats(a).get("vocal_band_ratio")
        rec["n_artists"] = None
        artist_src = rec["lyrics_meta"].get("ar") or ""
        if not artist_src and "-" in (ly.stem if ly else ""):
            artist_src = (ly.stem if ly else "").split("-", 1)[1]
        if artist_src:
            rec["n_artists"] = len([a2 for a2 in re.split(r"[/,、;]", artist_src) if a2.strip()])
        rec["multi_singer_hint"] = bool(rec["n_artists"] and rec["n_artists"] >= 3)
        records.append(rec)
    return {"root": str(root), "n_audio": len(audio), "n_lyrics_files": len(lyrics),
            "records": records}


def assign_splits(records: list[dict]) -> dict:
    """Deterministic tune/val/blind split by song family (one recording per song).

    With a small corpus we keep the split honest: we report the real counts and
    never claim a blind set larger than it is. Selection songs (tune/val) and
    blind songs are disjoint.
    """
    out: dict[str, str] = {}
    for lang in sorted({r.get("language") or "unknown" for r in records}):
        if lang == "unknown":
            continue
        fams = sorted(r["song_id"] for r in records if r.get("language") == lang)
        n = len(fams)
        if n >= 6:
            n_blind, n_val = 2, 2
        elif n >= 3:
            n_blind, n_val = 1, 1
        else:
            n_blind, n_val = 1, 0
        for i, fam in enumerate(fams):
            if i < n_blind:
                out[fam] = "blind"
            elif i < n_blind + n_val:
                out[fam] = "val"
            else:
                out[fam] = "tune"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="Test")
    ap.add_argument("--out", default="experiments/playlist")
    ap.add_argument("--report", default="docs/PLAYLIST_DATASET.md")
    args = ap.parse_args()

    root = Path(args.root)
    data = build(root)
    splits = assign_splits(data["records"])
    for rec in data["records"]:
        rec["split"] = splits.get(rec["song_id"], "unassigned")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "dataset_manifest.json"
    manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    by_lang = Counter(r.get("language") for r in data["records"])
    by_split = Counter(r["split"] for r in data["records"])
    problems = Counter(p.split(":")[0] for r in data["records"] for p in r.get("problems", []))
    summary = {
        "n_audio": data["n_audio"], "n_lyrics_files": data["n_lyrics_files"],
        "by_language": dict(by_lang), "by_split": dict(by_split),
        "problems": dict(problems),
        "records": [{"song_id": r["song_id"], "language": r.get("language"),
                     "split": r["split"], "duration_s": r.get("duration_s"),
                     "n_singing_lines": r.get("n_singing_lines"),
                     "n_chars": r.get("n_chars"), "scene": r.get("scene"),
                     "problems": r.get("problems")} for r in data["records"]],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_audio", "n_lyrics_files", "by_language",
                                              "by_split", "problems")},
                     ensure_ascii=False, indent=2))

    # committed report: aggregate only, no lyric text
    lines = ["# Test/ playlist dataset (auto-generated)", "",
             f"Source: `{args.root}` (third-party audio; never committed).",
             f"Audio files: **{summary['n_audio']}**, lyric files: **{summary['n_lyrics_files']}**.",
             "", "## Language", ""]
    for k, v in sorted(by_lang.items(), key=lambda kv: str(kv[0])):
        lines.append(f"- {k}: {v}")
    lines += ["", "## Splits (by song family; tune/val select, blind is frozen)", ""]
    for k, v in sorted(by_split.items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## Per song", "",
              "| song_id | lang | split | dur s | singing lines | chars | scene | problems |",
              "|---|---|---|---:|---:|---:|---|---|"]
    for r in summary["records"]:
        sc = r["scene"] or {}
        tags = [k for k in ("fast_singing", "dense_lyrics", "repeated_chorus") if sc.get(k)]
        vr = sc.get("vocal_band_ratio")
        if vr is not None:
            tags.append(f"vocalband={vr:.2f}")
        lines.append(f"| `{r['song_id']}` | {r['language']} | {r['split']} | "
                     f"{r['duration_s']} | {r['n_singing_lines']} | {r['n_chars']} | "
                     f"{', '.join(tags) or '—'} | {', '.join(r['problems']) or '—'} |")
    lines += ["", ("> Scene tags are observable proxies; anything not measurable is marked `unknown`."
               " No lyric text is stored in this report."), ""]
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"manifest -> {manifest}\nsummary  -> {out / 'summary.json'}\nreport   -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
