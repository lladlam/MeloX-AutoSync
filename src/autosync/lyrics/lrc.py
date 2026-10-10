"""LRC / enhanced-LRC parsing.

We keep the raw text exactly as written and preserve all timestamps. We do not
invent an ``end`` time: if the file only carries line starts, ``end`` stays
``None`` until acoustic alignment decides it.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..types import LyricLine, LyricSegment

# [mm:ss.xx] or [mm:ss:xx] or [mm:ss]
_LRC_TIME = re.compile(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
# <mm:ss.xx> inline segments (enhanced LRC)
_SEG_TIME = re.compile(r"<(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?>")
_META = re.compile(r"^\[(ar|ti|al|by|offset|re|ve|length|kana|language):(.*?)\]$", re.IGNORECASE)


def _frac_to_seconds(mm: str, ss: str, frac: str | None) -> float:
    minutes = int(mm)
    seconds = int(ss)
    if frac is None:
        return minutes * 60 + seconds
    # 2 digits => centiseconds, 3 digits => milliseconds, 1 digit => tenths
    value = int(frac)
    divisor = 10 ** len(frac)
    return minutes * 60 + seconds + value / divisor


def parse_lrc_text(text: str) -> tuple[list[LyricLine], dict[str, str]]:
    """Parse LRC content into lines plus a metadata dict.

    Supports multiple timestamps per line (``[00:01.00][00:05.00]text``) and
    the ``offset`` metadata tag (milliseconds, applied to every timestamp).
    """
    meta: dict[str, str] = {}
    offset_s = 0.0
    raw_lines: list[tuple[float, str]] = []
    seg_map: dict[float, list[LyricSegment]] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _META.match(line)
        if m:
            key, value = m.group(1).lower(), m.group(2).strip()
            meta[key] = value
            if key == "offset":
                try:
                    offset_s = int(value) / 1000.0
                except ValueError:
                    pass
            continue

        times = [(_frac_to_seconds(*g)) for g in _LRC_TIME.findall(line)]
        if not times:
            continue
        # Text is what remains after stripping all time tags.
        content = _LRC_TIME.sub("", line).strip()

        # Enhanced LRC segments (<mm:ss>word).
        seg_matches = list(_SEG_TIME.finditer(content))
        if seg_matches:
            segments: list[LyricSegment] = []
            for i, sm in enumerate(seg_matches):
                seg_start = _frac_to_seconds(sm.group(1), sm.group(2), sm.group(3))
                seg_text_start = sm.end()
                seg_text_end = (
                    seg_matches[i + 1].start() if i + 1 < len(seg_matches) else len(content)
                )
                seg_text = content[seg_text_start:seg_text_end].strip()
                if seg_text:
                    segments.append(LyricSegment(start=seg_start, text=seg_text))

        for t in times:
            raw_lines.append((t, content))
            if seg_matches:
                seg_map[t] = [LyricSegment(s.start, s.text) for s in segments]

    raw_lines.sort(key=lambda x: x[0])
    lines: list[LyricLine] = []
    for i, (t, content) in enumerate(raw_lines):
        lines.append(
            LyricLine(
                index=i,
                start=max(0.0, t + offset_s),
                text=content,
                end=None,
                segments=seg_map.get(t),
            )
        )

    # Apply offset to enhanced segments too.
    if offset_s:
        for ln in lines:
            if ln.segments:
                for s in ln.segments:
                    s.start = max(0.0, s.start + offset_s)

    return lines, meta


def parse_lrc(path: str | Path) -> tuple[list[LyricLine], dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"lyrics file not found: {path}")
    for enc in ("utf-8-sig", "utf-8", "gb18030", "shift_jis", "latin-1"):
        try:
            text = path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:  # pragma: no cover - all encodings failed
        text = path.read_bytes().decode("utf-8", "replace")
    return parse_lrc_text(text)
