"""Non-singing LRC line classification (V0.9).

Credit lines ("作词/作曲/编曲/…", "Lyrics by …", "作詞・作曲：…") and
non-lyric cues ("(間奏)", "Instrumental", a line of only symbols) carry LRC
timestamps but are **not sung**. Feeding them to forced alignment wastes the
acoustic model, can drag the previous line's window, and would produce fake
word-level highlighting.

This module classifies each line **before any acoustic inference**:

* ``SINGING``      – ordinary lyrics (the default, conservative choice);
* ``METADATA``     – a confident credit/signature line;
* ``INSTRUMENTAL`` – a non-lyric cue (interlude/intro/outro, symbols only, empty);
* ``UNKNOWN``      – looks credit-like but is not a confident match.

The original text is never modified; callers keep the raw line and only use the
classification to decide whether to align it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from ..types import LyricLine


class LineKind(str, Enum):
    SINGING = "singing"
    METADATA = "metadata"
    INSTRUMENTAL = "instrumental"
    UNKNOWN = "unknown"

    @property
    def alignable(self) -> bool:
        return self is LineKind.SINGING


@dataclass(frozen=True)
class LineClassification:
    kind: LineKind
    reason: str = ""
    matched: str = ""


# --------------------------------------------------------------------------- setup
_BRACKETS = re.compile(r"[\[\]【】()（）〔〕「」『』]")
_WS = re.compile(r"\s+")
_SENTENCE_PUNCT = "。！？!?…"

# Multi-character credit keywords (Chinese + Japanese). Single-character ones are
# only trusted with an explicit separator (see below) to avoid matching lyrics.
_STRONG_KEYWORDS = [
    # Chinese — roles
    "作词人", "作曲人", "编曲人", "混音师", "母带工程师", "制作人", "製作人", "监制", "監製",
    "作词", "作詞", "作曲", "编曲", "編曲", "混音", "母带", "母帶", "制作", "製作", "出品",
    "发行", "發行", "录音室", "錄音室", "录音", "錄音", "配唱", "和声", "和聲", "合声", "合聲",
    "词曲", "詞曲", "歌词", "歌詞", "策划", "企劃", "统筹", "統籌", "宣传", "宣傳", "封面",
    "视觉", "視覺", "导演", "導演", "特别鸣谢", "特別鳴謝", "鸣谢", "鳴謝", "版权", "版權",
    "演唱", "原唱", "翻唱", "改编", "改編", "监唱", "伴唱", "合唱", "童声", "女声", "男声",
    "声乐", "音乐总监", "音乐制作", "制作助理", "后期", "混缩", "录音棚", "工作室", "人声",
    "音频编辑", "人声编辑", "和音", "管弦乐", "弦乐", "弦樂",
    # Chinese — instruments
    "吉他", "贝斯", "貝斯", "贝司", "键盘", "鍵盤", "鼓手", "鼓组", "鼓組", "打击乐", "打擊樂",
    "钢琴", "鋼琴", "笛子", "长笛", "長笛", "二胡", "古筝", "古箏", "琵琶", "小提琴", "中提琴",
    "大提琴", "低音提琴", "萨克斯", "薩克斯", "小号", "小號", "长号", "長號", "圆号", "圓號",
    # Japanese
    "ミックス", "マスタリング", "プロデュース", "レコーディング", "サウンド", "アレンジ",
    "コーラス", "ギター", "ベース", "ドラム", "ピアノ", "ストリングス", "録音", "音響",
    "バイオリン", "チェロ", "フルート", "トランペット", "作詞", "作曲", "編曲", "歌唱",
]
# Labels that are credits only when they are the *entire* label (avoids matching
# the character inside an ordinary lyric word, e.g. 唱歌).
_EXACT_LABELS = {"詞", "曲", "歌", "唄", "词", "词曲", "詞曲", "op", "sp", "OST"}
_EN_KEYWORDS = [
    "lyrics", "lyric", "music", "composed", "composer", "arranged", "arranger", "produced",
    "producer", "written", "mixed", "mixing", "mastered", "mastering", "recorded", "recording",
    "vocals", "vocal", "guitar", "guitars", "bass", "drums", "piano", "strings", "programming",
    "lyricist", "studio", "label", "op", "sp", "chorus", "harmony", "backing", "percussion",
    "violin", "cello", "flute", "trumpet", "saxophone", "brass", "orchestra", "arrangement",
    "production", "engineer", "engineered", "edited", "edit", "sound", "mix", "master",
    "additional", "executive", "co-produced", "assistant",
]
# NOTE: `_EN_KEYWORDS` is defined above (broad list); single-character labels are
# only trusted through `_EXACT_LABELS` so they cannot match inside a word.
_NON_LYRIC_CUES = [
    "间奏", "間奏", "间奏曲", "間奏曲", "前奏", "尾奏", "纯音乐", "純音樂", "伴奏",
    "instrumental", "interlude", "intro", "outro", "solo", "karaoke",
]
_SYMBOLS_ONLY = re.compile(r"^[♪♫♩♬~～\-—–_…\.、,。·・\s]+$")

_K = "|".join(re.escape(k) for k in sorted(set(_STRONG_KEYWORDS), key=len, reverse=True))
_K_EN = "|".join(re.escape(k) for k in sorted(set(_EN_KEYWORDS), key=len, reverse=True))
_CUES = "|".join(re.escape(c) for c in _NON_LYRIC_CUES)

_RE_CUE = re.compile(rf"^(?:{_CUES})$", re.IGNORECASE)
_RE_CREDIT_SPACE = re.compile(rf"^(?:{_K})\s+\S")
_RE_EN_BY = re.compile(rf"^(?:{_K_EN})\b[^:：]{{0,60}}?\bby\b\s+\S", re.IGNORECASE)
_RE_KEYWORD = re.compile(rf"(?:{_K})")
_COLON_SPLIT = re.compile(r"[:：]")
_MAX_LABEL_CHARS = 14
_MAX_LABEL_CHARS_EN = 30
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def _flatten(text: str) -> str:
    """NFKC + bracket→space + collapsed whitespace, for matching only."""
    s = unicodedata.normalize("NFKC", text)
    s = _BRACKETS.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _has_sentence_punct(flat: str) -> bool:
    return any(ch in _SENTENCE_PUNCT for ch in flat)


def _label_is_credit(label: str) -> bool:
    """True when a short ``label: value`` prefix names a credit role/instrument."""
    compact = _WS.sub("", label)
    if not compact:
        return False
    if compact.lower() in {k.lower() for k in _EXACT_LABELS}:
        return True
    if any(k in compact for k in _STRONG_KEYWORDS):
        return True
    low = label.lower()
    return any(re.search(rf"\b{re.escape(k)}\b", low) for k in _EN_KEYWORDS)


def classify_line(text: str) -> LineClassification:
    """Classify one LRC line. Never raises; defaults to ``SINGING``."""
    flat = _flatten(text)
    if not flat:
        return LineClassification(LineKind.INSTRUMENTAL, "empty text")
    if _RE_CUE.match(flat):
        return LineClassification(LineKind.INSTRUMENTAL, f"non-lyric cue {flat!r}", flat)
    if _SYMBOLS_ONLY.match(flat):
        return LineClassification(LineKind.INSTRUMENTAL, "symbols only", flat)

    # "label : value" where the label is short and names a credit role/instrument.
    # This covers real-world variants (吉他演奏：/ 上海录音 : / 笛子 : / Produced by).
    m = _COLON_SPLIT.search(flat)
    if m is not None:
        label = flat[: m.start()].strip()
        value = flat[m.end() :].strip()
        compact = _WS.sub("", label)
        # CJK labels are kept very short to avoid matching lyric lines; English
        # role labels are naturally longer ("Recording Engineer").
        limit = _MAX_LABEL_CHARS if _CJK_RE.search(label) else _MAX_LABEL_CHARS_EN
        if (
            value
            and len(compact) <= limit
            and not _has_sentence_punct(label)
            and _label_is_credit(label)
        ):
            return LineClassification(LineKind.METADATA, f"credit line (label {label!r})", flat)

    if _RE_EN_BY.match(flat) and not _has_sentence_punct(flat):
        return LineClassification(LineKind.METADATA, "credit line (… by …)", flat)
    if _RE_CREDIT_SPACE.match(flat) and not _has_sentence_punct(flat):
        return LineClassification(LineKind.METADATA, "credit line (keyword value)", flat)

    # Ambiguous: a probable credit list without separators, e.g. "作詞作曲米津玄師".
    hits = {m.group(0) for m in _RE_KEYWORD.finditer(flat)}
    if len(hits) >= 2 and len(flat) <= 30:
        return LineClassification(LineKind.UNKNOWN, f"probable credit list {sorted(hits)}", flat)
    return LineClassification(LineKind.SINGING, "", "")


def classify_lines(lines: list[LyricLine]) -> list[LineClassification]:
    return [classify_line(line.text) for line in lines]
