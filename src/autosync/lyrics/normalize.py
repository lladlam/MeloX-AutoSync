"""Text normalisation for alignment.

Guiding principle: **never lose the original text.** We produce a normalised
form used for acoustic matching while keeping the raw surface text around for
display. This matters especially for Japanese, where kanji must be preserved
for the user even though the acoustic model matches on readings.
"""

from __future__ import annotations

import re
import unicodedata

# Unicode ranges
_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_HIRAGANA = r"\u3041-\u309f"
_KATAKANA = r"\u30a0-\u30ff"
_HALFWIDTH_KANA = r"\uff66-\uff9f"
_CJK_PUNCT = r"\u3000-\u303f\uff00-\uffef"

_EN_KEEP = re.compile(r"[^A-Za-z0-9' ]+")
_CJK_KEEP = re.compile(rf"[^{_CJK}{_HIRAGANA}{_KATAKANA}\uff10-\uff19\d]+")
_WS = re.compile(r"\s+")


def normalize_en(text: str) -> str:
    """Normalise English for a character-level uppercase CTC model.

    Uppercases, drops punctuation that the model vocabulary cannot emit, and
    collapses whitespace. Apostrophes are kept if the vocabulary supports them
    (``wav2vec2-base-960h`` does) but will simply be dropped by the tokenizer
    when unsupported.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = _EN_KEEP.sub(" ", text)
    text = _WS.sub(" ", text).strip().upper()
    return text


def normalize_cjk(text: str) -> str:
    """Normalise Chinese/Japanese: keep CJK chars, kana and digits only."""
    text = unicodedata.normalize("NFKC", text)
    text = _CJK_KEEP.sub("", text)
    return text


def normalize_for(language: str, text: str) -> str:
    if language == "en":
        return normalize_en(text)
    if language in ("zh", "ja"):
        return normalize_cjk(text)
    raise ValueError(f"unsupported language: {language!r}")


def words_en(text: str) -> list[str]:
    """Split normalised English into words."""
    return [w for w in text.split(" ") if w]


def chars_cjk(text: str) -> list[str]:
    """Split normalised CJK text into per-character units (ignoring spaces)."""
    return [c for c in text if not c.isspace()]


def display_units(language: str, text: str) -> list[str]:
    """The natural display units for a language: words (en) or chars (zh/ja)."""
    if language == "en":
        import re

        return re.findall(r"[A-Za-z0-9']+", text)
    if language in ("zh", "ja"):
        return chars_cjk(normalize_cjk(text))
    raise ValueError(f"unsupported language: {language!r}")
