"""Japanese grapheme-to-phoneme and phoneme↔original-text mapping.

Motivation
----------
pyshiro / TIFA / HuBERT-CTC all work at the **phoneme** level, but the user
wants their *original* lyrics back. This module:

* converts kanji/kana lyrics to a kana reading (G2P, via pyopenjtalk);
* maps every original character to its kana span (a kanji may be several morae,
  a kana character is one mora — we **never** assume one kanji == one syllable);
* converts the kana to the SHIRO/ENUNU phoneme inventory and attaches, to each
  original character, the phonemes it covers;
* supports pronunciation overrides and reports uncertainty when the reading
  cannot be established;
* guarantees the original text can be reconstructed losslessly (characters are
  never dropped).

All external backends (``pyopenjtalk`` for G2P, ``pyshiro`` for the kana→
phoneme table) are injectable so tests run without them.
"""

from __future__ import annotations

import itertools
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

_KATAKANA_TO_HIRAGANA = str.maketrans(
    {chr(c): chr(c - 0x60) for c in range(0x30A1, 0x30F7)}
)
# Long vowel mark and iteration marks that belong with the previous mora.
_MORA_EXTENDERS = {"ー", "ゝ", "ゞ", "ヽ", "ヾ"}
_MAX_KANJI_MORAE = 12


def katakana_to_hiragana(text: str) -> str:
    return text.translate(_KATAKANA_TO_HIRAGANA)


def _is_kanji(ch: str) -> bool:
    o = ord(ch)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF


def _is_kana(ch: str) -> bool:
    o = ord(ch)
    return 0x3041 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF


@dataclass
class CharReading:
    """Reading of one original lyric character."""

    char: str
    kana: str = ""
    phonemes: list[str] = field(default_factory=list)
    uncertain: bool = False


def _default_g2p(text: str) -> str:
    import pyopenjtalk

    return pyopenjtalk.g2p(text, kana=True)


def _default_convert(kana: str, table: dict) -> list[str]:
    import pyshiro

    return pyshiro.text_to_phonemes(kana, table)


def _default_table() -> dict:
    import pyshiro

    return pyshiro.load_table()


class JapanesePronunciationMapper:
    """Kanji/kana → kana → phonemes, with per-character attribution."""

    def __init__(
        self,
        table: dict | None = None,
        g2p: Callable[[str], str] | None = None,
        convert: Callable[[str, dict], list[str]] | None = None,
        overrides: dict[str, str] | None = None,
    ) -> None:
        self._table = table
        self._g2p = g2p
        self._convert = convert
        self.overrides = dict(overrides or {})

    # ---------------------------------------------------------------- backends
    def _ensure(self) -> None:
        if self._table is None:
            self._table = _default_table()
        if self._g2p is None:
            self._g2p = _default_g2p
        if self._convert is None:
            self._convert = _default_convert

    # ------------------------------------------------------------------- G2P
    def apply_overrides(self, text: str) -> str:
        if not self.overrides:
            return text
        out = text
        for surface in sorted(self.overrides, key=len, reverse=True):
            out = out.replace(surface, self.overrides[surface])
        return out

    def to_kana(self, text: str) -> str:
        self._ensure()
        normalized = unicodedata.normalize("NFKC", self.apply_overrides(text))
        return katakana_to_hiragana(self._g2p(normalized))

    # ------------------------------------------------------- kana → phonemes
    def _convert_prefixes(self, kana: str) -> list[list[str]] | None:
        """Inner phonemes (no leading/trailing ``pau``) for every kana prefix."""
        self._ensure()
        prefixes: list[list[str]] = []
        whole = self._convert(kana, self._table)
        inner_whole = self._strip_pau(whole)
        for i in range(len(kana) + 1):
            inner = self._strip_pau(self._convert(kana[:i], self._table))
            prefixes.append(inner)
        # Prefix conversion must be monotonic and end at the whole conversion,
        # otherwise attribution is unreliable.
        if prefixes[-1] != inner_whole:
            return None
        for a, b in itertools.pairwise(prefixes):
            if len(b) < len(a):
                return None
        return prefixes

    @staticmethod
    def _strip_pau(phonemes: list[str]) -> list[str]:
        out = list(phonemes)
        if out and out[0] == "pau":
            out = out[1:]
        if out and out[-1] == "pau":
            out = out[:-1]
        return out

    # ------------------------------------------------------ char ↔ kana spans
    def align_kana_to_chars(self, chars: list[str], kana: str) -> list[tuple[int, int]] | None:
        """Return, per original char, its ``[a, b)`` kana span (or ``None``).

        Kana characters consume one kana char; kanji consume one or more;
        other characters (latin, digits, punctuation) consume none.
        """
        n, m = len(chars), len(kana)
        inf = float("inf")
        # dp[i][j] = min mismatch cost to consume chars[:i] and kana[:j]
        dp = [[inf] * (m + 1) for _ in range(n + 1)]
        parent: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0
        for i in range(n):
            ch = chars[i]
            hira = katakana_to_hiragana(unicodedata.normalize("NFKC", ch))
            for j in range(m + 1):
                if dp[i][j] == inf:
                    continue
                if _is_kana(ch):
                    if j < m:
                        cost = 0.0 if kana[j] == hira else 1.0
                        if dp[i + 1][j + 1] > dp[i][j] + cost:
                            dp[i + 1][j + 1] = dp[i][j] + cost
                            parent[i + 1][j + 1] = (i, j)
                elif _is_kanji(ch):
                    for k in range(1, min(_MAX_KANJI_MORAE, m - j) + 1):
                        if dp[i + 1][j + k] > dp[i][j]:
                            dp[i + 1][j + k] = dp[i][j]
                            parent[i + 1][j + k] = (i, j)
                else:  # latin / digit / punctuation consume no kana
                    if dp[i + 1][j] > dp[i][j]:
                        dp[i + 1][j] = dp[i][j]
                        parent[i + 1][j] = (i, j)
        if dp[n][m] == inf:
            return None
        # Backtrack spans.
        spans: list[tuple[int, int] | None] = [None] * n
        i, j = n, m
        while i > 0:
            prev = parent[i][j]
            if prev is None:
                return None
            pi, pj = prev
            spans[pi] = (pj, j)
            i, j = pi, pj
        return spans

    # ------------------------------------------------------------------- main
    def map(self, text: str) -> tuple[list[CharReading], list[str]]:
        """Map ``text`` to per-character readings and the full phoneme sequence.

        Returns ``(readings, full_phonemes)`` where ``full_phonemes`` includes the
        leading/trailing ``pau`` used for alignment. On failure every character is
        marked ``uncertain`` and the phoneme list is empty.
        """
        self._ensure()
        # Keep the ORIGINAL characters so the text can be reconstructed
        # losslessly; overrides only affect the reading, never the returned text.
        chars = list(text)
        try:
            kana = self.to_kana(text)
        except Exception:  # noqa: BLE001 - G2P backend missing/failed
            return [CharReading(char=c, uncertain=True) for c in chars], []

        readings = [CharReading(char=c) for c in chars]
        spans = self.align_kana_to_chars(chars, kana)
        prefixes = self._convert_prefixes(kana) if spans else None
        if spans is None or prefixes is None:
            for r in readings:
                r.uncertain = True
            return readings, self._convert(kana, self._table)

        for r, span in zip(readings, spans):
            if span is None:
                r.uncertain = True
                continue
            a, b = span
            r.kana = kana[a:b]
            r.phonemes = prefixes[b][len(prefixes[a]) :]
            if not r.phonemes:
                # No phoneme could be attributed (e.g. the second mora of a
                # youon such as きょ, or a latin character). The boundary is not
                # well defined, so mark it uncertain rather than pretend.
                r.uncertain = True
        full = self._convert(kana, self._table)
        return readings, full

    @staticmethod
    def reconstruct(readings: list[CharReading]) -> str:
        """Losslessly rebuild the original text from the readings."""
        return "".join(r.char for r in readings)
