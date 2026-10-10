"""Per-line language detection for mixed-language songs (V1.0 P2).

Script counting alone is not enough: 不安灵魂收容所 and 提瓦特民谣 are Chinese songs
with English sections, and Latin letters do **not** imply English. This module
returns an explicit, auditable decision (language + routable + reason) so a
low-confidence line can fall back instead of silently using the wrong model.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")

# A compact list of very common English function words. Presence of one is
# treated as evidence that a Latin-script line is actually English. This is a
# heuristic: it is deliberately conservative (absence => ambiguous => fallback).
_ENGLISH_EVIDENCE = {
    "the", "a", "an", "and", "or", "but", "if", "you", "your", "yours", "i", "me", "my", "mine",
    "we", "us", "our", "he", "she", "it", "its", "they", "them", "their", "is", "are", "was",
    "were", "be", "been", "am", "do", "does", "did", "dont", "wanna", "gonna", "gotta", "can",
    "could", "will", "would", "should", "shall", "may", "might", "must", "to", "of", "in", "on",
    "at", "for", "with", "without", "from", "into", "that", "this", "these", "those", "there",
    "here", "not", "no", "never", "all", "so", "just", "like", "love", "night", "day", "time",
    "when", "where", "what", "who", "how", "why", "oh", "yeah", "baby", "please", "let", "lets",
    "up", "down", "out", "over", "under", "again", "still", "ever", "always", "together",
    "alone", "home", "away", "back", "feel", "know", "see", "go", "going", "come", "take",
    "give", "make", "say", "tell", "hear", "hold", "keep", "leave", "want", "need", "call",
    "welcome", "party", "asleep", "pick", "phone", "clap", "world", "life", "heart", "song",
}


@dataclass(frozen=True)
class LanguageDecision:
    language: str  # "zh" | "ja" | "en" | "latin_ambiguous" | "none"
    routable: bool
    reason: str = ""
    han: int = 0
    kana: int = 0
    latin: int = 0

    @property
    def script_summary(self) -> str:
        return f"han={self.han} kana={self.kana} latin={self.latin}"


def _counts(text: str) -> tuple[int, int, int]:
    han = kana = latin = 0
    for ch in text:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            han += 1
        elif 0x3040 <= o <= 0x30FF:
            kana += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
    return han, kana, latin


def _english_evidence(text: str) -> bool:
    words = [w.lower().strip("'-") for w in _LATIN_WORD.findall(unicodedata.normalize("NFKC", text))]
    return any(w in _ENGLISH_EVIDENCE for w in words)


def detect_line_language(text: str, primary: str) -> LanguageDecision:
    """Decide which model should align ``text`` (or that it must fall back)."""
    han, kana, latin = _counts(text)
    cjk = han + kana
    if cjk == 0 and latin == 0:
        return LanguageDecision("none", False, "no letters", han, kana, latin)

    # Japanese: kana anywhere is decisive.
    if kana > 0:
        if latin == 0:
            return LanguageDecision("ja", True, "kana present", han, kana, latin)
        ratio = cjk / (cjk + latin)
        if ratio >= 0.5:
            return LanguageDecision("ja", True, "kana-dominant mixed; ja G2P covers Latin tokens", han, kana, latin)
        return LanguageDecision(
            "mixed", False, f"Latin-dominant mixed line (cjk ratio {ratio:.2f}); not reliably separable",
            han, kana, latin,
        )

    # Kanji-only: Japanese kanji-only lines exist, so use the song's primary.
    if han > 0:
        if latin == 0:
            lang = "ja" if primary == "ja" else "zh"
            return LanguageDecision(lang, True, f"kanji-only; routed by song primary={primary}", han, kana, latin)
        ratio = han / (han + latin)
        if ratio >= 0.5:
            lang = "ja" if primary == "ja" else "zh"
            return LanguageDecision(lang, True, "CJK-dominant mixed; Latin token handled by the CJK model", han, kana, latin)
        return LanguageDecision(
            "mixed", False, f"Latin-dominant mixed line (han ratio {ratio:.2f}); not reliably separable",
            han, kana, latin,
        )

    # Pure Latin script.
    if primary == "en":
        return LanguageDecision("en", True, "Latin text in an English-primary song", han, kana, latin)
    if _english_evidence(text):
        return LanguageDecision("en", True, "Latin text with English function words", han, kana, latin)
    return LanguageDecision(
        "latin_ambiguous", False,
        "Latin text without English evidence (may be romanised or stylised); conservative fallback",
        han, kana, latin,
    )
