"""Regression tests for the HuBERT phoneme→character index mapping (V0.5 P0)."""

from __future__ import annotations

from autosync.alignment.ctc import TokenSpan
from autosync.lyrics.ja_pronunciation import CharReading, locate_inner_offset
from autosync.models.phoneme_ctc_aligner import map_spans_to_char_units

FS = 0.02


def _spans(*pairs: tuple[int, int]) -> dict[int, TokenSpan]:
    return {i: TokenSpan(token=i, start_frame=a, end_frame=b, score=0.0) for i, (a, b) in enumerate(pairs)}


def test_locate_inner_offset_skips_leading_pau() -> None:
    assert locate_inner_offset(["pau", "s", "a", "i", "pau"], ["s", "a", "i"]) == 1
    assert locate_inner_offset(["s", "a", "pau"], ["s", "a"]) == 0


def test_first_char_does_not_inherit_leading_pau() -> None:
    full = ["pau", "s", "a", "i", "pau"]
    readings = [CharReading(char="最", kana="さ", phonemes=["s", "a"]),
                CharReading(char="後", kana="い", phonemes=["i"])]
    spans = _spans((0, 9), (10, 11), (12, 14), (15, 19), (20, 24))
    units = map_spans_to_char_units(readings, full, spans, FS)
    # 最 must start at the 's' frame (10), NOT at the leading pau (0).
    assert units[0].text == "最"
    assert abs(units[0].start - 10 * FS) < 1e-9
    assert abs(units[0].end - (14 + 1) * FS) < 1e-9
    assert abs(units[1].start - 15 * FS) < 1e-9


def test_multi_morae_kanji_spans_all_its_phonemes() -> None:
    # 後 -> いご -> i g o (a kanji is NOT assumed to be one syllable)
    full = ["pau", "i", "g", "o", "pau"]
    readings = [CharReading(char="後", kana="いご", phonemes=["i", "g", "o"])]
    spans = _spans((0, 4), (5, 6), (7, 9), (10, 12), (13, 14))
    units = map_spans_to_char_units(readings, full, spans, FS)
    assert abs(units[0].start - 5 * FS) < 1e-9
    assert abs(units[0].end - (12 + 1) * FS) < 1e-9


def test_repeated_phonemes_long_vowel() -> None:
    # ぞ -> z o ; ー -> o   (two consecutive identical 'o' phonemes)
    full = ["pau", "sh", "i", "N", "z", "o", "o", "pau"]
    readings = [CharReading(char="ぞ", kana="ぞ", phonemes=["z", "o"]),
                CharReading(char="ー", kana="ー", phonemes=["o"])]
    spans = _spans((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 15), (16, 17))
    units = map_spans_to_char_units(readings, full, spans, FS)
    assert abs(units[0].start - 8 * FS) < 1e-9
    assert abs(units[0].end - (11 + 1) * FS) < 1e-9   # z + first o
    assert abs(units[1].start - 12 * FS) < 1e-9        # second o (long vowel)
    assert abs(units[1].end - (15 + 1) * FS) < 1e-9


def test_sokuon_gets_closure_span() -> None:
    full = ["pau", "g", "a", "cl", "k", "o", "pau"]
    readings = [CharReading(char="が", kana="が", phonemes=["g", "a"]),
                CharReading(char="っ", kana="っ", phonemes=["cl"]),
                CharReading(char="こ", kana="こ", phonemes=["k", "o"])]
    spans = _spans((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13))
    units = map_spans_to_char_units(readings, full, spans, FS)
    assert abs(units[1].start - 6 * FS) < 1e-9
    assert abs(units[1].end - (7 + 1) * FS) < 1e-9


def test_unknown_phoneme_is_zero_width_at_neighbour_not_zero() -> None:
    # 'x' is not in the vocabulary -> dropped from ids -> no span.
    full = ["pau", "a", "x", "i", "pau"]
    readings = [CharReading(char="あ", kana="あ", phonemes=["a"]),
                CharReading(char="ｘ", kana="", phonemes=["x"], uncertain=True),
                CharReading(char="い", kana="い", phonemes=["i"])]
    # Index 2 ('x') has NO span (it was dropped as unknown); 0/1/3 exist.
    spans = {
        0: TokenSpan(0, 0, 1, 0.0),
        1: TokenSpan(1, 2, 4, 0.0),
        3: TokenSpan(3, 7, 9, 0.0),
    }
    units = map_spans_to_char_units(readings, full, spans, FS)
    assert units[1].uncertain is True
    # zero-width, placed at the previous character's end, not at 0
    assert units[1].start == units[1].end
    assert abs(units[1].start - (4 + 1) * FS) < 1e-9
    assert units[1].start != 0.0


def test_all_characters_preserved() -> None:
    full = ["pau", "k", "a", "pau"]
    readings = [CharReading(char="か", kana="か", phonemes=["k", "a"]),
                CharReading(char="！", kana="", phonemes=[], uncertain=True)]
    spans = _spans((0, 1), (2, 3), (4, 5), (6, 7))
    units = map_spans_to_char_units(readings, full, spans, FS)
    assert "".join(u.text for u in units) == "か！"
