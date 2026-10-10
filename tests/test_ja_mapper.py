"""Tests for the Japanese pronunciation mapper (V0.4 §2).

Uses injected fake G2P / kana→phoneme backends so no external dependency is
required.
"""

from __future__ import annotations

from autosync.lyrics.ja_pronunciation import JapanesePronunciationMapper, katakana_to_hiragana

# A tiny kana→phoneme table (longest-match), in the SHIRO/ENUNU style.
_TABLE = {
    "き": ["k", "i"], "きょ": ["ky", "o"], "ょ": ["y", "o"],
    "う": ["u"], "し": ["sh", "i"], "ん": ["N"], "ぶ": ["b", "u"],
    "っ": ["cl"], "こ": ["k", "o"], "が": ["g", "a"], "く": ["k", "u"],
    "さ": ["s", "a"], "い": ["i"], "ご": ["g", "o"], "の": ["n", "o"],
    "わ": ["w", "a"], "た": ["t", "a"], "ば": ["b", "a"], "じ": ["j", "i"],
}
_READINGS = {"最": "さ", "後": "いご", "味": "あじ", "歩": "ある"}


def _fake_g2p(text: str) -> str:
    out = []
    for ch in text:
        out.append(_READINGS.get(ch, ch))
    return "".join(out)


def _fake_convert(kana: str, table: dict) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(kana):
        for length in (2, 1):
            piece = kana[i : i + length]
            if piece in table:
                out.extend(table[piece])
                i += length
                break
        else:
            i += 1
    return ["pau", *out, "pau"]


def _mapper(**kw) -> JapanesePronunciationMapper:
    return JapanesePronunciationMapper(table=_TABLE, g2p=_fake_g2p, convert=_fake_convert, **kw)


def test_katakana_to_hiragana() -> None:
    assert katakana_to_hiragana("キス") == "きす"


def test_reconstruction_is_lossless() -> None:
    m = _mapper()
    readings, _ = m.map("最後のキス")
    assert m.reconstruct(readings) == "最後のキス"


def test_kanji_maps_to_multiple_morae() -> None:
    m = _mapper()
    readings, full = m.map("最後")
    # 最=さ, 後=いご -> 後 covers two morae; never 1 kanji == 1 syllable.
    assert readings[0].char == "最" and readings[0].phonemes == ["s", "a"]
    assert readings[1].char == "後" and readings[1].kana == "いご"
    assert full[0] == "pau" and full[-1] == "pau"


def test_youon_second_mora_marked_uncertain() -> None:
    m = _mapper()
    readings, _ = m.map("きょう")
    by_char = {r.char: r for r in readings}
    assert by_char["ょ"].uncertain is True  # merged into きょ


def test_g2p_failure_marks_uncertain_but_keeps_text() -> None:
    def boom(_t: str) -> str:
        raise RuntimeError("no g2p")

    m = JapanesePronunciationMapper(table=_TABLE, g2p=boom, convert=_fake_convert)
    readings, phonemes = m.map("あい")
    assert phonemes == []
    assert all(r.uncertain for r in readings)
    assert m.reconstruct(readings) == "あい"


def test_overrides_change_reading() -> None:
    m = _mapper(overrides={"キス": "きす"})
    readings, _ = m.map("キス")
    assert readings[0].kana == "き"
    assert readings[1].kana == "す" or readings[1].char == "ス"
    assert m.reconstruct(readings) == "キス"


def test_align_kana_to_chars_handles_particle_ha() -> None:
    # は is read as わ; the mapper must still map 1 kana char -> 1 kana mora.
    m = _mapper()
    spans = m.align_kana_to_chars(["は"], "わ")
    assert spans == [(0, 1)]
