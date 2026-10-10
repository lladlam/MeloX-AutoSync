from __future__ import annotations

from autosync.lyrics.normalize import (
    chars_cjk,
    normalize_cjk,
    normalize_en,
    words_en,
)


def test_normalize_en_uppercases_and_strips_punct() -> None:
    assert normalize_en("Hello, world!") == "HELLO WORLD"
    assert normalize_en("Don't stop") == "DON'T STOP"


def test_words_en() -> None:
    assert words_en("HELLO WORLD") == ["HELLO", "WORLD"]
    assert words_en("") == []


def test_normalize_cjk_keeps_hanzi_and_kana() -> None:
    assert normalize_cjk("你好，世界！") == "你好世界"
    assert normalize_cjk("こんにちは、世界。") == "こんにちは世界"


def test_chars_cjk() -> None:
    assert chars_cjk("你好世界") == ["你", "好", "世", "界"]
    assert chars_cjk("こ ん") == ["こ", "ん"]
