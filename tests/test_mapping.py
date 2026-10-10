"""Regression tests for token -> original-text mapping and reconstruction."""

from __future__ import annotations

from autosync.alignment.segment import check_reconstruction
from autosync.models.wav2vec2_ctc import normalized_index_map
from autosync.types import AlignUnit


def test_normalized_index_map_skips_punctuation() -> None:
    assert normalized_index_map("a\uFF0Cb", "ab") == [0, 2]
    assert normalized_index_map("AB", "AB") == [0, 1]


def test_normalized_index_map_exact_only() -> None:
    # 'X' is not present in the original -> mapping is not exact -> None
    assert normalized_index_map("abc", "abd") is None


def test_reconstruction_cjk_mismatch_detected() -> None:
    units = [AlignUnit(text="你", start=0, end=1)]
    assert check_reconstruction("zh", "你好，世界", units) is not None


def test_reconstruction_cjk_ok() -> None:
    units = [
        AlignUnit(text="你", start=0, end=1),
        AlignUnit(text="好", start=1, end=2),
        AlignUnit(text="世", start=2, end=3),
        AlignUnit(text="界", start=3, end=4),
    ]
    assert check_reconstruction("zh", "你好，世界", units) is None


def test_reconstruction_english_ok() -> None:
    units = [
        AlignUnit(text="Hello", start=0, end=1),
        AlignUnit(text="World", start=1, end=2),
    ]
    assert check_reconstruction("en", "Hello, world!", units) is None


def test_reconstruction_english_mismatch() -> None:
    units = [AlignUnit(text="Hello", start=0, end=1)]
    assert check_reconstruction("en", "Hello world", units) is not None
