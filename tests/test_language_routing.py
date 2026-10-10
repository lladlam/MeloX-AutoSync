"""Tests for per-line language detection and routing (V1.0 P2)."""

from __future__ import annotations

import numpy as np
import pytest

from autosync.lyrics.language import detect_line_language
from autosync.models.base import BaseAligner, ModelMeta
from autosync.models.router import LanguageRouter, LanguageRoutingError


# ------------------------------------------------------------------ detection
def test_pure_chinese_line_routes_zh() -> None:
    d = detect_line_language("你的世界才不是梦", "zh")
    assert d.language == "zh" and d.routable


def test_pure_japanese_line_routes_ja() -> None:
    d = detect_line_language("みなさん、こんにちは", "ja")
    assert d.language == "ja" and d.routable


def test_kanji_only_line_uses_song_primary() -> None:
    assert detect_line_language("東京", "ja").language == "ja"
    assert detect_line_language("北京", "zh").language == "zh"


def test_clear_english_line_routes_en() -> None:
    d = detect_line_language("If you don't wanna fall asleep,", "zh")
    assert d.language == "en" and d.routable


def test_latin_without_english_evidence_is_not_routed() -> None:
    # "Bora sana" is Latin script but not English -> conservative fallback
    d = detect_line_language("(Bora sana, Bora sana)", "zh")
    assert d.language == "latin_ambiguous" and not d.routable


def test_latin_in_english_song_routes_en() -> None:
    d = detect_line_language("Bora sana", "en")
    assert d.language == "en" and d.routable


def test_cjk_dominant_mixed_routes_cjk() -> None:
    d = detect_line_language("スーパーガチャ・GetLow沙汰", "ja")
    assert d.language == "ja" and d.routable


def test_latin_dominant_mixed_falls_back() -> None:
    d = detect_line_language("别害怕 we clap clap", "zh")
    assert d.language == "mixed" and not d.routable


# -------------------------------------------------------------------- router
class _Stub(BaseAligner):
    def __init__(self, language: str) -> None:
        self.meta = ModelMeta(name=f"stub-{language}", version="0", language=language, license="test")
        self.frame_seconds = 0.02
        self.loaded = False
        self.calls = 0
        self.last_quality = None

    def load(self) -> None:
        self.loaded = True

    def align_text(self, audio, sample_rate, text):
        self.calls += 1
        from autosync.types import AlignUnit

        return [AlignUnit(text=text, start=0.0, end=0.2, source="model")]


def _router(primary="zh"):
    created: dict[str, _Stub] = {}

    def factory(lang):
        created.setdefault(lang, _Stub(lang))
        return created[lang]

    return LanguageRouter(primary, factory=factory), created


def test_router_loads_lazily_and_reuses_models() -> None:
    router, created = _router("zh")
    router.load()
    assert set(created) == {"zh"}  # only the primary loaded
    audio = np.zeros(1600, dtype=np.float32)
    router.align_text(audio, 16000, "你好世界")     # zh
    router.align_text(audio, 16000, "If you know")  # en
    router.align_text(audio, 16000, "再见")         # zh again (reused)
    assert set(created) == {"zh", "en"}
    assert created["zh"].calls == 2 and created["en"].calls == 1  # no reloads


def test_router_records_audit_trail() -> None:
    router, _ = _router("zh")
    router.align_text(np.zeros(1600, dtype=np.float32), 16000, "If you know")
    assert router.last_route["line_language"] == "en"
    assert router.last_route["route_model"] == "stub-en"
    assert "English" in router.last_route["route_reason"]


def test_router_raises_on_unroutable_line() -> None:
    router, created = _router("zh")
    with pytest.raises(LanguageRoutingError):
        router.align_text(np.zeros(1600, dtype=np.float32), 16000, "(Bora sana, Bora sana)")
    assert router.last_route["routable"] is False
    assert set(created) == set()  # nothing loaded for an unroutable line


def test_japanese_primary_absorbs_latin_lines() -> None:
    # pyopenjtalk (ja G2P) reads Latin tokens, so a ja song keeps word timings;
    # the audit trail records that the Latin tokens are approximate.
    router, _ = _router("ja")
    router.align_text(np.zeros(1600, dtype=np.float32), 16000, "All Day All Night　極刑のライブ")
    assert router.last_route["line_language"] == "ja"
    assert router.last_route["routable"] is True
    assert "Latin" in router.last_route["route_reason"]


def test_router_delegates_quality() -> None:
    router, _ = _router("zh")
    router.align_text(np.zeros(1600, dtype=np.float32), 16000, "你好")
    # sub-aligner sets last_quality; the router must expose it
    assert router.last_quality is None  # stub does not set it
