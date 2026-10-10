"""Per-line language routing across independent language packs (V1.0 P2).

Keeps the "one model per language, loaded on demand" design: the router lazily
loads the model a line actually needs and **reuses** it for the rest of the song
(no per-line model reloading).

Routing rules live in :mod:`autosync.lyrics.language`; when a line cannot be
confidently routed the router raises :class:`LanguageRoutingError` so the caller
records an explicit failure/fallback instead of silently using a wrong model.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..lyrics.language import detect_line_language
from .base import BaseAligner, ModelMeta
from .registry import get_aligner


class LanguageRoutingError(RuntimeError):
    """The line language could not be determined confidently enough to route."""


class LanguageRouter(BaseAligner):
    def __init__(
        self,
        primary: str,
        cache_dir: Path | None = None,
        factory: Callable[[str], BaseAligner] | None = None,
    ) -> None:
        self.primary = primary
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._factory = factory or (lambda lang: get_aligner(lang, cache_dir=self.cache_dir))
        self._aligners: dict[str, BaseAligner] = {}
        self.last_route: dict | None = None
        self.last_used_lang: str | None = None
        self.infer_seconds = 0.0
        self.meta = ModelMeta(
            name="language-router",
            version="v1.0",
            language=primary,
            license="see routed models",
            status="routing",
            description="per-line language routing across independent language packs",
        )

    # ------------------------------------------------------------------ models
    def _get(self, language: str) -> BaseAligner:
        if language not in self._aligners:
            aligner = self._factory(language)
            aligner.load()
            self._aligners[language] = aligner
        return self._aligners[language]

    def model_names(self) -> dict[str, str]:
        return {lang: a.meta.name for lang, a in self._aligners.items()}

    def load(self) -> None:
        self._get(self.primary)

    def unload(self) -> None:
        for aligner in self._aligners.values():
            aligner.unload()
        self._aligners.clear()
        self.last_used_lang = None

    # ------------------------------------------------------------------ state
    @property
    def frame_seconds(self) -> float:  # type: ignore[override]
        if self.last_used_lang and self.last_used_lang in self._aligners:
            return float(getattr(self._aligners[self.last_used_lang], "frame_seconds", 0.02) or 0.02)
        return 0.02

    @property
    def last_quality(self):
        aligner = self._aligners.get(self.last_used_lang) if self.last_used_lang else None
        return getattr(aligner, "last_quality", None) if aligner is not None else None

    @last_quality.setter
    def last_quality(self, _value) -> None:
        # the caller resets quality before each line; propagate to all packs
        for aligner in self._aligners.values():
            if hasattr(aligner, "last_quality"):
                aligner.last_quality = None

    # ------------------------------------------------------------------ align
    def align_text(self, audio: np.ndarray, sample_rate: int, text: str):
        decision = detect_line_language(text, self.primary)
        language, routable, reason = decision.language, decision.routable, decision.reason
        # pyopenjtalk (the ja G2P) reads Latin tokens as katakana, so a Japanese
        # song can absorb Latin words; the zh normaliser drops them, so zh stays
        # conservative. This mirrors the pre-routing behaviour for ja songs.
        if not routable and self.primary == "ja" and language in ("mixed", "latin_ambiguous"):
            language, routable = "ja", True
            reason = f"{reason}; ja G2P reads Latin tokens (approximate)"

        self.last_route = {
            "line_language": language,
            "routable": routable,
            "route_model": None,
            "route_reason": reason,
            "script": decision.script_summary,
        }
        if not routable:
            raise LanguageRoutingError(f"{language}: {reason}")
        aligner = self._get(language)
        self.last_used_lang = language
        self.last_route["route_model"] = aligner.meta.name
        return aligner.align_text(audio, sample_rate, text)
