"""Model registry (V0.8).

Explicit, per-language backends. Only the requested language/backend is
constructed, and weights are loaded lazily in :meth:`BaseAligner.load`.

Defaults (V0.8)
---------------
* en: ``wav2vec2-base-960h``            (stable)
* ja: ``hubert-phoneme-ctc-v4``         (experimental — the best Japanese candidate)
* zh: ``funasr-fa-zh``                  (stable)

Additional backends are selectable via ``autosync align --backend`` /
:func:`get_aligner`:

* ``reazonspeech-wav2vec2-base`` (ja, experimental baseline — the pre-V0.4 model)
* ``pyshiro-v2`` (ja, research only — **weights have no explicit licence**)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import BaseAligner

SUPPORTED_LANGUAGES = ("en", "ja", "zh")


@dataclass(frozen=True)
class BackendSpec:
    key: str
    language: str
    repo: str
    license: str
    status: str  # stable | experimental | research
    description: str

    @property
    def distributable(self) -> bool:
        return self.status == "stable" or self.license == "apache-2.0"


BACKENDS: dict[tuple[str, str], BackendSpec] = {
    ("en", "wav2vec2-base-960h"): BackendSpec(
        "wav2vec2-base-960h", "en", "facebook/wav2vec2-base-960h", "apache-2.0", "stable",
        "character CTC, word-level alignment",
    ),
    ("ja", "hubert-phoneme-ctc-v4"): BackendSpec(
        "hubert-phoneme-ctc-v4", "ja", "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4",
        "apache-2.0", "experimental", "HuBERT phoneme CTC + our Viterbi (best JA candidate)",
    ),
    ("ja", "reazonspeech-wav2vec2-base"): BackendSpec(
        "reazonspeech-wav2vec2-base", "ja", "reazon-research/japanese-wav2vec2-base-rs35kh",
        "apache-2.0", "experimental", "speech-trained char/subword CTC baseline",
    ),
    ("ja", "pyshiro-v2"): BackendSpec(
        "pyshiro-v2", "ja", "pyshiro-jp-v2", "GPLv3 code / weights: no explicit licence", "research",
        "singing HSMM; weights not distributable — local research only",
    ),
    ("zh", "funasr-fa-zh"): BackendSpec(
        "funasr-fa-zh", "zh", "funasr/fa-zh", "model-license (FunASR)", "stable",
        "Paraformer timestamp prediction, hanzi timestamps",
    ),
}

DEFAULT_BACKENDS: dict[str, str] = {
    "en": "wav2vec2-base-960h",
    "ja": "hubert-phoneme-ctc-v4",
    "zh": "funasr-fa-zh",
}


def list_backends(language: str | None = None) -> list[BackendSpec]:
    specs = [s for (lang, _), s in BACKENDS.items() if language is None or lang == language]
    return sorted(specs, key=lambda s: (s.language, s.key != DEFAULT_BACKENDS.get(s.language, ""), s.key))


def resolve_backend(language: str, backend: str | None = None) -> BackendSpec:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported language: {language!r} (expected one of {SUPPORTED_LANGUAGES})")
    key = backend or DEFAULT_BACKENDS[language]
    spec = BACKENDS.get((language, key))
    if spec is None:
        available = [s.key for s in list_backends(language)]
        raise ValueError(f"unknown backend {key!r} for {language}; available: {available}")
    return spec


def get_aligner(
    language: str,
    backend: str | None = None,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> BaseAligner:
    """Construct the requested (or default) backend for ``language``.

    Weights are **not** loaded here; call :meth:`BaseAligner.load`.
    """
    spec = resolve_backend(language, backend)
    if spec.key == "wav2vec2-base-960h":
        from .wav2vec2_ctc import Wav2Vec2CTCAligner

        aligner: BaseAligner = Wav2Vec2CTCAligner(
            spec.repo, language="en", license=spec.license, unit_mode="word",
            cache_dir=cache_dir, approx_size_mb=380,
        )
    elif spec.key == "reazonspeech-wav2vec2-base":
        from .wav2vec2_ctc import Wav2Vec2CTCAligner

        aligner = Wav2Vec2CTCAligner(
            spec.repo, language="ja", license=spec.license, unit_mode="char",
            cache_dir=cache_dir, approx_size_mb=369,
        )
    elif spec.key == "hubert-phoneme-ctc-v4":
        from .phoneme_ctc_aligner import PhonemeCTCAligner

        aligner = PhonemeCTCAligner(repo_id=spec.repo, cache_dir=cache_dir)
    elif spec.key == "pyshiro-v2":
        from .pyshiro_aligner import PySHIROAligner

        aligner = PySHIROAligner("v2")
    elif spec.key == "funasr-fa-zh":
        from .funasr_zh import ChineseFunASRAligner

        aligner = ChineseFunASRAligner(cache_dir=cache_dir)
    else:  # pragma: no cover - defensive
        raise ValueError(f"backend {spec.key!r} not implemented")

    # Single source of truth for metadata.
    aligner.meta.name = spec.repo
    aligner.meta.license = spec.license
    aligner.meta.status = spec.status
    aligner.meta.description = spec.description
    return aligner
