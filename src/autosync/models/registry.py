"""Model registry.

Only the language that is actually requested gets its adapter constructed, and
the adapter only loads weights in :meth:`load`. This keeps the "do not force
downloading all models" requirement.
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseAligner
from .wav2vec2_ctc import Wav2Vec2CTCAligner

SUPPORTED_LANGUAGES = ("en", "ja", "zh")

# Repository ids and licensing (verified against the model cards).
EN_REPO = "facebook/wav2vec2-base-960h"
EN_LICENSE = "apache-2.0"

JA_REPO = "reazon-research/japanese-wav2vec2-base-rs35kh"
JA_LICENSE = "apache-2.0"

ZH_REPO = "funasr/fa-zh"
ZH_LICENSE = "model-license (FunASR)"


def get_aligner(
    language: str,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> BaseAligner:
    if language == "en":
        return Wav2Vec2CTCAligner(
            EN_REPO,
            language="en",
            license=EN_LICENSE,
            unit_mode="word",
            cache_dir=cache_dir,
            approx_size_mb=380,
        )
    if language == "ja":
        from .wav2vec2_ja import JapaneseAligner

        return JapaneseAligner(cache_dir=cache_dir)
    if language == "zh":
        from .funasr_zh import ChineseFunASRAligner

        return ChineseFunASRAligner(cache_dir=cache_dir)
    raise ValueError(f"unsupported language: {language!r} (expected one of {SUPPORTED_LANGUAGES})")
