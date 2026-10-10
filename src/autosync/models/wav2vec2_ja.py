"""Japanese adapter.

Model: ``reazon-research/japanese-wav2vec2-base-rs35kh`` (Apache-2.0, 96.7M
parameters, Wav2Vec2ForCTC), fine-tuned on ReazonSpeech.

Tokenisation research (see docs/MODEL_RESEARCH.md): the vocabulary is
character/sub-word based (kana, kanji and ``▁``-prefixed pieces), *not* a
phoneme inventory that requires an external G2P front-end. We therefore align
directly on the surface characters/sub-words of the lyric text, which also lets
us preserve the original Japanese text.

The mapping between *pronunciation units* (model tokens) and *original
characters* is exposed via ``AlignUnit.raw``. A single model token may cover
several kana/kanji, and one kanji may span several tokens — we never assume a
1:1 kanji↔syllable relation.
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseAligner
from .wav2vec2_ctc import Wav2Vec2CTCAligner


class JapaneseAligner(BaseAligner):
    def __new__(cls, cache_dir: Path | None = None) -> BaseAligner:  # type: ignore[misc]
        import os

        from .registry import JA_LICENSE, JA_REPO

        # Research-only override for alternative Japanese acoustic models.
        repo = os.environ.get("AUTOSYNC_JA_REPO", JA_REPO)
        return Wav2Vec2CTCAligner(
            repo,
            language="ja",
            license=JA_LICENSE,
            unit_mode="char",
            reading_fn=None,
            cache_dir=cache_dir,
            approx_size_mb=369,
        )


def build(cache_dir: Path | None = None) -> BaseAligner:
    return JapaneseAligner(cache_dir)
