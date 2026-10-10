"""Japanese ReazonSpeech wav2vec2 baseline adapter (kept as an optional backend).

V0.8: the *default* Japanese backend is now the HuBERT phoneme CTC model; this
speech-trained character/subword CTC model remains selectable as the
``reazonspeech-wav2vec2-base`` experimental baseline. ``AUTOSYNC_JA_REPO`` may
still override the repository for research (e.g. a large variant), but that is
never a shipped default.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import BaseAligner
from .wav2vec2_ctc import Wav2Vec2CTCAligner

DEFAULT_JA_REPO = "reazon-research/japanese-wav2vec2-base-rs35kh"
JA_LICENSE = "apache-2.0"


class JapaneseAligner(BaseAligner):
    def __new__(cls, cache_dir: Path | None = None) -> BaseAligner:  # type: ignore[misc]
        repo = os.environ.get("AUTOSYNC_JA_REPO", DEFAULT_JA_REPO)
        aligner = Wav2Vec2CTCAligner(
            repo,
            language="ja",
            license=JA_LICENSE,
            unit_mode="char",
            reading_fn=None,
            cache_dir=cache_dir,
            approx_size_mb=369,
        )
        aligner.meta.status = "experimental"
        aligner.meta.description = "speech-trained char/subword CTC baseline"
        return aligner


def build(cache_dir: Path | None = None) -> BaseAligner:
    return JapaneseAligner(cache_dir)
