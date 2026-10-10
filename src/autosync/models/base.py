"""Language model adapter interface.

Each language gets its own adapter. Adapters are deliberately decoupled: they
know nothing about LRC parsing or JSON export, and the alignment pipeline knows
nothing about which framework (transformers / FunASR) is underneath.

Two families exist:

* CTC models (wav2vec2 family): expose frame-level log-probabilities and a
  tokenizer; the shared :mod:`autosync.alignment.ctc` aligner does the work.
* Native timestamp models (FunASR ``fa-zh``): directly return per-unit times.

Both are hidden behind :class:`BaseAligner.align_text`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..types import AlignUnit


@dataclass
class ModelMeta:
    name: str
    version: str
    language: str
    license: str
    approx_size_mb: int = 0


class BaseAligner(ABC):
    """Common interface implemented by every language model adapter."""

    meta: ModelMeta
    frame_seconds: float = 0.02

    @abstractmethod
    def load(self) -> None:
        """Download (if needed) and load weights into memory."""

    @abstractmethod
    def align_text(self, audio: np.ndarray, sample_rate: int, text: str) -> list[AlignUnit]:
        """Align ``text`` against ``audio``.

        Returns units with times **relative to the start of ``audio``** (seconds).
        The caller maps them back to absolute song time.
        """

    def unload(self) -> None:
        """Release resources. Optional."""
        return
