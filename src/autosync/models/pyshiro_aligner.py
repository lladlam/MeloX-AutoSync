"""Japanese singing forced alignment via pyshiro (SHIRO / HSMM).

pyshiro (https://github.com/wavtechyukky/pyshiro) is a Python reimplementation
of the SHIRO HSMM phoneme aligner, targeted at **Japanese singing**. It ships
phoneme-level checkpoints (``v1``, ``v2``) trained on Japanese singing corpora.

This adapter:
1. uses :class:`JapanesePronunciationMapper` to turn kanji/kana lyrics into a
   kana reading and the SHIRO/ENUNU phoneme sequence;
2. runs pyshiro's two-pass HMM→HSMM alignment to get phoneme intervals;
3. maps phoneme intervals back to the **original characters** (a kanji may span
   several morae; a merged youon is marked uncertain).

Licensing
---------
pyshiro **code is GPLv3**. The pretrained ``.hsmm`` weights ship in the repo
(and a submodule) **without an explicit weight licence**; they are therefore
used for local research only and must **not** be bundled/redistributed. See
docs/JAPANESE_MODEL_COMPARISON.md and docs/JAPANESE_ANDROID_FEASIBILITY.md.

The pyshiro package is NOT vendored; point ``AUTOSYNC_PYSHIRO_PATH`` at a clone
(default: ``models/pyshiro``).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from ..config import MODELS_DIR
from ..lyrics.ja_pronunciation import JapanesePronunciationMapper, locate_inner_offset
from ..types import AlignUnit
from .base import BaseAligner, ModelMeta

PYSHIRO_HOP_SECONDS = 0.005


def _ensure_pyshiro_on_path() -> None:
    path = os.environ.get("AUTOSYNC_PYSHIRO_PATH", str(MODELS_DIR / "pyshiro"))
    if path and path not in sys.path:
        sys.path.insert(0, path)


def phoneme_intervals_from_starts(
    phonemes: list[str], state_starts: list[int], segments
) -> list[tuple[int, int, str]]:
    """Aggregate per-state segments into per-phoneme ``(start, end, phoneme)``.

    ``state_starts[i]`` is the index of the first state belonging to phoneme
    ``i`` and ``state_starts[-1]`` the total state count. Deriving the blocks
    from the **actual built state sequence** keeps us consistent with the
    backtracked path: a ``pskip`` phoneme still owns its states (skip is an
    optional edge, not a forced removal), and two consecutive identical
    phonemes stay separate. A phoneme that the path skipped receives a
    zero-width interval.
    """
    intervals: list[tuple[int, int, str]] = []
    prev_end = 0
    for i, ph in enumerate(phonemes):
        a = state_starts[i]
        b = state_starts[i + 1]
        if b > a:
            start = segments[a][0]
            end = segments[b - 1][1]
        else:
            pos = segments[a][0] if a < len(segments) else prev_end
            start = end = pos
        intervals.append((start, end, ph))
        prev_end = end
    return intervals


class PySHIROAligner(BaseAligner):
    def __init__(
        self,
        version: str = "v2",
        checkpoint_dir: Path | None = None,
        mapper: JapanesePronunciationMapper | None = None,
    ) -> None:
        if version not in ("v1", "v2"):
            raise ValueError("pyshiro version must be 'v1' or 'v2'")
        self.version = version
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else (
            MODELS_DIR / "pyshiro" / "checkpoint"
        )
        self.meta = ModelMeta(
            name=f"pyshiro-jp-{version}",
            version=version,
            language="ja",
            license="GPLv3 code / weights: no explicit licence",
            approx_size_mb=1,
        )
        self.frame_seconds = PYSHIRO_HOP_SECONDS
        self.load_seconds: float | None = None
        self.infer_seconds: float = 0.0
        self._mapper = mapper
        self._model = None
        self._phonemap = None
        self._pyshiro = None

    def load(self) -> None:
        _ensure_pyshiro_on_path()
        import pyshiro  # type: ignore

        t0 = time.time()
        model_path = self.checkpoint_dir / f"pyshiro-jp-{self.version}.hsmm"
        phonemap_path = self.checkpoint_dir / f"pyshiro-jp-{self.version}_phonemap.json"
        if not model_path.exists():
            raise FileNotFoundError(f"pyshiro checkpoint not found: {model_path}")
        self._pyshiro = pyshiro
        self._model = pyshiro.load_hsmm(model_path)
        self._phonemap = pyshiro.load_phonemap(phonemap_path)
        if self._mapper is None:
            self._mapper = JapanesePronunciationMapper()
        self.load_seconds = time.time() - t0

    def align_text(self, audio: np.ndarray, sample_rate: int, text: str) -> list[AlignUnit]:
        if self._model is None:
            self.load()
        assert self._mapper is not None and self._phonemap is not None and self._pyshiro is not None

        readings, phonemes = self._mapper.map(text)
        if not phonemes:
            return []
        hop = PYSHIRO_HOP_SECONDS
        streams = self._pyshiro.extract_mfcc(np.ascontiguousarray(audio, dtype=np.float32), sample_rate)
        total_frames = streams[0].shape[0]
        state_seq = self._pyshiro.build_state_sequence(phonemes, self._phonemap, total_frames)
        refined, _loglik = self._pyshiro.forced_align_2pass(self._model, streams, state_seq)

        # Per-phoneme state blocks, derived from the *actual* built state
        # sequence (robust to pskip and consecutive identical phonemes).
        state_starts = [0]
        for i in range(1, len(phonemes) + 1):
            state_starts.append(
                len(self._pyshiro.build_state_sequence(phonemes[:i], self._phonemap, total_frames))
            )
        intervals = phoneme_intervals_from_starts(phonemes, state_starts, refined)

        inner = [p for r in readings for p in r.phonemes]
        ph_index = locate_inner_offset(phonemes, inner)
        last_end = 0.0
        units: list[AlignUnit] = []
        for r in readings:
            n = len(r.phonemes)
            if n == 0:
                pos = intervals[ph_index][0] * hop if ph_index < len(intervals) else last_end
                units.append(
                    AlignUnit(text=r.char, start=pos, end=pos, source="model", raw=r.kana or None, uncertain=True)
                )
                continue
            start = intervals[ph_index][0] * hop
            end = intervals[min(ph_index + n - 1, len(intervals) - 1)][1] * hop
            if end < start:
                start, end = end, start
            units.append(
                AlignUnit(
                    text=r.char,
                    start=start,
                    end=end,
                    source="model",
                    raw=r.kana or None,
                    uncertain=r.uncertain,
                )
            )
            ph_index += n
            last_end = end
        return units
