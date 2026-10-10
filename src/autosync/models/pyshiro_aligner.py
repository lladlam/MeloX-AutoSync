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
from ..lyrics.ja_pronunciation import JapanesePronunciationMapper
from ..types import AlignUnit
from .base import BaseAligner, ModelMeta

PYSHIRO_HOP_SECONDS = 0.005


def _ensure_pyshiro_on_path() -> None:
    path = os.environ.get("AUTOSYNC_PYSHIRO_PATH", str(MODELS_DIR / "pyshiro"))
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _state_count(phonemap: dict, phoneme: str) -> int:
    entry = phonemap.get(phoneme)
    if not entry:
        return 3
    if float(entry.get("pskip", 0.0)) > 0.0:
        return 0
    states = entry.get("states")
    return len(states) if states else 3


def phoneme_intervals(phonemap: dict, phonemes: list[str], segments) -> list[tuple[int, int, str]]:
    """Aggregate per-state segments into per-phoneme ``(start, end, phoneme)``.

    Uses the phonemap state counts (not a fixed ``3``) so that skipped phonemes
    and consecutive identical phonemes are handled correctly.
    """
    intervals: list[tuple[int, int, str]] = []
    cursor = 0
    for ph in phonemes:
        count = _state_count(phonemap, ph)
        if count <= 0:
            pos = segments[cursor][0] if cursor < len(segments) else intervals[-1][1]
            intervals.append((pos, pos, ph))
            continue
        start = segments[cursor][0]
        end = segments[min(cursor + count - 1, len(segments) - 1)][1]
        intervals.append((start, end, ph))
        cursor += count
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
        intervals = phoneme_intervals(self._phonemap, phonemes, refined)

        offset = 1 if phonemes and phonemes[0] == "pau" else 0
        ph_index = offset
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
