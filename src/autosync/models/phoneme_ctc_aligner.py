"""Japanese phoneme CTC alignment adapter (candidate C).

Uses a Japanese HuBERT phoneme-CTC model (Apache-2.0,
``prj-beatrice/japanese-hubert-base-phoneme-ctc-v4``) to obtain a phoneme
posterior, aligns the **known** phoneme sequence with our own CTC Viterbi, and
maps phoneme intervals back to the original characters via
:class:`JapanesePronunciationMapper` (with a pyopenjtalk phoneme converter).

This is a research baseline: it is trained on speech (ReazonSpeech), so it may
suffer the same singing-domain mismatch as the wav2vec2 model. Reported honestly.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..alignment.ctc import InfeasibleAlignmentError, TokenSpan, forced_align
from ..config import HF_CACHE_DIR
from ..lyrics.ja_pronunciation import JapanesePronunciationMapper, locate_inner_offset
from ..types import AlignUnit
from .base import BaseAligner, ModelMeta

DEFAULT_REPO = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4"


def pyopenjtalk_kana(text: str) -> str:
    import pyopenjtalk

    return pyopenjtalk.g2p(text, kana=True)


def pyopenjtalk_phonemes(kana: str, _table: dict) -> list[str]:
    import pyopenjtalk

    phonemes = pyopenjtalk.g2p(kana).split()
    return ["pau", *phonemes, "pau"]


def map_spans_to_char_units(
    readings,
    phonemes: list[str],
    spans_by_phoneme: dict,
    frame_seconds: float,
):
    """Map per-phoneme CTC spans back to the original characters.

    * Uses the verified :func:`locate_inner_offset` (no leading-``pau`` leak).
    * Unknown / skipped phonemes simply have no span; a character with no known
      phoneme gets a zero-width interval placed at its nearest known boundary
      (never at time 0 by accident) and is marked ``uncertain``.
    * Every original character is kept, so the text stays reconstructible.
    """
    from ..types import AlignUnit

    inner = [p for r in readings for p in r.phonemes]
    offset = locate_inner_offset(phonemes, inner)

    resolved: list[tuple[float, float, bool] | None] = []
    cursor = offset
    for r in readings:
        n = len(r.phonemes)
        idxs = [cursor + k for k in range(n)]
        known = [spans_by_phoneme[i] for i in idxs if i in spans_by_phoneme]
        if known:
            start = known[0].start_frame * frame_seconds
            end = (known[-1].end_frame + 1) * frame_seconds
            resolved.append((start, end, bool(r.uncertain)))
        else:
            resolved.append(None)
        cursor += n

    # Nearest known boundaries for characters without any known phoneme.
    prev_end = [0.0] * len(resolved)
    pe = 0.0
    for i, val in enumerate(resolved):
        prev_end[i] = pe
        if val is not None:
            pe = val[1]
    next_start: list[float | None] = [None] * len(resolved)
    ns: float | None = None
    for i in range(len(resolved) - 1, -1, -1):
        if resolved[i] is not None:
            ns = resolved[i][0]
        next_start[i] = ns

    prev_known = [False] * len(resolved)
    flag = False
    for i, val in enumerate(resolved):
        prev_known[i] = flag
        if val is not None:
            flag = True

    units: list[AlignUnit] = []
    for i, r in enumerate(readings):
        val = resolved[i]
        if val is not None:
            s, e, unc = val
            units.append(AlignUnit(text=r.char, start=s, end=e, source="model", raw=r.kana or None, uncertain=unc))
            continue
        pos = prev_end[i] if prev_known[i] else (next_start[i] if next_start[i] is not None else 0.0)
        units.append(AlignUnit(text=r.char, start=pos, end=pos, source="model", raw=r.kana or None, uncertain=True))
    return units


class PhonemeCTCAligner(BaseAligner):
    def __init__(self, repo_id: str = DEFAULT_REPO, cache_dir: Path | None = None) -> None:
        self.repo_id = repo_id
        self.meta = ModelMeta(
            name=repo_id,
            version=repo_id,
            language="ja",
            license="apache-2.0",
            approx_size_mb=378,
        )
        # HuBERT base downsamples 16 kHz by 320 -> 20 ms frames.
        self.frame_seconds = 0.02
        self.load_seconds: float | None = None
        self.infer_seconds: float = 0.0
        self.cache_dir = Path(cache_dir) if cache_dir else HF_CACHE_DIR
        self._processor = None
        self._model = None
        self._torch = None
        self._vocab: dict[str, int] = {}
        self._mapper = JapanesePronunciationMapper(
            table={}, g2p=pyopenjtalk_kana, convert=pyopenjtalk_phonemes
        )

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self._torch = torch
        t0 = time.time()
        self._processor = AutoProcessor.from_pretrained(self.repo_id, cache_dir=str(self.cache_dir))
        self._model = AutoModelForCTC.from_pretrained(self.repo_id, cache_dir=str(self.cache_dir))
        self._model.eval()
        self._vocab = dict(self._processor.tokenizer.get_vocab())
        self.load_seconds = time.time() - t0

    def _blank_id(self) -> int:
        pad = self._processor.tokenizer.pad_token_id
        if pad is not None:
            return int(pad)
        return int(self._vocab.get("PAD", 0))

    def _phoneme_ids(self, phonemes: list[str]) -> tuple[list[int], list[int]]:
        """Return (ids, idx) keeping only phonemes present in the vocabulary."""
        ids: list[int] = []
        idx: list[int] = []
        for i, ph in enumerate(phonemes):
            tid = self._vocab.get(ph)
            if tid is None:
                continue
            ids.append(int(tid))
            idx.append(i)
        return ids, idx

    def _log_probs(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        assert self._processor is not None and self._model is not None
        torch = self._torch
        inputs = self._processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        with torch.inference_mode():
            logits = self._model(inputs.input_values).logits[0]
        return torch.log_softmax(logits.float(), dim=-1).numpy()

    def align_text(self, audio: np.ndarray, sample_rate: int, text: str) -> list[AlignUnit]:
        if self._model is None:
            self.load()
        assert self._processor is not None

        readings, phonemes = self._mapper.map(text)
        ids, keep = self._phoneme_ids(phonemes)
        if not ids:
            raise InfeasibleAlignmentError("no known phonemes for the lyric text")

        logp = self._log_probs(audio, sample_rate)
        al = forced_align(logp, ids, blank=self._blank_id())
        if not al.feasible:
            raise InfeasibleAlignmentError("phoneme alignment infeasible: " + "; ".join(al.reasons))

        spans_by_phoneme: dict[int, TokenSpan] = {}
        for pos, span in zip(keep, al.spans):
            if span is not None:
                spans_by_phoneme[pos] = span
        return map_spans_to_char_units(readings, phonemes, spans_by_phoneme, self.frame_seconds)
