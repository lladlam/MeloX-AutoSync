"""Generic adapter for Hugging Face ``Wav2Vec2ForCTC`` acoustic models.

Used for English (``facebook/wav2vec2-base-960h``) and Japanese
(``reazon-research/japanese-wav2vec2-base-rs35kh``). The heavy lifting (Viterbi
alignment) is delegated to :mod:`autosync.alignment.ctc`.

Original-text mapping (V0.2)
----------------------------
The acoustic model matches tokens; the *user* wants the original lyrics. We map
every model token back to the original surface characters:

* For each model token we know the normalised substring it emits; we rebuild the
  original characters at the same positions.
* A token that covers **more than one original character** (common in Japanese,
  e.g. ``世界`` or ``あり``) is emitted as one unit and flagged
  ``uncertain=True``: its outer boundaries are model-derived but the internal
  character split cannot be recovered exactly, so we do not pretend it can.
* The concatenated unit texts are checked against the normalised line, so that
  "the original text can be recovered" is verified, not assumed.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..alignment.ctc import InfeasibleAlignmentError, TokenSpan, forced_align
from ..config import HF_CACHE_DIR
from ..lyrics.normalize import normalize_cjk, normalize_en
from ..types import AlignUnit
from .base import BaseAligner, ModelMeta

# wav2vec2-base downsamples the 16 kHz waveform by 320 -> 20 ms frames.
_WAV2VEC2_FRAME_SECONDS = 0.02

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SPECIALS = {"<pad>", "<s>", "</s>", "<unk>", "|", " "}
_WORD_DELIM = "|"


def normalized_index_map(original: str, normalized: str) -> list[int] | None:
    """Map each normalised character index to its index in ``original``.

    ``normalize_*`` only *deletes* characters (it never reorders), so a greedy
    scan recovers the positions. Returns ``None`` if the mapping is not exact.
    """
    positions: list[int] = []
    j = 0
    for i, ch in enumerate(original):
        if j < len(normalized) and ch == normalized[j]:
            positions.append(i)
            j += 1
    if j == len(normalized) and len(positions) == len(normalized):
        return positions
    return None


class Wav2Vec2CTCAligner(BaseAligner):
    def __init__(
        self,
        repo_id: str,
        language: str,
        license: str,
        unit_mode: str = "word",
        reading_fn: Callable[[str], tuple[str, list[str]]] | None = None,
        cache_dir: Path | None = None,
        approx_size_mb: int = 0,
    ) -> None:
        if unit_mode not in ("word", "char"):
            raise ValueError(f"unsupported unit_mode: {unit_mode}")
        self.meta = ModelMeta(
            name=repo_id,
            version="",
            language=language,
            license=license,
            approx_size_mb=approx_size_mb,
        )
        self.repo_id = repo_id
        self.unit_mode = unit_mode
        self.reading_fn = reading_fn
        self.cache_dir = Path(cache_dir) if cache_dir else HF_CACHE_DIR
        self.frame_seconds = _WAV2VEC2_FRAME_SECONDS
        self.load_seconds: float | None = None
        self.infer_seconds: float = 0.0
        self._processor = None
        self._model = None
        self._torch = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self._torch = torch
        t0 = time.time()
        self._processor = AutoProcessor.from_pretrained(
            self.repo_id, cache_dir=str(self.cache_dir)
        )
        self._model = Wav2Vec2ForCTC.from_pretrained(
            self.repo_id, cache_dir=str(self.cache_dir)
        )
        self._model.eval()
        self.load_seconds = time.time() - t0
        self.meta.version = getattr(self._model.config, "_name_or_path", self.repo_id)

    # --------------------------------------------------------------- inference
    def _log_probs(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        assert self._model is not None and self._processor is not None
        torch = self._torch
        inputs = self._processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        with torch.inference_mode():
            logits = self._model(inputs.input_values).logits[0]
            logp = torch.log_softmax(logits.float(), dim=-1)
        return logp.numpy()

    # ------------------------------------------------------------------ align
    def align_text(self, audio: np.ndarray, sample_rate: int, text: str) -> list[AlignUnit]:
        if self._model is None:
            self.load()
        assert self._processor is not None

        if self.unit_mode == "word":
            return self._align_words(audio, sample_rate, text)
        return self._align_chars(audio, sample_rate, text)

    def _align_words(self, audio: np.ndarray, sample_rate: int, text: str) -> list[AlignUnit]:
        normalized = normalize_en(text)
        tokens = self._tokenize(normalized)
        if not tokens:
            return []
        spans = self._align(audio, sample_rate, tokens)
        chars = self._ids_to_chars(tokens)

        groups: list[list[TokenSpan]] = []
        current: list[TokenSpan] = []
        for ch, span in zip(chars, spans):
            if ch == _WORD_DELIM or ch == " ":
                if current:
                    groups.append(current)
                    current = []
                continue
            if ch in _SPECIALS:
                continue
            if span is not None:
                current.append(span)
        if current:
            groups.append(current)

        original_words = _WORD_RE.findall(text)
        norm_words = normalized.split()
        display = original_words if len(original_words) == len(groups) else norm_words

        fs = self.frame_seconds
        units: list[AlignUnit] = []
        for idx, group in enumerate(groups):
            start = group[0].start_frame * fs
            end = (group[-1].end_frame + 1) * fs
            score = float(np.mean([s.score for s in group]))
            word = display[idx] if idx < len(display) else ""
            units.append(
                AlignUnit(text=word, start=start, end=end, source="model", score=score, raw=word)
            )
        return units

    def _align_chars(self, audio: np.ndarray, sample_rate: int, text: str) -> list[AlignUnit]:
        normalized = normalize_cjk(text)
        if self.reading_fn is not None:
            reading, _ = self.reading_fn(normalized)
        else:
            reading = normalized
        tokens = self._tokenize(reading)
        if not tokens:
            return []
        spans = self._align(audio, sample_rate, tokens)
        chars = self._ids_to_chars(tokens)

        pieces: list[tuple[str, TokenSpan]] = []
        for raw_tok, span in zip(chars, spans):
            if raw_tok in _SPECIALS or span is None:
                continue
            clean = raw_tok.replace("\u2581", "").strip()
            if clean:
                pieces.append((clean, span))

        joined = "".join(p[0] for p in pieces)
        index_map = normalized_index_map(text, normalized) if reading == normalized else None
        if index_map is not None and joined != normalized:
            # Tokenisation did not reproduce the normalised text exactly; do not
            # pretend we can map positions onto the original characters.
            index_map = None

        fs = self.frame_seconds
        units: list[AlignUnit] = []
        cursor = 0  # position within ``normalized`` (== concatenation of pieces)
        for clean, span in pieces:
            piece_len = len(clean)
            start = span.start_frame * fs
            end = (span.end_frame + 1) * fs
            surface = clean
            if index_map is not None and cursor + piece_len <= len(normalized):
                positions = index_map[cursor : cursor + piece_len]
                surface = "".join(text[i] for i in positions)
            units.append(
                AlignUnit(
                    text=surface,
                    start=start,
                    end=end,
                    source="model",
                    score=span.score,
                    raw=clean,
                    # A token covering several original characters has an
                    # internal boundary we cannot recover exactly.
                    uncertain=piece_len > 1,
                )
            )
            cursor += piece_len
        return units

    # ---------------------------------------------------------------- helpers
    def _align(self, audio: np.ndarray, sample_rate: int, tokens: list[int]) -> list[TokenSpan | None]:
        logp = self._log_probs(audio, sample_rate)
        al = forced_align(logp, tokens, blank=self._blank_id())
        if not al.feasible:
            raise InfeasibleAlignmentError(
                "alignment infeasible: " + "; ".join(al.reasons)
            )
        return al.spans

    def _blank_id(self) -> int:
        return int(self._processor.tokenizer.pad_token_id or 0)

    def _tokenize(self, text: str) -> list[int]:
        enc = self._processor.tokenizer(text)
        return list(enc["input_ids"] if isinstance(enc, dict) else enc.input_ids)

    def _ids_to_chars(self, tokens: list[int]) -> list[str]:
        return [self._processor.tokenizer.convert_ids_to_tokens(t) for t in tokens]
