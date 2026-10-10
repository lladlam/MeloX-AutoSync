"""Generic adapter for Hugging Face ``Wav2Vec2ForCTC`` acoustic models.

Used for English (``facebook/wav2vec2-base-960h``) and Japanese
(``reazon-research/japanese-wav2vec2-base-rs35kh``). The heavy lifting (Viterbi
alignment) is delegated to :mod:`autosync.alignment.ctc`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..alignment.ctc import TokenSpan, forced_align
from ..config import HF_CACHE_DIR
from ..lyrics.normalize import normalize_cjk, normalize_en, words_en
from ..types import AlignUnit
from .base import BaseAligner, ModelMeta

# wav2vec2-base downsamples the 16 kHz waveform by 320 -> 20 ms frames.
_WAV2VEC2_FRAME_SECONDS = 0.02


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
        # Record the resolved revision as the version string.
        self.meta.version = getattr(self._model.config, "_name_or_path", self.repo_id)

    # --------------------------------------------------------------- inference
    def _log_probs(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        assert self._model is not None and self._processor is not None
        torch = self._torch
        inputs = self._processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
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
            normalized = normalize_en(text)
            display_units = words_en(text.upper()) or words_en(normalized)
        else:
            normalized = normalize_cjk(text)
            display_units = [c for c in normalized]

        # Optional reading conversion (e.g. kanji -> kana) for languages whose
        # acoustic model matches on readings rather than surface characters.
        if self.reading_fn is not None:
            reading_text, _ = self.reading_fn(normalized)
        else:
            reading_text = normalized

        tokens = self._tokenize(reading_text)
        if not tokens:
            return []

        logp = self._log_probs(audio, sample_rate)
        al = forced_align(logp, tokens, blank=self._blank_id())

        char_tokens = self._ids_to_chars(tokens)
        units = self._group_units(char_tokens, al.spans, display_units)
        return units

    # ---------------------------------------------------------------- helpers
    def _blank_id(self) -> int:
        # wav2vec2 uses <pad> as the CTC blank.
        return int(self._processor.tokenizer.pad_token_id or 0)

    def _tokenize(self, text: str) -> list[int]:
        enc = self._processor.tokenizer(text)
        return list(enc["input_ids"] if isinstance(enc, dict) else enc.input_ids)

    def _ids_to_chars(self, tokens: list[int]) -> list[str]:
        return [self._processor.tokenizer.convert_ids_to_tokens(t) for t in tokens]

    def _group_units(
        self,
        char_tokens: list[str],
        spans: list[TokenSpan | None],
        display_units: list[str],
    ) -> list[AlignUnit]:
        """Collapse character-level spans into word (en) or char (cjk) units."""
        fs = self.frame_seconds

        if self.unit_mode == "char":
            # Character / sub-word units. Used for languages whose CTC model
            # emits characters or sub-words (e.g. Japanese). We strip the
            # SentencePiece-style word-boundary marker ``▁`` (U+2581) so the
            # unit text is the original surface character(s).
            out: list[AlignUnit] = []
            specials = {"<pad>", "<s>", "</s>", "<unk>", "|", " "}
            for ch, span in zip(char_tokens, spans):
                if span is None or ch in specials:
                    continue
                clean = ch.replace("\u2581", "").strip()
                if not clean:
                    continue
                out.append(
                    AlignUnit(
                        text=clean,
                        start=span.start_frame * fs,
                        end=(span.end_frame + 1) * fs,
                        source="model",
                        score=span.score,
                        raw=clean,
                    )
                )
            return _attach_raw(out, display_units)

        # word mode: group on the wav2vec2 word-delimiter token "|".
        groups: list[list[TokenSpan]] = []
        current: list[TokenSpan] = []
        for ch, span in zip(char_tokens, spans):
            if ch in ("|", " "):
                if current:
                    groups.append(current)
                    current = []
                continue
            if ch in ("<pad>", "<s>", "</s>", "<unk>"):
                continue
            if span is not None:
                current.append(span)
        if current:
            groups.append(current)

        out = []
        for span_group in groups:
            if not span_group:
                continue
            start = span_group[0].start_frame * fs
            end = (span_group[-1].end_frame + 1) * fs
            score = float(np.mean([s.score for s in span_group]))
            out.append(AlignUnit(text="", start=start, end=end, source="model", score=score))
        return _attach_raw(out, display_units)


def _attach_raw(units: list[AlignUnit], display_units: list[str]) -> list[AlignUnit]:
    """Attach the display/original surface text to acoustic units.

    The acoustic model may match on a normalised reading; we still want to show
    the user the original word. When counts match trivially we map 1:1,
    otherwise we fall back to the normalised text already in ``unit.text``.
    """
    if len(display_units) == len(units):
        for u, d in zip(units, display_units):
            u.text = d
            u.raw = d
    return units
