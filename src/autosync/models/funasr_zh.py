"""Chinese adapter using FunASR ``fa-zh`` (MonotonicAligner).

Model: ``funasr/fa-zh`` (alias of ``iic/speech_timestamp_prediction-v1-16k-offline``).

Unlike wav2vec2, this model performs **native** text-audio forced alignment:
it takes the audio and the known text and returns a timestamp (ms) range for
every character token via a monotonic attention / CIF predictor. There is no
CTC Viterbi step here.

Licence: Apache-2.0 for the weights. The upstream model card
(``iic/speech_timestamp_prediction-v1-16k-offline`` on ModelScope) declares
``license: Apache License 2.0`` in its README front matter and carries no
restrictive-use clauses; the FunASR *toolkit* is MIT. Verified 2026-10-11 —
see ``docs/LICENSE_AUDIT.md``. The weights (~151 MB, ~38M parameters) are loaded
only when Chinese is requested; the ``funasr`` package is an optional
dependency.

Because this model computes timestamps directly we do not synthesise any
numbers: if the model fails we raise and the caller records a failure.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import MODELS_DIR
from ..lyrics.normalize import normalize_cjk
from ..types import AlignUnit
from .base import BaseAligner, ModelMeta

DEFAULT_MODEL_DIR = MODELS_DIR / "funasr" / "fa-zh"


class ChineseFunASRAligner(BaseAligner):
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.model_dir = DEFAULT_MODEL_DIR
        self.meta = ModelMeta(
            name="funasr/fa-zh",
            version="fa-zh (MonotonicAligner, speech_timestamp_prediction-v1-16k-offline)",
            language="zh",
            license="apache-2.0",
            approx_size_mb=151,
        )
        # Not frame-based; keep a nominal value for the interface.
        self.frame_seconds = 0.01
        self.load_seconds: float | None = None
        self.infer_seconds: float = 0.0
        self._model = None

    def load(self) -> None:
        from funasr import AutoModel

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"fa-zh weights not found at {self.model_dir}. "
                "Download model.pt/config.yaml/tokens.json/seg_dict/am.mvn first."
            )
        t0 = time.time()
        self._model = AutoModel(
            model=str(self.model_dir),
            device="cpu",
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
        )
        self.load_seconds = time.time() - t0

    def align_text(self, audio: np.ndarray, sample_rate: int, text: str) -> list[AlignUnit]:
        from funasr import AutoModel  # noqa: F401  (ensures dependency present)

        if self._model is None:
            self.load()
        normalized = normalize_cjk(text)
        if not normalized:
            # V1.3: never silently produce an empty timeline. A line whose text
            # has no CJK/kana after normalization (e.g. pure Latin backing
            # vocals) must fail loudly so the caller records an explicit
            # fallback instead of dropping the line from the coverage count.
            raise ValueError(
                "no CJK/kana characters after normalize_cjk; this Latin line must be "
                "routed to the English model or explicitly marked unsupported"
            )

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        res = self._model.generate(
            input=(audio, normalized),
            data_type=("sound", "text"),
            fs=sample_rate,
            batch_size=1,
            disable_pbar=True,
        )
        if not res:
            return []
        item = res[0]
        stamps = item.get("timestamp") or []
        out_text = item.get("text") or normalized

        # FunASR returns the post-processed text as space-separated character
        # tokens. Split on whitespace so tokens[i] corresponds to stamps[i].
        tokens = out_text.split() if " " in out_text else list(out_text)

        units: list[AlignUnit] = []
        n = min(len(stamps), len(tokens))
        for i in range(n):
            seg = stamps[i]
            if not seg or len(seg) < 2:
                continue
            start_ms, end_ms = float(seg[0]), float(seg[1])
            tok = tokens[i]
            units.append(
                AlignUnit(
                    text=tok,
                    start=start_ms / 1000.0,
                    end=end_ms / 1000.0,
                    source="model",
                    score=None,
                    raw=tok,
                )
            )
        return units
