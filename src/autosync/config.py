"""Project paths and default configuration."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face weights are cached inside the project (gitignored).
HF_CACHE_DIR = MODELS_DIR / "hf"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

DEFAULT_SAMPLE_RATE = 16_000

# Margin (seconds) added before/after a lyric line when cropping audio, so the
# model sees natural onsets and trailing vowels.
CROP_PADDING_SECONDS = 0.30

# If two consecutive lyric lines are closer than this, we still crop each line
# independently but avoid double-counting padding beyond the previous line.
MIN_LINE_GAP_SECONDS = 0.05
