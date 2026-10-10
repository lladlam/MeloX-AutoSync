#!/usr/bin/env bash
# Reproduce the MeloX-AutoSync V0.1 experiments.
#
# Requirements: uv, ffmpeg, espeak-ng. Network access on first run to download
# model weights into models/ (gitignored). Chinese experiments need the funasr
# extra (`uv sync --extra funasr`).
#
# Runtime on a CPU-only machine: a few minutes (model download excluded).
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p experiments/audio experiments/output experiments/research

echo "=============================================================="
echo " Experiment A/B/E: English synthetic clip with word boundaries"
echo "=============================================================="
uv run --extra dev python tools/make_synthetic_sample.py \
  --text "hello world this is a test" \
  --out experiments/audio/synthetic_en.wav

uv run autosync align --audio experiments/audio/synthetic_en.wav \
  --lyrics experiments/audio/synthetic_en.lrc --language en \
  --engine heuristic --output experiments/output/synthetic_en.heuristic.json -q

OMP_NUM_THREADS=4 uv run autosync align --audio experiments/audio/synthetic_en.wav \
  --lyrics experiments/audio/synthetic_en.lrc --language en \
  --output experiments/output/synthetic_en.model.json -q

uv run python benchmarks/evaluate.py \
  --result experiments/output/synthetic_en.heuristic.json \
  --reference experiments/audio/synthetic_en.ref.json \
  --json-out experiments/output/synthetic_en.heuristic.eval.json

uv run python benchmarks/evaluate.py \
  --result experiments/output/synthetic_en.model.json \
  --reference experiments/audio/synthetic_en.ref.json \
  --json-out experiments/output/synthetic_en.model.eval.json

echo "=============================================================="
echo " Experiment B (real speech): LibriSpeech sample (English)"
echo "=============================================================="
echo "   (run manually once librispeech_sample1.flac is present)"
OMP_NUM_THREADS=4 uv run autosync align --audio experiments/audio/librispeech_sample1.flac \
  --lyrics experiments/audio/librispeech_sample1.lrc --language en \
  --output experiments/output/librispeech_sample1.model.json -q || true

echo "=============================================================="
echo " Experiment B/E: Chinese synthetic clip with char boundaries"
echo "=============================================================="
uv run --extra dev python tools/make_synthetic_sample.py \
  --text "巨龟是科隆群岛的食草动物" \
  --out experiments/audio/synthetic_zh.wav --voice cmn --split char

OMP_NUM_THREADS=4 uv run --extra funasr autosync align \
  --audio experiments/audio/synthetic_zh.wav \
  --lyrics experiments/audio/synthetic_zh.lrc --language zh \
  --output experiments/output/synthetic_zh.model.json -q

uv run python benchmarks/evaluate.py \
  --result experiments/output/synthetic_zh.model.json \
  --reference experiments/audio/synthetic_zh.ref.json \
  --json-out experiments/output/synthetic_zh.model.eval.json

echo "=============================================================="
echo " All experiments finished. See experiments/output/*.json"
echo "=============================================================="
