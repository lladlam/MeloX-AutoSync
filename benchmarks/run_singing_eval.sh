#!/usr/bin/env bash
# Reproduce the V0.2 singing evaluation on the curated GTSinger subset.
#
# Prerequisites:
#   * GTSinger subset downloaded locally (accepts its CC BY-NC-SA 4.0 licence):
#       uv run python tools/fetch_gtsinger.py --out experiments/singing/gtsinger
#   * Chinese needs the funasr extra: uv sync --extra funasr --extra dev
#
# Outputs aggregate reports to experiments/reports/ (committed). Raw audio and
# annotations stay under experiments/singing/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiments/reports

if [ ! -d experiments/singing/gtsinger ]; then
  echo "GTSinger subset missing; downloading (accepts CC BY-NC-SA 4.0)..."
  uv run python tools/fetch_gtsinger.py --out experiments/singing/gtsinger --max-per-group 3
fi

echo "== English =="
OMP_NUM_THREADS=4 uv run python benchmarks/singing_eval.py \
  --languages en --out experiments/reports/singing_en.json

echo "== Japanese =="
OMP_NUM_THREADS=4 uv run python benchmarks/singing_eval.py \
  --languages ja --out experiments/reports/singing_ja.json

echo "== Chinese =="
OMP_NUM_THREADS=4 uv run --extra funasr python benchmarks/singing_eval.py \
  --languages zh --out experiments/reports/singing_zh.json

echo "== Accompaniment (controlled synthetic bed) =="
OMP_NUM_THREADS=4 uv run --extra funasr python benchmarks/accompaniment_eval.py \
  --language zh --out experiments/reports/accompaniment_zh.json
OMP_NUM_THREADS=4 uv run python benchmarks/accompaniment_eval.py \
  --language en --out experiments/reports/accompaniment_en.json

echo "done -> experiments/reports/"
