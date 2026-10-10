#!/usr/bin/env bash
# Reproduce the V0.3 singing evaluation (lrc_only blind + oracle research).
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

ENG="heuristic,model,model+energy,model+onset,model+reseg"
MODES="lrc_only,oracle"

if [ ! -d experiments/singing/gtsinger ]; then
  echo "GTSinger subset missing; downloading (accepts CC BY-NC-SA 4.0)..."
  uv run python tools/fetch_gtsinger.py --out experiments/singing/gtsinger --max-per-group 3
fi

echo "== English =="
OMP_NUM_THREADS=4 uv run python benchmarks/singing_eval.py --languages en \
  --engines "$ENG" --modes "$MODES" --out experiments/reports/singing_en.json

echo "== Japanese =="
OMP_NUM_THREADS=4 uv run python benchmarks/singing_eval.py --languages ja \
  --engines "$ENG" --modes "$MODES" --out experiments/reports/singing_ja.json

echo "== Japanese singing models (pyshiro + HuBERT phoneme CTC) =="
uv sync --extra dev --extra shiro --extra funasr >/dev/null 2>&1 || true
OMP_NUM_THREADS=4 uv run --extra shiro python benchmarks/singing_eval.py --languages ja \
  --engines heuristic,model,pyshiro,pyshiro_v1,phoneme_ctc --modes "$MODES" \
  --out experiments/reports/singing_ja.json

echo "== Chinese =="
OMP_NUM_THREADS=4 uv run --extra funasr python benchmarks/singing_eval.py --languages zh \
  --engines "$ENG" --modes "$MODES" --out experiments/reports/singing_zh.json

echo "== Per-segment diagnosis (ja/en) =="
OMP_NUM_THREADS=4 uv run python benchmarks/diagnosis.py --language ja --out experiments/reports/diagnosis_ja.json
OMP_NUM_THREADS=4 uv run python benchmarks/diagnosis.py --language en --out experiments/reports/diagnosis_en.json

echo "== Accompaniment (controlled synthetic bed, per engine) =="
OMP_NUM_THREADS=4 uv run --extra funasr python benchmarks/accompaniment_eval.py \
  --language zh --out experiments/reports/accompaniment_zh.json
OMP_NUM_THREADS=4 uv run python benchmarks/accompaniment_eval.py \
  --language en --out experiments/reports/accompaniment_en.json

echo "== Alternative models (research overrides, NOT shipped defaults) =="
AUTOSYNC_JA_REPO=reazon-research/japanese-wav2vec2-large-rs35kh OMP_NUM_THREADS=4 uv run python \
  benchmarks/singing_eval.py --languages ja --engines model,model+energy \
  --out experiments/reports/singing_ja_large.json || true
AUTOSYNC_EN_REPO=facebook/wav2vec2-large-960h OMP_NUM_THREADS=4 uv run python \
  benchmarks/singing_eval.py --languages en --engines model,model+energy \
  --out experiments/reports/singing_en_large.json || true

echo "done -> experiments/reports/"
