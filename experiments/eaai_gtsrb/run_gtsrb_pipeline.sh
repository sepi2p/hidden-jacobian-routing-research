#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
OUTPUT="$ROOT/analysis_outputs/eaai_gtsrb"
LOGS="$ROOT/logs/eaai_gtsrb"

mkdir -p "$OUTPUT" "$LOGS"
cd "$ROOT"

"$PYTHON" -u experiments/eaai_gtsrb/train_gtsrb_models.py \
  --data-dir data/gtsrb \
  --output-dir "$OUTPUT/checkpoints" \
  --models resnet18,convnext_tiny \
  --image-size 112 \
  --epochs 15 \
  --download \
  2>&1 | tee -a "$LOGS/training.log"

touch "$OUTPUT/TRAINING_COMPLETE"
