#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
OUTPUT="$ROOT/analysis_outputs/eaai_gtsrb"
LOGS="$ROOT/logs/eaai_gtsrb"

mkdir -p "$LOGS"
while [[ ! -f "$OUTPUT/TRAINING_COMPLETE" ]]; do
  sleep 300
done

cd "$ROOT"
"$PYTHON" -u experiments/eaai_gtsrb/run_gtsrb_compact_audit.py \
  --data-dir data/gtsrb \
  --checkpoint-dir "$OUTPUT/checkpoints" \
  --output-dir "$OUTPUT/diagnostics" \
  --models resnet18,convnext_tiny \
  --images 1000 \
  --jvp-images 200 \
  --strong-scope \
  2>&1 | tee -a "$LOGS/audit.log"

touch "$OUTPUT/AUDIT_COMPLETE"
