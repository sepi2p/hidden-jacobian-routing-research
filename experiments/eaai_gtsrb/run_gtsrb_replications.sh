#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
BASE="$ROOT/analysis_outputs/eaai_gtsrb/replications"
LOG_BASE="$ROOT/logs/eaai_gtsrb/replications"

mkdir -p "$BASE" "$LOG_BASE"
cd "$ROOT"

run_replication() {
  local train_seed="$1"
  local split_seed="$2"
  local run_dir="$BASE/seed_${train_seed}"
  local log_dir="$LOG_BASE/seed_${train_seed}"
  mkdir -p "$run_dir" "$log_dir"

  if [[ ! -f "$run_dir/TRAINING_COMPLETE" ]]; then
    "$PYTHON" -u experiments/eaai_gtsrb/train_gtsrb_models.py \
      --data-dir data/gtsrb \
      --output-dir "$run_dir/checkpoints" \
      --models resnet18,convnext_tiny \
      --image-size 112 \
      --epochs 15 \
      --seed "$train_seed" \
      --split-seed "$split_seed" \
      2>&1 | tee -a "$log_dir/training.log"
    touch "$run_dir/TRAINING_COMPLETE"
  fi

  if [[ ! -f "$run_dir/AUDIT_COMPLETE" ]]; then
    "$PYTHON" -u experiments/eaai_gtsrb/run_gtsrb_compact_audit.py \
      --data-dir data/gtsrb \
      --checkpoint-dir "$run_dir/checkpoints" \
      --output-dir "$run_dir/diagnostics" \
      --models resnet18,convnext_tiny \
      --images 1000 \
      --jvp-images 200 \
      --seed "$train_seed" \
      --split-seed "$split_seed" \
      2>&1 | tee -a "$log_dir/audit.log"
    touch "$run_dir/AUDIT_COMPLETE"
  fi
}

run_replication 20260714 1308
run_replication 20260715 1309

touch "$BASE/REPLICATIONS_COMPLETE"
