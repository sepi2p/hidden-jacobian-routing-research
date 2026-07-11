#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

required=(
  artifacts/splits/cifar10_exact_splits.csv
  artifacts/splits/model_registry.csv
  artifacts/splits/layer_registry.csv
  artifacts/splits/attack_registry.csv
  artifacts/analysis_summaries/ko_exact_run_metrics.csv
  artifacts/analysis_summaries/ko_exact_summary.csv
  reproducibility/configs/checkpoint_registry.csv
  reproducibility/configs/claim_evidence_map.csv
)

for path in "${required[@]}"; do
  test -s "$path" || { echo "missing required release file: $path" >&2; exit 1; }
done

count=$(find artifacts/table_inputs -maxdepth 1 -name 'table_*.csv' -type f | wc -l)
test "$count" -eq 35 || { echo "expected 35 frozen paper tables, found $count" >&2; exit 1; }

echo "release layout OK: 35 table inputs, exact comparator summaries, splits, and protocol registries"
