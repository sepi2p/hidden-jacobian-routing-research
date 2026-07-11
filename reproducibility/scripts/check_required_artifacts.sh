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
  artifacts/analysis_summaries/ko_grouped_cv_incremental_models.csv
  artifacts/analysis_summaries/ko_grouped_cv_incremental_deltas.csv
  artifacts/analysis_summaries/ko_grouped_cv_incremental_summary.csv
  artifacts/analysis_summaries/ko_grouped_cv_delta_image_bootstrap.csv
  artifacts/analysis_summaries/ko_realized_jvp_metrics_resnet50_n25.csv
  artifacts/analysis_summaries/ko_proposal_coverage_summary.csv
  artifacts/analysis_summaries/ko_sign_selection_summary.csv
  artifacts/analysis_summaries/ko_radius_selection_summary.csv
  artifacts/analysis_summaries/checkpoint_metrics.csv
  artifacts/analysis_summaries/difficulty_control_split_summaries.csv
  artifacts/analysis_summaries/linf_comparator_paired_bootstrap.csv
  artifacts/analysis_summaries/mechanism_breaking/mechanism_breaking_balanced_eps1__bbb_resnet50__paired.csv
  artifacts/analysis_summaries/mechanism_breaking/mechanism_breaking_balanced_eps1__bbb_vgg19_bn__paired.csv
  reproducibility/configs/checkpoint_registry.csv
  reproducibility/configs/claim_evidence_map.csv
)

for path in "${required[@]}"; do
  test -s "$path" || { echo "missing required release file: $path" >&2; exit 1; }
done

count=$(find artifacts/table_inputs -maxdepth 1 -name 'table_*.csv' -type f | wc -l)
test "$count" -eq 32 || { echo "expected 32 frozen paper tables, found $count" >&2; exit 1; }

echo "release layout OK: 32 active table inputs, exact splits, protocol registries, and current conditional/functional summaries"
