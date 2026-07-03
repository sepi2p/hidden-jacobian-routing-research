#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

required=(
  "analysis_outputs/hidden_jacobian_routing/multimodel_road_routing_cifar_c200/summary.csv"
  "analysis_outputs/hidden_jacobian_routing/matched_jacobian_intervention_controls/summary.csv"
  "analysis_outputs/hidden_jacobian_routing/two_stage_mobility_margin_sweep/selector_summary.csv"
  "analysis_outputs/hidden_jacobian_routing/jvp_mobility_multimodel/summary.csv"
  "analysis_outputs/hidden_jacobian_routing/actual_trajectory_jvp_linearity/summary.csv"
)

missing=0
for path in "${required[@]}"; do
  if [[ -e "$path" ]]; then
    printf "OK      %s\n" "$path"
  else
    printf "MISSING %s\n" "$path"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "One or more core summary artifacts are missing."
  echo "Download the external artifact bundle or rerun the mapped scripts in reproducibility/MANIFEST.md."
  exit 1
fi

echo "All required core summary artifacts are present."
