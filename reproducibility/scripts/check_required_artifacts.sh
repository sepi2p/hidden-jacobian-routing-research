#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

required=(
  "analysis_outputs/hidden_jacobian_routing/transport_hybrid_benchmark_a/posthoc_summary/asr_curve_table_by_target.csv"
  "analysis_outputs/hidden_jacobian_routing/transport_hybrid_benchmark_a/posthoc_summary/delta_table_q100.csv"
  "analysis_outputs/hidden_jacobian_routing/transport_hybrid_benchmark_a/posthoc_summary/success_overlap_table_q100.csv"
  "analysis_outputs/hidden_jacobian_routing/transport_hybrid_benchmark_a/posthoc_summary/paired_bootstrap_ci_q100_q250.csv"
  "analysis_outputs/hidden_jacobian_routing/transport_hybrid_benchmark_a/posthoc_summary/proposal_acceptance_margin_diagnostics.csv"
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
  echo "One or more required artifacts are missing."
  exit 1
fi

echo "All required core artifacts are present."
