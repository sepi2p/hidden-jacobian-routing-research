# Research Artifact Manifest

This manifest maps research outputs to the scripts and external artifacts needed to audit them. It deliberately avoids paper/PDF-specific files.

## Core Mechanism Artifacts

| Artifact family | Purpose | Main scripts | External outputs |
|---|---|---|---|
| Initial transport concentration | Measure concentration of successful hidden displacements | `experiments/pure_af_geometry/analyze_flow_tube_dimensionality.py`, `analyze_cifar_benchmark_optimizer_transport.py` | summary CSVs under `analysis_outputs/pure_af_geometry/` |
| Success/failure separability | Held-out projection-energy tests | `experiments/pure_af_geometry/analyze_flow_subspace_predictiveness.py`, `run_section4_fast_resnet_layer3_separability.py` | projection-energy metrics CSVs |
| Non-adversarial controls | Compare attack transport to generic representation optimization | `experiments/pure_af_geometry/analyze_cifar_nonadversarial_optimization_controls.py` | non-adversarial control summary CSVs |
| Objective-neutral mobility | Test whether label-free high-mobility directions overlap transport coordinates | `experiments/pure_af_geometry/analyze_cifar_objective_neutral_mobility_flow.py` | mobility summary CSVs |
| Mobility versus margin selection | Test proposal/selection decomposition | `experiments/pure_af_geometry/test_mobility_margin_two_stage_selection.py`, `summarize_two_stage_mobility_margin_sweep.py` | selector sweep CSVs |
| Hidden-Jacobian controls | Compare finite-difference mobility, exact JVP gain, and JVP-sketch bases | `experiments/pure_af_geometry/test_mobility_vs_jacobian_gain.py`, `test_jacobian_basis_and_residual_transport.py`, `test_clean_whitened_mobility_jvp.py` | JVP/mobility/residual summaries |
| Attack-step JVP linearization | Compare recorded attack hidden steps with local JVP predictions | `experiments/pure_af_geometry/test_actual_trajectory_jvp_linearization.py` | actual-step JVP summary CSVs |
| Matched pullback interventions | Compare transport PCs with JVP, failed-attack, residual, and matched-random bases | `experiments/pure_af_geometry/run_matched_jacobian_intervention_controls.py` | intervention summary CSVs |

## Machine-Readable Manifest

See `configs/research_artifact_manifest.csv` for a compact machine-readable mapping from claim/component to scripts and expected artifacts.

## Query-Refined Transfer Artifacts

| Artifact | Expected path |
|---|---|
| Benchmark summary | `analysis_outputs/pure_af_geometry/transport_hybrid_benchmark_a/posthoc_summary/transport_hybrid_benchmark_summary.md` |
| ASR curves | `analysis_outputs/pure_af_geometry/transport_hybrid_benchmark_a/posthoc_summary/asr_curve_table_by_target.csv` |
| Q=100 deltas | `analysis_outputs/pure_af_geometry/transport_hybrid_benchmark_a/posthoc_summary/delta_table_q100.csv` |
| Success overlap | `analysis_outputs/pure_af_geometry/transport_hybrid_benchmark_a/posthoc_summary/success_overlap_table_q100.csv` |
| Bootstrap intervals | `analysis_outputs/pure_af_geometry/transport_hybrid_benchmark_a/posthoc_summary/paired_bootstrap_ci_q100_q250.csv` |
| Proposal diagnostics | `analysis_outputs/pure_af_geometry/transport_hybrid_benchmark_a/posthoc_summary/proposal_acceptance_margin_diagnostics.csv` |
| Per-image outcomes | `analysis_outputs/pure_af_geometry/transport_hybrid_benchmark_a/posthoc_summary/combined_per_image_results.csv` |

The full `combined_query_curves.csv` can be large and may be distributed separately.

## Expensive Outputs Not Stored in Git

- raw trajectory segment vectors;
- model checkpoints;
- full per-query traces;
- dense JVP sketch matrices;
- raw image samples and generated visualizations.
