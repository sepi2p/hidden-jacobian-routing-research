# Reproducibility Manifest

This release is locked to the submitted study and contains only its mapped experiments and lightweight audit artifacts.

## Promoted Pipelines

| Evidence block | Script | Frozen lightweight artifact |
|---|---|---|
| Exact CIFAR splits and registries | `create_exact_cifar_splits.py` | `artifacts/splits/` |
| Nested layer selection | `run_exact_nested_layer_selection.py` | external raw archive; table inputs tracked here |
| Initial-difficulty control | `run_success_difficulty_control.py` | split summaries and difficulty table input |
| K&O clean-start comparator | `run_exact_ko_cleanstart_comparator.py`; `summarize_ko_grouped_cv_incremental.py` | K&O table inputs, grouped OOF increments, conditional image-bootstrap intervals, and `artifacts/analysis_summaries/ko_exact_*.csv` |
| Realized candidate JVP pilot | `run_ko_realized_jvp_gain_pilot.py` | `artifacts/analysis_summaries/ko_realized_jvp_*` |
| Proposal/sign/radius decomposition | `analyze_ko_proposal_sign_radius.py` | `artifacts/analysis_summaries/ko_*_selection_summary.csv` and table input |
| Concentration and held-out separability | `analyze_flow_tube_dimensionality.py`; `analyze_flow_subspace_predictiveness.py` | concentration, layerwise, and separability table inputs |
| Generic optimization controls | `analyze_cifar_nonadversarial_optimization_controls.py` | non-adversarial-control table input |
| Objective-neutral mobility and selector | `analyze_cifar_objective_neutral_mobility_flow.py`; `test_mobility_margin_two_stage_selection.py` | mobility and selector table inputs |
| Attack-step selection diagnostic | `analyze_attack_road_selection_diagnostic.py` | proposal-summary table inputs |
| Local hidden-Jacobian mechanism | `test_mobility_vs_jacobian_gain.py`; `test_jacobian_basis_and_residual_transport.py`; `test_clean_whitened_mobility_jvp.py` | JVP and overlap table inputs |
| Norm-native comparator | `run_linf_induced_jacobian_comparator.py`; `summarize_linf_comparator_paired.py` | paired comparator summary and table input |
| Finite-budget JVP/residual analysis | `run_finite_budget_jvp_residual.py`; `test_actual_trajectory_jvp_linearization.py` | finite-budget and recorded-step table inputs |
| Coordinate dependence | `run_function_preserving_coordinate_rescaling.py`; `analyze_decomposition_sensitivity.py` | coordinate-stress table inputs |
| ImageNet supporting pilot | `run_imagenet_supporting_pilot.py` | ImageNet pilot table input |
| RobustBench local-mobility pilot | `run_robustbench_local_mobility_pilot.py` | RobustBench pilot table input |
| GTSRB application validation | `train_gtsrb_models.py`; `run_gtsrb_compact_audit.py`; `aggregate_gtsrb_replications.py` | `artifacts/analysis_summaries/gtsrb_*.csv` |

The CIFAR/ImageNet/RobustBench scripts are under `experiments/hidden_jacobian_routing/`; the traffic-sign application is under `experiments/eaai_gtsrb/`. Exact paper-item mappings are in `reproducibility/configs/claim_evidence_map.csv`.

## Frozen Inputs and Checks

- `artifacts/table_inputs/`: every numeric table currently referenced by the manuscript, stored as lightweight CSV.
- `artifacts/analysis_summaries/`: run-level and grouped values for the exact K&O comparator, difficulty control, norm-native comparator, checkpoint metrics, and associated conditional bootstrap analyses.
- `artifacts/splits/`: exact CIFAR split and protocol registries.
- `reproducibility/configs/checkpoint_registry.csv`: checkpoint paths and SHA256 hashes.
- `reproducibility/SHA256SUMS`: checksums for all tracked release artifacts and registries.

Run:

```bash
make smoke
make tables
make verify-checksums
```

## External Raw Archive

Raw trajectories, dense arrays, checkpoints, and per-query logs are intentionally outside Git. The frozen table inputs, exact splits, checkpoint hashes, and mapped scripts form the tracked audit layer.
