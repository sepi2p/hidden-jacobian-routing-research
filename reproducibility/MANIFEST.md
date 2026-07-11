# Reproducibility Manifest

This release is locked to the submitted study and contains only its mapped experiments and lightweight audit artifacts.

## Promoted Pipelines

| Evidence block | Script | Frozen lightweight artifact |
|---|---|---|
| Exact CIFAR splits and registries | `create_exact_cifar_splits.py` | `artifacts/splits/` |
| Nested layer selection | `run_exact_nested_layer_selection.py` | external raw archive; table inputs tracked here |
| K&O clean-start comparator | `run_exact_ko_cleanstart_comparator.py` | K&O table inputs and `artifacts/analysis_summaries/ko_exact_*.csv` |
| Concentration and held-out separability | `analyze_flow_tube_dimensionality.py`; `analyze_flow_subspace_predictiveness.py` | concentration, layerwise, and separability table inputs |
| Generic optimization controls | `analyze_cifar_nonadversarial_optimization_controls.py` | non-adversarial-control table input |
| Objective-neutral mobility and selector | `analyze_cifar_objective_neutral_mobility_flow.py`; `test_mobility_margin_two_stage_selection.py` | mobility and selector table inputs |
| Attack-step selection diagnostic | `analyze_attack_road_selection_diagnostic.py` | proposal-summary table inputs |
| Local hidden-Jacobian mechanism | `test_mobility_vs_jacobian_gain.py`; `test_jacobian_basis_and_residual_transport.py`; `test_clean_whitened_mobility_jvp.py` | JVP and overlap table inputs |
| Finite-budget JVP/residual analysis | `run_finite_budget_jvp_residual.py`; `test_actual_trajectory_jvp_linearization.py` | finite-budget and recorded-step table inputs |
| Coordinate dependence | `run_function_preserving_coordinate_rescaling.py`; `analyze_decomposition_sensitivity.py` | coordinate-stress table inputs |
| Matched pullback activity | `run_matched_jacobian_intervention_controls.py` | intervention table inputs |
| Routing diagnostics | `run_same_harness_routing_efficiency.py`; `benchmark_multimodel_road_routing.py` | routing table inputs |
| ImageNet supporting pilot | `run_imagenet_supporting_pilot.py` | ImageNet pilot table input |
| RobustBench local-mobility pilot | `run_robustbench_local_mobility_pilot.py` | RobustBench pilot table input |

All scripts above are under `experiments/hidden_jacobian_routing/`. Exact paper-item mappings are in `reproducibility/configs/claim_evidence_map.csv`.

## Frozen Inputs and Checks

- `artifacts/table_inputs/`: every numeric table currently referenced by the manuscript, stored as lightweight CSV.
- `artifacts/analysis_summaries/`: all run-level and grouped values for the exact K&O clean-start comparator.
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
