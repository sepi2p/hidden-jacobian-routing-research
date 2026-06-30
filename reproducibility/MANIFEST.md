# Research Artifact Manifest

This manifest maps paper components to the scripts retained in this public code release. It deliberately avoids manuscript/PDF files and excludes scripts that are not part of the paper.

## Core Mechanism Scripts

| Component | Purpose | Main scripts | Expected external outputs |
|---|---|---|---|
| Initial transport concentration | Measure concentration of successful hidden displacements | `experiments/hidden_jacobian_routing/analyze_flow_tube_dimensionality.py` | dimensionality CSVs |
| Success/failure separability | Held-out projection-energy tests | `experiments/hidden_jacobian_routing/analyze_flow_subspace_predictiveness.py` | projection-energy metrics CSVs |
| Non-adversarial controls | Compare attack transport to generic representation optimization | `experiments/hidden_jacobian_routing/analyze_cifar_nonadversarial_optimization_controls.py` | non-adversarial control summary CSVs |
| Objective-neutral mobility | Test whether label-free high-mobility directions overlap transport coordinates | `experiments/hidden_jacobian_routing/analyze_cifar_objective_neutral_mobility_flow.py` | mobility summary CSVs |
| Mobility versus margin selection | Test proposal/selection decomposition | `experiments/hidden_jacobian_routing/test_mobility_margin_two_stage_selection.py`, `experiments/hidden_jacobian_routing/summarize_two_stage_mobility_margin_sweep.py` | selector sweep CSVs |
| Hidden-Jacobian controls | Compare finite-difference mobility, exact JVP gain, and JVP-sketch bases | `experiments/hidden_jacobian_routing/test_mobility_vs_jacobian_gain.py`, `experiments/hidden_jacobian_routing/test_jacobian_basis_and_residual_transport.py`, `experiments/hidden_jacobian_routing/test_clean_whitened_mobility_jvp.py` | JVP/mobility/residual summaries |
| Attack-step JVP linearization | Compare recorded attack hidden steps with local JVP predictions | `experiments/hidden_jacobian_routing/test_actual_trajectory_jvp_linearization.py` | actual-step JVP summary CSVs |
| Matched pullback interventions | Compare transport PCs with JVP, failed-attack, residual, and matched-random bases | `experiments/hidden_jacobian_routing/run_matched_jacobian_intervention_controls.py` | intervention summary CSVs |
| Sign/time optimizer comparison | Compare successful PGD/Square hidden trajectories under sign- and time-sensitive metrics | `experiments/hidden_jacobian_routing/analyze_sign_time_optimizer_similarity.py` | optimizer-signature summary CSVs |
| Road tracing diagnostics | Trace high-mobility hidden-Jacobian roads as integral curves | `experiments/hidden_jacobian_routing/trace_jacobian_singular_roads.py` | road tracing CSVs |
| Training dynamics / seed support | Support appendix checks on recurrence across independently trained ResNet18 checkpoints | `experiments/hidden_jacobian_routing/run_cifar_training_dynamics_transport.py` | checkpoint transport summaries |

## Query-Refined Transfer Scripts

| Component | Purpose | Main scripts | Expected external outputs |
|---|---|---|---|
| Surrogate-assisted query-refined transfer | Evaluate CE/transport proposal families under score-query access | `experiments/hidden_jacobian_routing/evaluate_square_learned_correction_transfer.py` | per-image and summary CSVs |
| Query-refined transfer post-processing | Build ASR curves, delta tables, overlap tables, and bootstrap summaries | `experiments/hidden_jacobian_routing/summarize_transport_hybrid_benchmark.py` | posthoc summary CSVs |

## Helper Modules

| Helper | Purpose |
|---|---|
| `experiments/hidden_jacobian_routing/common.py` | Shared layer hooks, margins, projections, model loading wrappers, and Square trajectory helper |
| `experiments/hidden_jacobian_routing/analyze_jacobian_null_response_pilot.py` | Balanced trajectory generation and PGD helper reused by JVP analyses |
| `experiments/hidden_jacobian_routing/evaluate_square_learned_correction_policy.py` | Source-side proposal-policy utilities imported by the query-refined transfer script |
| `attacks/square.py` | Square Attack probability schedule |

## Expensive Outputs Not Stored in Git

- model checkpoints;
- raw trajectory segment vectors;
- full per-query traces;
- dense JVP sketch matrices;
- raw image samples and generated visualizations.
