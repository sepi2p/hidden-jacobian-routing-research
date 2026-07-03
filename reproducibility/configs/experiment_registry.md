# Experiment Registry

This registry records the experiments that support the main paper. Large checkpoints and raw outputs are external artifacts; this file records the model, layer, and experiment conventions expected by the public code release.

## Common Dataset and Model Conventions

- Dataset: CIFAR-10 test images unless otherwise stated.
- Main clean-correct filtering: use images correctly classified by all models involved in the comparison.
- Main perturbation norm: \(L_\infty\).
- White-box road-routing benchmark: `bbb_resnet50`, `bbb_vgg19_bn`, `bbb_densenet`, and `bbb_inception_v3`.
- Road-routing access model: white-box hidden activations, logits, gradients, and JVP/VJP operations.

## Model Registry

| Model id | Architecture | Checkpoint artifact | Preprocessing | Eval mode |
|---|---|---|---|---|
| `bbb_resnet50` | ResNet50, CIFAR-10 | External checkpoint bundle | BlackboxBench CIFAR-10 preprocessing | `model.eval()` |
| `bbb_vgg19_bn` | VGG19-BN, CIFAR-10 | External checkpoint bundle | BlackboxBench CIFAR-10 preprocessing | `model.eval()` |
| `bbb_densenet` | DenseNet, CIFAR-10 | External checkpoint bundle | BlackboxBench CIFAR-10 preprocessing | `model.eval()` |
| `bbb_inception_v3` | Inception-v3, CIFAR-10 | External checkpoint bundle | BlackboxBench CIFAR-10 preprocessing | `model.eval()` |

## Layer Registry

| Model id | Hidden layer used in mechanism tests | Penultimate | Logits |
|---|---|---|---|
| `bbb_resnet50` | Registered hidden hook in the artifact metadata | Registered penultimate hook in the artifact metadata | Classifier output |
| `bbb_vgg19_bn` | Registered hidden hook in the artifact metadata | Registered penultimate hook in the artifact metadata | Classifier output |
| `bbb_densenet` | Registered hidden hook in the artifact metadata | Registered penultimate hook in the artifact metadata | Classifier output |
| `bbb_inception_v3` | Registered hidden hook in the artifact metadata | Registered penultimate hook in the artifact metadata | Classifier output |

## Main Experiment Families

| Family | Main script(s) | Output directory | Main manuscript use |
|---|---|---|---|
| Initial concentration/separability | `experiments/hidden_jacobian_routing/analyze_flow_tube_dimensionality.py`, `experiments/hidden_jacobian_routing/analyze_flow_subspace_predictiveness.py` | `analysis_outputs/hidden_jacobian_routing/` | Sections 3--5 |
| Objective-neutral mobility | `experiments/hidden_jacobian_routing/analyze_cifar_objective_neutral_mobility_flow.py` | `analysis_outputs/hidden_jacobian_routing/` | Mechanism section |
| Mobility versus margin selector | `experiments/hidden_jacobian_routing/test_mobility_margin_two_stage_selection.py` | `analysis_outputs/hidden_jacobian_routing/` | Mechanism section |
| JVP mechanism controls | `experiments/hidden_jacobian_routing/test_mobility_vs_jacobian_gain.py`, `experiments/hidden_jacobian_routing/test_jacobian_basis_and_residual_transport.py`, `experiments/hidden_jacobian_routing/test_clean_whitened_mobility_jvp.py` | `analysis_outputs/hidden_jacobian_routing/` | Mechanism section |
| Matched pullback interventions | `experiments/hidden_jacobian_routing/run_matched_jacobian_intervention_controls.py` | `analysis_outputs/hidden_jacobian_routing/` | Intervention section |
| Road tracing diagnostics | `experiments/hidden_jacobian_routing/trace_jacobian_singular_roads.py` | `analysis_outputs/hidden_jacobian_routing/` | Mechanism/appendix |
| White-box road-routing attack | `experiments/hidden_jacobian_routing/benchmark_multimodel_road_routing.py`, `experiments/hidden_jacobian_routing/benchmark_whitebox_road_routing.py`, `experiments/hidden_jacobian_routing/evaluate_topk_margin_selected_singular_roads_on_balanced.py` | `analysis_outputs/hidden_jacobian_routing/` | Road-routing section |
| Road-map figures | `experiments/hidden_jacobian_routing/plot_hidden_jacobian_road_map.py`, `experiments/hidden_jacobian_routing/plot_margin_selected_singular_road_vs_pgd.py` | `analysis_outputs/hidden_jacobian_routing/` | Mechanism figures |

## External Artifact Metadata

Full artifact bundles should include checkpoint hashes, exact module hook names, GPU metadata, wall-clock times, and image-id lists for all train/test splits. These details are treated as artifact metadata rather than Git-tracked large outputs.
