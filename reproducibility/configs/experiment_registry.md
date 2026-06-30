# Experiment Registry

This registry records the experiments that support the main paper. Fill in exact checkpoint hashes and wall-clock times before public release.

## Common Dataset and Model Conventions

- Dataset: CIFAR-10 test images unless otherwise stated.
- Main clean-correct filtering: use images correctly classified by all models involved in the comparison.
- Main perturbation norm: \(L_\infty\).
- Main source model for query-refined transfer: `bbb_resnet50`.
- Main target models for query-refined transfer: `bbb_vgg19_bn`, `bbb_densenet`, `bbb_inception_v3`.
- Target-output access in Section 9: class logits or equivalent class scores, not top-1 labels only.

## Model Registry

| Model id | Architecture | Checkpoint path/hash | Preprocessing | Eval mode |
|---|---|---|---|---|
| `bbb_resnet50` | ResNet50, CIFAR-10 | TODO | TODO | `model.eval()` |
| `bbb_vgg19_bn` | VGG19-BN, CIFAR-10 | TODO | TODO | `model.eval()` |
| `bbb_densenet` | DenseNet, CIFAR-10 | TODO | TODO | `model.eval()` |
| `bbb_inception_v3` | Inception-v3, CIFAR-10 | TODO | TODO | `model.eval()` |

## Layer Registry

| Model id | Hidden layer used in mechanism tests | Penultimate | Logits |
|---|---|---|---|
| `bbb_resnet50` | TODO exact module hook | TODO | classifier output |
| `bbb_vgg19_bn` | TODO exact module hook | TODO | classifier output |
| `bbb_densenet` | TODO exact module hook | TODO | classifier output |
| `bbb_inception_v3` | TODO exact module hook | TODO | classifier output |

## Main Experiment Families

| Family | Main script(s) | Output directory | Main manuscript use |
|---|---|---|---|
| Initial concentration/separability | `experiments/pure_af_geometry/analyze_*success*`, paper figure scripts | `analysis_outputs/pure_af_geometry/` | Sections 3--5 |
| Objective-neutral mobility | `experiments/pure_af_geometry/analyze_cifar_objective_neutral_mobility_flow.py` | `analysis_outputs/pure_af_geometry/` | Section 6 |
| Mobility versus margin selector | `experiments/pure_af_geometry/analyze_*mobility_margin*` | `analysis_outputs/pure_af_geometry/` | Section 6 |
| JVP mechanism controls | `experiments/pure_af_geometry/analyze_jacobian_residual_nulls.py` and related final scripts | `analysis_outputs/pure_af_geometry/` | Section 6 |
| Matched pullback interventions | matched intervention scripts under `experiments/pure_af_geometry/` | `analysis_outputs/pure_af_geometry/` | Section 7 |
| Cross-architecture / cross-seed validation | final multimodel and seed scripts | `analysis_outputs/pure_af_geometry/` | Section 8 |
| Query-refined transfer | `experiments/pure_af_geometry/evaluate_square_learned_correction_transfer.py`, `experiments/pure_af_geometry/summarize_transport_hybrid_benchmark.py` | `analysis_outputs/pure_af_geometry/transport_hybrid_benchmark_a/` | Section 9 |

## Release TODOs

- Fill checkpoint hashes.
- Fill exact layer hook names.
- Record GPU type and memory.
- Record exact wall-clock time for the final mechanism and query-refinement jobs.
- Include image-id lists for all train/test splits.
