# Compute-Cost Reporting

Generated: 2026-06-14T15:30:42

This report separates measured target-query counts from local surrogate-compute estimates.

## Data Sources

- Target-query metrics are read from canonical CSV outputs.
- Wall-clock values are parsed from run log timestamps.
- Surrogate backward-pass counts are derived from explicit run configuration: `surpgd_num_restarts * checkpoint_or_default_steps`.
- The code did not instrument exact CUDA kernel time, exact forward-pass count, or per-image wall-clock time, so those are not claimed as measured.

## Compute-Cost Table

| experiment_group | policy | rows | asr_observed | target_query_mean_observed | target_query_median_observed | candidate_target_queries_mean_observed | candidate_count_mean_observed | surpgd_restarts_config | surpgd_steps_config | surrogate_backward_passes_per_image_estimated | wall_clock_seconds_observed_group | wall_clock_seconds_per_row_observed_group | timing_source | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_fresh_matrix | base | 12000 | 0.9992 | 123.95 | 9.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 297860.00 | 24.8217 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log;logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log | Main non-ablation CSV metrics; SurPGD default steps from attack.py parser default. |
| main_fresh_matrix | surpgd_pre_default40 | 12000 | 0.9995 | 85.2177 | 3.0000 | 5.1018 | 9.8323 | 32 | 40 | 1280 | 297860.00 | 24.8217 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log;logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log | Main non-ablation CSV metrics; SurPGD default steps from attack.py parser default. |
| main_fresh_matrix | surpgd_only_default40 | 12000 | 0.9997 | 109.96 | 2.0000 | 8.0488 | 19.9715 | 32 | 40 | 1280 | 297860.00 | 24.8217 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log;logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log | Main non-ablation CSV metrics; SurPGD default steps from attack.py parser default. |
| surpgd_checkpoint_ablation | surpgd_only_steps5 | 12000 | 0.9992 | 146.36 | 3.0000 | 9.1729 |  | 32 | 5 | 160 | 180739.00 | 2.5103 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log | Checkpoint ablation uses exact-step candidate pools from the completed checkpoint-ablation run. |
| surpgd_checkpoint_ablation | surpgd_only_steps10 | 12000 | 0.9994 | 119.80 | 1.0000 | 7.9612 |  | 32 | 10 | 320 | 180739.00 | 2.5103 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log | Checkpoint ablation uses exact-step candidate pools from the completed checkpoint-ablation run. |
| surpgd_checkpoint_ablation | surpgd_only_steps20 | 12000 | 0.9994 | 111.14 | 1.0000 | 7.6360 |  | 32 | 20 | 640 | 180739.00 | 2.5103 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log | Checkpoint ablation uses exact-step candidate pools from the completed checkpoint-ablation run. |
| surpgd_checkpoint_ablation | surpgd_pre_steps5 | 12000 | 0.9992 | 108.02 | 3.0000 | 1.9900 | 5.2606 | 32 | 5 | 160 | 180739.00 | 2.5103 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log | Checkpoint ablation uses exact-step candidate pools from the completed checkpoint-ablation run. |
| surpgd_checkpoint_ablation | surpgd_pre_steps10 | 12000 | 0.9995 | 87.0305 | 3.0000 | 3.6421 | 13.4897 | 32 | 10 | 320 | 180739.00 | 2.5103 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log | Checkpoint ablation uses exact-step candidate pools from the completed checkpoint-ablation run. |
| surpgd_checkpoint_ablation | surpgd_pre_steps20 | 12000 | 0.9998 | 78.7876 | 3.0000 | 3.9954 | 15.8316 | 32 | 20 | 640 | 180739.00 | 2.5103 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log | Checkpoint ablation uses exact-step candidate pools from the completed checkpoint-ablation run. |

## Timing Rows

| experiment_group | run_id | duration_seconds | source_log |
| --- | --- | --- | --- |
| main_fresh_matrix | cifar10_fulltest_bbb_vgg19_bn_to_bbb_densenet_square_unseen_common_seed0_k1000 | 872.00 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_vgg19_bn_to_bbb_resnet50_square_unseen_common_seed0_k1000 | 737.00 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_vgg19_bn_to_bbb_inception_v3_square_unseen_common_seed0_k1000 | 847.00 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_densenet_to_bbb_vgg19_bn_square_unseen_common_seed0_k1000 | 4885.00 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_densenet_to_bbb_resnet50_square_unseen_common_seed0_k1000 | 4260.00 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_densenet_to_bbb_inception_v3_square_unseen_common_seed0_k1000 | 4765.00 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_resnet50_to_bbb_vgg19_bn_square_unseen_common_seed0_k1000 | 1380.00 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_resnet50_to_bbb_densenet_square_unseen_common_seed0_k1000 | 1364.00 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_resnet50_to_bbb_inception_v3_square_unseen_common_seed0_k1000 | 1348.00 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_inception_v3_to_bbb_vgg19_bn_square_unseen_common_seed0_k1000 | 2728.00 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_inception_v3_to_bbb_densenet_square_unseen_common_seed0_k1000 | 2633.00 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_inception_v3_to_bbb_resnet50_square_unseen_common_seed0_k1000 | 2366.00 | logs/cifar10_unseen_common_seed0_k1000_checkpoint100.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_vgg19_bn_to_bbb_densenet_square_unseen_common_seed0_k1000 | 8276.00 | logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_vgg19_bn_to_bbb_resnet50_square_unseen_common_seed0_k1000 | 6866.00 | logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_vgg19_bn_to_bbb_inception_v3_square_unseen_common_seed0_k1000 | 8463.00 | logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_densenet_to_bbb_vgg19_bn_square_unseen_common_seed0_k1000 | 48180.00 | logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_densenet_to_bbb_resnet50_square_unseen_common_seed0_k1000 | 39425.00 | logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_densenet_to_bbb_inception_v3_square_unseen_common_seed0_k1000 | 44653.00 | logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_resnet50_to_bbb_vgg19_bn_square_unseen_common_seed0_k1000 | 13675.00 | logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_resnet50_to_bbb_densenet_square_unseen_common_seed0_k1000 | 13229.00 | logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_resnet50_to_bbb_inception_v3_square_unseen_common_seed0_k1000 | 13246.00 | logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_inception_v3_to_bbb_vgg19_bn_square_unseen_common_seed0_k1000 | 26914.00 | logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_inception_v3_to_bbb_densenet_square_unseen_common_seed0_k1000 | 24783.00 | logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log |
| main_fresh_matrix | cifar10_fulltest_bbb_inception_v3_to_bbb_resnet50_square_unseen_common_seed0_k1000 | 21965.00 | logs/cifar10_unseen_common_seed0_k1000_continuation.stdout.log |
| surpgd_checkpoint_ablation | cifar10_fulltest_bbb_vgg19_bn_to_bbb_densenet_square_unseen_common_seed0_surpgd_ckpt_ablation_k1000 | 7379.00 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log |
| surpgd_checkpoint_ablation | cifar10_fulltest_bbb_vgg19_bn_to_bbb_resnet50_square_unseen_common_seed0_surpgd_ckpt_ablation_k1000 | 4707.00 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log |
| surpgd_checkpoint_ablation | cifar10_fulltest_bbb_vgg19_bn_to_bbb_inception_v3_square_unseen_common_seed0_surpgd_ckpt_ablation_k1000 | 7257.00 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log |
| surpgd_checkpoint_ablation | cifar10_fulltest_bbb_densenet_to_bbb_vgg19_bn_square_unseen_common_seed0_surpgd_ckpt_ablation_k1000 | 29293.00 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log |
| surpgd_checkpoint_ablation | cifar10_fulltest_bbb_densenet_to_bbb_resnet50_square_unseen_common_seed0_surpgd_ckpt_ablation_k1000 | 23801.00 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log |
| surpgd_checkpoint_ablation | cifar10_fulltest_bbb_densenet_to_bbb_inception_v3_square_unseen_common_seed0_surpgd_ckpt_ablation_k1000 | 28911.00 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log |
| surpgd_checkpoint_ablation | cifar10_fulltest_bbb_resnet50_to_bbb_vgg19_bn_square_unseen_common_seed0_surpgd_ckpt_ablation_k1000 | 9349.00 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log |
| surpgd_checkpoint_ablation | cifar10_fulltest_bbb_resnet50_to_bbb_densenet_square_unseen_common_seed0_surpgd_ckpt_ablation_k1000 | 10917.00 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log |
| surpgd_checkpoint_ablation | cifar10_fulltest_bbb_resnet50_to_bbb_inception_v3_square_unseen_common_seed0_surpgd_ckpt_ablation_k1000 | 10453.00 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log |
| surpgd_checkpoint_ablation | cifar10_fulltest_bbb_inception_v3_to_bbb_vgg19_bn_square_unseen_common_seed0_surpgd_ckpt_ablation_k1000 | 17108.00 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log |
| surpgd_checkpoint_ablation | cifar10_fulltest_bbb_inception_v3_to_bbb_densenet_square_unseen_common_seed0_surpgd_ckpt_ablation_k1000 | 17803.00 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log |
| surpgd_checkpoint_ablation | cifar10_fulltest_bbb_inception_v3_to_bbb_resnet50_square_unseen_common_seed0_surpgd_ckpt_ablation_k1000 | 13761.00 | logs/cifar10_unseen_common_seed0_surpgd_ckpt_ablation_k1000.launcher.log |

## Reporting Guardrail

Use target queries as the primary black-box cost. Report surrogate PGD computation separately because it is local compute, not target-model access.
