# Reporting Schema

This document explains the run artifacts and metrics produced by the current MCG + SurPGD experiment code. It is meant to make the CSV files paper-safe and prevent ambiguous interpretation later.

## Run Artifacts

Detached CIFAR-10 runs started with `scripts/run_cifar10_blackboxbench_detached.sh` write:

- `logs/<run_id>.log`: main experiment log, including final aggregate metrics.
- `logs/<run_id>.stdout.log`: stdout/stderr captured by `nohup`.
- `logs/<run_id>.pid`: process id for the detached run.
- `logs/<run_id>_valid_indices_seed<seed>_k<max_images>.txt`: dataset indices selected for the run.
- `logs/<run_id>_perimage_randomvalid.csv`: one row per image and variant.
- `logs/<run_id>_case_summary_randomvalid.csv`: one row per image in comparison mode.
- `logs/runs_summary_randomvalid.csv`: aggregate one-row-per-variant summary across runs.

The current comparison mode writes three variants for each image:

- `base`: MCG candidate initialization, then the selected black-box attack.
- `surpgd_pre`: MCG candidate initialization, SurPGD refinement on the first surrogate model, then the selected black-box attack.
- `surpgd_only`: SurPGD candidates from the clean image, then the selected black-box attack. This is a control to show whether MCG contributes beyond surrogate PGD alone.

## Aggregate Metrics

Aggregate metrics are written in `<run_id>.log` and `runs_summary_randomvalid.csv`.

- `valid_images`: number of target-clean-correct images actually attacked. For the current CIFAR-10 runner, this is collected from the repository validation subset by `attack.py::build_valid_buffer_random`.
- `ASR`: attack success rate over `valid_images`.
- `FASR`: fraction of images that succeed with `query_cnt == 1`. In this implementation, it means immediate first-query success for the variant, not necessarily the exact FASR definition from every paper table.
- `MeanQueries`: mean target-model query count for that variant.
- `MedianQueries`: median target-model query count for that variant.
- `linf_eps`: perturbation budget used by the attack. For the CIFAR-10 setup this is currently `8 / 255`.
- `max_query`: maximum target-model query budget.

Important query-count caveat:

- `query_cnt` counts target-model evaluations made by the implemented variant.
- Surrogate PGD gradient steps are not target-model queries.
- SurPGD candidate checks against the target model are counted.
- For `surpgd_pre`, the count includes the MCG candidate target check, SurPGD candidate target checks, a post-refinement target check, and then any queries used by the main black-box attack.

## Per-Image CSV

`<run_id>_perimage_randomvalid.csv` has one row per `(image, variant)` pair.

Identity and configuration columns:

- `run_id`: timestamped run identifier derived from the log filename.
- `buffer_idx`: index inside the valid-image buffer for this run.
- `dataset_idx`: original dataset index. Use this, not `buffer_idx`, for cross-run image matching.
- `original_label`: dataset label.
- `attack_label`: untargeted source label or targeted objective label, depending on `targeted`.
- `targeted`: `1` for targeted attack, `0` for untargeted attack.
- `target_label`: configured target label. For untargeted runs this is still present as a config value.
- `attack_method`: selected black-box attack, for example `square`.
- `variant`: `base`, `surpgd_pre`, or `surpgd_only`.
- `success`: whether this variant succeeded.
- `query_cnt`: target-model query count for this variant.
- `max_query`: target-model query budget.
- `linf_eps`: perturbation budget.

Model-free image descriptor columns:

- `image_mean`: mean pixel value in `[0, 1]`.
- `image_std`: pixel standard deviation.
- `image_entropy_16`: 16-bin grayscale entropy.
- `image_edge_energy`: Sobel-style edge energy.
- `image_high_freq_ratio`: high-frequency FFT energy ratio.

Clean target and surrogate columns:

- `clean_target_*`: target-model prediction/probability metrics on the clean image.
- `clean_surrogate_*`: first-surrogate prediction/probability metrics on the clean image.
- `*_pred`: predicted class.
- `*_pred_prob`: probability assigned to the predicted class.
- `*_label_prob`: probability assigned to the attacked label.
- `*_score`: `1 - label_prob`; larger is better for untargeted misclassification pressure.
- `*_prob_margin`: attacked-label probability minus highest non-attacked-label probability. Smaller is closer to misclassification.
- `clean_surrogate_target_agree`: whether clean target and clean surrogate predictions match.

MCG candidate columns:

- `mcg_target_*`: target-model metrics on the MCG-generated initial candidate.
- `mcg_surrogate_*`: first-surrogate metrics on the same MCG candidate.
- `mcg_target_margin`: raw target-side margin from the repository loss function.
- `mcg_target_success`: `1` if the MCG candidate is already adversarial for the target model.
- `mcg_surrogate_success`: `1` if the first surrogate misclassifies the MCG candidate.

SurPGD columns:

- `surpgd_pre_*`: diagnostics for MCG candidate plus SurPGD refinement.
- `surpgd_only_*`: diagnostics for SurPGD from the clean image without MCG initialization.
- `*_candidate_count`: number of SurPGD candidates returned for target checking.
- `*_surrogate_top1_score`: best surrogate-side candidate score.
- `*_surrogate_mean_score`: mean surrogate-side score across returned candidates.
- `*_best_target_score`: best target-side score among checked candidates.
- `*_best_target_margin`: target-side margin of the best checked candidate.
- `*_used_queries`: number of target-model checks used during SurPGD candidate selection.
- `*_early_success`: `1` if a checked candidate was already adversarial before the main attack.
- `*_target_*`: target-model metrics after the SurPGD stage.
- `*_target_success`: whether that post-SurPGD candidate is target-adversarial.

Some SurPGD fields are blank for variants where that stage does not apply.

## Case Summary CSV

`<run_id>_case_summary_randomvalid.csv` has one row per image in comparison mode. It combines final outcomes for all three variants and repeats diagnostic columns needed for image-level analysis.

Variant outcome columns:

- `base_success`, `base_query_cnt`: final result for MCG + selected attack.
- `surpgd_pre_success`, `surpgd_pre_query_cnt`: final result for MCG + SurPGD-pre + selected attack.
- `surpgd_only_success`, `surpgd_only_query_cnt`: final result for SurPGD-only + selected attack.
- `base_minus_pre_queries`: `base_query_cnt - surpgd_pre_query_cnt`; positive means SurPGD-pre used fewer target queries than base.
- `base_minus_only_queries`: `base_query_cnt - surpgd_only_query_cnt`; positive means SurPGD-only used fewer target queries than base.
- `pre_minus_only_queries`: `surpgd_pre_query_cnt - surpgd_only_query_cnt`; positive means SurPGD-only used fewer target queries than SurPGD-pre.

Current descriptive case tags:

- `mcg_immediate`: base succeeds with exactly one target query.
- `pgd_helped`: SurPGD-pre succeeds with fewer target queries than base, or rescues a base failure.
- `pgd_hurt`: base and SurPGD-pre both succeed, but SurPGD-pre uses more target queries.
- `pgd_rescue`: base fails and SurPGD-pre succeeds.
- `pgd_irrelevant`: base and SurPGD-pre both succeed with the same query count.
- `surpgd_only_fast`: SurPGD-only succeeds within `surpgd_pre_max_queries`.
- `mcg_needed`: SurPGD-pre beats SurPGD-only, or SurPGD-only fails.
- `universally_hard_candidate`: all three variants need more than 1,000 target queries.

These tags are descriptive diagnostics, not final paper categories. Thresholds such as 1,000 queries and `surpgd_pre_max_queries` should be reported explicitly if used in analysis.

## First Completed CIFAR-10 Run

The first completed controlled CIFAR-10 run used:

- target model: `bbb_densenet`
- surrogate model: `bbb_vgg19_bn`
- attack: `square`
- variants: `base`, `surpgd_pre`, `surpgd_only`
- valid images: 970
- files: `logs/cifar10_blackboxbench_bbb_densenet_square_compare_surpgd_20260531_032215_*`

Final aggregate metrics:

| variant | ASR | FASR | mean queries | median queries |
| --- | ---: | ---: | ---: | ---: |
| `base` | 1.0000 | 0.5454 | 96.61 | 1.0 |
| `surpgd_pre` | 1.0000 | 0.5454 | 33.57 | 1.0 |
| `surpgd_only` | 1.0000 | 0.6856 | 40.67 | 1.0 |

Case-tag counts from the same run:

| tag | count |
| --- | ---: |
| `mcg_immediate` | 529 |
| `pgd_helped` | 386 |
| `pgd_hurt` | 54 |
| `pgd_rescue` | 0 |
| `pgd_irrelevant` | 530 |
| `surpgd_only_fast` | 819 |
| `mcg_needed` | 204 |
| `universally_hard_candidate` | 1 |

Interpretation for planning, not yet a final claim: SurPGD-pre reduced mean target queries substantially compared with base, but SurPGD-only was also very strong in this model pair. This makes cross-model surrogate-target analysis essential before claiming that MCG is consistently necessary.

## Schema Safety

If the per-image or case-summary CSV header does not match the current code, `attack.py::_prepare_csv_path` writes a new schema-suffixed CSV instead of appending incompatible rows. Do not merge old and new schema files without checking headers.
