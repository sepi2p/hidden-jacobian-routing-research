# Hidden-Jacobian Mobility in Adversarial Trajectories

> **Important:** the pullback-avoidance summaries in releases through `v1.2.2` contain classes 0--4 only because of a class-sorting/truncation bug. They do not validate the manuscript's ten-class functional claim. See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

This repository contains the research code and reproducibility scaffolding for a coordinate-level empirical audit of adversarial trajectories against hidden-Jacobian mobility, initial attack difficulty, finite-step residual motion, and budget-dependent pullback avoidance.

It intentionally does **not** include manuscript source, PDFs, generated paper figures, checkpoints, or dense raw trajectory arrays. The tracked split registries and table-ready summaries provide a lightweight audit layer; the mapped scripts regenerate the larger outputs.

## What This Repo Contains

- `experiments/hidden_jacobian_routing/`: experiment and analysis scripts for transport concentration, nested layer selection, difficulty controls, mobility/JVP tests, the norm-native comparator, coordinate stress tests, pullback avoidance, matched interventions, and supporting pilots.
- `attacks/`: the Square Attack probability schedule used by the paper scripts.
- `surro_models/`: CIFAR-10 model definitions for the evaluated BlackboxBench architectures and the ResNet18 seed study.
- `utils/`: a minimal CIFAR model loader for the evaluated models.
- `artifacts/table_inputs/`: the lightweight numeric table inputs used by the current manuscript.
- `artifacts/analysis_summaries/`: run-level and aggregated metrics for the exact clean-start comparator, including grouped out-of-fold increments, conditional image-bootstrap intervals, and the realized-JVP pilot.
- `artifacts/splits/`: exact CIFAR split, model, layer, and attack registries.
- `reproducibility/`: claim-to-evidence mapping, checkpoint hashes, release metadata, and deterministic checks.

## Core Scientific Claim

The repository supports a scoped empirical claim:

> Successful-trajectory PCA is a checkpoint-coordinate summary that is largely explained by realizable hidden-Jacobian motion. The stronger claim about functional reliance on fitted pullbacks is under revalidation and should be treated as provisional.

## Access Models

- Mechanism diagnostics: white-box model internals, hidden activations, gradients, and JVPs.
- Pullback-avoidance and matched interventions: white-box source-model features, JVP/VJP operations, and margins.

## Setup

```bash
conda env create -f environment.yml
conda activate hidden-jacobian-routing
```

If using an existing environment, install the standard scientific Python and PyTorch stack:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm pillow pyyaml torch torchvision robustbench timm
```

## Artifact Check

The tracked release can be audited without downloading raw trajectories:

```bash
bash reproducibility/scripts/check_required_artifacts.sh
make smoke
make tables
make verify-checksums
```

`make tables` regenerates table-ready tabular files under `generated/tables/`. The repository does not build or contain the manuscript.

## Exact Clean-Start Comparator

The promoted CIFAR protocol is resumable and uses the tracked 40/20/40 splits:

```bash
python experiments/hidden_jacobian_routing/create_exact_cifar_splits.py
for model in bbb_resnet50 bbb_vgg19_bn bbb_densenet bbb_inception_v3; do
  for seed in 1001 1002 1003; do
    python experiments/hidden_jacobian_routing/run_exact_nested_layer_selection.py \
      --model "$model" --split-seed "$seed" \
      --output-dir "analysis_outputs/hidden_jacobian_routing/exact_protocol/phase1_nested_layer_selection/$model/split_seed_$seed"
  done
done
python reproducibility/scripts/run_exact_ko_queue.py
python reproducibility/scripts/summarize_exact_ko.py \
  --input-root analysis_outputs/hidden_jacobian_routing/exact_protocol/phase1a_ko_cleanstart_comparator \
  --output-dir analysis_outputs/hidden_jacobian_routing/exact_protocol/ko_summary
python experiments/hidden_jacobian_routing/summarize_ko_grouped_cv_incremental.py \
  --input-root analysis_outputs/hidden_jacobian_routing/exact_protocol/phase1a_ko_cleanstart_comparator \
  --output-dir analysis_outputs/hidden_jacobian_routing/exact_protocol/ko_grouped_cv_incremental
python experiments/hidden_jacobian_routing/analyze_ko_proposal_sign_radius.py \
  --input-root analysis_outputs/hidden_jacobian_routing/exact_protocol/phase1a_ko_cleanstart_comparator \
  --output-dir analysis_outputs/hidden_jacobian_routing/exact_protocol/ko_proposal_sign_radius
```

The queue skips completed `DONE` shards, so interruption does not discard finished model/seed combinations. The grouped-CV script fits standardization and logistic regression inside each image-grouped training fold and predicts held-out image groups only. Its bootstrap intervals are conditional on the fitted OOF models and selected layer; they are not full-pipeline uncertainty intervals.

## Difficulty, Norm-Native, and Avoidance Tests

The current release adds the three experiments used to resolve the strongest alternative explanations:

```bash
python experiments/hidden_jacobian_routing/run_success_difficulty_control.py --help
python experiments/hidden_jacobian_routing/run_linf_induced_jacobian_comparator.py --help
python experiments/hidden_jacobian_routing/run_mechanism_breaking_attacks.py --help
```

The first measures the incremental out-of-fold contribution of transport energy after clean difficulty and first-step progress. The second compares signed Euclidean singular directions with an approximate induced \((\infty,2)\) maximizer. The third removes a fixed five-dimensional transport pullback during margin-PGD and includes a dimension-matched random-subspace control. Frozen aggregate outputs are under `artifacts/analysis_summaries/`.

## Large Files

The following are intentionally excluded from Git:

- `analysis_outputs/`
- `checkpoints/`
- `logs/`
- `*.npz`, `*.pt`, `*.pth`, `*.ckpt`
- generated figures/PDFs/images

Large outputs can be regenerated with the mapped scripts; the tracked summaries and checksums provide the lightweight audit layer used by the paper.

## Release Boundary

The repository contains the exact scripts, frozen lightweight table inputs, exact split registries, and model hashes needed to audit the submitted results. It excludes manuscript files, abandoned experiments, unrelated attacks, checkpoints, raw trajectories, and generated media.
