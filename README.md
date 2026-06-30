# Hidden-Jacobian Routing and Margin-Based Selection

This repository contains the research code and reproducibility scaffolding for the empirical study of adversarial search as hidden-Jacobian proposal geometry plus margin-based selection.

It intentionally does **not** include the manuscript source, PDFs, generated paper figures, or large raw artifacts. Large outputs should be released separately, for example through Zenodo, OSF, Hugging Face Datasets, or a GitHub Release asset.

## What This Repo Contains

- `experiments/hidden_jacobian_routing/`: paper experiment and analysis scripts for transport concentration, mobility/JVP controls, selector diagnostics, matched interventions, trajectory-road probes, and query-refined transfer.
- `attacks/`: the Square Attack probability schedule used by the paper scripts.
- `surro_models/`: CIFAR-10 model definitions for the evaluated BlackboxBench architectures and the ResNet18 seed study.
- `utils/`: a minimal CIFAR model loader for the evaluated models.
- `reproducibility/`: manifests, environment file, and artifact checks.

## Core Scientific Claim

The repository supports a scoped empirical claim:

> Apparent hidden transport structure in successful adversarial trajectories is best explained as hidden-Jacobian high-mobility proposal geometry plus margin/gradient-based selection, not as a distinct adversarial-only flow.

The query-refined transfer scripts test a scoped application: surrogate CE and transport proposals can improve low-query target selection under score-query access. This is not a state-of-the-art black-box attack claim.

## Access Models

- Mechanism diagnostics: white-box model internals, hidden activations, gradients, and JVPs.
- Matched interventions: white-box source-model pullbacks.
- Query-refined transfer: white-box surrogate model plus score-query target access. Target queries return logits or equivalent class scores, not top-1 labels only.

## Setup

```bash
conda env create -f environment.yml
conda activate hidden-jacobian-routing
```

If using an existing environment, install the standard scientific Python and PyTorch stack:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm pillow pyyaml torch torchvision
```

## Artifact Check

After downloading external artifacts into their expected locations:

```bash
bash reproducibility/scripts/check_required_artifacts.sh
```

The checker validates only the core released research artifacts. It does not build the paper.

## Large Files

The following are intentionally excluded from Git:

- `analysis_outputs/`
- `checkpoints/`
- `logs/`
- `*.npz`, `*.pt`, `*.pth`, `*.ckpt`
- generated figures/PDFs/images

Distribute large artifacts outside Git, for example through a GitHub Release, Zenodo, OSF, or another archival service.

## Public Release Status

This public repository contains the source code, manifests, and lightweight documentation needed to inspect and rerun the research pipeline. Large generated artifacts and model checkpoints are intentionally distributed outside Git.

Current artifact policy:

1. Source code and reproducibility scaffolding are tracked here.
2. Large outputs, checkpoints, logs, and generated figures remain outside Git history.
3. External artifact bundles should include checksums and should satisfy `bash reproducibility/scripts/check_required_artifacts.sh` after download.
