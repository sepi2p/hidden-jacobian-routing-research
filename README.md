# Hidden-Jacobian Routing and Margin-Based Selection

This repository contains the research code and reproducibility scaffolding for the empirical study of adversarial search as hidden-Jacobian proposal geometry plus margin-based selection.

It intentionally does **not** include the manuscript source, PDFs, generated paper figures, or large raw artifacts. Large outputs should be released separately, for example through Zenodo, OSF, Hugging Face Datasets, or a GitHub Release asset.

## What This Repo Contains

- `experiments/pure_af_geometry/`: experiment and analysis scripts for transport trajectories, mobility/JVP controls, selector diagnostics, matched interventions, and query-refined transfer.
- `attacks/`, `data/`, `models/`, `utils/`, `configs/`, `trainners/`: supporting code used by the experiments.
- `reproducibility/`: manifests, environment file, artifact checks, and release notes.
- `docs/`: lightweight setup and reporting notes.
- `artifacts/`: placeholder directory for small manifests only. Large `.csv`, `.npz`, checkpoints, logs, images, and generated figures are ignored by default.

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
- `attack_prior_data/`
- `save/`
- `logs/`
- `*.npz`, `*.pt`, `*.pth`, `*.ckpt`
- generated figures/PDFs/images

Use `reproducibility/ARTIFACTS.md` to document external download URLs and hashes.

## Recommended Release Checklist

Before making the GitHub repo public:

1. Fill checkpoint hashes in `reproducibility/configs/experiment_registry.md`.
2. Fill exact layer hook names.
3. Add artifact download links and checksums to `reproducibility/ARTIFACTS.md`.
4. Verify `bash reproducibility/scripts/check_required_artifacts.sh` passes after downloading artifacts.
5. Keep all large outputs out of Git history.

