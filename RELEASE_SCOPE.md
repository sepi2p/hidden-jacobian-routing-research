# Release Scope

This repository contains both canonical experiment scripts and exploratory analysis utilities that were useful during the research process.

For reproduction of the paper's main claims, use:

- `reproducibility/MANIFEST.md`
- `reproducibility/configs/research_artifact_manifest.csv`
- `reproducibility/configs/experiment_registry.md`

Scripts outside the manifest are included for transparency and auditability, but they should not be interpreted as promoted paper claims. In particular, older scripts may use historical names such as `flow`, `success_flow`, `away`, or `pure`; these names reflect development history, not the final scientific framing. The final interpretation is:

> hidden-Jacobian mobility proposes feasible high-motion directions, and margin/gradient dynamics select adversarially useful directions under budget.

The repository excludes:

- manuscript LaTeX source;
- PDF files;
- generated paper figures;
- model checkpoints;
- raw trajectory vectors;
- large CSV/NPZ artifacts;
- logs.

Large artifacts should be distributed separately and documented in `reproducibility/ARTIFACTS.md`.
