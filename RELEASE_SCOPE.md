# Release Scope

This is a curated paper-only code release for the hidden-Jacobian routing study.

The repository includes:

- experiment scripts mapped in `reproducibility/MANIFEST.md`;
- helper scripts imported by those mapped experiments;
- the Square Attack probability schedule used by the query-search experiments;
- CIFAR-10 model definitions for the architectures evaluated in the paper;
- lightweight reproducibility manifests and environment files.

The repository intentionally excludes:

- manuscript LaTeX source and PDFs;
- generated figures and tables;
- model checkpoints;
- raw trajectory vectors and large CSV/NPZ outputs;
- scripts and model families outside this paper.

The final scientific framing is:

> Hidden-Jacobian mobility proposes feasible high-motion directions, and margin/gradient dynamics select adversarially useful directions under budget.

Large artifacts must be distributed outside Git, for example through a GitHub Release, Zenodo, OSF, or another archival service.
