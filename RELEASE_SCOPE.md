# Release Scope

This is a curated paper-only code release for the hidden-Jacobian mobility study.

The repository includes:

- experiment scripts mapped in `reproducibility/MANIFEST.md`;
- helper scripts imported by those mapped experiments;
- the Square Attack probability schedule used by the query-search experiments;
- CIFAR-10 model definitions for the architectures evaluated in the paper;
- lightweight reproducibility manifests, exact aggregate summaries, and environment files.

The repository intentionally excludes:

- manuscript LaTeX source and PDFs;
- generated figures and tables;
- model checkpoints;
- raw trajectory vectors and large CSV/NPZ outputs;
- scripts and model families outside this paper.

The final scientific framing is:

> Successful-trajectory PCA summarizes realizable hidden-Jacobian mobility in checkpoint coordinates; the fitted pullbacks are important under the tested tight budget but bypassable at a larger budget.

The tracked artifact bundle contains only lightweight table-ready summaries, exact clean-start run summaries, split registries, and checksums. Raw trajectories, checkpoints, and dense arrays are regenerated with the mapped scripts and are not committed to Git.
