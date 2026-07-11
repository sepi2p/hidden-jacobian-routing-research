# Reproducing the Submitted Results

1. Create the environment with `conda env create -f environment.yml`.
2. Place CIFAR checkpoints at the paths in `reproducibility/configs/checkpoint_registry.csv` and verify their SHA256 hashes.
3. Run `make smoke` and `make verify-checksums`.
4. Run `make tables` to rebuild all table-ready LaTeX tabular files from the frozen CSV inputs.
5. Use `reproducibility/configs/claim_evidence_map.csv` to locate the producer for each paper figure or table.
6. Inspect `artifacts/analysis_summaries/ko_exact_run_metrics.csv` for all 105 layer-rule measurements underlying the exact clean-start comparator.
7. For a full rerun, regenerate the relevant block with the mapped producer script and registered checkpoints.

The Git repository intentionally excludes manuscript files, PDFs, checkpoints, raw trajectory arrays, and unrelated research scripts.
