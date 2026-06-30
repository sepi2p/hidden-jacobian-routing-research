# Reproducibility Notes

This directory documents how to reproduce the research artifacts without storing the manuscript, PDFs, or generated paper figures in Git.

## What the Scripts Do

- `scripts/check_required_artifacts.sh` checks whether externally downloaded summary artifacts are present.
- `configs/experiment_registry.md` records model IDs, checkpoint hashes, layer hooks, and experiment families.
- `configs/research_artifact_manifest.csv` records the logical mapping from research outputs to summary artifacts. It is a manifest, not a paper build script.

## What Is Excluded

The Git repository intentionally excludes:

- manuscript source;
- paper PDFs;
- generated figure images;
- raw trajectory NPZ files;
- model checkpoints;
- large CSV logs;
- attack logs.

Distribute external artifacts separately with download URLs and checksums. The script `scripts/check_required_artifacts.sh` validates the expected file layout after download.

## Reproducibility Levels

**Level 1: Summary-artifact audit**

Download summary CSV/JSON/MD artifacts and run:

```bash
bash reproducibility/scripts/check_required_artifacts.sh
```

**Level 2: Experiment rerun**

Run the experiment scripts listed in `configs/experiment_registry.md`. Expensive runs include JVP sketches, matched interventions, trajectory extraction, and query-refined transfer.

**Level 3: Full raw rerun**

Requires public model checkpoints, image-id lists, exact layer hooks, and sufficient GPU time. This is expected to be substantially more expensive than Level 1.
