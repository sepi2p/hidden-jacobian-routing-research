#!/usr/bin/env python3
"""Fast release smoke test that does not download data or checkpoints."""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    importlib.import_module("experiments.hidden_jacobian_routing.common")
    importlib.import_module("experiments.eaai_gtsrb.gtsrb_common")
    table_inputs = sorted((ROOT / "artifacts/table_inputs").glob("table_*.csv"))
    if not table_inputs:
        raise RuntimeError("frozen table inputs are missing")
    for path in table_inputs:
        with path.open(newline="", encoding="utf-8") as handle:
            if not next(csv.reader(handle), None):
                raise RuntimeError(f"empty table input: {path}")
    with (ROOT / "reproducibility/configs/claim_evidence_map.csv").open(newline="", encoding="utf-8") as handle:
        mapped = list(csv.DictReader(handle))
    mapped_inputs = {
        ROOT / row["frozen_input_or_output"]
        for row in mapped
        if row["frozen_input_or_output"].startswith("artifacts/table_inputs/table_")
    }
    missing_mappings = sorted(str(path) for path in table_inputs if path not in mapped_inputs)
    if missing_mappings:
        raise RuntimeError("unmapped frozen tables: " + ", ".join(missing_mappings))
    required = [
        ROOT / "artifacts/splits/cifar10_exact_splits.csv",
        ROOT / "artifacts/analysis_summaries/gtsrb_architecture_summary.csv",
        ROOT / "artifacts/analysis_summaries/gtsrb_seed_level_summary.csv",
        ROOT / "artifacts/analysis_summaries/gtsrb_coordinate_stress_all_seeds.csv",
        ROOT / "reproducibility/configs/checkpoint_registry.csv",
        ROOT / "reproducibility/configs/claim_evidence_map.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing release files: " + ", ".join(missing))
    print(f"smoke passed: {len(table_inputs)} frozen table inputs, all mapped")


if __name__ == "__main__":
    main()
