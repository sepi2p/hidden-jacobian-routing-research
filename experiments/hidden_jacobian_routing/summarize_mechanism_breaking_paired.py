#!/usr/bin/env python3
"""Paired image-level uncertainty for mechanism-breaking attacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "success",
    "final_margin",
    "hidden_mobility",
    "transport_projection_energy",
    "clean_state_jvp_gain",
    "transport_input_projection_ratio",
]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, reps: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    values = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = rng.integers(0, len(diff), size=len(diff))
        values[i] = float(diff[idx].mean())
    return float(diff.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    done = args.input_dir / "ANALYSIS_DONE.json"
    if done.exists():
        print(f"[SKIP] {done}")
        return
    best = pd.read_csv(args.input_dir / "mechanism_breaking_best_per_image.csv")
    baseline_name = "baseline_margin_pgd"
    baseline = best[best.method == baseline_name].set_index("dataset_idx")
    rows = []
    for method in sorted(set(best.method) - {baseline_name}):
        other = best[best.method == method].set_index("dataset_idx")
        common = baseline.index.intersection(other.index)
        a = other.loc[common]
        b = baseline.loc[common]
        record = {
            "method": method,
            "baseline": baseline_name,
            "n_images": len(common),
            "method_only_success": int(((a.success == 1) & (b.success == 0)).sum()),
            "baseline_only_success": int(((a.success == 0) & (b.success == 1)).sum()),
            "both_success": int(((a.success == 1) & (b.success == 1)).sum()),
            "neither_success": int(((a.success == 0) & (b.success == 0)).sum()),
        }
        for j, metric in enumerate(METRICS):
            mean, lo, hi = paired_bootstrap(
                a[metric].to_numpy(float), b[metric].to_numpy(float), args.bootstrap, args.seed + j * 101
            )
            record[f"delta_{metric}"] = mean
            record[f"delta_{metric}_ci_low"] = lo
            record[f"delta_{metric}_ci_high"] = hi
        rows.append(record)
    result = pd.DataFrame(rows)
    result.to_csv(args.input_dir / "mechanism_breaking_paired_comparisons.csv", index=False)
    done.write_text(
        json.dumps(
            {
                "status": "complete",
                "bootstrap": args.bootstrap,
                "unit": "image",
                "difference": "method_minus_baseline_margin_pgd",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
