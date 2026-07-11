#!/usr/bin/env python3
"""Paired image-bootstrap intervals for the L_inf-native comparator."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["candidate_coverage", "best_margin_drop", "unit_realized_jvp_gain", "max_fd_mobility"]


def bootstrap_delta(a: np.ndarray, b: np.ndarray, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    diff = np.asarray(a, float) - np.asarray(b, float)
    values = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, len(diff), size=len(diff))
        values[i] = diff[idx].mean()
    return float(diff.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("analysis_outputs/hidden_jacobian_routing/linf_induced_comparator"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    frames = []
    for path in sorted(args.input_root.glob("*/split_seed_1001/linf_comparator_per_image.csv")):
        frames.append(pd.read_csv(path))
    data = pd.concat(frames, ignore_index=True)
    rows = []
    for model, group in data.groupby("model"):
        baseline = group[group.method == "l2_singular_sign"].set_index("dataset_idx")
        for method in ["linf_induced", "ce_gradient_sign", "random_sign"]:
            other = group[group.method == method].set_index("dataset_idx")
            common = baseline.index.intersection(other.index)
            record = {"model": model, "method": method, "baseline": "l2_singular_sign", "n_images": len(common)}
            for j, metric in enumerate(METRICS):
                mean, lo, hi = bootstrap_delta(
                    other.loc[common, metric].to_numpy(float),
                    baseline.loc[common, metric].to_numpy(float),
                    args.bootstrap,
                    args.seed + j * 101,
                )
                record[f"delta_{metric}"] = mean
                record[f"delta_{metric}_ci_low"] = lo
                record[f"delta_{metric}_ci_high"] = hi
            rows.append(record)
    result = pd.DataFrame(rows)
    result.to_csv(args.input_root / "linf_comparator_paired_bootstrap.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
