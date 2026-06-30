#!/usr/bin/env python3
"""Summarize class transitions along traced singular roads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ordered_unique(xs):
    out = []
    for x in xs:
        x = int(x)
        if not out or out[-1] != x:
            out.append(x)
    return out


def summarize_group(g: pd.DataFrame) -> dict:
    g = g.sort_values("step")
    pred0 = int(g.pred0.iloc[0])
    label = int(g.label.iloc[0])
    preds = g.pred.to_numpy(dtype=int)
    steps = g.step.to_numpy(dtype=int)
    changed = preds != pred0
    success = preds != label
    first_change = int(steps[np.argmax(changed)]) if changed.any() else -1
    first_success = int(steps[np.argmax(success)]) if success.any() else -1
    after_leave = g[steps > first_change] if first_change >= 0 else g.iloc[0:0]
    returned_original = int((after_leave.pred.to_numpy(dtype=int) == pred0).any()) if len(after_leave) else 0
    sequence = ordered_unique(preds)
    sigma0 = float(g.sigma1_est.iloc[0])
    sigma_end = float(g.sigma1_est.iloc[-1])
    sigma_min = float(g.sigma1_est.min())
    sigma_died_25 = int(sigma_min < 0.25 * sigma0)
    boundary_steps = int((g.linf_from_start >= 0.031 - 1e-4).sum())
    return {
        "image_id": int(g.image_id.iloc[0]),
        "direction": str(g.direction.iloc[0]),
        "label": label,
        "pred0": pred0,
        "final_pred": int(preds[-1]),
        "first_change_step": first_change,
        "first_success_step": first_success,
        "n_unique_classes": int(len(set(preds.tolist()))),
        "n_segments": int(len(sequence)),
        "class_sequence": "->".join(map(str, sequence)),
        "returned_to_original": returned_original,
        "sigma0": sigma0,
        "sigma_end": sigma_end,
        "sigma_min": sigma_min,
        "sigma_end_ratio": sigma_end / max(sigma0, 1e-12),
        "sigma_min_ratio": sigma_min / max(sigma0, 1e-12),
        "sigma_died_below_25pct": sigma_died_25,
        "final_margin": float(g.margin.iloc[-1]),
        "min_margin": float(g.margin.min()),
        "max_margin": float(g.margin.max()),
        "final_linf": float(g.linf_from_start.iloc[-1]),
        "boundary_fraction": float(boundary_steps / max(len(g), 1)),
        "final_hidden_distance": float(g.hidden_dist_from_start.iloc[-1]),
        "max_hidden_distance": float(g.hidden_dist_from_start.max()),
        "mean_cos_prev": float(g.cos_prev.dropna().mean()),
        "min_cos_prev": float(g.cos_prev.dropna().min()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    rows = [summarize_group(g) for _, g in df.groupby(["image_id", "direction"])]
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "singular_road_class_transition_summary.csv", index=False)
    aggregate = (
        summary.groupby("direction")
        .agg(
            n=("image_id", "count"),
            mean_first_success_step=("first_success_step", lambda x: np.mean([v for v in x if v >= 0]) if any(v >= 0 for v in x) else np.nan),
            success_rate=("first_success_step", lambda x: np.mean(np.asarray(x) >= 0)),
            mean_unique_classes=("n_unique_classes", "mean"),
            max_unique_classes=("n_unique_classes", "max"),
            return_rate=("returned_to_original", "mean"),
            sigma_death_rate=("sigma_died_below_25pct", "mean"),
            mean_sigma_end_ratio=("sigma_end_ratio", "mean"),
            mean_boundary_fraction=("boundary_fraction", "mean"),
            mean_final_hidden_distance=("final_hidden_distance", "mean"),
            mean_min_cos_prev=("min_cos_prev", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(out / "singular_road_class_transition_aggregate.csv", index=False)
    (out / "class_transition_examples.json").write_text(
        json.dumps(summary[["image_id", "direction", "class_sequence"]].head(40).to_dict(orient="records"), indent=2)
    )
    print("Aggregate:", flush=True)
    print(aggregate.to_string(index=False), flush=True)
    print("\nExamples:", flush=True)
    print(summary[["image_id", "direction", "class_sequence", "first_success_step", "returned_to_original"]].head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
