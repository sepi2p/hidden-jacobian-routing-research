#!/usr/bin/env python3
"""Aggregate completed exact K&O comparator shards into frozen summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCORES = {
    "singular_value": "singular_auprc",
    "jvp_gain": "jvp_auprc",
    "margin_drop": "margin_auprc",
    "transport_projection_energy": "transport_auprc",
}


def load_run(run_dir: Path) -> list[dict]:
    metric = pd.read_csv(run_dir / "ko_candidate_metric_auroc_auprc.csv")
    incremental = pd.read_csv(run_dir / "ko_incremental_models.csv")
    rows = []
    for keys, group in metric.groupby(["model", "split_seed", "candidate_seed", "layer_rule", "layer"]):
        model, split_seed, candidate_seed, layer_rule, layer = keys
        row = {
            "model": model,
            "split_seed": int(split_seed),
            "candidate_seed": int(candidate_seed),
            "layer_rule": layer_rule,
            "layer": layer,
        }
        by_score = group.set_index("score")
        for score, column in SCORES.items():
            row[column] = float(by_score.loc[score, "auprc"])
        inc = incremental[
            (incremental.model == model)
            & (incremental.split_seed == split_seed)
            & (incremental.candidate_seed == candidate_seed)
            & (incremental.layer_rule == layer_rule)
            & (incremental.layer == layer)
        ].set_index("nested_model")
        row["m3_auprc"] = float(inc.loc["M3_gradient", "auprc"])
        row["m4_auprc"] = float(inc.loc["M4_transport", "auprc"])
        row["delta_transport_auprc"] = row["m4_auprc"] - row["m3_auprc"]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_rows = []
    for done in sorted(args.input_root.glob("*/split_seed_*/candidate_seed_*/DONE")):
        run_rows.extend(load_run(done.parent))
    runs = pd.DataFrame(run_rows)
    # ResNet50's selected avgpool and pre-logit representation are the same
    # registered tensor, so it contributes one layer rule; the other models
    # contribute selected and pre-logit rules.
    expected = 15 * (1 + 2 + 2 + 2)
    if len(runs) != expected:
        raise SystemExit(f"expected {expected} layer-rule rows from 60 runs; found {len(runs)}")
    group_cols = ["model", "layer_rule"]
    value_cols = list(SCORES.values()) + ["m3_auprc", "m4_auprc", "delta_transport_auprc"]
    summary = runs.groupby(group_cols)[value_cols].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = ["_".join(x).rstrip("_") if isinstance(x, tuple) else x for x in summary.columns]
    counts = runs.groupby(group_cols).size().rename("n_runs").reset_index()
    layers = runs.groupby(group_cols)["layer"].agg(lambda values: "/".join(sorted(set(values)))).rename("layers").reset_index()
    summary = counts.merge(layers, on=group_cols).merge(summary, on=group_cols)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "ko_exact_run_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "ko_exact_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
