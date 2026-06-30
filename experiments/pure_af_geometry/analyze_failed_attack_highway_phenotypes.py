#!/usr/bin/env python3
"""Classify failed attacks by how they interact with mobility highways."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def classify(row: pd.Series, success_refs: dict[str, dict[str, float]]) -> str:
    refs = success_refs.get(str(row.source), {})
    success_q25_highway = refs.get("q25_mean_highway", np.nan)
    success_q25_drop = refs.get("q25_margin_drop", np.nan)
    entered = bool(row.highway_q50_entered)
    early = bool(row.highway_q50_early20_dwell_frac > 0)
    dwell = float(row.highway_q50_dwell_frac)
    total_drop = float(row.margin_drop_total)
    mean_highway = float(row.mean_highway_energy)
    frac_bad = float(row.frac_class_support_or_neutral)

    if not entered:
        return "no_q50_highway_entry"
    if not early:
        return "late_highway_entry"
    if dwell < 0.25:
        return "brief_highway_contact"
    if frac_bad >= 0.25:
        return "wrong_or_neutral_highway_steps"
    if np.isfinite(success_q25_highway) and mean_highway < success_q25_highway:
        return "weak_highway_alignment"
    if np.isfinite(success_q25_drop) and total_drop < success_q25_drop:
        return "highway_but_insufficient_margin_progress"
    return "highway_progress_but_no_crossing"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--highway-dir",
        default="analysis_outputs/pure_af_geometry/jacobian_null_response/highway_entry_exit_bbb_resnet50_c200_auto",
    )
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/failed_attack_highway_phenotypes")
    p.add_argument("--sources", default="pgd,square")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--neutral-margin-drop", type=float, default=1e-6)
    args = p.parse_args()

    highway_dir = Path(args.highway_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [x.strip() for x in args.sources.split(",") if x.strip()]

    traj = pd.read_csv(highway_dir / "highway_trajectory_metrics.csv")
    step = pd.read_csv(highway_dir / "highway_step_metrics.csv")
    traj = traj[traj.source.isin(sources)].copy()
    step = step[step.source.isin(sources)].copy()
    if args.highway_source:
        traj = traj[traj.highway_source == args.highway_source].copy()
        step = step[step.highway_source == args.highway_source].copy()

    step_agg = (
        step.groupby(["source", "image_ord"], dropna=False)
        .agg(
            frac_boundary_leading=("margin_drop_step", lambda x: float((x > args.neutral_margin_drop).mean())),
            frac_class_supporting=("margin_drop_step", lambda x: float((x < -args.neutral_margin_drop).mean())),
            frac_neutral=("margin_drop_step", lambda x: float((np.abs(x) <= args.neutral_margin_drop).mean())),
            mean_step_margin_drop=("margin_drop_step", "mean"),
            min_step_margin_drop=("margin_drop_step", "min"),
            max_step_margin_drop=("margin_drop_step", "max"),
        )
        .reset_index()
    )
    merged = traj.merge(step_agg, on=["source", "image_ord"], how="left")
    merged["frac_class_support_or_neutral"] = merged["frac_class_supporting"].fillna(0) + merged["frac_neutral"].fillna(0)

    success_refs = {}
    for source, g in merged[merged.final_success == 1].groupby("source"):
        success_refs[str(source)] = {
            "q25_mean_highway": float(g.mean_highway_energy.quantile(0.25)),
            "q25_margin_drop": float(g.margin_drop_total.quantile(0.25)),
            "q25_feature_speed": float(g.mean_feature_speed.quantile(0.25)),
        }

    failed = merged[merged.final_success == 0].copy()
    failed["failure_phenotype"] = failed.apply(lambda r: classify(r, success_refs), axis=1)
    failed.to_csv(out_dir / "failed_highway_phenotypes_per_image.csv", index=False)

    summary = (
        failed.groupby(["source", "failure_phenotype"], dropna=False)
        .agg(
            n=("image_ord", "size"),
            mean_highway_energy=("mean_highway_energy", "mean"),
            mean_early20_highway=("early20_mean_highway_energy", "mean"),
            mean_dwell=("highway_q50_dwell_frac", "mean"),
            mean_margin_drop=("margin_drop_total", "mean"),
            mean_feature_speed=("mean_feature_speed", "mean"),
            mean_boundary_fraction=("frac_boundary_leading", "mean"),
            mean_bad_fraction=("frac_class_support_or_neutral", "mean"),
        )
        .reset_index()
    )
    total = failed.groupby("source")["image_ord"].size().rename("total").reset_index()
    summary = summary.merge(total, on="source", how="left")
    summary["fraction"] = summary["n"] / summary["total"]
    summary = summary.sort_values(["source", "fraction"], ascending=[True, False])
    summary.to_csv(out_dir / "failed_highway_phenotypes_summary.csv", index=False)

    compare = (
        merged[merged.source.isin(sources)]
        .groupby(["source", "final_success"])
        .agg(
            n=("image_ord", "size"),
            mean_highway_energy=("mean_highway_energy", "mean"),
            mean_early20_highway=("early20_mean_highway_energy", "mean"),
            q50_entry_rate=("highway_q50_entered", "mean"),
            q50_early_entry_rate=("highway_q50_early20_dwell_frac", lambda x: float((x > 0).mean())),
            mean_dwell=("highway_q50_dwell_frac", "mean"),
            mean_margin_drop=("margin_drop_total", "mean"),
            mean_feature_speed=("mean_feature_speed", "mean"),
            mean_boundary_fraction=("frac_boundary_leading", "mean"),
            mean_bad_fraction=("frac_class_support_or_neutral", "mean"),
        )
        .reset_index()
    )
    compare.to_csv(out_dir / "success_failed_highway_comparison.csv", index=False)

    lines = ["# Failed Attack Highway Phenotypes", "", "## Summary", ""]
    for r in summary.itertuples(index=False):
        lines.append(
            f"- `{r.source}` / `{r.failure_phenotype}`: {int(r.n)}/{int(r.total)} "
            f"({r.fraction:.3f}), highway={r.mean_highway_energy:.3f}, "
            f"dwell={r.mean_dwell:.3f}, margin_drop={r.mean_margin_drop:.3f}"
        )
    (out_dir / "failed_highway_phenotypes_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nComparison")
    print(compare.to_string(index=False))
    print(f"[SAVED] {out_dir}")


if __name__ == "__main__":
    main()
