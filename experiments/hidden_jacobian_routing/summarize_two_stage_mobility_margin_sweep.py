#!/usr/bin/env python3
"""Summarize two-stage mobility/margin selector sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="analysis_outputs/hidden_jacobian_routing/jacobian_null_response",
    )
    p.add_argument(
        "--glob",
        default="two_stage_mobility_margin_selection_bbb_resnet50_c200_eps*_alpha*",
    )
    p.add_argument(
        "--output",
        default="analysis_outputs/hidden_jacobian_routing/jacobian_null_response/two_stage_mobility_margin_sweep_bbb_resnet50_summary.csv",
    )
    args = p.parse_args()

    rows = []
    topk_rows = []
    pred_rows = []
    for d in sorted(Path(args.root).glob(args.glob)):
        meta_path = d / "metadata.json"
        selector_path = d / "two_stage_selector_summary.csv"
        topk_path = d / "two_stage_topk_selector_summary.csv"
        pred_path = d / "two_stage_predictive_models.csv"
        if not selector_path.exists() or not pred_path.exists():
            continue
        meta = pd.read_json(meta_path, typ="series") if meta_path.exists() else pd.Series(dtype=object)
        tag = d.name.replace("two_stage_mobility_margin_selection_bbb_resnet50_c200_", "")
        selector = pd.read_csv(selector_path)
        for r in selector.itertuples(index=False):
            rows.append(
                {
                    "config": tag,
                    "probe_eps_over_255": float(meta.get("probe_eps_over_255", float("nan"))),
                    "attack_eps_over_255": float(meta.get("attack_eps_over_255", float("nan"))),
                    "images": int(meta.get("images_evaluated", 0)),
                    "directions_per_image": int(meta.get("directions_per_image", 0)),
                    "selector": r.selector,
                    "top_k": 1,
                    "asr": float(r.asr),
                    "mean_full_margin_drop": float(r.mean_full_margin_drop),
                    "mean_probe_margin_drop": float(r.mean_probe_margin_drop),
                    "mean_probe_mobility": float(r.mean_probe_mobility),
                }
            )
        if topk_path.exists():
            topk = pd.read_csv(topk_path)
            for r in topk.itertuples(index=False):
                topk_rows.append(
                    {
                        "config": tag,
                        "probe_eps_over_255": float(meta.get("probe_eps_over_255", float("nan"))),
                        "attack_eps_over_255": float(meta.get("attack_eps_over_255", float("nan"))),
                        "images": int(meta.get("images_evaluated", 0)),
                        "directions_per_image": int(meta.get("directions_per_image", 0)),
                        "selector": r.selector,
                        "top_k": int(r.top_k),
                        "topk_asr": float(r.topk_asr),
                        "topk_precision": float(r.topk_precision),
                        "mean_best_full_margin_drop": float(r.mean_best_full_margin_drop),
                    }
                )
        pred = pd.read_csv(pred_path)
        for r in pred.itertuples(index=False):
            pred_rows.append(
                {
                    "config": tag,
                    "probe_eps_over_255": float(meta.get("probe_eps_over_255", float("nan"))),
                    "attack_eps_over_255": float(meta.get("attack_eps_over_255", float("nan"))),
                    "model_name": r.model_name,
                    "mode": r.mode,
                    "test_auc": float(r.test_auc),
                    "test_auprc": float(r.test_auprc),
                }
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(topk_rows).to_csv(out.with_name(out.stem + "_topk.csv"), index=False)
    pd.DataFrame(pred_rows).to_csv(out.with_name(out.stem + "_predictive.csv"), index=False)

    if rows:
        print(pd.DataFrame(rows).pivot_table(index=["config"], columns="selector", values="asr", aggfunc="first").to_string())
    if topk_rows:
        sub = pd.DataFrame(topk_rows)
        keep = sub[(sub.top_k == 10) & (sub.selector.isin(["probe_margin_drop", "mobility_x_margin", "probe_mobility", "random_direction"]))]
        print("\nTop-10 ASR:")
        print(keep.pivot_table(index="config", columns="selector", values="topk_asr", aggfunc="first").to_string())
    if pred_rows:
        pr = pd.DataFrame(pred_rows)
        keep = pr[(pr["mode"] == "logistic") & (pr.model_name.isin(["probe_margin_only", "margin_plus_mobility", "all_features"]))]
        print("\nPredictive AUROC:")
        print(keep.pivot_table(index="config", columns="model_name", values="test_auc", aggfunc="first").to_string())
        print("\nPredictive AUPRC:")
        print(keep.pivot_table(index="config", columns="model_name", values="test_auprc", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
