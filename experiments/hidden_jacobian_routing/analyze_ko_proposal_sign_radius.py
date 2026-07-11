#!/usr/bin/env python3
"""Separate proposal coverage, sign selection, and radius selection.

This analysis uses the frozen clean-start K&O candidate pools. It avoids the
misleading comparison between sign-symmetric singular values and sign-aware
candidate margins by asking three identified questions separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def iter_nested_files(root: Path):
    yield from sorted(root.glob("bbb_*/*/candidate_seed_*/ko_candidate_scores_nested_selected_nonlogit_*.csv"))


def proposal_coverage(df: pd.DataFrame, top_ks: list[int]) -> pd.DataFrame:
    rows = []
    for k in top_ks:
        selected = df[df["rank"] <= k]
        per_image = selected.groupby("dataset_idx")["candidate_success"].max()
        rows.append(
            {
                "top_k": k,
                "n_images": int(per_image.size),
                "proposal_coverage_asr": float(per_image.mean()),
            }
        )
    return pd.DataFrame(rows)


def sign_selection(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_idx", "rank", "alpha_255"]
    valid = df.groupby(keys)["sign"].nunique().rename("n_signs").reset_index()
    work = df.merge(valid[valid.n_signs == 2][keys], on=keys, how="inner")
    oracle_idx = work.groupby(keys, sort=False)["margin_drop"].idxmax()
    oracle = work.loc[oracle_idx, keys + ["sign", "margin_drop", "candidate_success"]].rename(
        columns={
            "sign": "oracle_sign",
            "margin_drop": "oracle_margin_drop",
            "candidate_success": "oracle_success",
        }
    )
    outputs = []
    for selector, score in (("ce_gradient", "ce_grad_cos"), ("dlr_gradient", "dlr_grad_cos")):
        selected_idx = work.groupby(keys, sort=False)[score].idxmax()
        selected = work.loc[selected_idx, keys + ["sign", "margin_drop", "candidate_success"]].rename(
            columns={
                "sign": "selected_sign",
                "margin_drop": "selected_margin_drop",
                "candidate_success": "selected_success",
            }
        )
        result = selected.merge(oracle, on=keys, how="inner")
        result["selector"] = selector
        result["sign_match"] = (result.selected_sign == result.oracle_sign).astype(int)
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def radius_selection(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_idx", "rank"]
    oracle_idx = df.groupby(keys, sort=False)["margin_drop"].idxmax()
    oracle = df.loc[oracle_idx, keys + ["sign", "alpha_255", "margin_drop", "candidate_success"]].rename(
        columns={
            "sign": "oracle_sign",
            "alpha_255": "oracle_alpha_255",
            "margin_drop": "oracle_margin_drop",
            "candidate_success": "oracle_success",
        }
    )
    max_alpha = df.groupby(keys)["alpha_255"].transform("max")
    max_radius = df[np.isclose(df.alpha_255, max_alpha)].copy()
    max_idx = max_radius.groupby(keys, sort=False)["ce_grad_cos"].idxmax()
    max_pick = max_radius.loc[max_idx, keys + ["margin_drop", "candidate_success"]].rename(
        columns={"margin_drop": "max_radius_margin_drop", "candidate_success": "max_radius_success"}
    )
    outputs = []
    for selector, score in (("ce_linear", "ce_grad_cos"), ("dlr_linear", "dlr_grad_cos")):
        work = df.copy()
        work["local_score"] = work[score] * work["alpha_255"]
        selected_idx = work.groupby(keys, sort=False)["local_score"].idxmax()
        selected = work.loc[
            selected_idx, keys + ["sign", "alpha_255", "margin_drop", "candidate_success"]
        ].rename(
            columns={
                "sign": "selected_sign",
                "alpha_255": "selected_alpha_255",
                "margin_drop": "selected_margin_drop",
                "candidate_success": "selected_success",
            }
        )
        result = selected.merge(oracle, on=keys, how="inner").merge(max_pick, on=keys, how="inner")
        result["selector"] = selector
        result["exact_candidate_match"] = (
            (result.selected_sign == result.oracle_sign)
            & np.isclose(result.selected_alpha_255, result.oracle_alpha_255)
        ).astype(int)
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def summarize_selection(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col)
        .agg(
            n=("dataset_idx", "size"),
            image_count=("dataset_idx", "nunique"),
            match_rate=("sign_match" if "sign_match" in df else "exact_candidate_match", "mean"),
            selected_margin_drop=("selected_margin_drop", "mean"),
            oracle_margin_drop=("oracle_margin_drop", "mean"),
            selected_success=("selected_success", "mean"),
            oracle_success=("oracle_success", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="analysis_outputs/pure_af_geometry/q1_reviewer_validation/exact_protocol/phase1a_ko_cleanstart_comparator",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/pure_af_geometry/q1_reviewer_validation/exact_protocol/ko_proposal_sign_radius",
    )
    parser.add_argument("--top-ks", default="1,5,10,20")
    args = parser.parse_args()

    root = Path(args.input_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    top_ks = [int(value) for value in args.top_ks.split(",")]
    coverage_rows = []
    sign_rows = []
    radius_rows = []
    for path in iter_nested_files(root):
        df = pd.read_csv(path)
        run = {
            "model": str(df.model.iloc[0]),
            "split_seed": int(df.split_seed.iloc[0]),
            "candidate_seed": int(df.candidate_seed.iloc[0]),
            "layer": str(df.layer.iloc[0]),
        }
        coverage_rows.append(proposal_coverage(df, top_ks).assign(**run))
        sign_rows.append(sign_selection(df).assign(**run))
        radius_rows.append(radius_selection(df).assign(**run))

    coverage = pd.concat(coverage_rows, ignore_index=True)
    signs = pd.concat(sign_rows, ignore_index=True)
    radii = pd.concat(radius_rows, ignore_index=True)
    coverage.to_csv(out / "proposal_coverage_runs.csv", index=False)
    signs.to_csv(out / "sign_selection_rows.csv", index=False)
    radii.to_csv(out / "radius_selection_rows.csv", index=False)

    coverage_summary = (
        coverage.groupby(["model", "top_k"])
        .agg(
            n_runs=("proposal_coverage_asr", "size"),
            mean_coverage=("proposal_coverage_asr", "mean"),
            min_coverage=("proposal_coverage_asr", "min"),
            max_coverage=("proposal_coverage_asr", "max"),
        )
        .reset_index()
    )
    sign_summary = (
        signs.groupby(["model", "selector"])
        .agg(
            n=("dataset_idx", "size"),
            image_count=("dataset_idx", "nunique"),
            sign_match=("sign_match", "mean"),
            selected_margin_drop=("selected_margin_drop", "mean"),
            oracle_margin_drop=("oracle_margin_drop", "mean"),
            selected_success=("selected_success", "mean"),
            oracle_success=("oracle_success", "mean"),
        )
        .reset_index()
    )
    radius_summary = (
        radii.groupby(["model", "selector"])
        .agg(
            n=("dataset_idx", "size"),
            image_count=("dataset_idx", "nunique"),
            exact_candidate_match=("exact_candidate_match", "mean"),
            selected_alpha_255=("selected_alpha_255", "mean"),
            oracle_alpha_255=("oracle_alpha_255", "mean"),
            selected_margin_drop=("selected_margin_drop", "mean"),
            oracle_margin_drop=("oracle_margin_drop", "mean"),
            selected_success=("selected_success", "mean"),
            oracle_success=("oracle_success", "mean"),
            max_radius_success=("max_radius_success", "mean"),
        )
        .reset_index()
    )
    coverage_summary.to_csv(out / "proposal_coverage_summary.csv", index=False)
    sign_summary.to_csv(out / "sign_selection_summary.csv", index=False)
    radius_summary.to_csv(out / "radius_selection_summary.csv", index=False)
    print("\nProposal coverage\n", coverage_summary.to_string(index=False))
    print("\nSign selection\n", sign_summary.to_string(index=False))
    print("\nRadius selection\n", radius_summary.to_string(index=False))


if __name__ == "__main__":
    main()
