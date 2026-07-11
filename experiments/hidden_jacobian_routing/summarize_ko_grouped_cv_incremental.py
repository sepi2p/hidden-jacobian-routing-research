#!/usr/bin/env python3
"""Grouped out-of-sample incremental models for the clean-start comparator.

The original clean-start comparator wrote in-sample logistic-regression scores.
This post-processing script recomputes the incremental AUPRC/AUROC with
GroupKFold over image IDs, so all candidate rows from a held-out image are
excluded from training before evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_SETS = [
    ("M0_singular", ["rank", "singular_value"]),
    ("M1_jvp_proxy", ["rank", "singular_value", "jvp_gain"]),
    ("M2_margin", ["rank", "singular_value", "jvp_gain", "margin_drop"]),
    ("M3_gradient", ["rank", "singular_value", "jvp_gain", "margin_drop", "ce_grad_cos", "dlr_grad_cos"]),
    (
        "M4_transport",
        ["rank", "singular_value", "jvp_gain", "margin_drop", "ce_grad_cos", "dlr_grad_cos", "transport_projection_energy"],
    ),
]


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    y = np.asarray(y, dtype=int)[ok]
    s = np.asarray(s, dtype=float)[ok]
    if len(y) < 4 or len(np.unique(y)) < 2 or np.nanstd(s) < 1e-12:
        return np.nan
    return float(roc_auc_score(y, s))


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    y = np.asarray(y, dtype=int)[ok]
    s = np.asarray(s, dtype=float)[ok]
    if len(y) < 4 or len(np.unique(y)) < 2 or np.nanstd(s) < 1e-12:
        return np.nan
    return float(average_precision_score(y, s))


def grouped_oof_scores(df: pd.DataFrame, cols: list[str], n_splits: int) -> tuple[np.ndarray, int]:
    x = df[cols].to_numpy(float)
    y = df["candidate_success"].to_numpy(int)
    groups = df["dataset_idx"].to_numpy(int)
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
    scores = np.full(len(df), np.nan, dtype=float)
    valid_groups = np.unique(groups[ok])
    folds = min(n_splits, len(valid_groups))
    if folds < 2 or len(np.unique(y[ok])) < 2:
        return scores, folds
    splitter = GroupKFold(n_splits=folds)
    ok_indices = np.flatnonzero(ok)
    for train_rel, test_rel in splitter.split(x[ok], y[ok], groups[ok]):
        train_idx = ok_indices[train_rel]
        test_idx = ok_indices[test_rel]
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
        )
        clf.fit(x[train_idx], y[train_idx])
        scores[test_idx] = clf.predict_proba(x[test_idx])[:, 1]
    return scores, folds


def paired_group_bootstrap_delta(
    y: np.ndarray,
    score_m3: np.ndarray,
    score_m4: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, int]:
    """Conditional image-bootstrap CI for the paired OOF AUPRC increment.

    The fitted OOF models and folds are held fixed. Each replicate resamples
    image IDs and carries all candidate rows belonging to the sampled image.
    This quantifies image-sampling uncertainty, not full pipeline uncertainty.
    """
    ok = np.isfinite(score_m3) & np.isfinite(score_m4)
    y = np.asarray(y, dtype=int)[ok]
    score_m3 = np.asarray(score_m3, dtype=float)[ok]
    score_m4 = np.asarray(score_m4, dtype=float)[ok]
    groups = np.asarray(groups)[ok]
    unique_groups = np.unique(groups)
    row_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([row_indices[group] for group in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(average_precision_score(y[idx], score_m4[idx]) - average_precision_score(y[idx], score_m3[idx]))
    if not deltas:
        return np.nan, np.nan, np.nan, 0
    values = np.asarray(deltas, dtype=float)
    return float(values.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), int(len(values))


def iter_score_files(root: Path):
    yield from sorted(root.glob("bbb_*/*/candidate_seed_*/ko_candidate_scores_*.csv"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-root",
        default="analysis_outputs/pure_af_geometry/q1_reviewer_validation/exact_protocol/phase1a_ko_cleanstart_comparator",
    )
    p.add_argument(
        "--output-dir",
        default="analysis_outputs/pure_af_geometry/q1_reviewer_validation/exact_protocol/ko_grouped_cv_incremental",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--bootstrap-seed", type=int, default=20260712)
    p.add_argument(
        "--bootstrap-candidate-seed",
        type=int,
        default=0,
        help="Candidate seed used for the conditional image-bootstrap audit; all seeds remain in point summaries.",
    )
    args = p.parse_args()

    root = Path(args.input_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    usecols = sorted(
        {
            "model",
            "split_seed",
            "candidate_seed",
            "layer_rule",
            "layer",
            "dataset_idx",
            "candidate_success",
            *[c for _name, cols in FEATURE_SETS for c in cols],
        }
    )
    rows = []
    bootstrap_rows = []
    for path in iter_score_files(root):
        df = pd.read_csv(path, usecols=usecols)
        if df.empty:
            continue
        run_scores: dict[str, np.ndarray] = {}
        for name, cols in FEATURE_SETS:
            scores, folds = grouped_oof_scores(df, cols, args.n_splits)
            run_scores[name] = scores
            y = df["candidate_success"].to_numpy(int)
            rows.append(
                {
                    "model": str(df["model"].iloc[0]),
                    "split_seed": int(df["split_seed"].iloc[0]),
                    "candidate_seed": int(df["candidate_seed"].iloc[0]),
                    "layer_rule": str(df["layer_rule"].iloc[0]),
                    "layer": str(df["layer"].iloc[0]),
                    "nested_model": name,
                    "features": ",".join(cols),
                    "n_candidates": int(len(df)),
                    "n_images": int(df["dataset_idx"].nunique()),
                    "n_positive": int(y.sum()),
                    "positive_rate": float(y.mean()),
                    "group_folds": int(folds),
                    "oof_auroc": safe_auroc(y, scores),
                    "oof_auprc": safe_auprc(y, scores),
                }
            )
        if (
            str(df["layer_rule"].iloc[0]) == "nested_selected_nonlogit"
            and int(df["candidate_seed"].iloc[0]) == args.bootstrap_candidate_seed
        ):
            mean_delta, ci_low, ci_high, n_valid = paired_group_bootstrap_delta(
                y=df["candidate_success"].to_numpy(int),
                score_m3=run_scores["M3_gradient"],
                score_m4=run_scores["M4_transport"],
                groups=df["dataset_idx"].to_numpy(int),
                n_boot=args.bootstrap,
                seed=args.bootstrap_seed + int(df["split_seed"].iloc[0]),
            )
            point_delta = safe_auprc(df["candidate_success"].to_numpy(int), run_scores["M4_transport"]) - safe_auprc(
                df["candidate_success"].to_numpy(int), run_scores["M3_gradient"]
            )
            bootstrap_rows.append(
                {
                    "model": str(df["model"].iloc[0]),
                    "split_seed": int(df["split_seed"].iloc[0]),
                    "candidate_seed": int(df["candidate_seed"].iloc[0]),
                    "layer_rule": str(df["layer_rule"].iloc[0]),
                    "layer": str(df["layer"].iloc[0]),
                    "n_images": int(df["dataset_idx"].nunique()),
                    "point_delta_transport_oof_auprc": float(point_delta),
                    "bootstrap_delta_mean": mean_delta,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "n_boot_valid": n_valid,
                    "uncertainty_scope": "conditional_on_fitted_oof_models_and_selected_layer",
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "ko_grouped_cv_incremental_models.csv", index=False)
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(out / "ko_grouped_cv_delta_image_bootstrap.csv", index=False)
    if metrics.empty:
        raise SystemExit("No score files found")

    wide = metrics.pivot_table(
        index=["model", "split_seed", "candidate_seed", "layer_rule", "layer"],
        columns="nested_model",
        values="oof_auprc",
        aggfunc="first",
    ).reset_index()
    if {"M3_gradient", "M4_transport"}.issubset(wide.columns):
        wide["delta_transport_oof_auprc"] = wide["M4_transport"] - wide["M3_gradient"]
    wide.to_csv(out / "ko_grouped_cv_incremental_deltas.csv", index=False)

    summary = (
        wide.groupby(["model", "layer_rule"], dropna=False)
        .agg(
            n_runs=("delta_transport_oof_auprc", "size"),
            layers=("layer", lambda x: "/".join(sorted(set(map(str, x))))),
            m0_singular_mean=("M0_singular", "mean"),
            m1_jvp_proxy_mean=("M1_jvp_proxy", "mean"),
            m2_margin_mean=("M2_margin", "mean"),
            m3_gradient_mean=("M3_gradient", "mean"),
            m4_transport_mean=("M4_transport", "mean"),
            delta_transport_mean=("delta_transport_oof_auprc", "mean"),
            delta_transport_min=("delta_transport_oof_auprc", "min"),
            delta_transport_max=("delta_transport_oof_auprc", "max"),
            delta_transport_std=("delta_transport_oof_auprc", "std"),
        )
        .reset_index()
    )
    prev = (
        metrics.groupby(["model", "layer_rule"], dropna=False)
        .agg(
            positive_rate_mean=("positive_rate", "mean"),
            positive_rate_min=("positive_rate", "min"),
            positive_rate_max=("positive_rate", "max"),
        )
        .reset_index()
    )
    summary = summary.merge(prev, on=["model", "layer_rule"], how="left")
    summary.to_csv(out / "ko_grouped_cv_incremental_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
