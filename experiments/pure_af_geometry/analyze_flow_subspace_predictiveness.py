#!/usr/bin/env python3
"""Test whether successful-flow PCA subspace predicts trajectory success."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


LAYERS = ["clf_avgpool", "clf_layer4", "clf_logits"]
KS = [5, 10, 20, 50, 100]


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def collect_segments(manifest: pd.DataFrame, layer: str):
    success = []
    failed = []
    meta_success = []
    meta_failed = []
    for _idx, row in manifest.iterrows():
        z = np.load(row["trajectory_features_npz"])
        feats = z[layer].astype(np.float64)
        is_success = int(row.get("success", 0)) == 1
        for t in range(len(feats) - 1):
            seg = feats[t + 1] - feats[t]
            if np.linalg.norm(seg) <= 1e-12:
                continue
            item = {
                "run_name": row["run_name"],
                "target_class": int(row["target_class"]),
                "seed": int(row["seed"]),
                "segment_index": int(t),
            }
            if is_success:
                success.append(seg)
                meta_success.append(item)
            else:
                failed.append(seg)
                meta_failed.append(item)
    return (
        normalize_rows(np.stack(success)) if success else np.empty((0, 0)),
        normalize_rows(np.stack(failed)) if failed else np.empty((0, 0)),
        meta_success,
        meta_failed,
    )


def fit_pca_basis(x_train: np.ndarray, max_k: int):
    mean = x_train.mean(axis=0, keepdims=True)
    xc = x_train - mean
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return mean, vt[:max_k]


def projection_energies(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, ks: list[int]) -> pd.DataFrame:
    xc = x - mean
    denom = np.sum(xc * xc, axis=1)
    coeff = xc @ basis.T
    rows = {}
    for k in ks:
        kk = min(k, basis.shape[0])
        num = np.sum(coeff[:, :kk] ** 2, axis=1)
        rows[f"energy_k{k}"] = num / np.clip(denom, 1e-12, None)
    return pd.DataFrame(rows)


def auroc_safe(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def mwu_safe(pos: np.ndarray, neg: np.ndarray):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
    # Rank-biserial / common-language effect in [-1, 1].
    auc = float(stat / (len(pos) * len(neg)))
    return float(p), float(2 * auc - 1)


def logistic_cv(rows: pd.DataFrame, label_col: str, feature_cols: list[str], seed: int):
    y = rows[label_col].to_numpy(dtype=int)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        return {"cv_accuracy_mean": np.nan, "cv_accuracy_std": np.nan}
    n_splits = min(5, int(min(np.bincount(y))))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, rows[feature_cols].to_numpy(), y, cv=cv, scoring="accuracy")
    return {"cv_accuracy_mean": float(scores.mean()), "cv_accuracy_std": float(scores.std())}


def percentile_bins(success_rows: pd.DataFrame, random_rows: pd.DataFrame, layer: str, k: int):
    col = f"energy_k{k}"
    combined = pd.concat([
        success_rows.assign(label=1),
        random_rows.assign(label=0),
    ], ignore_index=True)
    combined["percentile_bin"] = pd.qcut(combined[col].rank(method="first"), 10, labels=False)
    out = combined.groupby("percentile_bin").agg(
        layer=("layer", "first"),
        k=("layer", lambda _s: k),
        n=("label", "size"),
        success_rate=("label", "mean"),
        mean_energy=(col, "mean"),
        min_energy=(col, "min"),
        max_energy=(col, "max"),
    ).reset_index()
    out["layer"] = layer
    out["k"] = k
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="analysis_outputs/pure_af_geometry/random_ga_classifier_trajectories_resnet18_c10_s5/manifest.csv")
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-multiplier", type=int, default=2)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["trajectory_features_npz"].notna()].copy()

    score_rows = []
    metric_rows = []
    bin_rows = []
    logreg_rows = []

    for layer in LAYERS:
        success, failed, meta_success, meta_failed = collect_segments(manifest, layer)
        if len(success) == 0:
            continue
        idx = np.arange(len(success))
        strat = np.array([m["target_class"] for m in meta_success])
        train_idx, test_idx = train_test_split(
            idx,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=strat if min(np.bincount(strat)) >= 2 else None,
        )
        x_train = success[train_idx]
        x_success_test = success[test_idx]
        max_k = min(max(KS), x_train.shape[0] - 1, x_train.shape[1])
        mean, basis = fit_pca_basis(x_train, max_k)
        n_random = max(len(x_success_test) * args.random_multiplier, len(failed), 1)
        random_x = normalize_rows(rng.normal(size=(n_random, success.shape[1])))

        groups = {
            "success_heldout": x_success_test,
            "failed": failed,
            "random": random_x,
        }
        group_frames = {}
        for group_name, x in groups.items():
            energies = projection_energies(x, mean, basis, KS)
            energies["layer"] = layer
            energies["group"] = group_name
            energies["label_success"] = 1 if group_name == "success_heldout" else 0
            energies["label_success_vs_random"] = 1 if group_name == "success_heldout" else (0 if group_name == "random" else -1)
            energies["label_success_vs_failed"] = 1 if group_name == "success_heldout" else (0 if group_name == "failed" else -1)
            group_frames[group_name] = energies
            score_rows.append(energies)

        for k in KS:
            col = f"energy_k{k}"
            pos = group_frames["success_heldout"][col].to_numpy()
            for neg_name in ["random", "failed"]:
                neg = group_frames[neg_name][col].to_numpy()
                y = np.r_[np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]
                score = np.r_[pos, neg]
                p_value, rank_biserial = mwu_safe(pos, neg)
                metric_rows.append({
                    "layer": layer,
                    "k": k,
                    "comparison": f"success_vs_{neg_name}",
                    "n_success": len(pos),
                    "n_negative": len(neg),
                    "success_mean_energy": float(np.mean(pos)),
                    "negative_mean_energy": float(np.mean(neg)) if len(neg) else np.nan,
                    "mean_difference": float(np.mean(pos) - np.mean(neg)) if len(neg) else np.nan,
                    "auroc": auroc_safe(y, score),
                    "mannwhitney_p": p_value,
                    "rank_biserial": rank_biserial,
                })

        feature_cols = [f"energy_k{k}" for k in KS]
        sr = pd.concat([group_frames["success_heldout"], group_frames["random"]], ignore_index=True)
        sr = sr[sr["label_success_vs_random"] >= 0].copy()
        lr = logistic_cv(sr, "label_success_vs_random", feature_cols, args.seed)
        lr.update({"layer": layer, "comparison": "success_vs_random", "n": len(sr)})
        logreg_rows.append(lr)

        sf = pd.concat([group_frames["success_heldout"], group_frames["failed"]], ignore_index=True)
        sf = sf[sf["label_success_vs_failed"] >= 0].copy()
        lr = logistic_cv(sf, "label_success_vs_failed", feature_cols, args.seed)
        lr.update({"layer": layer, "comparison": "success_vs_failed", "n": len(sf)})
        logreg_rows.append(lr)

        for k in KS:
            bin_rows.append(percentile_bins(group_frames["success_heldout"], group_frames["random"], layer, k))

    scores = pd.concat(score_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    bins = pd.concat(bin_rows, ignore_index=True)
    logreg = pd.DataFrame(logreg_rows)
    scores_path = out_dir / "flow_subspace_projection_scores.csv"
    metrics_path = out_dir / "flow_subspace_predictiveness_metrics.csv"
    bins_path = out_dir / "flow_subspace_success_probability_by_percentile.csv"
    logreg_path = out_dir / "flow_subspace_logistic_regression.csv"
    scores.to_csv(scores_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    bins.to_csv(bins_path, index=False)
    logreg.to_csv(logreg_path, index=False)
    meta = {
        "manifest": args.manifest,
        "layers": LAYERS,
        "ks": KS,
        "test_size": args.test_size,
        "random_multiplier": args.random_multiplier,
        "outputs": [str(scores_path), str(metrics_path), str(bins_path), str(logreg_path)],
    }
    (out_dir / "flow_subspace_predictiveness_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("\nMETRICS")
    print(metrics.sort_values(["layer", "comparison", "k"]).to_string(index=False))
    print("\nLOGISTIC")
    print(logreg.sort_values(["layer", "comparison"]).to_string(index=False))


if __name__ == "__main__":
    main()
