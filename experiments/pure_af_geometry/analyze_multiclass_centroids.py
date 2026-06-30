#!/usr/bin/env python3
"""Analyze multi-class pure AF/VF distances to clean class centroids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

LAYERS = ["sem0", "sem1", "sem4", "vf0", "vf1", "vf4"]
AF_LAYERS = ["sem0", "sem1", "sem4"]


def load_inputs(stats_path: Path, npz_path: Path):
    stats = pd.read_csv(stats_path)
    data = np.load(npz_path, allow_pickle=True)
    sample_ids = data["sample_id"].astype(str)
    if list(sample_ids) != list(stats["sample_id"].astype(str)):
        raise RuntimeError("sample_id order mismatch between stats CSV and pooled NPZ")
    features = {layer: data[layer].astype(np.float32) for layer in LAYERS if layer in data}
    return stats, features


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def summarize(x: pd.Series) -> dict[str, float | int]:
    vals = x.dropna().astype(float)
    if len(vals) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "std": float(vals.std(ddof=0)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def make_pca_plot(stats: pd.DataFrame, feature: np.ndarray, output: Path, title: str) -> None:
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(feature)
    plot_df = stats.copy()
    plot_df["pc1"] = xy[:, 0]
    plot_df["pc2"] = xy[:, 1]

    plt.figure(figsize=(9, 7))
    clean = plot_df[plot_df["source"] == "clean"]
    plt.scatter(clean["pc1"], clean["pc2"], c=clean["label"], cmap="tab10", s=18, alpha=0.55, label="clean")

    markers = {"real": "o", "random": "X"}
    colors = {"real": "black", "random": "red"}
    for init_mode, marker in markers.items():
        cur = plot_df[(plot_df["source"] == "pure") & (plot_df["init_mode"] == init_mode)]
        if cur.empty:
            continue
        plt.scatter(cur["pc1"], cur["pc2"], c=colors[init_mode], marker=marker, s=110, edgecolors="white", linewidths=0.8, label=f"pure {init_mode}")
        for _, row in cur.iterrows():
            plt.text(row["pc1"], row["pc2"], str(int(row["label"])), fontsize=8)

    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-stats", required=True)
    parser.add_argument("--pooled-features", required=True)
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/multiclass_centroids")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, features = load_inputs(Path(args.feature_stats), Path(args.pooled_features))
    stats["label"] = stats["label"].astype(int)

    clean_mask = stats["source"] == "clean"
    pure_mask = stats["source"] == "pure"
    labels = sorted(stats.loc[clean_mask, "label"].unique())

    centroid_rows = []
    distance_rows = []
    ratio_rows = []

    for layer, x in features.items():
        centroids = {}
        for label in labels:
            mask = clean_mask & (stats["label"] == label)
            if mask.sum() == 0:
                continue
            centroids[int(label)] = x[mask.to_numpy()].mean(axis=0)
            centroid_rows.append({"layer": layer, "label": int(label), "clean_n": int(mask.sum())})

        for idx, row in stats[pure_mask].iterrows():
            label = int(row["label"])
            vec = x[idx]
            all_d = {c: dist(vec, centroid) for c, centroid in centroids.items()}
            if label not in all_d:
                continue
            sorted_centroids = sorted(all_d.items(), key=lambda kv: kv[1])
            rank = 1 + [c for c, _d in sorted_centroids].index(label)
            nearest_label, nearest_dist = sorted_centroids[0]
            other_vals = [d for c, d in all_d.items() if c != label]
            distance_rows.append({
                "layer": layer,
                "label": label,
                "sample_id": row["sample_id"],
                "init_mode": row.get("init_mode", ""),
                "regularization": row.get("regularization", ""),
                "d_own": all_d[label],
                "nearest_label": int(nearest_label),
                "nearest_distance": float(nearest_dist),
                "own_rank": int(rank),
                "own_is_nearest": int(rank == 1),
                "mean_other_distance": float(np.mean(other_vals)) if other_vals else np.nan,
                "min_other_distance": float(np.min(other_vals)) if other_vals else np.nan,
                "own_vs_mean_other_ratio": all_d[label] / float(np.mean(other_vals)) if other_vals else np.nan,
                "own_vs_min_other_ratio": all_d[label] / float(np.min(other_vals)) if other_vals else np.nan,
            })

    dist_df = pd.DataFrame(distance_rows)
    dist_path = out_dir / "per_class_centroid_distances.csv"
    dist_df.to_csv(dist_path, index=False)

    for (layer, label, reg), group in dist_df.groupby(["layer", "label", "regularization"], dropna=False):
        real = group[group["init_mode"] == "real"]
        random = group[group["init_mode"] == "random"]
        if real.empty or random.empty:
            continue
        d_real = float(real["d_own"].mean())
        d_random = float(random["d_own"].mean())
        ratio_rows.append({
            "layer": layer,
            "label": int(label),
            "regularization": reg,
            "d_real": d_real,
            "d_random": d_random,
            "random_over_real": d_random / d_real if d_real else np.nan,
            "real_own_rank": float(real["own_rank"].mean()),
            "random_own_rank": float(random["own_rank"].mean()),
            "real_own_is_nearest": float(real["own_is_nearest"].mean()),
            "random_own_is_nearest": float(random["own_is_nearest"].mean()),
        })
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_path = out_dir / "real_random_distance_ratios.csv"
    ratio_df.to_csv(ratio_path, index=False)

    aggregate_rows = []
    for layer, group in ratio_df.groupby("layer", dropna=False):
        row = {"layer": layer}
        row.update({f"random_over_real_{k}": v for k, v in summarize(group["random_over_real"]).items()})
        row["classes_with_random_farther"] = int((group["random_over_real"] > 1.0).sum())
        row["classes_total"] = int(len(group))
        row["real_nearest_rate"] = float(group["real_own_is_nearest"].mean()) if len(group) else np.nan
        row["random_nearest_rate"] = float(group["random_own_is_nearest"].mean()) if len(group) else np.nan
        aggregate_rows.append(row)
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("random_over_real_mean", ascending=False)

    pair_map = {"sem0": "vf0", "sem1": "vf1", "sem4": "vf4"}
    af_vf_rows = []
    for af_layer, vf_layer in pair_map.items():
        af = aggregate_df[aggregate_df["layer"] == af_layer]
        vf = aggregate_df[aggregate_df["layer"] == vf_layer]
        if af.empty or vf.empty:
            continue
        af_mean = float(af["random_over_real_mean"].iloc[0])
        vf_mean = float(vf["random_over_real_mean"].iloc[0])
        af_vf_rows.append({
            "af_layer": af_layer,
            "vf_layer": vf_layer,
            "separation_af_random_over_real_mean": af_mean,
            "separation_vf_random_over_real_mean": vf_mean,
            "af_minus_vf": af_mean - vf_mean,
            "af_over_vf": af_mean / vf_mean if vf_mean else np.nan,
            "interpretation_hint": "af_stronger" if af_mean > vf_mean else "vf_stronger_or_equal",
        })
    af_vf_df = pd.DataFrame(af_vf_rows)
    af_vf_path = out_dir / "af_vs_vf_separation.csv"
    af_vf_df.to_csv(af_vf_path, index=False)

    aggregate_path = out_dir / "aggregate_layer_summary.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    ranking_df = aggregate_df[aggregate_df["layer"].isin(AF_LAYERS)].copy()
    ranking_df["rank_by_random_over_real_mean"] = ranking_df["random_over_real_mean"].rank(ascending=False, method="min").astype(int)
    ranking_path = out_dir / "af_layer_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)

    if "sem1" in features:
        make_pca_plot(stats, features["sem1"], out_dir / "sem1_pca_clean_pure.png", "sem1 PCA: clean centroids and pure images")

    metadata = {
        "feature_stats": args.feature_stats,
        "pooled_features": args.pooled_features,
        "outputs": {
            "per_class_centroid_distances": str(dist_path),
            "real_random_distance_ratios": str(ratio_path),
            "aggregate_layer_summary": str(aggregate_path),
            "af_layer_ranking": str(ranking_path),
            "af_vs_vf_separation": str(af_vf_path),
            "sem1_pca": str(out_dir / "sem1_pca_clean_pure.png"),
        },
        "clean_classes": [int(x) for x in labels],
        "n_rows": int(len(stats)),
        "n_pure": int(pure_mask.sum()),
        "n_clean": int(clean_mask.sum()),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
