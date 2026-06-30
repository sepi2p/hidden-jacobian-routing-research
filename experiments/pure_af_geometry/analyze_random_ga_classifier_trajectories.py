#!/usr/bin/env python3
"""Analyze random-noise-to-pure GA trajectories in classifier and AF/VF spaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_LAYERS = ["clf_conv1", "clf_layer1", "clf_layer2", "clf_layer3", "clf_layer4", "clf_avgpool", "clf_logits"]
SECONDARY_LAYERS = ["afvf_sem0", "afvf_sem1", "afvf_sem4", "afvf_vf0", "afvf_vf1", "afvf_vf4"]


def cos(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0:
        return float("nan")
    return float(np.dot(a, b) / den)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(x: np.ndarray) -> np.ndarray:
    # Average ranks for ties, enough for small trajectory diagnostics.
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2:
        return float("nan")
    return corr(rankdata(a), rankdata(b))


def load_run(row: pd.Series):
    traj = pd.read_csv(row["trajectory_csv"])
    feats = np.load(row["trajectory_features_npz"], allow_pickle=True)
    return traj, feats


def available_layers(npz) -> list[str]:
    return [k for k in PRIMARY_LAYERS + SECONDARY_LAYERS if k in npz.files]


def endpoint_direction(feats, layer: str) -> np.ndarray:
    x = feats[layer]
    return x[-1].astype(float) - x[0].astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/random_ga_classifier_trajectory_analysis")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["trajectory_features_npz"].notna()].copy()

    run_cache = {}
    layers = None
    for idx, row in manifest.iterrows():
        traj, feats = load_run(row)
        run_cache[idx] = (traj, feats)
        cur_layers = available_layers(feats)
        layers = cur_layers if layers is None else [x for x in layers if x in cur_layers]
    layers = layers or []

    direction_rows = []
    smooth_rows = []
    endpoint_rows = []

    for idx, row in manifest.iterrows():
        traj, feats = run_cache[idx]
        target = int(row["target_class"])
        run_name = row["run_name"]
        prob = traj["prob"].to_numpy(dtype=float)
        margin = traj["margin"].to_numpy(dtype=float)
        gen = traj["generation"].to_numpy(dtype=float)
        endpoint = {"run_name": run_name, "target_class": target, "seed": int(row["seed"]), "success": int(row.get("success", 0)), "saved_points": len(traj)}
        for layer in layers:
            x = feats[layer].astype(float)
            d = x - x[0:1]
            dist = np.linalg.norm(d, axis=1)
            endpoint[f"{layer}_endpoint_l2"] = float(dist[-1])
            endpoint[f"{layer}_path_l2_sum"] = float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum()) if len(x) > 1 else 0.0
            endpoint[f"{layer}_path_direct_ratio"] = endpoint[f"{layer}_path_l2_sum"] / endpoint[f"{layer}_endpoint_l2"] if endpoint[f"{layer}_endpoint_l2"] else np.nan
            smooth_rows.append({
                "run_name": run_name,
                "target_class": target,
                "seed": int(row["seed"]),
                "layer": layer,
                "n_points": len(traj),
                "pearson_dist_prob": corr(dist, prob),
                "spearman_dist_prob": spearman(dist, prob),
                "pearson_dist_margin": corr(dist, margin),
                "spearman_dist_margin": spearman(dist, margin),
                "pearson_generation_prob": corr(gen, prob),
                "spearman_generation_prob": spearman(gen, prob),
            })
        endpoint_rows.append(endpoint)

    endpoints = {}
    for idx, row in manifest.iterrows():
        _traj, feats = run_cache[idx]
        endpoints[idx] = {layer: endpoint_direction(feats, layer) for layer in layers}

    indices = list(manifest.index)
    for pos_i, i in enumerate(indices):
        for j in indices[pos_i + 1:]:
            ri = manifest.loc[i]
            rj = manifest.loc[j]
            same_class = int(ri["target_class"] == rj["target_class"])
            for layer in layers:
                direction_rows.append({
                    "run_i": ri["run_name"],
                    "run_j": rj["run_name"],
                    "class_i": int(ri["target_class"]),
                    "class_j": int(rj["target_class"]),
                    "seed_i": int(ri["seed"]),
                    "seed_j": int(rj["seed"]),
                    "same_class": same_class,
                    "layer": layer,
                    "endpoint_direction_cosine": cos(endpoints[i][layer], endpoints[j][layer]),
                })

    direction_df = pd.DataFrame(direction_rows)
    smooth_df = pd.DataFrame(smooth_rows)
    endpoint_df = pd.DataFrame(endpoint_rows)

    direction_path = out_dir / "endpoint_direction_pairwise_cosines.csv"
    smooth_path = out_dir / "trajectory_confidence_smoothness.csv"
    endpoint_path = out_dir / "trajectory_endpoint_stats.csv"
    direction_df.to_csv(direction_path, index=False)
    smooth_df.to_csv(smooth_path, index=False)
    endpoint_df.to_csv(endpoint_path, index=False)

    if len(direction_df):
        direction_summary = direction_df.groupby(["layer", "same_class"]).agg(
            n=("endpoint_direction_cosine", "size"),
            mean_cos=("endpoint_direction_cosine", "mean"),
            median_cos=("endpoint_direction_cosine", "median"),
            std_cos=("endpoint_direction_cosine", "std"),
        ).reset_index()
    else:
        direction_summary = pd.DataFrame()
    smooth_summary = smooth_df.groupby("layer").agg(
        n=("run_name", "size"),
        mean_spearman_dist_prob=("spearman_dist_prob", "mean"),
        median_spearman_dist_prob=("spearman_dist_prob", "median"),
        mean_spearman_dist_margin=("spearman_dist_margin", "mean"),
        median_spearman_dist_margin=("spearman_dist_margin", "median"),
    ).reset_index() if len(smooth_df) else pd.DataFrame()

    direction_summary_path = out_dir / "endpoint_direction_cosine_summary.csv"
    smooth_summary_path = out_dir / "trajectory_confidence_smoothness_summary.csv"
    direction_summary.to_csv(direction_summary_path, index=False)
    smooth_summary.to_csv(smooth_summary_path, index=False)

    metadata = {
        "manifest": args.manifest,
        "runs": int(len(manifest)),
        "layers": layers,
        "outputs": {
            "endpoint_direction_pairwise_cosines": str(direction_path),
            "endpoint_direction_cosine_summary": str(direction_summary_path),
            "trajectory_confidence_smoothness": str(smooth_path),
            "trajectory_confidence_smoothness_summary": str(smooth_summary_path),
            "trajectory_endpoint_stats": str(endpoint_path),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
