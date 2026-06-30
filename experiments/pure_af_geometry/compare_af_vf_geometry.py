#!/usr/bin/env python3
"""Compare clean, pure, and adversarial AF/VF geometry for pure_af_geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LAYERS = ["sem0", "sem1", "sem4", "vf0", "vf1", "vf4"]


def l2_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x - y, axis=1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def summarize(vals: np.ndarray) -> dict[str, float | int]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(len(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def load_features(stats_csv: Path, features_npz: Path):
    stats = pd.read_csv(stats_csv)
    data = np.load(features_npz, allow_pickle=True)
    sample_ids = data["sample_id"].astype(str)
    if list(sample_ids) != list(stats["sample_id"].astype(str)):
        raise RuntimeError("sample_id order mismatch between stats CSV and pooled NPZ")
    features = {layer: data[layer].astype(np.float32) for layer in LAYERS if layer in data}
    return stats, features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-stats", required=True)
    parser.add_argument("--pooled-features", required=True)
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/comparisons")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, feats = load_features(Path(args.feature_stats), Path(args.pooled_features))

    norm_rows = []
    for layer in LAYERS:
        for col in [f"{layer}_norm_l2", f"{layer}_norm_l1_mean"]:
            if col not in stats.columns:
                continue
            group_cols = ["source", "label"]
            if "init_mode" in stats.columns:
                group_cols.append("init_mode")
            if "regularization" in stats.columns:
                group_cols.append("regularization")
            for keys, group in stats.groupby(group_cols, dropna=False):
                key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
                row = {"metric": col, "layer": layer, **key_map, **summarize(group[col].to_numpy())}
                norm_rows.append(row)
    norm_df = pd.DataFrame(norm_rows)
    norm_path = out_dir / "norm_summary.csv"
    norm_df.to_csv(norm_path, index=False)

    distance_rows = []
    for label in sorted(stats["label"].dropna().astype(int).unique()):
        clean_mask = (stats["source"] == "clean") & (stats["label"].astype(int) == label)
        if clean_mask.sum() == 0:
            continue
        for layer, x in feats.items():
            clean_mean = x[clean_mask.to_numpy()].mean(axis=0, keepdims=True)
            for source in ["pure", "adv", "adv_clean"]:
                src_mask = (stats["source"] == source) & (stats["label"].astype(int) == label)
                if src_mask.sum() == 0:
                    continue
                sub = stats[src_mask]
                d = l2_rows(x[src_mask.to_numpy()], clean_mean)
                for keys, idxs in sub.groupby(["init_mode", "regularization"], dropna=False).groups.items():
                    local_positions = [sub.index.get_loc(i) for i in idxs]
                    key_init, key_reg = keys if isinstance(keys, tuple) else (keys, "")
                    distance_rows.append({
                        "label": label,
                        "layer": layer,
                        "source": source,
                        "init_mode": key_init,
                        "regularization": key_reg,
                        "distance_to_clean_class_mean": True,
                        **summarize(d[local_positions]),
                    })
    dist_df = pd.DataFrame(distance_rows)
    dist_path = out_dir / "distance_to_clean_class_mean.csv"
    dist_df.to_csv(dist_path, index=False)

    pure_pair_rows = []
    pure = stats[stats["source"] == "pure"].copy()
    for label in sorted(pure["label"].dropna().astype(int).unique()):
        for reg in sorted(str(x) for x in pure[pure["label"].astype(int) == label]["regularization"].fillna("").unique()):
            real_mask = (stats["source"] == "pure") & (stats["label"].astype(int) == label) & (stats["init_mode"] == "real") & (stats["regularization"].fillna("") == reg)
            random_mask = (stats["source"] == "pure") & (stats["label"].astype(int) == label) & (stats["init_mode"] == "random") & (stats["regularization"].fillna("") == reg)
            if real_mask.sum() == 0 or random_mask.sum() == 0:
                continue
            for layer, x in feats.items():
                real_mean = x[real_mask.to_numpy()].mean(axis=0, keepdims=True)
                random_mean = x[random_mask.to_numpy()].mean(axis=0, keepdims=True)
                pure_pair_rows.append({
                    "label": label,
                    "regularization": reg,
                    "layer": layer,
                    "real_random_mean_l2": float(np.linalg.norm(real_mean - random_mean)),
                    "real_n": int(real_mask.sum()),
                    "random_n": int(random_mask.sum()),
                })
    pair_df = pd.DataFrame(pure_pair_rows)
    pair_path = out_dir / "real_vs_random_pure_distance.csv"
    pair_df.to_csv(pair_path, index=False)

    align_rows = []
    if {"adv", "adv_clean", "pure"}.issubset(set(stats["source"].unique())):
        for label in sorted(stats["label"].dropna().astype(int).unique()):
            pure_label = stats[(stats["source"] == "pure") & (stats["label"].astype(int) == label)]
            adv_pairs = stats[(stats["source"].isin(["adv", "adv_clean"])) & (stats["label"].astype(int) == label) & (stats["pair_id"].fillna("") != "")]
            if pure_label.empty or adv_pairs.empty:
                continue
            for layer, x in feats.items():
                clean_pool_mask = (stats["source"] == "clean") & (stats["label"].astype(int) == label)
                if clean_pool_mask.sum() == 0:
                    continue
                clean_mean = x[clean_pool_mask.to_numpy()].mean(axis=0)
                for (init_mode, reg), pure_group in pure_label.groupby(["init_mode", "regularization"], dropna=False):
                    pure_vec = x[pure_group.index.to_numpy()].mean(axis=0)
                    pure_shift = pure_vec - clean_mean
                    cos_vals = []
                    for pair_id, pair in adv_pairs.groupby("pair_id"):
                        if set(pair["source"]) != {"adv", "adv_clean"}:
                            continue
                        adv_idx = pair[pair["source"] == "adv"].index[0]
                        clean_idx = pair[pair["source"] == "adv_clean"].index[0]
                        adv_shift = x[adv_idx] - x[clean_idx]
                        cos_vals.append(cosine(pure_shift, adv_shift))
                    align_rows.append({
                        "label": label,
                        "layer": layer,
                        "pure_init_mode": init_mode,
                        "pure_regularization": reg,
                        "alignment": "cosine_pure_shift_vs_adv_shift",
                        **summarize(np.array(cos_vals)),
                    })
    align_df = pd.DataFrame(align_rows)
    align_path = out_dir / "adv_shift_alignment.csv"
    align_df.to_csv(align_path, index=False)

    metadata = {
        "feature_stats": args.feature_stats,
        "pooled_features": args.pooled_features,
        "outputs": {
            "norm_summary": str(norm_path),
            "distance_to_clean_class_mean": str(dist_path),
            "real_vs_random_pure_distance": str(pair_path),
            "adv_shift_alignment": str(align_path),
        },
        "layers": LAYERS,
        "rows": int(len(stats)),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
