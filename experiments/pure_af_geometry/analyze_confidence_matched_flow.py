#!/usr/bin/env python3
"""Compare local same-class trajectory directions at matched confidence levels."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


LAYERS = ["clf_logits", "clf_avgpool", "clf_layer4"]


def cos(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else float("nan")


def nearest_prob_index(probs: np.ndarray, level: float) -> int:
    return int(np.argmin(np.abs(probs.astype(float) - float(level))))


def local_vector(feats: np.ndarray, idx: int, window: int) -> tuple[np.ndarray, int, int]:
    lo = max(0, idx - window)
    hi = min(len(feats) - 1, idx + window)
    if hi == lo:
        if hi < len(feats) - 1:
            hi += 1
        elif lo > 0:
            lo -= 1
    return feats[hi].astype(float) - feats[lo].astype(float), lo, hi


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="analysis_outputs/pure_af_geometry/random_ga_classifier_trajectories_resnet18_c10_s5/manifest.csv")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry")
    p.add_argument("--levels", default="0.10,0.30,0.50,0.70,0.90")
    p.add_argument("--window", type=int, default=1, help="Saved-checkpoint radius around nearest confidence point.")
    p.add_argument("--success-only", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    levels = [float(x) for x in args.levels.split(",") if x.strip()]
    manifest = pd.read_csv(args.manifest)
    if args.success_only and "success" in manifest.columns:
        manifest = manifest[manifest["success"].astype(int) == 1].copy()
    manifest = manifest[manifest["trajectory_features_npz"].notna()].copy()

    runs = {}
    for i, row in manifest.iterrows():
        traj = pd.read_csv(row["trajectory_csv"])
        feats = np.load(row["trajectory_features_npz"])
        runs[i] = {"row": row, "traj": traj, "feats": feats}

    point_rows = []
    pair_rows = []
    for layer in LAYERS:
        for cls, group in manifest.groupby("target_class"):
            idxs = list(group.index)
            for level in levels:
                vectors = {}
                for idx in idxs:
                    item = runs[idx]
                    probs = item["traj"]["prob"].to_numpy(dtype=float)
                    nearest = nearest_prob_index(probs, level)
                    vec, lo, hi = local_vector(item["feats"][layer], nearest, args.window)
                    vectors[idx] = vec
                    point_rows.append({
                        "layer": layer,
                        "target_class": int(cls),
                        "run_name": item["row"]["run_name"],
                        "seed": int(item["row"]["seed"]),
                        "level": level,
                        "nearest_index": nearest,
                        "nearest_generation": int(item["traj"]["generation"].iloc[nearest]),
                        "nearest_prob": float(probs[nearest]),
                        "lo_index": lo,
                        "hi_index": hi,
                        "local_l2": float(np.linalg.norm(vec)),
                    })
                for i, j in combinations(idxs, 2):
                    pair_rows.append({
                        "layer": layer,
                        "target_class": int(cls),
                        "level": level,
                        "run_i": runs[i]["row"]["run_name"],
                        "run_j": runs[j]["row"]["run_name"],
                        "seed_i": int(runs[i]["row"]["seed"]),
                        "seed_j": int(runs[j]["row"]["seed"]),
                        "local_direction_cos": cos(vectors[i], vectors[j]),
                        "local_l2_i": float(np.linalg.norm(vectors[i])),
                        "local_l2_j": float(np.linalg.norm(vectors[j])),
                    })

    point_df = pd.DataFrame(point_rows)
    pair_df = pd.DataFrame(pair_rows)
    point_df.to_csv(out_dir / "confidence_matched_flow_points.csv", index=False)
    pair_df.to_csv(out_dir / "confidence_matched_flow_pairwise.csv", index=False)
    summary = pair_df.groupby(["layer", "level"]).agg(
        n=("local_direction_cos", "size"),
        mean_cos=("local_direction_cos", "mean"),
        median_cos=("local_direction_cos", "median"),
        std_cos=("local_direction_cos", "std"),
        min_cos=("local_direction_cos", "min"),
        max_cos=("local_direction_cos", "max"),
    ).reset_index()
    summary.to_csv(out_dir / "confidence_matched_flow_summary.csv", index=False)
    meta = {
        "manifest": args.manifest,
        "levels": levels,
        "window": args.window,
        "success_only": bool(args.success_only),
        "runs": int(len(manifest)),
        "outputs": [
            str(out_dir / "confidence_matched_flow_points.csv"),
            str(out_dir / "confidence_matched_flow_pairwise.csv"),
            str(out_dir / "confidence_matched_flow_summary.csv"),
        ],
    }
    (out_dir / "confidence_matched_flow_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
