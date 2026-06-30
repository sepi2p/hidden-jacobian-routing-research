#!/usr/bin/env python3
"""Analyze whether class-pure GA trajectories form point attractors or parallel flows."""

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


def pair_stats(vectors: list[np.ndarray]) -> dict[str, float]:
    cosines = []
    l2s = []
    for a, b in combinations(vectors, 2):
        cosines.append(cos(a, b))
        l2s.append(float(np.linalg.norm(a - b)))
    return {
        "pair_count": len(cosines),
        "mean_pair_cos": float(np.nanmean(cosines)) if cosines else np.nan,
        "median_pair_cos": float(np.nanmedian(cosines)) if cosines else np.nan,
        "mean_pair_l2": float(np.nanmean(l2s)) if l2s else np.nan,
        "median_pair_l2": float(np.nanmedian(l2s)) if l2s else np.nan,
    }


def nearest_index(gens: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(gens.astype(float) - float(target))))


def segment_vectors(feats: np.ndarray, gens: np.ndarray):
    final_gen = float(gens[-1])
    idxs = [nearest_index(gens, final_gen * frac) for frac in [0.0, 0.25, 0.5, 0.75, 1.0]]
    idxs[0] = 0
    idxs[-1] = len(gens) - 1
    return [feats[idxs[i + 1]].astype(float) - feats[idxs[i]].astype(float) for i in range(4)], idxs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="analysis_outputs/pure_af_geometry/random_ga_classifier_trajectories_resnet18_c10_s5/manifest.csv")
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry")
    parser.add_argument("--success-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    manifest = pd.read_csv(args.manifest)
    if args.success_only and "success" in manifest.columns:
        manifest = manifest[manifest["success"].astype(int) == 1].copy()
    manifest = manifest[manifest["trajectory_features_npz"].notna()].copy()

    cache = {}
    for i, row in manifest.iterrows():
        cache[i] = np.load(row["trajectory_features_npz"])

    flow_rows = []
    for layer in LAYERS:
        for cls, group in manifest.groupby("target_class"):
            idxs = list(group.index)
            starts = [cache[i][layer][0].astype(float) for i in idxs]
            finals = [cache[i][layer][-1].astype(float) for i in idxs]
            dirs = [cache[i][layer][-1].astype(float) - cache[i][layer][0].astype(float) for i in idxs]
            s = pair_stats(starts)
            f = pair_stats(finals)
            d = pair_stats(dirs)
            flow_rows.append({
                "layer": layer,
                "target_class": int(cls),
                "n": len(idxs),
                "start_mean_pair_cos": s["mean_pair_cos"],
                "start_mean_pair_l2": s["mean_pair_l2"],
                "final_mean_pair_cos": f["mean_pair_cos"],
                "final_mean_pair_l2": f["mean_pair_l2"],
                "direction_mean_pair_cos": d["mean_pair_cos"],
                "direction_mean_pair_l2": d["mean_pair_l2"],
                "final_l2_over_start_l2": f["mean_pair_l2"] / s["mean_pair_l2"] if s["mean_pair_l2"] else np.nan,
                "final_cos_minus_start_cos": f["mean_pair_cos"] - s["mean_pair_cos"],
            })
    flow_df = pd.DataFrame(flow_rows)
    flow_df.to_csv(out_dir / "attractor_flow_vs_point_summary.csv", index=False)
    flow_agg = flow_df.groupby("layer").agg(
        classes=("target_class", "size"),
        mean_start_l2=("start_mean_pair_l2", "mean"),
        mean_final_l2=("final_mean_pair_l2", "mean"),
        mean_final_l2_over_start_l2=("final_l2_over_start_l2", "mean"),
        mean_start_cos=("start_mean_pair_cos", "mean"),
        mean_final_cos=("final_mean_pair_cos", "mean"),
        mean_direction_cos=("direction_mean_pair_cos", "mean"),
    ).reset_index()
    flow_agg.to_csv(out_dir / "attractor_flow_vs_point_aggregate.csv", index=False)

    curvature_rows = []
    for i, row in manifest.iterrows():
        z = cache[i]
        gens = z["generation"]
        for layer in LAYERS:
            feats = z[layer].astype(float)
            segs, idxs = segment_vectors(feats, gens)
            seg_norms = [float(np.linalg.norm(v)) for v in segs]
            path_length = float(sum(seg_norms))
            straight = float(np.linalg.norm(feats[-1] - feats[0]))
            curvature_rows.append({
                "run_name": row["run_name"],
                "target_class": int(row["target_class"]),
                "seed": int(row["seed"]),
                "success": int(row.get("success", 0)),
                "layer": layer,
                "idx_0": idxs[0],
                "idx_25": idxs[1],
                "idx_50": idxs[2],
                "idx_75": idxs[3],
                "idx_100": idxs[4],
                "cos_v1_v2": cos(segs[0], segs[1]),
                "cos_v2_v3": cos(segs[1], segs[2]),
                "cos_v3_v4": cos(segs[2], segs[3]),
                "cos_v1_v4": cos(segs[0], segs[3]),
                "path_length": path_length,
                "straight_line_length": straight,
                "curvature_ratio": path_length / straight if straight else np.nan,
            })
    curv_df = pd.DataFrame(curvature_rows)
    curv_df.to_csv(out_dir / "trajectory_curvature_per_run.csv", index=False)
    curv_summary = curv_df.groupby("layer").agg(
        n=("run_name", "size"),
        mean_cos_v1_v2=("cos_v1_v2", "mean"),
        mean_cos_v2_v3=("cos_v2_v3", "mean"),
        mean_cos_v3_v4=("cos_v3_v4", "mean"),
        mean_cos_v1_v4=("cos_v1_v4", "mean"),
        median_curvature_ratio=("curvature_ratio", "median"),
        mean_curvature_ratio=("curvature_ratio", "mean"),
        mean_path_length=("path_length", "mean"),
        mean_straight_line_length=("straight_line_length", "mean"),
    ).reset_index()
    curv_summary.to_csv(out_dir / "trajectory_curvature_summary.csv", index=False)

    meta = {
        "manifest": args.manifest,
        "success_only": bool(args.success_only),
        "rows": int(len(manifest)),
        "layers": LAYERS,
        "outputs": [
            str(out_dir / "attractor_flow_vs_point_summary.csv"),
            str(out_dir / "attractor_flow_vs_point_aggregate.csv"),
            str(out_dir / "trajectory_curvature_per_run.csv"),
            str(out_dir / "trajectory_curvature_summary.csv"),
        ],
    }
    (out_dir / "attractor_flow_geometry_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("\nFLOW AGGREGATE")
    print(flow_agg.to_string(index=False))
    print("\nCURVATURE SUMMARY")
    print(curv_summary.to_string(index=False))


if __name__ == "__main__":
    main()
