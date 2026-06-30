#!/usr/bin/env python3
"""Compare GA trajectory directions with analytic class-evidence fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import models


LAYERS = ["clf_logits", "clf_avgpool", "clf_layer4"]


def softmax_np(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64) - float(np.max(x))
    e = np.exp(z)
    return e / float(e.sum())


def cos(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else float("nan")


def nearest_level(prob: float, levels: list[float]) -> float:
    return min(levels, key=lambda x: abs(float(prob) - x))


def logp_field(layer: str, logits: np.ndarray, target: int, fc_weight: np.ndarray) -> np.ndarray:
    probs = softmax_np(logits)
    g_logits = -probs
    g_logits[int(target)] += 1.0
    if layer == "clf_logits":
        return g_logits
    if layer in {"clf_avgpool", "clf_layer4"}:
        return np.matmul(g_logits, fc_weight)
    raise ValueError(layer)


def score_field(layer: str, target: int, fc_weight: np.ndarray) -> np.ndarray:
    if layer == "clf_logits":
        out = np.zeros((1000,), dtype=np.float64)
        out[int(target)] = 1.0
        return out
    if layer in {"clf_avgpool", "clf_layer4"}:
        return fc_weight[int(target)].astype(np.float64)
    raise ValueError(layer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="analysis_outputs/pure_af_geometry/random_ga_classifier_trajectories_resnet18_c10_s5/manifest.csv")
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry")
    parser.add_argument("--levels", default="0.10,0.30,0.50,0.70,0.90")
    parser.add_argument("--success-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    levels = [float(x) for x in args.levels.split(",") if x.strip()]
    manifest = pd.read_csv(args.manifest)
    if args.success_only and "success" in manifest.columns:
        manifest = manifest[manifest["success"].astype(int) == 1].copy()
    manifest = manifest[manifest["trajectory_features_npz"].notna()].copy()

    resnet = models.resnet18(pretrained=True).eval()
    fc_weight = resnet.fc.weight.detach().cpu().numpy().astype(np.float64)

    rows = []
    matched_rows = []
    for _idx, row in manifest.iterrows():
        target = int(row["target_class"])
        traj = pd.read_csv(row["trajectory_csv"])
        z = np.load(row["trajectory_features_npz"])
        probs = traj["prob"].to_numpy(dtype=float)
        generations = traj["generation"].to_numpy(dtype=int)
        logits = z["clf_logits"].astype(np.float64)
        for layer in LAYERS:
            feats = z[layer].astype(np.float64)
            for t in range(len(feats) - 1):
                v = feats[t + 1] - feats[t]
                f_logp = logp_field(layer, logits[t], target, fc_weight)
                f_score = score_field(layer, target, fc_weight)
                rows.append({
                    "run_name": row["run_name"],
                    "target_class": target,
                    "seed": int(row["seed"]),
                    "success": int(row.get("success", 0)),
                    "layer": layer,
                    "segment_index": t,
                    "generation": int(generations[t]),
                    "next_generation": int(generations[t + 1]),
                    "prob": float(probs[t]),
                    "nearest_level": nearest_level(float(probs[t]), levels),
                    "segment_l2": float(np.linalg.norm(v)),
                    "field_logp_l2": float(np.linalg.norm(f_logp)),
                    "field_score_l2": float(np.linalg.norm(f_score)),
                    "cos_segment_with_grad_logp": cos(v, f_logp),
                    "cos_segment_with_grad_score": cos(v, f_score),
                })

            for level in levels:
                # Matched-confidence local segment: use nearest checkpoint with a following point.
                candidates = np.arange(max(0, len(feats) - 1))
                nearest = int(candidates[np.argmin(np.abs(probs[candidates] - level))])
                v = feats[nearest + 1] - feats[nearest]
                f_logp = logp_field(layer, logits[nearest], target, fc_weight)
                f_score = score_field(layer, target, fc_weight)
                matched_rows.append({
                    "run_name": row["run_name"],
                    "target_class": target,
                    "seed": int(row["seed"]),
                    "success": int(row.get("success", 0)),
                    "layer": layer,
                    "level": level,
                    "nearest_index": nearest,
                    "generation": int(generations[nearest]),
                    "next_generation": int(generations[nearest + 1]),
                    "prob": float(probs[nearest]),
                    "segment_l2": float(np.linalg.norm(v)),
                    "field_logp_l2": float(np.linalg.norm(f_logp)),
                    "field_score_l2": float(np.linalg.norm(f_score)),
                    "cos_segment_with_grad_logp": cos(v, f_logp),
                    "cos_segment_with_grad_score": cos(v, f_score),
                })

    all_df = pd.DataFrame(rows)
    matched_df = pd.DataFrame(matched_rows)
    all_path = out_dir / "trajectory_field_alignment_segments.csv"
    matched_path = out_dir / "trajectory_field_alignment_confidence_matched.csv"
    summary_path = out_dir / "trajectory_field_alignment_summary.csv"
    matched_summary_path = out_dir / "trajectory_field_alignment_confidence_matched_summary.csv"
    all_df.to_csv(all_path, index=False)
    matched_df.to_csv(matched_path, index=False)

    summary = all_df.groupby(["layer", "nearest_level"]).agg(
        n=("cos_segment_with_grad_logp", "size"),
        mean_cos_logp=("cos_segment_with_grad_logp", "mean"),
        median_cos_logp=("cos_segment_with_grad_logp", "median"),
        mean_cos_score=("cos_segment_with_grad_score", "mean"),
        median_cos_score=("cos_segment_with_grad_score", "median"),
        mean_prob=("prob", "mean"),
    ).reset_index()
    matched_summary = matched_df.groupby(["layer", "level"]).agg(
        n=("cos_segment_with_grad_logp", "size"),
        mean_cos_logp=("cos_segment_with_grad_logp", "mean"),
        median_cos_logp=("cos_segment_with_grad_logp", "median"),
        std_cos_logp=("cos_segment_with_grad_logp", "std"),
        mean_cos_score=("cos_segment_with_grad_score", "mean"),
        median_cos_score=("cos_segment_with_grad_score", "median"),
        mean_prob=("prob", "mean"),
    ).reset_index()
    summary.to_csv(summary_path, index=False)
    matched_summary.to_csv(matched_summary_path, index=False)

    meta = {
        "manifest": args.manifest,
        "success_only": bool(args.success_only),
        "runs": int(len(manifest)),
        "levels": levels,
        "layers": LAYERS,
        "outputs": [str(all_path), str(matched_path), str(summary_path), str(matched_summary_path)],
        "field": {
            "logp": "grad_h log p_y(h)",
            "score": "grad_h s_y(h)",
            "avgpool_layer4": "analytic through ResNet18 fc weight; layer4 is pooled layer4 feature",
        },
    }
    (out_dir / "trajectory_field_alignment_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("\nCONFIDENCE MATCHED SUMMARY")
    print(matched_summary.to_string(index=False))


if __name__ == "__main__":
    main()
