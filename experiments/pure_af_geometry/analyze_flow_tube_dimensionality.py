#!/usr/bin/env python3
"""Estimate dimensionality of normalized GA trajectory segment flow tubes."""

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


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def pca_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n < 2:
        return {
            "n": n, "d": d, "pc1_var": np.nan, "dim80": np.nan, "dim90": np.nan,
            "dim95": np.nan, "effective_rank": np.nan,
        }
    xc = x - x.mean(axis=0, keepdims=True)
    _u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    var = s ** 2
    total = float(var.sum())
    if total <= 0:
        ratios = np.zeros_like(var)
    else:
        ratios = var / total
    csum = np.cumsum(ratios)
    entropy = -float(np.sum(ratios[ratios > 0] * np.log(ratios[ratios > 0])))
    return {
        "n": int(n),
        "d": int(d),
        "pc1_var": float(ratios[0]) if len(ratios) else np.nan,
        "pc2_var": float(ratios[1]) if len(ratios) > 1 else np.nan,
        "pc5_cum_var": float(csum[min(4, len(csum) - 1)]) if len(csum) else np.nan,
        "pc10_cum_var": float(csum[min(9, len(csum) - 1)]) if len(csum) else np.nan,
        "dim80": int(np.searchsorted(csum, 0.80) + 1) if len(csum) else np.nan,
        "dim90": int(np.searchsorted(csum, 0.90) + 1) if len(csum) else np.nan,
        "dim95": int(np.searchsorted(csum, 0.95) + 1) if len(csum) else np.nan,
        "effective_rank": float(np.exp(entropy)),
    }


def logp_field(layer: str, logits: np.ndarray, target: int, fc_weight: np.ndarray) -> np.ndarray:
    probs = softmax_np(logits)
    g_logits = -probs
    g_logits[int(target)] += 1.0
    if layer == "clf_logits":
        return g_logits
    if layer in {"clf_avgpool", "clf_layer4"}:
        return np.matmul(g_logits, fc_weight)
    raise ValueError(layer)


def collect_vectors(manifest: pd.DataFrame, layer: str, fc_weight: np.ndarray):
    success_segments = []
    failed_segments = []
    grad_success = []
    grad_failed = []
    class_segments: dict[int, list[np.ndarray]] = {}
    class_grads: dict[int, list[np.ndarray]] = {}
    for _idx, row in manifest.iterrows():
        z = np.load(row["trajectory_features_npz"])
        feats = z[layer].astype(np.float64)
        logits = z["clf_logits"].astype(np.float64)
        target = int(row["target_class"])
        is_success = int(row.get("success", 0)) == 1
        for t in range(len(feats) - 1):
            seg = feats[t + 1] - feats[t]
            if np.linalg.norm(seg) <= 1e-12:
                continue
            grad = logp_field(layer, logits[t], target, fc_weight)
            if is_success:
                success_segments.append(seg)
                grad_success.append(grad)
                class_segments.setdefault(target, []).append(seg)
                class_grads.setdefault(target, []).append(grad)
            else:
                failed_segments.append(seg)
                grad_failed.append(grad)
    return {
        "success_segments": normalize_rows(np.stack(success_segments)) if success_segments else np.empty((0, feats.shape[1])),
        "failed_segments": normalize_rows(np.stack(failed_segments)) if failed_segments else np.empty((0, feats.shape[1])),
        "grad_success": normalize_rows(np.stack(grad_success)) if grad_success else np.empty((0, feats.shape[1])),
        "grad_failed": normalize_rows(np.stack(grad_failed)) if grad_failed else np.empty((0, feats.shape[1])),
        "class_segments": {k: normalize_rows(np.stack(v)) for k, v in class_segments.items()},
        "class_grads": {k: normalize_rows(np.stack(v)) for k, v in class_grads.items()},
    }


def random_baseline(n: int, d: int, reps: int, rng: np.random.Generator):
    rows = []
    for rep in range(reps):
        x = rng.normal(size=(n, d))
        x = normalize_rows(x)
        stats = pca_stats(x)
        stats["rep"] = rep
        rows.append(stats)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="analysis_outputs/pure_af_geometry/random_ga_classifier_trajectories_resnet18_c10_s5/manifest.csv")
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry")
    parser.add_argument("--random-reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["trajectory_features_npz"].notna()].copy()
    fc_weight = models.resnet18(pretrained=True).eval().fc.weight.detach().cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(args.seed)

    rows = []
    random_rows = []
    class_rows = []
    for layer in LAYERS:
        vecs = collect_vectors(manifest, layer, fc_weight)
        datasets = {
            "success_segments": vecs["success_segments"],
            "failed_segments": vecs["failed_segments"],
            "grad_success": vecs["grad_success"],
            "grad_failed": vecs["grad_failed"],
        }
        for name, x in datasets.items():
            stats = pca_stats(x)
            stats.update({"layer": layer, "set": name})
            rows.append(stats)
        n, d = vecs["success_segments"].shape
        rb = random_baseline(n, d, args.random_reps, rng)
        agg = rb.drop(columns=["rep"]).mean(numeric_only=True).to_dict()
        agg.update({"layer": layer, "set": "random_unit_mean"})
        rows.append(agg)
        rb["layer"] = layer
        random_rows.append(rb)

        for cls, x in sorted(vecs["class_segments"].items()):
            stats = pca_stats(x)
            stats.update({"layer": layer, "target_class": cls, "set": "success_segments_by_class"})
            class_rows.append(stats)
        for cls, x in sorted(vecs["class_grads"].items()):
            stats = pca_stats(x)
            stats.update({"layer": layer, "target_class": cls, "set": "grad_success_by_class"})
            class_rows.append(stats)

    summary = pd.DataFrame(rows)
    class_summary = pd.DataFrame(class_rows)
    random_detail = pd.concat(random_rows, ignore_index=True)
    summary_path = out_dir / "flow_tube_dimensionality_summary.csv"
    class_path = out_dir / "flow_tube_dimensionality_by_class.csv"
    random_path = out_dir / "flow_tube_dimensionality_random_reps.csv"
    summary.to_csv(summary_path, index=False)
    class_summary.to_csv(class_path, index=False)
    random_detail.to_csv(random_path, index=False)
    meta = {
        "manifest": args.manifest,
        "layers": LAYERS,
        "random_reps": args.random_reps,
        "outputs": [str(summary_path), str(class_path), str(random_path)],
    }
    (out_dir / "flow_tube_dimensionality_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(summary.sort_values(["layer", "set"]).to_string(index=False))


if __name__ == "__main__":
    main()
