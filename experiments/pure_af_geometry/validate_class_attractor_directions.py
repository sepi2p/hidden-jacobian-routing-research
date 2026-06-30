#!/usr/bin/env python3
"""Validate class-attractor directions from random-noise GA trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_LAYERS = [
    "clf_conv1",
    "clf_layer1",
    "clf_layer2",
    "clf_layer3",
    "clf_layer4",
    "clf_avgpool",
    "clf_logits",
]
SECONDARY_LAYERS = ["afvf_sem0", "afvf_sem1", "afvf_sem4", "afvf_vf0", "afvf_vf1", "afvf_vf4"]
DEFAULT_LAYERS = PRIMARY_LAYERS + SECONDARY_LAYERS


def cos(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.dot(a, b) / den)


def topk_rank(scores: dict[int, float], correct_class: int) -> int:
    ordered = sorted(scores.items(), key=lambda kv: (-np.nan_to_num(kv[1], nan=-np.inf), kv[0]))
    for rank, (cls, _score) in enumerate(ordered, start=1):
        if cls == correct_class:
            return rank
    return len(ordered) + 1


def parse_layers(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def available_layers(npz: np.lib.npyio.NpzFile) -> list[str]:
    return [k for k in DEFAULT_LAYERS if k in npz.files]


def load_manifest(path: str, success_only: bool) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    manifest = manifest[manifest["trajectory_features_npz"].notna()].copy()
    if success_only and "success" in manifest.columns:
        manifest = manifest[manifest["success"].astype(int) == 1].copy()
    manifest["target_class"] = manifest["target_class"].astype(int)
    manifest["seed"] = manifest["seed"].astype(int)
    return manifest.reset_index(drop=True)


def endpoint_direction(feats: np.lib.npyio.NpzFile, layer: str) -> np.ndarray:
    x = feats[layer].astype(np.float64)
    return x[-1] - x[0]


def checkpoint_delta(feats: np.lib.npyio.NpzFile, layer: str, fraction: float) -> tuple[np.ndarray, int, int]:
    generations = feats["generation"].astype(np.int64)
    final_generation = int(generations[-1])
    target_generation = int(round(float(final_generation) * fraction))
    idx = int(np.argmin(np.abs(generations - target_generation)))
    x = feats[layer].astype(np.float64)
    return x[idx] - x[0], int(generations[idx]), idx


def mean_direction(vectors: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(vectors, axis=0), axis=0)


def class_means_for_layer(
    rows: pd.DataFrame,
    directions: dict[tuple[int, str], np.ndarray],
    layer: str,
    classes: list[int],
) -> dict[int, np.ndarray]:
    means = {}
    for cls in classes:
        run_indices = rows.index[rows["target_class"] == cls].tolist()
        means[cls] = mean_direction([directions[(idx, layer)] for idx in run_indices])
    return means


def class_means_minus_one(
    rows: pd.DataFrame,
    directions: dict[tuple[int, str], np.ndarray],
    layer: str,
    classes: list[int],
    heldout_idx: int,
) -> dict[int, np.ndarray]:
    means = {}
    heldout_class = int(rows.loc[heldout_idx, "target_class"])
    for cls in classes:
        run_indices = rows.index[rows["target_class"] == cls].tolist()
        if cls == heldout_class:
            run_indices = [idx for idx in run_indices if idx != heldout_idx]
        if not run_indices:
            continue
        means[cls] = mean_direction([directions[(idx, layer)] for idx in run_indices])
    return means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default=None, help="Comma-separated layer list. Defaults to all available layers.")
    parser.add_argument("--fractions", default="0.25,0.5,0.75")
    parser.add_argument("--success-only", action="store_true", help="Use only strict P(target)>=0.9999 successful trajectories.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest, args.success_only)
    if manifest.empty:
        raise SystemExit("No trajectories found after filtering.")

    run_cache: dict[int, np.lib.npyio.NpzFile] = {}
    layers: list[str] | None = parse_layers(args.layers)
    for idx, row in manifest.iterrows():
        feats = np.load(row["trajectory_features_npz"], allow_pickle=True)
        run_cache[idx] = feats
        cur_layers = available_layers(feats)
        layers = cur_layers if layers is None else [x for x in layers if x in cur_layers]
    layers = layers or []
    if not layers:
        raise SystemExit("No requested layers were available in the trajectory feature archives.")

    classes = sorted(int(x) for x in manifest["target_class"].unique())
    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    directions: dict[tuple[int, str], np.ndarray] = {}
    for idx, feats in run_cache.items():
        for layer in layers:
            directions[(idx, layer)] = endpoint_direction(feats, layer)

    mean_npz: dict[str, np.ndarray] = {"classes": np.asarray(classes, dtype=np.int64)}
    norm_rows = []
    class_means_by_layer: dict[str, dict[int, np.ndarray]] = {}
    for layer in layers:
        class_means = class_means_for_layer(manifest, directions, layer, classes)
        class_means_by_layer[layer] = class_means
        packed = np.stack([class_means[cls] for cls in classes], axis=0)
        mean_npz[layer] = packed.astype(np.float32)
        for cls, vec in class_means.items():
            norm_rows.append({
                "layer": layer,
                "target_class": cls,
                "n_runs": int((manifest["target_class"] == cls).sum()),
                "mean_direction_l2": float(np.linalg.norm(vec)),
            })
    np.savez_compressed(out_dir / "class_mean_directions.npz", **mean_npz)
    pd.DataFrame(norm_rows).to_csv(out_dir / "class_direction_norms.csv", index=False)

    loo_rows = []
    for idx, row in manifest.iterrows():
        correct = int(row["target_class"])
        for layer in layers:
            means = class_means_minus_one(manifest, directions, layer, classes, idx)
            if correct not in means:
                continue
            vec = directions[(idx, layer)]
            scores = {cls: cos(vec, mu) for cls, mu in means.items()}
            competitor_scores = [score for cls, score in scores.items() if cls != correct]
            best_comp = float(np.nanmax(competitor_scores)) if competitor_scores else float("nan")
            rank = topk_rank(scores, correct)
            loo_rows.append({
                "run_name": row["run_name"],
                "target_class": correct,
                "seed": int(row["seed"]),
                "success": int(row.get("success", 0)),
                "layer": layer,
                "same_class_cos": scores[correct],
                "best_competing_class_cos": best_comp,
                "cosine_margin": scores[correct] - best_comp,
                "correct_rank": rank,
                "top1": int(rank == 1),
                "top3": int(rank <= 3),
            })

    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(out_dir / "leave_one_seed_class_identification.csv", index=False)
    loo_summary = loo_df.groupby("layer").agg(
        n=("run_name", "size"),
        top1_acc=("top1", "mean"),
        top3_acc=("top3", "mean"),
        mean_same_class_cos=("same_class_cos", "mean"),
        mean_best_competing_cos=("best_competing_class_cos", "mean"),
        mean_cosine_margin=("cosine_margin", "mean"),
        median_correct_rank=("correct_rank", "median"),
    ).reset_index()
    loo_summary.to_csv(out_dir / "leave_one_seed_summary.csv", index=False)

    early_rows = []
    for idx, row in manifest.iterrows():
        correct = int(row["target_class"])
        feats = run_cache[idx]
        for layer in layers:
            means = class_means_minus_one(manifest, directions, layer, classes, idx)
            if correct not in means:
                continue
            for fraction in fractions:
                vec, used_generation, used_index = checkpoint_delta(feats, layer, fraction)
                scores = {cls: cos(vec, mu) for cls, mu in means.items()}
                rank = topk_rank(scores, correct)
                ordered = sorted(scores.items(), key=lambda kv: (-np.nan_to_num(kv[1], nan=-np.inf), kv[0]))
                pred = int(ordered[0][0])
                early_rows.append({
                    "run_name": row["run_name"],
                    "target_class": correct,
                    "seed": int(row["seed"]),
                    "success": int(row.get("success", 0)),
                    "layer": layer,
                    "fraction": fraction,
                    "used_generation": used_generation,
                    "used_index": used_index,
                    "predicted_class": pred,
                    "same_class_cos": scores[correct],
                    "best_cos": float(ordered[0][1]),
                    "correct_rank": rank,
                    "top1": int(rank == 1),
                    "top3": int(rank <= 3),
                })

    early_df = pd.DataFrame(early_rows)
    early_df.to_csv(out_dir / "early_trajectory_classification.csv", index=False)
    early_summary = early_df.groupby(["layer", "fraction"]).agg(
        n=("run_name", "size"),
        top1_acc=("top1", "mean"),
        top3_acc=("top3", "mean"),
        mean_same_class_cos=("same_class_cos", "mean"),
        median_correct_rank=("correct_rank", "median"),
    ).reset_index()
    early_summary.to_csv(out_dir / "early_trajectory_summary.csv", index=False)

    space_rows = []
    for layer, group in loo_summary.groupby("layer"):
        if layer.startswith("clf_"):
            space = "classifier"
        elif layer.startswith("afvf_sem"):
            space = "af"
        elif layer.startswith("afvf_vf"):
            space = "vf"
        else:
            space = "other"
        row = group.iloc[0].to_dict()
        row["space"] = space
        space_rows.append(row)
    pd.DataFrame(space_rows).sort_values(["top1_acc", "mean_cosine_margin"], ascending=False).to_csv(
        out_dir / "space_comparison_summary.csv", index=False
    )

    metadata = {
        "manifest": args.manifest,
        "output_dir": str(out_dir),
        "runs": int(len(manifest)),
        "classes": classes,
        "layers": layers,
        "fractions": fractions,
        "success_only": bool(args.success_only),
        "outputs": {
            "class_mean_directions": str(out_dir / "class_mean_directions.npz"),
            "class_direction_norms": str(out_dir / "class_direction_norms.csv"),
            "leave_one_seed_class_identification": str(out_dir / "leave_one_seed_class_identification.csv"),
            "leave_one_seed_summary": str(out_dir / "leave_one_seed_summary.csv"),
            "early_trajectory_classification": str(out_dir / "early_trajectory_classification.csv"),
            "early_trajectory_summary": str(out_dir / "early_trajectory_summary.csv"),
            "space_comparison_summary": str(out_dir / "space_comparison_summary.csv"),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
