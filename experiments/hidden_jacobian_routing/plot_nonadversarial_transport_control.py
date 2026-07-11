#!/usr/bin/env python3
"""Plot adversarial and non-adversarial optimization motion in one transport space."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def load_group(meta: pd.DataFrame, npz, model: str, source: str, objective: str, layer_group: str):
    g = meta[
        (meta["model"] == model)
        & (meta["source"] == source)
        & (meta["objective"] == objective)
        & (meta["layer_group"] == layer_group)
    ].copy()
    if g.empty:
        raise RuntimeError(f"No rows for {model}/{source}/{objective}/{layer_group}")
    key = g["vector_key"].iloc[0]
    arr = npz[f"vectors__{key}"]
    x = normalize_rows(arr[g["vector_idx"].to_numpy(int)])
    return g, x


def fit_pca2(x: np.ndarray):
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return mean, vt[:2]


def sample_project(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, n: int, seed: int):
    rng = np.random.default_rng(seed)
    if len(x) > n:
        idx = rng.choice(len(x), size=n, replace=False)
        x = x[idx]
    return (x - mean) @ basis.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("analysis_outputs/hidden_jacobian_routing/nonadversarial_optimization_controls_c200_s40"))
    ap.add_argument("--output", type=Path, default=Path("figures/nonadversarial_transport_control"))
    ap.add_argument("--model", default="bbb_resnet50")
    ap.add_argument("--layer-group", default="hidden")
    ap.add_argument("--sample", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_csv(args.input_dir / "control_segment_metadata.csv")
    npz = np.load(args.input_dir / "control_segment_vectors.npz", allow_pickle=False)
    adv_meta, adv_x = load_group(meta, npz, args.model, "adversarial", "pgd", args.layer_group)
    adv_x = adv_x[adv_meta["final_adversarial_success"].to_numpy(int) == 1]
    mean, basis = fit_pca2(adv_x)
    adv_proj = sample_project(adv_x, mean, basis, args.sample, args.seed)

    objectives = [
        ("same_class_feature_match", "same-class feature match"),
        ("different_image_feature_match", "different-image feature match"),
        ("activation_max", "activation maximization"),
        ("random_feature_direction", "random feature direction"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.9), constrained_layout=True)
    axes = axes.ravel()
    adv_color = "#2563eb"
    ctrl_color = "#f97316"
    for ax, (objective, title) in zip(axes, objectives):
        _ctrl_meta, ctrl_x = load_group(meta, npz, args.model, "control", objective, args.layer_group)
        ctrl_proj = sample_project(ctrl_x, mean, basis, args.sample, args.seed + 17)
        ax.scatter(ctrl_proj[:, 0], ctrl_proj[:, 1], s=5, alpha=0.20, color=ctrl_color, linewidths=0, label="generic optimization")
        ax.scatter(adv_proj[:, 0], adv_proj[:, 1], s=5, alpha=0.22, color=adv_color, linewidths=0, label="adversarial PGD")
        ax.axhline(0, color="#e5e7eb", lw=0.8, zorder=0)
        ax.axvline(0, color="#e5e7eb", lw=0.8, zorder=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("adversarial PC1")
        ax.set_ylabel("adversarial PC2")
        ax.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
