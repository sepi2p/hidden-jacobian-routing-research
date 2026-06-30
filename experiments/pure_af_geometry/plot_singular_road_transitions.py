#!/usr/bin/env python3
"""Plot class transitions along continued hidden-Jacobian singular roads."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


CIFAR10_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_COLORS = [
    "#4C78A8",  # airplane
    "#F58518",  # automobile
    "#54A24B",  # bird
    "#E45756",  # cat
    "#72B7B2",  # deer
    "#B279A2",  # dog
    "#EECA3B",  # frog
    "#FF9DA6",  # horse
    "#9D755D",  # ship
    "#BAB0AC",  # truck
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", default="singular_road_continuation")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    df = df.sort_values(["direction", "image_id", "step"])

    groups = list(df.groupby(["direction", "image_id"], sort=True))
    max_step = int(df.step.max())
    road_labels = [f"{direction[0].upper()} img {image_id}" for (direction, image_id), _ in groups]
    pred_grid = np.full((len(groups), max_step + 1), np.nan)
    margin_grid = np.full_like(pred_grid, np.nan, dtype=float)
    sigma_grid = np.full_like(pred_grid, np.nan, dtype=float)
    first_success = []
    return_steps = []

    for i, ((direction, image_id), g) in enumerate(groups):
        g = g.sort_values("step")
        steps = g.step.to_numpy(dtype=int)
        preds = g.pred.to_numpy(dtype=int)
        pred0 = int(g.pred0.iloc[0])
        pred_grid[i, steps] = preds
        margin_grid[i, steps] = g.margin.to_numpy(dtype=float)
        sigma = g.sigma1_est.to_numpy(dtype=float)
        sigma_grid[i, steps] = sigma / max(float(sigma[0]), 1e-12)

        success_mask = preds != pred0
        fs = steps[np.argmax(success_mask)] if success_mask.any() else None
        first_success.append(fs)
        if fs is None:
            return_steps.append([])
        else:
            after = steps[(steps > fs) & (preds == pred0)]
            return_steps.append(after.tolist())

    cmap = ListedColormap(CIFAR10_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)

    fig = plt.figure(figsize=(11.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)

    im = ax0.imshow(pred_grid, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax0.set_title("Budget-Constrained Continuation of Hidden-Jacobian Roads", fontsize=12)
    ax0.set_ylabel("Road")
    ax0.set_yticks(np.arange(len(road_labels)))
    ax0.set_yticklabels(road_labels, fontsize=8)
    ax0.set_xlim(-0.5, max_step + 0.5)
    ax0.set_xticks(np.arange(0, max_step + 1, max(1, max_step // 8)))

    for i, fs in enumerate(first_success):
        if fs is not None:
            ax0.scatter(fs, i, marker="|", s=120, color="black", linewidth=1.2)
        for rs in return_steps[i]:
            ax0.scatter(rs, i, marker=".", s=9, color="white", edgecolor="black", linewidth=0.25)

    legend_handles = [Patch(facecolor=CIFAR10_COLORS[i], label=f"{i}: {name}") for i, name in enumerate(CIFAR10_NAMES)]
    ax0.legend(
        handles=legend_handles,
        ncol=5,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        frameon=False,
    )
    ax0.text(
        1.01,
        0.5,
        "black tick: first non-original class\nwhite dot: return to original class",
        transform=ax0.transAxes,
        va="center",
        fontsize=8,
    )

    steps = np.arange(max_step + 1)
    for i in range(len(groups)):
        color = "#4C78A8" if road_labels[i].startswith("F") else "#72B7B2"
        ax1.plot(steps, margin_grid[i], color=color, alpha=0.35, lw=1.0)
    ax1.plot(steps, np.nanmean(margin_grid, axis=0), color="black", lw=2.0, label="mean")
    ax1.axhline(0, color="#D62728", lw=1.0, ls="--")
    ax1.set_ylabel("true-class margin")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")

    for i in range(len(groups)):
        color = "#4C78A8" if road_labels[i].startswith("F") else "#72B7B2"
        ax2.plot(steps, sigma_grid[i], color=color, alpha=0.35, lw=1.0)
    ax2.plot(steps, np.nanmean(sigma_grid, axis=0), color="black", lw=2.0, label="mean")
    ax2.axhline(0.25, color="#D62728", lw=1.0, ls="--", label="25% initial")
    ax2.set_xlabel("road-continuation step")
    ax2.set_ylabel(r"$\sigma_1(J_h)$ / initial")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")

    png = out / f"{args.prefix}_class_transitions.png"
    pdf = out / f"{args.prefix}_class_transitions.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)

    print(f"saved {png}")
    print(f"saved {pdf}")


if __name__ == "__main__":
    main()
