#!/usr/bin/env python3
"""Polar/orbital view of local-step PC vectors from a saved step-vector CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


CLASS_NAMES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}
CLASS_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#e377c2",
    7: "#7f7f7f",
    8: "#bcbd22",
    9: "#17becf",
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["radius"] = np.sqrt(df.dx.to_numpy(float) ** 2 + df.dy.to_numpy(float) ** 2)
    df["theta"] = np.arctan2(df.dy.to_numpy(float), df.dx.to_numpy(float))
    # Non-crossing class-0 vectors stay blue; final crossing vectors use adversarial predicted class.
    df["plot_class"] = df["arrow_class"].astype(int)
    return df[df.radius > 1e-12].copy()


def draw_polar_panel(ax, df: pd.DataFrame, attack: str):
    d = df[df.attack == attack].copy()
    normal = d[d.is_crossing_arrow == 0]
    crossing = d[d.is_crossing_arrow == 1]
    if not normal.empty:
        ax.scatter(
            normal.theta,
            normal.radius,
            s=8,
            color=CLASS_COLORS[0],
            alpha=0.42,
            edgecolors="none",
            label="pre-adversarial class 0",
        )
    for cls, g in crossing.groupby("plot_class"):
        ax.scatter(
            g.theta,
            g.radius,
            s=22,
            color=CLASS_COLORS[int(cls)],
            alpha=0.9,
            edgecolors="black",
            linewidths=0.25,
        )
    ax.set_title(f"{attack.upper()} local-step vectors ({d.run_id.nunique()} runs)")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.grid(alpha=0.22)


def draw_angle_hist(ax, df: pd.DataFrame):
    bins = np.linspace(-np.pi, np.pi, 37)
    for attack, color in [("pgd", "#111827"), ("square", "#64748b")]:
        d = df[(df.attack == attack) & (df.is_crossing_arrow == 0)]
        ax.hist(d.theta, bins=bins, density=True, histtype="step", lw=2.0, color=color, label=attack.upper())
    ax.set_xlabel("local-step angle in PC1/PC2 plane")
    ax.set_ylabel("density")
    ax.set_xlim(-np.pi, np.pi)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)


def draw_origin_vector_panel(ax, df: pd.DataFrame, attack: str):
    d = df[df.attack == attack].copy()
    normal = d[d.is_crossing_arrow == 0]
    crossing = d[d.is_crossing_arrow == 1]
    if not normal.empty:
        ax.plot(
            np.column_stack([np.zeros(len(normal)), normal.dx]).T,
            np.column_stack([np.zeros(len(normal)), normal.dy]).T,
            color=CLASS_COLORS[0],
            alpha=0.32,
            lw=0.7,
        )
    for cls, g in crossing.groupby("plot_class"):
        ax.quiver(
            np.zeros(len(g)),
            np.zeros(len(g)),
            g.dx,
            g.dy,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=CLASS_COLORS[int(cls)],
            alpha=0.8,
            width=0.002,
        )
    ax.scatter([0], [0], color="black", s=30, zorder=5)
    ax.axhline(0, color="black", lw=0.8, alpha=0.18)
    ax.axvline(0, color="black", lw=0.8, alpha=0.18)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.16)
    ax.set_title(f"{attack.upper()} local-step vectors from common origin")
    ax.set_xlabel("local-step PC1")
    ax.set_ylabel("local-step PC2")


def set_origin_limits(axes, df: pd.DataFrame):
    x = df.dx.to_numpy(float)
    y = df.dy.to_numpy(float)
    lim = max(float(np.nanmax(np.abs(x))), float(np.nanmax(np.abs(y))), 1e-6)
    lim *= 1.08
    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)


def draw_origin_vector_figure(df: pd.DataFrame, out_dir: Path, stem: str):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2), constrained_layout=True)
    draw_origin_vector_panel(axes[0], df, "pgd")
    draw_origin_vector_panel(axes[1], df, "square")
    set_origin_limits(axes, df)
    handles = [Line2D([0], [0], color=CLASS_COLORS[i], lw=4, label=f"{i}: {CLASS_NAMES[i]}") for i in range(10)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=True)
    fig.suptitle("Class-0 local transport vectors drawn from a common origin", fontsize=12)
    fig.savefig(out_dir / f"{stem}_origin_vectors.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_origin_vectors.pdf", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare(pd.read_csv(args.step_vectors_csv))
    stem = Path(args.step_vectors_csv).stem.replace("_step_vectors", "_orbital_pc1_pc2")

    df.to_csv(out_dir / f"{stem}_points.csv", index=False)

    fig = plt.figure(figsize=(14.2, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 0.95])
    ax0 = fig.add_subplot(gs[0, 0], projection="polar")
    ax1 = fig.add_subplot(gs[0, 1], projection="polar")
    ax2 = fig.add_subplot(gs[0, 2])
    draw_polar_panel(ax0, df, "pgd")
    draw_polar_panel(ax1, df, "square")
    draw_angle_hist(ax2, df)

    handles = [Line2D([0], [0], marker="o", linestyle="", color=CLASS_COLORS[i], label=f"{i}: {CLASS_NAMES[i]}") for i in range(10)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=True)
    fig.suptitle("Orbital view of class-0 local transport vectors in the PC1/PC2 plane", fontsize=12)
    fig.savefig(out_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    draw_origin_vector_figure(df, out_dir, stem)

    summary = (
        df.groupby(["attack", "is_crossing_arrow"])
        .agg(n=("radius", "size"), mean_radius=("radius", "mean"), median_radius=("radius", "median"))
        .reset_index()
    )
    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)
    print(f"[SAVED] {out_dir / (stem + '.png')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step-vectors-csv", required=True)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/pgd_square_class_colored_flow")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
