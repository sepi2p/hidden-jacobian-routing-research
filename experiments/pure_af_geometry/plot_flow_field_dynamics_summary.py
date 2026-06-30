#!/usr/bin/env python3
"""Plot aggregate flow-field dynamics summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {"pgd": "#2563eb", "square": "#dc2626", "ga": "#16a34a"}
LABELS = {"pgd": "PGD", "square": "Square", "ga": "GA"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary_csv)
    attacks = ["pgd", "square", "ga"]
    metrics = [
        ("frac_convergent", "Fraction convergent cells"),
        ("mean_divergence", "Mean divergence"),
        ("mean_abs_curl", "Mean |curl|"),
        ("mean_speed", "Mean projected speed"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        data = [df[df.attack == a][metric].to_numpy() for a in attacks]
        bp = ax.boxplot(data, labels=[LABELS[a] for a in attacks], patch_artist=True, showmeans=True)
        for patch, a in zip(bp["boxes"], attacks):
            patch.set_facecolor(COLORS[a])
            patch.set_alpha(0.24)
            patch.set_edgecolor(COLORS[a])
        for med in bp["medians"]:
            med.set_color("#111827")
            med.set_linewidth(1.4)
        for i, a in enumerate(attacks, start=1):
            y = df[df.attack == a][metric]
            x = [i] * len(y)
            ax.scatter(x, y, color=COLORS[a], alpha=0.55, s=22, zorder=3)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18)
        if metric == "mean_divergence":
            ax.axhline(0, color="black", lw=0.9, alpha=0.5)
            ax.text(0.55, 0.02, "positive = divergent", transform=ax.transAxes, fontsize=8)
            ax.text(0.55, 0.08, "negative = convergent", transform=ax.transAxes, fontsize=8)
    fig.suptitle("Empirical flow-field dynamics across CIFAR models and layers", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "flow_field_dynamics_summary_all_models.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / "flow_field_dynamics_summary_all_models.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_dir / 'flow_field_dynamics_summary_all_models.png'}")


if __name__ == "__main__":
    main()
