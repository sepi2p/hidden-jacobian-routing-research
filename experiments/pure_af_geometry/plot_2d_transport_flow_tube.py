#!/usr/bin/env python3
"""Plot adversarial trajectories as mean flow tubes in 2D transport coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd


ATTACK_COLORS = {"pgd": "#2563eb", "square": "#dc2626", "ga": "#16a34a"}
ATTACK_LABELS = {"pgd": "PGD", "square": "Square", "ga": "GA pure"}


def resample_run(g: pd.DataFrame, n_points: int) -> pd.DataFrame:
    g = g.sort_values("normalized_progress")
    x = g.normalized_progress.to_numpy(float)
    t = np.linspace(0.0, 1.0, n_points)
    if len(np.unique(x)) < 2:
        pc1 = np.full(n_points, g.pc1_coeff.iloc[-1])
        pc2 = np.full(n_points, g.pc2_coeff.iloc[-1])
    else:
        pc1 = np.interp(t, x, g.pc1_coeff.to_numpy(float))
        pc2 = np.interp(t, x, g.pc2_coeff.to_numpy(float))
    return pd.DataFrame({"progress": t, "pc1": pc1, "pc2": pc2})


def covariance_ellipse(points: np.ndarray, n_std: float = 1.0):
    if len(points) < 3:
        return None
    cov = np.cov(points.T)
    if not np.isfinite(cov).all():
        return None
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-10)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    return width, height, angle


def build_resampled(df: pd.DataFrame, n_points: int) -> pd.DataFrame:
    rows = []
    for (attack, run_id), g in df.groupby(["attack", "run_id"], sort=False):
        rr = resample_run(g, n_points)
        rr["attack"] = attack
        rr["run_id"] = run_id
        rows.append(rr)
    return pd.concat(rows, ignore_index=True)


def draw_attack(ax, rr: pd.DataFrame, attack: str, max_faint: int, seed: int):
    color = ATTACK_COLORS.get(attack, "#111827")
    sub = rr[rr.attack == attack]
    if sub.empty:
        return
    rng = np.random.default_rng(seed)
    runs = sub.run_id.drop_duplicates().to_numpy()
    if len(runs) > max_faint:
        runs = rng.choice(runs, size=max_faint, replace=False)
    for run_id in runs:
        g = sub[sub.run_id == run_id].sort_values("progress")
        ax.plot(g.pc1, g.pc2, color=color, lw=0.9, alpha=0.13, zorder=1)

    mean = sub.groupby("progress", as_index=False)[["pc1", "pc2"]].mean().sort_values("progress")
    ax.plot(mean.pc1, mean.pc2, color=color, lw=3.0, label=ATTACK_LABELS.get(attack, attack), zorder=4)
    ax.scatter(mean.pc1.iloc[0], mean.pc2.iloc[0], color=color, marker="o", s=42, zorder=5)
    ax.scatter(mean.pc1.iloc[-1], mean.pc2.iloc[-1], color=color, marker="^", s=72, zorder=5)

    # Direction arrows along the actual mean flow.
    idx = np.linspace(0, len(mean) - 2, 8, dtype=int)
    starts = mean.iloc[idx]
    ends = mean.iloc[idx + 1]
    ax.quiver(
        starts.pc1,
        starts.pc2,
        ends.pc1.to_numpy() - starts.pc1.to_numpy(),
        ends.pc2.to_numpy() - starts.pc2.to_numpy(),
        color=color,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.006,
        alpha=0.95,
        zorder=6,
    )

    # Ellipses at a few progress slices show the trajectory tube width.
    ellipse_progress = np.linspace(0.15, 0.95, 5)
    for p in ellipse_progress:
        nearest = mean.progress.iloc[np.argmin(np.abs(mean.progress.to_numpy() - p))]
        pts = sub[np.isclose(sub.progress, nearest)][["pc1", "pc2"]].to_numpy(float)
        ell = covariance_ellipse(pts, n_std=1.0)
        if ell is None:
            continue
        width, height, angle = ell
        center = pts.mean(axis=0)
        patch = Ellipse(center, width, height, angle=angle, facecolor=color, edgecolor=color, alpha=0.08, lw=1.0, zorder=2)
        ax.add_patch(patch)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.timeseries_csv)
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    df = df[
        (df.model == args.model)
        & (df.layer_group == args.layer_group)
        & (df.attack.isin(attacks))
        & (df.final_success == 1)
    ].dropna(subset=["pc1_coeff", "pc2_coeff", "normalized_progress"])
    rr = build_resampled(df, args.n_points)
    rr.to_csv(out_dir / f"flow_tube_resampled_{args.model}_{args.layer_group}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for attack in attacks:
        draw_attack(ax, rr, attack, args.max_faint, args.seed)
    ax.set_title(f"Recorded adversarial flow tube: {args.model} / {args.layer_group}")
    ax.set_xlabel("transport PC1")
    ax.set_ylabel("transport PC2")
    ax.axhline(0, color="black", lw=0.8, alpha=0.22)
    ax.axvline(0, color="black", lw=0.8, alpha=0.22)
    ax.grid(alpha=0.2)
    ax.legend(frameon=True)
    fig.tight_layout()
    stem = f"transport_flow_tube_{args.model}_{args.layer_group}"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {png}")
    print(f"[SAVED] {pdf}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeseries-csv", default="analysis_outputs/pure_af_geometry/cifar_attack_transport_decomposition/attack_transport_projection_timeseries.csv")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_attack_transport_decomposition/flow_tubes")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer-group", default="penultimate", choices=["hidden", "penultimate", "logits"])
    p.add_argument("--attacks", default="pgd,square,ga")
    p.add_argument("--n-points", type=int, default=45)
    p.add_argument("--max-faint", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
