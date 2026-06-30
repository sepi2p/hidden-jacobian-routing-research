#!/usr/bin/env python3
"""Plot one-class pure/adversarial flow with shared axes and safer overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd


COLORS = {"pgd": "#2563eb", "square": "#dc2626", "ga": "#16a34a"}
FLOW_COLORS = {"pgd": "#93c5fd", "square": "#fca5a5", "ga": "#86efac"}
LABELS = {"pgd": "PGD", "square": "Square", "ga": "GA pure"}


def add_step_vectors(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model", "attack", "layer_group", "layer", "run_id"]
    for _, g in df.sort_values(keys + ["step"]).groupby(keys, sort=False):
        g = g.copy()
        g["dx"] = g["pc1_coeff"].shift(-1) - g["pc1_coeff"]
        g["dy"] = g["pc2_coeff"].shift(-1) - g["pc2_coeff"]
        rows.append(g.iloc[:-1])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna(
        subset=["pc1_coeff", "pc2_coeff", "dx", "dy"]
    )


def limits(df: pd.DataFrame):
    x = df.pc1_coeff.to_numpy(float)
    y = df.pc2_coeff.to_numpy(float)
    xlo, xhi = np.nanpercentile(x, [1, 99])
    ylo, yhi = np.nanpercentile(y, [1, 99])
    px = 0.08 * max(xhi - xlo, 1e-6)
    py = 0.08 * max(yhi - ylo, 1e-6)
    return (xlo - px, xhi + px), (ylo - py, yhi + py)


def smooth_field(vecs: pd.DataFrame, xlim, ylim, grid_n: int, bandwidth_scale: float):
    x = vecs.pc1_coeff.to_numpy(float)
    y = vecs.pc2_coeff.to_numpy(float)
    dx = vecs.dx.to_numpy(float)
    dy = vecs.dy.to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & np.isfinite(dy)
    x, y, dx, dy = x[keep], y[keep], dx[keep], dy[keep]
    if len(x) < 10:
        return None
    gx = np.linspace(xlim[0], xlim[1], grid_n)
    gy = np.linspace(ylim[0], ylim[1], grid_n)
    X, Y = np.meshgrid(gx, gy)
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0], 1e-6)
    sigma = bandwidth_scale * span
    points = np.column_stack([x, y])
    vectors = np.column_stack([dx, dy])
    flat = np.column_stack([X.ravel(), Y.ravel()])
    u = np.zeros(len(flat))
    v = np.zeros(len(flat))
    wsum = np.zeros(len(flat))
    for start in range(0, len(flat), 512):
        q = flat[start : start + 512]
        dist2 = ((q[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        w = np.exp(-0.5 * dist2 / (sigma**2))
        ws = w.sum(axis=1)
        u[start : start + len(q)] = (w @ vectors[:, 0]) / np.maximum(ws, 1e-12)
        v[start : start + len(q)] = (w @ vectors[:, 1]) / np.maximum(ws, 1e-12)
        wsum[start : start + len(q)] = ws
    U = u.reshape(X.shape)
    V = v.reshape(Y.shape)
    W = wsum.reshape(X.shape)
    mask = W < np.nanpercentile(W, 25)
    return X, Y, np.ma.array(U, mask=mask), np.ma.array(V, mask=mask), W


def mean_path(df: pd.DataFrame, attack: str) -> pd.DataFrame:
    rows = []
    for run_id, g in df[df.attack == attack].groupby("run_id", sort=False):
        g = g.sort_values("normalized_progress")
        t0 = g.normalized_progress.to_numpy(float)
        if len(np.unique(t0)) < 2:
            continue
        t = np.linspace(0, 1, 60)
        rows.append(
            pd.DataFrame(
                {
                    "run_id": run_id,
                    "t": t,
                    "pc1_coeff": np.interp(t, t0, g.pc1_coeff.to_numpy(float)),
                    "pc2_coeff": np.interp(t, t0, g.pc2_coeff.to_numpy(float)),
                }
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).groupby("t", as_index=False)[["pc1_coeff", "pc2_coeff"]].mean()


def draw_attack(ax, df, vecs, attack, xlim, ylim, args):
    color = COLORS[attack]
    flow_color = FLOW_COLORS[attack]
    d = df[(df.attack == attack) & (df.final_success == 1)]
    v = vecs[(vecs.attack == attack) & (vecs.final_success == 1)]
    field = smooth_field(v, xlim, ylim, args.grid_n, args.bandwidth_scale)
    if field is not None:
        X, Y, U, V, W = field
        ax.streamplot(
            X,
            Y,
            U,
            V,
            color=flow_color,
            density=args.stream_density,
            linewidth=1.15,
            arrowsize=1.1,
        )
    for _, g in d.groupby("run_id"):
        ax.plot(g.pc1_coeff, g.pc2_coeff, color=color, alpha=0.08, lw=0.8)
    mp = mean_path(d, attack)
    if not mp.empty:
        ax.plot(mp.pc1_coeff, mp.pc2_coeff, color=color, lw=3, label=LABELS[attack])
        ax.scatter(mp.pc1_coeff.iloc[0], mp.pc2_coeff.iloc[0], color=color, marker="o", s=35)
        ax.scatter(mp.pc1_coeff.iloc[-1], mp.pc2_coeff.iloc[-1], color=color, marker="^", s=55)
    ax.set_title(LABELS[attack])
    ax.legend(loc="best", fontsize=8)


def draw_overlay(ax, df, attacks, xlim, ylim):
    for attack in attacks:
        color = COLORS[attack]
        flow_color = to_rgba(FLOW_COLORS[attack], 0.55)
        d = df[(df.attack == attack) & (df.final_success == 1)]
        v = add_step_vectors(d)
        field = smooth_field(v, xlim, ylim, 58, 0.07)
        if field is not None:
            X, Y, U, V, _W = field
            ax.streamplot(
                X,
                Y,
                U,
                V,
                color=flow_color,
                density=0.75,
                linewidth=0.85,
                arrowsize=0.85,
            )
        for _, g in d.groupby("run_id"):
            ax.plot(g.pc1_coeff, g.pc2_coeff, color=color, alpha=0.035, lw=0.7)
        mp = mean_path(d, attack)
        if not mp.empty:
            ax.plot(mp.pc1_coeff, mp.pc2_coeff, color=color, lw=3, label=LABELS[attack])
            ax.scatter(mp.pc1_coeff.iloc[0], mp.pc2_coeff.iloc[0], color=color, marker="o", s=35)
            ax.scatter(mp.pc1_coeff.iloc[-1], mp.pc2_coeff.iloc[-1], color=color, marker="^", s=55)
            # A few explicit arrows on the mean path make direction readable even
            # when streamlines overlap in the shared panel.
            idxs = np.linspace(4, len(mp) - 6, 3).astype(int) if len(mp) > 12 else []
            for i in idxs:
                x0, y0 = float(mp.pc1_coeff.iloc[i]), float(mp.pc2_coeff.iloc[i])
                x1, y1 = float(mp.pc1_coeff.iloc[i + 3]), float(mp.pc2_coeff.iloc[i + 3])
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.7, shrinkA=0, shrinkB=0),
                )
    ax.set_title("Overlay: separate colored fields, no mixed field")
    ax.legend(loc="best", fontsize=8)


def format_ax(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0, color="black", lw=0.8, alpha=0.2)
    ax.axvline(0, color="black", lw=0.8, alpha=0.2)
    ax.grid(alpha=0.16)
    ax.set_xlabel("transport PC1")
    ax.set_ylabel("transport PC2")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeseries-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer-group", default="penultimate")
    p.add_argument("--attacks", default="pgd,square,ga")
    p.add_argument("--grid-n", type=int, default=58)
    p.add_argument("--bandwidth-scale", type=float, default=0.07)
    p.add_argument("--stream-density", type=float, default=1.1)
    args = p.parse_args()

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.timeseries_csv, low_memory=False)
    df = df[(df.model == args.model) & (df.layer_group == args.layer_group) & df.attack.isin(attacks)].copy()
    df = df.dropna(subset=["run_id", "pc1_coeff", "pc2_coeff", "normalized_progress", "step"])
    d_success = df[df.final_success == 1]
    xlim, ylim = limits(d_success)
    vecs = add_step_vectors(df)
    panels = attacks + ["overlay"]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.0), squeeze=False)
    for ax, panel in zip(axes.ravel(), panels):
        if panel == "overlay":
            draw_overlay(ax, df, attacks, xlim, ylim)
        else:
            draw_attack(ax, df, vecs, panel, xlim, ylim, args)
        format_ax(ax, xlim, ylim)
    fig.suptitle(
        f"Class-specific pure/adversarial flow in shared axes: {args.model} / {args.layer_group}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    stem = f"shared_axes_flow_{args.model}_{args.layer_group}"
    fig.savefig(out_dir / f"{stem}.png", dpi=250, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_dir / (stem + '.png')}")


if __name__ == "__main__":
    main()
