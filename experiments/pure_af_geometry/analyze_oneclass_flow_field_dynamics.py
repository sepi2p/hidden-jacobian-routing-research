#!/usr/bin/env python3
"""Analyze convergence/divergence/curl in one-class 2D transport flow fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"pgd": "#2563eb", "square": "#dc2626", "ga": "#16a34a"}
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


def field_limits(df: pd.DataFrame):
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
    u = np.zeros(len(flat), dtype=np.float64)
    v = np.zeros(len(flat), dtype=np.float64)
    wsum = np.zeros(len(flat), dtype=np.float64)
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
    return X, Y, U, V, W


def derivatives(X, Y, U, V):
    x = X[0, :]
    y = Y[:, 0]
    dU_dy, dU_dx = np.gradient(U, y, x, edge_order=1)
    dV_dy, dV_dx = np.gradient(V, y, x, edge_order=1)
    div = dU_dx + dV_dy
    curl = dV_dx - dU_dy
    speed = np.sqrt(np.maximum(U * U + V * V, 0.0))
    return div, curl, speed


def summarize_grid(model, layer_group, attack, X, Y, U, V, W, density_percentile):
    div, curl, speed = derivatives(X, Y, U, V)
    dense = W >= np.nanpercentile(W, density_percentile)
    if not np.any(dense):
        dense = np.isfinite(W)
    vals = {
        "model": model,
        "layer_group": layer_group,
        "attack": attack,
        "n_dense_cells": int(np.sum(dense)),
        "density_percentile": float(density_percentile),
        "mean_speed": float(np.nanmean(speed[dense])),
        "median_speed": float(np.nanmedian(speed[dense])),
        "mean_divergence": float(np.nanmean(div[dense])),
        "median_divergence": float(np.nanmedian(div[dense])),
        "min_divergence": float(np.nanmin(div[dense])),
        "max_divergence": float(np.nanmax(div[dense])),
        "mean_abs_curl": float(np.nanmean(np.abs(curl[dense]))),
        "median_abs_curl": float(np.nanmedian(np.abs(curl[dense]))),
        "max_abs_curl": float(np.nanmax(np.abs(curl[dense]))),
        "frac_convergent": float(np.mean(div[dense] < 0)),
        "frac_divergent": float(np.mean(div[dense] > 0)),
    }
    rows = []
    candidates = [
        ("strong_convergence", div, np.argsort(div[dense])[:5]),
        ("strong_divergence", div, np.argsort(-div[dense])[:5]),
        ("high_curl", np.abs(curl), np.argsort(-np.abs(curl[dense]))[:5]),
        ("low_speed", speed, np.argsort(speed[dense])[:5]),
    ]
    dense_idx = np.argwhere(dense)
    for kind, metric, order in candidates:
        for rank, oi in enumerate(order, start=1):
            iy, ix = dense_idx[int(oi)]
            rows.append(
                {
                    "model": model,
                    "layer_group": layer_group,
                    "attack": attack,
                    "region_type": kind,
                    "rank": rank,
                    "pc1": float(X[iy, ix]),
                    "pc2": float(Y[iy, ix]),
                    "speed": float(speed[iy, ix]),
                    "divergence": float(div[iy, ix]),
                    "curl": float(curl[iy, ix]),
                    "density_weight": float(W[iy, ix]),
                    "metric_value": float(metric[iy, ix]),
                }
            )
    return vals, rows, div, curl, speed, dense


def plot_diagnostics(out_dir, model, layer_group, attack, X, Y, div, curl, speed, dense):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), squeeze=False)
    items = [
        ("Divergence (- = convergence)", div, "coolwarm"),
        ("|Curl|", np.abs(curl), "magma"),
        ("Projected speed", speed, "viridis"),
    ]
    for ax, (title, arr, cmap) in zip(axes.ravel(), items):
        masked = np.ma.array(arr, mask=~dense)
        im = ax.imshow(
            masked,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            aspect="auto",
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel("transport PC1")
        ax.set_ylabel("transport PC2")
        ax.axhline(0, color="black", lw=0.6, alpha=0.25)
        ax.axvline(0, color="black", lw=0.6, alpha=0.25)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Flow-field diagnostics: {model} / {layer_group} / {LABELS.get(attack, attack)}")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_dir / f"flow_diagnostics_{model}_{layer_group}_{attack}.png", dpi=230, bbox_inches="tight")
    fig.savefig(out_dir / f"flow_diagnostics_{model}_{layer_group}_{attack}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeseries-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer-groups", default="hidden,penultimate")
    p.add_argument("--attacks", default="pgd,square,ga")
    p.add_argument("--grid-n", type=int, default=70)
    p.add_argument("--bandwidth-scale", type=float, default=0.07)
    p.add_argument("--density-percentile", type=float, default=55.0)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_groups = [x.strip() for x in args.layer_groups.split(",") if x.strip()]
    attacks = [x.strip() for x in args.attacks.split(",") if x.strip()]
    df = pd.read_csv(args.timeseries_csv, low_memory=False)
    df = df[(df.model == args.model) & df.layer_group.isin(layer_groups) & df.attack.isin(attacks)].copy()
    df = df.dropna(subset=["run_id", "pc1_coeff", "pc2_coeff", "normalized_progress", "step"])
    vecs = add_step_vectors(df)

    summary_rows = []
    region_rows = []
    for layer_group in layer_groups:
        layer_df = df[(df.layer_group == layer_group) & (df.final_success == 1)]
        if layer_df.empty:
            continue
        xlim, ylim = field_limits(layer_df)
        for attack in attacks:
            v = vecs[(vecs.layer_group == layer_group) & (vecs.attack == attack) & (vecs.final_success == 1)]
            field = smooth_field(v, xlim, ylim, args.grid_n, args.bandwidth_scale)
            if field is None:
                continue
            X, Y, U, V, W = field
            vals, regions, div, curl, speed, dense = summarize_grid(
                args.model, layer_group, attack, X, Y, U, V, W, args.density_percentile
            )
            summary_rows.append(vals)
            region_rows.extend(regions)
            plot_diagnostics(out_dir, args.model, layer_group, attack, X, Y, div, curl, speed, dense)

    summary = pd.DataFrame(summary_rows)
    regions = pd.DataFrame(region_rows)
    summary.to_csv(out_dir / "flow_field_dynamics_summary.csv", index=False)
    regions.to_csv(out_dir / "flow_field_dynamic_regions.csv", index=False)
    print(f"[SAVED] {out_dir / 'flow_field_dynamics_summary.csv'}")
    print(f"[SAVED] {out_dir / 'flow_field_dynamic_regions.csv'}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
