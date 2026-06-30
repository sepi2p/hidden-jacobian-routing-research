#!/usr/bin/env python3
"""Create clear 2D phase portraits of recorded adversarial transport flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ATTACK_COLORS = {"pgd": "#2563eb", "square": "#dc2626", "ga": "#16a34a"}
ATTACK_LABELS = {"pgd": "PGD", "square": "Square", "ga": "GA pure", "all": "All attacks"}


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
    out = pd.concat(rows, ignore_index=True)
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["pc1_coeff", "pc2_coeff", "dx", "dy"])


def smooth_field(vecs: pd.DataFrame, grid_n: int, bandwidth_scale: float, min_weight: float):
    x = vecs.pc1_coeff.to_numpy(float)
    y = vecs.pc2_coeff.to_numpy(float)
    dx = vecs.dx.to_numpy(float)
    dy = vecs.dy.to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & np.isfinite(dy)
    x, y, dx, dy = x[keep], y[keep], dx[keep], dy[keep]
    if len(x) < 10:
        raise RuntimeError("Not enough vectors for a phase portrait")

    xlo, xhi = np.nanpercentile(x, [1, 99])
    ylo, yhi = np.nanpercentile(y, [1, 99])
    pad_x = 0.06 * max(xhi - xlo, 1e-6)
    pad_y = 0.06 * max(yhi - ylo, 1e-6)
    gx = np.linspace(xlo - pad_x, xhi + pad_x, grid_n)
    gy = np.linspace(ylo - pad_y, yhi + pad_y, grid_n)
    X, Y = np.meshgrid(gx, gy)

    sigma = bandwidth_scale * max(xhi - xlo, yhi - ylo, 1e-6)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    W = np.zeros_like(X)

    # Chunked Gaussian smoothing to keep memory tame.
    points = np.column_stack([x, y])
    vectors = np.column_stack([dx, dy])
    flat = np.column_stack([X.ravel(), Y.ravel()])
    u = np.zeros(len(flat))
    v = np.zeros(len(flat))
    wsum = np.zeros(len(flat))
    chunk = 512
    for start in range(0, len(flat), chunk):
        stop = start + chunk
        q = flat[start:stop]
        dist2 = ((q[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        w = np.exp(-0.5 * dist2 / (sigma**2))
        ws = w.sum(axis=1)
        u[start:stop] = (w @ vectors[:, 0]) / np.maximum(ws, 1e-12)
        v[start:stop] = (w @ vectors[:, 1]) / np.maximum(ws, 1e-12)
        wsum[start:stop] = ws

    U[:] = u.reshape(X.shape)
    V[:] = v.reshape(Y.shape)
    W[:] = wsum.reshape(X.shape)

    threshold = np.nanpercentile(W, min_weight)
    mask = W < threshold
    U = np.ma.array(U, mask=mask)
    V = np.ma.array(V, mask=mask)
    W = np.ma.array(W, mask=mask)
    speed = np.sqrt(U**2 + V**2)
    return X, Y, U, V, W, speed


def mean_paths(df: pd.DataFrame, attack: str | None) -> pd.DataFrame:
    sub = df.copy()
    if attack is not None:
        sub = sub[sub.attack == attack]
    rows = []
    for key, g in sub.groupby(["attack", "run_id"], sort=False):
        g = g.sort_values("normalized_progress")
        x = g.normalized_progress.to_numpy(float)
        if len(np.unique(x)) < 2:
            continue
        t = np.linspace(0, 1, 50)
        rows.append(
            pd.DataFrame(
                {
                    "attack": key[0],
                    "run_id": key[1],
                    "t": t,
                    "pc1_coeff": np.interp(t, x, g.pc1_coeff.to_numpy(float)),
                    "pc2_coeff": np.interp(t, x, g.pc2_coeff.to_numpy(float)),
                }
            )
        )
    if not rows:
        return pd.DataFrame()
    rr = pd.concat(rows, ignore_index=True)
    return rr.groupby(["attack", "t"], as_index=False)[["pc1_coeff", "pc2_coeff"]].mean()


def draw_panel(ax, df: pd.DataFrame, vecs: pd.DataFrame, attack_name: str, args: argparse.Namespace):
    if attack_name == "all":
        d = df.copy()
        v = vecs.copy()
    else:
        d = df[df.attack == attack_name].copy()
        v = vecs[vecs.attack == attack_name].copy()
    d = d[d.final_success == 1]
    v = v[v.final_success == 1]
    X, Y, U, V, W, speed = smooth_field(v, args.grid_n, args.bandwidth_scale, args.min_weight_percentile)

    density = np.log1p(W.filled(0))
    ax.imshow(
        density,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        cmap="Greys",
        alpha=0.22,
        aspect="auto",
        zorder=0,
    )
    ax.streamplot(
        X,
        Y,
        U,
        V,
        color=speed,
        cmap="viridis",
        density=args.stream_density,
        linewidth=1.25,
        arrowsize=1.25,
        minlength=0.08,
        zorder=2,
    )

    rng = np.random.default_rng(args.seed)
    runs = d["run_id"].drop_duplicates().to_numpy()
    if len(runs) > args.max_paths:
        runs = rng.choice(runs, size=args.max_paths, replace=False)
    for run_id in runs:
        g = d[d.run_id == run_id].sort_values("step")
        color = ATTACK_COLORS.get(g.attack.iloc[0], "#111827")
        ax.plot(g.pc1_coeff, g.pc2_coeff, color=color, alpha=0.16, lw=1.1, zorder=3)

    mp = mean_paths(d, None if attack_name == "all" else attack_name)
    for attack, g in mp.groupby("attack"):
        color = ATTACK_COLORS.get(attack, "#111827")
        ax.plot(g.pc1_coeff, g.pc2_coeff, color=color, lw=3.0, alpha=0.96, zorder=4, label=ATTACK_LABELS.get(attack, attack))
        ax.scatter(g.pc1_coeff.iloc[0], g.pc2_coeff.iloc[0], color=color, marker="o", s=40, zorder=5)
        ax.scatter(g.pc1_coeff.iloc[-1], g.pc2_coeff.iloc[-1], color=color, marker="^", s=65, zorder=5)

    ax.set_title(ATTACK_LABELS.get(attack_name, attack_name))
    ax.set_xlabel("transport PC1")
    ax.set_ylabel("transport PC2")
    ax.axhline(0, color="black", lw=0.8, alpha=0.22)
    ax.axvline(0, color="black", lw=0.8, alpha=0.22)
    ax.grid(alpha=0.17)
    ax.legend(loc="best", frameon=True, fontsize=8)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.timeseries_csv)
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    df = df[
        (df.model == args.model)
        & (df.layer_group == args.layer_group)
        & (df.attack.isin(attacks))
    ].copy()
    df = df.dropna(subset=["pc1_coeff", "pc2_coeff", "normalized_progress", "step"])
    vecs = add_step_vectors(df)
    vecs.to_csv(out_dir / f"phase_portrait_vectors_{args.model}_{args.layer_group}.csv", index=False)

    panels = attacks + ["all"] if args.include_combined else attacks
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5.5), squeeze=False)
    for ax, attack in zip(axes.ravel(), panels):
        draw_panel(ax, df, vecs, attack, args)
    fig.suptitle(
        f"Smoothed empirical adversarial flow in transport-PC space: {args.model} / {args.layer_group}\n"
        "Streamlines are kernel-smoothed recorded step vectors; faint paths are individual successful trajectories",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    stem = f"transport_phase_portrait_{args.model}_{args.layer_group}"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / f"{stem}_metadata.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[SAVED] {png}")
    print(f"[SAVED] {pdf}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeseries-csv", default="analysis_outputs/pure_af_geometry/cifar_attack_transport_decomposition/attack_transport_projection_timeseries.csv")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_attack_transport_decomposition/phase_portraits")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer-group", default="penultimate", choices=["hidden", "penultimate", "logits"])
    p.add_argument("--attacks", default="pgd,square")
    p.add_argument("--include-combined", action="store_true", default=True)
    p.add_argument("--grid-n", type=int, default=55)
    p.add_argument("--bandwidth-scale", type=float, default=0.075)
    p.add_argument("--min-weight-percentile", type=float, default=18.0)
    p.add_argument("--stream-density", type=float, default=1.35)
    p.add_argument("--max-paths", type=int, default=28)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
