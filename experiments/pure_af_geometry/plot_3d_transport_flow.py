#!/usr/bin/env python3
"""Plot recorded adversarial transport trajectories in 3D PC coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


ATTACK_COLORS = {"pgd": "#2563eb", "square": "#dc2626", "ga": "#16a34a"}
ATTACK_LABELS = {"pgd": "PGD", "square": "Square", "ga": "GA pure"}


def resample_run(g: pd.DataFrame, n_points: int) -> pd.DataFrame:
    g = g.sort_values("normalized_progress")
    x = g.normalized_progress.to_numpy(float)
    t = np.linspace(0.0, 1.0, n_points)
    out = {"progress": t}
    for src, dst in [("pc1_coeff", "pc1"), ("pc2_coeff", "pc2"), ("pc3_coeff", "pc3")]:
        y = g[src].to_numpy(float)
        if len(np.unique(x)) < 2:
            out[dst] = np.full(n_points, y[-1])
        else:
            out[dst] = np.interp(t, x, y)
    return pd.DataFrame(out)


def build_resampled(df: pd.DataFrame, n_points: int) -> pd.DataFrame:
    rows = []
    for (attack, run_id), g in df.groupby(["attack", "run_id"], sort=False):
        rr = resample_run(g, n_points)
        rr["attack"] = attack
        rr["run_id"] = run_id
        rows.append(rr)
    return pd.concat(rows, ignore_index=True)


def mean_paths(rr: pd.DataFrame) -> pd.DataFrame:
    return rr.groupby(["attack", "progress"], as_index=False)[["pc1", "pc2", "pc3"]].mean()


def add_plotly_path(fig: go.Figure, g: pd.DataFrame, attack: str, name: str, width: float, opacity: float, showlegend: bool) -> None:
    color = ATTACK_COLORS.get(attack, "#111827")
    fig.add_trace(
        go.Scatter3d(
            x=g.pc1,
            y=g.pc2,
            z=g.pc3,
            mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            name=name,
            showlegend=showlegend,
            hoverinfo="skip",
        )
    )


def add_plotly_mean(fig: go.Figure, g: pd.DataFrame, attack: str) -> None:
    color = ATTACK_COLORS.get(attack, "#111827")
    label = ATTACK_LABELS.get(attack, attack)
    fig.add_trace(
        go.Scatter3d(
            x=g.pc1,
            y=g.pc2,
            z=g.pc3,
            mode="lines+markers",
            line=dict(color=color, width=8),
            marker=dict(size=2.5, color=color),
            name=f"{label} mean flow",
            hovertemplate="progress=%{customdata:.2f}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>",
            customdata=g.progress,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[g.pc1.iloc[0], g.pc1.iloc[-1]],
            y=[g.pc2.iloc[0], g.pc2.iloc[-1]],
            z=[g.pc3.iloc[0], g.pc3.iloc[-1]],
            mode="markers",
            marker=dict(size=[6, 9], color=color, symbol=["circle", "diamond"]),
            name=f"{label} start/end",
            showlegend=False,
        )
    )
    idx = np.linspace(0, len(g) - 2, 8, dtype=int)
    starts = g.iloc[idx]
    ends = g.iloc[idx + 1]
    fig.add_trace(
        go.Cone(
            x=starts.pc1,
            y=starts.pc2,
            z=starts.pc3,
            u=ends.pc1.to_numpy() - starts.pc1.to_numpy(),
            v=ends.pc2.to_numpy() - starts.pc2.to_numpy(),
            w=ends.pc3.to_numpy() - starts.pc3.to_numpy(),
            sizemode="absolute",
            sizeref=0.45,
            anchor="tail",
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=f"{label} direction arrows",
            showlegend=False,
            opacity=0.82,
        )
    )


def save_interactive(rr: pd.DataFrame, mean: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> Path:
    fig = go.Figure()
    rng = np.random.default_rng(args.seed)
    for attack, sub in rr.groupby("attack", sort=False):
        runs = sub.run_id.drop_duplicates().to_numpy()
        if len(runs) > args.max_faint:
            runs = rng.choice(runs, size=args.max_faint, replace=False)
        for run_id in runs:
            g = sub[sub.run_id == run_id].sort_values("progress")
            add_plotly_path(fig, g, attack, ATTACK_LABELS.get(attack, attack), 2.0, 0.12, False)
    for attack, g in mean.groupby("attack", sort=False):
        add_plotly_mean(fig, g.sort_values("progress"), attack)

    fig.update_layout(
        title=(
            f"3D recorded adversarial transport flow: {args.model} / {args.layer_group}<br>"
            "<sup>Coordinates are PC1/PC2/PC3 coefficients; faint lines are individual successful trajectories</sup>"
        ),
        scene=dict(
            xaxis_title="transport PC1",
            yaxis_title="transport PC2",
            zaxis_title="transport PC3",
            aspectmode="data",
        ),
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        width=1100,
        height=850,
    )
    html = out_dir / f"transport_3d_flow_{args.model}_{args.layer_group}.html"
    fig.write_html(html, include_plotlyjs=True, full_html=True)
    return html


def set_equal_3d(ax, vals: np.ndarray) -> None:
    lo = np.nanpercentile(vals, 2, axis=0)
    hi = np.nanpercentile(vals, 98, axis=0)
    center = (lo + hi) / 2
    radius = max(float(np.max(hi - lo) / 2), 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_static(rr: pd.DataFrame, mean: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(8.4, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    rng = np.random.default_rng(args.seed)
    for attack, sub in rr.groupby("attack", sort=False):
        color = ATTACK_COLORS.get(attack, "#111827")
        runs = sub.run_id.drop_duplicates().to_numpy()
        if len(runs) > args.max_faint:
            runs = rng.choice(runs, size=args.max_faint, replace=False)
        for run_id in runs:
            g = sub[sub.run_id == run_id].sort_values("progress")
            ax.plot(g.pc1, g.pc2, g.pc3, color=color, alpha=0.09, lw=0.8)
    for attack, g in mean.groupby("attack", sort=False):
        color = ATTACK_COLORS.get(attack, "#111827")
        label = ATTACK_LABELS.get(attack, attack)
        g = g.sort_values("progress")
        ax.plot(g.pc1, g.pc2, g.pc3, color=color, lw=3.4, label=label)
        ax.scatter(g.pc1.iloc[0], g.pc2.iloc[0], g.pc3.iloc[0], color=color, marker="o", s=42)
        ax.scatter(g.pc1.iloc[-1], g.pc2.iloc[-1], g.pc3.iloc[-1], color=color, marker="^", s=76)
        idx = np.linspace(0, len(g) - 2, 8, dtype=int)
        starts = g.iloc[idx]
        ends = g.iloc[idx + 1]
        ax.quiver(
            starts.pc1,
            starts.pc2,
            starts.pc3,
            ends.pc1.to_numpy() - starts.pc1.to_numpy(),
            ends.pc2.to_numpy() - starts.pc2.to_numpy(),
            ends.pc3.to_numpy() - starts.pc3.to_numpy(),
            color=color,
            length=1.0,
            normalize=False,
            arrow_length_ratio=0.25,
            alpha=0.92,
        )
    vals = rr[["pc1", "pc2", "pc3"]].to_numpy(float)
    set_equal_3d(ax, vals)
    ax.set_title(f"3D recorded adversarial transport flow: {args.model} / {args.layer_group}")
    ax.set_xlabel("transport PC1")
    ax.set_ylabel("transport PC2")
    ax.set_zlabel("transport PC3")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    stem = f"transport_3d_flow_{args.model}_{args.layer_group}"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    df = pd.read_csv(args.timeseries_csv)
    df = df[
        (df.model == args.model)
        & (df.layer_group == args.layer_group)
        & (df.attack.isin(attacks))
        & (df.final_success == 1)
    ].dropna(subset=["pc1_coeff", "pc2_coeff", "pc3_coeff", "normalized_progress"])
    if df.empty:
        raise RuntimeError("No successful trajectory rows found for the requested selection")
    rr = build_resampled(df, args.n_points)
    mean = mean_paths(rr)
    rr.to_csv(out_dir / f"transport_3d_flow_resampled_{args.model}_{args.layer_group}.csv", index=False)
    mean.to_csv(out_dir / f"transport_3d_flow_mean_{args.model}_{args.layer_group}.csv", index=False)
    html = save_interactive(rr, mean, args, out_dir)
    png, pdf = save_static(rr, mean, args, out_dir)
    print(f"[SAVED] {html}")
    print(f"[SAVED] {png}")
    print(f"[SAVED] {pdf}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeseries-csv", default="analysis_outputs/pure_af_geometry/cifar_attack_transport_decomposition/attack_transport_projection_timeseries.csv")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_attack_transport_decomposition/transport_3d")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer-group", default="penultimate", choices=["hidden", "penultimate", "logits"])
    p.add_argument("--attacks", default="pgd,square,ga")
    p.add_argument("--n-points", type=int, default=55)
    p.add_argument("--max-faint", type=int, default=45)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim", type=float, default=-55.0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
