#!/usr/bin/env python3
"""Interactive 3D phase portraits from recorded adversarial transport steps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


ATTACK_COLORS = {"pgd": "#2563eb", "square": "#dc2626", "ga": "#16a34a"}
ATTACK_LABELS = {"pgd": "PGD", "square": "Square", "ga": "GA pure", "all": "All attacks"}


def add_step_vectors(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model", "attack", "layer_group", "layer", "run_id"]
    for _, g in df.sort_values(keys + ["step"]).groupby(keys, sort=False):
        g = g.copy()
        for pc in [1, 2, 3]:
            g[f"dpc{pc}"] = g[f"pc{pc}_coeff"].shift(-1) - g[f"pc{pc}_coeff"]
        rows.append(g.iloc[:-1])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    cols = ["pc1_coeff", "pc2_coeff", "pc3_coeff", "dpc1", "dpc2", "dpc3"]
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)


def binned_3d_flow(vecs: pd.DataFrame, bins: int, min_count: int, max_cones: int) -> pd.DataFrame:
    cols = ["pc1_coeff", "pc2_coeff", "pc3_coeff", "dpc1", "dpc2", "dpc3"]
    arr = vecs[cols].to_numpy(float)
    if len(arr) < min_count:
        return pd.DataFrame()

    xyz = arr[:, :3]
    dxyz = arr[:, 3:]
    lo = np.nanpercentile(xyz, 2, axis=0)
    hi = np.nanpercentile(xyz, 98, axis=0)
    edges = [np.linspace(lo[i], hi[i], bins + 1) for i in range(3)]
    idx = [np.digitize(xyz[:, i], edges[i]) - 1 for i in range(3)]
    rows = []
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                mask = (idx[0] == i) & (idx[1] == j) & (idx[2] == k)
                n = int(mask.sum())
                if n < min_count:
                    continue
                center = xyz[mask].mean(axis=0)
                vec = dxyz[mask].mean(axis=0)
                mag = float(np.linalg.norm(vec))
                if mag <= 1e-12:
                    continue
                rows.append(
                    {
                        "x": center[0],
                        "y": center[1],
                        "z": center[2],
                        "u": vec[0],
                        "v": vec[1],
                        "w": vec[2],
                        "count": n,
                        "magnitude": mag,
                    }
                )
    flow = pd.DataFrame(rows)
    if flow.empty:
        return flow
    # Keep the most data-supported bins to avoid a visually noisy cone cloud.
    flow = flow.sort_values(["count", "magnitude"], ascending=False).head(max_cones).copy()
    return flow.sort_values("magnitude")


def resample_mean(df: pd.DataFrame, attack: str, n_points: int) -> pd.DataFrame:
    rows = []
    sub = df[df.attack == attack].copy()
    for run_id, g in sub.groupby("run_id", sort=False):
        g = g.sort_values("normalized_progress")
        x = g.normalized_progress.to_numpy(float)
        if len(np.unique(x)) < 2:
            continue
        t = np.linspace(0.0, 1.0, n_points)
        rows.append(
            pd.DataFrame(
                {
                    "run_id": run_id,
                    "progress": t,
                    "pc1": np.interp(t, x, g.pc1_coeff.to_numpy(float)),
                    "pc2": np.interp(t, x, g.pc2_coeff.to_numpy(float)),
                    "pc3": np.interp(t, x, g.pc3_coeff.to_numpy(float)),
                }
            )
        )
    if not rows:
        return pd.DataFrame()
    rr = pd.concat(rows, ignore_index=True)
    return rr.groupby("progress", as_index=False)[["pc1", "pc2", "pc3"]].mean()


def add_paths(fig: go.Figure, df: pd.DataFrame, attacks: list[str], max_paths: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for attack in attacks:
        color = ATTACK_COLORS.get(attack, "#111827")
        label = ATTACK_LABELS.get(attack, attack)
        sub = df[(df.attack == attack) & (df.final_success == 1)]
        runs = sub.run_id.drop_duplicates().to_numpy()
        if len(runs) > max_paths:
            runs = rng.choice(runs, size=max_paths, replace=False)
        for run_id in runs:
            g = sub[sub.run_id == run_id].sort_values("step")
            fig.add_trace(
                go.Scatter3d(
                    x=g.pc1_coeff,
                    y=g.pc2_coeff,
                    z=g.pc3_coeff,
                    mode="lines",
                    line=dict(color=color, width=2),
                    opacity=0.10,
                    name=label,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        mean = resample_mean(sub, attack, 55)
        if mean.empty:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=mean.pc1,
                y=mean.pc2,
                z=mean.pc3,
                mode="lines+markers",
                line=dict(color=color, width=8),
                marker=dict(size=2.3, color=color),
                name=f"{label} mean path",
            )
        )


def add_cones(fig: go.Figure, flow: pd.DataFrame, name: str, sizeref: float) -> None:
    if flow.empty:
        return
    fig.add_trace(
        go.Cone(
            x=flow.x,
            y=flow.y,
            z=flow.z,
            u=flow.u,
            v=flow.v,
            w=flow.w,
            sizemode="absolute",
            sizeref=sizeref,
            anchor="tail",
            colorscale="Viridis",
            cmin=float(flow.magnitude.min()),
            cmax=float(flow.magnitude.max()),
            colorbar=dict(title="mean step<br>norm"),
            opacity=0.58,
            name=name,
            hovertemplate=(
                "PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}"
                "<br>dPC=(%{u:.2f}, %{v:.2f}, %{w:.2f})<extra></extra>"
            ),
        )
    )


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    df = pd.read_csv(args.timeseries_csv)
    df = df[
        (df.model == args.model)
        & (df.layer_group == args.layer_group)
        & (df.attack.isin(attacks))
    ].dropna(subset=["pc1_coeff", "pc2_coeff", "pc3_coeff", "normalized_progress", "step"])
    vecs = add_step_vectors(df)
    vecs = vecs[vecs.final_success == 1].copy()
    flow = binned_3d_flow(vecs, args.bins, args.min_count, args.max_cones)

    stem = f"transport_3d_phase_portrait_{args.model}_{args.layer_group}"
    vecs.to_csv(out_dir / f"{stem}_step_vectors.csv", index=False)
    flow.to_csv(out_dir / f"{stem}_binned_flow.csv", index=False)

    fig = go.Figure()
    add_cones(fig, flow, "empirical binned flow", args.cone_size)
    add_paths(fig, df, attacks, args.max_paths, args.seed)
    fig.update_layout(
        title=(
            f"3D empirical transport phase portrait: {args.model} / {args.layer_group}<br>"
            "<sup>Cones are binned mean recorded step vectors in PC1/PC2/PC3; lines are actual successful trajectories</sup>"
        ),
        template="plotly_white",
        width=1120,
        height=860,
        scene=dict(
            xaxis_title="transport PC1",
            yaxis_title="transport PC2",
            zaxis_title="transport PC3",
            aspectmode="data",
        ),
        legend=dict(x=0.02, y=0.98),
    )
    html = out_dir / f"{stem}.html"
    fig.write_html(html, include_plotlyjs=True, full_html=True)
    print(f"[SAVED] {html}")
    print(f"[SAVED] {out_dir / f'{stem}_binned_flow.csv'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeseries-csv", default="analysis_outputs/pure_af_geometry/cifar_attack_transport_decomposition/attack_transport_projection_timeseries.csv")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_attack_transport_decomposition/phase_portraits_3d")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer-group", default="penultimate", choices=["hidden", "penultimate", "logits"])
    p.add_argument("--attacks", default="pgd,square,ga")
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--max-cones", type=int, default=140)
    p.add_argument("--cone-size", type=float, default=0.7)
    p.add_argument("--max-paths", type=int, default=35)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
