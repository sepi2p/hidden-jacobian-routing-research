#!/usr/bin/env python3
"""Plot actual adversarial trajectories and local flow vectors in 2D PC space.

The input time series is already projected onto learned transport PCs. This
script does not synthesize trajectories: every foreground arrow is a recorded
step-to-step displacement, and every background quiver arrow is an average of
recorded local displacements inside a PC1/PC2 bin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ATTACK_COLORS = {
    "pgd": "#2563eb",
    "square": "#dc2626",
    "ga": "#16a34a",
}

ATTACK_LABELS = {
    "pgd": "PGD",
    "square": "Square",
    "ga": "GA pure",
}


def add_step_vectors(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model", "attack", "layer_group", "layer", "run_id"]
    for _, g in df.sort_values(group_cols + ["step"]).groupby(group_cols, sort=False):
        g = g.copy()
        g["dx"] = g["pc1_coeff"].shift(-1) - g["pc1_coeff"]
        g["dy"] = g["pc2_coeff"].shift(-1) - g["pc2_coeff"]
        rows.append(g.iloc[:-1])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["pc1_coeff", "pc2_coeff", "dx", "dy"])
    return out


def binned_flow(vecs: pd.DataFrame, bins: int, min_count: int) -> pd.DataFrame:
    if vecs.empty:
        return pd.DataFrame()
    x = vecs["pc1_coeff"].to_numpy(float)
    y = vecs["pc2_coeff"].to_numpy(float)
    dx = vecs["dx"].to_numpy(float)
    dy = vecs["dy"].to_numpy(float)
    xlo, xhi = np.nanpercentile(x, [2, 98])
    ylo, yhi = np.nanpercentile(y, [2, 98])
    if not np.isfinite([xlo, xhi, ylo, yhi]).all() or xlo == xhi or ylo == yhi:
        return pd.DataFrame()
    xb = np.linspace(xlo, xhi, bins + 1)
    yb = np.linspace(ylo, yhi, bins + 1)
    xi = np.digitize(x, xb) - 1
    yi = np.digitize(y, yb) - 1
    rows = []
    for i in range(bins):
        for j in range(bins):
            mask = (xi == i) & (yi == j)
            n = int(mask.sum())
            if n < min_count:
                continue
            rows.append(
                {
                    "x": float(x[mask].mean()),
                    "y": float(y[mask].mean()),
                    "dx": float(dx[mask].mean()),
                    "dy": float(dy[mask].mean()),
                    "count": n,
                }
            )
    return pd.DataFrame(rows)


def choose_runs(df: pd.DataFrame, model: str, layer_group: str, n_images: int, attacks: list[str]) -> list[tuple[str, float]]:
    sub = df[(df.model == model) & (df.layer_group == layer_group) & (df.attack.isin(attacks))].copy()
    if sub.empty:
        return []
    # Prefer image_ord values that have both PGD and Square successful runs.
    candidates = []
    for image_ord, g in sub.groupby("image_ord", dropna=True):
        got = set(g[g.final_success == 1].attack)
        if {"pgd", "square"}.issubset(got):
            candidates.append((len(g), float(image_ord)))
    if not candidates:
        for image_ord, g in sub.groupby("image_ord", dropna=True):
            candidates.append((len(g), float(image_ord)))
    candidates = sorted(candidates, reverse=True)[:n_images]
    return [(model, image_ord) for _, image_ord in candidates]


def plot_image_panel(ax, df: pd.DataFrame, vecs: pd.DataFrame, model: str, layer_group: str, image_ord: float, attacks: list[str]):
    sub = df[
        (df.model == model)
        & (df.layer_group == layer_group)
        & (df.image_ord == image_ord)
        & (df.attack.isin(attacks))
    ].copy()
    vsub = vecs[(vecs.model == model) & (vecs.layer_group == layer_group) & (vecs.attack.isin(attacks))].copy()
    flow = binned_flow(vsub[vsub.final_success == 1], bins=13, min_count=3)
    if not flow.empty:
        mag = np.sqrt(flow.dx.to_numpy() ** 2 + flow.dy.to_numpy() ** 2)
        scale = np.nanpercentile(mag, 90)
        if scale > 0:
            flow["dx_plot"] = flow["dx"] / scale
            flow["dy_plot"] = flow["dy"] / scale
        else:
            flow["dx_plot"] = flow["dx"]
            flow["dy_plot"] = flow["dy"]
        ax.scatter(flow.x, flow.y, s=10, color="#525252", alpha=0.22, zorder=1)
        ax.quiver(
            flow.x,
            flow.y,
            flow.dx_plot,
            flow.dy_plot,
            color="#525252",
            alpha=0.58,
            angles="xy",
            scale_units="xy",
            scale=5.8,
            width=0.0042,
            zorder=2,
        )

    for attack in attacks:
        color = ATTACK_COLORS.get(attack, "black")
        runs = sub[sub.attack == attack]["run_id"].drop_duplicates().tolist()
        if not runs:
            continue
        # For PGD/Square there should be one run per image. For anything else,
        # draw the first available run to keep the panel readable.
        run_id = runs[0]
        g = sub[(sub.attack == attack) & (sub.run_id == run_id)].sort_values("step")
        if g.empty:
            continue
        ax.plot(g.pc1_coeff, g.pc2_coeff, color=color, lw=2.0, alpha=0.92, label=ATTACK_LABELS.get(attack, attack), zorder=3)
        ax.scatter(g.pc1_coeff.iloc[0], g.pc2_coeff.iloc[0], color=color, marker="o", s=34, zorder=4)
        ax.scatter(g.pc1_coeff.iloc[-1], g.pc2_coeff.iloc[-1], color=color, marker="^", s=56, zorder=4)
        if len(g) > 1:
            stride = max(1, len(g) // 12)
            gg = g.iloc[::stride].copy()
            gg["dx"] = gg.pc1_coeff.shift(-1) - gg.pc1_coeff
            gg["dy"] = gg.pc2_coeff.shift(-1) - gg.pc2_coeff
            gg = gg.iloc[:-1]
            ax.quiver(
                gg.pc1_coeff,
                gg.pc2_coeff,
                gg.dx,
                gg.dy,
                color=color,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.006,
                alpha=0.75,
                zorder=5,
            )

    label = sub["label"].dropna().astype(int)
    title_label = f", y={label.iloc[0]}" if len(label) else ""
    ax.set_title(f"{model} {layer_group}: image {int(image_ord)}{title_label}")
    ax.set_xlabel("transport PC1 coefficient")
    ax.set_ylabel("transport PC2 coefficient")
    ax.axhline(0, color="black", lw=0.8, alpha=0.18)
    ax.axvline(0, color="black", lw=0.8, alpha=0.18)
    ax.grid(alpha=0.22)
    ax.legend(loc="best", frameon=True, fontsize=8)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.timeseries_csv)
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    df = df[df.attack.isin(attacks)].copy()
    df = df.dropna(subset=["pc1_coeff", "pc2_coeff", "step", "run_id", "model", "layer_group"])
    vecs = add_step_vectors(df)
    vecs.to_csv(out_dir / "actual_2d_flow_step_vectors.csv", index=False)

    selected = choose_runs(df, args.model, args.layer_group, args.n_images, attacks)
    if not selected:
        raise RuntimeError(f"No runs found for model={args.model}, layer_group={args.layer_group}")

    fig, axes = plt.subplots(1, len(selected), figsize=(6.2 * len(selected), 5.3), squeeze=False)
    for ax, (model, image_ord) in zip(axes.ravel(), selected):
        plot_image_panel(ax, df, vecs, model, args.layer_group, image_ord, attacks)

    fig.suptitle(
        "Actual adversarial trajectory flow in learned 2D transport coordinates\n"
        "Colored paths/arrows are recorded image trajectories; grey arrows are binned average recorded step vectors",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    png = out_dir / f"actual_2d_flow_{args.model}_{args.layer_group}.png"
    pdf = out_dir / f"actual_2d_flow_{args.model}_{args.layer_group}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "timeseries_csv": args.timeseries_csv,
        "model": args.model,
        "layer_group": args.layer_group,
        "attacks": attacks,
        "selected": [{"model": m, "image_ord": image_ord} for m, image_ord in selected],
        "note": "2D coordinates are transport PC1/PC2 coefficients from recorded trajectories; flow arrows are binned mean recorded step displacements.",
    }
    with open(out_dir / "actual_2d_flow_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] {png}")
    print(f"[SAVED] {pdf}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeseries-csv", default="analysis_outputs/pure_af_geometry/cifar_attack_transport_decomposition/attack_transport_projection_timeseries.csv")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_attack_transport_decomposition/actual_2d_flow")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer-group", default="penultimate", choices=["hidden", "penultimate", "logits"])
    p.add_argument("--attacks", default="pgd,square")
    p.add_argument("--n-images", type=int, default=2)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
