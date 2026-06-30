#!/usr/bin/env python3
"""Plot PGD, Square, and GA trajectories in learned transport PC1-3 space."""

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
    "ga": "GA",
}


def resample_run(g: pd.DataFrame, n_points: int) -> pd.DataFrame:
    g = g.sort_values("normalized_progress")
    x = g["normalized_progress"].to_numpy(dtype=float)
    out = {"normalized_progress": np.linspace(0.0, 1.0, n_points)}
    for col in ["pc1_coeff", "pc2_coeff", "pc3_coeff"]:
        y = g[col].to_numpy(dtype=float)
        if len(np.unique(x)) == 1:
            out[col] = np.full(n_points, y[-1])
        else:
            out[col] = np.interp(out["normalized_progress"], x, y)
    return pd.DataFrame(out)


def build_mean_paths(df: pd.DataFrame, n_points: int) -> pd.DataFrame:
    rows = []
    keys = ["model", "attack", "layer_group", "layer", "run_id"]
    for key, g in df.groupby(keys):
        model, attack, layer_group, layer, run_id = key
        rr = resample_run(g, n_points)
        rr["model"] = model
        rr["attack"] = attack
        rr["layer_group"] = layer_group
        rr["layer"] = layer
        rr["run_id"] = run_id
        rows.append(rr)
    resampled = pd.concat(rows, ignore_index=True)
    mean = resampled.groupby(["model", "attack", "layer_group", "layer", "normalized_progress"], as_index=False)[
        ["pc1_coeff", "pc2_coeff", "pc3_coeff"]
    ].mean()
    return resampled, mean


def set_equalish_3d(ax, sub: pd.DataFrame):
    vals = sub[["pc1_coeff", "pc2_coeff", "pc3_coeff"]].to_numpy(dtype=float)
    if len(vals) == 0:
        return
    lo = np.nanpercentile(vals, 2, axis=0)
    hi = np.nanpercentile(vals, 98, axis=0)
    center = (lo + hi) / 2
    radius = max(float(np.max(hi - lo) / 2), 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_layer(df: pd.DataFrame, resampled: pd.DataFrame, mean: pd.DataFrame, layer_group: str, out_dir: Path, max_individual: int, seed: int):
    sub = df[(df.layer_group == layer_group) & (df.final_success == 1)].copy()
    if sub.empty:
        return
    rsub = resampled[resampled.layer_group == layer_group]
    msub = mean[mean.layer_group == layer_group]
    models = list(dict.fromkeys(sub.model))
    fig = plt.figure(figsize=(13.5, 11))
    rng = np.random.default_rng(seed)
    for i, model in enumerate(models, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        model_sub = sub[sub.model == model]
        model_resampled = rsub[rsub.model == model]
        model_mean = msub[msub.model == model]
        for attack in ["pgd", "square", "ga"]:
            color = ATTACK_COLORS[attack]
            runs = model_resampled[model_resampled.attack == attack]["run_id"].drop_duplicates().to_numpy()
            if len(runs) > max_individual:
                runs = rng.choice(runs, size=max_individual, replace=False)
            for run_id in runs:
                g = model_resampled[(model_resampled.attack == attack) & (model_resampled.run_id == run_id)].sort_values("normalized_progress")
                ax.plot(
                    g.pc1_coeff,
                    g.pc2_coeff,
                    g.pc3_coeff,
                    color=color,
                    alpha=0.07,
                    linewidth=0.8,
                )
            mg = model_mean[model_mean.attack == attack].sort_values("normalized_progress")
            if not mg.empty:
                ax.plot(
                    mg.pc1_coeff,
                    mg.pc2_coeff,
                    mg.pc3_coeff,
                    color=color,
                    linewidth=3.0,
                    label=ATTACK_LABELS[attack],
                )
                ax.scatter(
                    [mg.pc1_coeff.iloc[0]],
                    [mg.pc2_coeff.iloc[0]],
                    [mg.pc3_coeff.iloc[0]],
                    color=color,
                    marker="o",
                    s=22,
                    alpha=0.9,
                )
                ax.scatter(
                    [mg.pc1_coeff.iloc[-1]],
                    [mg.pc2_coeff.iloc[-1]],
                    [mg.pc3_coeff.iloc[-1]],
                    color=color,
                    marker="^",
                    s=55,
                    alpha=0.95,
                )
        layer = model_sub.layer.iloc[0]
        ax.set_title(f"{model} ({layer})", pad=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.view_init(elev=22, azim=-52)
        set_equalish_3d(ax, model_sub)
        ax.grid(alpha=0.2)
        if i == 1:
            ax.legend(loc="upper left", bbox_to_anchor=(-0.08, 1.02))
    fig.suptitle(
        f"Different attacks in learned transport coordinates: {layer_group}\n"
        "Faint lines: individual successful trajectories; thick lines: mean path; triangles: final states",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / f"attack_pc123_trajectory_convergence_{layer_group}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"attack_pc123_trajectory_convergence_{layer_group}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_endpoint_overlay(df: pd.DataFrame, out_dir: Path):
    sub = df[(df.layer_group == "penultimate") & (df.final_success == 1)].copy()
    if sub.empty:
        return
    final = sub.sort_values(["normalized_progress", "step"]).groupby(["model", "attack", "run_id"], as_index=False).tail(1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    axes = axes.ravel()
    for ax, (model, g) in zip(axes, final.groupby("model")):
        for attack, ag in g.groupby("attack"):
            ax.scatter(
                ag.pc1_coeff,
                ag.pc2_coeff,
                s=18,
                alpha=0.65,
                color=ATTACK_COLORS[attack],
                label=ATTACK_LABELS[attack],
            )
        ax.axhline(0, color="black", lw=0.7, alpha=0.25)
        ax.axvline(0, color="black", lw=0.7, alpha=0.25)
        ax.set_title(model)
        ax.set_xlabel("PC1 final coefficient")
        ax.set_ylabel("PC2 final coefficient")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Successful attack endpoints in penultimate transport coordinates")
    fig.savefig(out_dir / "attack_pc12_endpoint_convergence_penultimate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.timeseries_csv)
    df = df[df.attack.isin(["pgd", "square", "ga"])].copy()
    df = df.dropna(subset=["pc1_coeff", "pc2_coeff", "pc3_coeff", "normalized_progress"])
    # Keep successful trajectories for the main convergence plot.
    success = df[df.final_success == 1].copy()
    resampled, mean = build_mean_paths(success, args.resample_points)
    resampled.to_csv(out_dir / "attack_pc123_resampled_success_trajectories.csv", index=False)
    mean.to_csv(out_dir / "attack_pc123_mean_success_paths.csv", index=False)
    for layer_group in ["hidden", "penultimate", "logits"]:
        plot_layer(success, resampled, mean, layer_group, out_dir, args.max_individual, args.seed)
    plot_endpoint_overlay(success, out_dir)
    with open(out_dir / "attack_pc123_convergence_metadata.json", "w") as f:
        json.dump({"args": vars(args), "n_success_rows": int(len(success))}, f, indent=2)
    print(f"[SAVED] {out_dir}", flush=True)
    print((out_dir / "attack_pc123_trajectory_convergence_penultimate.png").as_posix(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeseries-csv", default="analysis_outputs/pure_af_geometry/cifar_attack_transport_decomposition/attack_transport_projection_timeseries.csv")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_attack_transport_decomposition")
    p.add_argument("--resample-points", type=int, default=40)
    p.add_argument("--max-individual", type=int, default=35)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
