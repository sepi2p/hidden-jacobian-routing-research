#!/usr/bin/env python3
"""Plot turning angle between consecutive local transport vectors over time."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_turning_angles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_attack, run_id), g in df.sort_values(["attack", "run_id", "step"]).groupby(["attack", "run_id"], sort=False):
        g = g.copy().reset_index(drop=True)
        vx = g.dx.to_numpy(float)
        vy = g.dy.to_numpy(float)
        for i in range(1, len(g)):
            prev = np.array([vx[i - 1], vy[i - 1]], dtype=float)
            curr = np.array([vx[i], vy[i]], dtype=float)
            nprev = float(np.linalg.norm(prev))
            ncurr = float(np.linalg.norm(curr))
            if nprev <= 1e-12 or ncurr <= 1e-12:
                continue
            dot = float(np.clip(np.dot(prev, curr) / (nprev * ncurr), -1.0, 1.0))
            unsigned = float(np.arccos(dot))
            signed = float(np.arctan2(prev[0] * curr[1] - prev[1] * curr[0], np.dot(prev, curr)))
            r = g.iloc[i].to_dict()
            r["run_id"] = run_id
            r["turn_step"] = int(g.loc[i, "step"])
            r["prev_step"] = int(g.loc[i - 1, "step"])
            r["angle_rad"] = unsigned
            r["angle_deg"] = float(np.degrees(unsigned))
            r["signed_angle_rad"] = signed
            r["signed_angle_deg"] = float(np.degrees(signed))
            r["prev_radius"] = nprev
            r["curr_radius"] = ncurr
            rows.append(r)
    if not rows:
        raise RuntimeError("No consecutive local vectors available for turning-angle analysis.")
    return pd.DataFrame(rows)


def summarize(turns: pd.DataFrame) -> pd.DataFrame:
    return (
        turns.groupby(["attack", "turn_step"], as_index=False)
        .agg(
            n=("angle_deg", "size"),
            mean_angle_deg=("angle_deg", "mean"),
            median_angle_deg=("angle_deg", "median"),
            q25_angle_deg=("angle_deg", lambda x: float(np.percentile(x, 25))),
            q75_angle_deg=("angle_deg", lambda x: float(np.percentile(x, 75))),
            mean_signed_angle_deg=("signed_angle_deg", "mean"),
        )
    )


def plot_turns(turns: pd.DataFrame, summary: pd.DataFrame, out_path: Path):
    colors = {"pgd": "#111827", "square": "#2563eb"}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True, constrained_layout=True)
    for ax, attack in zip(axes, ["pgd", "square"]):
        d = turns[turns.attack == attack]
        s = summary[summary.attack == attack]
        ax.scatter(
            d.turn_step,
            d.angle_deg,
            s=10,
            color=colors[attack],
            alpha=0.24,
            edgecolors="none",
        )
        if not s.empty:
            ax.plot(s.turn_step, s.median_angle_deg, color="#dc2626", lw=2.0, label="median")
            ax.fill_between(
                s.turn_step.to_numpy(float),
                s.q25_angle_deg.to_numpy(float),
                s.q75_angle_deg.to_numpy(float),
                color="#dc2626",
                alpha=0.16,
                label="IQR",
            )
        ax.set_title(f"{attack.upper()} turning angle over steps")
        ax.set_xlabel("step index of current local vector")
        ax.grid(alpha=0.18)
        ax.set_ylim(0, 180)
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("angle from previous local vector (degrees)")
    fig.suptitle("Turning angle between consecutive class-0 local transport vectors", fontsize=12)
    fig.savefig(out_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_signed_turns(turns: pd.DataFrame, summary: pd.DataFrame, out_path: Path):
    colors = {"pgd": "#111827", "square": "#2563eb"}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True, constrained_layout=True)
    for ax, attack in zip(axes, ["pgd", "square"]):
        d = turns[turns.attack == attack]
        s = summary[summary.attack == attack]
        ax.scatter(
            d.turn_step,
            d.signed_angle_deg,
            s=10,
            color=colors[attack],
            alpha=0.24,
            edgecolors="none",
        )
        if not s.empty:
            ax.plot(s.turn_step, s.mean_signed_angle_deg, color="#dc2626", lw=2.0, label="mean signed")
        ax.axhline(0, color="black", lw=0.8, alpha=0.25)
        ax.set_title(f"{attack.upper()} signed turning angle")
        ax.set_xlabel("step index of current local vector")
        ax.grid(alpha=0.18)
        ax.set_ylim(-180, 180)
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("signed angle from previous local vector (degrees)")
    fig.suptitle("Signed turning angle between consecutive class-0 local transport vectors", fontsize=12)
    fig.savefig(out_path.with_name(out_path.name + "_signed").with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_name(out_path.name + "_signed").with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.step_vectors_csv)
    turns = compute_turning_angles(df)
    summary = summarize(turns)

    stem = Path(args.step_vectors_csv).stem.replace("_step_vectors", "_turning_angles")
    turns.to_csv(out_dir / f"{stem}.csv", index=False)
    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)
    plot_turns(turns, summary, out_dir / stem)
    plot_signed_turns(turns, summary, out_dir / stem)
    print(f"[SAVED] {out_dir / (stem + '.png')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step-vectors-csv", required=True)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/pgd_square_class_colored_flow")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
