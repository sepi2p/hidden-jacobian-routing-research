#!/usr/bin/env python3
"""Compare progress-normalized trajectory curvature between attacks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resample_polyline(points: np.ndarray, n_points: int) -> np.ndarray | None:
    if len(points) < 3:
        return None
    seg = points[1:] - points[:-1]
    lengths = np.linalg.norm(seg, axis=1)
    keep = lengths > 1e-12
    if keep.sum() < 2:
        return None
    # Keep original vertices but collapse zero-length segments for interpolation.
    kept_points = [points[0]]
    for i, ok in enumerate(keep):
        if ok:
            kept_points.append(points[i + 1])
    points = np.asarray(kept_points, dtype=float)
    seg = points[1:] - points[:-1]
    lengths = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cum[-1])
    if total <= 1e-12:
        return None
    target = np.linspace(0.0, total, n_points)
    out = np.zeros((n_points, 2), dtype=float)
    out[:, 0] = np.interp(target, cum, points[:, 0])
    out[:, 1] = np.interp(target, cum, points[:, 1])
    return out


def turning_angles(points: np.ndarray) -> np.ndarray:
    tangents = points[1:] - points[:-1]
    norms = np.linalg.norm(tangents, axis=1)
    rows = []
    for i in range(1, len(tangents)):
        if norms[i - 1] <= 1e-12 or norms[i] <= 1e-12:
            continue
        dot = np.clip(float(np.dot(tangents[i - 1], tangents[i]) / (norms[i - 1] * norms[i])), -1.0, 1.0)
        rows.append(float(np.degrees(np.arccos(dot))))
    return np.asarray(rows, dtype=float)


def compute(df: pd.DataFrame, n_points: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    curve_rows = []
    for (attack, run_id), g in df.sort_values(["attack", "run_id", "step"]).groupby(["attack", "run_id"], sort=False):
        if int(g.final_success.max()) != 1:
            continue
        points = g[["pc1", "pc2"]].to_numpy(float)
        resampled = resample_polyline(points, n_points)
        if resampled is None:
            continue
        angles = turning_angles(resampled)
        if len(angles) == 0:
            continue
        for i, p in enumerate(resampled):
            curve_rows.append(
                {
                    "attack": attack,
                    "run_id": run_id,
                    "progress_idx": i,
                    "progress": i / max(n_points - 1, 1),
                    "x": float(p[0]),
                    "y": float(p[1]),
                }
            )
        for i, angle in enumerate(angles, start=1):
            rows.append(
                {
                    "attack": attack,
                    "run_id": run_id,
                    "turn_idx": i,
                    "progress": i / max(n_points - 2, 1),
                    "angle_deg": float(angle),
                    "path_length": float(np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1))),
                    "original_steps": int(len(points) - 1),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(curve_rows)


def summarize(angles: pd.DataFrame) -> pd.DataFrame:
    return (
        angles.groupby(["attack", "turn_idx"], as_index=False)
        .agg(
            n=("angle_deg", "size"),
            mean_angle_deg=("angle_deg", "mean"),
            median_angle_deg=("angle_deg", "median"),
            q25_angle_deg=("angle_deg", lambda x: float(np.percentile(x, 25))),
            q75_angle_deg=("angle_deg", lambda x: float(np.percentile(x, 75))),
        )
    )


def distribution_stats(angles: pd.DataFrame) -> pd.DataFrame:
    attacks = sorted(angles.attack.unique())
    rows = []
    for attack in attacks:
        d = angles[angles.attack == attack].angle_deg.to_numpy(float)
        rows.append(
            {
                "attack": attack,
                "n": int(len(d)),
                "mean": float(np.mean(d)),
                "median": float(np.median(d)),
                "q25": float(np.percentile(d, 25)),
                "q75": float(np.percentile(d, 75)),
                "std": float(np.std(d)),
            }
        )
    if set(["pgd", "square"]).issubset(set(attacks)):
        a = np.sort(angles[angles.attack == "pgd"].angle_deg.to_numpy(float))
        b = np.sort(angles[angles.attack == "square"].angle_deg.to_numpy(float))
        grid = np.sort(np.unique(np.concatenate([a, b])))
        cdf_a = np.searchsorted(a, grid, side="right") / max(len(a), 1)
        cdf_b = np.searchsorted(b, grid, side="right") / max(len(b), 1)
        ks = float(np.max(np.abs(cdf_a - cdf_b)))
        # 1D Wasserstein with quantile interpolation.
        qs = np.linspace(0, 1, 1001)
        wa = np.quantile(a, qs)
        wb = np.quantile(b, qs)
        wasserstein = float(np.mean(np.abs(wa - wb)))
        rows.append({"attack": "pgd_vs_square", "n": int(min(len(a), len(b))), "mean": ks, "median": wasserstein, "q25": np.nan, "q75": np.nan, "std": np.nan})
    return pd.DataFrame(rows)


def plot_angle_over_progress(angles: pd.DataFrame, summary: pd.DataFrame, out_path: Path):
    colors = {"pgd": "#111827", "square": "#2563eb"}
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for attack in ["pgd", "square"]:
        d = angles[angles.attack == attack]
        s = summary[summary.attack == attack]
        ax.scatter(d.progress, d.angle_deg, s=8, alpha=0.12, color=colors[attack], edgecolors="none")
        ax.plot(s.turn_idx / max(float(s.turn_idx.max()), 1.0), s.median_angle_deg, color=colors[attack], lw=2.2, label=f"{attack.upper()} median")
        ax.fill_between(
            s.turn_idx.to_numpy(float) / max(float(s.turn_idx.max()), 1.0),
            s.q25_angle_deg.to_numpy(float),
            s.q75_angle_deg.to_numpy(float),
            color=colors[attack],
            alpha=0.12,
        )
    ax.set_xlabel("normalized arc-length progress")
    ax.set_ylabel("turning angle after resampling (degrees)")
    ax.set_ylim(0, 180)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    ax.set_title("Progress-normalized curvature of successful trajectories")
    fig.savefig(out_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_hist(angles: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    bins = np.linspace(0, 180, 46)
    for attack, color in [("pgd", "#111827"), ("square", "#2563eb")]:
        d = angles[angles.attack == attack].angle_deg
        ax.hist(d, bins=bins, density=True, histtype="step", lw=2.2, color=color, label=attack.upper())
    ax.set_xlabel("progress-normalized turning angle (degrees)")
    ax.set_ylabel("density")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    ax.set_title("Turning-angle distribution after arc-length resampling")
    fig.savefig(out_path.with_name(out_path.name + "_hist").with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_name(out_path.name + "_hist").with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.timeseries_csv)
    angles, curves = compute(df, args.resample_points)
    summary = summarize(angles)
    stats = distribution_stats(angles)
    stem = Path(args.timeseries_csv).stem.replace("_timeseries", f"_progress_curvature_r{args.resample_points}")
    angles.to_csv(out_dir / f"{stem}_angles.csv", index=False)
    curves.to_csv(out_dir / f"{stem}_resampled_curves.csv", index=False)
    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)
    stats.to_csv(out_dir / f"{stem}_distribution_stats.csv", index=False)
    plot_angle_over_progress(angles, summary, out_dir / stem)
    plot_hist(angles, out_dir / stem)
    print(f"[SAVED] {out_dir / (stem + '.png')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeseries-csv", required=True)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/pgd_square_class_colored_flow")
    p.add_argument("--resample-points", type=int, default=100)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
