#!/usr/bin/env python3
"""Analyze target-class reachability along continued singular roads."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CIFAR10_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def ordered_unique(xs) -> list[int]:
    out: list[int] = []
    for x in xs:
        x = int(x)
        if not out or out[-1] != x:
            out.append(x)
    return out


def first_all_classes_step(preds: np.ndarray, steps: np.ndarray, n_classes: int = 10) -> int:
    seen = set()
    for pred, step in zip(preds, steps):
        seen.add(int(pred))
        if len(seen) >= n_classes:
            return int(step)
    return -1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--source-class", type=int, default=0)
    p.add_argument("--prefix", default="singular_road_target_sequences")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv).sort_values(["direction", "image_id", "step"])

    coverage_rows = []
    hit_rows = []
    road_names = []
    first_hit_grid = []

    for road_idx, ((direction, image_id), g) in enumerate(df.groupby(["direction", "image_id"], sort=True)):
        g = g.sort_values("step")
        preds = g.pred.to_numpy(dtype=int)
        steps = g.step.to_numpy(dtype=int)
        seq_full = ordered_unique(preds)
        visited = sorted(set(preds.tolist()))
        all_step = first_all_classes_step(preds, steps)
        road_name = f"{direction[0].upper()}-{int(image_id)}"
        road_names.append(road_name)
        row_hits = []
        coverage_rows.append(
            {
                "road_id": road_idx,
                "image_id": int(image_id),
                "direction": str(direction),
                "n_unique_classes": int(len(visited)),
                "visited_classes": " ".join(map(str, visited)),
                "class_sequence": "->".join(map(str, seq_full)),
                "all_classes_step": all_step,
                "reached_all_classes": int(all_step >= 0),
            }
        )
        for target in range(10):
            hit = g[g.pred == target]
            if len(hit):
                first_step = int(hit.step.iloc[0])
                prefix_preds = g[g.step <= first_step].pred.to_numpy(dtype=int)
                seq = ordered_unique(prefix_preds)
                hit_rows.append(
                    {
                        "road_id": road_idx,
                        "road_name": road_name,
                        "image_id": int(image_id),
                        "direction": str(direction),
                        "source_class": int(args.source_class),
                        "target_class": target,
                        "target_name": CIFAR10_NAMES[target],
                        "first_hit_step": first_step,
                        "sequence_length": len(seq),
                        "n_intermediate_classes": max(0, len(seq) - 2),
                        "direct": int(seq == [args.source_class, target] or target == args.source_class),
                        "sequence": "->".join(map(str, seq)),
                    }
                )
                row_hits.append(first_step)
            else:
                row_hits.append(np.nan)
        first_hit_grid.append(row_hits)

    coverage = pd.DataFrame(coverage_rows)
    hits = pd.DataFrame(hit_rows)

    # Best means fewest distinct class transitions; ties broken by earliest hit step.
    best_rows = []
    for target in range(10):
        subset = hits[hits.target_class == target].copy()
        if len(subset) == 0:
            best_rows.append(
                {
                    "target_class": target,
                    "target_name": CIFAR10_NAMES[target],
                    "reachable": 0,
                    "best_sequence": "",
                    "best_sequence_length": np.nan,
                    "best_first_hit_step": np.nan,
                    "best_image_id": np.nan,
                    "best_direction": "",
                    "direct_path_exists": 0,
                    "n_roads_reaching_target": 0,
                }
            )
            continue
        subset = subset.sort_values(["sequence_length", "first_hit_step", "road_id"])
        best = subset.iloc[0]
        best_rows.append(
            {
                "target_class": target,
                "target_name": CIFAR10_NAMES[target],
                "reachable": 1,
                "best_sequence": best.sequence,
                "best_sequence_length": int(best.sequence_length),
                "best_first_hit_step": int(best.first_hit_step),
                "best_image_id": int(best.image_id),
                "best_direction": str(best.direction),
                "direct_path_exists": int((subset.direct == 1).any()),
                "n_roads_reaching_target": int(subset.road_id.nunique()),
            }
        )

    best_df = pd.DataFrame(best_rows)
    coverage.to_csv(out / f"{args.prefix}_road_coverage.csv", index=False)
    hits.to_csv(out / f"{args.prefix}_all_first_hits.csv", index=False)
    best_df.to_csv(out / f"{args.prefix}_best_by_target.csv", index=False)

    # Plot target reachability.
    grid = np.asarray(first_hit_grid, dtype=float)
    fig = plt.figure(figsize=(12.0, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(grid, aspect="auto", interpolation="nearest", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("first hit step")
    ax0.set_title("First Step at Which Each Road Visits Each Class")
    ax0.set_xlabel("target class")
    ax0.set_ylabel("road")
    ax0.set_xticks(range(10))
    ax0.set_xticklabels([f"{i}\n{name[:4]}" for i, name in enumerate(CIFAR10_NAMES)], fontsize=8)
    ax0.set_yticks(range(len(road_names)))
    ax0.set_yticklabels(road_names, fontsize=6)

    ax1 = fig.add_subplot(gs[0, 1])
    targets = best_df.target_class.to_numpy(dtype=int)
    lengths = best_df.best_sequence_length.to_numpy(dtype=float)
    steps = best_df.best_first_hit_step.to_numpy(dtype=float)
    colors = ["#4C78A8" if bool(v) else "#BAB0AC" for v in best_df.direct_path_exists.to_numpy(dtype=int)]
    ax1.bar(targets, lengths, color=colors, alpha=0.9)
    ax1.set_title("Shortest Observed Class Sequence from Class 0")
    ax1.set_xlabel("target class")
    ax1.set_ylabel("distinct classes in sequence")
    ax1.set_xticks(targets)
    ax1.set_xticklabels([str(t) for t in targets])
    for _, row in best_df.iterrows():
        if row.reachable:
            label = f"{row.best_sequence}\nstep {int(row.best_first_hit_step)}"
            ax1.text(
                int(row.target_class),
                float(row.best_sequence_length) + 0.12,
                label,
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )
    ax1.set_ylim(0, max(5, np.nanmax(lengths) + 3))
    ax1.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#4C78A8", label="direct sequence exists"),
            plt.Rectangle((0, 0), 1, 1, color="#BAB0AC", label="requires intermediate class"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper left",
    )

    png = out / f"{args.prefix}.png"
    pdf = out / f"{args.prefix}.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)

    aggregate = {
        "n_roads": int(len(coverage)),
        "n_roads_reaching_all_classes": int((coverage.all_classes_step >= 0).sum()),
        "mean_unique_classes": float(coverage.n_unique_classes.mean()),
        "max_unique_classes": int(coverage.n_unique_classes.max()),
        "targets_reachable_by_any_road": int(best_df.reachable.sum()),
        "targets_with_direct_sequence": int(best_df.direct_path_exists.sum()),
    }
    pd.DataFrame([aggregate]).to_csv(out / f"{args.prefix}_aggregate.csv", index=False)

    print("Aggregate:")
    print(pd.DataFrame([aggregate]).to_string(index=False))
    print("\nBest target sequences:")
    print(best_df.to_string(index=False))
    print(f"\nsaved {png}")
    print(f"saved {pdf}")


if __name__ == "__main__":
    main()
