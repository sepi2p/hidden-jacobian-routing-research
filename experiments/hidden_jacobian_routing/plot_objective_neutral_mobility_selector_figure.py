#!/usr/bin/env python3
"""Build the objective-neutral mobility and margin-selection figure."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)


def first_number(cell: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", cell)
    if not match:
        raise ValueError(f"No numeric value found in cell: {cell!r}")
    return float(match.group(0))


def parse_mobility_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    rows = []
    for line in path.read_text().splitlines():
        if "transport" not in line or "&" not in line:
            continue
        cells = [c.strip().rstrip("\\") for c in line.split("&")]
        if len(cells) < 7:
            continue
        rows.append(
            {
                "basis": cells[0].replace(" transport", ""),
                "rho": first_number(cells[1]),
                "hi_lo_auroc": first_number(cells[2]),
                "low_asr": first_number(cells[3]),
                "high_asr": first_number(cells[4]),
                "ratio": first_number(cells[5]),
                "adv_auroc": first_number(cells[6]),
            }
        )
    return pd.DataFrame(rows)


def parse_selector_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    rows = []
    for line in path.read_text().splitlines():
        if "&" not in line or any(skip in line for skip in ("toprule", "midrule", "bottomrule", "Selector")):
            continue
        cells = [c.strip().rstrip("\\") for c in line.split("&")]
        if len(cells) < 4:
            continue
        rows.append(
            {
                "selector": cells[0].replace("$\\times$", "x"),
                "top1": first_number(cells[1]),
                "top5": first_number(cells[2]),
                "top10": first_number(cells[3]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mobility-table", default="artifacts/table_inputs/objective_neutral_mobility.csv")
    p.add_argument("--selector-table", default="artifacts/table_inputs/two_stage_selector_sweep.csv")
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/figure7_objective_neutral_mobility_selector")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mobility = parse_mobility_table(Path(args.mobility_table))
    selector = parse_selector_table(Path(args.selector_table))
    keep = ["Random", "Mobility", "Margin", "Mobility x margin"]
    selector_small = selector[selector["selector"].isin(keep)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    colors = {"PGD": "#3267d6", "Square": "#d84a3a"}

    ax = axes[0]
    x = np.arange(len(mobility))
    width = 0.34
    ax.bar(x - width / 2, mobility["rho"], width=width, color="#6f8fd8", label=r"$\rho(m,E)$")
    ax.bar(x + width / 2, mobility["hi_lo_auroc"], width=width, color="#5bbd91", label="hi/lo AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(mobility["basis"])
    ax.set_ylim(0, 1.0)
    ax.set_title("A. mobility predicts transport energy")
    ax.set_ylabel("score")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.16)

    ax = axes[1]
    x = np.arange(len(mobility))
    ax.bar(x - width / 2, mobility["low_asr"], width=width, color="#cfcfcf", label="low energy")
    ax.bar(x + width / 2, mobility["high_asr"], width=width, color="#6f55a3", label="high energy")
    ax.set_xticks(x)
    ax.set_xticklabels(mobility["basis"])
    ax.set_ylim(0, max(mobility["high_asr"].max() * 1.35, 12))
    ax.set_title("B. high-energy random moves")
    ax.set_ylabel("post-hoc ASR (%)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.16)

    ax = axes[2]
    label_map = {
        "Random": "random",
        "Mobility": "mobility",
        "Margin": "margin",
        "Mobility x margin": "mobility\nx margin",
    }
    selector_small["label"] = selector_small["selector"].map(label_map)
    ax.bar(selector_small["label"], selector_small["top5"], color=["#bdbdbd", "#6f55a3", "#e18a2c", "#4f9f73"])
    ax.set_ylim(0, max(selector_small["top5"].max() * 1.25, 36))
    ax.set_title("C. margin is needed for selection")
    ax.set_ylabel("top-5 candidate ASR (%)")
    ax.grid(axis="y", alpha=0.16)

    fig.suptitle("Objective-neutral mobility is a proposal signal, not a selector", fontsize=10)
    png = out / "objective_neutral_mobility_selector_summary.png"
    pdf = out / "objective_neutral_mobility_selector_summary.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    mobility.to_csv(out / "mobility_values_used.csv", index=False)
    selector_small.to_csv(out / "selector_values_used.csv", index=False)
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
