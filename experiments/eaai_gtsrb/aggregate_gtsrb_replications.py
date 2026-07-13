#!/usr/bin/env python3
"""Aggregate the three GTSRB application runs and generate paper artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = (
    (20260713, 1307, None),
    (20260714, 1308, "replications/seed_20260714"),
    (20260715, 1309, "replications/seed_20260715"),
)
ARCHITECTURES = ("resnet18", "convnext_tiny")
LABELS = {"resnet18": "ResNet-18", "convnext_tiny": "ConvNeXt-Tiny"}
COLORS = {"resnet18": "#167D8D", "convnext_tiny": "#D45D3F"}


def read_json(path: Path):
    return json.loads(path.read_text())


def run_root(base: Path, relative: str | None) -> Path:
    return base if relative is None else base / relative


def collect(base: Path):
    rows = []
    coordinate = []
    for train_seed, split_seed, relative in RUNS:
        root = run_root(base, relative)
        for architecture in ARCHITECTURES:
            checkpoint = read_json(root / "checkpoints" / architecture / "complete.json")
            manifest = read_json(root / "diagnostics" / architecture / "manifest.json")
            core = manifest["core_summary"]
            jvp = manifest["jvp_summary"]
            attack = manifest["selected_weak_attack"]
            rows.append(
                {
                    "train_seed": train_seed,
                    "split_seed": split_seed,
                    "architecture": architecture,
                    "test_accuracy": checkpoint["test"]["accuracy"],
                    "test_nll": checkpoint["test"]["nll"],
                    "test_ece15": checkpoint["test"]["ece15"],
                    "weak_eps_255": attack["eps_255"],
                    "weak_steps": attack["steps"],
                    "weak_step_size_255": attack["step_size_255"],
                    **core,
                    **jvp,
                }
            )
            stress = pd.read_csv(
                root / "diagnostics" / architecture / "function_preserving_coordinate_stress.csv"
            )
            stress.insert(0, "architecture", architecture)
            stress.insert(0, "split_seed", split_seed)
            stress.insert(0, "train_seed", train_seed)
            coordinate.append(stress)
    return pd.DataFrame(rows), pd.concat(coordinate, ignore_index=True)


def architecture_summary(seed_level: pd.DataFrame):
    records = []
    metrics = [
        "test_accuracy",
        "test_nll",
        "test_ece15",
        "final_test_auroc",
        "incremental_auprc",
        "median_fd_jvp_cosine",
        "median_stabilized_residual_ratio",
        "fd_jvp_norm_spearman",
    ]
    for architecture, group in seed_level.groupby("architecture"):
        row = {
            "architecture": architecture,
            "runs": len(group),
            "total_final_test_n": int(group.final_test_n.sum()),
            "total_successes": int(group.final_test_successes.sum()),
            "total_failures": int(group.final_test_failures.sum()),
            "selected_layers": ", ".join(sorted(group.selected_layer.unique())),
            "selected_k_values": ", ".join(str(int(x)) for x in sorted(group.selected_k.unique())),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        records.append(row)
    return pd.DataFrame(records)


def paper_figure(seed_level: pd.DataFrame, coordinate: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.15), constrained_layout=True)

    ax = axes[0, 0]
    for offset, architecture in zip((-0.08, 0.08), ARCHITECTURES):
        group = seed_level[seed_level.architecture == architecture].sort_values("train_seed")
        x = np.arange(len(group)) + offset
        y = group.final_test_auroc.to_numpy()
        lo = y - group.final_test_auroc_ci_low.to_numpy()
        hi = group.final_test_auroc_ci_high.to_numpy() - y
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([lo, hi]),
            fmt="o",
            capsize=3,
            color=COLORS[architecture],
            label=LABELS[architecture],
        )
        ax.axhline(y.mean(), color=COLORS[architecture], alpha=0.35, linewidth=1)
    ax.set_xticks(range(3), ["Seed 1", "Seed 2", "Seed 3"])
    ax.set_ylim(0.5, 1.01)
    ax.set_ylabel("Success/failure AUROC")
    ax.set_title("(a) Held-out trajectory separation", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    width = 0.34
    x = np.arange(2)
    for run_idx in range(3):
        values = []
        for architecture in ARCHITECTURES:
            group = seed_level[seed_level.architecture == architecture].sort_values("train_seed")
            values.append(group.iloc[run_idx].incremental_auprc)
        ax.plot(x, values, color="#8C8C8C", alpha=0.5, linewidth=1)
    means = [
        seed_level[seed_level.architecture == architecture].incremental_auprc.mean()
        for architecture in ARCHITECTURES
    ]
    sds = [
        seed_level[seed_level.architecture == architecture].incremental_auprc.std(ddof=1)
        for architecture in ARCHITECTURES
    ]
    ax.bar(x, means, yerr=sds, width=width, color=[COLORS[a] for a in ARCHITECTURES], capsize=3)
    ax.set_xticks(x, [LABELS[a] for a in ARCHITECTURES])
    ax.set_ylabel("Incremental AUPRC")
    ax.set_title("(b) Value beyond difficulty/progress", loc="left", fontsize=10)

    ax = axes[1, 0]
    metric_names = ["FD/JVP cosine", "Residual ratio"]
    positions = np.arange(2)
    for arch_idx, architecture in enumerate(ARCHITECTURES):
        group = seed_level[seed_level.architecture == architecture]
        means = [
            group.median_fd_jvp_cosine.mean(),
            group.median_stabilized_residual_ratio.mean(),
        ]
        sds = [
            group.median_fd_jvp_cosine.std(ddof=1),
            group.median_stabilized_residual_ratio.std(ddof=1),
        ]
        ax.bar(
            positions + (arch_idx - 0.5) * width,
            means,
            yerr=sds,
            width=width,
            color=COLORS[architecture],
            label=LABELS[architecture],
            capsize=3,
        )
    ax.set_xticks(positions, metric_names)
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) Finite-step Jacobian agreement", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for architecture in ARCHITECTURES:
        group = coordinate[coordinate.architecture == architecture]
        summary = group.groupby("sigma").auroc_k20.agg(["mean", "std"]).reset_index()
        ax.errorbar(
            summary.sigma,
            summary["mean"],
            yerr=summary["std"].fillna(0),
            marker="o",
            capsize=3,
            color=COLORS[architecture],
            label=LABELS[architecture],
        )
    ax.set_xticks([0, 0.5, 1.0, 2.0])
    ax.set_ylim(0.5, 1.01)
    ax.set_xlabel("Lognormal channel-scaling sigma")
    ax.set_ylabel("Projection AUROC (k=20)")
    ax.set_title("(d) Function-preserving coordinate stress", loc="left", fontsize=10)

    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def latex_table(summary: pd.DataFrame, path: Path):
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Model & Test acc. & Success/failure & AUROC & $\\Delta$AUPRC & FD/JVP cos. \\\\",
        "\\midrule",
    ]
    for architecture in ARCHITECTURES:
        row = summary[summary.architecture == architecture].iloc[0]
        lines.append(
            f"{LABELS[architecture]} & "
            f"{100 * row.test_accuracy_mean:.2f}$\\pm${100 * row.test_accuracy_sd:.2f} & "
            f"{int(row.total_successes)}/{int(row.total_failures)} & "
            f"{row.final_test_auroc_mean:.3f}$\\pm${row.final_test_auroc_sd:.3f} & "
            f"{row.incremental_auprc_mean:+.3f}$\\pm${row.incremental_auprc_sd:.3f} & "
            f"{row.median_fd_jvp_cosine_mean:.3f}$\\pm${row.median_fd_jvp_cosine_sd:.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def latex_seed_table(seed_level: pd.DataFrame, path: Path):
    lines = [
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "Model & Seed & Acc. & Layer/$k$ & S/F & AUROC [95\\% CI] & $\\Delta$AUPRC & JVP cos. & Residual \\\\",
        "\\midrule",
    ]
    for architecture in ARCHITECTURES:
        group = seed_level[seed_level.architecture == architecture].sort_values("train_seed")
        for run_number, (_, row) in enumerate(group.iterrows(), start=1):
            lines.append(
                f"{LABELS[architecture]} & {run_number} & {100 * row.test_accuracy:.2f} & "
                f"{row.selected_layer}/{int(row.selected_k)} & "
                f"{int(row.final_test_successes)}/{int(row.final_test_failures)} & "
                f"{row.final_test_auroc:.3f} "
                f"[{row.final_test_auroc_ci_low:.3f}, {row.final_test_auroc_ci_high:.3f}] & "
                f"{row.incremental_auprc:+.3f} & {row.median_fd_jvp_cosine:.3f} & "
                f"{row.median_stabilized_residual_ratio:.3f} \\\\"
            )
        if architecture != ARCHITECTURES[-1]:
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def latex_coordinate_table(coordinate: pd.DataFrame, path: Path):
    grouped = (
        coordinate.groupby(["architecture", "sigma"])
        .agg(
            auroc_mean=("auroc_k20", "mean"),
            auroc_sd=("auroc_k20", "std"),
            dim80_mean=("dim80", "mean"),
            dim80_sd=("dim80", "std"),
            max_logit_error=("max_compensated_logit_error", "max"),
        )
        .reset_index()
    )
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Model & Scaling $\\sigma$ & AUROC ($k=20$) & dim80 & Max logit error \\\\",
        "\\midrule",
    ]
    for architecture in ARCHITECTURES:
        group = grouped[grouped.architecture == architecture]
        for _, row in group.iterrows():
            lines.append(
                f"{LABELS[architecture]} & {row.sigma:.1f} & "
                f"{row.auroc_mean:.3f}$\\pm${row.auroc_sd:.3f} & "
                f"{row.dim80_mean:.1f}$\\pm${row.dim80_sd:.1f} & "
                f"{row.max_logit_error:.1e} \\\\"
            )
        if architecture != ARCHITECTURES[-1]:
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="analysis_outputs/eaai_gtsrb")
    parser.add_argument("--output-dir", default="analysis_outputs/eaai_gtsrb/aggregate")
    parser.add_argument("--figure-dir", default="generated/gtsrb/figures")
    parser.add_argument("--table-dir", default="generated/gtsrb/tables")
    args = parser.parse_args()

    base = Path(args.base_dir)
    output = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    table_dir = Path(args.table_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    seed_level, coordinate = collect(base)
    summary = architecture_summary(seed_level)
    seed_level.to_csv(output / "gtsrb_seed_level_summary.csv", index=False)
    coordinate.to_csv(output / "gtsrb_coordinate_stress_all_seeds.csv", index=False)
    summary.to_csv(output / "gtsrb_architecture_summary.csv", index=False)
    paper_figure(seed_level, coordinate, figure_dir / "gtsrb_application_audit")
    latex_table(summary, table_dir / "gtsrb_application_summary.tex")
    latex_seed_table(seed_level, table_dir / "gtsrb_seed_details.tex")
    latex_coordinate_table(coordinate, table_dir / "gtsrb_coordinate_stress.tex")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
