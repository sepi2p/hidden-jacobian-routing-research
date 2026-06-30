#!/usr/bin/env python3
"""Compare off-flow/on-flow images and success-flow/random-basis entry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.offflow_entry_diagnostic import (  # noqa: E402
    LAYER_MAP,
    feature_vector,
    load_success_basis,
    pgd_step,
    projection_energy,
    score_initial_flow,
    select_clean_correct_class,
)
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import load_model, margin  # noqa: E402


def random_basis(dim: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((dim, k)).astype(np.float32)
    Q, _r = np.linalg.qr(X)
    return Q[:, :k].T.astype(np.float32)


def attack_and_track_controls(
    dataset,
    wrapper,
    layer: str,
    candidates: pd.DataFrame,
    success_basis: np.ndarray,
    random_basis_: np.ndarray,
    eps: float,
    step_size: float,
    steps: int,
    device,
):
    rows = []
    for image_ord, row in candidates.reset_index(drop=True).iterrows():
        idx = int(row.dataset_idx)
        label = int(row.label)
        x_cpu, _ = dataset[idx]
        x0 = x_cpu.unsqueeze(0).to(device)
        x_adv = x0.clone()
        y = torch.tensor([label], device=device)
        prev_h = None
        crossed = False
        for step in range(steps + 1):
            logits, h = feature_vector(wrapper, x_adv, layer)
            pred = int(logits.argmax(1).item())
            if prev_h is None:
                success_pe = np.nan
                random_pe = np.nan
                feature_step_norm = np.nan
            else:
                v = h - prev_h
                success_pe = projection_energy(v, success_basis)
                random_pe = projection_energy(v, random_basis_)
                feature_step_norm = float(np.linalg.norm(v))
            now_success = pred != label
            rows.append(
                {
                    "group": row.group,
                    "image_ord": int(image_ord),
                    "dataset_idx": idx,
                    "label": label,
                    "step": int(step),
                    "pred": pred,
                    "margin": float(margin(logits, y).item()),
                    "success_projection_energy": success_pe,
                    "random_projection_energy": random_pe,
                    "feature_step_norm": feature_step_norm,
                    "success": int(now_success),
                    "first_success": int(now_success and not crossed),
                    "initial_success_projection_energy": float(row.initial_projection_energy),
                }
            )
            if now_success:
                crossed = True
                break
            prev_h = h
            if step < steps:
                x_adv = pgd_step(wrapper, x_adv, x0, y, eps, step_size)
    return pd.DataFrame(rows)


def summarize_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, image_ord), g in tracks.groupby(["group", "image_ord"]):
        g = g.sort_values("step")
        success_step = g[g.success == 1].step.min() if (g.success == 1).any() else np.nan
        pre = g[g.step > 0]
        cross = pre[pre.step == success_step] if not np.isnan(success_step) else pre.iloc[0:0]
        before = pre[pre.step < success_step] if not np.isnan(success_step) else pre
        rows.append(
            {
                "group": group,
                "image_ord": int(image_ord),
                "dataset_idx": int(g.dataset_idx.iloc[0]),
                "initial_success_projection_energy": float(g.initial_success_projection_energy.iloc[0]),
                "success": int((g.success == 1).any()),
                "success_step": float(success_step) if not np.isnan(success_step) else np.nan,
                "mean_pre_success_success_pe": float(before.success_projection_energy.mean()) if len(before) else np.nan,
                "max_pre_success_success_pe": float(before.success_projection_energy.max()) if len(before) else np.nan,
                "crossing_success_pe": float(cross.success_projection_energy.iloc[0]) if len(cross) else np.nan,
                "crossing_random_pe": float(cross.random_projection_energy.iloc[0]) if len(cross) else np.nan,
                "start_margin": float(g.margin.iloc[0]),
                "final_margin": float(g.margin.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("group", as_index=False)
        .agg(
            n=("success", "size"),
            success_rate=("success", "mean"),
            median_initial_success_pe=("initial_success_projection_energy", "median"),
            median_crossing_success_pe=("crossing_success_pe", "median"),
            median_crossing_random_pe=("crossing_random_pe", "median"),
            median_success_step=("success_step", "median"),
            mean_success_step=("success_step", "mean"),
            median_margin_drop=("final_margin", lambda x: float(np.nanmedian(summary.loc[x.index, "start_margin"] - x))),
        )
    )


def plot_tracks(tracks: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4), sharex="col", constrained_layout=True)
    colors = {"offflow": "#2563eb", "onflow": "#dc2626"}
    for group, g0 in tracks.groupby("group"):
        for image_ord, g in g0.groupby("image_ord"):
            g = g.sort_values("step")
            axes[0, 0].plot(g.step, g.success_projection_energy, color=colors[group], alpha=0.17, lw=0.9)
            axes[0, 1].plot(g.step, g.random_projection_energy, color=colors[group], alpha=0.17, lw=0.9)
            axes[1, 0].plot(g.step, g.margin, color=colors[group], alpha=0.17, lw=0.9)
        med = g0.groupby("step", as_index=False).agg(
            success_pe=("success_projection_energy", "median"),
            random_pe=("random_projection_energy", "median"),
            margin=("margin", "median"),
        )
        axes[0, 0].plot(med.step, med.success_pe, color=colors[group], lw=2.4, label=group)
        axes[0, 1].plot(med.step, med.random_pe, color=colors[group], lw=2.4, label=group)
        axes[1, 0].plot(med.step, med.margin, color=colors[group], lw=2.4, label=group)
    axes[0, 0].set_title("Projection into learned success-flow basis")
    axes[0, 1].set_title("Projection into random basis")
    axes[1, 0].set_title("Margin")
    axes[1, 1].axis("off")
    for ax in [axes[0, 0], axes[0, 1]]:
        ax.set_ylim(0, 1)
        ax.set_ylabel("projection energy")
    axes[1, 0].axhline(0, color="black", lw=0.8, alpha=0.3)
    axes[1, 0].set_ylabel("true-vs-best-other margin")
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xlabel("PGD step")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
    fig.suptitle("Off-flow vs on-flow PGD entry into learned success-flow coordinates", fontsize=12)
    fig.savefig(out_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_crossing(summary: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), constrained_layout=True)
    colors = {"offflow": "#2563eb", "onflow": "#dc2626"}
    for group, g in summary.groupby("group"):
        axes[0].scatter(g.initial_success_projection_energy, g.crossing_success_pe, color=colors[group], alpha=0.75, label=group)
        axes[1].scatter(g.crossing_random_pe, g.crossing_success_pe, color=colors[group], alpha=0.75, label=group)
    axes[0].plot([0, 1], [0, 1], color="black", alpha=0.25)
    axes[0].set_xlabel("initial learned-basis PE")
    axes[0].set_ylabel("crossing learned-basis PE")
    axes[1].plot([0, 1], [0, 1], color="black", alpha=0.25)
    axes[1].set_xlabel("crossing random-basis PE")
    axes[1].set_ylabel("crossing learned-basis PE")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
    fig.suptitle("Boundary-crossing projection energy controls", fontsize=12)
    fig.savefig(out_path.with_name(out_path.stem + "_crossing_controls").with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_name(out_path.stem + "_crossing_controls").with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    wrapper = load_model(args.model, device)
    layer = LAYER_MAP[args.model][args.layer_group]
    success_basis, explained = load_success_basis(Path(args.success_feature_npz), args.k)
    rand_basis = random_basis(success_basis.shape[1], args.k, args.seed + 1009)
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0 if args.step_size > 0 else eps / max(args.steps, 1)
    selected = select_clean_correct_class(dataset, wrapper, args.class_id, args.candidate_pool, device)
    scores = score_initial_flow(dataset, wrapper, layer, selected, success_basis, eps, step_size, device)
    off = scores.sort_values("initial_projection_energy").head(args.n_each).copy()
    on = scores.sort_values("initial_projection_energy").tail(args.n_each).copy()
    off["group"] = "offflow"
    on["group"] = "onflow"
    candidates = pd.concat([off, on], ignore_index=True)
    tracks = attack_and_track_controls(dataset, wrapper, layer, candidates, success_basis, rand_basis, eps, step_size, args.steps, device)
    summary = summarize_tracks(tracks)
    agg = aggregate(summary)
    stem = f"off_on_control_{args.model}_{args.layer_group}_class{args.class_id}_k{args.k}_n{args.n_each}"
    scores.to_csv(out_dir / f"{stem}_candidate_scores.csv", index=False)
    candidates.to_csv(out_dir / f"{stem}_selected.csv", index=False)
    tracks.to_csv(out_dir / f"{stem}_tracks.csv", index=False)
    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)
    agg.to_csv(out_dir / f"{stem}_aggregate.csv", index=False)
    plot_tracks(tracks, out_dir / stem)
    plot_crossing(summary, out_dir / stem)
    with open(out_dir / f"{stem}_metadata.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "layer": layer,
                "success_basis_explained": [float(x) for x in explained],
                "n_selected_clean_correct": len(selected),
                "aggregate": agg.to_dict(orient="records"),
                "note": "Off/on groups are lowest/highest initial first-PGD-step projection energy into learned class success-flow basis. Random basis is a same-dimensional orthonormal control.",
            },
            f,
            indent=2,
        )
    print(f"[SAVED] {out_dir / (stem + '.png')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--success-feature-npz", required=True)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/offflow_entry_diagnostic")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer-group", default="penultimate", choices=["hidden", "penultimate", "logits"])
    p.add_argument("--class-id", type=int, default=0)
    p.add_argument("--candidate-pool", type=int, default=300)
    p.add_argument("--n-each", type=int, default=40)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--step-size", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=31)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
