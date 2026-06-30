#!/usr/bin/env python3
"""Test whether off-flow class-0 points enter the success-flow basis before failure.

This diagnostic uses a saved class-0 feature-displacement archive to build a
local-step success-flow PCA basis. It then finds clean-correct class-0 images
whose first PGD feature step has low projection energy into that basis, attacks
them, and tracks projection energy and margin over time.
"""

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

from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import (  # noqa: E402
    load_model,
    margin,
    project_linf,
)


LAYER_MAP = {
    "bbb_resnet50": {"hidden": "layer4", "penultimate": "avgpool", "logits": "logits"},
    "bbb_vgg19_bn": {"hidden": "block5", "penultimate": "penultimate", "logits": "logits"},
    "bbb_densenet": {"hidden": "denseblock3", "penultimate": "penultimate", "logits": "logits"},
    "bbb_inception_v3": {"hidden": "mixed6", "penultimate": "penultimate", "logits": "logits"},
}


def load_success_basis(path: Path, k: int):
    z = np.load(path, allow_pickle=False)
    disp = z["feature_displacements"].astype(np.float32)
    meta = pd.DataFrame(json.loads(str(z["meta_json"])))
    local = []
    for _rid, g in meta.sort_values(["run_id", "step"]).groupby("run_id", sort=False):
        if int(g.final_success.max()) != 1:
            continue
        idx = g.index.to_numpy()
        for a, b in zip(idx[:-1], idx[1:]):
            v = disp[b] - disp[a]
            if np.linalg.norm(v) > 1e-12:
                local.append(v)
    X = np.stack(local).astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(X, full_matrices=False)
    explained = (s * s) / max(float(np.sum(s * s)), 1e-12)
    return vt[:k].astype(np.float32), explained[:k]


def feature_vector(wrapper, x: torch.Tensor, layer: str):
    with torch.no_grad():
        logits, feats, _raw = wrapper.forward_with_features(x)
    if layer == "logits":
        h = logits.detach().cpu().numpy()[0].astype(np.float32)
    else:
        h = feats[layer].detach().cpu().numpy()[0].astype(np.float32)
    return logits.detach(), h.reshape(-1)


def projection_energy(v: np.ndarray, basis: np.ndarray) -> float:
    n2 = float(np.dot(v, v))
    if n2 <= 1e-12:
        return 0.0
    coeff = basis @ v
    return float(np.dot(coeff, coeff) / n2)


def pgd_step(wrapper, x_adv: torch.Tensor, x0: torch.Tensor, y: torch.Tensor, eps: float, step_size: float):
    probe = x_adv.detach().requires_grad_(True)
    logits = wrapper(probe)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, probe)[0]
    return project_linf(x_adv + step_size * grad.sign(), x0, eps)


def select_clean_correct_class(dataset, wrapper, class_id: int, max_images: int, device):
    rows = []
    for idx in range(len(dataset)):
        if len(rows) >= max_images:
            break
        x_cpu, y0 = dataset[idx]
        if int(y0) != class_id:
            continue
        x = x_cpu.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(wrapper(x).argmax(1).item())
        if pred == class_id:
            rows.append((idx, int(y0)))
    return rows


def score_initial_flow(dataset, wrapper, layer: str, selected, basis: np.ndarray, eps: float, step_size: float, device):
    rows = []
    for idx, label in selected:
        x_cpu, _ = dataset[idx]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        logits0, h0 = feature_vector(wrapper, x0, layer)
        x1 = pgd_step(wrapper, x0, x0, y, eps, step_size)
        logits1, h1 = feature_vector(wrapper, x1, layer)
        v = h1 - h0
        rows.append(
            {
                "dataset_idx": int(idx),
                "label": int(label),
                "initial_projection_energy": projection_energy(v, basis),
                "initial_margin": float(margin(logits0, y).item()),
                "after_one_margin": float(margin(logits1, y).item()),
                "after_one_pred": int(logits1.argmax(1).item()),
            }
        )
    return pd.DataFrame(rows)


def attack_and_track(dataset, wrapper, layer: str, candidates: pd.DataFrame, basis: np.ndarray, eps: float, step_size: float, steps: int, device):
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
            m = float(margin(logits, y).item())
            if prev_h is None:
                pe = np.nan
                step_norm = np.nan
            else:
                v = h - prev_h
                pe = projection_energy(v, basis)
                step_norm = float(np.linalg.norm(v))
            now_success = pred != label
            rows.append(
                {
                    "image_ord": int(image_ord),
                    "dataset_idx": idx,
                    "label": label,
                    "step": int(step),
                    "pred": pred,
                    "margin": m,
                    "projection_energy": pe,
                    "feature_step_norm": step_norm,
                    "success": int(now_success),
                    "first_success": int(now_success and not crossed),
                    "initial_projection_energy": float(row.initial_projection_energy),
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
    out = []
    for image_ord, g in tracks.groupby("image_ord"):
        g = g.sort_values("step")
        success_step = g[g.success == 1].step.min() if (g.success == 1).any() else np.nan
        pre = g[g.step > 0]
        if not np.isnan(success_step):
            before = pre[pre.step < success_step]
            cross = pre[pre.step == success_step]
        else:
            before = pre
            cross = pre.iloc[0:0]
        out.append(
            {
                "image_ord": int(image_ord),
                "dataset_idx": int(g.dataset_idx.iloc[0]),
                "initial_projection_energy": float(g.initial_projection_energy.iloc[0]),
                "success": int((g.success == 1).any()),
                "success_step": float(success_step) if not np.isnan(success_step) else np.nan,
                "mean_pre_success_projection_energy": float(before.projection_energy.mean()) if len(before) else np.nan,
                "max_pre_success_projection_energy": float(before.projection_energy.max()) if len(before) else np.nan,
                "crossing_projection_energy": float(cross.projection_energy.iloc[0]) if len(cross) else np.nan,
                "start_margin": float(g.margin.iloc[0]),
                "final_margin": float(g.margin.iloc[-1]),
            }
        )
    return pd.DataFrame(out)


def plot_tracks(tracks: pd.DataFrame, summary: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    for image_ord, g in tracks.groupby("image_ord"):
        g = g.sort_values("step")
        axes[0].plot(g.step, g.projection_energy, color="#2563eb", alpha=0.22, lw=1.0)
        axes[1].plot(g.step, g.margin, color="#111827", alpha=0.22, lw=1.0)
        fs = g[g.first_success == 1]
        if not fs.empty:
            axes[0].scatter(fs.step, fs.projection_energy, color="#dc2626", s=18, zorder=3)
            axes[1].scatter(fs.step, fs.margin, color="#dc2626", s=18, zorder=3)
    med = tracks.groupby("step", as_index=False).agg(
        median_projection_energy=("projection_energy", "median"),
        median_margin=("margin", "median"),
    )
    axes[0].plot(med.step, med.median_projection_energy, color="#1d4ed8", lw=2.4, label="median")
    axes[1].plot(med.step, med.median_margin, color="black", lw=2.4, label="median")
    axes[0].set_xlabel("PGD step")
    axes[0].set_ylabel("projection energy into success-flow basis")
    axes[0].set_ylim(0, 1)
    axes[1].axhline(0, color="#dc2626", lw=1.0, alpha=0.45)
    axes[1].set_xlabel("PGD step")
    axes[1].set_ylabel("true-vs-best-other margin")
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
    fig.suptitle("Do off-flow class-0 images enter success-flow coordinates before misclassification?", fontsize=12)
    fig.savefig(out_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    ok = summary[summary.success == 1].copy()
    ax.scatter(ok.initial_projection_energy, ok.crossing_projection_energy, color="#2563eb", alpha=0.75)
    ax.plot([0, 1], [0, 1], color="black", lw=1.0, alpha=0.25)
    ax.set_xlim(0, max(0.25, float(ok[["initial_projection_energy", "crossing_projection_energy"]].max().max()) * 1.05))
    ax.set_ylim(0, 1)
    ax.set_xlabel("initial one-step projection energy")
    ax.set_ylabel("boundary-crossing step projection energy")
    ax.grid(alpha=0.18)
    ax.set_title("Projection energy at entry vs boundary crossing")
    fig.savefig(out_path.with_name(out_path.stem + "_entry_vs_crossing").with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_path.with_name(out_path.stem + "_entry_vs_crossing").with_suffix(".pdf"), bbox_inches="tight")
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
    basis, explained = load_success_basis(Path(args.success_feature_npz), args.k)
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0 if args.step_size > 0 else eps / max(args.steps, 1)
    selected = select_clean_correct_class(dataset, wrapper, args.class_id, args.candidate_pool, device)
    scores = score_initial_flow(dataset, wrapper, layer, selected, basis, eps, step_size, device)
    candidates = scores.sort_values("initial_projection_energy").head(args.n_offflow).copy()
    tracks = attack_and_track(dataset, wrapper, layer, candidates, basis, eps, step_size, args.steps, device)
    summary = summarize_tracks(tracks)

    stem = f"offflow_entry_{args.model}_{args.layer_group}_class{args.class_id}_k{args.k}_n{len(candidates)}"
    scores.to_csv(out_dir / f"{stem}_candidate_scores.csv", index=False)
    candidates.to_csv(out_dir / f"{stem}_selected_offflow.csv", index=False)
    tracks.to_csv(out_dir / f"{stem}_tracks.csv", index=False)
    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)
    plot_tracks(tracks, summary, out_dir / stem)
    with open(out_dir / f"{stem}_metadata.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "layer": layer,
                "success_basis_explained": [float(x) for x in explained],
                "n_selected_clean_correct": len(selected),
                "n_offflow": int(len(candidates)),
                "success_rate": float(summary.success.mean()) if len(summary) else np.nan,
                "median_initial_projection_energy": float(candidates.initial_projection_energy.median()),
                "median_crossing_projection_energy": float(summary.crossing_projection_energy.median()),
                "note": "Off-flow images are clean-correct class images with lowest first-PGD-step projection energy into the saved class success-flow basis.",
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
    p.add_argument("--n-offflow", type=int, default=40)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--step-size", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=31)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
