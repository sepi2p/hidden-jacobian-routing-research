#!/usr/bin/env python3
"""Compare early Square success predictors against transport-axis energy.

Predictors:
  A) learned class-flow transport-axis energy
  B) CE loss / margin
  C) random PCA-style axes
  D) clean-motion PCA axes
  E) gradient-only PCA axes
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar_attack_axis_projection import (  # noqa: E402
    feature_state_rows,
    square_trajectory,
)
from experiments.pure_af_geometry.analyze_cifar_global_vs_class_success_flow import LAYER_GROUPS  # noqa: E402
from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import get_npz, normalize_rows  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import (  # noqa: E402
    build_mu,
    eval_all,
    load_model,
    select_clean_correct,
)
from experiments.pure_af_geometry.run_cifar_pc_transport_mode_attack import build_pc_directions  # noqa: E402


def fit_pca_basis(x: np.ndarray, k: int) -> np.ndarray:
    x = normalize_rows(x.astype(np.float32))
    x = x[np.linalg.norm(x, axis=1) > 1e-12]
    if len(x) == 0:
        return np.empty((0, 0), dtype=np.float32)
    xc = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return vt[: min(k, vt.shape[0])].astype(np.float32)


def random_basis(d: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _r = np.linalg.qr(rng.standard_normal((d, min(k, d))).astype(np.float32))
    return q.T.astype(np.float32)


def build_bases(layerwise_dir: Path, top_k: int, seed: int):
    mu = build_mu(layerwise_dir)
    pc_dirs, pc_meta = build_pc_directions(mu, top_k)
    clean_npz = np.load(layerwise_dir / "clean_vectors.npz")
    grad_npz = np.load(layerwise_dir / "segment_grads.npz")
    bases = {}
    rows = []
    for group, mapping in LAYER_GROUPS.items():
        for model, layer in mapping.items():
            transport = []
            for pc in range(1, top_k + 1):
                v = pc_dirs.get((model, layer, pc))
                if v is not None:
                    transport.append(v)
            if not transport:
                continue
            transport = np.stack(transport).astype(np.float32)
            d = transport.shape[1]
            clean_key = f"clean__{model}__{layer}"
            grad_key = f"grads__{model}__{layer}"
            basis_map = {
                "transport_axis_energy": transport,
                "random_axis_energy": random_basis(d, top_k, seed + abs(hash((model, layer))) % 1_000_000),
            }
            if clean_key in clean_npz.files:
                basis_map["clean_motion_axis_energy"] = fit_pca_basis(clean_npz[clean_key], top_k)
            if grad_key in grad_npz.files:
                basis_map["gradient_only_axis_energy"] = fit_pca_basis(grad_npz[grad_key], top_k)
            for name, basis in basis_map.items():
                if basis.shape[0] == 0:
                    continue
                bases[(model, layer, name)] = basis
                rows.append({
                    "model": model,
                    "layer_group": group,
                    "layer": layer,
                    "basis_type": name,
                    "k": int(basis.shape[0]),
                    "d": int(basis.shape[1]),
                })
    return bases, pc_meta, pd.DataFrame(rows)


def projection_fraction(delta: np.ndarray, basis: np.ndarray) -> tuple[float, float, float]:
    coeff = basis @ delta
    energy = float(np.sum(coeff * coeff))
    total = float(np.sum(delta * delta))
    return energy / max(total, 1e-12), energy, total


def collect_model_rows(args, model_name: str, wrapper, dataset, selected, bases, layer_groups, device):
    rows = []
    eps = args.eps / 255.0
    for image_ord, (dataset_idx, label) in enumerate(selected):
        x_cpu, _ = dataset[dataset_idx]
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        states = square_trajectory(
            wrapper,
            x,
            y,
            eps,
            args.square_steps,
            args.seed + image_ord * 1009 + len(model_name) * 17,
            args.square_min_size,
        )
        final_eval = eval_all({model_name: wrapper}, states[-1], y)[model_name]
        state_rows, feat_by_step = feature_state_rows(wrapper, states, y)
        n_steps = max(len(states) - 1, 1)
        for group in layer_groups:
            layer = LAYER_GROUPS[group][model_name]
            h0 = feat_by_step[0].get(layer)
            if h0 is None:
                continue
            for meta in state_rows:
                step = int(meta["step"])
                ht = feat_by_step[step].get(layer)
                if ht is None:
                    continue
                delta = ht - h0
                time_bin = min(4, int(np.floor((step / n_steps) * 5.0))) if step < n_steps else 4
                ce_loss = -float(np.log(max(float(meta["true_prob"]), 1e-12)))
                base = {
                    "model": model_name,
                    "attack": "square",
                    "dataset_idx": int(dataset_idx),
                    "image_ord": int(image_ord),
                    "label": int(label),
                    "layer_group": group,
                    "layer": layer,
                    "step": step,
                    "normalized_progress": float(step / n_steps),
                    "time_bin": int(time_bin),
                    "final_success": int(final_eval["success"]),
                    "step_success": int(meta["success_at_step"]),
                    "margin": float(meta["margin"]),
                    "neg_margin": -float(meta["margin"]),
                    "true_prob": float(meta["true_prob"]),
                    "ce_loss": ce_loss,
                }
                for basis_type in [
                    "transport_axis_energy",
                    "random_axis_energy",
                    "clean_motion_axis_energy",
                    "gradient_only_axis_energy",
                ]:
                    basis = bases.get((model_name, layer, basis_type))
                    if basis is None:
                        continue
                    frac, energy, total = projection_fraction(delta, basis)
                    rows.append({
                        **base,
                        "predictor": basis_type,
                        "score": frac,
                        "raw_energy": energy,
                        "total_feature_energy": total,
                    })
                rows.append({**base, "predictor": "ce_loss", "score": ce_loss, "raw_energy": np.nan, "total_feature_energy": np.nan})
                rows.append({**base, "predictor": "neg_margin", "score": -float(meta["margin"]), "raw_energy": np.nan, "total_feature_energy": np.nan})
        if (image_ord + 1) % args.checkpoint_every == 0:
            print(f"  {model_name}: {image_ord + 1}/{len(selected)}", flush=True)
    return rows


def summarize(df: pd.DataFrame, out_dir: Path):
    rows = []
    for keys, g in df[df.step > 0].groupby(["model", "layer_group", "layer", "time_bin", "predictor"]):
        model, group, layer, time_bin, predictor = keys
        per_img = g.sort_values("step").groupby("dataset_idx", as_index=False).tail(1)
        y = per_img["final_success"].to_numpy(dtype=int)
        score = per_img["score"].to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            auroc = np.nan
            oriented = np.nan
            note = "single_class"
        else:
            auroc = float(roc_auc_score(y, score))
            oriented = max(auroc, 1.0 - auroc)
            note = ""
        rows.append({
            "model": model,
            "layer_group": group,
            "layer": layer,
            "time_bin": int(time_bin),
            "predictor": predictor,
            "n": int(len(per_img)),
            "positive_rate": float(np.mean(y)) if len(y) else np.nan,
            "auroc": auroc,
            "oriented_auroc": oriented,
            "mean_success_score": float(per_img.loc[per_img.final_success == 1, "score"].mean()),
            "mean_failed_score": float(per_img.loc[per_img.final_success == 0, "score"].mean()),
            "note": note,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "square_early_predictor_comparison_summary.csv", index=False)

    early = summary[summary.time_bin.isin([0, 1])].copy()
    rank = early.groupby(["model", "layer_group", "predictor"], dropna=False).agg(
        mean_early_auroc=("auroc", "mean"),
        mean_early_oriented_auroc=("oriented_auroc", "mean"),
    ).reset_index()
    rank.to_csv(out_dir / "square_early_predictor_rankings.csv", index=False)
    return summary, rank


def plot_summary(summary: pd.DataFrame, out_dir: Path):
    pred_order = [
        "transport_axis_energy",
        "ce_loss",
        "neg_margin",
        "random_axis_energy",
        "clean_motion_axis_energy",
        "gradient_only_axis_energy",
    ]
    models = list(dict.fromkeys(summary.model))
    layer_groups = ["hidden", "penultimate", "logits"]
    fig, axes = plt.subplots(len(models), len(layer_groups), figsize=(16, 3.5 * len(models)), sharey=True, constrained_layout=True)
    if len(models) == 1:
        axes = np.expand_dims(axes, 0)
    for r, model in enumerate(models):
        for c, group in enumerate(layer_groups):
            ax = axes[r, c]
            sub = summary[(summary.model == model) & (summary.layer_group == group) & (summary.time_bin.isin([0, 1]))]
            means = sub.groupby("predictor")["auroc"].mean().reindex(pred_order)
            ax.bar(np.arange(len(means)), means.values)
            ax.axhline(0.5, color="black", lw=1, alpha=0.5)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{model} {group}")
            ax.set_xticks(np.arange(len(means)))
            ax.set_xticklabels([p.replace("_axis_energy", "").replace("_", "\n") for p in means.index], rotation=0, fontsize=7)
            ax.set_ylabel("Early AUROC")
            ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "square_early_predictor_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    layer_groups = [g.strip() for g in args.layer_groups.split(",") if g.strip()]
    bases, pc_meta, basis_meta = build_bases(Path(args.layerwise_dir), args.top_k, args.seed)
    pc_meta.to_csv(out_dir / "transport_axis_metadata.csv", index=False)
    basis_meta.to_csv(out_dir / "predictor_basis_metadata.csv", index=False)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    all_rows = []
    for model_name in args.models:
        wrapper = load_model(model_name, device)
        select_args = SimpleNamespace(**{**vars(args), "models": [model_name]})
        selected = select_clean_correct(dataset, {model_name: wrapper}, select_args, device)
        print(f"[MODEL] {model_name} n={len(selected)}", flush=True)
        rows = collect_model_rows(args, model_name, wrapper, dataset, selected, bases, layer_groups, device)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(out_dir / "partial_square_early_predictor_comparison.csv", index=False)
        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "square_early_predictor_comparison.csv", index=False)
    summary, rank = summarize(df, out_dir)
    plot_summary(summary, out_dir)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"args": vars(args), "n_rows": int(len(df))}, f, indent=2)
    print(f"[SAVED] {out_dir}", flush=True)
    print(rank.sort_values("mean_early_auroc", ascending=False).head(30).to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layerwise-dir", default="analysis_outputs/pure_af_geometry/cifar_layerwise_success_flow_c10_s3_g120")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_square_early_predictor_comparison")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--models", default="bbb_resnet50,bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--square-steps", type=int, default=100)
    p.add_argument("--square-min-size", type=int, default=2)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--layer-groups", default="hidden,penultimate,logits")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
