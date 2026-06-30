#!/usr/bin/env python3
"""Global-vs-class-specific CIFAR success-flow geometry and away attack."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import (  # noqa: E402
    DOMINANT,
    MODELS,
    PENULTIMATE,
    attack_one,
    build_mu,
    eval_all,
    load_model,
    normalize_rows,
    select_clean_correct,
)


LAYER_GROUPS = {
    "hidden": DOMINANT,
    "penultimate": PENULTIMATE,
    "logits": {m: "logits" for m in MODELS},
}


def pca_stats(x: np.ndarray) -> dict:
    xc = x - x.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s * s
    ratios = var / np.clip(var.sum(), 1e-12, None)
    csum = np.cumsum(ratios)
    entropy = -float(np.sum(ratios[ratios > 0] * np.log(ratios[ratios > 0])))
    return {
        "pc1_var": float(ratios[0]),
        "pc2_var": float(ratios[1]) if len(ratios) > 1 else np.nan,
        "pc3_var": float(ratios[2]) if len(ratios) > 2 else np.nan,
        "pc5_cum_var": float(csum[min(4, len(csum) - 1)]),
        "pc10_cum_var": float(csum[min(9, len(csum) - 1)]),
        "dim80": int(np.searchsorted(csum, 0.8) + 1),
        "dim90": int(np.searchsorted(csum, 0.9) + 1),
        "effective_rank": float(np.exp(entropy)),
    }


def class_mu_matrix(mu: dict, model: str, layer: str):
    classes = sorted(c for (m, l, c) in mu if m == model and l == layer)
    if not classes:
        return classes, np.empty((0, 0))
    arr = np.stack([mu[(model, layer, c)] for c in classes])
    return classes, normalize_rows(arr)


def compute_geometry(mu: dict, out_dir: Path):
    cos_rows, pca_rows, decomp_rows = [], [], []
    global_mu = {}
    for group_name, mapping in LAYER_GROUPS.items():
        for model, layer in mapping.items():
            classes, arr = class_mu_matrix(mu, model, layer)
            if len(classes) < 2:
                continue
            cos = arr @ arr.T
            for i, ci in enumerate(classes):
                for j, cj in enumerate(classes):
                    cos_rows.append({
                        "model": model,
                        "layer_group": group_name,
                        "layer": layer,
                        "class_i": int(ci),
                        "class_j": int(cj),
                        "cosine": float(cos[i, j]),
                    })
            stats = pca_stats(arr)
            stats.update({"model": model, "layer_group": group_name, "layer": layer, "n_classes": len(classes), "d": arr.shape[1]})
            pca_rows.append(stats)

            g_raw = arr.mean(axis=0)
            g_norm = float(np.linalg.norm(g_raw))
            g_unit = g_raw / np.clip(g_norm, 1e-12, None)
            global_mu[(model, layer)] = g_unit.astype(np.float32)
            proj = arr @ g_unit
            residual = arr - proj[:, None] * g_unit[None, :]
            residual_norm = np.linalg.norm(residual, axis=1)
            frac_norm_shared = proj**2 / np.clip(np.sum(arr * arr, axis=1), 1e-12, None)
            for c, p, rn, frac in zip(classes, proj, residual_norm, frac_norm_shared):
                decomp_rows.append({
                    "model": model,
                    "layer_group": group_name,
                    "layer": layer,
                    "class": int(c),
                    "mu_global_norm_before_normalization": g_norm,
                    "projection_on_global": float(p),
                    "residual_norm": float(rn),
                    "fraction_squared_norm_explained_by_global": float(frac),
                })

    cos_df = pd.DataFrame(cos_rows)
    pca_df = pd.DataFrame(pca_rows)
    decomp_df = pd.DataFrame(decomp_rows)
    cos_df.to_csv(out_dir / "class_direction_cosine_matrix.csv", index=False)
    pca_df.to_csv(out_dir / "class_direction_pca.csv", index=False)
    decomp_df.to_csv(out_dir / "global_vs_class_specific_decomposition.csv", index=False)
    plot_heatmaps(cos_df, out_dir)
    plot_scree(pca_df, out_dir)
    return global_mu


def plot_heatmaps(cos_df: pd.DataFrame, out_dir: Path):
    for group_name in ["hidden", "penultimate", "logits"]:
        sub = cos_df[cos_df.layer_group == group_name]
        if sub.empty:
            continue
        models = list(dict.fromkeys(sub.model))
        fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
        axes = axes.ravel()
        for ax, model in zip(axes, models):
            g = sub[sub.model == model]
            mat = g.pivot(index="class_i", columns="class_j", values="cosine").sort_index().sort_index(axis=1)
            im = ax.imshow(mat.values, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_title(f"{model} {g.layer.iloc[0]}")
            ax.set_xticks(range(len(mat.columns)))
            ax.set_yticks(range(len(mat.index)))
            ax.set_xticklabels(mat.columns)
            ax.set_yticklabels(mat.index)
        fig.colorbar(im, ax=axes.tolist(), shrink=0.75)
        fig.suptitle(f"Class pure-flow direction cosine: {group_name}")
        fig.savefig(out_dir / f"class_direction_cosine_heatmap_{group_name}.png", dpi=180)
        plt.close(fig)


def plot_scree(pca_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True, sharey=True)
    for ax, group_name in zip(axes, ["hidden", "penultimate", "logits"]):
        sub = pca_df[pca_df.layer_group == group_name]
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub.pc1_var, width=0.4, label="PC1")
        ax.bar(x + 0.2, sub.pc2_var, width=0.4, label="PC2")
        ax.set_xticks(x)
        ax.set_xticklabels(sub.model, rotation=35, ha="right", fontsize=8)
        ax.set_title(group_name)
        ax.set_ylabel("variance explained")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.savefig(out_dir / "class_direction_scree_plot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_global_attack(args, global_mu: dict, out_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    wrappers = {m: load_model(m, device) for m in args.models}
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    selected = select_clean_correct(dataset, wrappers, args, device)
    eps_values = [float(x) / 255.0 for x in args.eps.split(",")]
    steps_values = [int(x) for x in args.steps.split(",")]
    rows = []
    completed = set()
    partial_path = out_dir / "partial_global_vs_class_awayflow_attack_per_image.csv"
    if args.resume and partial_path.exists():
        old = pd.read_csv(partial_path)
        rows = old.to_dict("records")
        source_done = old[old.source_model == old.target_model]
        for r in source_done[["source_model", "dataset_idx", "eps_255", "steps", "variant"]].drop_duplicates().itertuples():
            completed.add((r.source_model, int(r.dataset_idx), float(r.eps_255), int(r.steps), r.variant))
        print(f"[RESUME] loaded {len(rows)} rows", flush=True)

    for source_model in args.models:
        print(f"[GLOBAL ATTACK] {source_model}", flush=True)
        wrapper = wrappers[source_model]
        variants = [
            ("global_hidden", DOMINANT[source_model]),
            ("global_penultimate", PENULTIMATE[source_model]),
            ("global_logits", "logits"),
        ]
        for image_ord, (dataset_idx, label) in enumerate(selected):
            x_cpu, _ = dataset[dataset_idx]
            x = x_cpu.unsqueeze(0).to(device)
            y = torch.tensor([label], device=device)
            clean_eval = eval_all(wrappers, x, y)
            for eps in eps_values:
                for steps in steps_values:
                    step_size = eps / max(steps, 1)
                    for variant, layer in variants:
                        done = (source_model, int(dataset_idx), float(eps * 255), int(steps), variant)
                        if done in completed:
                            continue
                        mu_np = global_mu.get((source_model, layer))
                        if mu_np is None:
                            continue
                        adv, ce_cos, local_cos = attack_one(
                            wrapper, x, y, "away_hidden", layer, mu_np, eps, steps, step_size,
                            args.seed + image_ord * 1000 + steps * 10 + int(eps * 255),
                            device,
                        )
                        evals = eval_all(wrappers, adv, y)
                        src = evals[source_model]
                        for target_model, ev in evals.items():
                            rows.append({
                                "source_model": source_model,
                                "target_model": target_model,
                                "dataset_idx": int(dataset_idx),
                                "image_ord": int(image_ord),
                                "label": int(label),
                                "variant": variant,
                                "layer": layer,
                                "eps": float(eps),
                                "eps_255": float(eps * 255),
                                "steps": int(steps),
                                "source_success": int(src["success"]),
                                "target_success": int(ev["success"]),
                                "target_pred": int(ev["pred"]),
                                "target_margin": float(ev["margin"]),
                                "target_true_prob": float(ev["true_prob"]),
                                "clean_target_margin": float(clean_eval[target_model]["margin"]),
                                "clean_target_true_prob": float(clean_eval[target_model]["true_prob"]),
                                "ce_grad_cosine": ce_cos,
                                "feature_logp_grad_cosine": local_cos,
                            })
                        completed.add(done)
            if (image_ord + 1) % args.checkpoint_every == 0:
                pd.DataFrame(rows).to_csv(partial_path, index=False)
                print(f"  {source_model}: {image_ord + 1}/{len(selected)} rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "global_vs_class_awayflow_attack_per_image.csv", index=False)
    summary = df[df.source_model == df.target_model].groupby(["source_model", "variant", "layer", "eps_255", "steps"], dropna=False).agg(
        asr=("target_success", "mean"),
        n=("target_success", "size"),
        mean_margin=("target_margin", "mean"),
        mean_ce_grad_cosine=("ce_grad_cosine", "mean"),
        mean_feature_logp_grad_cosine=("feature_logp_grad_cosine", "mean"),
    ).reset_index()

    class_report_path = Path(args.class_attack_dir) / "cifar_away_flow_attack_summary.csv"
    if class_report_path.exists():
        class_summary = pd.read_csv(class_report_path)
        class_summary = class_summary.rename(columns={"mean_true_prob": "mean_target_true_prob"})
        summary["attack_family"] = "global"
        class_summary["attack_family"] = "class_specific_or_baseline"
        compare = pd.concat([summary, class_summary], ignore_index=True, sort=False)
    else:
        compare = summary
    compare.to_csv(out_dir / "global_vs_class_awayflow_attack.csv", index=False)
    return df, summary


def cross_arch_similarity(args, global_mu: dict, out_dir: Path):
    rows = []
    for group_name, mapping in LAYER_GROUPS.items():
        items = [(m, mapping[m], global_mu[(m, mapping[m])]) for m in MODELS if (m, mapping[m]) in global_mu]
        for i, (ma, la, va) in enumerate(items):
            for mb, lb, vb in items[i + 1:]:
                row = {"layer_group": group_name, "model_a": ma, "layer_a": la, "model_b": mb, "layer_b": lb}
                if len(va) == len(vb):
                    row["direct_cosine_if_same_dimension"] = float(np.dot(va, vb) / np.clip(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12, None))
                else:
                    row["direct_cosine_if_same_dimension"] = np.nan
                rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "global_direction_cross_arch_similarity.csv", index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layerwise-dir", default="analysis_outputs/pure_af_geometry/cifar_layerwise_success_flow_c10_s3_g120")
    p.add_argument("--class-attack-dir", default="analysis_outputs/pure_af_geometry/cifar_away_flow_attack_100")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_global_vs_class_success_flow")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--models", default="bbb_resnet50,bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--eps", default="8")
    p.add_argument("--steps", default="1,2,5,10,20")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-attack", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mu = build_mu(Path(args.layerwise_dir))
    global_mu = compute_geometry(mu, out_dir)
    cross_arch_similarity(args, global_mu, out_dir)
    if not args.skip_attack:
        run_global_attack(args, global_mu, out_dir)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"args": vars(args), "notes": ["Direct cross-architecture cosine is reported only when dimensions match, mainly logits."]}, f, indent=2)
    print(f"[SAVED] {out_dir}")


if __name__ == "__main__":
    main()
