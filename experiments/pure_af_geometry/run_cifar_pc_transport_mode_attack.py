#!/usr/bin/env python3
"""Attack with individual PCA transport modes of class-pure directions."""

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

from experiments.pure_af_geometry.analyze_cifar_global_vs_class_success_flow import (  # noqa: E402
    LAYER_GROUPS,
    class_mu_matrix,
)
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import (  # noqa: E402
    MODELS,
    attack_one,
    build_mu,
    eval_all,
    load_model,
    select_clean_correct,
)


def build_pc_directions(mu: dict, max_pcs: int):
    pc = {}
    rows = []
    for layer_group, mapping in LAYER_GROUPS.items():
        for model, layer in mapping.items():
            classes, arr = class_mu_matrix(mu, model, layer)
            if len(classes) < 2:
                continue
            mean = arr.mean(axis=0, keepdims=True)
            xc = arr - mean
            _u, s, vt = np.linalg.svd(xc, full_matrices=False)
            ratios = (s * s) / np.clip(np.sum(s * s), 1e-12, None)
            for i in range(min(max_pcs, vt.shape[0])):
                v = vt[i]
                v = v / np.clip(np.linalg.norm(v), 1e-12, None)
                # Fix sign so the PC roughly points with the mean class-flow direction.
                if float(np.dot(v, mean.reshape(-1))) < 0:
                    v = -v
                pc[(model, layer, i + 1)] = v.astype(np.float32)
                rows.append({
                    "model": model,
                    "layer_group": layer_group,
                    "layer": layer,
                    "pc": i + 1,
                    "variance_explained": float(ratios[i]),
                    "cumulative_variance": float(np.sum(ratios[: i + 1])),
                    "n_classes": len(classes),
                    "d": arr.shape[1],
                })
    return pc, pd.DataFrame(rows)


def write_attack_outputs(rows, out_dir: Path, final: bool):
    prefix = "" if final else "partial_"
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df.to_csv(out_dir / f"{prefix}pc_transport_mode_attack_per_image.csv", index=False)
    source = df[df.source_model == df.target_model]
    summary = source.groupby(["source_model", "layer_group", "layer", "pc", "eps_255", "steps"], dropna=False).agg(
        asr=("target_success", "mean"),
        n=("target_success", "size"),
        mean_margin=("target_margin", "mean"),
        mean_true_prob=("target_true_prob", "mean"),
        mean_ce_grad_cosine=("ce_grad_cosine", "mean"),
        mean_feature_logp_grad_cosine=("feature_logp_grad_cosine", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / f"{prefix}pc_transport_mode_attack_summary.csv", index=False)
    transfer = df.groupby(["source_model", "target_model", "layer_group", "layer", "pc", "eps_255", "steps"], dropna=False).agg(
        asr=("target_success", "mean"),
        n=("target_success", "size"),
        mean_margin=("target_margin", "mean"),
    ).reset_index()
    transfer.to_csv(out_dir / f"{prefix}pc_transport_mode_transfer_summary.csv", index=False)
    if final:
        plot_summary(summary, out_dir)


def plot_summary(summary: pd.DataFrame, out_dir: Path):
    max_steps = int(summary.steps.max())
    eps = float(summary.eps_255.max())
    sub = summary[(summary.steps == max_steps) & (summary.eps_255 == eps)]
    for layer_group in ["hidden", "penultimate", "logits"]:
        gsub = sub[sub.layer_group == layer_group]
        if gsub.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, constrained_layout=True)
        axes = axes.ravel()
        for ax, (model, g) in zip(axes, gsub.groupby("source_model")):
            g = g.sort_values("pc")
            ax.bar(g.pc.astype(str), g.asr)
            ax.set_title(f"{model} {g.layer.iloc[0]}")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("PC")
            ax.set_ylabel("source ASR")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"PC transport-mode away attack: {layer_group}, eps={eps:.0f}/255, steps={max_steps}")
        fig.savefig(out_dir / f"pc_transport_mode_asr_{layer_group}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    layer_groups = [x.strip() for x in args.layer_groups.split(",") if x.strip()]
    eps_values = [float(x) / 255.0 for x in args.eps.split(",")]
    steps_values = [int(x) for x in args.steps.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mu = build_mu(Path(args.layerwise_dir))
    pc_dirs, pc_meta = build_pc_directions(mu, args.pcs)
    pc_meta.to_csv(out_dir / "pc_transport_mode_directions.csv", index=False)

    wrappers = {m: load_model(m, device) for m in args.models}
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    selected = select_clean_correct(dataset, wrappers, args, device)
    rows = []
    completed = set()
    partial = out_dir / "partial_pc_transport_mode_attack_per_image.csv"
    final = out_dir / "pc_transport_mode_attack_per_image.csv"
    resume_path = final if final.exists() else partial
    if args.resume and resume_path.exists():
        old = pd.read_csv(resume_path)
        rows = old.to_dict("records")
        source_done = old[old.source_model == old.target_model]
        for r in source_done[["source_model", "dataset_idx", "eps_255", "steps", "layer_group", "pc"]].drop_duplicates().itertuples():
            completed.add((r.source_model, int(r.dataset_idx), float(r.eps_255), int(r.steps), r.layer_group, int(r.pc)))
        print(f"[RESUME] rows={len(rows)} completed={len(completed)}", flush=True)

    for source_model in args.models:
        wrapper = wrappers[source_model]
        print(f"[MODEL] {source_model}", flush=True)
        for image_ord, (dataset_idx, label) in enumerate(selected):
            x_cpu, _ = dataset[dataset_idx]
            x = x_cpu.unsqueeze(0).to(device)
            y = torch.tensor([label], device=device)
            clean_eval = eval_all(wrappers, x, y)
            for eps in eps_values:
                for steps in steps_values:
                    step_size = eps / max(steps, 1)
                    for layer_group in layer_groups:
                        layer = LAYER_GROUPS[layer_group][source_model]
                        for pc_idx in range(1, args.pcs + 1):
                            done = (source_model, int(dataset_idx), float(eps * 255), int(steps), layer_group, int(pc_idx))
                            if done in completed:
                                continue
                            direction = pc_dirs.get((source_model, layer, pc_idx))
                            if direction is None:
                                continue
                            adv, ce_cos, local_cos = attack_one(
                                wrapper, x, y, "away_hidden", layer, direction, eps, steps, step_size,
                                args.seed + image_ord * 1000 + steps * 10 + pc_idx,
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
                                    "layer_group": layer_group,
                                    "layer": layer,
                                    "pc": int(pc_idx),
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
                write_attack_outputs(rows, out_dir, final=False)
                print(f"  {source_model}: {image_ord + 1}/{len(selected)} rows={len(rows)}", flush=True)
    write_attack_outputs(rows, out_dir, final=True)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"args": vars(args), "n_images": len(selected)}, f, indent=2)
    print(f"[SAVED] {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layerwise-dir", default="analysis_outputs/pure_af_geometry/cifar_layerwise_success_flow_c10_s3_g120")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_pc_transport_mode_attack")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--models", default="bbb_resnet50,bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--eps", default="8")
    p.add_argument("--steps", default="1,2,5,10,20")
    p.add_argument("--pcs", type=int, default=5)
    p.add_argument("--layer-groups", default="hidden,penultimate,logits")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
