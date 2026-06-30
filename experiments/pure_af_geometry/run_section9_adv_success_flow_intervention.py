#!/usr/bin/env python3
"""Intervene on adversarial success-flow PCs in CIFAR-10 models.

This script learns PCA directions from successful PGD transport vectors in a
chosen hidden layer, pulls individual PCs back to pixel space, and measures
whether moving clean held-out images along those directions changes classifier
behavior. Unlike the older pure-flow intervention scripts, the basis here is
fit only from successful adversarial PGD trajectories.
"""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import (  # noqa: E402
    load_model,
    normalize_rows,
)
from experiments.pure_af_geometry.analyze_cifar_benchmark_optimizer_transport import pgd_trajectory  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import margin  # noqa: E402


BEST_LAYER = {
    "bbb_resnet50": "layer4",
    "bbb_vgg19_bn": "block5",
    "bbb_densenet": "denseblock3",
    "bbb_inception_v3": "mixed6",
}


def select_clean_correct(dataset, wrapper, n_total: int, device):
    selected = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            pred = int(wrapper(x).argmax(1).item())
        if pred == int(y.item()):
            selected.append((idx, int(y.item())))
        if len(selected) >= n_total:
            break
    return selected


def eval_one(wrapper, x, y):
    with torch.no_grad():
        logits = wrapper(x)
        probs = F.softmax(logits, dim=1)
        pred = int(logits.argmax(1).item())
        return {
            "pred": pred,
            "success": int(pred != int(y.item())),
            "margin": float(margin(logits, y).item()),
            "true_prob": float(probs[0, int(y.item())].item()),
        }


def project_linf(x_adv, x0, eps):
    return torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)


def fit_pca(vectors: np.ndarray, k: int):
    mean = vectors.mean(axis=0, keepdims=True)
    xc = vectors - mean
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s * s
    ratios = var / np.clip(var.sum(), 1e-12, None)
    basis = vt[: min(k, vt.shape[0])].astype(np.float32)
    return mean.astype(np.float32), basis, ratios.astype(np.float32)


def ce_pixel_grad(wrapper, x_adv, y):
    x_probe = x_adv.detach().requires_grad_(True)
    logits = wrapper(x_probe)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_probe)[0]
    return grad, logits.detach()


def feature_pc_grad(wrapper, x_adv, layer: str, direction_np: np.ndarray):
    x_probe = x_adv.detach().requires_grad_(True)
    _logits, feats, _raw = wrapper.forward_with_features(x_probe)
    h = feats[layer]
    direction = torch.as_tensor(direction_np, dtype=h.dtype, device=h.device).view_as(h)
    scalar = (h * direction).sum()
    grad = torch.autograd.grad(scalar, x_probe)[0]
    return grad


def attack_with_direction(wrapper, x, y, layer, direction_np, sign, eps, steps, step_size):
    x0 = x.detach()
    x_adv = x0.clone()
    ce_cosines = []
    for _ in range(steps):
        grad = feature_pc_grad(wrapper, x_adv, layer, sign * direction_np)
        ce_grad, _ = ce_pixel_grad(wrapper, x_adv, y)
        ce_cos = torch.sum(
            F.normalize(grad.flatten(1), dim=1) * F.normalize(ce_grad.flatten(1), dim=1),
            dim=1,
        )
        ce_cosines.append(float(ce_cos.item()))
        x_adv = project_linf(x_adv + step_size * grad.sign(), x0, eps)
    return x_adv.detach(), float(np.mean(ce_cosines))


def ce_pgd(wrapper, x, y, eps, steps, step_size):
    x0 = x.detach()
    x_adv = x0.clone()
    for _ in range(steps):
        grad, _ = ce_pixel_grad(wrapper, x_adv, y)
        x_adv = project_linf(x_adv + step_size * grad.sign(), x0, eps)
    return x_adv.detach()


def random_direction_attack(wrapper, x, y, eps, steps, step_size, seed):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    direction = torch.randn(x.shape, generator=gen, device=x.device).sign()
    x0 = x.detach()
    x_adv = x0.clone()
    ce_cosines = []
    for _ in range(steps):
        ce_grad, _ = ce_pixel_grad(wrapper, x_adv, y)
        ce_cos = torch.sum(
            F.normalize(direction.flatten(1), dim=1) * F.normalize(ce_grad.flatten(1), dim=1),
            dim=1,
        )
        ce_cosines.append(float(ce_cos.item()))
        x_adv = project_linf(x_adv + step_size * direction, x0, eps)
    return x_adv.detach(), float(np.mean(ce_cosines))


def collect_success_transport(wrapper, dataset, train_items, layer, args, device):
    eps = args.basis_eps / 255.0
    step_size = args.basis_step_size / 255.0 if args.basis_step_size > 0 else eps / max(args.basis_steps, 1)
    vectors = []
    image_rows = []
    for image_ord, (idx, label) in enumerate(train_items):
        x_cpu, _ = dataset[idx]
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        states = pgd_trajectory(wrapper, x, y, eps, args.basis_steps, step_size)
        final = eval_one(wrapper, states[-1], y)
        image_rows.append({
            "dataset_idx": int(idx),
            "image_ord": int(image_ord),
            "label": int(label),
            "basis_final_success": int(final["success"]),
            "basis_final_margin": float(final["margin"]),
        })
        if not final["success"]:
            continue
        feats_by_step = []
        for state in states:
            with torch.no_grad():
                _logits, feats, _raw = wrapper.forward_with_features(state)
            feats_by_step.append(feats[layer].detach().cpu().numpy()[0].astype(np.float32))
        for step in range(len(feats_by_step) - 1):
            v = feats_by_step[step + 1] - feats_by_step[step]
            if np.linalg.norm(v) > 1e-12:
                vectors.append(v.astype(np.float32))
    return np.stack(vectors).astype(np.float32) if vectors else np.empty((0, 0), dtype=np.float32), image_rows


def choose_pc_signs(wrapper, dataset, train_items, layer, basis, args, device):
    eps = args.sign_eps / 255.0
    step_size = eps
    rows = []
    signs = {}
    for pc_idx, direction in enumerate(basis[: args.pcs], start=1):
        sign_scores = {}
        for sign in [-1, 1]:
            drops = []
            for idx, label in train_items[: args.sign_images]:
                x_cpu, _ = dataset[idx]
                x = x_cpu.unsqueeze(0).to(device)
                y = torch.tensor([label], device=device)
                clean = eval_one(wrapper, x, y)
                adv, _ce_cos = attack_with_direction(wrapper, x, y, layer, direction, sign, eps, 1, step_size)
                after = eval_one(wrapper, adv, y)
                drops.append(clean["margin"] - after["margin"])
            sign_scores[sign] = float(np.mean(drops)) if drops else float("-inf")
            rows.append({
                "pc": int(pc_idx),
                "sign": int(sign),
                "mean_train_margin_drop": float(sign_scores[sign]),
            })
        signs[pc_idx] = 1 if sign_scores[1] >= sign_scores[-1] else -1
    return signs, rows


def summarize(df: pd.DataFrame):
    source = df[df.source_model == df.target_model]
    return source.groupby(["source_model", "layer", "variant", "eps_255", "steps"], dropna=False).agg(
        asr=("target_success", "mean"),
        n=("target_success", "size"),
        mean_margin_drop=("margin_drop", "mean"),
        mean_prob_drop=("true_prob_drop", "mean"),
        mean_ce_grad_cosine=("ce_grad_cosine", "mean"),
    ).reset_index()


def plot_summary(summary: pd.DataFrame, out_dir: Path):
    sub = summary[(summary.eps_255 == summary.eps_255.max()) & (summary.steps == summary.steps.max())].copy()
    order = ["random_feature", "adv_pc1", "adv_pc2", "adv_pc3", "adv_pc4", "adv_pc5", "ce_pgd"]
    sub["variant"] = pd.Categorical(sub["variant"], categories=order, ordered=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True, constrained_layout=True)
    for ax, (model, g) in zip(axes.ravel(), sub.sort_values("variant").groupby("source_model", sort=False)):
        ax.bar(g["variant"].astype(str), g["asr"], color="#5b8fd9")
        ax.set_title(f"{model} ({g.layer.iloc[0]})")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("untargeted ASR")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "adv_success_flow_intervention_asr.png", dpi=190, bbox_inches="tight")
    fig.savefig(out_dir / "adv_success_flow_intervention_asr.pdf", bbox_inches="tight")
    plt.close(fig)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    eps_values = [float(x) / 255.0 for x in args.eps.split(",")]
    steps_values = [int(x) for x in args.steps.split(",")]
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())

    all_rows = []
    all_basis_rows = []
    all_sign_rows = []
    all_basis_image_rows = []

    for model in models:
        layer = BEST_LAYER[model]
        wrapper = load_model(model, device).eval()
        selected = select_clean_correct(dataset, wrapper, args.train_images + args.test_images, device)
        train_items = selected[: args.train_images]
        test_items = selected[args.train_images: args.train_images + args.test_images]
        print(f"[MODEL] {model} layer={layer} train={len(train_items)} test={len(test_items)}", flush=True)

        vectors, basis_image_rows = collect_success_transport(wrapper, dataset, train_items, layer, args, device)
        for row in basis_image_rows:
            row.update({"model": model, "layer": layer})
        all_basis_image_rows.extend(basis_image_rows)
        if len(vectors) < max(args.pcs + 2, 8):
            print(f"  [SKIP] not enough successful transport vectors: {len(vectors)}", flush=True)
            del wrapper
            continue
        norm_vectors = normalize_rows(vectors)
        _mean, basis, ratios = fit_pca(norm_vectors, args.pcs)
        for pc_idx in range(1, min(args.pcs, len(ratios)) + 1):
            all_basis_rows.append({
                "model": model,
                "layer": layer,
                "pc": int(pc_idx),
                "variance_explained": float(ratios[pc_idx - 1]),
                "cumulative_variance": float(np.sum(ratios[:pc_idx])),
                "n_success_transport_vectors": int(len(norm_vectors)),
                "d": int(norm_vectors.shape[1]),
            })
        signs, sign_rows = choose_pc_signs(wrapper, dataset, train_items, layer, basis, args, device)
        for row in sign_rows:
            row.update({"model": model, "layer": layer, "chosen": int(row["sign"] == signs[row["pc"]])})
        all_sign_rows.extend(sign_rows)

        for image_ord, (idx, label) in enumerate(test_items):
            x_cpu, _ = dataset[idx]
            x = x_cpu.unsqueeze(0).to(device)
            y = torch.tensor([label], device=device)
            clean = eval_one(wrapper, x, y)
            for eps in eps_values:
                for steps in steps_values:
                    step_size = eps / max(steps, 1)
                    variants = [("random_feature", None, 0), ("ce_pgd", None, 0)]
                    for pc_idx in range(1, min(args.pcs, basis.shape[0]) + 1):
                        variants.append((f"adv_pc{pc_idx}", basis[pc_idx - 1], signs[pc_idx]))
                    for variant, direction, sign in variants:
                        if variant == "ce_pgd":
                            adv = ce_pgd(wrapper, x, y, eps, steps, step_size)
                            ce_cos = 1.0
                        elif variant == "random_feature":
                            adv, ce_cos = random_direction_attack(
                                wrapper,
                                x,
                                y,
                                eps,
                                steps,
                                step_size,
                                args.seed + idx * 1000 + steps * 17,
                            )
                        else:
                            adv, ce_cos = attack_with_direction(wrapper, x, y, layer, direction, sign, eps, steps, step_size)
                        ev = eval_one(wrapper, adv, y)
                        all_rows.append({
                            "source_model": model,
                            "target_model": model,
                            "layer": layer,
                            "dataset_idx": int(idx),
                            "image_ord": int(image_ord),
                            "label": int(label),
                            "variant": variant,
                            "pc": int(variant.replace("adv_pc", "")) if variant.startswith("adv_pc") else np.nan,
                            "pc_sign": int(sign) if variant.startswith("adv_pc") else np.nan,
                            "eps": float(eps),
                            "eps_255": float(eps * 255),
                            "steps": int(steps),
                            "clean_pred": int(clean["pred"]),
                            "target_pred": int(ev["pred"]),
                            "target_success": int(ev["success"]),
                            "clean_margin": float(clean["margin"]),
                            "target_margin": float(ev["margin"]),
                            "margin_drop": float(clean["margin"] - ev["margin"]),
                            "clean_true_prob": float(clean["true_prob"]),
                            "target_true_prob": float(ev["true_prob"]),
                            "true_prob_drop": float(clean["true_prob"] - ev["true_prob"]),
                            "ce_grad_cosine": float(ce_cos),
                        })
            if (image_ord + 1) % args.checkpoint_every == 0:
                df = pd.DataFrame(all_rows)
                df.to_csv(out_dir / "partial_adv_success_flow_intervention_per_image.csv", index=False)
                summarize(df).to_csv(out_dir / "partial_adv_success_flow_intervention_summary.csv", index=False)
                print(f"  {model}: {image_ord + 1}/{len(test_items)} eval images rows={len(df)}", flush=True)
        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    summary = summarize(df)
    df.to_csv(out_dir / "adv_success_flow_intervention_per_image.csv", index=False)
    summary.to_csv(out_dir / "adv_success_flow_intervention_summary.csv", index=False)
    pd.DataFrame(all_basis_rows).to_csv(out_dir / "adv_success_flow_basis_metadata.csv", index=False)
    pd.DataFrame(all_sign_rows).to_csv(out_dir / "adv_success_flow_pc_sign_selection.csv", index=False)
    pd.DataFrame(all_basis_image_rows).to_csv(out_dir / "adv_success_flow_basis_image_outcomes.csv", index=False)
    plot_summary(summary, out_dir)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print("[SUMMARY]", flush=True)
    if not summary.empty:
        print(summary[(summary.eps_255 == summary.eps_255.max()) & (summary.steps == summary.steps.max())].to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/section9_adv_success_flow_intervention")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--models", default="bbb_resnet50,bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--train-images", type=int, default=60)
    p.add_argument("--test-images", type=int, default=40)
    p.add_argument("--basis-eps", type=float, default=2.0)
    p.add_argument("--basis-steps", type=int, default=5)
    p.add_argument("--basis-step-size", type=float, default=0.0)
    p.add_argument("--sign-images", type=int, default=30)
    p.add_argument("--sign-eps", type=float, default=2.0)
    p.add_argument("--eps", default="8")
    p.add_argument("--steps", default="1,5,10")
    p.add_argument("--pcs", type=int, default=5)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=29)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
