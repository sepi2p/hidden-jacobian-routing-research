#!/usr/bin/env python3
"""Fast Section 4 separability check for adversarial success-flow.

This script fits a PCA basis on raw layer-3 transport vectors from successful
adversarial trajectories, then compares held-out successful segments against
failed adversarial segments, clean class-preserving motion, and random feature
directions. It is intentionally small enough to refresh manuscript figures.
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
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar_benchmark_optimizer_transport import (  # noqa: E402
    gradient_variant_trajectory,
    pgd_trajectory,
)
from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import (  # noqa: E402
    load_model,
)
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import (  # noqa: E402
    margin,
    select_clean_correct,
)


def forward_feature(wrapper, x: torch.Tensor, layer: str) -> tuple[int, float, float, np.ndarray]:
    with torch.no_grad():
        logits, feats, _raw = wrapper.forward_with_features(x)
        pred = int(logits.argmax(1).item())
        prob = float(F.softmax(logits, dim=1).max(1).values.item())
    return pred, prob, float(logits.max(1).values.item()), feats[layer].detach().cpu().numpy()[0].astype(np.float32)


def clean_variants(x: torch.Tensor, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator(device=x.device).manual_seed(seed)
    out = []
    for sigma in (0.01, 0.02, 0.03):
        out.append((x + torch.randn(x.shape, generator=gen, device=x.device) * sigma).clamp(0, 1))
    out.append(TF.adjust_brightness(x, 1.08).clamp(0, 1))
    out.append(TF.adjust_contrast(x, 0.92).clamp(0, 1))
    return out


def fit_basis(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return mean.astype(np.float32), vt[: min(k, vt.shape[0])].astype(np.float32)


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    use = basis[: min(k, len(basis))]
    xc = x - mean
    denom = np.sum(xc * xc, axis=1)
    coeff = xc @ use.T
    return np.sum(coeff * coeff, axis=1) / np.clip(denom, 1e-12, None)


def pca_stats(x: np.ndarray) -> dict:
    xc = x - x.mean(axis=0, keepdims=True)
    _u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    var = s * s
    ratios = var / np.clip(var.sum(), 1e-12, None)
    csum = np.cumsum(ratios)
    entropy = -float(np.sum(ratios[ratios > 0] * np.log(ratios[ratios > 0])))
    return {
        "n": int(len(x)),
        "d": int(x.shape[1]),
        "pc1_var": float(ratios[0]),
        "pc10_cum_var": float(csum[min(9, len(csum) - 1)]),
        "dim80": int(np.searchsorted(csum, 0.8) + 1),
        "dim90": int(np.searchsorted(csum, 0.9) + 1),
        "effective_rank": float(np.exp(entropy)),
    }


def attack_states(wrapper, x, y, attack: str, eps: float, steps: int, step_size: float):
    if attack == "pgd":
        return pgd_trajectory(wrapper, x, y, eps, steps, step_size)
    return gradient_variant_trajectory(wrapper, x, y, eps, steps, step_size, attack)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = load_model(args.model, device).eval()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    select_args = argparse.Namespace(models=[args.model], images=args.images)
    selected = select_clean_correct(dataset, {args.model: wrapper}, select_args, device)
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0 if args.step_size > 0 else eps / max(args.steps, 1)

    adv_rows = []
    adv_vectors = []
    clean_rows = []
    clean_vectors = []
    print(f"[RUN] model={args.model} layer={args.layer} attack={args.attack} images={len(selected)}", flush=True)
    for image_ord, (dataset_idx, label) in enumerate(selected):
        x_cpu, _ = dataset[dataset_idx]
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        states = attack_states(wrapper, x, y, args.attack, eps, args.steps, step_size)
        with torch.no_grad():
            final_logits = wrapper(states[-1])
            final_success = int(final_logits.argmax(1).item() != label)
            final_margin = float(margin(final_logits, y).item())

        feats = []
        margins = []
        probs = []
        preds = []
        for state in states:
            pred, _prob, _logit, feat = forward_feature(wrapper, state, args.layer)
            with torch.no_grad():
                logits = wrapper(state)
                probs.append(float(F.softmax(logits, dim=1)[0, label].item()))
                margins.append(float(margin(logits, y).item()))
            preds.append(pred)
            feats.append(feat)
        for step in range(len(feats) - 1):
            adv_vectors.append((feats[step + 1] - feats[step]).astype(np.float32))
            adv_rows.append({
                "group": "success" if final_success else "failed",
                "dataset_idx": int(dataset_idx),
                "image_ord": int(image_ord),
                "label": int(label),
                "step": int(step),
                "final_success": int(final_success),
                "final_margin": final_margin,
                "step_margin": margins[step],
                "next_margin": margins[step + 1],
                "step_true_prob": probs[step],
                "next_true_prob": probs[step + 1],
                "pred": int(preds[step + 1]),
            })

        _pred0, _prob0, _logit0, h0 = forward_feature(wrapper, x, args.layer)
        for aug_ord, x_aug in enumerate(clean_variants(x, args.seed + 10000 + image_ord)):
            with torch.no_grad():
                logits_aug = wrapper(x_aug)
                if int(logits_aug.argmax(1).item()) != label:
                    continue
            _pred_aug, _prob_aug, _logit_aug, h_aug = forward_feature(wrapper, x_aug, args.layer)
            clean_vectors.append((h_aug - h0).astype(np.float32))
            clean_rows.append({
                "group": "clean",
                "dataset_idx": int(dataset_idx),
                "image_ord": int(image_ord),
                "label": int(label),
                "step": int(aug_ord),
                "final_success": 0,
                "final_margin": np.nan,
                "step_margin": np.nan,
                "next_margin": float(margin(logits_aug, y).item()),
                "step_true_prob": np.nan,
                "next_true_prob": float(F.softmax(logits_aug, dim=1)[0, label].item()),
                "pred": int(logits_aug.argmax(1).item()),
            })

    adv_df = pd.DataFrame(adv_rows)
    adv_arr = np.stack(adv_vectors).astype(np.float32)
    clean_df = pd.DataFrame(clean_rows)
    clean_arr = np.stack(clean_vectors).astype(np.float32)

    train_images = set(
        adv_df[["image_ord", "final_success"]]
        .drop_duplicates()
        .sample(frac=args.train_frac, random_state=args.seed)["image_ord"]
        .astype(int)
        .tolist()
    )
    train_mask = adv_df["image_ord"].isin(train_images).to_numpy()
    success_mask = (adv_df["group"] == "success").to_numpy()
    train_success = adv_arr[train_mask & success_mask]
    if len(train_success) < args.k + 2:
        raise RuntimeError(f"Not enough successful training segments for PCA: {len(train_success)}")

    mean, basis = fit_basis(train_success, args.max_k)
    rng = np.random.default_rng(args.seed)
    random_arr = rng.normal(size=(max(len(clean_arr), int((~train_mask & success_mask).sum()), 200), adv_arr.shape[1])).astype(np.float32)
    random_arr /= np.clip(np.linalg.norm(random_arr, axis=1, keepdims=True), 1e-12, None)

    eval_parts = []
    test_adv_df = adv_df[~train_mask].copy()
    test_adv_arr = adv_arr[~train_mask]
    for group in ("success", "failed"):
        mask = (test_adv_df["group"] == group).to_numpy()
        if mask.any():
            part = test_adv_df[mask].copy()
            part["projection_energy"] = projection_energy(test_adv_arr[mask], mean, basis, args.k)
            eval_parts.append(part)
    clean_part = clean_df.copy()
    clean_part["projection_energy"] = projection_energy(clean_arr, mean, basis, args.k)
    eval_parts.append(clean_part)
    random_part = pd.DataFrame({"group": "random", "dataset_idx": -1, "image_ord": np.arange(len(random_arr))})
    random_part["projection_energy"] = projection_energy(random_arr, mean, basis, args.k)
    eval_parts.append(random_part)
    scores = pd.concat(eval_parts, ignore_index=True, sort=False)

    metrics = []
    success_scores = scores[scores["group"] == "success"]["projection_energy"].to_numpy(float)
    for group in ("failed", "clean", "random"):
        other = scores[scores["group"] == group]["projection_energy"].to_numpy(float)
        if len(success_scores) and len(other):
            y_true = np.concatenate([np.ones(len(success_scores)), np.zeros(len(other))])
            values = np.concatenate([success_scores, other])
            auroc = float(roc_auc_score(y_true, values))
        else:
            auroc = np.nan
        metrics.append({
            "comparison": f"success_vs_{group}",
            "auroc": auroc,
            "n_success": int(len(success_scores)),
            f"n_{group}": int(len(other)),
            "mean_success": float(np.mean(success_scores)) if len(success_scores) else np.nan,
            f"mean_{group}": float(np.mean(other)) if len(other) else np.nan,
        })

    counts = adv_df[["image_ord", "final_success"]].drop_duplicates()["final_success"].value_counts().to_dict()
    metadata = {
        "model": args.model,
        "layer": args.layer,
        "attack": args.attack,
        "images": len(selected),
        "eps": eps,
        "steps": args.steps,
        "step_size": step_size,
        "k": args.k,
        "max_k": args.max_k,
        "n_success_images": int(counts.get(1, 0)),
        "n_failed_images": int(counts.get(0, 0)),
        "pca_stats_train_success": pca_stats(train_success),
    }
    scores.to_csv(out_dir / "section4_fast_projection_scores.csv", index=False)
    pd.DataFrame(metrics).to_csv(out_dir / "section4_fast_projection_metrics.csv", index=False)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/section4_fast_resnet_layer3_separability")
    parser.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    parser.add_argument("--model", default="bbb_resnet50")
    parser.add_argument("--layer", default="layer3")
    parser.add_argument("--attack", default="si_fgsm", choices=["pgd", "mi_fgsm", "ni_fgsm", "ti_fgsm", "si_fgsm"])
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--eps", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max-k", type=int, default=50)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
