#!/usr/bin/env python3
"""Fast adversarial layerwise transport-coordinate analysis for Section 6.

For each model/layer/attack, fit PCA on training-split successful adversarial
transport vectors and evaluate held-out success-vs-failed projection-energy
AUROC. This supplies true adversarial layerwise evidence, distinct from the
pure-flow layerwise artifacts used in earlier exploratory figures.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.analyze_cifar_benchmark_optimizer_transport import (  # noqa: E402
    gradient_variant_trajectory,
    pgd_trajectory,
    signhunter_trajectory,
)
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import (  # noqa: E402
    eval_all,
    select_clean_correct,
)


LAYER_ORDER = {
    "bbb_resnet50": ["layer1", "layer2", "layer3", "layer4", "avgpool", "logits"],
    "bbb_vgg19_bn": ["block1", "block2", "block3", "block4", "block5", "penultimate", "logits"],
    "bbb_densenet": ["denseblock1", "denseblock2", "denseblock3", "penultimate", "logits"],
    "bbb_inception_v3": ["mixed5", "mixed6", "mixed7", "penultimate", "logits"],
}


def fit_basis(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return mean.astype(np.float32), vt[: min(k, vt.shape[0])].astype(np.float32)


def pca_curve(x: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=0, keepdims=True)
    _u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    var = s * s
    ratios = var / np.clip(var.sum(), 1e-12, None)
    return np.cumsum(ratios)


def pca_stats(x: np.ndarray) -> dict:
    csum = pca_curve(x)
    xc = x - x.mean(axis=0, keepdims=True)
    _u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    var = s * s
    ratios = var / np.clip(var.sum(), 1e-12, None)
    csum = np.cumsum(ratios)
    entropy = -float(np.sum(ratios[ratios > 0] * np.log(ratios[ratios > 0])))
    return {
        "pc1_var": float(ratios[0]) if len(ratios) else np.nan,
        "pc10_cum_var": float(csum[min(9, len(csum) - 1)]) if len(csum) else np.nan,
        "dim80": int(np.searchsorted(csum, 0.8) + 1) if len(csum) else np.nan,
        "dim90": int(np.searchsorted(csum, 0.9) + 1) if len(csum) else np.nan,
        "effective_rank": float(np.exp(entropy)),
    }


def random_dim_stats(n: int, d: int, seed: int, reps: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        x = rng.normal(size=(n, d)).astype(np.float32)
        x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
        vals.append(pca_stats(x))
    out = {}
    for key in vals[0]:
        out[f"random_{key}"] = float(np.mean([v[key] for v in vals]))
    return out


def random_cum_curve(n: int, d: int, seed: int, reps: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(reps):
        x = rng.normal(size=(n, d)).astype(np.float32)
        x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
        curves.append(pca_curve(x))
    max_len = max(len(c) for c in curves)
    mat = np.full((len(curves), max_len), np.nan, dtype=np.float64)
    for i, c in enumerate(curves):
        mat[i, : len(c)] = c
    return np.nanmean(mat, axis=0), np.nanpercentile(mat, 10, axis=0), np.nanpercentile(mat, 90, axis=0)


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    xc = x - mean
    coeff = xc @ basis.T
    return np.sum(coeff * coeff, axis=1) / np.clip(np.sum(xc * xc, axis=1), 1e-12, None)


def attack_states(wrapper, x, y, attack: str, eps: float, steps: int, step_size: float, seed: int, checkpoints: int):
    if attack == "pgd":
        return pgd_trajectory(wrapper, x, y, eps, steps, step_size)
    if attack in {"mi_fgsm", "ni_fgsm", "ti_fgsm", "si_fgsm"}:
        return gradient_variant_trajectory(wrapper, x, y, eps, steps, step_size, attack)
    if attack == "signhunter":
        return signhunter_trajectory(wrapper, x, y, eps, steps, seed, checkpoints)
    raise ValueError(f"Unsupported attack: {attack}")


def collect_model_attack(wrapper, dataset, selected, model: str, attack: str, args, device):
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0 if args.step_size > 0 else eps / max(args.steps, 1)
    layer_vectors: dict[str, list[np.ndarray]] = {layer: [] for layer in LAYER_ORDER[model]}
    layer_rows: dict[str, list[dict]] = {layer: [] for layer in LAYER_ORDER[model]}
    image_outcomes = []
    for image_ord, (dataset_idx, label) in enumerate(selected):
        x_cpu, _ = dataset[dataset_idx]
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        states = attack_states(
            wrapper,
            x,
            y,
            attack,
            eps,
            args.steps if attack != "signhunter" else args.query_steps,
            step_size,
            args.seed + 1009 * image_ord + len(model),
            args.saved_checkpoints,
        )
        final = eval_all({model: wrapper}, states[-1], y)[model]
        final_success = int(final["success"])
        image_outcomes.append({"model": model, "attack": attack, "dataset_idx": dataset_idx, "image_ord": image_ord, "label": label, "final_success": final_success})

        feats_by_step = []
        for state in states:
            with torch.no_grad():
                _logits, feats, _raw = wrapper.forward_with_features(state)
            feats_by_step.append({k: v.detach().cpu().numpy()[0].astype(np.float32) for k, v in feats.items()})
        for step in range(len(feats_by_step) - 1):
            for layer in LAYER_ORDER[model]:
                if layer not in feats_by_step[step] or layer not in feats_by_step[step + 1]:
                    continue
                layer_vectors[layer].append((feats_by_step[step + 1][layer] - feats_by_step[step][layer]).astype(np.float32))
                layer_rows[layer].append({
                    "model": model,
                    "attack": attack,
                    "dataset_idx": int(dataset_idx),
                    "image_ord": int(image_ord),
                    "label": int(label),
                    "layer": layer,
                    "step": int(step),
                    "final_success": final_success,
                })
    return image_outcomes, layer_rows, layer_vectors


def analyze_layer(rows: pd.DataFrame, vectors: np.ndarray, args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_df = rows[["image_ord", "final_success"]].drop_duplicates()
    train_images = set(image_df.sample(frac=args.train_frac, random_state=args.seed)["image_ord"].astype(int).tolist())
    train_mask = rows["image_ord"].isin(train_images).to_numpy()
    success_mask = (rows["final_success"] == 1).to_numpy()
    train_success = vectors[train_mask & success_mask]
    if len(train_success) < max(8, args.k + 2):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    mean, basis = fit_basis(train_success, args.k)
    stats = pca_stats(train_success)
    rand_seed = args.seed + abs(hash((rows["model"].iloc[0], rows["layer"].iloc[0]))) % 100000
    rand_stats = random_dim_stats(len(train_success), train_success.shape[1], rand_seed)
    success_curve = pca_curve(train_success)
    random_mean, random_lo, random_hi = random_cum_curve(len(train_success), train_success.shape[1], rand_seed)
    curve_rows = []
    max_len = max(len(success_curve), len(random_mean))
    for component in range(1, max_len + 1):
        curve_rows.append({
            "model": rows["model"].iloc[0],
            "attack": rows["attack"].iloc[0],
            "layer": rows["layer"].iloc[0],
            "component": component,
            "success_cumvar": float(success_curve[component - 1]) if component <= len(success_curve) else np.nan,
            "random_cumvar_mean": float(random_mean[component - 1]) if component <= len(random_mean) else np.nan,
            "random_cumvar_lo": float(random_lo[component - 1]) if component <= len(random_lo) else np.nan,
            "random_cumvar_hi": float(random_hi[component - 1]) if component <= len(random_hi) else np.nan,
        })
    test_mask = ~train_mask
    test_rows = rows[test_mask].copy()
    test_rows["projection_energy"] = projection_energy(vectors[test_mask], mean, basis)
    if test_rows["final_success"].nunique() < 2:
        auroc = np.nan
    else:
        auroc = float(roc_auc_score(test_rows["final_success"], test_rows["projection_energy"]))
    metric = pd.DataFrame([{
        "model": rows["model"].iloc[0],
        "attack": rows["attack"].iloc[0],
        "layer": rows["layer"].iloc[0],
        "n_test_segments": int(len(test_rows)),
        "n_test_success_segments": int((test_rows["final_success"] == 1).sum()),
        "n_test_failed_segments": int((test_rows["final_success"] == 0).sum()),
        "n_train_success_segments": int(len(train_success)),
        "auroc_success_vs_failed": auroc,
        "success_mean_energy": float(test_rows[test_rows["final_success"] == 1]["projection_energy"].mean()),
        "failed_mean_energy": float(test_rows[test_rows["final_success"] == 0]["projection_energy"].mean()),
        "pc1_var_success": stats["pc1_var"],
        "pc10_cum_var_success": stats["pc10_cum_var"],
        "dim80_success": stats["dim80"],
        "dim90_success": stats["dim90"],
        "effective_rank_success": stats["effective_rank"],
        "dim80_random": rand_stats["random_dim80"],
        "dim90_random": rand_stats["random_dim90"],
        "effective_rank_random": rand_stats["random_effective_rank"],
        "random_dim80_over_success": rand_stats["random_dim80"] / max(float(stats["dim80"]), 1e-12),
    }])
    return metric, test_rows, pd.DataFrame(curve_rows)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    all_metrics = []
    all_scores = []
    all_curves = []
    all_images = []
    for model in models:
        wrapper = load_model(model, device).eval()
        selected = select_clean_correct(dataset, {model: wrapper}, argparse.Namespace(models=[model], images=args.images), device)
        print(f"[MODEL] {model} images={len(selected)}", flush=True)
        for attack in attacks:
            print(f"  [ATTACK] {attack}", flush=True)
            image_outcomes, layer_rows, layer_vectors = collect_model_attack(wrapper, dataset, selected, model, attack, args, device)
            all_images.extend(image_outcomes)
            print(f"    outcomes: success={sum(r['final_success'] for r in image_outcomes)} failed={sum(1-r['final_success'] for r in image_outcomes)}", flush=True)
            for layer in LAYER_ORDER[model]:
                if not layer_rows[layer]:
                    continue
                rows = pd.DataFrame(layer_rows[layer])
                vectors = np.stack(layer_vectors[layer]).astype(np.float32)
                metric, scores, curves = analyze_layer(rows, vectors, args)
                if not metric.empty:
                    all_metrics.append(metric)
                    all_scores.append(scores)
                    all_curves.append(curves)
        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    curves = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    images = pd.DataFrame(all_images)
    metrics.to_csv(out_dir / "layerwise_adversarial_projection_metrics.csv", index=False)
    scores.to_csv(out_dir / "layerwise_adversarial_projection_scores.csv", index=False)
    curves.to_csv(out_dir / "layerwise_adversarial_pca_spectra.csv", index=False)
    images.to_csv(out_dir / "layerwise_adversarial_image_outcomes.csv", index=False)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(metrics.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/section6_fast_layerwise_adversarial")
    parser.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    parser.add_argument("--models", default="bbb_resnet50,bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    parser.add_argument("--attacks", default="si_fgsm,signhunter")
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--eps", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--query-steps", type=int, default=160)
    parser.add_argument("--saved-checkpoints", type=int, default=21)
    parser.add_argument("--step-size", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
