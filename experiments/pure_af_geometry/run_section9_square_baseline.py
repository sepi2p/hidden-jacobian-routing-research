#!/usr/bin/env python3
"""Square Attack baseline for success-flow coefficient-search comparisons."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.square import p_selection  # noqa: E402
from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import margin, project_linf  # noqa: E402
from experiments.pure_af_geometry.run_section9_adv_success_flow_blackbox import eval_target, select_common_clean_correct  # noqa: E402


def square_attack(wrapper, x, y, eps, query_budget, seed, p_init, init_epochs):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    x0 = x.detach()
    c, h, w = x.shape[1:]
    stripe = torch.where(
        torch.rand((1, c, 1, w), generator=gen, device=x.device) < 0.5,
        -torch.ones((1, c, 1, w), device=x.device),
        torch.ones((1, c, 1, w), device=x.device),
    ) * eps
    x_best = (x0 + stripe).clamp(0, 1)
    ev = eval_target(wrapper, x_best, y)
    best = ev
    best_x = x_best
    if ev["success"]:
        return best_x.detach(), best, 1
    for q in range(2, query_budget + 1):
        perturbation = best_x - x0
        p = p_selection(p_init, q + init_epochs, query_budget)
        side = int(round(np.sqrt(p * c * h * w / c)))
        side = min(max(side, 1), h)
        top = int(torch.randint(0, h - side + 1, (1,), generator=gen, device=x.device).item())
        left = int(torch.randint(0, w - side + 1, (1,), generator=gen, device=x.device).item())
        patch = torch.where(
            torch.rand((1, c, 1, 1), generator=gen, device=x.device) < 0.5,
            -torch.ones((1, c, 1, 1), device=x.device),
            torch.ones((1, c, 1, 1), device=x.device),
        ) * eps
        perturbation[:, :, top : top + side, left : left + side] = patch
        candidate = (x0 + perturbation).clamp(0, 1)
        candidate = project_linf(candidate, x0, eps)
        ev = eval_target(wrapper, candidate, y)
        if ev["margin"] < best["margin"]:
            best = ev
            best_x = candidate.detach()
        if ev["success"]:
            return candidate.detach(), ev, q
    return best_x.detach(), best, np.nan


def summarize(df):
    return df.groupby(["target_model", "attack", "eps_255", "query_budget"], dropna=False).agg(
        asr=("target_success", "mean"),
        n=("target_success", "size"),
        mean_best_margin=("target_margin", "mean"),
        mean_margin_drop=("margin_drop", "mean"),
        mean_queries_to_success=("queries_to_success", "mean"),
        median_queries_to_success=("queries_to_success", "median"),
    ).reset_index()


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    all_models = [args.source] + [t for t in targets if t != args.source]
    wrappers = {m: load_model(m, device).eval() for m in all_models}
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    selected = select_common_clean_correct(dataset, wrappers, args.train_images + args.test_images, device)
    test_items = selected[args.train_images: args.train_images + args.test_images]
    eps = args.eps / 255.0
    rows = []
    print(f"[DATA] test={len(test_items)} targets={targets}", flush=True)
    for image_ord, (idx, label) in enumerate(test_items):
        x_cpu, _ = dataset[idx]
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        clean = {t: eval_target(wrappers[t], x, y) for t in targets}
        for target in targets:
            _adv, ev, q_success = square_attack(
                wrappers[target],
                x,
                y,
                eps,
                args.query_budget,
                args.seed + idx * 1009 + len(target),
                args.p_init,
                args.init_epochs,
            )
            rows.append(
                {
                    "target_model": target,
                    "dataset_idx": int(idx),
                    "image_ord": int(image_ord),
                    "label": int(label),
                    "attack": "square",
                    "eps": float(eps),
                    "eps_255": float(eps * 255),
                    "query_budget": int(args.query_budget),
                    "target_success": int(ev["success"]),
                    "target_pred": int(ev["pred"]),
                    "target_margin": float(ev["margin"]),
                    "target_true_prob": float(ev["true_prob"]),
                    "clean_margin": float(clean[target]["margin"]),
                    "clean_true_prob": float(clean[target]["true_prob"]),
                    "margin_drop": float(clean[target]["margin"] - ev["margin"]),
                    "true_prob_drop": float(clean[target]["true_prob"] - ev["true_prob"]),
                    "queries_to_success": float(q_success) if not np.isnan(q_success) else np.nan,
                }
            )
        if (image_ord + 1) % args.checkpoint_every == 0:
            df = pd.DataFrame(rows)
            df.to_csv(out_dir / "partial_square_baseline_per_image.csv", index=False)
            summarize(df).to_csv(out_dir / "partial_square_baseline_summary.csv", index=False)
            print(f"  {image_ord + 1}/{len(test_items)} rows={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    summary = summarize(df)
    df.to_csv(out_dir / "square_baseline_per_image.csv", index=False)
    summary.to_csv(out_dir / "square_baseline_summary.csv", index=False)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print("[SUMMARY]", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/section9_square_baseline_q100")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--source", default="bbb_resnet50")
    p.add_argument("--targets", default="bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--train-images", type=int, default=80)
    p.add_argument("--test-images", type=int, default=60)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--query-budget", type=int, default=100)
    p.add_argument("--p-init", type=float, default=0.3)
    p.add_argument("--init-epochs", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=59)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
