#!/usr/bin/env python3
"""Final black-box validation for adversarial success-flow coefficient search.

This compares a low-data success-flow basis against random coefficient search,
Square Attack, and source CE-PGD transfer under matched target-query budgets.
The success-flow basis is estimated from a small number of successful PGD
transport segments on a surrogate model.
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
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.square import p_selection  # noqa: E402
from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model, normalize_rows  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import project_linf  # noqa: E402
from experiments.pure_af_geometry.run_section9_adv_success_flow_blackbox import (  # noqa: E402
    BEST_LAYER,
    collect_success_transport,
    eval_target,
    fit_pca,
    make_pc_pixel_basis,
    make_random_pixel_basis,
    select_common_clean_correct,
)
from experiments.pure_af_geometry.run_section9_adv_success_flow_intervention import choose_pc_signs, ce_pgd  # noqa: E402


def coeff_search_trace(target_wrapper, x, y, pixel_basis, eps, max_queries, milestones, seed):
    rng = np.random.default_rng(seed)
    best = eval_target(target_wrapper, x, y)
    first_success = 0 if best["success"] else np.nan
    rows = {}
    for q in range(1, max_queries + 1):
        alpha = rng.normal(size=len(pixel_basis)).astype(np.float32)
        direction = torch.zeros_like(x)
        for a, u in zip(alpha, pixel_basis):
            direction = direction + float(a) * u
        adv = project_linf(x + eps * direction.sign(), x, eps)
        ev = eval_target(target_wrapper, adv, y)
        if ev["margin"] < best["margin"]:
            best = ev
        if ev["success"] and np.isnan(first_success):
            first_success = q
        if q in milestones:
            snap = dict(best)
            snap["queries_to_success"] = float(first_success) if not np.isnan(first_success) and first_success <= q else np.nan
            rows[q] = snap
    return rows


def square_trace(target_wrapper, x, y, eps, max_queries, milestones, seed, p_init, init_epochs):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    x0 = x.detach()
    c, h, w = x.shape[1:]
    stripe = torch.where(
        torch.rand((1, c, 1, w), generator=gen, device=x.device) < 0.5,
        -torch.ones((1, c, 1, w), device=x.device),
        torch.ones((1, c, 1, w), device=x.device),
    ) * eps
    x_best = (x0 + stripe).clamp(0, 1)
    best = eval_target(target_wrapper, x_best, y)
    first_success = 1 if best["success"] else np.nan
    rows = {}
    if 1 in milestones:
        snap = dict(best)
        snap["queries_to_success"] = float(first_success) if not np.isnan(first_success) else np.nan
        rows[1] = snap
    for q in range(2, max_queries + 1):
        perturbation = x_best - x0
        p = p_selection(p_init, q + init_epochs, max_queries)
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
        candidate = project_linf((x0 + perturbation).clamp(0, 1), x0, eps)
        ev = eval_target(target_wrapper, candidate, y)
        if ev["margin"] < best["margin"]:
            best = ev
            x_best = candidate.detach()
        if ev["success"] and np.isnan(first_success):
            first_success = q
        if q in milestones:
            snap = dict(best)
            snap["queries_to_success"] = float(first_success) if not np.isnan(first_success) and first_success <= q else np.nan
            rows[q] = snap
    return rows


def summarize(df):
    return df.groupby(["target_model", "method", "query_budget"], dropna=False).agg(
        asr=("target_success", "mean"),
        n=("target_success", "size"),
        mean_best_margin=("target_margin", "mean"),
        mean_margin_drop=("margin_drop", "mean"),
        mean_queries_to_success=("queries_to_success", "mean"),
        median_queries_to_success=("queries_to_success", "median"),
    ).reset_index()


def bootstrap_ci(values, rng, reps=1000):
    vals = np.asarray(values, dtype=np.float32)
    if len(vals) == 0:
        return np.nan, np.nan
    boots = [float(np.mean(vals[rng.integers(0, len(vals), len(vals))])) for _ in range(reps)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_tests(df):
    rows = []
    rng = np.random.default_rng(123)
    methods = ["success_flow_coeff", "random_coeff", "square", "source_ce_pgd_transfer"]
    for (target, budget), g in df.groupby(["target_model", "query_budget"]):
        pivot = g.pivot_table(index="dataset_idx", columns="method", values="target_success", aggfunc="max")
        for other in ["random_coeff", "square", "source_ce_pgd_transfer"]:
            if "success_flow_coeff" not in pivot or other not in pivot:
                continue
            diff = pivot["success_flow_coeff"].fillna(0).to_numpy() - pivot[other].fillna(0).to_numpy()
            lo, hi = bootstrap_ci(diff, rng)
            rows.append(
                {
                    "target_model": target,
                    "query_budget": budget,
                    "comparison": f"success_flow_coeff_minus_{other}",
                    "paired_success_diff": float(np.mean(diff)),
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "n": int(len(diff)),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary, out_dir):
    targets = list(summary.target_model.unique())
    methods = ["success_flow_coeff", "random_coeff", "square", "source_ce_pgd_transfer"]
    colors = {
        "success_flow_coeff": "#2563eb",
        "random_coeff": "#9ca3af",
        "square": "#dc2626",
        "source_ce_pgd_transfer": "#16a34a",
    }
    fig, axes = plt.subplots(1, len(targets), figsize=(4.6 * len(targets), 3.3), sharey=True, constrained_layout=True)
    if len(targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        g = summary[summary.target_model == target]
        for method in methods:
            m = g[g.method == method].sort_values("query_budget")
            if not m.empty:
                ax.plot(m.query_budget, m.asr, marker="o", label=method.replace("_", " "), color=colors[method])
        ax.set_title(target)
        ax.set_xlabel("target queries")
        ax.set_ylabel("ASR")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.savefig(out_dir / "final_blackbox_validation_asr_curves.png", dpi=190, bbox_inches="tight")
    fig.savefig(out_dir / "final_blackbox_validation_asr_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    all_models = [args.source] + [t for t in targets if t != args.source]
    wrappers = {m: load_model(m, device).eval() for m in all_models}
    source_wrapper = wrappers[args.source]
    source_layer = BEST_LAYER[args.source]
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    selected = select_common_clean_correct(dataset, wrappers, args.train_images + args.test_images, device)
    train_items = selected[: args.train_images]
    test_items = selected[args.train_images : args.train_images + args.test_images]
    eps = args.eps / 255.0
    milestones = sorted(int(x) for x in args.query_budgets.split(",") if x.strip())
    max_queries = max(milestones)
    print(f"[DATA] train={len(train_items)} test={len(test_items)} source={args.source} layer={source_layer}", flush=True)

    vectors, basis_image_rows = collect_success_transport(source_wrapper, dataset, train_items, source_layer, args, device)
    norm_vectors = normalize_rows(vectors)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(norm_vectors))
    nseg = min(args.basis_segments, len(norm_vectors))
    _mean, basis, ratios = fit_pca(norm_vectors[perm[:nseg]], args.pcs)
    signs, _sign_rows = choose_pc_signs(source_wrapper, dataset, train_items, source_layer, basis, args, device)
    print(f"[BASIS] nseg={nseg} pc1={ratios[0]:.3f} pc5cum={np.sum(ratios[:min(args.pcs, len(ratios))]):.3f}", flush=True)

    rows = []
    for image_ord, (idx, label) in enumerate(test_items):
        x_cpu, _ = dataset[idx]
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        pc_basis = make_pc_pixel_basis(source_wrapper, x, source_layer, basis, signs, args.pcs)
        rand_basis = make_random_pixel_basis(x, args.pcs, args.seed + idx * 19)
        source_adv = ce_pgd(source_wrapper, x, y, eps, args.source_pgd_steps, eps / max(args.source_pgd_steps, 1))
        clean = {t: eval_target(wrappers[t], x, y) for t in targets}
        traces_by_target = {}
        for target in targets:
            traces_by_target[(target, "success_flow_coeff")] = coeff_search_trace(
                wrappers[target],
                x,
                y,
                pc_basis,
                eps,
                max_queries,
                set(milestones),
                args.seed + idx * 1009 + len(target),
            )
            traces_by_target[(target, "random_coeff")] = coeff_search_trace(
                wrappers[target],
                x,
                y,
                rand_basis,
                eps,
                max_queries,
                set(milestones),
                args.seed + idx * 1009 + len(target) + 99,
            )
            traces_by_target[(target, "square")] = square_trace(
                wrappers[target],
                x,
                y,
                eps,
                max_queries,
                set(milestones),
                args.seed + idx * 1009 + len(target) + 199,
                args.p_init,
                args.init_epochs,
            )
            ce_ev = eval_target(wrappers[target], source_adv, y)
            for budget in milestones:
                for method in ["success_flow_coeff", "random_coeff", "square"]:
                    ev = traces_by_target[(target, method)][budget]
                    rows.append(
                        {
                            "target_model": target,
                            "dataset_idx": int(idx),
                            "image_ord": int(image_ord),
                            "label": int(label),
                            "method": method,
                            "query_budget": int(budget),
                            "eps_255": float(args.eps),
                            "target_success": int(ev["success"]),
                            "target_pred": int(ev["pred"]),
                            "target_margin": float(ev["margin"]),
                            "target_true_prob": float(ev["true_prob"]),
                            "clean_margin": float(clean[target]["margin"]),
                            "clean_true_prob": float(clean[target]["true_prob"]),
                            "margin_drop": float(clean[target]["margin"] - ev["margin"]),
                            "true_prob_drop": float(clean[target]["true_prob"] - ev["true_prob"]),
                            "queries_to_success": ev["queries_to_success"],
                        }
                    )
                rows.append(
                    {
                        "target_model": target,
                        "dataset_idx": int(idx),
                        "image_ord": int(image_ord),
                        "label": int(label),
                        "method": "source_ce_pgd_transfer",
                        "query_budget": int(budget),
                        "eps_255": float(args.eps),
                        "target_success": int(ce_ev["success"]),
                        "target_pred": int(ce_ev["pred"]),
                        "target_margin": float(ce_ev["margin"]),
                        "target_true_prob": float(ce_ev["true_prob"]),
                        "clean_margin": float(clean[target]["margin"]),
                        "clean_true_prob": float(clean[target]["true_prob"]),
                        "margin_drop": float(clean[target]["margin"] - ce_ev["margin"]),
                        "true_prob_drop": float(clean[target]["true_prob"] - ce_ev["true_prob"]),
                        "queries_to_success": 1.0 if ce_ev["success"] else np.nan,
                    }
                )
        if (image_ord + 1) % args.checkpoint_every == 0:
            df = pd.DataFrame(rows)
            df.to_csv(out_dir / "partial_final_blackbox_validation_per_image.csv", index=False)
            summarize(df).to_csv(out_dir / "partial_final_blackbox_validation_summary.csv", index=False)
            print(f"  {image_ord + 1}/{len(test_items)} rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    summary = summarize(df)
    tests = paired_tests(df)
    df.to_csv(out_dir / "final_blackbox_validation_per_image.csv", index=False)
    summary.to_csv(out_dir / "final_blackbox_validation_summary.csv", index=False)
    tests.to_csv(out_dir / "final_blackbox_validation_paired_tests.csv", index=False)
    pd.DataFrame(basis_image_rows).assign(source_model=args.source, source_layer=source_layer).to_csv(out_dir / "basis_image_outcomes.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_model": args.source,
                "source_layer": source_layer,
                "basis_segments": int(nseg),
                "pc": i + 1,
                "variance_explained": float(ratios[i]),
                "cumulative_variance": float(np.sum(ratios[: i + 1])),
                "n_available_success_transport_vectors": int(len(norm_vectors)),
                "d": int(norm_vectors.shape[1]),
            }
            for i in range(min(args.pcs, len(ratios)))
        ]
    ).to_csv(out_dir / "basis_metadata.csv", index=False)
    plot_summary(summary, out_dir)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print("[SUMMARY]", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("[PAIRED]", flush=True)
    print(tests.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/section9_final_blackbox_validation")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--source", default="bbb_resnet50")
    p.add_argument("--targets", default="bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--train-images", type=int, default=80)
    p.add_argument("--test-images", type=int, default=200)
    p.add_argument("--basis-segments", type=int, default=20)
    p.add_argument("--basis-eps", type=float, default=2.0)
    p.add_argument("--basis-steps", type=int, default=5)
    p.add_argument("--basis-step-size", type=float, default=0.0)
    p.add_argument("--sign-images", type=int, default=40)
    p.add_argument("--sign-eps", type=float, default=2.0)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--pcs", type=int, default=5)
    p.add_argument("--query-budgets", default="25,50,100,250")
    p.add_argument("--source-pgd-steps", type=int, default=10)
    p.add_argument("--p-init", type=float, default=0.3)
    p.add_argument("--init-epochs", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=67)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
