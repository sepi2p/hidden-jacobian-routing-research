#!/usr/bin/env python3
"""Black-box tests for adversarial success-flow coordinates.

The source model is used as a white-box surrogate to learn successful PGD
transport PCs and pull them back to pixel space. Target models are evaluated
only through their outputs, either by direct transfer or by low-dimensional
coefficient search over the pulled-back PC directions.
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

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model, normalize_rows  # noqa: E402
from experiments.pure_af_geometry.analyze_cifar_benchmark_optimizer_transport import pgd_trajectory  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import margin  # noqa: E402
from experiments.pure_af_geometry.run_section9_adv_success_flow_intervention import (  # noqa: E402
    BEST_LAYER,
    attack_with_direction,
    ce_pgd,
    collect_success_transport,
    eval_one,
    fit_pca,
    project_linf,
)


def select_common_clean_correct(dataset, wrappers, n_total: int, device):
    selected = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    model_names = list(wrappers)
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        ok = True
        with torch.no_grad():
            for name in model_names:
                if int(wrappers[name](x).argmax(1).item()) != int(y.item()):
                    ok = False
                    break
        if ok:
            selected.append((idx, int(y.item())))
        if len(selected) >= n_total:
            break
    return selected


def eval_target(wrapper, x, y):
    with torch.no_grad():
        logits = wrapper(x)
        pred = int(logits.argmax(1).item())
        probs = F.softmax(logits, dim=1)
        return {
            "pred": pred,
            "success": int(pred != int(y.item())),
            "margin": float(margin(logits, y).item()),
            "true_prob": float(probs[0, int(y.item())].item()),
        }


def feature_pc_grad(source_wrapper, x, layer, direction_np):
    x_probe = x.detach().requires_grad_(True)
    _logits, feats, _raw = source_wrapper.forward_with_features(x_probe)
    h = feats[layer]
    direction = torch.as_tensor(direction_np, dtype=h.dtype, device=h.device).view_as(h)
    scalar = (h * direction).sum()
    return torch.autograd.grad(scalar, x_probe)[0].detach()


def choose_pc_signs(source_wrapper, dataset, train_items, layer, basis, args, device):
    eps = args.sign_eps / 255.0
    rows = []
    signs = {}
    for pc_idx, direction in enumerate(basis[: args.pcs], start=1):
        scores = {}
        for sign in [-1, 1]:
            drops = []
            for idx, label in train_items[: args.sign_images]:
                x_cpu, _ = dataset[idx]
                x = x_cpu.unsqueeze(0).to(device)
                y = torch.tensor([label], device=device)
                clean = eval_one(source_wrapper, x, y)
                adv, _ = attack_with_direction(source_wrapper, x, y, layer, direction, sign, eps, 1, eps)
                after = eval_one(source_wrapper, adv, y)
                drops.append(clean["margin"] - after["margin"])
            scores[sign] = float(np.mean(drops)) if drops else float("-inf")
            rows.append({"pc": pc_idx, "sign": sign, "mean_source_margin_drop": scores[sign]})
        signs[pc_idx] = 1 if scores[1] >= scores[-1] else -1
    return signs, rows


def normalize_pixel_basis(vecs):
    out = []
    for v in vecs:
        flat_norm = torch.linalg.vector_norm(v.flatten(1), dim=1).view(-1, 1, 1, 1)
        out.append(v / torch.clamp(flat_norm, min=1e-12))
    return out


def make_pc_pixel_basis(source_wrapper, x, layer, basis, signs, pcs):
    vecs = []
    for pc_idx in range(1, min(pcs, basis.shape[0]) + 1):
        vecs.append(feature_pc_grad(source_wrapper, x, layer, signs[pc_idx] * basis[pc_idx - 1]))
    return normalize_pixel_basis(vecs)


def make_random_pixel_basis(x, k, seed):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    vecs = [torch.randn(x.shape, generator=gen, device=x.device) for _ in range(k)]
    return normalize_pixel_basis(vecs)


def coefficient_search(target_wrapper, x, y, pixel_basis, eps, queries, seed):
    rng = np.random.default_rng(seed)
    best = eval_target(target_wrapper, x, y)
    best_query = 0 if best["success"] else np.nan
    for q in range(1, queries + 1):
        alpha = rng.normal(size=len(pixel_basis)).astype(np.float32)
        direction = torch.zeros_like(x)
        for a, u in zip(alpha, pixel_basis):
            direction = direction + float(a) * u
        adv = project_linf(x + eps * direction.sign(), x, eps)
        ev = eval_target(target_wrapper, adv, y)
        if ev["margin"] < best["margin"]:
            best = ev
        if ev["success"]:
            best_query = q
            best = ev
            break
    return best, best_query


def summarize_transfer(df):
    return df.groupby(["source_model", "target_model", "variant", "eps_255", "steps"], dropna=False).agg(
        asr=("target_success", "mean"),
        n=("target_success", "size"),
        mean_margin_drop=("margin_drop", "mean"),
        mean_prob_drop=("true_prob_drop", "mean"),
    ).reset_index()


def summarize_search(df):
    return df.groupby(["source_model", "target_model", "variant", "eps_255", "query_budget"], dropna=False).agg(
        asr=("target_success", "mean"),
        n=("target_success", "size"),
        mean_best_margin=("target_margin", "mean"),
        mean_queries_to_success=("queries_to_success", "mean"),
        median_queries_to_success=("queries_to_success", "median"),
    ).reset_index()


def plot_transfer(summary, out_dir):
    sub = summary[(summary.eps_255 == summary.eps_255.max()) & (summary.steps == summary.steps.max())].copy()
    target_order = [x for x in sub.target_model.unique() if x != sub.source_model.iloc[0]]
    variants = ["random_pixel", "adv_pc1", "adv_pc2", "adv_pc3", "adv_pc4", "adv_pc5", "source_ce_pgd"]
    fig, axes = plt.subplots(1, len(target_order), figsize=(4.3 * len(target_order), 3.2), sharey=True, constrained_layout=True)
    if len(target_order) == 1:
        axes = [axes]
    for ax, target in zip(axes, target_order):
        g = sub[sub.target_model == target].copy()
        g["variant"] = pd.Categorical(g["variant"], categories=variants, ordered=True)
        g = g.sort_values("variant")
        ax.bar(g.variant.astype(str), g.asr, color="#6aaed6")
        ax.set_title(target)
        ax.set_ylim(0, 1)
        ax.set_ylabel("transfer ASR")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "adv_success_flow_transfer_asr.png", dpi=190, bbox_inches="tight")
    fig.savefig(out_dir / "adv_success_flow_transfer_asr.pdf", bbox_inches="tight")
    plt.close(fig)


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
    source_wrapper = wrappers[args.source]
    layer = BEST_LAYER[args.source]
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    selected = select_common_clean_correct(dataset, wrappers, args.train_images + args.test_images, device)
    train_items = selected[: args.train_images]
    test_items = selected[args.train_images: args.train_images + args.test_images]
    print(f"[DATA] common clean-correct train={len(train_items)} test={len(test_items)} source={args.source} layer={layer}", flush=True)

    vectors, basis_image_rows = collect_success_transport(source_wrapper, dataset, train_items, layer, args, device)
    if len(vectors) < max(args.pcs + 2, 8):
        raise RuntimeError(f"Not enough successful source transport vectors: {len(vectors)}")
    norm_vectors = normalize_rows(vectors)
    _mean, basis, ratios = fit_pca(norm_vectors, args.pcs)
    signs, sign_rows = choose_pc_signs(source_wrapper, dataset, train_items, layer, basis, args, device)

    transfer_rows = []
    search_rows = []
    eps_values = [float(x) / 255.0 for x in args.eps.split(",")]
    steps_values = [int(x) for x in args.steps.split(",")]
    for image_ord, (idx, label) in enumerate(test_items):
        x_cpu, _ = dataset[idx]
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        clean_by_model = {m: eval_target(w, x, y) for m, w in wrappers.items()}
        pc_pixel_basis = make_pc_pixel_basis(source_wrapper, x, layer, basis, signs, args.pcs)
        random_pixel_basis = make_random_pixel_basis(x, args.pcs, args.seed + idx * 19)
        for eps in eps_values:
            for steps in steps_values:
                step_size = eps / max(steps, 1)
                variants = [("random_pixel", None), ("source_ce_pgd", None)]
                for pc_idx in range(1, min(args.pcs, basis.shape[0]) + 1):
                    variants.append((f"adv_pc{pc_idx}", pc_idx))
                for variant, pc_idx in variants:
                    if variant == "source_ce_pgd":
                        adv = ce_pgd(source_wrapper, x, y, eps, steps, step_size)
                    elif variant == "random_pixel":
                        gen = torch.Generator(device=device).manual_seed(args.seed + idx * 101 + steps)
                        direction = torch.randn(x.shape, generator=gen, device=device).sign()
                        adv = x.clone()
                        for _ in range(steps):
                            adv = project_linf(adv + step_size * direction, x, eps)
                    else:
                        direction = signs[pc_idx] * basis[pc_idx - 1]
                        adv, _ = attack_with_direction(source_wrapper, x, y, layer, direction, 1, eps, steps, step_size)
                    for target_name, target_wrapper in wrappers.items():
                        ev = eval_target(target_wrapper, adv, y)
                        clean = clean_by_model[target_name]
                        transfer_rows.append({
                            "source_model": args.source,
                            "target_model": target_name,
                            "dataset_idx": int(idx),
                            "image_ord": int(image_ord),
                            "label": int(label),
                            "variant": variant,
                            "eps": float(eps),
                            "eps_255": float(eps * 255),
                            "steps": int(steps),
                            "target_success": int(ev["success"]),
                            "target_pred": int(ev["pred"]),
                            "target_margin": float(ev["margin"]),
                            "target_true_prob": float(ev["true_prob"]),
                            "clean_margin": float(clean["margin"]),
                            "clean_true_prob": float(clean["true_prob"]),
                            "margin_drop": float(clean["margin"] - ev["margin"]),
                            "true_prob_drop": float(clean["true_prob"] - ev["true_prob"]),
                        })
            for target_name in targets:
                for variant, basis_pixels in [("pc_coeff_search", pc_pixel_basis), ("random_coeff_search", random_pixel_basis)]:
                    ev, first_q = coefficient_search(
                        wrappers[target_name],
                        x,
                        y,
                        basis_pixels,
                        eps,
                        args.query_budget,
                        args.seed + idx * 1009 + len(target_name) + (0 if variant.startswith("pc") else 99),
                    )
                    clean = clean_by_model[target_name]
                    search_rows.append({
                        "source_model": args.source,
                        "target_model": target_name,
                        "dataset_idx": int(idx),
                        "image_ord": int(image_ord),
                        "label": int(label),
                        "variant": variant,
                        "eps": float(eps),
                        "eps_255": float(eps * 255),
                        "query_budget": int(args.query_budget),
                        "target_success": int(ev["success"]),
                        "target_pred": int(ev["pred"]),
                        "target_margin": float(ev["margin"]),
                        "target_true_prob": float(ev["true_prob"]),
                        "clean_margin": float(clean["margin"]),
                        "clean_true_prob": float(clean["true_prob"]),
                        "margin_drop": float(clean["margin"] - ev["margin"]),
                        "true_prob_drop": float(clean["true_prob"] - ev["true_prob"]),
                        "queries_to_success": float(first_q) if not np.isnan(first_q) else np.nan,
                    })
        if (image_ord + 1) % args.checkpoint_every == 0:
            pd.DataFrame(transfer_rows).to_csv(out_dir / "partial_blackbox_transfer_per_image.csv", index=False)
            pd.DataFrame(search_rows).to_csv(out_dir / "partial_blackbox_coeff_search_per_image.csv", index=False)
            print(f"  eval {image_ord + 1}/{len(test_items)} transfer_rows={len(transfer_rows)} search_rows={len(search_rows)}", flush=True)

    transfer_df = pd.DataFrame(transfer_rows)
    search_df = pd.DataFrame(search_rows)
    transfer_summary = summarize_transfer(transfer_df)
    search_summary = summarize_search(search_df)
    transfer_df.to_csv(out_dir / "blackbox_transfer_per_image.csv", index=False)
    search_df.to_csv(out_dir / "blackbox_coeff_search_per_image.csv", index=False)
    transfer_summary.to_csv(out_dir / "blackbox_transfer_summary.csv", index=False)
    search_summary.to_csv(out_dir / "blackbox_coeff_search_summary.csv", index=False)
    pd.DataFrame(sign_rows).assign(source_model=args.source, layer=layer).to_csv(out_dir / "pc_sign_selection.csv", index=False)
    pd.DataFrame(basis_image_rows).assign(source_model=args.source, layer=layer).to_csv(out_dir / "basis_image_outcomes.csv", index=False)
    pd.DataFrame([
        {
            "source_model": args.source,
            "layer": layer,
            "pc": i + 1,
            "variance_explained": float(ratios[i]),
            "cumulative_variance": float(np.sum(ratios[: i + 1])),
            "n_success_transport_vectors": int(len(norm_vectors)),
            "d": int(norm_vectors.shape[1]),
        }
        for i in range(min(args.pcs, len(ratios)))
    ]).to_csv(out_dir / "basis_metadata.csv", index=False)
    plot_transfer(transfer_summary, out_dir)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print("[TRANSFER]", flush=True)
    print(transfer_summary[(transfer_summary.eps_255 == transfer_summary.eps_255.max()) & (transfer_summary.steps == transfer_summary.steps.max())].to_string(index=False), flush=True)
    print("[COEFF SEARCH]", flush=True)
    print(search_summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/section9_adv_success_flow_blackbox")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--source", default="bbb_resnet50")
    p.add_argument("--targets", default="bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--train-images", type=int, default=60)
    p.add_argument("--test-images", type=int, default=40)
    p.add_argument("--basis-eps", type=float, default=2.0)
    p.add_argument("--basis-steps", type=int, default=5)
    p.add_argument("--basis-step-size", type=float, default=0.0)
    p.add_argument("--sign-images", type=int, default=30)
    p.add_argument("--sign-eps", type=float, default=2.0)
    p.add_argument("--eps", default="8")
    p.add_argument("--steps", default="10")
    p.add_argument("--pcs", type=int, default=5)
    p.add_argument("--query-budget", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=31)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
