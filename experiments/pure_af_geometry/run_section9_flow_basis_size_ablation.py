#!/usr/bin/env python3
"""Ablate how many successful adversarial transport vectors are needed.

The script estimates a surrogate success-flow PCA basis from increasing numbers
of successful PGD transport segments, then evaluates the resulting pullback
subspace with target black-box coefficient search.
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

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model, normalize_rows  # noqa: E402
from experiments.pure_af_geometry.run_section9_adv_success_flow_blackbox import (  # noqa: E402
    BEST_LAYER,
    coefficient_search,
    collect_success_transport,
    eval_target,
    fit_pca,
    make_pc_pixel_basis,
    make_random_pixel_basis,
    select_common_clean_correct,
)
from experiments.pure_af_geometry.run_section9_adv_success_flow_intervention import choose_pc_signs  # noqa: E402


def parse_segment_counts(spec: str, max_count: int):
    out = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        out.append(max_count if token in {"all", "full"} else int(token))
    return sorted(set(min(x, max_count) for x in out if x > 0))


def summarize(df: pd.DataFrame):
    return df.groupby(["target_model", "basis_segments", "variant", "eps_255", "query_budget"], dropna=False).agg(
        asr=("target_success", "mean"),
        n=("target_success", "size"),
        mean_best_margin=("target_margin", "mean"),
        mean_margin_drop=("margin_drop", "mean"),
        mean_queries_to_success=("queries_to_success", "mean"),
        median_queries_to_success=("queries_to_success", "median"),
    ).reset_index()


def plot_summary(summary: pd.DataFrame, out_dir: Path):
    pc = summary[summary.variant == "pc_coeff_search"].copy()
    rand = summary[summary.variant == "random_coeff_search"].copy()
    targets = list(pc.target_model.unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(4.4 * len(targets), 3.2), sharey=True, constrained_layout=True)
    if len(targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        g = pc[pc.target_model == target].sort_values("basis_segments")
        r = rand[rand.target_model == target].sort_values("basis_segments")
        ax.plot(g.basis_segments, g.asr, marker="o", label="success-flow")
        ax.plot(r.basis_segments, r.asr, marker="o", linestyle="--", label="random")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title(target)
        ax.set_xlabel("successful transport segments used")
        ax.set_ylabel("ASR")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out_dir / "flow_basis_size_ablation_asr.png", dpi=190, bbox_inches="tight")
    fig.savefig(out_dir / "flow_basis_size_ablation_asr.pdf", bbox_inches="tight")
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
    source_layer = BEST_LAYER[args.source]
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    selected = select_common_clean_correct(dataset, wrappers, args.train_images + args.test_images, device)
    train_items = selected[: args.train_images]
    test_items = selected[args.train_images: args.train_images + args.test_images]
    print(f"[DATA] train={len(train_items)} test={len(test_items)} source={args.source} layer={source_layer}", flush=True)

    vectors, basis_image_rows = collect_success_transport(source_wrapper, dataset, train_items, source_layer, args, device)
    if len(vectors) < max(args.pcs + 2, 8):
        raise RuntimeError(f"Not enough successful source transport vectors: {len(vectors)}")
    norm_vectors = normalize_rows(vectors)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(norm_vectors))
    segment_counts = parse_segment_counts(args.segment_counts, len(norm_vectors))
    eps = args.eps / 255.0
    rows = []
    basis_rows = []
    for nseg in segment_counts:
        basis_vectors = norm_vectors[perm[:nseg]]
        _mean, basis, ratios = fit_pca(basis_vectors, args.pcs)
        signs, sign_rows = choose_pc_signs(source_wrapper, dataset, train_items, source_layer, basis, args, device)
        basis_rows.extend(
            {
                "basis_segments": int(nseg),
                "pc": int(i + 1),
                "variance_explained": float(ratios[i]),
                "cumulative_variance": float(np.sum(ratios[: i + 1])),
                "n_available_success_transport_vectors": int(len(norm_vectors)),
                "d": int(norm_vectors.shape[1]),
            }
            for i in range(min(args.pcs, len(ratios)))
        )
        print(f"[BASIS] nseg={nseg} pc1={ratios[0]:.3f} pc5cum={np.sum(ratios[:min(args.pcs, len(ratios))]):.3f}", flush=True)
        for image_ord, (idx, label) in enumerate(test_items):
            x_cpu, _ = dataset[idx]
            x = x_cpu.unsqueeze(0).to(device)
            y = torch.tensor([label], device=device)
            pc_pixel_basis = make_pc_pixel_basis(source_wrapper, x, source_layer, basis, signs, args.pcs)
            random_pixel_basis = make_random_pixel_basis(x, args.pcs, args.seed + idx * 19 + nseg)
            clean_by_model = {m: eval_target(w, x, y) for m, w in wrappers.items()}
            for target in targets:
                for variant, pixel_basis in [("pc_coeff_search", pc_pixel_basis), ("random_coeff_search", random_pixel_basis)]:
                    ev, first_q = coefficient_search(
                        wrappers[target],
                        x,
                        y,
                        pixel_basis,
                        eps,
                        args.query_budget,
                        args.seed + idx * 1009 + nseg + len(target) + (0 if variant.startswith("pc") else 99),
                    )
                    clean = clean_by_model[target]
                    rows.append(
                        {
                            "source_model": args.source,
                            "target_model": target,
                            "source_layer": source_layer,
                            "basis_segments": int(nseg),
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
                        }
                    )
            if (image_ord + 1) % args.checkpoint_every == 0:
                df = pd.DataFrame(rows)
                df.to_csv(out_dir / "partial_flow_basis_size_ablation_per_image.csv", index=False)
                summarize(df).to_csv(out_dir / "partial_flow_basis_size_ablation_summary.csv", index=False)
                print(f"  nseg={nseg}: {image_ord + 1}/{len(test_items)} rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    summary = summarize(df)
    df.to_csv(out_dir / "flow_basis_size_ablation_per_image.csv", index=False)
    summary.to_csv(out_dir / "flow_basis_size_ablation_summary.csv", index=False)
    pd.DataFrame(basis_rows).to_csv(out_dir / "flow_basis_size_ablation_basis_metadata.csv", index=False)
    pd.DataFrame(basis_image_rows).assign(source_model=args.source, layer=source_layer).to_csv(out_dir / "basis_image_outcomes.csv", index=False)
    plot_summary(summary, out_dir)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print("[SUMMARY]", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/section9_flow_basis_size_ablation")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--source", default="bbb_resnet50")
    p.add_argument("--targets", default="bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--train-images", type=int, default=80)
    p.add_argument("--test-images", type=int, default=60)
    p.add_argument("--basis-eps", type=float, default=2.0)
    p.add_argument("--basis-steps", type=int, default=5)
    p.add_argument("--basis-step-size", type=float, default=0.0)
    p.add_argument("--sign-images", type=int, default=40)
    p.add_argument("--sign-eps", type=float, default=2.0)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--pcs", type=int, default=5)
    p.add_argument("--segment-counts", default="20,50,100,200,all")
    p.add_argument("--query-budget", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=59)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
