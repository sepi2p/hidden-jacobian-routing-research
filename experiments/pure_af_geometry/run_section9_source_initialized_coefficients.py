#!/usr/bin/env python3
"""Source-initialized universal coefficient adaptation.

Estimate universal success-flow coefficients on the surrogate/source model,
then adapt target-model coefficients by sampling around the source solution.
This tests whether target black-box calibration can be accelerated by a
cross-model coefficient prior.
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
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model, normalize_rows  # noqa: E402
from experiments.pure_af_geometry.run_section9_adv_success_flow_blackbox import (  # noqa: E402
    BEST_LAYER,
    collect_success_transport,
    fit_pca,
)
from experiments.pure_af_geometry.run_section9_universal_success_flow_coefficients import (  # noqa: E402
    build_image_bases,
    eval_alpha,
    evaluate_policy,
    random_search_alpha,
    select_splits,
    summarize,
)


def seeded_search_alpha(target_wrapper, records, eps, k, query_budget, seed_alpha, seed, radius: float):
    rng = np.random.default_rng(seed)
    if not records:
        return seed_alpha.astype(np.float32), 0, np.nan, np.nan
    n_candidates = max(1, int(query_budget // max(len(records), 1)))
    candidates = [np.asarray(seed_alpha, dtype=np.float32)]
    for scale in [0.25, 0.5, 1.0, 1.5]:
        candidates.append((scale * seed_alpha).astype(np.float32))
    for i in range(k):
        e = np.zeros(k, dtype=np.float32)
        e[i] = radius
        candidates.append((seed_alpha + e).astype(np.float32))
        candidates.append((seed_alpha - e).astype(np.float32))
    while len(candidates) < n_candidates:
        noise = rng.normal(size=k).astype(np.float32)
        noise = noise / np.clip(np.linalg.norm(noise), 1e-12, None)
        scale = float(rng.uniform(0.1, radius))
        candidates.append((seed_alpha + scale * noise).astype(np.float32))
    best_alpha = candidates[0]
    best_margin, best_asr = eval_alpha(target_wrapper, records, best_alpha, eps)
    for cand in candidates[1:n_candidates]:
        m, asr = eval_alpha(target_wrapper, records, cand, eps)
        if (asr > best_asr) or (asr == best_asr and m < best_margin):
            best_alpha, best_margin, best_asr = cand, m, asr
    return best_alpha.astype(np.float32), int(n_candidates * len(records)), float(best_margin), float(best_asr)


def plot_summary(summary, out_dir):
    test = summary[summary["mode"] == "test"].copy()
    order = [
        "source_global_alpha",
        "target_global_random_start",
        "target_global_source_init",
        "source_class_alpha",
        "target_class_random_start",
        "target_class_source_init",
    ]
    test["variant"] = pd.Categorical(test["variant"], categories=order, ordered=True)
    targets = list(test.target_model.unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 3.3), sharey=True, constrained_layout=True)
    if len(targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        g = test[test.target_model == target].sort_values("variant")
        ax.bar(g.variant.astype(str), g.asr, color="#8b79c9")
        ax.set_title(target)
        ax.set_ylim(0, 1)
        ax.set_ylabel("held-out one-shot ASR")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "source_initialized_coefficients_asr.png", dpi=190, bbox_inches="tight")
    fig.savefig(out_dir / "source_initialized_coefficients_asr.pdf", bbox_inches="tight")
    plt.close(fig)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch

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
    train_items, _calib_by_class, _test_by_class, calib_items, test_items = select_splits(
        dataset,
        wrappers,
        args.source,
        args.train_images,
        args.calib_per_class,
        args.test_per_class,
        device,
    )
    print(
        f"[DATA] source={args.source} layer={source_layer} train={len(train_items)} calib={len(calib_items)} test={len(test_items)}",
        flush=True,
    )
    vectors, basis_image_rows = collect_success_transport(source_wrapper, dataset, train_items, source_layer, args, device)
    if len(vectors) < max(args.pcs + 2, 8):
        raise RuntimeError(f"Not enough successful source transport vectors: {len(vectors)}")
    norm_vectors = normalize_rows(vectors)
    _mean, basis, ratios = fit_pca(norm_vectors, args.pcs)
    eps = args.eps / 255.0

    # Source-side coefficient policies, estimated once on source calibration records.
    source_calib_records = build_image_bases(source_wrapper, source_wrapper, dataset, calib_items, source_layer, basis, device)
    source_global, q_src_global, m_src_global, asr_src_global = random_search_alpha(
        source_wrapper,
        source_calib_records,
        eps,
        args.pcs,
        args.source_global_query_budget,
        args.seed + 100,
    )
    source_class = {}
    q_src_class = 0
    alpha_rows = [
        {
            "target_model": args.source,
            "policy": "source_global_alpha",
            "label": -1,
            "queries": q_src_global,
            "calib_mean_margin": m_src_global,
            "calib_asr": asr_src_global,
            **{f"alpha_{i+1}": float(source_global[i]) for i in range(args.pcs)},
        }
    ]
    for label in range(10):
        recs = [r for r in source_calib_records if r.label == label]
        alpha, q, m, asr = random_search_alpha(
            source_wrapper,
            recs,
            eps,
            args.pcs,
            args.source_class_query_budget,
            args.seed + 200 + label,
        )
        source_class[label] = alpha
        q_src_class += q
        alpha_rows.append(
            {
                "target_model": args.source,
                "policy": "source_class_alpha",
                "label": label,
                "queries": q,
                "calib_mean_margin": m,
                "calib_asr": asr,
                **{f"alpha_{i+1}": float(alpha[i]) for i in range(args.pcs)},
            }
        )

    all_rows = []
    for target in targets:
        print(f"[TARGET] {target}", flush=True)
        target_wrapper = wrappers[target]
        calib_records = build_image_bases(source_wrapper, target_wrapper, dataset, calib_items, source_layer, basis, device)
        test_records = build_image_bases(source_wrapper, target_wrapper, dataset, test_items, source_layer, basis, device)

        random_global, q_rg, m_rg, asr_rg = random_search_alpha(
            target_wrapper,
            calib_records,
            eps,
            args.pcs,
            args.target_global_query_budget,
            args.seed + len(target),
        )
        seeded_global, q_sg, m_sg, asr_sg = seeded_search_alpha(
            target_wrapper,
            calib_records,
            eps,
            args.pcs,
            args.target_global_query_budget,
            source_global,
            args.seed + 500 + len(target),
            args.init_radius,
        )
        random_class = {}
        seeded_class = {}
        q_rc = 0
        q_sc = 0
        for label in range(10):
            recs = [r for r in calib_records if r.label == label]
            a_r, q, m, asr = random_search_alpha(
                target_wrapper,
                recs,
                eps,
                args.pcs,
                args.target_class_query_budget,
                args.seed + label * 101 + len(target),
            )
            random_class[label] = a_r
            q_rc += q
            alpha_rows.append(
                {
                    "target_model": target,
                    "policy": "target_class_random_start",
                    "label": label,
                    "queries": q,
                    "calib_mean_margin": m,
                    "calib_asr": asr,
                    **{f"alpha_{i+1}": float(a_r[i]) for i in range(args.pcs)},
                }
            )
            a_s, q, m, asr = seeded_search_alpha(
                target_wrapper,
                recs,
                eps,
                args.pcs,
                args.target_class_query_budget,
                source_class[label],
                args.seed + 700 + label * 101 + len(target),
                args.init_radius,
            )
            seeded_class[label] = a_s
            q_sc += q
            alpha_rows.append(
                {
                    "target_model": target,
                    "policy": "target_class_source_init",
                    "label": label,
                    "queries": q,
                    "calib_mean_margin": m,
                    "calib_asr": asr,
                    **{f"alpha_{i+1}": float(a_s[i]) for i in range(args.pcs)},
                }
            )
        alpha_rows.append(
            {
                "target_model": target,
                "policy": "target_global_random_start",
                "label": -1,
                "queries": q_rg,
                "calib_mean_margin": m_rg,
                "calib_asr": asr_rg,
                **{f"alpha_{i+1}": float(random_global[i]) for i in range(args.pcs)},
            }
        )
        alpha_rows.append(
            {
                "target_model": target,
                "policy": "target_global_source_init",
                "label": -1,
                "queries": q_sg,
                "calib_mean_margin": m_sg,
                "calib_asr": asr_sg,
                **{f"alpha_{i+1}": float(seeded_global[i]) for i in range(args.pcs)},
            }
        )

        policies = [
            ("source_global_alpha", {-1: source_global}),
            ("target_global_random_start", {-1: random_global}),
            ("target_global_source_init", {-1: seeded_global}),
            ("source_class_alpha", source_class),
            ("target_class_random_start", random_class),
            ("target_class_source_init", seeded_class),
        ]
        for mode, records in [("calib", calib_records), ("test", test_records)]:
            for policy_name, mapping in policies:
                global_alpha = mapping.get(-1, source_global)
                class_mapping = mapping if -1 not in mapping else {}
                for row in evaluate_policy(target_wrapper, records, class_mapping, global_alpha, eps, mode, args.seed + 17):
                    # evaluate_policy emits generic names; keep only the policy row we want.
                    if class_mapping and row["variant"] != "class_alpha":
                        continue
                    if not class_mapping and row["variant"] != "global_alpha":
                        continue
                    row["variant"] = policy_name
                    row.update(
                        {
                            "source_model": args.source,
                            "target_model": target,
                            "source_layer": source_layer,
                            "online_queries": 1,
                            "source_global_queries": q_src_global,
                            "source_class_queries": q_src_class,
                            "target_global_random_queries": q_rg,
                            "target_global_source_init_queries": q_sg,
                            "target_class_random_queries": q_rc,
                            "target_class_source_init_queries": q_sc,
                        }
                    )
                    all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(out_dir / "partial_source_initialized_coefficients_per_image.csv", index=False)
        print(f"  done target={target} rows={len(all_rows)}", flush=True)

    df = pd.DataFrame(all_rows)
    summary = summarize(df)
    df.to_csv(out_dir / "source_initialized_coefficients_per_image.csv", index=False)
    summary.to_csv(out_dir / "source_initialized_coefficients_summary.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(out_dir / "source_initialized_coefficients_alpha.csv", index=False)
    pd.DataFrame(basis_image_rows).assign(source_model=args.source, layer=source_layer).to_csv(out_dir / "basis_image_outcomes.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_model": args.source,
                "source_layer": source_layer,
                "pc": i + 1,
                "variance_explained": float(ratios[i]),
                "cumulative_variance": float(np.sum(ratios[: i + 1])),
                "n_success_transport_vectors": int(len(norm_vectors)),
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
    print(f"[SAVED] {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/section9_source_initialized_coefficients")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--source", default="bbb_resnet50")
    p.add_argument("--targets", default="bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--train-images", type=int, default=80)
    p.add_argument("--calib-per-class", type=int, default=3)
    p.add_argument("--test-per-class", type=int, default=10)
    p.add_argument("--basis-eps", type=float, default=2.0)
    p.add_argument("--basis-steps", type=int, default=5)
    p.add_argument("--basis-step-size", type=float, default=0.0)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--pcs", type=int, default=5)
    p.add_argument("--source-global-query-budget", type=int, default=2000)
    p.add_argument("--source-class-query-budget", type=int, default=600)
    p.add_argument("--target-global-query-budget", type=int, default=2000)
    p.add_argument("--target-class-query-budget", type=int, default=600)
    p.add_argument("--init-radius", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=47)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
