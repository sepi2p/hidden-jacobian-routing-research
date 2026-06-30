#!/usr/bin/env python3
"""Universal coefficient policies in adversarial success-flow coordinates.

We spend black-box target queries on a calibration set to estimate either one
global coefficient vector or one class-specific coefficient vector in a
surrogate success-flow PC basis. The learned coefficients are then reused
one-shot on held-out images.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
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
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import margin  # noqa: E402
from experiments.pure_af_geometry.run_section9_adv_success_flow_blackbox import (  # noqa: E402
    BEST_LAYER,
    collect_success_transport,
    eval_target,
    feature_pc_grad,
    fit_pca,
    project_linf,
)


@dataclass
class ImageBasis:
    dataset_idx: int
    label: int
    x: torch.Tensor
    y: torch.Tensor
    pc_basis: list[torch.Tensor]
    clean: dict


def select_splits(dataset, wrappers, source: str, train_images: int, calib_per_class: int, test_per_class: int, device):
    train = []
    calib = {c: [] for c in range(10)}
    test = {c: [] for c in range(10)}
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for idx, (x, y) in enumerate(loader):
        label = int(y.item())
        x = x.to(device)
        ok = True
        with torch.no_grad():
            for wrapper in wrappers.values():
                if int(wrapper(x).argmax(1).item()) != label:
                    ok = False
                    break
        if not ok:
            continue
        if len(train) < train_images:
            train.append((idx, label))
        elif len(calib[label]) < calib_per_class:
            calib[label].append((idx, label))
        elif len(test[label]) < test_per_class:
            test[label].append((idx, label))
        if len(train) >= train_images and all(len(calib[c]) >= calib_per_class for c in range(10)) and all(len(test[c]) >= test_per_class for c in range(10)):
            break
    calib_items = [item for c in range(10) for item in calib[c]]
    test_items = [item for c in range(10) for item in test[c]]
    missing = {
        "calib": {c: len(calib[c]) for c in range(10) if len(calib[c]) < calib_per_class},
        "test": {c: len(test[c]) for c in range(10) if len(test[c]) < test_per_class},
    }
    if missing["calib"] or missing["test"]:
        print(f"[WARN] incomplete class-balanced split: {missing}", flush=True)
    return train, calib, test, calib_items, test_items


def normalize_pixel_basis(vecs):
    out = []
    for v in vecs:
        n = torch.linalg.vector_norm(v.flatten(1), dim=1).view(-1, 1, 1, 1)
        out.append(v / torch.clamp(n, min=1e-12))
    return out


def build_image_bases(source_wrapper, target_wrapper, dataset, items, layer, basis, device):
    records = []
    for idx, label in items:
        x_cpu, _ = dataset[idx]
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        pc_basis = [feature_pc_grad(source_wrapper, x, layer, basis[i]) for i in range(basis.shape[0])]
        records.append(
            ImageBasis(
                dataset_idx=int(idx),
                label=int(label),
                x=x.detach(),
                y=y,
                pc_basis=normalize_pixel_basis(pc_basis),
                clean=eval_target(target_wrapper, x, y),
            )
        )
    return records


def apply_alpha(record: ImageBasis, alpha: np.ndarray, eps: float):
    direction = torch.zeros_like(record.x)
    for a, u in zip(alpha, record.pc_basis):
        direction = direction + float(a) * u
    return project_linf(record.x + eps * direction.sign(), record.x, eps)


def eval_alpha(target_wrapper, records: list[ImageBasis], alpha: np.ndarray, eps: float):
    margins = []
    successes = []
    for rec in records:
        ev = eval_target(target_wrapper, apply_alpha(rec, alpha, eps), rec.y)
        margins.append(ev["margin"])
        successes.append(ev["success"])
    return float(np.mean(margins)), float(np.mean(successes))


def random_search_alpha(target_wrapper, records, eps, k, query_budget, seed):
    rng = np.random.default_rng(seed)
    if not records:
        return np.zeros(k, dtype=np.float32), 0, np.nan, np.nan
    n_candidates = max(1, int(query_budget // max(len(records), 1)))
    candidates = [np.zeros(k, dtype=np.float32)]
    for i in range(k):
        e = np.zeros(k, dtype=np.float32)
        e[i] = 1.0
        candidates.append(e.copy())
        candidates.append(-e.copy())
    while len(candidates) < n_candidates:
        a = rng.normal(size=k).astype(np.float32)
        a = a / np.clip(np.linalg.norm(a), 1e-12, None)
        scale = float(rng.uniform(0.5, 2.0))
        candidates.append((scale * a).astype(np.float32))
    best_alpha = candidates[0]
    best_margin, best_asr = eval_alpha(target_wrapper, records, best_alpha, eps)
    for cand in candidates[1:n_candidates]:
        m, asr = eval_alpha(target_wrapper, records, cand, eps)
        if (asr > best_asr) or (asr == best_asr and m < best_margin):
            best_alpha, best_margin, best_asr = cand, m, asr
    return best_alpha.astype(np.float32), int(n_candidates * len(records)), float(best_margin), float(best_asr)


def evaluate_policy(target_wrapper, records, alpha_by_label, alpha_global, eps, mode, random_seed):
    rng = np.random.default_rng(random_seed)
    rows = []
    k = len(alpha_global)
    random_global = rng.normal(size=k).astype(np.float32)
    random_global /= np.clip(np.linalg.norm(random_global), 1e-12, None)
    random_class = {}
    for label in range(10):
        a = rng.normal(size=k).astype(np.float32)
        random_class[label] = a / np.clip(np.linalg.norm(a), 1e-12, None)
    for rec in records:
        policies = {
            "global_alpha": alpha_global,
            "class_alpha": alpha_by_label.get(rec.label, alpha_global),
            "random_global_alpha": random_global,
            "random_class_alpha": random_class[rec.label],
        }
        for variant, alpha in policies.items():
            adv = apply_alpha(rec, alpha, eps)
            ev = eval_target(target_wrapper, adv, rec.y)
            rows.append(
                {
                    "mode": mode,
                    "dataset_idx": rec.dataset_idx,
                    "label": rec.label,
                    "variant": variant,
                    "target_success": int(ev["success"]),
                    "target_pred": int(ev["pred"]),
                    "target_margin": float(ev["margin"]),
                    "target_true_prob": float(ev["true_prob"]),
                    "clean_margin": float(rec.clean["margin"]),
                    "clean_true_prob": float(rec.clean["true_prob"]),
                    "margin_drop": float(rec.clean["margin"] - ev["margin"]),
                    "true_prob_drop": float(rec.clean["true_prob"] - ev["true_prob"]),
                }
            )
    return rows


def summarize(df):
    return df.groupby(["target_model", "mode", "variant"], dropna=False).agg(
        asr=("target_success", "mean"),
        n=("target_success", "size"),
        mean_margin_drop=("margin_drop", "mean"),
        mean_prob_drop=("true_prob_drop", "mean"),
        mean_margin=("target_margin", "mean"),
    ).reset_index()


def plot_summary(summary, out_dir):
    test = summary[summary["mode"] == "test"].copy()
    order = ["random_global_alpha", "global_alpha", "random_class_alpha", "class_alpha"]
    test["variant"] = pd.Categorical(test["variant"], categories=order, ordered=True)
    targets = list(test.target_model.unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(4.3 * len(targets), 3.2), sharey=True, constrained_layout=True)
    if len(targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        g = test[test.target_model == target].sort_values("variant")
        ax.bar(g.variant.astype(str), g.asr, color="#7aa974")
        ax.set_title(target)
        ax.set_ylim(0, 1)
        ax.set_ylabel("held-out one-shot ASR")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "universal_success_flow_coefficients_asr.png", dpi=190, bbox_inches="tight")
    fig.savefig(out_dir / "universal_success_flow_coefficients_asr.pdf", bbox_inches="tight")
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
    train_items, calib_by_class, test_by_class, calib_items, test_items = select_splits(
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
    all_rows = []
    alpha_rows = []
    for target in targets:
        target_wrapper = wrappers[target]
        print(f"[TARGET] {target}", flush=True)
        calib_records = build_image_bases(source_wrapper, target_wrapper, dataset, calib_items, source_layer, basis, device)
        test_records = build_image_bases(source_wrapper, target_wrapper, dataset, test_items, source_layer, basis, device)
        alpha_global, q_global, m_global, asr_global = random_search_alpha(
            target_wrapper,
            calib_records,
            eps,
            args.pcs,
            args.global_query_budget,
            args.seed + len(target),
        )
        alpha_by_label = {}
        class_queries = 0
        for label in range(10):
            class_records = [r for r in calib_records if r.label == label]
            alpha, q, m, asr = random_search_alpha(
                target_wrapper,
                class_records,
                eps,
                args.pcs,
                args.class_query_budget,
                args.seed + label * 101 + len(target),
            )
            alpha_by_label[label] = alpha
            class_queries += q
            alpha_rows.append(
                {
                    "target_model": target,
                    "policy": "class_alpha",
                    "label": label,
                    "queries": q,
                    "calib_mean_margin": m,
                    "calib_asr": asr,
                    **{f"alpha_{i+1}": float(alpha[i]) for i in range(args.pcs)},
                }
            )
        alpha_rows.append(
            {
                "target_model": target,
                "policy": "global_alpha",
                "label": -1,
                "queries": q_global,
                "calib_mean_margin": m_global,
                "calib_asr": asr_global,
                **{f"alpha_{i+1}": float(alpha_global[i]) for i in range(args.pcs)},
            }
        )
        for row in evaluate_policy(target_wrapper, calib_records, alpha_by_label, alpha_global, eps, "calib", args.seed):
            row.update(
                {
                    "source_model": args.source,
                    "target_model": target,
                    "source_layer": source_layer,
                    "online_queries": 1,
                    "global_calibration_queries": q_global,
                    "class_calibration_queries": class_queries,
                    "amortized_global_queries_per_test_image": q_global / max(len(test_records), 1),
                    "amortized_class_queries_per_test_image": class_queries / max(len(test_records), 1),
                }
            )
            all_rows.append(row)
        for row in evaluate_policy(target_wrapper, test_records, alpha_by_label, alpha_global, eps, "test", args.seed + 17):
            row.update(
                {
                    "source_model": args.source,
                    "target_model": target,
                    "source_layer": source_layer,
                    "online_queries": 1,
                    "global_calibration_queries": q_global,
                    "class_calibration_queries": class_queries,
                    "amortized_global_queries_per_test_image": q_global / max(len(test_records), 1),
                    "amortized_class_queries_per_test_image": class_queries / max(len(test_records), 1),
                }
            )
            all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(out_dir / "partial_universal_success_flow_coefficients_per_image.csv", index=False)
        print(f"  done target={target} rows={len(all_rows)}", flush=True)

    df = pd.DataFrame(all_rows)
    summary = summarize(df)
    df.to_csv(out_dir / "universal_success_flow_coefficients_per_image.csv", index=False)
    summary.to_csv(out_dir / "universal_success_flow_coefficients_summary.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(out_dir / "universal_success_flow_coefficients_alpha.csv", index=False)
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
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/section9_universal_success_flow_coefficients")
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
    p.add_argument("--global-query-budget", type=int, default=2000)
    p.add_argument("--class-query-budget", type=int, default=600)
    p.add_argument("--seed", type=int, default=41)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
