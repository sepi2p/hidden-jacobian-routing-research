#!/usr/bin/env python3
"""Project one-class pure and adversarial trajectories into class-specific pure-flow PCs."""

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

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import get_npz, normalize_rows  # noqa: E402
from experiments.pure_af_geometry.analyze_cifar_attack_axis_projection import (  # noqa: E402
    LAYER_GROUPS,
    pgd_trajectory,
    project_trajectory,
    square_trajectory,
)
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import (  # noqa: E402
    eval_all,
    load_model,
    margin,
)


def fit_pure_segment_pcs(layerwise_dir: Path, model: str, target_class: int, layer_groups: list[str], top_k: int):
    seg = pd.read_csv(layerwise_dir / "segments.csv")
    vec_npz = np.load(layerwise_dir / "segment_vectors.npz")
    pc_dirs = {}
    pc_rows = []
    for layer_group in layer_groups:
        layer = LAYER_GROUPS[layer_group][model]
        sub = seg[
            (seg.model == model)
            & (seg.layer == layer)
            & (seg.target_class == target_class)
            & (seg.success == 1)
        ].copy()
        if sub.empty:
            continue
        arr = get_npz(vec_npz, "vectors", model, layer)
        X = arr[sub.vector_idx.astype(int).to_numpy()]
        X = normalize_rows(X.astype(np.float32))
        X = X - X.mean(axis=0, keepdims=True)
        _u, s, vt = np.linalg.svd(X, full_matrices=False)
        ratios = (s * s) / max(float(np.sum(s * s)), 1e-12)
        for i in range(min(top_k, vt.shape[0])):
            v = vt[i].astype(np.float32)
            v = v / max(float(np.linalg.norm(v)), 1e-12)
            pc_dirs[(model, layer, i + 1)] = v
            pc_rows.append(
                {
                    "model": model,
                    "layer_group": layer_group,
                    "layer": layer,
                    "target_class": target_class,
                    "pc": i + 1,
                    "variance_explained": float(ratios[i]),
                    "cumulative_variance": float(np.sum(ratios[: i + 1])),
                    "n_segments": int(len(X)),
                    "d": int(X.shape[1]),
                }
            )
    return pc_dirs, pd.DataFrame(pc_rows)


def load_pure_timeseries(layerwise_dir: Path, pc_dirs: dict, model: str, target_class: int, layer_groups: list[str], top_k: int):
    seg = pd.read_csv(layerwise_dir / "segments.csv")
    points = pd.read_csv(layerwise_dir / "points.csv")
    runs = pd.read_csv(layerwise_dir / "runs.csv")
    vec_npz = np.load(layerwise_dir / "segment_vectors.npz")
    point_lookup = points.set_index(["model", "run_id", "layer", "generation"])
    run_success = runs.set_index(["model", "run_id"])["success"].to_dict()
    rows = []
    for layer_group in layer_groups:
        layer = LAYER_GROUPS[layer_group][model]
        key = (model, layer)
        if not any((model, layer, pc) in pc_dirs for pc in range(1, top_k + 1)):
            continue
        basis = np.stack([pc_dirs[(model, layer, pc)] for pc in range(1, top_k + 1)]).astype(np.float32)
        arr = get_npz(vec_npz, "vectors", model, layer)
        sub = seg[
            (seg.model == model)
            & (seg.layer == layer)
            & (seg.target_class == target_class)
        ].sort_values(["run_id", "start_generation"])
        for run_id, g in sub.groupby("run_id", sort=False):
            cumulative = np.zeros(basis.shape[1], dtype=np.float32)
            n = max(len(g), 1)
            final_success = int(run_success.get((model, run_id), int(g.success.max())))
            for step_i, r in enumerate(g.itertuples(), start=1):
                cumulative += arr[int(r.vector_idx)]
                coeff = basis @ cumulative
                try:
                    p = point_lookup.loc[(model, run_id, layer, int(r.end_generation))]
                    if isinstance(p, pd.DataFrame):
                        p = p.iloc[0]
                    pred = int(p.pred)
                    m = float(p.margin)
                    prob = float(p.prob)
                    step_success = int(pred == target_class)
                except KeyError:
                    pred, m, prob, step_success = -1, np.nan, np.nan, 0
                row = {
                    "model": model,
                    "attack": "ga",
                    "run_id": str(run_id),
                    "dataset_idx": np.nan,
                    "image_ord": np.nan,
                    "label": target_class,
                    "target_class": target_class,
                    "layer_group": layer_group,
                    "layer": layer,
                    "step": int(r.end_generation),
                    "normalized_progress": float(step_i / n),
                    "time_bin": min(4, int(np.floor((step_i / n) * 5.0))),
                    "final_success": final_success,
                    "step_success": step_success,
                    "pred": pred,
                    "margin": m,
                    "true_prob": prob,
                }
                for i, c in enumerate(coeff, start=1):
                    row[f"pc{i}_coeff"] = float(c)
                    row[f"pc{i}_abs_coeff"] = float(abs(c))
                    row[f"pc{i}_energy"] = float(c * c)
                row["transport_energy_top5"] = float(sum(row.get(f"pc{i}_energy", 0.0) for i in range(1, top_k + 1)))
                rows.append(row)
    return pd.DataFrame(rows)


def select_clean_correct_class(dataset, wrapper, model: str, target_class: int, n: int, device):
    selected = []
    for idx in range(len(dataset)):
        x_cpu, y0 = dataset[idx]
        if int(y0) != target_class:
            continue
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([target_class], device=device)
        with torch.no_grad():
            logits = wrapper(x)
        if int(logits.argmax(1).item()) == target_class:
            selected.append((idx, target_class))
            if len(selected) >= n:
                break
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layerwise-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument("--images", type=int, default=80)
    p.add_argument("--attacks", default="pgd,square")
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--pgd-steps", type=int, default=20)
    p.add_argument("--square-steps", type=int, default=120)
    p.add_argument("--square-min-size", type=int, default=2)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--layer-groups", default="hidden,penultimate,logits")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_groups = [x.strip() for x in args.layer_groups.split(",") if x.strip()]

    pc_dirs, pc_meta = fit_pure_segment_pcs(Path(args.layerwise_dir), args.model, args.target_class, layer_groups, args.top_k)
    pc_meta.to_csv(out_dir / "oneclass_pure_pc_metadata.csv", index=False)
    pure_df = load_pure_timeseries(Path(args.layerwise_dir), pc_dirs, args.model, args.target_class, layer_groups, args.top_k)

    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    wrapper = load_model(args.model, device)
    selected = select_clean_correct_class(dataset, wrapper, args.model, args.target_class, args.images, device)
    print(f"[SELECTED] class={args.target_class} clean_correct={len(selected)}", flush=True)
    eps = args.eps / 255.0
    step_size = eps / max(args.pgd_steps, 1)
    attacks = [x.strip() for x in args.attacks.split(",") if x.strip()]
    adv_rows = []
    for image_ord, (dataset_idx, label) in enumerate(selected):
        x_cpu, _ = dataset[dataset_idx]
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        for attack in attacks:
            if attack == "pgd":
                states = pgd_trajectory(wrapper, x, y, eps, args.pgd_steps, step_size)
            elif attack == "square":
                states = square_trajectory(wrapper, x, y, eps, args.square_steps, args.seed + image_ord * 997, args.square_min_size)
            else:
                raise ValueError(attack)
            final_eval = eval_all({args.model: wrapper}, states[-1], y)[args.model]
            start_len = len(adv_rows)
            adv_rows.extend(
                project_trajectory(
                    pc_dirs=pc_dirs,
                    wrapper=wrapper,
                    source_model=args.model,
                    attack=attack,
                    dataset_idx=int(dataset_idx),
                    image_ord=int(image_ord),
                    label=int(label),
                    final_success=int(final_eval["success"]),
                    states=states,
                    y=y,
                    layer_groups=layer_groups,
                    top_k=args.top_k,
                )
            )
            for row in adv_rows[start_len:]:
                row["run_id"] = f"{attack}_img{image_ord}"
        if (image_ord + 1) % 10 == 0:
            print(f"  projected {image_ord + 1}/{len(selected)} images rows={len(adv_rows)}", flush=True)

    adv_df = pd.DataFrame(adv_rows)
    df = pd.concat([adv_df, pure_df], ignore_index=True, sort=False)
    df.to_csv(out_dir / "oneclass_pure_adv_projection_timeseries.csv", index=False)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"args": vars(args), "n_rows": int(len(df)), "n_selected": len(selected)}, f, indent=2)
    print(f"[SAVED] {out_dir}", flush=True)
    if not adv_df.empty:
        print(adv_df.groupby(["attack", "layer_group"]).final_success.mean().reset_index().to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
