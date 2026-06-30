#!/usr/bin/env python3
"""Plot 2D maps of traced singular roads and attack trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.evaluate_road_damping_defense import project_linf  # noqa: E402
from experiments.pure_af_geometry.trace_jacobian_singular_roads import (  # noqa: E402
    estimate_top_direction,
    load_model,
    path_features,
    pgd_path,
    select_images,
    set_seed,
    square_path,
)


COLORS = {
    "road_forward": "#2563eb",
    "road_backward": "#0ea5a4",
    "pgd": "#dc2626",
    "square": "#f59e0b",
}


def trace_states(model, x0, args, image_id: int, sign: float):
    xs = []
    x = x0.detach().clone()
    prev_v = None
    for t in range(args.road_steps + 1):
        xs.append(x.detach())
        if t == args.road_steps:
            break
        v, _sigma = estimate_top_direction(model, x, args.power_iters, prev_v, args.seed + image_id * 7001 + t)
        if prev_v is not None and float((v.flatten(1) * prev_v.flatten(1)).sum().item()) < 0:
            v = -v
        step = sign * args.step_l2 * v
        if args.use_sign_step:
            step = sign * (args.step_linf / 255.0) * v.sign()
        x = project_linf(x + step, x0, args.eps / 255.0).detach()
        prev_v = v.detach()
    return xs


def select_images_class(dataset, model, n, device, class_filter: int | None):
    rows = []
    for idx in range(len(dataset)):
        x, y0 = dataset[idx]
        if class_filter is not None and int(y0) != int(class_filter):
            continue
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(model(x).argmax(1).item())
        if pred == int(y0):
            rows.append((idx, int(y0)))
        if len(rows) >= n:
            break
    return rows


def collect_paths(model, dataset, selected, args, device):
    rows = []
    features = []
    paths_by_image = {}
    for image_id, label in selected:
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        paths = {
            "road_forward": trace_states(model, x0, args, image_id, +1.0),
            "road_backward": trace_states(model, x0, args, image_id, -1.0),
            "pgd": pgd_path(model, x0, y, args),
            "square": square_path(model, x0, y, args, args.seed + image_id),
        }
        paths_by_image[image_id] = (label, paths)
        for name, xs in paths.items():
            h = path_features(model, xs)
            for step, vec in enumerate(h):
                rows.append({"image_id": image_id, "label": label, "path": name, "step": step})
                features.append(vec)
    return pd.DataFrame(rows), np.asarray(features, dtype=np.float32), paths_by_image


def draw_path(ax, pts, name, label=None, alpha=0.9):
    color = COLORS.get(name, "0.3")
    ax.plot(pts[:, 0], pts[:, 1], lw=1.6, color=color, alpha=alpha, label=label)
    ax.scatter(pts[0, 0], pts[0, 1], s=24, marker="o", color=color, edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(pts[-1, 0], pts[-1, 1], s=36, marker="x", color=color, zorder=3)


def plot_aggregate(coords: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    for name in ["road_forward", "road_backward", "pgd", "square"]:
        first = True
        for _image_id, g in coords[coords.path == name].sort_values("step").groupby("image_id"):
            draw_path(ax, g[["pc1", "pc2"]].to_numpy(), name, label=name if first else None, alpha=0.45)
            first = False
    ax.set_title("Singular-road traces and attack trajectories")
    ax.set_xlabel("PC1 of hidden states")
    ax.set_ylabel("PC2 of hidden states")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out / "singular_road_map_all.png", dpi=240)
    plt.close(fig)


def plot_grid(paths_by_image, model, out: Path, max_panels: int):
    items = list(paths_by_image.items())[:max_panels]
    n = len(items)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.9 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, (image_id, (label, paths)) in zip(axes, items):
        feats = []
        labels = []
        for name, xs in paths.items():
            h = path_features(model, xs)
            feats.append(h)
            labels.extend([name] * len(h))
        z = PCA(n_components=2, random_state=0).fit_transform(np.concatenate(feats, axis=0))
        cursor = 0
        for name, h in zip(paths.keys(), feats):
            pts = z[cursor : cursor + len(h)]
            cursor += len(h)
            draw_path(ax, pts, name, label=name)
        ax.set_title(f"image {image_id}, class {label}")
        ax.set_xlabel("local PC1")
        ax.set_ylabel("local PC2")
        ax.legend(frameon=False, fontsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.savefig(out / "singular_road_map_grid.png", dpi=240)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/singular_road_tracing_resnet50_quick/maps")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--images", type=int, default=6)
    p.add_argument("--class-filter", type=int, default=-1)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--road-steps", type=int, default=24)
    p.add_argument("--step-l2", type=float, default=0.18)
    p.add_argument("--use-sign-step", action="store_true")
    p.add_argument("--step-linf", type=float, default=1.0)
    p.add_argument("--power-iters", type=int, default=8)
    p.add_argument("--attack-steps", type=int, default=20)
    p.add_argument("--attack-step", type=float, default=2.0)
    p.add_argument("--square-queries", type=int, default=100)
    p.add_argument("--seed", type=int, default=15)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    class_filter = None if args.class_filter < 0 else int(args.class_filter)
    selected = select_images_class(dataset, model, args.images, device, class_filter)
    if len(selected) < args.images:
        print(f"[WARN] requested {args.images}, found {len(selected)} clean-correct for class_filter={class_filter}", flush=True)
    meta, feats, paths_by_image = collect_paths(model, dataset, selected, args, device)
    z = PCA(n_components=2, random_state=0).fit_transform(feats)
    coords = meta.copy()
    coords["pc1"] = z[:, 0]
    coords["pc2"] = z[:, 1]
    coords.to_csv(out / "singular_road_2d_coordinates.csv", index=False)
    plot_aggregate(coords, out)
    plot_grid(paths_by_image, model, out, args.images)
    (out / "metadata.json").write_text(json.dumps(vars(args) | {"device": str(device), "selected": selected}, indent=2))
    print(f"Wrote maps to {out}", flush=True)
    print(f"- {out / 'singular_road_map_all.png'}", flush=True)
    print(f"- {out / 'singular_road_map_grid.png'}", flush=True)


if __name__ == "__main__":
    main()
