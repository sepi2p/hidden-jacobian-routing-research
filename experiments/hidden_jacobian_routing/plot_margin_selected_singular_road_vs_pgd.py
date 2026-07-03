#!/usr/bin/env python3
"""Trace a margin-selected high-mobility road and compare it with PGD.

For one CIFAR-10 image, each step estimates the top local input-space singular
direction of the hidden feature Jacobian. Since the singular direction is an
axis, both signs are evaluated; the next state is the sign that gives the lower
true-class margin. The resulting path is plotted against PGD in the same hidden
feature PCA plane.
"""

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
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import margin, project_linf  # noqa: E402
from experiments.hidden_jacobian_routing.trace_jacobian_singular_roads import (  # noqa: E402
    estimate_top_direction,
    feat,
    load_model,
    set_seed,
)


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def logits_row(model, x, y):
    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(1).item())
        m = float(margin(logits, y).item())
        ce = float(F.cross_entropy(logits, y).item())
        prob = float(F.softmax(logits, dim=1)[0, pred].item())
    return pred, m, ce, prob


def trace_margin_selected_road(model, x0, y, args):
    rows = []
    xs = [x0.detach().clone()]
    x = x0.detach().clone()
    prev_v = None

    for t in range(args.road_steps + 1):
        pred, m, ce, prob = logits_row(model, x, y)
        rows.append(
            {
                "path": "margin_selected_road",
                "step": t,
                "chosen_sign": 0,
                "pred": pred,
                "pred_class": CIFAR10_CLASSES[pred],
                "margin": m,
                "ce": ce,
                "pred_prob": prob,
                "forward_margin": np.nan,
                "backward_margin": np.nan,
                "sigma1_est": np.nan,
                "linf_from_start": float((x - x0).abs().max().item()),
            }
        )
        if t == args.road_steps:
            break

        v, sigma = estimate_top_direction(model, x, args.power_iters, prev_v, args.seed + t)
        if prev_v is not None and float((v.flatten(1) * prev_v.flatten(1)).sum().item()) < 0:
            v = -v

        step = (args.step_linf / 255.0) * v.sign()
        x_forward = project_linf(x + step, x0, args.eps / 255.0).detach()
        x_backward = project_linf(x - step, x0, args.eps / 255.0).detach()
        _pf, mf, _cef, _probf = logits_row(model, x_forward, y)
        _pb, mb, _ceb, _probb = logits_row(model, x_backward, y)

        if mf <= mb:
            x = x_forward
            chosen = +1
        else:
            x = x_backward
            chosen = -1
            v = -v

        # Fill candidate margins into the current transition row.
        rows[-1]["chosen_sign"] = chosen
        rows[-1]["forward_margin"] = mf
        rows[-1]["backward_margin"] = mb
        rows[-1]["sigma1_est"] = sigma
        xs.append(x.detach())
        prev_v = v.detach()

    return xs, rows


def select_image(dataset, model, device, image_index: int | None, class_filter: int | None):
    if image_index is not None:
        x, y = dataset[image_index]
        x = x.unsqueeze(0).to(device)
        yy = torch.tensor([int(y)], device=device)
        pred, *_ = logits_row(model, x, yy)
        if pred != int(y):
            raise RuntimeError(f"Requested image {image_index} is not clean-correct: y={y}, pred={pred}")
        return image_index, int(y), x, yy

    for idx in range(len(dataset)):
        x, y = dataset[idx]
        if class_filter is not None and int(y) != int(class_filter):
            continue
        x = x.unsqueeze(0).to(device)
        yy = torch.tensor([int(y)], device=device)
        pred, *_ = logits_row(model, x, yy)
        if pred == int(y):
            return idx, int(y), x, yy
    raise RuntimeError("Could not find a clean-correct image matching the requested filters")


def path_features(model, xs):
    with torch.no_grad():
        return np.concatenate([feat(model, x).detach().cpu().numpy() for x in xs], axis=0)


def pgd_path_local(model, x0, y, args):
    xs = [x0.detach()]
    if args.pgd_random_start:
        x = project_linf(
            x0 + torch.empty_like(x0).uniform_(-args.eps / 255.0, args.eps / 255.0),
            x0,
            args.eps / 255.0,
        )
    else:
        x = x0.detach().clone()
    if args.pgd_random_start:
        xs.append(x.detach())
    for _ in range(args.attack_steps):
        x.requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        grad = torch.autograd.grad(loss, x)[0]
        x = project_linf(x.detach() + (args.attack_step / 255.0) * grad.detach().sign(), x0, args.eps / 255.0)
        xs.append(x.detach())
    return xs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/margin_selected_singular_road_vs_pgd")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--image-index", type=int, default=0)
    p.add_argument("--class-filter", type=int, default=-1)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--road-steps", type=int, default=8)
    p.add_argument("--step-linf", type=float, default=1.0)
    p.add_argument("--power-iters", type=int, default=5)
    p.add_argument("--attack-steps", type=int, default=8)
    p.add_argument("--attack-step", type=float, default=2.0)
    p.add_argument("--pgd-random-start", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    class_filter = None if args.class_filter < 0 else int(args.class_filter)
    image_index = None if args.image_index < 0 else int(args.image_index)
    idx, label, x0, y = select_image(dataset, model, device, image_index, class_filter)

    road_xs, road_rows = trace_margin_selected_road(model, x0, y, args)
    pgd_xs = pgd_path_local(model, x0, y, args)
    pgd_rows = []
    for t, x in enumerate(pgd_xs):
        pred, m, ce, prob = logits_row(model, x, y)
        pgd_rows.append(
            {
                "path": "pgd",
                "step": t,
                "chosen_sign": np.nan,
                "pred": pred,
                "pred_class": CIFAR10_CLASSES[pred],
                "margin": m,
                "ce": ce,
                "pred_prob": prob,
                "forward_margin": np.nan,
                "backward_margin": np.nan,
                "sigma1_est": np.nan,
                "linf_from_start": float((x - x0).abs().max().item()),
            }
        )

    road_h = path_features(model, road_xs)
    pgd_h = path_features(model, pgd_xs)
    z = PCA(n_components=2, random_state=args.seed).fit_transform(np.concatenate([road_h, pgd_h], axis=0))
    road_xy = z[: len(road_h)]
    pgd_xy = z[len(road_h) :]

    all_rows = pd.DataFrame(road_rows + pgd_rows)
    xy_rows = []
    for name, pts in [("margin_selected_road", road_xy), ("pgd", pgd_xy)]:
        for t, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            xy_rows.append(
                {
                    "path": name,
                    "step": t,
                    "pc1": float(a[0]),
                    "pc2": float(a[1]),
                    "pc1_next": float(b[0]),
                    "pc2_next": float(b[1]),
                    "pc1_delta": float(b[0] - a[0]),
                    "pc2_delta": float(b[1] - a[1]),
                }
            )
    all_rows.to_csv(out / "margin_selected_road_vs_pgd_steps.csv", index=False)
    pd.DataFrame(xy_rows).to_csv(out / "margin_selected_road_vs_pgd_2d_vectors.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
    styles = {
        "margin_selected_road": ("#2563eb", "margin-selected high-mobility road"),
        "pgd": ("#dc2626", "PGD"),
    }
    for name, pts in [("margin_selected_road", road_xy), ("pgd", pgd_xy)]:
        color, label_name = styles[name]
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=color, lw=2.0, ms=3.8, label=label_name)
        for a, b in zip(pts[:-1], pts[1:]):
            ax.annotate(
                "",
                xy=b,
                xytext=a,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.25, alpha=0.65, shrinkA=2, shrinkB=2),
            )
        ax.scatter(pts[0, 0], pts[0, 1], s=70, marker="o", color=color, edgecolor="white", linewidth=0.7, zorder=4)
        ax.scatter(pts[-1, 0], pts[-1, 1], s=80, marker="x", color=color, zorder=4)
    ax.set_title(f"Margin-selected high-mobility road vs PGD, image {idx} ({CIFAR10_CLASSES[label]})")
    ax.set_xlabel("PC1 of hidden states")
    ax.set_ylabel("PC2 of hidden states")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out / "margin_selected_road_vs_pgd.png", dpi=240)
    fig.savefig(out / "margin_selected_road_vs_pgd.pdf")
    plt.close(fig)

    metadata = {
        "image_index": idx,
        "label": label,
        "label_class": CIFAR10_CLASSES[label],
        "device": str(device),
        "settings": vars(args),
        "outputs": {
            "plot_png": str(out / "margin_selected_road_vs_pgd.png"),
            "plot_pdf": str(out / "margin_selected_road_vs_pgd.pdf"),
            "steps_csv": str(out / "margin_selected_road_vs_pgd_steps.csv"),
            "vectors_csv": str(out / "margin_selected_road_vs_pgd_2d_vectors.csv"),
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
