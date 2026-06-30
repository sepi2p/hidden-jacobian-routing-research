#!/usr/bin/env python3
"""Trace high-mobility roads as integral curves of top hidden-Jacobian directions.

This script tests whether the local hidden-Jacobian anisotropy forms coherent
curves rather than isolated local patches.  For each clean image x0, it estimates
the top input-space right singular direction of J_h(x), enforces sign continuity,
and integrates

    x_{t+1} = Proj_{Linf}(x_t + eta v_1(x_t)).

It records road strength, width, margin, prediction changes, curvature, return
distance, and simple comparisons with PGD/Square trajectories from the same x0.
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
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.evaluate_road_damping_defense import DampedResNet50, margin, project_linf  # noqa: E402
from utils.load_models import load_cifar_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(device: torch.device) -> DampedResNet50:
    seq = load_cifar_model("bbb_resnet50").to(device).eval()
    return DampedResNet50(seq, None).to(device).eval()


def feat(model: DampedResNet50, x: torch.Tensor) -> torch.Tensor:
    return model.pooled_layer4(x).flatten(1)


def logits_stats(model: DampedResNet50, x: torch.Tensor, y: torch.Tensor):
    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(1).item())
        m = float(margin(logits, y).item())
        conf = float(F.softmax(logits, dim=1).max(1).values.item())
    return pred, m, conf


def normalize_l2(v: torch.Tensor) -> torch.Tensor:
    return v / v.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)


def estimate_top_direction(
    model: DampedResNet50,
    x: torch.Tensor,
    n_iter: int,
    init_v: torch.Tensor | None,
    seed: int,
) -> tuple[torch.Tensor, float]:
    if init_v is None:
        g = torch.Generator(device=x.device).manual_seed(seed)
        v = torch.randn(x.shape, generator=g, device=x.device)
        v = normalize_l2(v)
    else:
        v = normalize_l2(init_v.detach())

    def f(inp):
        return feat(model, inp)

    for _ in range(n_iter):
        x_req = x.detach().requires_grad_(True)
        _h, jv = torch.autograd.functional.jvp(f, x_req, v, create_graph=False, strict=False)
        h = f(x_req)
        dot = (h * jv.detach()).sum()
        w = torch.autograd.grad(dot, x_req)[0]
        v = normalize_l2(w.detach())
    with torch.no_grad():
        _h, jv = torch.autograd.functional.jvp(f, x.detach(), v, create_graph=False, strict=False)
        sigma = float(jv.flatten(1).norm(dim=1).item())
    return v.detach(), sigma


def jvp_strengths_randomized(model: DampedResNet50, x: torch.Tensor, n_dirs: int, seed: int) -> np.ndarray:
    vals = []
    g = torch.Generator(device=x.device).manual_seed(seed)

    def f(inp):
        return feat(model, inp)

    for _ in range(n_dirs):
        v = normalize_l2(torch.randn(x.shape, generator=g, device=x.device))
        with torch.no_grad():
            _h, jv = torch.autograd.functional.jvp(f, x.detach(), v, create_graph=False, strict=False)
            vals.append(float(jv.flatten(1).norm(dim=1).item()))
    return np.asarray(vals, dtype=np.float32)


def width_from_strengths(vals: np.ndarray, alpha: float) -> tuple[int, float]:
    if len(vals) == 0:
        return 0, np.nan
    s1 = float(vals.max())
    width = int((vals >= alpha * s1).sum())
    p = vals**2 / max(float((vals**2).sum()), 1e-12)
    erank = float(np.exp(-(p * np.log(np.clip(p, 1e-12, None))).sum()))
    return width, erank


def trace_one(model, x0, y, args, image_id: int, sign: float, device):
    rows = []
    x = x0.detach().clone()
    prev_v = None
    h0 = feat(model, x0).detach()
    pred0, m0, conf0 = logits_stats(model, x0, y)
    for t in range(args.road_steps + 1):
        v, sigma = estimate_top_direction(model, x, args.power_iters, prev_v, args.seed + image_id * 1009 + t)
        if prev_v is not None:
            cos_prev = float((v.flatten(1) * prev_v.flatten(1)).sum().item())
            if cos_prev < 0:
                v = -v
                cos_prev = -cos_prev
        else:
            cos_prev = np.nan
        vals = jvp_strengths_randomized(model, x, args.width_dirs, args.seed + image_id * 2003 + t)
        width, erank = width_from_strengths(np.r_[vals, sigma], args.width_alpha)
        h = feat(model, x).detach()
        pred, m, conf = logits_stats(model, x, y)
        linf = float((x - x0).abs().max().item())
        l2 = float((x - x0).flatten(1).norm(dim=1).item())
        hdist = float((h - h0).norm(dim=1).item())
        rows.append(
            {
                "image_id": image_id,
                "direction": "forward" if sign > 0 else "backward",
                "step": t,
                "pred0": pred0,
                "pred": pred,
                "label": int(y.item()),
                "changed_class": int(pred != pred0),
                "success": int(pred != int(y.item())),
                "margin0": m0,
                "margin": m,
                "confidence": conf,
                "sigma1_est": sigma,
                "width_count": width,
                "width_erank": erank,
                "cos_prev": cos_prev,
                "linf_from_start": linf,
                "l2_from_start": l2,
                "hidden_dist_from_start": hdist,
                "return_linf": linf,
                "return_hidden": hdist,
            }
        )
        if t == args.road_steps:
            break
        step = sign * args.step_l2 * v
        if args.use_sign_step:
            step = sign * (args.step_linf / 255.0) * v.sign()
        x = project_linf(x + step, x0, args.eps / 255.0)
        x = x.detach()
        prev_v = v.detach()
    return rows


def pgd_path(model, x0, y, args):
    xs = [x0.detach()]
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-args.eps / 255.0, args.eps / 255.0), x0, args.eps / 255.0)
    xs.append(x.detach())
    for _ in range(args.attack_steps):
        x.requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        grad = torch.autograd.grad(loss, x)[0]
        x = project_linf(x.detach() + (args.attack_step / 255.0) * grad.detach().sign(), x0, args.eps / 255.0)
        xs.append(x.detach())
    return xs


def square_path(model, x0, y, args, seed: int):
    rng = np.random.default_rng(seed)
    xs = [x0.detach()]
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-args.eps / 255.0, args.eps / 255.0), x0, args.eps / 255.0)
    xs.append(x.detach())
    with torch.no_grad():
        best_loss = float(F.cross_entropy(model(x), y).item())
    for q in range(args.square_queries):
        size = max(1, int(round(32 * (1 - q / max(args.square_queries, 1)) ** 0.5)))
        i = int(rng.integers(0, 33 - size))
        j = int(rng.integers(0, 33 - size))
        cand = x.clone()
        sign = -1.0 if rng.random() < 0.5 else 1.0
        cand[:, :, i : i + size, j : j + size] = x0[:, :, i : i + size, j : j + size] + sign * (args.eps / 255.0)
        cand = project_linf(cand, x0, args.eps / 255.0)
        with torch.no_grad():
            logits = model(cand)
            loss = float(F.cross_entropy(logits, y).item())
            succ = int(logits.argmax(1).item() != int(y.item()))
        if loss > best_loss or succ:
            x = cand.detach()
            best_loss = max(best_loss, loss)
            xs.append(x)
        if succ:
            break
    return xs


def path_features(model, xs):
    with torch.no_grad():
        return np.concatenate([feat(model, x).cpu().numpy() for x in xs], axis=0)


def compare_paths(road_h: np.ndarray, attack_h: np.ndarray) -> dict:
    d = np.linalg.norm(attack_h[:, None, :] - road_h[None, :, :], axis=2)
    return {
        "mean_attack_to_road_hidden_dist": float(d.min(axis=1).mean()),
        "median_attack_to_road_hidden_dist": float(np.median(d.min(axis=1))),
        "mean_road_to_attack_hidden_dist": float(d.min(axis=0).mean()),
        "endpoint_attack_to_road_hidden_dist": float(d[-1].min()),
    }


def select_images(dataset, model, n, device):
    rows = []
    for idx in range(len(dataset)):
        x, y0 = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([int(y0)], device=device)
        with torch.no_grad():
            pred = int(model(x).argmax(1).item())
        if pred == int(y0):
            rows.append((idx, int(y0)))
        if len(rows) >= n:
            break
    return rows


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


def make_plot(model, examples: dict, out: Path):
    feats = []
    labels = []
    for name, xs in examples.items():
        h = path_features(model, xs)
        feats.append(h)
        labels += [name] * len(h)
    z = PCA(n_components=2, random_state=0).fit_transform(np.concatenate(feats, axis=0))
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    start = 0
    colors = {"road_forward": "#4C78A8", "road_backward": "#72B7B2", "pgd": "#E45756", "square": "#F58518"}
    for name, h in examples.items():
        pts = z[start : start + len(h)]
        start += len(h)
        ax.plot(pts[:, 0], pts[:, 1], marker="o", ms=3, lw=1.6, color=colors.get(name, None), label=name)
        ax.scatter(pts[0, 0], pts[0, 1], s=40, marker="s", color=colors.get(name, None))
        ax.scatter(pts[-1, 0], pts[-1, 1], s=55, marker="x", color=colors.get(name, None))
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(fontsize=8)
    fig.savefig(out / "singular_road_vs_attacks_example.png", dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/singular_road_tracing_resnet50_quick")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--images", type=int, default=12)
    p.add_argument("--class-filter", type=int, default=-1)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--road-steps", type=int, default=24)
    p.add_argument("--step-l2", type=float, default=0.18)
    p.add_argument("--use-sign-step", action="store_true")
    p.add_argument("--step-linf", type=float, default=1.0)
    p.add_argument("--power-iters", type=int, default=8)
    p.add_argument("--width-dirs", type=int, default=16)
    p.add_argument("--width-alpha", type=float, default=0.5)
    p.add_argument("--attack-steps", type=int, default=20)
    p.add_argument("--attack-step", type=float, default=2.0)
    p.add_argument("--square-queries", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
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
        print(f"[WARN] requested {args.images}, found {len(selected)} clean-correct images for class_filter={class_filter}", flush=True)
    all_rows = []
    comp_rows = []
    example_paths = None
    for image_id, label in selected:
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        rows_f = trace_one(model, x0, y, args, image_id, +1.0, device)
        rows_b = trace_one(model, x0, y, args, image_id, -1.0, device)
        all_rows += rows_f + rows_b
        # Reconstruct road feature paths by rerunning trace cheaply? Use row hidden distances not enough,
        # so collect attack comparisons against freshly traced image states below.
        # For comparison we approximate with feature sequence generated by PGD/Square and the road rows
        # are not available as x states; rerun a compact path collector inline.
        pgd_xs = pgd_path(model, x0, y, args)
        sq_xs = square_path(model, x0, y, args, args.seed + image_id)
        # Collect road x states for forward/backward once for path comparison/plot.
        road_xs = []
        x = x0.detach().clone()
        prev_v = None
        for t in range(args.road_steps + 1):
            road_xs.append(x.detach())
            if t == args.road_steps:
                break
            v, _ = estimate_top_direction(model, x, args.power_iters, prev_v, args.seed + image_id * 3001 + t)
            if prev_v is not None and float((v.flatten(1) * prev_v.flatten(1)).sum().item()) < 0:
                v = -v
            step = args.step_l2 * v
            if args.use_sign_step:
                step = (args.step_linf / 255.0) * v.sign()
            x = project_linf(x + step, x0, args.eps / 255.0).detach()
            prev_v = v.detach()
        road_h = path_features(model, road_xs)
        for attack_name, xs in [("pgd", pgd_xs), ("square", sq_xs)]:
            comp = compare_paths(road_h, path_features(model, xs))
            comp.update({"image_id": image_id, "label": label, "attack": attack_name, "n_attack_points": len(xs)})
            comp_rows.append(comp)
        if example_paths is None:
            # Also collect backward road for the plot.
            back_xs = []
            x = x0.detach().clone()
            prev_v = None
            for t in range(args.road_steps + 1):
                back_xs.append(x.detach())
                if t == args.road_steps:
                    break
                v, _ = estimate_top_direction(model, x, args.power_iters, prev_v, args.seed + image_id * 4001 + t)
                if prev_v is not None and float((v.flatten(1) * prev_v.flatten(1)).sum().item()) < 0:
                    v = -v
                x = project_linf(x - args.step_l2 * v, x0, args.eps / 255.0).detach()
                prev_v = v.detach()
            example_paths = {"road_forward": road_xs, "road_backward": back_xs, "pgd": pgd_xs, "square": sq_xs}

    roads = pd.DataFrame(all_rows)
    comps = pd.DataFrame(comp_rows)
    roads.to_csv(out / "singular_road_trace_steps.csv", index=False)
    comps.to_csv(out / "singular_road_attack_comparison.csv", index=False)
    summary = (
        roads.groupby(["direction"])
        .agg(
            n=("image_id", "nunique"),
            mean_final_linf=("linf_from_start", "max"),
            mean_sigma1=("sigma1_est", "mean"),
            mean_width=("width_count", "mean"),
            mean_erank=("width_erank", "mean"),
            class_change_rate=("changed_class", "max"),
            success_rate=("success", "max"),
            mean_hidden_return=("return_hidden", "max"),
        )
        .reset_index()
    )
    comp_summary = comps.groupby("attack").mean(numeric_only=True).reset_index()
    summary.to_csv(out / "singular_road_summary.csv", index=False)
    comp_summary.to_csv(out / "singular_road_attack_comparison_summary.csv", index=False)
    if example_paths is not None:
        make_plot(model, example_paths, out)
    (out / "metadata.json").write_text(json.dumps(vars(args) | {"device": str(device)}, indent=2))
    print("Road summary:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nAttack comparison:", flush=True)
    print(comp_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
