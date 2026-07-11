#!/usr/bin/env python3
"""Build a paper figure explaining hidden-Jacobian road directions.

The figure is intentionally pedagogical and uses real calculations.  Panel A
shows one clean image in a two-dimensional hidden-feature PCA plane and two
local hidden-Jacobian singular directions with lengths scaled by singular gain.
Subsequent panels trace
routes obtained by repeatedly recomputing the top-k singular directions and
choosing the feasible sign/rank with the largest immediate true-class margin
decrease.  Thus the roads are not objective-free in the route panels: mobility
proposes directions, and margin selects the plotted step.
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
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import load_model as load_feature_model, project_linf  # noqa: E402


CLASSES = [
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

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(device: torch.device):
    return load_feature_model("bbb_resnet50", device).eval()


def feat(model, x: torch.Tensor) -> torch.Tensor:
    return model.pooled_layer4(x).flatten(1)


def true_margin(logits: torch.Tensor, y: int) -> float:
    vals = logits.detach().clone()
    true_val = float(vals[0, y].item())
    vals[0, y] = -1e9
    return float(true_val - vals.max(dim=1).values[0].item())


def normalize_l2(v: torch.Tensor) -> torch.Tensor:
    return v / v.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)


def orthogonalize(v: torch.Tensor, basis: list[torch.Tensor]) -> torch.Tensor:
    out = v
    for b in basis:
        coeff = (out.flatten(1) * b.flatten(1)).sum(dim=1).view(-1, 1, 1, 1)
        out = out - coeff * b
    return out


def estimate_topk(
    model,
    x: torch.Tensor,
    *,
    k: int,
    power_iters: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[float], list[np.ndarray]]:
    """Power iteration for top input-space right singular directions of J_h(x)."""

    def f(inp: torch.Tensor) -> torch.Tensor:
        return feat(model, inp)

    gen = torch.Generator(device=x.device).manual_seed(seed)
    dirs: list[torch.Tensor] = []
    sigmas: list[float] = []
    jvps: list[np.ndarray] = []
    for _rank in range(k):
        v = normalize_l2(torch.randn(x.shape, generator=gen, device=x.device))
        v = normalize_l2(orthogonalize(v, dirs))
        for _ in range(power_iters):
            x_req = x.detach().requires_grad_(True)
            _h, jv = torch.autograd.functional.jvp(f, x_req, v, create_graph=False, strict=False)
            h = f(x_req)
            dot = (h * jv.detach()).sum()
            w = torch.autograd.grad(dot, x_req)[0]
            v = normalize_l2(orthogonalize(w.detach(), dirs))
        with torch.no_grad():
            _h, jv = torch.autograd.functional.jvp(f, x.detach(), v, create_graph=False, strict=False)
            sigmas.append(float(jv.flatten(1).norm(dim=1).item()))
            jvps.append(jv.squeeze(0).detach().cpu().numpy())
        dirs.append(v.detach())
    return dirs, sigmas, jvps


def select_clean_correct(dataset, model, n: int, device: torch.device) -> list[tuple[int, int]]:
    rows = []
    for idx in range(len(dataset)):
        x, y0 = dataset[idx]
        xb = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(model(xb).argmax(1).item())
        if pred == int(y0):
            rows.append((idx, int(y0)))
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise RuntimeError(f"Only found {len(rows)} clean-correct images, requested {n}.")
    return rows


def trace_rank_road(
    model,
    x0: torch.Tensor,
    *,
    rank: int,
    steps: int,
    eps: float,
    eta: float,
    power_iters: int,
    seed: int,
    sign: float = 1.0,
) -> tuple[list[torch.Tensor], list[np.ndarray], list[float]]:
    xs = [x0.detach().clone()]
    feats = [feat(model, x0).squeeze(0).detach().cpu().numpy()]
    sigmas = []
    prev_v = None
    x = x0.detach().clone()
    for t in range(steps):
        dirs, step_sigmas, _jvps = estimate_topk(model, x, k=rank, power_iters=power_iters, seed=seed + 1009 * t)
        v = dirs[rank - 1]
        if prev_v is not None:
            cos = float((v.flatten(1) * prev_v.flatten(1)).sum().item())
            if cos < 0:
                v = -v
        step = sign * eta * v.sign()
        x = project_linf(x + step, x0, eps).detach()
        xs.append(x)
        feats.append(feat(model, x).squeeze(0).detach().cpu().numpy())
        sigmas.append(float(step_sigmas[rank - 1]))
        prev_v = v.detach()
    return xs, feats, sigmas


def trace_margin_selected_route(
    model,
    x0: torch.Tensor,
    y: int,
    *,
    k: int,
    steps: int,
    eps: float,
    eta: float,
    power_iters: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[np.ndarray], list[dict]]:
    xs = [x0.detach().clone()]
    feats = [feat(model, x0).squeeze(0).detach().cpu().numpy()]
    route_rows: list[dict] = []
    x = x0.detach().clone()
    for t in range(steps):
        with torch.no_grad():
            margin_before = true_margin(model(x), y)
            pred_before = int(model(x).argmax(1).item())
        dirs, step_sigmas, _jvps = estimate_topk(model, x, k=k, power_iters=power_iters, seed=seed + 1009 * t)
        candidates = []
        for rank, v in enumerate(dirs, start=1):
            for sign in (-1.0, 1.0):
                x_cand = project_linf(x + sign * eta * v.sign(), x0, eps).detach()
                with torch.no_grad():
                    logits = model(x_cand)
                    margin_after = true_margin(logits, y)
                    pred_after = int(logits.argmax(1).item())
                candidates.append(
                    {
                        "rank": rank,
                        "sign": sign,
                        "x": x_cand,
                        "margin_after": margin_after,
                        "margin_drop": margin_before - margin_after,
                        "pred_after": pred_after,
                        "sigma": float(step_sigmas[rank - 1]),
                    }
                )
        chosen = max(candidates, key=lambda row: (row["margin_drop"], row["sigma"]))
        x = chosen["x"].detach()
        xs.append(x)
        feats.append(feat(model, x).squeeze(0).detach().cpu().numpy())
        route_rows.append(
            {
                "step": t + 1,
                "chosen_rank": chosen["rank"],
                "chosen_sign": chosen["sign"],
                "sigma": chosen["sigma"],
                "margin_before": margin_before,
                "margin_after": chosen["margin_after"],
                "margin_drop": chosen["margin_drop"],
                "pred_before": pred_before,
                "pred_after": chosen["pred_after"],
            }
        )
    return xs, feats, route_rows


def fit_pca(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return mean, vt[:2]


def transform(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (x - mean) @ basis.T


def arrow(ax, start, end, color, lw=1.6, alpha=1.0, style="-|>"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, alpha=alpha, shrinkA=0, shrinkB=0),
        zorder=5,
    )


def set_limits(ax, pts: np.ndarray, pad_frac: float = 0.14) -> None:
    xmin, ymin = np.nanmin(pts, axis=0)
    xmax, ymax = np.nanmax(pts, axis=0)
    dx = max(xmax - xmin, 1e-3)
    dy = max(ymax - ymin, 1e-3)
    ax.set_xlim(xmin - pad_frac * dx, xmax + pad_frac * dx)
    ax.set_ylim(ymin - pad_frac * dy, ymax + pad_frac * dy)


def smooth_vector_field(starts: np.ndarray, deltas: np.ndarray, nx: int = 28, ny: int = 28):
    xmin, ymin = starts.min(axis=0)
    xmax, ymax = starts.max(axis=0)
    pad_x = max((xmax - xmin) * 0.12, 1e-3)
    pad_y = max((ymax - ymin) * 0.12, 1e-3)
    xs = np.linspace(xmin - pad_x, xmax + pad_x, nx)
    ys = np.linspace(ymin - pad_y, ymax + pad_y, ny)
    u_sum = np.zeros((ny, nx), dtype=float)
    v_sum = np.zeros((ny, nx), dtype=float)
    count = np.zeros((ny, nx), dtype=float)
    xi = np.clip(np.searchsorted(xs, starts[:, 0]) - 1, 0, nx - 1)
    yi = np.clip(np.searchsorted(ys, starts[:, 1]) - 1, 0, ny - 1)
    for i, j, d in zip(xi, yi, deltas):
        u_sum[j, i] += d[0]
        v_sum[j, i] += d[1]
        count[j, i] += 1.0
    u = np.divide(u_sum, count, out=np.zeros_like(u_sum), where=count > 0)
    v = np.divide(v_sum, count, out=np.zeros_like(v_sum), where=count > 0)
    weight = count.copy()
    for _ in range(7):
        u = (
            u
            + np.roll(u, 1, axis=0)
            + np.roll(u, -1, axis=0)
            + np.roll(u, 1, axis=1)
            + np.roll(u, -1, axis=1)
        ) / 5.0
        v = (
            v
            + np.roll(v, 1, axis=0)
            + np.roll(v, -1, axis=0)
            + np.roll(v, 1, axis=1)
            + np.roll(v, -1, axis=1)
        ) / 5.0
        weight = (
            weight
            + np.roll(weight, 1, axis=0)
            + np.roll(weight, -1, axis=0)
            + np.roll(weight, 1, axis=1)
            + np.roll(weight, -1, axis=1)
        ) / 5.0
    u[weight < 0.015] = np.nan
    v[weight < 0.015] = np.nan
    return xs, ys, u, v


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/hidden_jacobian_road_map")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--clean-offset", type=int, default=0)
    p.add_argument("--many-images", type=int, default=20)
    p.add_argument("--steps-single", type=int, default=8)
    p.add_argument("--steps-many", type=int, default=6)
    p.add_argument("--route-k", type=int, default=2)
    p.add_argument("--proposal-ranks", type=int, nargs=2, default=(1, 20))
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--power-iters", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    proposal_ranks = tuple(int(r) for r in args.proposal_ranks)
    if any(r < 1 for r in proposal_ranks):
        raise ValueError("--proposal-ranks must contain positive one-indexed ranks.")
    max_rank = max(max(proposal_ranks), args.route_k)
    selected_all = select_clean_correct(dataset, model, args.clean_offset + max(args.many_images, 1), device)
    selected = selected_all[args.clean_offset :]

    eps = args.eps / 255.0
    eta = args.eta / 255.0

    single_idx, single_y = selected[0]
    x0 = dataset[single_idx][0].unsqueeze(0).to(device)
    h0 = feat(model, x0).squeeze(0).detach().cpu().numpy()
    dirs0, sigmas0, jvps0 = estimate_topk(model, x0, k=max_rank, power_iters=args.power_iters, seed=args.seed + 7)
    _xs, fs, single_route = trace_margin_selected_route(
        model,
        x0,
        single_y,
        k=args.route_k,
        steps=args.steps_single,
        eps=eps,
        eta=eta,
        power_iters=args.power_iters,
        seed=args.seed + 4000,
    )
    single_paths = {"margin_selected": {"features": np.stack(fs), "route": single_route}}

    many_paths = []
    for image_no, (idx, y) in enumerate(selected[: args.many_images]):
        xi = dataset[idx][0].unsqueeze(0).to(device)
        _xs, fs, route = trace_margin_selected_route(
            model,
            xi,
            y,
            k=args.route_k,
            steps=args.steps_many,
            eps=eps,
            eta=eta,
            power_iters=args.power_iters,
            seed=args.seed + 9000 + 97 * image_no,
        )
        many_paths.append({"dataset_idx": idx, "label": y, "features": np.stack(fs), "route": route})

    all_features = [h0[None, :], *(p0["features"] for p0 in single_paths.values())]
    all_features += [m["features"] for m in many_paths]
    mean, basis = fit_pca(np.concatenate(all_features, axis=0))
    single_xy = {name: transform(path["features"], mean, basis) for name, path in single_paths.items()}
    many_xy = [(m, transform(m["features"], mean, basis)) for m in many_paths]

    jvp_xy = []
    selected_steps = single_xy["margin_selected"][1:] - single_xy["margin_selected"][:-1]
    step_norms = np.linalg.norm(selected_steps, axis=1)
    scale = 1.15 * float(np.median(step_norms[step_norms > 1e-12])) if np.any(step_norms > 1e-12) else 1.0
    sigma_ref = max(sigmas0[proposal_ranks[0] - 1], 1e-12)
    for rank in proposal_ranks:
        jvp = jvps0[rank - 1]
        raw = basis @ jvp
        raw = raw / (np.linalg.norm(raw) + 1e-12)
        raw = raw * max(scale, 1e-6) * (sigmas0[rank - 1] / sigma_ref)
        jvp_xy.append(raw)
    h0_xy = transform(h0[None, :], mean, basis)[0]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter([h0_xy[0]], [h0_xy[1]], s=42, color="black", zorder=6, label="start")
    proposal_pts = [h0_xy]
    proposal_specs = [
        (proposal_ranks[0], jvp_xy[0], "#d95f02"),
        (proposal_ranks[1], jvp_xy[1], "#1b9e77"),
    ]
    for rank, direction, color in proposal_specs:
        sigma_ratio = sigmas0[rank - 1] / sigma_ref
        label = rf"$v_{{{rank}}}$ ({sigma_ratio:.2f}$\sigma_1$)"
        arrow(ax, h0_xy, h0_xy + direction, color, lw=2.0, alpha=0.80)
        arrow(ax, h0_xy, h0_xy - direction, color, lw=1.3, alpha=0.22, style="-")
        proposal_pts.extend([h0_xy + direction, h0_xy - direction])
        ax.text(
            *(h0_xy + 1.12 * direction),
            label,
            fontsize=8,
            color=color,
            ha="center",
            va="center",
        )
    ax.set_title(f"A. rank {proposal_ranks[0]} vs rank {proposal_ranks[1]} proposals")
    ax.set_xlabel("hidden PCA 1")
    ax.set_ylabel("hidden PCA 2")
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, fontsize=7, loc="best")
    set_limits(ax, np.stack(proposal_pts), pad_frac=0.22)

    ax = axes[0, 1]
    pts = single_xy["margin_selected"]
    ax.plot(pts[:, 0], pts[:, 1], "-o", color="#5b4b8a", lw=1.9, ms=3.0)
    ax.scatter([pts[0, 0]], [pts[0, 1]], s=36, color="black", zorder=6)
    for a, b in zip(pts[:-1], pts[1:]):
        arrow(ax, a, b, "#5b4b8a", lw=1.1, alpha=0.55)
    ax.set_title("B. one margin-selected route")
    ax.set_xlabel("hidden PCA 1")
    ax.set_ylabel("hidden PCA 2")
    ax.grid(alpha=0.16)
    set_limits(ax, pts, pad_frac=0.10)

    ax = axes[1, 0]
    route_color = "#5b4b8a"
    for m, pts in many_xy:
        ax.plot(pts[:, 0], pts[:, 1], "-", color=route_color, lw=1.0, alpha=0.42)
        ax.scatter([pts[0, 0]], [pts[0, 1]], s=14, color="black", alpha=0.45, zorder=4)
        arrow(ax, pts[-2], pts[-1], route_color, lw=0.8, alpha=0.38)
    ax.set_title("C. selected routes across images")
    ax.set_xlabel("hidden PCA 1")
    ax.set_ylabel("hidden PCA 2")
    ax.grid(alpha=0.16)
    set_limits(ax, np.concatenate([pts for _m, pts in many_xy], axis=0), pad_frac=0.07)

    ax = axes[1, 1]
    starts = []
    deltas = []
    for _m, pts in many_xy:
        starts.append(pts[:-1])
        deltas.append(pts[1:] - pts[:-1])
    starts_arr = np.concatenate(starts, axis=0)
    deltas_arr = np.concatenate(deltas, axis=0)
    xs_grid, ys_grid, u, v = smooth_vector_field(starts_arr, deltas_arr)
    ax.streamplot(xs_grid, ys_grid, u, v, color="#5b4b8a", density=1.15, linewidth=1.0, arrowsize=0.85)
    ax.scatter(starts_arr[:, 0], starts_arr[:, 1], s=4, color="black", alpha=0.12)
    ax.set_title("D. smoothed selected-route field")
    ax.set_xlabel("hidden PCA 1")
    ax.set_ylabel("hidden PCA 2")
    ax.grid(alpha=0.16)
    set_limits(ax, starts_arr, pad_frac=0.09)

    fig.suptitle("Hidden-Jacobian routing map", fontsize=10)
    png = out / "hidden_jacobian_road_map.png"
    pdf = out / "hidden_jacobian_road_map.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for name, pts in single_xy.items():
        for step, xy in enumerate(pts):
            route_meta = {}
            if step > 0 and name == "margin_selected":
                route_meta = single_paths[name]["route"][step - 1]
            rows.append(
                {
                    "panel": "single",
                    "path": name,
                    "dataset_idx": single_idx,
                    "label": single_y,
                    "class": CLASSES[single_y],
                    "step": step,
                    "pc1": float(xy[0]),
                    "pc2": float(xy[1]),
                    **route_meta,
                }
            )
    for m, pts in many_xy:
        for step, xy in enumerate(pts):
            route_meta = {}
            if step > 0:
                route_meta = m["route"][step - 1]
            rows.append(
                {
                    "panel": "many",
                    "path": "margin_selected",
                    "dataset_idx": m["dataset_idx"],
                    "label": m["label"],
                    "class": CLASSES[m["label"]],
                    "step": step,
                    "pc1": float(xy[0]),
                    "pc2": float(xy[1]),
                    **route_meta,
                }
            )
    pd.DataFrame(rows).to_csv(out / "hidden_jacobian_road_map_points.csv", index=False)
    feature_rows = []
    feature_arrays = []

    def add_feature_path(panel: str, dataset_idx: int, label: int, path_name: str, features: np.ndarray) -> None:
        start = len(feature_arrays)
        feature_arrays.append(features.astype(np.float32, copy=False))
        for step in range(features.shape[0]):
            feature_rows.append(
                {
                    "panel": panel,
                    "path": path_name,
                    "dataset_idx": int(dataset_idx),
                    "label": int(label),
                    "step": int(step),
                    "array_index": len(feature_arrays) - 1,
                    "row_index": start,
                }
            )

    add_feature_path("single", single_idx, single_y, "margin_selected", single_paths["margin_selected"]["features"])
    for m in many_paths:
        add_feature_path("many", int(m["dataset_idx"]), int(m["label"]), "margin_selected", m["features"])
    np.savez_compressed(
        out / "hidden_jacobian_road_map_features.npz",
        features=np.concatenate(feature_arrays, axis=0),
        lengths=np.array([arr.shape[0] for arr in feature_arrays], dtype=np.int64),
        dataset_idx=np.array([single_idx, *[int(m["dataset_idx"]) for m in many_paths]], dtype=np.int64),
        labels=np.array([single_y, *[int(m["label"]) for m in many_paths]], dtype=np.int64),
    )
    pd.DataFrame(feature_rows).to_csv(out / "hidden_jacobian_road_map_features_index.csv", index=False)
    metadata = {
        "model": "bbb_resnet50",
        "layer": "pooled layer4",
        "dataset": "CIFAR-10 test clean-correct",
        "single_dataset_idx": single_idx,
        "single_class": CLASSES[single_y],
        "clean_offset": args.clean_offset,
        "many_images": args.many_images,
        "eps_over_255": args.eps,
        "eta_over_255": args.eta,
        "power_iters": args.power_iters,
        "proposal_ranks": list(proposal_ranks),
        "proposal_sigmas": {f"rank_{rank}": sigmas0[rank - 1] for rank in proposal_ranks},
        "proposal_sigma_ratios_to_first_rank": {
            f"rank_{rank}": sigmas0[rank - 1] / sigma_ref for rank in proposal_ranks
        },
        "route_k": args.route_k,
        "steps_single": args.steps_single,
        "steps_many": args.steps_many,
        "route_rule": "at each state evaluate both signs of the top-k hidden-Jacobian singular directions and choose the candidate with the largest immediate true-class margin decrease",
        "outputs": {"png": str(png), "pdf": str(pdf)},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
