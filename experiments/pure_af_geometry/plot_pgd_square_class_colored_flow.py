#!/usr/bin/env python3
"""Side-by-side class-colored PGD/Square transport flow diagnostics.

The script uses the same clean-correct CIFAR-10 images for PGD and Square.
Trajectories stop at the first adversarial step. A shared 2D PCA basis is
fitted on successful local transport vectors from both attacks, and plotted
paths are cumulative sums of projected local steps.

Arrow colors encode classes: non-crossing arrows use the source class color;
the final arrow that first causes misclassification uses the adversarial
predicted class color.
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
from matplotlib.lines import Line2D
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import (  # noqa: E402
    load_model,
    margin,
    project_linf,
)


LAYER_MAP = {
    "bbb_resnet50": {"hidden": "layer4", "penultimate": "avgpool", "logits": "logits"},
    "bbb_vgg19_bn": {"hidden": "block5", "penultimate": "penultimate", "logits": "logits"},
    "bbb_densenet": {"hidden": "denseblock3", "penultimate": "penultimate", "logits": "logits"},
    "bbb_inception_v3": {"hidden": "mixed6", "penultimate": "penultimate", "logits": "logits"},
}

CLASS_NAMES = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}
CLASS_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#e377c2",
    7: "#7f7f7f",
    8: "#bcbd22",
    9: "#17becf",
}


def true_prob(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1).gather(1, y.view(-1, 1)).squeeze(1)


def pgd_trajectory(wrapper, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int, step_size: float):
    x0 = x.detach()
    x_adv = x0.clone()
    states = [x_adv.detach().clone()]
    for _ in range(steps):
        probe = x_adv.detach().requires_grad_(True)
        logits = wrapper(probe)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, probe)[0]
        x_adv = project_linf(x_adv + step_size * grad.sign(), x0, eps)
        states.append(x_adv.detach().clone())
        with torch.no_grad():
            if int(wrapper(x_adv).argmax(1).item()) != int(y.item()):
                break
    return states


def square_size(step: int, max_steps: int, image_size: int, min_square: int) -> int:
    frac = 1.0 - (step / max(max_steps, 1))
    side = int(round(image_size * (0.12 + 0.28 * frac)))
    return max(min_square, min(image_size, side))


def square_trajectory(wrapper, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int, seed: int, min_square: int):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    x0 = x.detach()
    x_adv = x0.clone()
    states = [x_adv.detach().clone()]
    with torch.no_grad():
        best_margin = margin(wrapper(x_adv), y)
    _, _, h, w = x.shape
    for step in range(1, steps + 1):
        side = square_size(step, steps, h, min_square)
        top = int(torch.randint(0, h - side + 1, (1,), generator=gen, device=x.device).item())
        left = int(torch.randint(0, w - side + 1, (1,), generator=gen, device=x.device).item())
        candidate = x_adv.clone()
        patch = (torch.rand((1, x.shape[1], side, side), generator=gen, device=x.device) * 2.0 - 1.0) * eps
        candidate[:, :, top : top + side, left : left + side] = x0[:, :, top : top + side, left : left + side] + patch
        candidate = project_linf(candidate, x0, eps)
        with torch.no_grad():
            cand_logits = wrapper(candidate)
            cand_margin = margin(cand_logits, y)
        if float(cand_margin.item()) < float(best_margin.item()):
            x_adv = candidate.detach()
            best_margin = cand_margin.detach()
        states.append(x_adv.detach().clone())
        with torch.no_grad():
            if int(wrapper(x_adv).argmax(1).item()) != int(y.item()):
                break
    return states


def feature_vector(wrapper, x: torch.Tensor, layer: str):
    with torch.no_grad():
        logits, feats, _raw = wrapper.forward_with_features(x)
    if layer == "logits":
        h = logits.detach().cpu().numpy()[0].astype(np.float32)
    else:
        h = feats[layer].detach().cpu().numpy()[0].astype(np.float32)
    return logits.detach(), h.reshape(-1)


def select_balanced_clean_correct(dataset, wrapper, per_class: int, device, class_filter: int | None = None):
    selected = []
    counts = {c: 0 for c in range(10)}
    for idx in range(len(dataset)):
        if class_filter is None and all(v >= per_class for v in counts.values()):
            break
        if class_filter is not None and counts[class_filter] >= per_class:
            break
        x_cpu, y0 = dataset[idx]
        y0 = int(y0)
        if class_filter is not None and y0 != class_filter:
            continue
        if counts[y0] >= per_class:
            continue
        x = x_cpu.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(wrapper(x).argmax(1).item())
        if pred == y0:
            selected.append((idx, y0))
            counts[y0] += 1
    return selected, counts


def fit_pca_2d(X: np.ndarray):
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    explained = (s * s) / max(float(np.sum(s * s)), 1e-12)
    return mean.reshape(-1).astype(np.float32), vt[:2].astype(np.float32), explained[:2]


def collect_trajectories(args, wrapper, dataset, selected, layer: str, device):
    eps = args.eps / 255.0
    step_size = args.pgd_step_size / 255.0 if args.pgd_step_size > 0 else eps / max(args.pgd_steps, 1)
    rows = []
    features = []
    meta = []
    for image_ord, (dataset_idx, label) in enumerate(selected):
        x_cpu, _ = dataset[dataset_idx]
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        for attack in ["pgd", "square"]:
            if attack == "pgd":
                states = pgd_trajectory(wrapper, x, y, eps, args.pgd_steps, step_size)
            else:
                states = square_trajectory(
                    wrapper,
                    x,
                    y,
                    eps,
                    args.square_steps,
                    seed=args.seed + image_ord * 997,
                    min_square=args.square_min_size,
                )
            h0 = None
            run_feats = []
            run_meta = []
            for step, state in enumerate(states):
                logits, h = feature_vector(wrapper, state, layer)
                if h0 is None:
                    h0 = h.copy()
                delta = h - h0
                pred = int(logits.argmax(1).item())
                run_feats.append(delta)
                run_meta.append(
                    {
                        "model": args.model,
                        "attack": attack,
                        "dataset_idx": int(dataset_idx),
                        "image_ord": int(image_ord),
                        "run_id": f"{attack}_img{image_ord}",
                        "label": int(label),
                        "layer_group": args.layer_group,
                        "layer": layer,
                        "step": int(step),
                        "normalized_progress": float(step / max(len(states) - 1, 1)),
                        "pred": pred,
                        "step_success": int(pred != int(label)),
                        "margin": float(margin(logits, y).item()),
                        "true_prob": float(true_prob(logits, y).item()),
                    }
                )
            final_success = int(run_meta[-1]["step_success"])
            final_pred = int(run_meta[-1]["pred"])
            for m in run_meta:
                m["final_success"] = final_success
                m["final_pred"] = final_pred
            features.extend(run_feats)
            meta.extend(run_meta)
        if (image_ord + 1) % 10 == 0:
            print(f"  images {image_ord + 1}/{len(selected)}", flush=True)
    return np.stack(features).astype(np.float32), meta


def project_local_steps(feat: np.ndarray, meta: list[dict]):
    run_to_indices: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        run_to_indices.setdefault(m["run_id"], []).append(i)

    local_vecs = []
    success_runs = {rid for rid, g in pd.DataFrame(meta).groupby("run_id") if int(g.final_success.max()) == 1}
    for rid, idxs in run_to_indices.items():
        if rid not in success_runs:
            continue
        idxs = sorted(idxs, key=lambda i: meta[i]["step"])
        for a, b in zip(idxs[:-1], idxs[1:]):
            local_vecs.append(feat[b] - feat[a])
    if not local_vecs:
        raise RuntimeError("No successful local vectors")
    local_vecs = np.stack(local_vecs).astype(np.float32)
    keep = np.linalg.norm(local_vecs, axis=1) > 1e-12
    mean, basis, explained = fit_pca_2d(local_vecs[keep])

    coords_by_index = {}
    for _rid, idxs in run_to_indices.items():
        idxs = sorted(idxs, key=lambda i: meta[i]["step"])
        z = np.zeros(2, dtype=np.float32)
        coords_by_index[idxs[0]] = z.copy()
        for a, b in zip(idxs[:-1], idxs[1:]):
            local = feat[b] - feat[a]
            step_coord = (local - mean) @ basis.T
            z = z + step_coord.astype(np.float32)
            coords_by_index[b] = z.copy()

    rows = []
    for i, m in enumerate(meta):
        r = dict(m)
        r["pc1"] = float(coords_by_index[i][0])
        r["pc2"] = float(coords_by_index[i][1])
        rows.append(r)
    return pd.DataFrame(rows), explained


def project_displacements(feat: np.ndarray, meta: list[dict]):
    meta_df = pd.DataFrame(meta)
    success_runs = {
        rid for rid, g in meta_df.groupby("run_id") if int(g.final_success.max()) == 1
    }
    fit_rows = []
    for i, m in enumerate(meta):
        if m["run_id"] in success_runs and int(m["step"]) > 0:
            fit_rows.append(feat[i])
    if not fit_rows:
        raise RuntimeError("No successful displacement vectors")
    X = np.stack(fit_rows).astype(np.float32)
    keep = np.linalg.norm(X, axis=1) > 1e-12
    _mean, basis, explained = fit_pca_2d(X[keep])

    rows = []
    for i, m in enumerate(meta):
        coord = feat[i] @ basis.T
        r = dict(m)
        r["pc1"] = float(coord[0])
        r["pc2"] = float(coord[1])
        rows.append(r)
    return pd.DataFrame(rows), explained


def add_step_vectors(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df.sort_values(["attack", "run_id", "step"]).groupby(["attack", "run_id"], sort=False):
        g = g.copy()
        g["next_pred"] = g["pred"].shift(-1)
        g["dx"] = g["pc1"].shift(-1) - g["pc1"]
        g["dy"] = g["pc2"].shift(-1) - g["pc2"]
        rows.append(g.iloc[:-1])
    out = pd.concat(rows, ignore_index=True).dropna(subset=["pc1", "pc2", "dx", "dy"])
    out["arrow_class"] = out["label"].astype(int)
    crossing = (out["step_success"] == 0) & (out["next_pred"].notna()) & (out["next_pred"].astype(int) != out["label"].astype(int))
    out.loc[crossing, "arrow_class"] = out.loc[crossing, "next_pred"].astype(int)
    out["is_crossing_arrow"] = crossing.astype(int)
    return out


def smooth_field(vecs: pd.DataFrame, grid_n: int, bandwidth_scale: float, min_weight_percentile: float):
    x = vecs.pc1.to_numpy(float)
    y = vecs.pc2.to_numpy(float)
    dx = vecs.dx.to_numpy(float)
    dy = vecs.dy.to_numpy(float)
    xlo, xhi = np.percentile(x, [1, 99])
    ylo, yhi = np.percentile(y, [1, 99])
    pad_x = 0.08 * max(xhi - xlo, 1e-6)
    pad_y = 0.08 * max(yhi - ylo, 1e-6)
    gx = np.linspace(xlo - pad_x, xhi + pad_x, grid_n)
    gy = np.linspace(ylo - pad_y, yhi + pad_y, grid_n)
    X, Y = np.meshgrid(gx, gy)
    sigma = bandwidth_scale * max(xhi - xlo, yhi - ylo, 1e-6)

    points = np.column_stack([x, y])
    vectors = np.column_stack([dx, dy])
    flat = np.column_stack([X.ravel(), Y.ravel()])
    u = np.zeros(len(flat))
    v = np.zeros(len(flat))
    wsum = np.zeros(len(flat))
    for start in range(0, len(flat), 512):
        stop = start + 512
        q = flat[start:stop]
        dist2 = ((q[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        w = np.exp(-0.5 * dist2 / (sigma**2))
        ws = w.sum(axis=1)
        u[start:stop] = (w @ vectors[:, 0]) / np.maximum(ws, 1e-12)
        v[start:stop] = (w @ vectors[:, 1]) / np.maximum(ws, 1e-12)
        wsum[start:stop] = ws
    U = u.reshape(X.shape)
    V = v.reshape(Y.shape)
    W = wsum.reshape(X.shape)
    mask = W < np.percentile(W, min_weight_percentile)
    U = np.ma.array(U, mask=mask)
    V = np.ma.array(V, mask=mask)
    W = np.ma.array(W, mask=mask)
    speed = np.sqrt(U**2 + V**2)
    return X, Y, U, V, W, speed


def draw_class_arrows(ax, vecs: pd.DataFrame, alpha: float, scale: float = 1.0):
    for cls, g in vecs.groupby("arrow_class"):
        color = CLASS_COLORS[int(cls)]
        normal = g[g.is_crossing_arrow == 0]
        cross = g[g.is_crossing_arrow == 1]
        if not normal.empty:
            ax.plot(
                np.column_stack([normal.pc1, normal.pc1 + normal.dx]).T,
                np.column_stack([normal.pc2, normal.pc2 + normal.dy]).T,
                color=color,
                alpha=alpha,
                lw=0.7,
            )
        if not cross.empty:
            ax.quiver(
                cross.pc1,
                cross.pc2,
                cross.dx,
                cross.dy,
                angles="xy",
                scale_units="xy",
                scale=scale,
                color=color,
                alpha=alpha,
                width=0.002,
            )


def draw_panel(ax, df: pd.DataFrame, vecs: pd.DataFrame, attack: str, with_stream: bool = True):
    d = df[(df.attack == attack) & (df.final_success == 1)].copy()
    v = vecs[(vecs.attack == attack) & (vecs.final_success == 1)].copy()
    if d.empty or v.empty:
        ax.set_title(f"{attack}: no successful runs")
        return
    if with_stream:
        X, Y, U, V, W, speed = smooth_field(v, 60, 0.075, 18.0)
        density = np.log1p(W.filled(0))
        ax.imshow(
            density,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin="lower",
            cmap="Greys",
            alpha=0.16,
            aspect="auto",
            zorder=0,
        )
        ax.streamplot(X, Y, U, V, color="#334155", density=1.1, linewidth=0.9, arrowsize=1.0, minlength=0.08, zorder=1)
    for run_id, g in d.groupby("run_id"):
        ax.plot(g.pc1, g.pc2, color="#94a3b8", alpha=0.12, lw=0.8, zorder=2)
    draw_class_arrows(ax, v, alpha=0.72)
    ax.scatter([0], [0], color="black", marker="o", s=36, zorder=5)
    ax.axhline(0, color="black", lw=0.8, alpha=0.18)
    ax.axvline(0, color="black", lw=0.8, alpha=0.18)
    ax.grid(alpha=0.16)
    ax.set_title(f"{attack.upper()} successful trajectories ({d.run_id.nunique()})")
    ax.set_xlabel("shared local-transport PC1")
    ax.set_ylabel("shared local-transport PC2")


def filter_common_success(feat: np.ndarray, meta: list[dict]):
    meta_df = pd.DataFrame(meta)
    success = (
        meta_df.groupby(["image_ord", "attack"])["final_success"]
        .max()
        .unstack(fill_value=0)
    )
    common = set(success[(success.get("pgd", 0) == 1) & (success.get("square", 0) == 1)].index.astype(int))
    keep_idx = [i for i, m in enumerate(meta) if int(m["image_ord"]) in common]
    return feat[keep_idx], [meta[i] for i in keep_idx], common


def set_shared_limits(axes, df: pd.DataFrame):
    if df.empty:
        return
    x = df.pc1.to_numpy(float)
    y = df.pc2.to_numpy(float)
    xlo, xhi = np.nanmin(x), np.nanmax(x)
    ylo, yhi = np.nanmin(y), np.nanmax(y)
    pad_x = 0.08 * max(xhi - xlo, 1e-6)
    pad_y = 0.08 * max(yhi - ylo, 1e-6)
    for ax in axes:
        ax.set_xlim(xlo - pad_x, xhi + pad_x)
        ax.set_ylim(ylo - pad_y, yhi + pad_y)


def save_feature_displacements(out_dir: Path, stem: str, feat: np.ndarray, meta: list[dict], args: argparse.Namespace, layer: str):
    """Save high-dimensional h(x_t)-h(x_0) vectors for projection-only reruns."""
    meta_json = json.dumps(meta)
    np.savez_compressed(
        out_dir / f"{stem}_feature_displacements.npz",
        feature_displacements=feat.astype(np.float32),
        meta_json=np.array(meta_json),
        layer=np.array(layer),
        args_json=np.array(json.dumps(vars(args))),
    )


def run(args: argparse.Namespace):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    wrapper = load_model(args.model, device)
    layer = LAYER_MAP[args.model][args.layer_group]
    selected, counts = select_balanced_clean_correct(dataset, wrapper, args.per_class, device, args.class_filter)
    print(f"[SELECTED] n={len(selected)} counts={counts} model={args.model} layer={layer}", flush=True)
    feat, meta = collect_trajectories(args, wrapper, dataset, selected, layer, device)
    common_success_image_ords = None
    if args.common_success_only:
        feat, meta, common_success_image_ords = filter_common_success(feat, meta)
        print(f"[COMMON SUCCESS] n={len(common_success_image_ords)} image IDs plotted for both attacks", flush=True)
        if len(common_success_image_ords) == 0:
            raise RuntimeError("No images succeeded for both PGD and Square; rerun with more images or Square steps.")

    raw_meta_df = pd.DataFrame(meta)
    n_feature_images = raw_meta_df.image_ord.nunique()
    feature_suffix = "common_success" if args.common_success_only else "all_success"
    if args.class_filter is not None:
        feature_suffix = f"class{args.class_filter}_{feature_suffix}"
    feature_stem = f"pgd_square_class_colored_{args.model}_{args.layer_group}_{feature_suffix}_n{n_feature_images}"
    save_feature_displacements(out_dir, feature_stem, feat, meta, args, layer)
    print(f"[SAVED FEATURES] {out_dir / (feature_stem + '_feature_displacements.npz')}", flush=True)

    if args.coordinate_mode == "local_steps":
        df, explained = project_local_steps(feat, meta)
    elif args.coordinate_mode == "displacement":
        df, explained = project_displacements(feat, meta)
    else:
        raise ValueError(f"Unknown coordinate mode: {args.coordinate_mode}")
    vecs = add_step_vectors(df)

    n_plot_images = df.image_ord.nunique()
    suffix = "common_success" if args.common_success_only else "all_success"
    if args.class_filter is not None:
        suffix = f"class{args.class_filter}_{suffix}"
    field_suffix = "stream" if args.streamlines else "raw_vectors"
    stem = f"pgd_square_class_colored_{args.model}_{args.layer_group}_{suffix}_{args.coordinate_mode}_{field_suffix}_n{n_plot_images}"
    df.to_csv(out_dir / f"{stem}_timeseries.csv", index=False)
    vecs.to_csv(out_dir / f"{stem}_step_vectors.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.2), constrained_layout=True)
    draw_panel(axes[0], df, vecs, "pgd", with_stream=args.streamlines)
    draw_panel(axes[1], df, vecs, "square", with_stream=args.streamlines)
    set_shared_limits(axes, df)
    handles = [
        Line2D([0], [0], color=CLASS_COLORS[i], lw=4, label=f"{i}: {CLASS_NAMES[i]}")
        for i in range(10)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=True)
    fig.suptitle(
        "Class-colored PGD and Square transport on identical images",
        fontsize=12,
    )
    fig.savefig(out_dir / f"{stem}_side_by_side.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_side_by_side.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / f"{stem}_metadata.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "layer": layer,
                "selected_counts": counts,
                "n_selected": len(selected),
                "common_success_only": bool(args.common_success_only),
                "streamlines": bool(args.streamlines),
                "coordinate_mode": args.coordinate_mode,
                "n_plotted_images": int(n_plot_images),
                "common_success_image_ords": sorted(int(i) for i in common_success_image_ords) if common_success_image_ords is not None else None,
                "n_success": df.groupby("attack").apply(lambda g: int(g.groupby("run_id").final_success.max().sum())).to_dict(),
                "pca_explained": [float(explained[0]), float(explained[1])],
                "note": "Shared PCA basis fitted on the requested successful representation vectors from the plotted runs. Non-crossing arrows use source class color; final crossing arrows use adversarial predicted class color.",
            },
            f,
            indent=2,
        )
    print(f"[SAVED] {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/pgd_square_class_colored_flow")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer-group", default="penultimate", choices=["hidden", "penultimate", "logits"])
    p.add_argument("--per-class", type=int, default=10)
    p.add_argument("--class-filter", type=int, default=None, choices=list(range(10)))
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--pgd-steps", type=int, default=20)
    p.add_argument("--pgd-step-size", type=float, default=0.0)
    p.add_argument("--square-steps", type=int, default=120)
    p.add_argument("--square-min-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument(
        "--coordinate-mode",
        default="local_steps",
        choices=["local_steps", "displacement"],
        help="local_steps uses cumulative projected h_t-h_{t-1}; displacement uses projected h_t-h_0.",
    )
    p.add_argument(
        "--common-success-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot only images where both PGD and Square reach adversarial success.",
    )
    p.add_argument(
        "--streamlines",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overlay smoothed vector-field streamlines. Disabled by default for raw-vector diagnostics.",
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
