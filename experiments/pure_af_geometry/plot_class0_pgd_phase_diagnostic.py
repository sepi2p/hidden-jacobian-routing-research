#!/usr/bin/env python3
"""Build staged class-0 attack phase-portrait diagnostics.

This script intentionally avoids pure-flow / GA coordinates. It:
1. selects clean-correct CIFAR-10 class-0 images,
2. runs attack trajectories,
3. fits a 2D PCA coordinate system on attack representation displacements,
4. constructs local step vectors in that PC plane,
5. saves three increasingly processed plots:
   raw trajectories + local step vectors, smoothed vector field, streamlines.
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


def true_prob(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1).gather(1, y.view(-1, 1)).squeeze(1)


def pgd_trajectory(wrapper, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int, step_size: float, stop_on_success: bool = False):
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
        if stop_on_success:
            with torch.no_grad():
                pred = int(wrapper(x_adv).argmax(1).item())
            if pred != int(y.item()):
                break
    return states


def square_size(step: int, max_steps: int, image_size: int, min_square: int) -> int:
    frac = 1.0 - (step / max(max_steps, 1))
    side = int(round(image_size * (0.12 + 0.28 * frac)))
    return max(min_square, min(image_size, side))


def square_trajectory(wrapper, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int, seed: int, min_square: int, stop_on_success: bool = False):
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
        if stop_on_success:
            with torch.no_grad():
                pred = int(wrapper(x_adv).argmax(1).item())
            if pred != int(y.item()):
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


def select_class_clean_correct(dataset, wrapper, target_class: int, n: int, device):
    selected = []
    for idx in range(len(dataset)):
        x_cpu, y0 = dataset[idx]
        if int(y0) != target_class:
            continue
        x = x_cpu.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(wrapper(x).argmax(1).item())
        if pred == target_class:
            selected.append(idx)
            if len(selected) >= n:
                break
    return selected


def fit_pca_2d(X: np.ndarray):
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    explained = (s * s) / max(float(np.sum(s * s)), 1e-12)
    basis = vt[:2].astype(np.float32)
    return mean.reshape(-1).astype(np.float32), basis, explained[:2]


def add_step_vectors(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df.sort_values(["run_id", "step"]).groupby("run_id", sort=False):
        g = g.copy()
        g["dx"] = g["pc1"].shift(-1) - g["pc1"]
        g["dy"] = g["pc2"].shift(-1) - g["pc2"]
        rows.append(g.iloc[:-1])
    return pd.concat(rows, ignore_index=True).dropna(subset=["pc1", "pc2", "dx", "dy"])


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
    u = np.zeros(len(flat), dtype=np.float64)
    v = np.zeros(len(flat), dtype=np.float64)
    wsum = np.zeros(len(flat), dtype=np.float64)
    chunk = 512
    for start in range(0, len(flat), chunk):
        stop = start + chunk
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


def plot_base_paths(ax, df: pd.DataFrame, color: str = "#2563eb", max_paths: int = 100):
    rng = np.random.default_rng(0)
    runs = df.run_id.drop_duplicates().to_numpy()
    if len(runs) > max_paths:
        runs = rng.choice(runs, size=max_paths, replace=False)
    for run_id in runs:
        g = df[df.run_id == run_id].sort_values("step")
        ax.plot(g.pc1, g.pc2, color=color, alpha=0.14, lw=0.9)
    mean_path = []
    for step, g in df.groupby("step"):
        mean_path.append((step, float(g.pc1.mean()), float(g.pc2.mean())))
    mp = pd.DataFrame(mean_path, columns=["step", "pc1", "pc2"]).sort_values("step")
    ax.plot(mp.pc1, mp.pc2, color=color, lw=3.0, label="mean PGD path")
    ax.scatter(mp.pc1.iloc[0], mp.pc2.iloc[0], color=color, marker="o", s=45, label="start")
    ax.scatter(mp.pc1.iloc[-1], mp.pc2.iloc[-1], color=color, marker="^", s=70, label="end")
    ax.axhline(0, color="black", lw=0.8, alpha=0.2)
    ax.axvline(0, color="black", lw=0.8, alpha=0.2)
    ax.grid(alpha=0.18)
    ax.set_xlabel("transport PC1")
    ax.set_ylabel("transport PC2")


def save_raw_plot(df: pd.DataFrame, vecs: pd.DataFrame, out_dir: Path, stem: str):
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    plot_base_paths(ax, df)
    q = vecs.iloc[:: max(len(vecs) // 500, 1)]
    ax.quiver(q.pc1, q.pc2, q.dx, q.dy, angles="xy", scale_units="xy", scale=1.0, color="#1d4ed8", alpha=0.28, width=0.002)
    ax.set_title("Class-0 successful attack: projected steps and local vectors")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_01_raw_vectors.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_01_raw_vectors.pdf", bbox_inches="tight")
    plt.close(fig)


def save_smoothed_plot(df: pd.DataFrame, X, Y, U, V, W, speed, out_dir: Path, stem: str):
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    density = np.log1p(W.filled(0))
    ax.imshow(density, extent=[X.min(), X.max(), Y.min(), Y.max()], origin="lower", cmap="Greys", alpha=0.22, aspect="auto")
    plot_base_paths(ax, df)
    skip = max(X.shape[0] // 18, 1)
    ax.quiver(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        U[::skip, ::skip],
        V[::skip, ::skip],
        speed[::skip, ::skip],
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        alpha=0.86,
    )
    ax.set_title("Class-0 successful attack: smoothed local vector field")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_02_smoothed_field.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_02_smoothed_field.pdf", bbox_inches="tight")
    plt.close(fig)


def save_streamline_plot(df: pd.DataFrame, X, Y, U, V, W, speed, out_dir: Path, stem: str, stream_density: float):
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    density = np.log1p(W.filled(0))
    ax.imshow(density, extent=[X.min(), X.max(), Y.min(), Y.max()], origin="lower", cmap="Greys", alpha=0.22, aspect="auto")
    ax.streamplot(X, Y, U, V, color=speed, cmap="viridis", density=stream_density, linewidth=1.25, arrowsize=1.25, minlength=0.08)
    plot_base_paths(ax, df)
    ax.set_title("Class-0 successful attack: streamline phase portrait")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_03_streamlines.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_03_streamlines.pdf", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    wrapper = load_model(args.model, device)
    layer = LAYER_MAP[args.model][args.layer_group]
    selected = select_class_clean_correct(dataset, wrapper, args.target_class, args.images, device)
    print(f"[SELECTED] class={args.target_class} clean_correct={len(selected)} model={args.model} layer={layer}", flush=True)

    eps = args.eps / 255.0
    step_size = args.step_size / 255.0 if args.step_size > 0 else eps / max(args.steps, 1)
    rows = []
    features = []
    meta = []
    for image_ord, dataset_idx in enumerate(selected):
        x_cpu, y0 = dataset[dataset_idx]
        y = torch.tensor([int(y0)], device=device)
        x = x_cpu.unsqueeze(0).to(device)
        if args.attack == "pgd":
            states = pgd_trajectory(wrapper, x, y, eps, args.steps, step_size, stop_on_success=args.stop_on_success)
        elif args.attack == "square":
            states = square_trajectory(
                wrapper,
                x,
                y,
                eps,
                args.steps,
                seed=args.seed + image_ord * 997,
                min_square=args.square_min_size,
                stop_on_success=args.stop_on_success,
            )
        else:
            raise ValueError(args.attack)
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
                    "attack": args.attack,
                    "target_class": args.target_class,
                    "dataset_idx": int(dataset_idx),
                    "image_ord": int(image_ord),
                    "run_id": f"{args.attack}_class{args.target_class}_img{image_ord}",
                    "label": int(y0),
                    "layer_group": args.layer_group,
                    "layer": layer,
                    "step": int(step),
                    "normalized_progress": float(step / max(len(states) - 1, 1)),
                    "pred": pred,
                    "step_success": int(pred != int(y0)),
                    "margin": float(margin(logits, y).item()),
                    "true_prob": float(true_prob(logits, y).item()),
                }
            )
        final_success = int(run_meta[-1]["step_success"])
        for m in run_meta:
            m["final_success"] = final_success
        features.extend(run_feats)
        meta.extend(run_meta)
        if (image_ord + 1) % 10 == 0:
            seen = {}
            for m in meta:
                seen[m["run_id"]] = m["final_success"]
            print(f"  {args.attack} {image_ord + 1}/{len(selected)} final_success={sum(seen.values())}", flush=True)

    feat = np.stack(features).astype(np.float32)
    if args.fit_local_steps:
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
            raise RuntimeError("No successful local step vectors available for PCA")
        local_vecs = np.stack(local_vecs).astype(np.float32)
        fit_mask = np.linalg.norm(local_vecs, axis=1) > 1e-12
        mean, basis, explained = fit_pca_2d(local_vecs[fit_mask])

        coords_by_index = {}
        for idxs in run_to_indices.values():
            idxs = sorted(idxs, key=lambda i: meta[i]["step"])
            z = np.zeros(2, dtype=np.float32)
            coords_by_index[idxs[0]] = z.copy()
            for a, b in zip(idxs[:-1], idxs[1:]):
                local = feat[b] - feat[a]
                step_coord = (local - mean) @ basis.T
                z = z + step_coord.astype(np.float32)
                coords_by_index[b] = z.copy()
        for i, m in enumerate(meta):
            c = coords_by_index[i]
            r = dict(m)
            r["pc1"] = float(c[0])
            r["pc2"] = float(c[1])
            rows.append(r)
        note = f"PCs are fitted on successful local {args.attack} transport vectors h(x_t)-h(x_{{t-1}}); plotted paths are cumulative sums of projected local steps."
    else:
        # Fit coordinates on nonzero cumulative displacement states, then project all states.
        fit_mask = np.linalg.norm(feat, axis=1) > 1e-12
        mean, basis, explained = fit_pca_2d(feat[fit_mask])
        coords = (feat - mean[None, :]) @ basis.T
        for m, c in zip(meta, coords):
            r = dict(m)
            r["pc1"] = float(c[0])
            r["pc2"] = float(c[1])
            rows.append(r)
        note = f"PCs are fitted on class-0 {args.attack} cumulative representation displacements; no pure-flow or GA basis is used."

    df = pd.DataFrame(rows)
    df_success = df[df.final_success == 1].copy()
    vecs = add_step_vectors(df_success)
    stem_suffix = "_localsteps" if args.fit_local_steps else ""
    stem = f"class{args.target_class}_{args.attack}_{args.model}_{args.layer_group}{stem_suffix}"
    df.to_csv(out_dir / f"{stem}_timeseries.csv", index=False)
    vecs.to_csv(out_dir / f"{stem}_step_vectors.csv", index=False)

    save_raw_plot(df_success, vecs, out_dir, stem)
    X, Y, U, V, W, speed = smooth_field(vecs, args.grid_n, args.bandwidth_scale, args.min_weight_percentile)
    pd.DataFrame(
        {
            "pc1": X.ravel(),
            "pc2": Y.ravel(),
            "u": np.asarray(U.filled(np.nan)).ravel(),
            "v": np.asarray(V.filled(np.nan)).ravel(),
            "weight": np.asarray(W.filled(np.nan)).ravel(),
            "speed": np.asarray(speed.filled(np.nan)).ravel(),
        }
    ).to_csv(out_dir / f"{stem}_smoothed_field.csv", index=False)
    save_smoothed_plot(df_success, X, Y, U, V, W, speed, out_dir, stem)
    save_streamline_plot(df_success, X, Y, U, V, W, speed, out_dir, stem, args.stream_density)

    with open(out_dir / f"{stem}_metadata.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "layer": layer,
                "n_selected": len(selected),
                "n_success": int(df.groupby("run_id").final_success.max().sum()),
                "mean_recorded_steps": float(df.groupby("run_id").step.max().mean()),
                "max_recorded_steps": int(df.groupby("run_id").step.max().max()),
                "pca_explained": [float(explained[0]), float(explained[1])],
                "fit_local_steps": bool(args.fit_local_steps),
                "note": note,
            },
            f,
            indent=2,
        )
    print(f"[SAVED] {out_dir}", flush=True)
    print(f"[SUCCESS] {int(df.groupby('run_id').final_success.max().sum())}/{len(selected)}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/class0_pgd_phase_diagnostic")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--attack", default="pgd", choices=["pgd", "square"])
    p.add_argument("--layer-group", default="penultimate", choices=["hidden", "penultimate", "logits"])
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--square-min-size", type=int, default=2)
    p.add_argument("--stop-on-success", action="store_true")
    p.add_argument("--fit-local-steps", action="store_true")
    p.add_argument("--step-size", type=float, default=0.0)
    p.add_argument("--grid-n", type=int, default=60)
    p.add_argument("--bandwidth-scale", type=float, default=0.075)
    p.add_argument("--min-weight-percentile", type=float, default=18.0)
    p.add_argument("--stream-density", type=float, default=1.35)
    p.add_argument("--seed", type=int, default=31)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
