#!/usr/bin/env python3
"""Plot untargeted trajectories in an absolute class-contrast feature plane."""

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
from matplotlib.lines import Line2D
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import load_model  # noqa: E402


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


def feature_vector(wrapper, x: torch.Tensor, layer: str):
    with torch.no_grad():
        logits, feats, _raw = wrapper.forward_with_features(x)
    if layer == "logits":
        h = logits.detach().cpu().numpy()[0].astype(np.float32)
    else:
        h = feats[layer].detach().cpu().numpy()[0].astype(np.float32)
    return int(logits.argmax(1).item()), h.reshape(-1)


def load_archive(path: Path, archive_id: int):
    z = np.load(path, allow_pickle=False)
    disp = z["feature_displacements"].astype(np.float32)
    meta = json.loads(str(z["meta_json"]))
    for m in meta:
        m["archive_id"] = int(archive_id)
        m["source_archive"] = str(path)
        m["run_id"] = f"a{archive_id}_{m['run_id']}"
    return disp, meta


def load_archives(paths: list[Path]):
    disps = []
    metas = []
    for archive_id, path in enumerate(paths):
        disp, meta = load_archive(path, archive_id)
        disps.append(disp)
        metas.extend(meta)
    return np.concatenate(disps, axis=0), metas


def extract_needed_clean_features(dataset, wrapper, layer: str, device, indices: set[int]):
    out = {}
    for idx in sorted(indices):
        x_cpu, _y = dataset[idx]
        x = x_cpu.unsqueeze(0).to(device)
        _pred, h = feature_vector(wrapper, x, layer)
        out[int(idx)] = h
    return out


def extract_clean_centroid_features(dataset, wrapper, layer: str, device, classes: tuple[int, int], max_per_class: int):
    feats = {int(c): [] for c in classes}
    for idx in range(len(dataset)):
        x_cpu, y0 = dataset[idx]
        y0 = int(y0)
        if y0 not in feats or len(feats[y0]) >= max_per_class:
            continue
        x = x_cpu.unsqueeze(0).to(device)
        pred, h = feature_vector(wrapper, x, layer)
        if pred == y0:
            feats[y0].append(h)
        if all(len(v) >= max_per_class for v in feats.values()):
            break
    return {c: np.stack(v).astype(np.float32) for c, v in feats.items()}


def build_absolute_dataframe(disp: np.ndarray, meta: list[dict], clean_by_idx: dict[int, np.ndarray]):
    rows = []
    H = []
    for i, m in enumerate(meta):
        h_abs = clean_by_idx[int(m["dataset_idx"])] + disp[i]
        H.append(h_abs.astype(np.float32))
        rows.append(dict(m))
    return pd.DataFrame(rows), np.stack(H).astype(np.float32)


def class_contrast_basis(clean_feats: dict[int, np.ndarray], traj_H: np.ndarray, class_a: int, class_b: int):
    mu_a = clean_feats[class_a].mean(axis=0)
    mu_b = clean_feats[class_b].mean(axis=0)
    midpoint = ((mu_a + mu_b) / 2.0).astype(np.float32)
    e1 = (mu_b - mu_a).astype(np.float32)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1e-12)

    clean_stack = np.concatenate([clean_feats[class_a], clean_feats[class_b]], axis=0)
    X = np.concatenate([clean_stack, traj_H], axis=0).astype(np.float32)
    R = X - midpoint[None, :]
    R = R - (R @ e1)[:, None] * e1[None, :]
    R = R - R.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(R, full_matrices=False)
    e2 = vt[0].astype(np.float32)
    explained_residual = float((s[0] * s[0]) / max(float(np.sum(s * s)), 1e-12))
    return midpoint, e1, e2, mu_a, mu_b, explained_residual


def project(H: np.ndarray, midpoint: np.ndarray, e1: np.ndarray, e2: np.ndarray):
    R = H - midpoint[None, :]
    return np.column_stack([R @ e1, R @ e2]).astype(np.float32)


def add_step_vectors(df: pd.DataFrame):
    rows = []
    for _, g in df.sort_values(["attack", "run_id", "step"]).groupby(["attack", "run_id"], sort=False):
        g = g.copy()
        g["next_pred"] = g["pred"].shift(-1)
        g["dx"] = g["z1"].shift(-1) - g["z1"]
        g["dy"] = g["z2"].shift(-1) - g["z2"]
        rows.append(g.iloc[:-1])
    out = pd.concat(rows, ignore_index=True).dropna(subset=["z1", "z2", "dx", "dy"])
    out["arrow_class"] = out["label"].astype(int)
    crossing = (out["step_success"] == 0) & (out["next_pred"].notna()) & (out["next_pred"].astype(int) != out["label"].astype(int))
    out.loc[crossing, "arrow_class"] = out.loc[crossing, "next_pred"].astype(int)
    out["is_crossing_arrow"] = crossing.astype(int)
    return out


def draw_vectors(ax, vecs: pd.DataFrame, alpha: float):
    for cls, g in vecs.groupby("arrow_class"):
        color = CLASS_COLORS[int(cls)]
        normal = g[g.is_crossing_arrow == 0]
        cross = g[g.is_crossing_arrow == 1]
        if not normal.empty:
            ax.plot(
                np.column_stack([normal.z1, normal.z1 + normal.dx]).T,
                np.column_stack([normal.z2, normal.z2 + normal.dy]).T,
                color=color,
                alpha=alpha,
                lw=0.7,
            )
        if not cross.empty:
            ax.quiver(
                cross.z1,
                cross.z2,
                cross.dx,
                cross.dy,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=color,
                alpha=alpha,
                width=0.0018,
            )


def draw_panel(ax, df: pd.DataFrame, vecs: pd.DataFrame, clean_proj: pd.DataFrame, attack: str, class_a: int, class_b: int):
    d = df[(df.attack == attack) & (df.final_success == 1)].copy()
    v = vecs[(vecs.attack == attack) & (vecs.final_success == 1)].copy()
    for cls in [class_a, class_b]:
        c = clean_proj[clean_proj.label == cls]
        ax.scatter(c.z1, c.z2, s=7, color=CLASS_COLORS[cls], alpha=0.08, edgecolors="none", zorder=0)
    for _rid, g in d.groupby("run_id"):
        ax.plot(g.z1, g.z2, color="#94a3b8", alpha=0.06, lw=0.55, zorder=1)
    draw_vectors(ax, v, alpha=0.72)
    cent = clean_proj.groupby("label")[["z1", "z2"]].mean()
    ax.scatter(cent.loc[class_a, "z1"], cent.loc[class_a, "z2"], color=CLASS_COLORS[class_a], marker="o", s=90, edgecolor="black", linewidth=0.8, zorder=5)
    ax.scatter(cent.loc[class_b, "z1"], cent.loc[class_b, "z2"], color=CLASS_COLORS[class_b], marker="o", s=90, edgecolor="black", linewidth=0.8, zorder=5)
    ax.text(cent.loc[class_a, "z1"], cent.loc[class_a, "z2"], f" {class_a}", fontsize=9, weight="bold")
    ax.text(cent.loc[class_b, "z1"], cent.loc[class_b, "z2"], f" {class_b}", fontsize=9, weight="bold")
    ax.axhline(0, color="black", lw=0.8, alpha=0.18)
    ax.axvline(0, color="black", lw=0.8, alpha=0.18)
    ax.grid(alpha=0.16)
    ax.set_title(f"{attack.upper()} untargeted successful trajectories ({d.run_id.nunique()})")
    ax.set_xlabel(f"class contrast axis: {class_a} -> {class_b}")
    ax.set_ylabel("orthogonal residual PC1")


def set_shared_limits(axes, df: pd.DataFrame, clean_proj: pd.DataFrame):
    x = np.concatenate([df.z1.to_numpy(float), clean_proj.z1.to_numpy(float)])
    y = np.concatenate([df.z2.to_numpy(float), clean_proj.z2.to_numpy(float)])
    xlo, xhi = np.nanpercentile(x, [0.5, 99.5])
    ylo, yhi = np.nanpercentile(y, [0.5, 99.5])
    pad_x = 0.10 * max(xhi - xlo, 1e-6)
    pad_y = 0.10 * max(yhi - ylo, 1e-6)
    for ax in axes:
        ax.set_xlim(xlo - pad_x, xhi + pad_x)
        ax.set_ylim(ylo - pad_y, yhi + pad_y)


def run(args: argparse.Namespace):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    wrapper = load_model(args.model, device)
    layer = LAYER_MAP[args.model][args.layer_group]

    disp, meta = load_archives([Path(p) for p in args.feature_npz])
    needed_indices = {int(m["dataset_idx"]) for m in meta}
    clean_by_idx = extract_needed_clean_features(dataset, wrapper, layer, device, needed_indices)
    df, H = build_absolute_dataframe(disp, meta, clean_by_idx)

    clean_feats = extract_clean_centroid_features(
        dataset, wrapper, layer, device, (args.class_a, args.class_b), args.clean_per_class
    )
    midpoint, e1, e2, mu_a, mu_b, residual_ev = class_contrast_basis(clean_feats, H, args.class_a, args.class_b)
    Z = project(H, midpoint, e1, e2)
    df["z1"] = Z[:, 0]
    df["z2"] = Z[:, 1]
    vecs = add_step_vectors(df)

    clean_rows = []
    for cls, feats in clean_feats.items():
        coords = project(feats, midpoint, e1, e2)
        for z1, z2 in coords:
            clean_rows.append({"label": int(cls), "z1": float(z1), "z2": float(z2)})
    clean_proj = pd.DataFrame(clean_rows)

    stem = f"untargeted_class_contrast_{args.model}_{args.layer_group}_class{args.class_a}_class{args.class_b}_n{df.run_id.nunique()}"
    df.to_csv(out_dir / f"{stem}_timeseries.csv", index=False)
    vecs.to_csv(out_dir / f"{stem}_step_vectors.csv", index=False)
    clean_proj.to_csv(out_dir / f"{stem}_clean_projection.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.2), constrained_layout=True)
    draw_panel(axes[0], df, vecs, clean_proj, "pgd", args.class_a, args.class_b)
    draw_panel(axes[1], df, vecs, clean_proj, "square", args.class_a, args.class_b)
    set_shared_limits(axes, df, clean_proj)
    handles = [Line2D([0], [0], color=CLASS_COLORS[i], lw=4, label=f"{i}: {CLASS_NAMES[i]}") for i in range(10)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=True)
    fig.suptitle("Untargeted trajectories in an absolute class-contrast representation plane", fontsize=12)
    fig.savefig(out_dir / f"{stem}_side_by_side.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_side_by_side.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / f"{stem}_metadata.json", "w") as f:
        json.dump(
            {
                "feature_npz": args.feature_npz,
                "model": args.model,
                "layer": layer,
                "class_a": int(args.class_a),
                "class_b": int(args.class_b),
                "clean_per_class": int(args.clean_per_class),
                "n_runs": int(df.run_id.nunique()),
                "n_runs_by_attack_class": {
                    f"{attack}_class{label}": int(count)
                    for (attack, label), count in df.groupby(["attack", "label"])["run_id"].nunique().items()
                },
                "residual_pc1_explained": residual_ev,
                "note": "x-axis is the clean centroid contrast direction mu_b - mu_a. y-axis is PC1 of residual variation after removing the class-contrast direction.",
            },
            f,
            indent=2,
        )
    print(f"[SAVED] {out_dir / (stem + '_side_by_side.png')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-npz", nargs="+", required=True)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/pgd_square_class_colored_flow")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer-group", default="penultimate", choices=["hidden", "penultimate", "logits"])
    p.add_argument("--class-a", type=int, default=0)
    p.add_argument("--class-b", type=int, default=2)
    p.add_argument("--clean-per-class", type=int, default=500)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
