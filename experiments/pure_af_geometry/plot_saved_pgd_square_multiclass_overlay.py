#!/usr/bin/env python3
"""Overlay saved class-specific PGD/Square trajectories in one shared 2D plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


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


def fit_pca_2d(X: np.ndarray):
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    ev = (s * s) / max(float(np.sum(s * s)), 1e-12)
    return mean.reshape(-1).astype(np.float32), vt[:2].astype(np.float32), ev[:2]


def load_archives(paths: list[Path]):
    feats = []
    metas = []
    for archive_id, path in enumerate(paths):
        z = np.load(path, allow_pickle=False)
        feat = z["feature_displacements"].astype(np.float32)
        meta = json.loads(str(z["meta_json"]))
        for m in meta:
            m["archive_id"] = int(archive_id)
            m["source_archive"] = str(path)
            m["run_id"] = f"a{archive_id}_{m['run_id']}"
        feats.append(feat)
        metas.extend(meta)
    return np.concatenate(feats, axis=0), metas


def success_runs(meta: list[dict]):
    df = pd.DataFrame(meta)
    return {rid for rid, g in df.groupby("run_id") if int(g.final_success.max()) == 1}


def run_indices(meta: list[dict]):
    out: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        out.setdefault(m["run_id"], []).append(i)
    return {rid: sorted(idxs, key=lambda j: meta[j]["step"]) for rid, idxs in out.items()}


def project_local_steps(feat: np.ndarray, meta: list[dict]):
    ok = success_runs(meta)
    rid_to_idxs = run_indices(meta)
    local_vecs = []
    for rid, idxs in rid_to_idxs.items():
        if rid not in ok:
            continue
        for a, b in zip(idxs[:-1], idxs[1:]):
            local_vecs.append(feat[b] - feat[a])
    local_vecs = np.stack(local_vecs).astype(np.float32)
    keep = np.linalg.norm(local_vecs, axis=1) > 1e-12
    mean, basis, explained = fit_pca_2d(local_vecs[keep])

    coords_by_index = {}
    for _rid, idxs in rid_to_idxs.items():
        z = np.zeros(2, dtype=np.float32)
        coords_by_index[idxs[0]] = z.copy()
        for a, b in zip(idxs[:-1], idxs[1:]):
            step_coord = (feat[b] - feat[a] - mean) @ basis.T
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
    ok = success_runs(meta)
    X = np.stack([feat[i] for i, m in enumerate(meta) if m["run_id"] in ok and int(m["step"]) > 0]).astype(np.float32)
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


def add_step_vectors(df: pd.DataFrame):
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


def draw_vectors(ax, vecs: pd.DataFrame, alpha: float):
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
                lw=0.65,
            )
        if not cross.empty:
            ax.quiver(
                cross.pc1,
                cross.pc2,
                cross.dx,
                cross.dy,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=color,
                alpha=alpha,
                width=0.0018,
            )


def draw_panel(ax, df: pd.DataFrame, vecs: pd.DataFrame, attack: str):
    d = df[(df.attack == attack) & (df.final_success == 1)].copy()
    v = vecs[(vecs.attack == attack) & (vecs.final_success == 1)].copy()
    for _rid, g in d.groupby("run_id"):
        ax.plot(g.pc1, g.pc2, color="#94a3b8", alpha=0.07, lw=0.55, zorder=1)
    draw_vectors(ax, v, alpha=0.72)
    ax.scatter([0], [0], color="black", marker="o", s=34, zorder=5)
    ax.axhline(0, color="black", lw=0.8, alpha=0.18)
    ax.axvline(0, color="black", lw=0.8, alpha=0.18)
    ax.grid(alpha=0.16)
    ax.set_title(f"{attack.upper()} successful trajectories ({d.run_id.nunique()})")
    ax.set_xlabel("shared PC1")
    ax.set_ylabel("shared PC2")


def set_shared_limits(axes, df: pd.DataFrame):
    x = df.pc1.to_numpy(float)
    y = df.pc2.to_numpy(float)
    xlo, xhi = np.nanmin(x), np.nanmax(x)
    ylo, yhi = np.nanmin(y), np.nanmax(y)
    pad_x = 0.08 * max(xhi - xlo, 1e-6)
    pad_y = 0.08 * max(yhi - ylo, 1e-6)
    for ax in axes:
        ax.set_xlim(xlo - pad_x, xhi + pad_x)
        ax.set_ylim(ylo - pad_y, yhi + pad_y)


def run(args: argparse.Namespace):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat, meta = load_archives([Path(p) for p in args.feature_npz])
    if args.coordinate_mode == "local_steps":
        df, explained = project_local_steps(feat, meta)
        axis_label = "local-step transport"
    elif args.coordinate_mode == "displacement":
        df, explained = project_displacements(feat, meta)
        axis_label = "displacement from clean image"
    else:
        raise ValueError(args.coordinate_mode)
    vecs = add_step_vectors(df)

    classes = sorted(int(x) for x in df.label.unique())
    class_tag = "_".join(f"class{c}" for c in classes)
    stem = f"pgd_square_overlay_{class_tag}_{args.coordinate_mode}_n{df.image_ord.nunique()}"
    df.to_csv(out_dir / f"{stem}_timeseries.csv", index=False)
    vecs.to_csv(out_dir / f"{stem}_step_vectors.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.2), constrained_layout=True)
    draw_panel(axes[0], df, vecs, "pgd")
    draw_panel(axes[1], df, vecs, "square")
    set_shared_limits(axes, df)
    for ax in axes:
        ax.set_xlabel(f"shared {axis_label} PC1")
        ax.set_ylabel(f"shared {axis_label} PC2")
    handles = [Line2D([0], [0], color=CLASS_COLORS[i], lw=4, label=f"{i}: {CLASS_NAMES[i]}") for i in range(10)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=True)
    fig.suptitle(f"Class overlay in a shared {axis_label} plane", fontsize=12)
    fig.savefig(out_dir / f"{stem}_side_by_side.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_side_by_side.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / f"{stem}_metadata.json", "w") as f:
        run_counts = {
            f"{attack}_class{label}": int(count)
            for (attack, label), count in df.groupby(["attack", "label"])["run_id"].nunique().items()
        }
        json.dump(
            {
                "feature_npz": args.feature_npz,
                "coordinate_mode": args.coordinate_mode,
                "classes": classes,
                "n_runs_by_attack_class": run_counts,
                "pca_explained": [float(explained[0]), float(explained[1])],
                "note": "Both classes are projected into one PCA plane fitted jointly from the loaded archives.",
            },
            f,
            indent=2,
        )
    print(f"[SAVED] {out_dir / (stem + '_side_by_side.png')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-npz", nargs="+", required=True)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/pgd_square_class_colored_flow")
    p.add_argument("--coordinate-mode", choices=["local_steps", "displacement"], default="local_steps")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
