#!/usr/bin/env python3
"""Plot saved PGD/Square local-step trajectories on an arbitrary PCA PC pair."""

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


def fit_pca(X: np.ndarray, n_components: int):
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    ev = (s * s) / max(float(np.sum(s * s)), 1e-12)
    return mean.reshape(-1).astype(np.float32), vt[:n_components].astype(np.float32), ev[:n_components]


def load_saved(path: Path):
    z = np.load(path, allow_pickle=False)
    feat = z["feature_displacements"].astype(np.float32)
    meta = json.loads(str(z["meta_json"]))
    return feat, meta


def project_local_steps(feat: np.ndarray, meta: list[dict], pc_x: int, pc_y: int):
    max_pc = max(pc_x, pc_y)
    meta_df = pd.DataFrame(meta)
    success_runs = {rid for rid, g in meta_df.groupby("run_id") if int(g.final_success.max()) == 1}
    run_to_indices: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        run_to_indices.setdefault(m["run_id"], []).append(i)

    local_vecs = []
    for rid, idxs in run_to_indices.items():
        if rid not in success_runs:
            continue
        idxs = sorted(idxs, key=lambda i: meta[i]["step"])
        for a, b in zip(idxs[:-1], idxs[1:]):
            local_vecs.append(feat[b] - feat[a])
    local_vecs = np.stack(local_vecs).astype(np.float32)
    keep = np.linalg.norm(local_vecs, axis=1) > 1e-12
    mean, basis, explained = fit_pca(local_vecs[keep], max_pc)

    coords_by_index = {}
    for _rid, idxs in run_to_indices.items():
        idxs = sorted(idxs, key=lambda i: meta[i]["step"])
        z = np.zeros(max_pc, dtype=np.float32)
        coords_by_index[idxs[0]] = z.copy()
        for a, b in zip(idxs[:-1], idxs[1:]):
            step_coord = (feat[b] - feat[a] - mean) @ basis.T
            z = z + step_coord.astype(np.float32)
            coords_by_index[b] = z.copy()

    rows = []
    for i, m in enumerate(meta):
        r = dict(m)
        r["pcx"] = float(coords_by_index[i][pc_x - 1])
        r["pcy"] = float(coords_by_index[i][pc_y - 1])
        rows.append(r)
    return pd.DataFrame(rows), explained


def add_step_vectors(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df.sort_values(["attack", "run_id", "step"]).groupby(["attack", "run_id"], sort=False):
        g = g.copy()
        g["next_pred"] = g["pred"].shift(-1)
        g["dx"] = g["pcx"].shift(-1) - g["pcx"]
        g["dy"] = g["pcy"].shift(-1) - g["pcy"]
        rows.append(g.iloc[:-1])
    out = pd.concat(rows, ignore_index=True).dropna(subset=["pcx", "pcy", "dx", "dy"])
    out["arrow_class"] = out["label"].astype(int)
    crossing = (out["step_success"] == 0) & (out["next_pred"].notna()) & (out["next_pred"].astype(int) != out["label"].astype(int))
    out.loc[crossing, "arrow_class"] = out.loc[crossing, "next_pred"].astype(int)
    out["is_crossing_arrow"] = crossing.astype(int)
    return out


def draw_segments(ax, vecs: pd.DataFrame, alpha: float):
    for cls, g in vecs.groupby("arrow_class"):
        color = CLASS_COLORS[int(cls)]
        normal = g[g.is_crossing_arrow == 0]
        cross = g[g.is_crossing_arrow == 1]
        if not normal.empty:
            ax.plot(
                np.column_stack([normal.pcx, normal.pcx + normal.dx]).T,
                np.column_stack([normal.pcy, normal.pcy + normal.dy]).T,
                color=color,
                alpha=alpha,
                lw=0.7,
            )
        if not cross.empty:
            ax.quiver(
                cross.pcx,
                cross.pcy,
                cross.dx,
                cross.dy,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=color,
                alpha=alpha,
                width=0.002,
            )


def set_shared_limits(axes, df: pd.DataFrame):
    x = df.pcx.to_numpy(float)
    y = df.pcy.to_numpy(float)
    xlo, xhi = np.nanmin(x), np.nanmax(x)
    ylo, yhi = np.nanmin(y), np.nanmax(y)
    pad_x = 0.08 * max(xhi - xlo, 1e-6)
    pad_y = 0.08 * max(yhi - ylo, 1e-6)
    for ax in axes:
        ax.set_xlim(xlo - pad_x, xhi + pad_x)
        ax.set_ylim(ylo - pad_y, yhi + pad_y)


def draw_panel(ax, df: pd.DataFrame, vecs: pd.DataFrame, attack: str, pc_x: int, pc_y: int):
    d = df[(df.attack == attack) & (df.final_success == 1)].copy()
    v = vecs[(vecs.attack == attack) & (vecs.final_success == 1)].copy()
    for _rid, g in d.groupby("run_id"):
        ax.plot(g.pcx, g.pcy, color="#94a3b8", alpha=0.10, lw=0.7, zorder=1)
    draw_segments(ax, v, alpha=0.72)
    ax.scatter([0], [0], color="black", marker="o", s=36, zorder=5)
    ax.axhline(0, color="black", lw=0.8, alpha=0.18)
    ax.axvline(0, color="black", lw=0.8, alpha=0.18)
    ax.grid(alpha=0.16)
    ax.set_title(f"{attack.upper()} successful trajectories ({d.run_id.nunique()})")
    ax.set_xlabel(f"shared local-transport PC{pc_x}")
    ax.set_ylabel(f"shared local-transport PC{pc_y}")


def run(args: argparse.Namespace):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat, meta = load_saved(Path(args.feature_npz))
    df, explained = project_local_steps(feat, meta, args.pc_x, args.pc_y)
    vecs = add_step_vectors(df)

    stem = Path(args.feature_npz).stem.replace("_feature_displacements", "")
    stem = f"{stem}_localsteps_pc{args.pc_x}_pc{args.pc_y}_raw_vectors_n{df.image_ord.nunique()}"
    df.to_csv(out_dir / f"{stem}_timeseries.csv", index=False)
    vecs.to_csv(out_dir / f"{stem}_step_vectors.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.2), constrained_layout=True)
    draw_panel(axes[0], df, vecs, "pgd", args.pc_x, args.pc_y)
    draw_panel(axes[1], df, vecs, "square", args.pc_x, args.pc_y)
    set_shared_limits(axes, df)
    handles = [Line2D([0], [0], color=CLASS_COLORS[i], lw=4, label=f"{i}: {CLASS_NAMES[i]}") for i in range(10)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=True)
    fig.suptitle(f"Class-colored PGD and Square local-step transport: PC{args.pc_x} vs PC{args.pc_y}", fontsize=12)
    fig.savefig(out_dir / f"{stem}_side_by_side.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}_side_by_side.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / f"{stem}_metadata.json", "w") as f:
        json.dump(
            {
                "feature_npz": args.feature_npz,
                "pc_x": args.pc_x,
                "pc_y": args.pc_y,
                "n_plotted_images": int(df.image_ord.nunique()),
                "n_success": df.groupby("attack").apply(lambda g: int(g.groupby("run_id").final_success.max().sum())).to_dict(),
                "pca_explained_first_components": [float(x) for x in explained],
                "note": "PCA fitted on high-dimensional local steps reconstructed from saved h(x_t)-h(x_0) displacements.",
            },
            f,
            indent=2,
        )
    print(f"[SAVED] {out_dir / (stem + '_side_by_side.png')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-npz", required=True)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/pgd_square_class_colored_flow")
    p.add_argument("--pc-x", type=int, default=1)
    p.add_argument("--pc-y", type=int, default=3)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
