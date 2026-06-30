#!/usr/bin/env python3
"""Estimate class-conditioned flow-channel width and branch count.

Consumes saved training-dynamics trajectory shards. For each class, checkpoint,
attack, and layer, it computes:

* channel width: dim80, effective rank, residual energy outside top-k PCs;
* concentration: PC5/PC10 cumulative variance, mean pairwise cosine;
* branch count: a conservative KMeans/silhouette estimate in PCA coordinates.

This is an exploratory diagnostic for the bottleneck interpretation. Branch
counts should be read as mode estimates, not as literal topological branches.
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.run_cifar_training_dynamics_transport import (  # noqa: E402
    LAYERS,
    normalize_rows,
    pca_basis,
    transport_vectors,
)


def parse_csv(s: str, typ=str):
    return [typ(x.strip()) for x in s.split(",") if x.strip()]


def dim_stats(ratio: np.ndarray):
    csum = np.cumsum(ratio)
    ent = -float(np.sum(ratio[ratio > 0] * np.log(ratio[ratio > 0]))) if len(ratio) else np.nan
    return {
        "pc1_var": float(ratio[0]) if len(ratio) else np.nan,
        "pc5_cum_var": float(csum[min(4, len(csum) - 1)]) if len(csum) else np.nan,
        "pc10_cum_var": float(csum[min(9, len(csum) - 1)]) if len(csum) else np.nan,
        "dim80": int(np.searchsorted(csum, 0.8) + 1) if len(csum) else np.nan,
        "dim90": int(np.searchsorted(csum, 0.9) + 1) if len(csum) else np.nan,
        "effective_rank": float(np.exp(ent)) if len(csum) else np.nan,
    }


def estimate_branches(x: np.ndarray, basis: np.ndarray, mean: np.ndarray, args):
    if len(x) < args.min_vectors_for_branch:
        return np.nan, np.nan, "{}"
    rng = np.random.default_rng(args.seed)
    if len(x) > args.max_branch_vectors:
        idx = rng.choice(len(x), args.max_branch_vectors, replace=False)
        x = x[idx]
    z = (normalize_rows(x.astype(np.float32)) - mean) @ basis[: min(args.branch_pcs, basis.shape[0])].T
    if len(z) < args.min_vectors_for_branch or z.shape[1] < 2:
        return np.nan, np.nan, "{}"
    best_k = 1
    best_score = -1.0
    scores = {}
    max_k = min(args.max_branches, len(z) - 1)
    for k in range(2, max_k + 1):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(z)
            score = float(silhouette_score(z, km.labels_))
        except Exception:
            continue
        scores[str(k)] = score
        if score > best_score:
            best_score = score
            best_k = k
    if best_score < args.silhouette_threshold:
        best_k = 1
    return int(best_k), float(best_score), json.dumps(scores, sort_keys=True)


def mean_pairwise_cosine(x: np.ndarray, args) -> float:
    if len(x) < 2:
        return np.nan
    rng = np.random.default_rng(args.seed)
    x = normalize_rows(x.astype(np.float32))
    if len(x) > args.max_pairwise_vectors:
        x = x[rng.choice(len(x), args.max_pairwise_vectors, replace=False)]
    sims = x @ x.T
    tri = sims[np.triu_indices(len(x), k=1)]
    return float(np.mean(tri)) if len(tri) else np.nan


def residual_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> float:
    x = normalize_rows(x.astype(np.float32))
    xc = x - mean
    kk = min(k, basis.shape[0])
    coeff = xc @ basis[:kk].T
    proj = coeff @ basis[:kk]
    resid = xc - proj
    return float(np.mean(np.sum(resid * resid, axis=1) / np.clip(np.sum(xc * xc, axis=1), 1e-12, None)))


def analyze(args):
    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    include_tags = set(parse_csv(args.include_tags)) if args.include_tags else set()
    include_attacks = set(parse_csv(args.attacks)) if args.attacks else set()
    rows_out = []

    for meta_path in sorted((inp / "trajectory_shards").glob("seed*/*/*/meta_*.csv")):
        parts = meta_path.parts
        tag = parts[-3]
        attack = parts[-2]
        if include_tags and tag not in include_tags:
            continue
        if include_attacks and attack not in include_attacks:
            continue
        npz_path = meta_path.with_name(meta_path.name.replace("meta_", "states_").replace(".csv", ".npz"))
        if not npz_path.exists():
            continue
        meta = pd.read_csv(meta_path)
        if meta.empty:
            continue
        npz = np.load(npz_path, allow_pickle=False)
        ck = meta.iloc[0]
        for layer in LAYERS:
            tv = transport_vectors(meta, npz, layer)
            if tv is None:
                continue
            trows, x = tv
            if trows.empty:
                continue
            success = trows.final_success.to_numpy(int) == 1
            for label, idxs in trows[success].groupby("label").groups.items():
                idxs = np.asarray(list(idxs), dtype=int)
                if len(idxs) < args.min_vectors:
                    continue
                xx = normalize_rows(x[idxs].astype(np.float32))
                if len(xx) > args.max_vectors:
                    rng = np.random.default_rng(args.seed + int(ck.seed) * 1009 + int(label) * 917)
                    xx = xx[rng.choice(len(xx), args.max_vectors, replace=False)]
                mean, basis, ratio = pca_basis(xx, min(args.max_pcs, len(xx), xx.shape[1]))
                stats = dim_stats(ratio)
                branches, sil, scores = estimate_branches(xx, basis, mean, args)
                rows_out.append(
                    {
                        "seed": int(ck.seed),
                        "tag": str(ck.tag),
                        "epoch": int(ck.epoch),
                        "checkpoint_acc": float(ck.checkpoint_acc),
                        "attack": attack,
                        "layer": layer,
                        "label": int(label),
                        "n_vectors": int(len(xx)),
                        "mean_pairwise_cosine": mean_pairwise_cosine(xx, args),
                        "residual_energy_top5": residual_energy(xx, mean, basis, 5),
                        "residual_energy_top10": residual_energy(xx, mean, basis, 10),
                        "branch_count": branches,
                        "branch_silhouette": sil,
                        "branch_silhouette_scores": scores,
                        **stats,
                    }
                )
    df = pd.DataFrame(rows_out)
    df.to_csv(out / "class_flow_channel_metrics.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "class_flow_channel_summary.csv", index=False)
    make_plots(df, out)
    (out / "metadata.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"[DONE] wrote {out}", flush=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(["tag", "attack", "layer"], as_index=False)
        .agg(
            checkpoint_acc=("checkpoint_acc", "mean"),
            mean_dim80=("dim80", "mean"),
            mean_effective_rank=("effective_rank", "mean"),
            mean_pc5=("pc5_cum_var", "mean"),
            mean_residual_top5=("residual_energy_top5", "mean"),
            mean_pairwise_cosine=("mean_pairwise_cosine", "mean"),
            mean_branch_count=("branch_count", "mean"),
            median_branch_count=("branch_count", "median"),
            mean_branch_silhouette=("branch_silhouette", "mean"),
            n_class_seed_units=("label", "size"),
        )
        .sort_values(["checkpoint_acc", "attack", "layer"])
    )


def make_plots(df: pd.DataFrame, out: Path):
    if df.empty:
        return
    focus = df[(df.layer.isin(["layer2", "layer3", "layer4", "avgpool"]))].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    for attack, marker in [("pgd", "o"), ("square", "s")]:
        sub = focus[focus.attack == attack].groupby(["checkpoint_acc"], as_index=False).agg(
            dim80=("dim80", "mean"),
            residual=("residual_energy_top5", "mean"),
            branches=("branch_count", "mean"),
        )
        axes[0].plot(sub.checkpoint_acc, sub.dim80, marker=marker, label=attack.upper())
        axes[1].plot(sub.checkpoint_acc, sub.residual, marker=marker, label=attack.upper())
        axes[2].plot(sub.checkpoint_acc, sub.branches, marker=marker, label=attack.upper())
    axes[0].set_title("Class-conditioned width")
    axes[0].set_ylabel("mean dim80")
    axes[1].set_title("Residual outside top-5 PCs")
    axes[1].set_ylabel("mean residual energy")
    axes[2].set_title("Estimated branches")
    axes[2].set_ylabel("mean branch count")
    for ax in axes:
        ax.set_xlabel("checkpoint clean accuracy")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
    fig.savefig(out / "class_flow_channel_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/cifar_training_dynamics_transport_v1")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_class_flow_channels_v1")
    p.add_argument("--include-tags", default="init,acc15,acc25,acc40,acc55,acc70,acc82,acc90,final")
    p.add_argument("--attacks", default="pgd,square")
    p.add_argument("--min-vectors", type=int, default=40)
    p.add_argument("--max-vectors", type=int, default=3000)
    p.add_argument("--max-pcs", type=int, default=50)
    p.add_argument("--branch-pcs", type=int, default=10)
    p.add_argument("--max-branches", type=int, default=6)
    p.add_argument("--min-vectors-for-branch", type=int, default=80)
    p.add_argument("--max-branch-vectors", type=int, default=2000)
    p.add_argument("--silhouette-threshold", type=float, default=0.08)
    p.add_argument("--max-pairwise-vectors", type=int, default=1000)
    p.add_argument("--seed", type=int, default=777)
    args = p.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()

