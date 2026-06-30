#!/usr/bin/env python3
"""Second-pass analysis for CIFAR training-dynamics transport artifacts.

This script consumes the full feature-state shards produced by
run_cifar_training_dynamics_transport.py. It does not rerun training or attacks.

The analysis asks whether adversarial transport structure changes as the model
learns, and whether PGD/Square bases transfer across optimizers at each
checkpoint.
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.run_cifar_training_dynamics_transport import (  # noqa: E402
    LAYERS,
    normalize_rows,
    pca_basis,
    projection_energy,
    transport_vectors,
)


def parse_csv(s: str, typ=str):
    return [typ(x.strip()) for x in s.split(",") if x.strip()]


def acc_stage(acc: float) -> str:
    if acc < 0.15:
        return "init_or_random"
    if acc < 0.40:
        return "early"
    if acc < 0.70:
        return "middle"
    if acc < 0.85:
        return "late"
    return "mature"


def safe_auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    if np.nanstd(s) <= 1e-12:
        return np.nan
    return float(roc_auc_score(y, s))


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


def load_clean_vectors(base: Path, seed: int, tag: str, layer: str) -> np.ndarray:
    npz_path = base / "clean_motion" / f"seed{seed}" / tag / "clean_motion_vectors.npz"
    key = f"vectors__{layer}"
    if not npz_path.exists():
        return np.empty((0, 1), dtype=np.float32)
    npz = np.load(npz_path, allow_pickle=False)
    if key not in npz.files:
        return np.empty((0, 1), dtype=np.float32)
    return normalize_rows(npz[key].astype(np.float32))


def load_attack_layer_vectors(meta_path: Path, layer: str):
    npz_path = meta_path.with_name(meta_path.name.replace("meta_", "states_").replace(".csv", ".npz"))
    if not npz_path.exists():
        return None
    meta = pd.read_csv(meta_path)
    if meta.empty:
        return None
    npz = np.load(npz_path, allow_pickle=False)
    rows, x = transport_vectors(meta, npz, layer)
    if rows.empty or len(x) == 0:
        return None
    rows = rows.copy()
    rows["stage"] = rows["checkpoint_acc"].map(acc_stage)
    return rows, x.astype(np.float32)


def split_success_vectors(rows: pd.DataFrame, x: np.ndarray, rng: np.random.Generator):
    success = rows.final_success.to_numpy(int) == 1
    ids = np.array(sorted(rows.dataset_idx.unique()))
    rng.shuffle(ids)
    train_ids = set(ids[: max(1, int(0.6 * len(ids)))])
    is_train_id = rows.dataset_idx.isin(train_ids).to_numpy()
    train_success = is_train_id & success
    test_success = (~is_train_id) & success
    test_failed = (~is_train_id) & (~success)
    return train_success, test_success, test_failed


def analyze(args):
    base = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ks = parse_csv(args.ks, int)
    max_k = max(ks)

    # Cache vectors by (seed, tag, attack, layer) so cross-basis transfer can
    # compare PGD and Square for the exact same checkpoint.
    cache = {}
    basis_cache = {}
    projection_rows = []
    cross_rows = []
    dim_rows = []
    signature_rows = []

    meta_paths = sorted((base / "trajectory_shards").glob("seed*/*/*/meta_*.csv"))
    for meta_path in meta_paths:
        for layer in LAYERS:
            loaded = load_attack_layer_vectors(meta_path, layer)
            if loaded is None:
                continue
            rows, x = loaded
            ck = rows.iloc[0]
            key = (int(ck.seed), str(ck.tag), str(ck.attack), layer)
            cache[key] = (rows, x)
            train_success, test_success, test_failed = split_success_vectors(rows, x, rng)
            if train_success.sum() < args.min_train_success or test_success.sum() < args.min_test_success:
                continue
            mean, basis, ratio = pca_basis(x[train_success], max_k)
            basis_cache[key] = (mean, basis)
            stats = dim_stats(ratio)
            dim_rows.append(
                {
                    "seed": int(ck.seed),
                    "tag": str(ck.tag),
                    "epoch": int(ck.epoch),
                    "checkpoint_acc": float(ck.checkpoint_acc),
                    "stage": acc_stage(float(ck.checkpoint_acc)),
                    "attack": str(ck.attack),
                    "layer": layer,
                    "n_train_success": int(train_success.sum()),
                    "n_test_success": int(test_success.sum()),
                    "n_test_failed": int(test_failed.sum()),
                    **stats,
                }
            )

            coeff = (normalize_rows(x[train_success]) - mean) @ basis[:5].T
            en = np.mean(coeff * coeff, axis=0)
            frac = en / np.clip(en.sum(), 1e-12, None)
            signature_rows.append(
                {
                    "seed": int(ck.seed),
                    "tag": str(ck.tag),
                    "epoch": int(ck.epoch),
                    "checkpoint_acc": float(ck.checkpoint_acc),
                    "stage": acc_stage(float(ck.checkpoint_acc)),
                    "attack": str(ck.attack),
                    "layer": layer,
                    **{f"pc{i+1}_frac": float(frac[i]) for i in range(len(frac))},
                }
            )

            clean = load_clean_vectors(base, int(ck.seed), str(ck.tag), layer)
            rand = normalize_rows(rng.normal(size=(max(int(test_success.sum()), 1000), x.shape[1])).astype(np.float32))
            negatives = {"random": rand}
            if clean.shape[0] > 0 and clean.shape[1] == x.shape[1]:
                negatives["clean_motion"] = clean
            if test_failed.sum() >= args.min_negative:
                negatives["failed"] = x[test_failed]

            for k in ks:
                es = projection_energy(x[test_success], mean, basis, k)
                for comparison, neg in negatives.items():
                    eneg = projection_energy(neg, mean, basis, k)
                    projection_rows.append(
                        {
                            "seed": int(ck.seed),
                            "tag": str(ck.tag),
                            "epoch": int(ck.epoch),
                            "checkpoint_acc": float(ck.checkpoint_acc),
                            "stage": acc_stage(float(ck.checkpoint_acc)),
                            "attack": str(ck.attack),
                            "layer": layer,
                            "basis_attack": str(ck.attack),
                            "eval_attack": str(ck.attack),
                            "comparison": f"success_vs_{comparison}",
                            "k": k,
                            "auroc": safe_auroc(es, eneg),
                            "success_energy_mean": float(np.mean(es)),
                            "negative_energy_mean": float(np.mean(eneg)),
                            "n_success": int(len(es)),
                            "n_negative": int(len(eneg)),
                        }
                    )

    # Cross-optimizer basis transfer: PGD basis scores Square success and vice versa.
    for (seed, tag, basis_attack, layer), (mean, basis) in list(basis_cache.items()):
        for eval_attack in ["pgd", "square"]:
            if eval_attack == basis_attack:
                continue
            eval_key = (seed, tag, eval_attack, layer)
            if eval_key not in cache:
                continue
            rows, x = cache[eval_key]
            ck = rows.iloc[0]
            train_success, test_success, test_failed = split_success_vectors(rows, x, rng)
            if test_success.sum() < args.min_test_success:
                continue
            clean = load_clean_vectors(base, seed, tag, layer)
            rand = normalize_rows(rng.normal(size=(max(int(test_success.sum()), 1000), x.shape[1])).astype(np.float32))
            negatives = {"random": rand}
            if clean.shape[0] > 0 and clean.shape[1] == x.shape[1]:
                negatives["clean_motion"] = clean
            if test_failed.sum() >= args.min_negative:
                negatives["failed"] = x[test_failed]
            for k in ks:
                es = projection_energy(x[test_success], mean, basis, k)
                for comparison, neg in negatives.items():
                    eneg = projection_energy(neg, mean, basis, k)
                    cross_rows.append(
                        {
                            "seed": seed,
                            "tag": tag,
                            "epoch": int(ck.epoch),
                            "checkpoint_acc": float(ck.checkpoint_acc),
                            "stage": acc_stage(float(ck.checkpoint_acc)),
                            "basis_attack": basis_attack,
                            "eval_attack": eval_attack,
                            "layer": layer,
                            "comparison": f"success_vs_{comparison}",
                            "k": k,
                            "auroc": safe_auroc(es, eneg),
                            "success_energy_mean": float(np.mean(es)),
                            "negative_energy_mean": float(np.mean(eneg)),
                            "n_success": int(len(es)),
                            "n_negative": int(len(eneg)),
                        }
                    )

    dims = pd.DataFrame(dim_rows)
    projections = pd.DataFrame(projection_rows)
    crosses = pd.DataFrame(cross_rows)
    signatures = pd.DataFrame(signature_rows)
    sigsim = signature_similarity(signatures)
    stage = stage_summary(dims, projections, crosses, sigsim)

    dims.to_csv(out / "formation_dimensionality.csv", index=False)
    projections.to_csv(out / "formation_projection_metrics.csv", index=False)
    crosses.to_csv(out / "formation_cross_optimizer_basis_transfer.csv", index=False)
    signatures.to_csv(out / "formation_transport_signatures.csv", index=False)
    sigsim.to_csv(out / "formation_optimizer_signature_similarity.csv", index=False)
    stage.to_csv(out / "formation_stage_summary.csv", index=False)
    make_plots(out, dims, projections, crosses, sigsim)
    (out / "metadata.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"[DONE] wrote {out}", flush=True)


def signature_similarity(signatures: pd.DataFrame) -> pd.DataFrame:
    if signatures.empty:
        return pd.DataFrame()
    rows = []
    pc_cols = [c for c in signatures.columns if c.startswith("pc") and c.endswith("_frac")]
    for (seed, tag, layer), g in signatures.groupby(["seed", "tag", "layer"]):
        if {"pgd", "square"} - set(g.attack):
            continue
        a = g[g.attack == "pgd"].iloc[0]
        b = g[g.attack == "square"].iloc[0]
        va = a[pc_cols].to_numpy(float)
        vb = b[pc_cols].to_numpy(float)
        cos = float(np.dot(va, vb) / np.clip(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12, None))
        rows.append(
            {
                "seed": seed,
                "tag": tag,
                "epoch": int(a.epoch),
                "checkpoint_acc": float(a.checkpoint_acc),
                "stage": str(a.stage),
                "layer": layer,
                "attack_a": "pgd",
                "attack_b": "square",
                "signature_cosine": cos,
            }
        )
    return pd.DataFrame(rows)


def stage_summary(dims: pd.DataFrame, projections: pd.DataFrame, crosses: pd.DataFrame, sigsim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not dims.empty:
        g = dims.groupby(["stage", "attack", "layer"], as_index=False).agg(
            checkpoint_acc_mean=("checkpoint_acc", "mean"),
            dim80_mean=("dim80", "mean"),
            dim80_std=("dim80", "std"),
            effective_rank_mean=("effective_rank", "mean"),
            n=("dim80", "size"),
        )
        g["metric_family"] = "dimensionality"
        rows.append(g)
    if not projections.empty:
        sub = projections[(projections["k"] == 20) & (projections["comparison"].isin(["success_vs_clean_motion", "success_vs_failed", "success_vs_random"]))]
        g = sub.groupby(["stage", "attack", "layer", "comparison"], as_index=False).agg(
            checkpoint_acc_mean=("checkpoint_acc", "mean"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            success_energy_mean=("success_energy_mean", "mean"),
            negative_energy_mean=("negative_energy_mean", "mean"),
            n=("auroc", "size"),
        )
        g["metric_family"] = "within_attack_projection"
        rows.append(g)
    if not crosses.empty:
        sub = crosses[(crosses["k"] == 20) & (crosses["comparison"].isin(["success_vs_clean_motion", "success_vs_failed", "success_vs_random"]))]
        g = sub.groupby(["stage", "basis_attack", "eval_attack", "layer", "comparison"], as_index=False).agg(
            checkpoint_acc_mean=("checkpoint_acc", "mean"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            success_energy_mean=("success_energy_mean", "mean"),
            negative_energy_mean=("negative_energy_mean", "mean"),
            n=("auroc", "size"),
        )
        g["metric_family"] = "cross_optimizer_projection"
        rows.append(g)
    if not sigsim.empty:
        g = sigsim.groupby(["stage", "layer"], as_index=False).agg(
            checkpoint_acc_mean=("checkpoint_acc", "mean"),
            signature_cosine_mean=("signature_cosine", "mean"),
            signature_cosine_std=("signature_cosine", "std"),
            n=("signature_cosine", "size"),
        )
        g["metric_family"] = "optimizer_signature_similarity"
        rows.append(g)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def mean_by_acc(df: pd.DataFrame, value: str, filters: dict, layer: str):
    sub = df.copy()
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    sub = sub[sub.layer == layer]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("checkpoint_acc")[value].mean().sort_index()


def make_plots(out: Path, dims: pd.DataFrame, projections: pd.DataFrame, crosses: pd.DataFrame, sigsim: pd.DataFrame):
    layers = ["layer1", "layer2", "layer3", "layer4", "avgpool", "logits"]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    axes = axes.ravel()
    for layer in layers:
        if not dims.empty:
            y = mean_by_acc(dims[dims.attack == "pgd"], "dim80", {}, layer)
            if len(y):
                axes[0].plot(y.index, y.values, marker="o", label=layer)
        if not projections.empty:
            y = mean_by_acc(
                projections,
                "auroc",
                {"attack": "pgd", "comparison": "success_vs_clean_motion", "k": 20},
                layer,
            )
            if len(y):
                axes[1].plot(y.index, y.values, marker="o", label=layer)
            y = mean_by_acc(
                projections,
                "auroc",
                {"attack": "square", "comparison": "success_vs_clean_motion", "k": 20},
                layer,
            )
            if len(y):
                axes[2].plot(y.index, y.values, marker="o", label=layer)
        if not crosses.empty:
            y = mean_by_acc(
                crosses,
                "auroc",
                {"basis_attack": "pgd", "eval_attack": "square", "comparison": "success_vs_clean_motion", "k": 20},
                layer,
            )
            if len(y):
                axes[3].plot(y.index, y.values, marker="o", label=layer)
            y = mean_by_acc(
                crosses,
                "auroc",
                {"basis_attack": "square", "eval_attack": "pgd", "comparison": "success_vs_clean_motion", "k": 20},
                layer,
            )
            if len(y):
                axes[4].plot(y.index, y.values, marker="o", label=layer)
        if not sigsim.empty:
            y = mean_by_acc(sigsim, "signature_cosine", {}, layer)
            if len(y):
                axes[5].plot(y.index, y.values, marker="o", label=layer)

    titles = [
        "PGD success-flow dim80",
        "PGD basis: success vs clean",
        "Square basis: success vs clean",
        "PGD basis -> Square success",
        "Square basis -> PGD success",
        "PGD/Square signature cosine",
    ]
    ylabels = ["dim80", "AUROC", "AUROC", "AUROC", "AUROC", "cosine"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title)
        ax.set_xlabel("checkpoint clean accuracy")
        ax.set_ylabel(ylabel)
        if ylabel in {"AUROC", "cosine"}:
            ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.18)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.savefig(out / "formation_learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Compact stage plot for manuscript/notes.
    if projections.empty:
        return
    focus_layers = ["layer2", "layer3", "avgpool", "logits"]
    stage_order = ["init_or_random", "early", "middle", "late", "mature"]
    sub = projections[
        (projections.k == 20)
        & (projections.comparison == "success_vs_clean_motion")
        & (projections.layer.isin(focus_layers))
    ].copy()
    if sub.empty:
        return
    sub["stage"] = pd.Categorical(sub["stage"], stage_order, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True, sharey=True)
    for ax, attack in zip(axes, ["pgd", "square"]):
        aa = sub[sub.attack == attack]
        for layer, g in aa.groupby("layer"):
            y = g.groupby("stage", observed=False).auroc.mean().reindex(stage_order)
            ax.plot(range(len(stage_order)), y.values, marker="o", label=layer)
        ax.set_title(f"{attack.upper()} success vs clean")
        ax.set_xticks(range(len(stage_order)), stage_order, rotation=25, ha="right")
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("AUROC")
    axes[1].legend(frameon=False, loc="lower right")
    fig.savefig(out / "formation_stage_success_clean.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/cifar_training_dynamics_transport_v1")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_training_dynamics_transport_v1/formation_analysis")
    p.add_argument("--ks", default="5,10,20,50")
    p.add_argument("--min-train-success", type=int, default=12)
    p.add_argument("--min-test-success", type=int, default=6)
    p.add_argument("--min-negative", type=int, default=6)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()

