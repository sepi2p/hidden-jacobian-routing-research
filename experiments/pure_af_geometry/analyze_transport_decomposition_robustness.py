#!/usr/bin/env python3
"""Decomposition robustness checks for transport-coordinate analyses.

The main paper uses PCA-derived transport coordinates. This script evaluates
whether the success-vs-comparison separation is specific to PCA or persists
under alternative decompositions and control bases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, KernelPCA, PCA, SparsePCA, TruncatedSVD
from sklearn.metrics import roc_auc_score


KS = [5, 10, 20, 50]


def parse_csv(x: str) -> list[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def npz_vector_key(model: str, attack: str, layer: str) -> str:
    return f"vectors__{model}__{attack}__{layer}"


def clean_vector_key(model: str, layer: str) -> str:
    return f"clean__{model}__{layer}"


def sample_rows(x: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if max_n <= 0 or len(x) <= max_n:
        return x
    idx = rng.choice(len(x), size=max_n, replace=False)
    return x[idx]


def train_test_image_split(meta: pd.DataFrame, seed: int, train_frac: float):
    id_col = "image_id" if "image_id" in meta.columns else "image_ord"
    images = np.array(sorted(meta[id_col].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(images)
    n_train = max(1, int(round(len(images) * train_frac)))
    train_images = set(images[:n_train].tolist())
    train = meta[id_col].isin(train_images).to_numpy()
    return train, ~train


class LinearBasis:
    def __init__(self, mean: np.ndarray, basis: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.basis = basis.astype(np.float32)

    def score(self, x: np.ndarray, k: int) -> np.ndarray:
        kk = min(k, self.basis.shape[0])
        xc = x - self.mean
        coeff = xc @ self.basis[:kk].T
        return np.sum(coeff * coeff, axis=1) / np.clip(np.sum(xc * xc, axis=1), 1e-12, None)


class TransformBasis:
    def __init__(self, model):
        self.model = model

    def score(self, x: np.ndarray, k: int) -> np.ndarray:
        z = self.model.transform(x)
        kk = min(k, z.shape[1])
        return np.sum(z[:, :kk] * z[:, :kk], axis=1)


def fit_basis(method: str, train_x: np.ndarray, kmax: int, rng: np.random.Generator) -> LinearBasis | TransformBasis:
    if method == "pca":
        model = PCA(n_components=min(kmax, train_x.shape[0], train_x.shape[1]), random_state=0).fit(train_x)
        return LinearBasis(model.mean_, model.components_)
    if method == "svd":
        mean = np.zeros((1, train_x.shape[1]), dtype=np.float32)
        model = TruncatedSVD(n_components=min(kmax, train_x.shape[0] - 1, train_x.shape[1]), random_state=0).fit(train_x)
        return LinearBasis(mean, model.components_)
    if method == "sparse_pca":
        model = SparsePCA(
            n_components=min(kmax, train_x.shape[0] - 1, train_x.shape[1]),
            alpha=1.0,
            ridge_alpha=0.01,
            max_iter=1000,
            random_state=0,
        ).fit(train_x)
        return LinearBasis(train_x.mean(axis=0, keepdims=True), model.components_)
    if method == "fast_ica":
        n_comp = min(kmax, train_x.shape[0] - 1, train_x.shape[1])
        model = FastICA(n_components=n_comp, whiten="unit-variance", max_iter=1000, random_state=0, tol=1e-3).fit(train_x)
        return TransformBasis(model)
    if method == "kernel_pca":
        n_comp = min(kmax, train_x.shape[0] - 1)
        ref = sample_rows(train_x, min(400, len(train_x)), rng)
        d2 = np.sum((ref[:, None, :] - ref[None, :, :]) ** 2, axis=2)
        med = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / max(med, 1e-12)
        model = KernelPCA(n_components=n_comp, kernel="rbf", gamma=gamma, random_state=0).fit(train_x)
        return TransformBasis(model)
    if method == "random":
        q, _r = np.linalg.qr(rng.normal(size=(train_x.shape[1], min(kmax, train_x.shape[1]))))
        return LinearBasis(np.zeros((1, train_x.shape[1]), dtype=np.float32), q.T.astype(np.float32))
    raise ValueError(method)


def rows_for_scores(dataset: str, model: str, attack: str, layer: str, method: str, comparison: str, k: int, pos: np.ndarray, neg: np.ndarray):
    if len(pos) < 3 or len(neg) < 3:
        return None
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    score = np.r_[pos, neg]
    return {
        "dataset": dataset,
        "model": model,
        "attack_family": attack,
        "layer": layer,
        "method": method,
        "comparison": comparison,
        "k": int(k),
        "auroc": float(roc_auc_score(y, score)),
        "positive_mean_score": float(np.mean(pos)),
        "negative_mean_score": float(np.mean(neg)),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
    }


def load_clean(root: Path, model: str, layer: str, dim: int, rng: np.random.Generator, max_eval: int):
    clean_npz = root / "clean_motion_vectors.npz"
    if not clean_npz.exists():
        return np.empty((0, dim), dtype=np.float32)
    z = np.load(clean_npz)
    key = clean_vector_key(model, layer)
    if key not in z:
        return np.empty((0, dim), dtype=np.float32)
    arr = z[key].astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != dim:
        return np.empty((0, dim), dtype=np.float32)
    return sample_rows(normalize_rows(arr), max_eval, rng)


def analyze_root(root: Path, dataset_name: str, methods: list[str], args) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(root / "standardized_segment_metadata.csv")
    vectors = np.load(root / "standardized_segment_vectors.npz")
    rng = np.random.default_rng(args.seed + abs(hash(dataset_name)) % 100000)
    metric_rows = []
    basis_rows = []
    for (model, attack, layer), group in meta.groupby(["model", "attack_family", "layer"], dropna=False):
        key = npz_vector_key(model, attack, layer)
        if key not in vectors:
            continue
        arr_all = vectors[key].astype(np.float32)
        group = group.reset_index(drop=True)
        x_all = normalize_rows(arr_all[group["vector_idx"].to_numpy(int)])
        train_mask, test_mask = train_test_image_split(group, args.seed, args.train_frac)
        success = group["final_success"].to_numpy(int) == 1
        train_success = sample_rows(x_all[train_mask & success], args.max_train, rng)
        test_success = sample_rows(x_all[test_mask & success], args.max_eval, rng)
        test_failed = sample_rows(x_all[test_mask & ~success], args.max_eval, rng)
        if len(train_success) < 10 or len(test_success) < 5:
            continue
        clean = load_clean(root, model, layer, train_success.shape[1], rng, args.max_eval)
        random_neg = rng.normal(size=(min(args.max_eval, len(test_success)), train_success.shape[1])).astype(np.float32)
        random_neg = normalize_rows(random_neg)
        fit_sources = {}
        for method in methods:
            if method == "clean_pca":
                if len(clean) < 10:
                    continue
                fit_sources[method] = sample_rows(clean, args.max_train, rng)
            elif method == "failed_pca":
                failed_train = x_all[train_mask & ~success]
                if len(failed_train) < 10:
                    continue
                fit_sources[method] = sample_rows(failed_train, args.max_train, rng)
            else:
                fit_sources[method] = train_success

        for method, fit_x in fit_sources.items():
            actual_method = "pca" if method in {"clean_pca", "failed_pca"} else method
            try:
                basis = fit_basis(actual_method, fit_x, max(KS), rng)
            except Exception as exc:
                basis_rows.append({
                    "dataset": dataset_name,
                    "model": model,
                    "attack_family": attack,
                    "layer": layer,
                    "method": method,
                    "status": "fit_failed",
                    "message": str(exc),
                    "n_train": int(len(fit_x)),
                    "d": int(fit_x.shape[1]),
                })
                continue
            basis_rows.append({
                "dataset": dataset_name,
                "model": model,
                "attack_family": attack,
                "layer": layer,
                "method": method,
                "status": "ok",
                "message": "",
                "n_train": int(len(fit_x)),
                "d": int(fit_x.shape[1]),
            })
            for k in KS:
                pos = basis.score(test_success, k)
                for comparison, neg_x in [
                    ("success_vs_random", random_neg),
                    ("success_vs_failed", test_failed),
                    ("success_vs_clean", clean),
                ]:
                    if len(neg_x) == 0:
                        continue
                    row = rows_for_scores(
                        dataset_name,
                        model,
                        attack,
                        layer,
                        method,
                        comparison,
                        k,
                        pos,
                        basis.score(neg_x, k),
                    )
                    if row is not None:
                        metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(basis_rows)


def write_summary(metrics: pd.DataFrame, out_dir: Path):
    if metrics.empty:
        return
    view = metrics[(metrics["k"] == 20) & (metrics["comparison"].isin(["success_vs_failed", "success_vs_clean", "success_vs_random"]))]
    if view.empty:
        return
    summary = view.groupby(["dataset", "method", "comparison"]).agg(
        mean_auroc=("auroc", "mean"),
        median_auroc=("auroc", "median"),
        min_auroc=("auroc", "min"),
        max_auroc=("auroc", "max"),
        n_settings=("auroc", "size"),
    ).reset_index()
    summary.to_csv(out_dir / "decomposition_robustness_summary_k20.csv", index=False)


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = parse_csv(args.methods)
    all_metrics, all_basis = [], []
    for item in parse_csv(args.roots):
        name, root_s = item.split("=", 1) if "=" in item else (Path(item).name, item)
        root = Path(root_s)
        metrics, basis = analyze_root(root, name, methods, args)
        all_metrics.append(metrics)
        all_basis.append(basis)
        print(f"[ROOT] {name} metrics={len(metrics)} basis_rows={len(basis)}", flush=True)
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    basis = pd.concat(all_basis, ignore_index=True) if all_basis else pd.DataFrame()
    metrics.to_csv(out_dir / "decomposition_robustness_metrics.csv", index=False)
    basis.to_csv(out_dir / "decomposition_robustness_basis_status.csv", index=False)
    write_summary(metrics, out_dir)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    if not metrics.empty:
        view = metrics[(metrics.k == 20) & (metrics.comparison == "success_vs_clean")]
        print(view.groupby(["dataset", "method"])["auroc"].agg(["mean", "median", "min", "max", "count"]).to_string(), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/decomposition_robustness")
    p.add_argument(
        "--roots",
        default="cifar=analysis_outputs/pure_af_geometry/paper_multi_attack,imagenet=analysis_outputs/pure_af_geometry/imagenet_pgd_square_generalization_resnet18/paper_multi_attack",
    )
    p.add_argument("--methods", default="pca,svd,sparse_pca,fast_ica,kernel_pca,random,clean_pca,failed_pca")
    p.add_argument("--max-train", type=int, default=1200)
    p.add_argument("--max-eval", type=int, default=1200)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
