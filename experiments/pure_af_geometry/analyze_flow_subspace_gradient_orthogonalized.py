#!/usr/bin/env python3
"""Analyze successful-flow subspaces after removing local class-gradient components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models


LAYERS = ["clf_avgpool", "clf_layer4", "clf_logits"]
KS = [5, 10, 20, 50, 100]


def softmax_np(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64) - float(np.max(x))
    e = np.exp(z)
    return e / float(e.sum())


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def logp_field(layer: str, logits: np.ndarray, target: int, fc_weight: np.ndarray) -> np.ndarray:
    probs = softmax_np(logits)
    g_logits = -probs
    g_logits[int(target)] += 1.0
    if layer == "clf_logits":
        return g_logits
    if layer in {"clf_avgpool", "clf_layer4"}:
        return np.matmul(g_logits, fc_weight)
    raise ValueError(layer)


def orthogonalize_local(vectors: np.ndarray, grads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = normalize_rows(vectors)
    g = normalize_rows(grads)
    coeff = np.sum(v * g, axis=1, keepdims=True)
    residual = v - coeff * g
    residual_norm = np.linalg.norm(residual, axis=1)
    keep = residual_norm > 1e-12
    out = normalize_rows(residual[keep])
    diagnostics = np.column_stack([coeff[keep, 0], residual_norm[keep]])
    return out, diagnostics


def pca_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n < 2:
        return {
            "n": n, "d": d, "pc1_var": np.nan, "pc2_var": np.nan,
            "pc5_cum_var": np.nan, "pc10_cum_var": np.nan, "dim80": np.nan,
            "dim90": np.nan, "dim95": np.nan, "effective_rank": np.nan,
        }
    xc = x - x.mean(axis=0, keepdims=True)
    _u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    var = s ** 2
    total = float(var.sum())
    ratios = var / total if total > 0 else np.zeros_like(var)
    csum = np.cumsum(ratios)
    entropy = -float(np.sum(ratios[ratios > 0] * np.log(ratios[ratios > 0])))
    return {
        "n": int(n),
        "d": int(d),
        "pc1_var": float(ratios[0]) if len(ratios) else np.nan,
        "pc2_var": float(ratios[1]) if len(ratios) > 1 else np.nan,
        "pc5_cum_var": float(csum[min(4, len(csum) - 1)]) if len(csum) else np.nan,
        "pc10_cum_var": float(csum[min(9, len(csum) - 1)]) if len(csum) else np.nan,
        "dim80": int(np.searchsorted(csum, 0.80) + 1) if len(csum) else np.nan,
        "dim90": int(np.searchsorted(csum, 0.90) + 1) if len(csum) else np.nan,
        "dim95": int(np.searchsorted(csum, 0.95) + 1) if len(csum) else np.nan,
        "effective_rank": float(np.exp(entropy)),
    }


def collect_segments_and_grads(manifest: pd.DataFrame, layer: str, fc_weight: np.ndarray):
    success = []
    success_grads = []
    success_meta = []
    failed = []
    failed_grads = []
    failed_meta = []
    for _idx, row in manifest.iterrows():
        z = np.load(row["trajectory_features_npz"])
        feats = z[layer].astype(np.float64)
        logits = z["clf_logits"].astype(np.float64)
        target = int(row["target_class"])
        is_success = int(row.get("success", 0)) == 1
        for t in range(len(feats) - 1):
            seg = feats[t + 1] - feats[t]
            if np.linalg.norm(seg) <= 1e-12:
                continue
            grad = logp_field(layer, logits[t], target, fc_weight)
            item = {
                "run_name": row["run_name"],
                "target_class": target,
                "seed": int(row["seed"]),
                "segment_index": int(t),
            }
            if is_success:
                success.append(seg)
                success_grads.append(grad)
                success_meta.append(item)
            else:
                failed.append(seg)
                failed_grads.append(grad)
                failed_meta.append(item)
    d = np.load(manifest.iloc[0]["trajectory_features_npz"])[layer].shape[1]
    return {
        "success": np.stack(success) if success else np.empty((0, d)),
        "success_grads": np.stack(success_grads) if success_grads else np.empty((0, d)),
        "success_meta": success_meta,
        "failed": np.stack(failed) if failed else np.empty((0, d)),
        "failed_grads": np.stack(failed_grads) if failed_grads else np.empty((0, d)),
        "failed_meta": failed_meta,
    }


def fit_pca_basis(x_train: np.ndarray, max_k: int):
    mean = x_train.mean(axis=0, keepdims=True)
    xc = x_train - mean
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return mean, vt[:max_k]


def projection_energies(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, ks: list[int]) -> pd.DataFrame:
    xc = x - mean
    denom = np.sum(xc * xc, axis=1)
    coeff = xc @ basis.T
    rows = {}
    for k in ks:
        kk = min(k, basis.shape[0])
        rows[f"energy_k{k}"] = np.sum(coeff[:, :kk] ** 2, axis=1) / np.clip(denom, 1e-12, None)
    return pd.DataFrame(rows)


def auroc_safe(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def mwu_safe(pos: np.ndarray, neg: np.ndarray):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
    auc = float(stat / (len(pos) * len(neg)))
    return float(p), float(2 * auc - 1)


def logistic_cv(rows: pd.DataFrame, label_col: str, feature_cols: list[str], seed: int):
    y = rows[label_col].to_numpy(dtype=int)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        return {"cv_accuracy_mean": np.nan, "cv_accuracy_std": np.nan}
    n_splits = min(5, int(min(np.bincount(y))))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, rows[feature_cols].to_numpy(), y, cv=cv, scoring="accuracy")
    return {"cv_accuracy_mean": float(scores.mean()), "cv_accuracy_std": float(scores.std())}


def random_orthogonalized(n: int, grads: np.ndarray, rng: np.random.Generator):
    raw = rng.normal(size=(n, grads.shape[1]))
    grad_idx = rng.integers(0, len(grads), size=n)
    residual, _diag = orthogonalize_local(raw, grads[grad_idx])
    return residual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="analysis_outputs/pure_af_geometry/random_ga_classifier_trajectories_resnet18_c10_s5/manifest.csv")
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/gradient_orthogonalized_flow")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-multiplier", type=int, default=2)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["trajectory_features_npz"].notna()].copy()
    fc_weight = models.resnet18(pretrained=True).eval().fc.weight.detach().cpu().numpy().astype(np.float64)

    dim_rows = []
    diag_rows = []
    score_rows = []
    metric_rows = []
    logreg_rows = []

    for layer in LAYERS:
        data = collect_segments_and_grads(manifest, layer, fc_weight)
        success_raw = normalize_rows(data["success"])
        failed_raw = normalize_rows(data["failed"]) if len(data["failed"]) else np.empty((0, success_raw.shape[1]))
        success_orth, success_diag = orthogonalize_local(data["success"], data["success_grads"])
        failed_orth, failed_diag = orthogonalize_local(data["failed"], data["failed_grads"]) if len(data["failed"]) else (np.empty((0, success_raw.shape[1])), np.empty((0, 2)))

        for variant, name, x in [
            ("raw", "success_segments", success_raw),
            ("gradient_orthogonalized", "success_segments", success_orth),
            ("raw", "failed_segments", failed_raw),
            ("gradient_orthogonalized", "failed_segments", failed_orth),
        ]:
            if len(x):
                stats = pca_stats(x)
                stats.update({"layer": layer, "variant": variant, "set": name})
                dim_rows.append(stats)

        n_random = max(len(success_orth), 1)
        random_orth = random_orthogonalized(n_random, normalize_rows(data["success_grads"]), rng)
        stats = pca_stats(random_orth)
        stats.update({"layer": layer, "variant": "gradient_orthogonalized", "set": "random_unit"})
        dim_rows.append(stats)

        for label, diag in [("success_segments", success_diag), ("failed_segments", failed_diag)]:
            if len(diag):
                diag_rows.append({
                    "layer": layer,
                    "set": label,
                    "n": int(len(diag)),
                    "mean_cos_with_local_grad_before_removal": float(np.mean(diag[:, 0])),
                    "median_cos_with_local_grad_before_removal": float(np.median(diag[:, 0])),
                    "mean_residual_norm_after_removal": float(np.mean(diag[:, 1])),
                    "median_residual_norm_after_removal": float(np.median(diag[:, 1])),
                })

        idx = np.arange(len(success_orth))
        strat = np.array([m["target_class"] for m in data["success_meta"]])
        train_idx, test_idx = train_test_split(
            idx,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=strat if min(np.bincount(strat)) >= 2 else None,
        )
        x_train = success_orth[train_idx]
        x_success_test = success_orth[test_idx]
        max_k = min(max(KS), x_train.shape[0] - 1, x_train.shape[1])
        mean, basis = fit_pca_basis(x_train, max_k)
        n_random_pred = max(len(x_success_test) * args.random_multiplier, len(failed_orth), 1)
        random_x = random_orthogonalized(n_random_pred, normalize_rows(data["success_grads"]), rng)

        groups = {
            "success_heldout": x_success_test,
            "failed": failed_orth,
            "random": random_x,
        }
        group_frames = {}
        for group_name, x in groups.items():
            if len(x) == 0:
                continue
            energies = projection_energies(x, mean, basis, KS)
            energies["layer"] = layer
            energies["variant"] = "gradient_orthogonalized"
            energies["group"] = group_name
            energies["label_success_vs_random"] = 1 if group_name == "success_heldout" else (0 if group_name == "random" else -1)
            energies["label_success_vs_failed"] = 1 if group_name == "success_heldout" else (0 if group_name == "failed" else -1)
            group_frames[group_name] = energies
            score_rows.append(energies)

        for k in KS:
            col = f"energy_k{k}"
            pos = group_frames["success_heldout"][col].to_numpy()
            for neg_name in ["random", "failed"]:
                if neg_name not in group_frames:
                    continue
                neg = group_frames[neg_name][col].to_numpy()
                y = np.r_[np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]
                score = np.r_[pos, neg]
                p_value, rank_biserial = mwu_safe(pos, neg)
                metric_rows.append({
                    "layer": layer,
                    "variant": "gradient_orthogonalized",
                    "k": k,
                    "comparison": f"success_vs_{neg_name}",
                    "n_success": len(pos),
                    "n_negative": len(neg),
                    "success_mean_energy": float(np.mean(pos)),
                    "negative_mean_energy": float(np.mean(neg)) if len(neg) else np.nan,
                    "mean_difference": float(np.mean(pos) - np.mean(neg)) if len(neg) else np.nan,
                    "auroc": auroc_safe(y, score),
                    "mannwhitney_p": p_value,
                    "rank_biserial": rank_biserial,
                })

        feature_cols = [f"energy_k{k}" for k in KS]
        for comparison, negative_name, label_col in [
            ("success_vs_random", "random", "label_success_vs_random"),
            ("success_vs_failed", "failed", "label_success_vs_failed"),
        ]:
            if negative_name not in group_frames:
                continue
            rows = pd.concat([group_frames["success_heldout"], group_frames[negative_name]], ignore_index=True)
            rows = rows[rows[label_col] >= 0].copy()
            lr = logistic_cv(rows, label_col, feature_cols, args.seed)
            lr.update({"layer": layer, "variant": "gradient_orthogonalized", "comparison": comparison, "n": len(rows)})
            logreg_rows.append(lr)

    dim = pd.DataFrame(dim_rows)
    diag = pd.DataFrame(diag_rows)
    scores = pd.concat(score_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    logreg = pd.DataFrame(logreg_rows)

    paths = {
        "dimensionality": out_dir / "gradient_orthogonalized_flow_dimensionality.csv",
        "diagnostics": out_dir / "gradient_orthogonalized_flow_diagnostics.csv",
        "projection_scores": out_dir / "gradient_orthogonalized_flow_projection_scores.csv",
        "predictiveness": out_dir / "gradient_orthogonalized_flow_predictiveness_metrics.csv",
        "logistic": out_dir / "gradient_orthogonalized_flow_logistic_regression.csv",
    }
    dim.to_csv(paths["dimensionality"], index=False)
    diag.to_csv(paths["diagnostics"], index=False)
    scores.to_csv(paths["projection_scores"], index=False)
    metrics.to_csv(paths["predictiveness"], index=False)
    logreg.to_csv(paths["logistic"], index=False)
    meta = {
        "manifest": args.manifest,
        "layers": LAYERS,
        "ks": KS,
        "test_size": args.test_size,
        "random_multiplier": args.random_multiplier,
        "orthogonalization": "row-wise residual v - <v, normalized grad_h log p_y> grad_h log p_y, then renormalize",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    (out_dir / "gradient_orthogonalized_flow_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("\nDIMENSIONALITY")
    print(dim.sort_values(["layer", "variant", "set"]).to_string(index=False))
    print("\nDIAGNOSTICS")
    print(diag.sort_values(["layer", "set"]).to_string(index=False))
    print("\nPREDICTIVENESS")
    print(metrics.sort_values(["layer", "comparison", "k"]).to_string(index=False))
    print("\nLOGISTIC")
    print(logreg.sort_values(["layer", "comparison"]).to_string(index=False))


if __name__ == "__main__":
    main()
