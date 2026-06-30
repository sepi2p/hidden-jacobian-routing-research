#!/usr/bin/env python3
"""Residualize balanced trajectories against Jacobian/mobility null bases.

This is Step 2 of the reviewer-critical rebuild plan.  It asks whether the
balanced success-vs-failed and PGD/Square similarities survive after removing
top directions from realizable high-gain nulls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import randomized_svd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_NULL_SOURCES = [
    "jacobian_probe_all",
    "jacobian_probe_top_mobility",
    "mobility_top_walk_pgd_budget",
    "mobility_top_walk_square_budget",
]

DEFAULT_EVAL_SOURCES = [
    "pgd",
    "square",
    "random_sign_walk_pgd_budget",
    "random_sign_walk_square_budget",
    "correlated_random_walk_pgd_budget",
    "correlated_random_walk_square_budget",
    "mobility_top_walk_pgd_budget",
    "mobility_top_walk_square_budget",
    "jacobian_probe_all",
    "jacobian_probe_top_mobility",
    "pgd_predicted_gradient_step",
]


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_csv(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def safe_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    scores = np.r_[pos, neg]
    if np.std(scores) < 1e-12:
        return np.nan
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(labels, scores))


def pca_basis(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < 2:
        raise ValueError("Need at least two rows for PCA.")
    kk = min(k, x.shape[0] - 1, x.shape[1])
    if kk < 1:
        raise ValueError("PCA rank is zero.")
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    xc = x - mean
    _u, _s, vt = randomized_svd(xc, n_components=kk, random_state=seed)
    return mean, vt.astype(np.float32)


def residualize(x: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, basis.shape[0], x.shape[1])
    if kk < 1:
        return x.astype(np.float32, copy=True)
    b = basis[:kk]
    return (x - (x @ b.T) @ b).astype(np.float32)


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, basis.shape[0], x.shape[1])
    if kk < 1 or len(x) == 0:
        return np.zeros(len(x), dtype=np.float64)
    xc = x - mean
    coeff = xc @ basis[:kk].T
    return np.sum(coeff * coeff, axis=1) / np.clip(np.sum(xc * xc, axis=1), 1e-12, None)


def coeff_profile(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, basis.shape[0], x.shape[1])
    if kk < 1 or len(x) == 0:
        return np.zeros(kk, dtype=np.float64)
    xc = x - mean
    denom = np.clip(np.sum(xc * xc, axis=1, keepdims=True), 1e-12, None)
    coeff = xc @ basis[:kk].T
    profile = np.mean((coeff * coeff) / denom, axis=0)
    return profile / np.clip(profile.sum(), 1e-12, None)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    kk = min(len(a), len(b))
    aa = a[:kk]
    bb = b[:kk]
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), 1e-12))


def subspace_overlap(a: np.ndarray, b: np.ndarray, k: int) -> dict:
    kk = min(k, a.shape[0], b.shape[0])
    if kk < 1:
        return {"projection_overlap": np.nan, "mean_principal_angle_deg": np.nan, "subspace_affinity": np.nan}
    s = np.linalg.svd(a[:kk] @ b[:kk].T, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return {
        "projection_overlap": float(np.sum(s * s) / kk),
        "mean_principal_angle_deg": float(np.mean(np.degrees(np.arccos(s)))),
        "subspace_affinity": float(np.linalg.norm(s) / math.sqrt(kk)),
    }


def vector_key(model: str, source: str, layer: str) -> str:
    return f"{model}__{source}__{layer}"


class ArtifactStore:
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.rows = pd.read_csv(input_dir / "segment_metadata.csv")
        self.splits = pd.read_csv(input_dir / "image_splits.csv")
        self.arrays = np.load(input_dir / "segment_vectors.npz")
        self.split_by_image = dict(zip(self.splits["image_ord"].astype(int), self.splits["split"].astype(str)))

    def rows_for(self, model: str, source: str, layer: str) -> tuple[pd.DataFrame, np.ndarray]:
        key = vector_key(model, source, layer)
        sub = self.rows[(self.rows.model == model) & (self.rows.source == source) & (self.rows.layer == layer)].copy()
        if sub.empty or key not in self.arrays.files:
            return sub, np.zeros((0, 0), dtype=np.float32)
        x = self.arrays[key][sub["vector_idx"].to_numpy(dtype=int)]
        return sub, x

    def split_mask(self, sub: pd.DataFrame, split: str) -> np.ndarray:
        vals = sub["image_ord"].map(self.split_by_image).fillna("").to_numpy()
        return vals == split

    def train_mask(self, sub: pd.DataFrame) -> np.ndarray:
        return self.split_mask(sub, "train")

    def test_mask(self, sub: pd.DataFrame) -> np.ndarray:
        return self.split_mask(sub, "test")


def fit_source_basis(store: ArtifactStore, model: str, source: str, layer: str, k: int, seed: int, success_only: bool) -> tuple[np.ndarray, np.ndarray, int] | None:
    sub, x = store.rows_for(model, source, layer)
    if sub.empty:
        return None
    mask = store.train_mask(sub)
    if success_only:
        mask &= sub["final_success"].to_numpy(dtype=int) == 1
    if mask.sum() < max(8, min(k, x.shape[1]) + 2):
        return None
    mean, basis = pca_basis(x[mask], k, seed)
    return mean, basis, int(mask.sum())


def fit_residual_basis(
    store: ArtifactStore,
    model: str,
    source: str,
    layer: str,
    null_basis: np.ndarray | None,
    residual_k: int,
    fit_k: int,
    seed: int,
    success_only: bool,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    sub, x = store.rows_for(model, source, layer)
    if sub.empty:
        return None
    xr = residualize(x, null_basis, residual_k) if null_basis is not None and residual_k > 0 else x
    mask = store.train_mask(sub)
    if success_only:
        mask &= sub["final_success"].to_numpy(dtype=int) == 1
    if mask.sum() < max(8, min(fit_k, xr.shape[1]) + 2):
        return None
    mean, basis = pca_basis(xr[mask], fit_k, seed)
    return mean, basis, int(mask.sum())


def bootstrap_image_auc(pos_df: pd.DataFrame, neg_df: pd.DataFrame, score: str, seed: int, reps: int) -> tuple[float, float, float]:
    if pos_df.empty or neg_df.empty:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    pos_images = pos_df["image_ord"].drop_duplicates().to_numpy()
    neg_images = neg_df["image_ord"].drop_duplicates().to_numpy()
    if len(pos_images) < 2 or len(neg_images) < 2:
        return np.nan, np.nan, np.nan
    vals = []
    for _ in range(reps):
        pi = rng.choice(pos_images, len(pos_images), replace=True)
        ni = rng.choice(neg_images, len(neg_images), replace=True)
        p = pos_df[pos_df.image_ord.isin(pi)].groupby("image_ord")[score].mean().to_numpy()
        n = neg_df[neg_df.image_ord.isin(ni)].groupby("image_ord")[score].mean().to_numpy()
        vals.append(safe_auc(p, n))
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    return float(vals.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def analyze_setting(
    store: ArtifactStore,
    model: str,
    layer: str,
    basis_source: str,
    null_source: str,
    residual_k: int,
    eval_sources: list[str],
    basis_k: int,
    seed: int,
    bootstrap_reps: int,
):
    if null_source == "none" or residual_k == 0:
        null_basis = None
        null_n = 0
    else:
        null_fit = fit_source_basis(store, model, null_source, layer, max(residual_k, basis_k), seed, success_only=False)
        if null_fit is None:
            return [], [], [], []
        _null_mean, null_basis, null_n = null_fit

    success_fit = fit_residual_basis(
        store, model, basis_source, layer, null_basis, residual_k, basis_k, seed, success_only=True
    )
    if success_fit is None:
        return [], [], [], []
    mean, basis, basis_n = success_fit

    source_rows = []
    score_frames = []
    for source in eval_sources:
        sub, x = store.rows_for(model, source, layer)
        if sub.empty:
            continue
        xr = residualize(x, null_basis, residual_k) if null_basis is not None and residual_k > 0 else x
        test = store.test_mask(sub)
        if not test.any():
            continue
        scores = projection_energy(xr[test], mean, basis, basis_k)
        sub_test = sub[test].copy()
        sub_test["residual_null_source"] = null_source
        sub_test["residual_k"] = residual_k
        sub_test["basis_source"] = basis_source
        sub_test["projection_energy"] = scores
        score_frames.append(sub_test)

    if not score_frames:
        return [], [], [], []
    scores_df = pd.concat(score_frames, ignore_index=True)
    pos = scores_df[(scores_df.source == basis_source) & (scores_df.final_success == 1)]
    metric_rows = []
    ci_rows = []
    for source, neg_all in scores_df.groupby("source"):
        if source == basis_source:
            neg = neg_all[neg_all.final_success == 0]
            comparison = "success_vs_failed_same_optimizer"
        else:
            neg = neg_all
            comparison = f"{basis_source}_success_vs_{source}"
        pos_scores = pos["projection_energy"].to_numpy()
        neg_scores = neg["projection_energy"].to_numpy()
        auc = safe_auc(pos_scores, neg_scores)
        bmean, blo, bhi = bootstrap_image_auc(pos, neg, "projection_energy", seed, bootstrap_reps)
        metric_rows.append(
            {
                "model": model,
                "layer": layer,
                "basis_source": basis_source,
                "residual_null_source": null_source,
                "residual_k": residual_k,
                "basis_k": basis_k,
                "basis_train_success_segments": basis_n,
                "null_train_segments": null_n,
                "comparison_source": source,
                "comparison": comparison,
                "segment_auroc": auc,
                "pos_mean_energy": float(pos_scores.mean()) if len(pos_scores) else np.nan,
                "neg_mean_energy": float(neg_scores.mean()) if len(neg_scores) else np.nan,
                "pos_segments": int(len(pos_scores)),
                "neg_segments": int(len(neg_scores)),
            }
        )
        ci_rows.append(
            {
                "model": model,
                "layer": layer,
                "basis_source": basis_source,
                "residual_null_source": null_source,
                "residual_k": residual_k,
                "basis_k": basis_k,
                "comparison_source": source,
                "comparison": comparison,
                "image_bootstrap_auroc_mean": bmean,
                "image_bootstrap_auroc_lo": blo,
                "image_bootstrap_auroc_hi": bhi,
                "pos_images": int(pos["image_ord"].nunique()),
                "neg_images": int(neg["image_ord"].nunique()),
            }
        )

    overlap_rows = []
    for other in eval_sources:
        other_fit = fit_residual_basis(store, model, other, layer, null_basis, residual_k, basis_k, seed, success_only=other in {"pgd", "square"})
        if other_fit is None:
            continue
        _om, obasis, other_n = other_fit
        overlap_rows.append(
            {
                "model": model,
                "layer": layer,
                "basis_source": basis_source,
                "other_source": other,
                "residual_null_source": null_source,
                "residual_k": residual_k,
                "basis_k": basis_k,
                "basis_train_success_segments": basis_n,
                "other_train_segments": other_n,
                **subspace_overlap(basis, obasis, basis_k),
            }
        )

    sim_rows = []
    profiles = {}
    for source in ["pgd", "square", "mobility_top_walk_pgd_budget", "mobility_top_walk_square_budget", "random_sign_walk_pgd_budget", "random_sign_walk_square_budget"]:
        sub, x = store.rows_for(model, source, layer)
        if sub.empty:
            continue
        xr = residualize(x, null_basis, residual_k) if null_basis is not None and residual_k > 0 else x
        test = store.test_mask(sub)
        for mode, extra in {
            "all_runs": np.ones(len(sub), dtype=bool),
            "success_only": sub["final_success"].to_numpy(dtype=int) == 1,
            "failed_only": sub["final_success"].to_numpy(dtype=int) == 0,
        }.items():
            mask = test & extra
            if mask.sum() < 2:
                continue
            profiles[(source, mode)] = coeff_profile(xr[mask], mean, basis, min(5, basis_k))
    keys = sorted(profiles)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            sim_rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "basis_source": basis_source,
                    "residual_null_source": null_source,
                    "residual_k": residual_k,
                    "signature_k": min(5, basis_k),
                    "source_a": a[0],
                    "mode_a": a[1],
                    "source_b": b[0],
                    "mode_b": b[1],
                    "cosine": cosine(profiles[a], profiles[b]),
                }
            )

    return metric_rows, overlap_rows, ci_rows, sim_rows


def write_summary(out_dir: Path, metrics: pd.DataFrame, overlaps: pd.DataFrame, ci: pd.DataFrame):
    lines = [
        "# Step 2 Residual Null Summary",
        "",
        "This analysis removes top directions from Jacobian/mobility null bases and recomputes success-flow metrics.",
        "",
    ]
    if metrics.empty:
        lines.append("No metrics were produced.")
        out_dir.joinpath("residual_null_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    key = metrics[(metrics["layer"] == "layer4") & (metrics["comparison"] == "success_vs_failed_same_optimizer")]
    lines += ["## Success-vs-Failed After Residualization", ""]
    for row in key.sort_values(["basis_source", "residual_null_source", "residual_k"]).itertuples():
        lines.append(
            f"- {row.basis_source}, remove {row.residual_null_source} k={row.residual_k}: AUROC={row.segment_auroc:.3f}, "
            f"pos={row.pos_segments}, neg={row.neg_segments}"
        )

    if not overlaps.empty:
        lines += ["", "## PGD/Square Versus Null Overlaps", ""]
        for basis_source in ["pgd", "square"]:
            sub = overlaps[
                (overlaps.layer == "layer4")
                & (overlaps.basis_source == basis_source)
                & (overlaps.other_source.isin(["pgd", "square", "mobility_top_walk_pgd_budget", "mobility_top_walk_square_budget", "jacobian_probe_all", "jacobian_probe_top_mobility"]))
            ]
            for row in sub.sort_values(["residual_null_source", "residual_k", "other_source"]).itertuples():
                lines.append(
                    f"- {basis_source}, remove {row.residual_null_source} k={row.residual_k}, vs {row.other_source}: "
                    f"overlap={row.projection_overlap:.3f}, angle={row.mean_principal_angle_deg:.1f}"
                )

    if not ci.empty:
        lines += ["", "## Image-Level CI Snapshot", ""]
        sf = ci[(ci.layer == "layer4") & (ci.comparison == "success_vs_failed_same_optimizer")]
        for row in sf.sort_values(["basis_source", "residual_null_source", "residual_k"]).itertuples():
            lines.append(
                f"- {row.basis_source}, remove {row.residual_null_source} k={row.residual_k}: "
                f"image AUROC={row.image_bootstrap_auroc_mean:.3f} "
                f"[{row.image_bootstrap_auroc_lo:.3f}, {row.image_bootstrap_auroc_hi:.3f}]"
            )

    lines += [
        "",
        "## Gate",
        "",
        "Interpret manually before scaling. If success-vs-failed survives but PGD/Square overlap remains matched or exceeded by mobility/Jacobian controls, pivot wording toward high-gain hidden-Jacobian transport with residual success-conditioned structure.",
    ]
    out_dir.joinpath("residual_null_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/residual_nulls_bbb_resnet50_c200_auto"))
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layers", default="layer4")
    p.add_argument("--basis-sources", default="pgd,square")
    p.add_argument("--null-sources", default=",".join(DEFAULT_NULL_SOURCES))
    p.add_argument("--eval-sources", default=",".join(DEFAULT_EVAL_SOURCES))
    p.add_argument("--residual-ks", default="0,5,10,20")
    p.add_argument("--basis-k", type=int, default=20)
    p.add_argument("--bootstrap-reps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(args.input_dir)
    layers = parse_csv(args.layers)
    basis_sources = parse_csv(args.basis_sources)
    null_sources = ["none"] + parse_csv(args.null_sources)
    eval_sources = parse_csv(args.eval_sources)
    residual_ks = parse_int_csv(args.residual_ks)

    metric_rows = []
    overlap_rows = []
    ci_rows = []
    sim_rows = []
    for layer in layers:
        for null_source in null_sources:
            for residual_k in residual_ks:
                if null_source == "none" and residual_k != 0:
                    continue
                if null_source != "none" and residual_k == 0:
                    continue
                for basis_source in basis_sources:
                    m, o, c, s = analyze_setting(
                        store,
                        args.model,
                        layer,
                        basis_source,
                        null_source,
                        residual_k,
                        eval_sources,
                        args.basis_k,
                        args.seed,
                        args.bootstrap_reps,
                    )
                    metric_rows.extend(m)
                    overlap_rows.extend(o)
                    ci_rows.extend(c)
                    sim_rows.extend(s)

    metrics = pd.DataFrame(metric_rows)
    overlaps = pd.DataFrame(overlap_rows)
    ci = pd.DataFrame(ci_rows)
    sim = pd.DataFrame(sim_rows)
    metrics.to_csv(args.output_dir / "residual_projection_metrics.csv", index=False)
    overlaps.to_csv(args.output_dir / "residual_subspace_overlap.csv", index=False)
    ci.to_csv(args.output_dir / "residual_image_level_ci.csv", index=False)
    sim.to_csv(args.output_dir / "residual_signature_similarity.csv", index=False)
    metadata = {
        "script": "experiments/pure_af_geometry/analyze_jacobian_residual_nulls.py",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "layers": layers,
        "basis_sources": basis_sources,
        "null_sources": null_sources,
        "eval_sources": eval_sources,
        "residual_ks": residual_ks,
        "basis_k": args.basis_k,
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(args.output_dir, metrics, overlaps, ci)
    print(f"[DONE] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
