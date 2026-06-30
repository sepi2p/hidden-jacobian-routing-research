#!/usr/bin/env python3
"""Basis-level hidden-Jacobian control and residual transport diagnostics.

This script is the basis-level follow-up to ``test_mobility_vs_jacobian_gain.py``.
It asks whether the learned adversarial transport PCA basis is mostly a basis
for high-gain hidden-Jacobian motion.  It then removes a JVP-sketch basis and
tests whether any residual transport energy remains predictive.

The script is intentionally conservative:

* subspace bases are fit only on train-split images from the balanced rerun;
* vectors are row-normalized by default before PCA, so the test is about
  directions rather than raw feature speed;
* random-subspace z-scores use orthonormal random bases in the same dimension;
* candidate residual scores are recomputed from the same deterministic random
  directions used by the exact-JVP candidate sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import load_model  # noqa: E402
from experiments.hidden_jacobian_routing.test_mobility_margin_two_stage_selection import (  # noqa: E402
    parse_csv,
    parse_int_csv,
    projection_energy,
    safe_auprc,
    safe_auroc,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def pca_basis(x: np.ndarray, k: int, seed: int, normalize: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    if normalize:
        x = normalize_rows(x.astype(np.float32, copy=False))
    n, d = x.shape
    kk = min(int(k), n - 1, d)
    if kk < 1:
        raise ValueError(f"Cannot fit PCA with shape={x.shape}, k={k}")
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    xc = x - mean
    _u, _s, vt = randomized_svd(xc, n_components=kk, random_state=seed)
    return mean, vt.astype(np.float32), kk


def orth_residual_basis(basis: np.ndarray, remove_basis: np.ndarray, remove_k: int, k: int) -> np.ndarray:
    kk = min(k, basis.shape[0], basis.shape[1])
    rr = min(remove_k, remove_basis.shape[0], basis.shape[1])
    b = basis[:kk].astype(np.float32, copy=True)
    if rr > 0:
        u = remove_basis[:rr]
        b = b - (b @ u.T) @ u
    # QR over columns of B^T gives an orthonormal row basis.
    q, _r = np.linalg.qr(b.T)
    out = q[:, : min(kk, q.shape[1])].T.astype(np.float32)
    return out


def residualize(x: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, basis.shape[0], x.shape[1])
    if kk < 1:
        return x.astype(np.float32, copy=True)
    b = basis[:kk]
    return (x - (x @ b.T) @ b).astype(np.float32)


def vector_key(model: str, source: str, layer: str) -> str:
    return f"{model}__{source}__{layer}"


def feature_tensor(wrapper, x: torch.Tensor, layer: str) -> torch.Tensor:
    _logits, feats, _raw = wrapper.forward_with_features(x)
    if layer not in feats:
        raise RuntimeError(f"Layer {layer} not captured.")
    return feats[layer]


def subspace_singular_values(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, a.shape[0], b.shape[0], a.shape[1], b.shape[1])
    if kk < 1:
        return np.zeros(0, dtype=np.float64)
    s = np.linalg.svd(a[:kk] @ b[:kk].T, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def subspace_metrics(a: np.ndarray, b: np.ndarray, k: int) -> dict:
    s = subspace_singular_values(a, b, k)
    if len(s) == 0:
        return {"overlap": np.nan, "mean_angle_deg": np.nan, "max_angle_deg": np.nan, "affinity": np.nan}
    angles = np.degrees(np.arccos(s))
    return {
        "overlap": float(np.sum(s * s) / len(s)),
        "mean_angle_deg": float(np.mean(angles)),
        "max_angle_deg": float(np.max(angles)),
        "affinity": float(np.linalg.norm(s) / math.sqrt(len(s))),
    }


def random_orth_basis(rng: np.random.Generator, d: int, k: int) -> np.ndarray:
    x = rng.normal(size=(d, k)).astype(np.float32)
    q, _ = np.linalg.qr(x)
    return q[:, :k].T.astype(np.float32)


class SegmentStore:
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.rows = pd.read_csv(input_dir / "segment_metadata.csv")
        self.splits = pd.read_csv(input_dir / "image_splits.csv")
        self.arrays = np.load(input_dir / "segment_vectors.npz")
        self.split_by_image = dict(zip(self.splits.image_ord.astype(int), self.splits.split.astype(str)))

    def rows_for(self, model: str, source: str, layer: str) -> tuple[pd.DataFrame, np.ndarray]:
        key = vector_key(model, source, layer)
        sub = self.rows[(self.rows.model == model) & (self.rows.source == source) & (self.rows.layer == layer)].copy()
        if sub.empty or key not in self.arrays.files:
            return sub, np.zeros((0, 0), dtype=np.float32)
        return sub, self.arrays[key][sub.vector_idx.to_numpy(dtype=int)].astype(np.float32)

    def split_mask(self, sub: pd.DataFrame, split: str) -> np.ndarray:
        return sub.image_ord.map(self.split_by_image).fillna("").to_numpy() == split


def load_images_for_split(input_dir: Path, model: str, split_name: str, max_images: int) -> pd.DataFrame:
    outcomes = pd.read_csv(input_dir / "image_outcomes.csv")
    splits = pd.read_csv(input_dir / "image_splits.csv")
    base = outcomes[(outcomes.model == model) & (outcomes.source == "pgd")][
        ["image_ord", "dataset_idx", "label", "clean_pred", "clean_margin"]
    ].drop_duplicates()
    base = base.merge(splits, on="image_ord", how="left")
    sub = base[base.split == split_name].sort_values("image_ord").reset_index(drop=True)
    if max_images > 0:
        sub = sub.head(max_images)
    return sub


def collect_vectors(
    store: SegmentStore,
    model: str,
    sources: list[str],
    layer: str,
    split: str | None = None,
    final_success: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    frames = []
    vecs = []
    for source in sources:
        sub, x = store.rows_for(model, source, layer)
        if sub.empty:
            continue
        mask = np.ones(len(sub), dtype=bool)
        if split is not None:
            mask &= store.split_mask(sub, split)
        if final_success is not None and "final_success" in sub.columns:
            mask &= sub.final_success.to_numpy(dtype=int) == int(final_success)
        if mask.any():
            frames.append(sub.loc[mask].copy())
            vecs.append(x[mask])
    if not vecs:
        return pd.DataFrame(), np.zeros((0, 0), dtype=np.float32)
    return pd.concat(frames, ignore_index=True), np.concatenate(vecs, axis=0).astype(np.float32)


def fit_named_bases(store: SegmentStore, args, transport_basis: np.ndarray) -> dict[str, np.ndarray]:
    basis_by_name: dict[str, np.ndarray] = {"transport": transport_basis}
    specs = {
        "failed_attack": (parse_csv(args.transport_sources), 0),
        "random_feasible": (parse_csv(args.random_sources), None),
        "clean_motion": (parse_csv(args.clean_sources), None),
    }
    for name, (sources, succ) in specs.items():
        if not sources:
            continue
        _rows, x = collect_vectors(store, args.model, sources, args.layer, split="train", final_success=succ)
        if x.size == 0:
            continue
        try:
            _mean, b, _kk = pca_basis(x, max(args.k_list), args.seed + 17, args.normalize_vectors)
            basis_by_name[name] = b
        except ValueError:
            continue
    return basis_by_name


def compute_jvp_sketch_basis(args, wrapper, dataset, train_images: pd.DataFrame, out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    meta_rows = []
    device = args.device_obj
    for image_i, row in enumerate(train_images.itertuples(index=False), start=1):
        x_cpu, _ = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        gen = torch.Generator(device=device).manual_seed(args.seed + int(row.dataset_idx) * 1543 + 31)
        remaining = args.n_jvp_dirs
        while remaining > 0:
            bs = min(args.jvp_batch_size, remaining)
            signs = torch.where(
                torch.rand((bs,) + tuple(x0.shape[1:]), generator=gen, device=device) < 0.5,
                -torch.ones((bs,) + tuple(x0.shape[1:]), device=device),
                torch.ones((bs,) + tuple(x0.shape[1:]), device=device),
            )
            if args.unit_l2_jvp_dirs:
                signs = signs / signs.flatten(1).norm(dim=1).view(bs, 1, 1, 1).clamp_min(1e-12)
            x_batch = x0.repeat(bs, 1, 1, 1)

            def feat(inp: torch.Tensor) -> torch.Tensor:
                return feature_tensor(wrapper, inp, args.layer)

            _val, jvp = torch.autograd.functional.jvp(feat, x_batch, signs, create_graph=False, strict=False)
            vectors.append(jvp.detach().cpu().numpy().astype(np.float32))
            start = len(meta_rows)
            for j in range(bs):
                meta_rows.append(
                    {
                        "vector_idx": int(start + j),
                        "image_ord": int(row.image_ord),
                        "dataset_idx": int(row.dataset_idx),
                        "label": int(row.label),
                        "split": "train",
                        "direction_local_idx": int(args.n_jvp_dirs - remaining + j),
                    }
                )
            remaining -= bs
        if image_i % max(1, args.progress_every) == 0:
            print(f"[jvp-sketch] {image_i}/{len(train_images)} train images", flush=True)
    z = np.concatenate(vectors, axis=0).astype(np.float32)
    np.savez_compressed(out_dir / "jvp_sketch_vectors.npz", vectors=z)
    pd.DataFrame(meta_rows).to_csv(out_dir / "jvp_sketch_metadata.csv", index=False)
    _mean, basis, _kk = pca_basis(z, max(args.k_list), args.seed + 101, args.normalize_vectors)
    return z, basis


def overlap_tables(basis_by_name: dict[str, np.ndarray], k_list: list[int], random_nulls: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    null_rows = []
    rng = np.random.default_rng(seed + 202)
    transport = basis_by_name["transport"]
    d = transport.shape[1]
    for k in k_list:
        random_overlaps = []
        for _ in range(random_nulls):
            rb = random_orth_basis(rng, d, min(k, d))
            random_overlaps.append(subspace_metrics(transport, rb, k)["overlap"])
        random_overlaps = np.asarray(random_overlaps, dtype=float)
        null_mean = float(np.nanmean(random_overlaps))
        null_std = float(np.nanstd(random_overlaps) + 1e-12)
        null_rows.append({"k": k, "random_null_mean_overlap": null_mean, "random_null_std_overlap": null_std, "n_null": random_nulls})
        for name, basis in basis_by_name.items():
            if name == "transport":
                continue
            m = subspace_metrics(transport, basis, k)
            rows.append(
                {
                    "comparison": f"transport_vs_{name}",
                    "k": k,
                    "overlap": m["overlap"],
                    "mean_principal_angle_deg": m["mean_angle_deg"],
                    "max_principal_angle_deg": m["max_angle_deg"],
                    "subspace_affinity": m["affinity"],
                    "random_null_z": float((m["overlap"] - null_mean) / null_std),
                    "random_null_mean_overlap": null_mean,
                    "random_null_std_overlap": null_std,
                }
            )
        rb = random_orth_basis(rng, d, min(k, d))
        m = subspace_metrics(transport, rb, k)
        rows.append(
            {
                "comparison": "transport_vs_random_orthogonal_basis",
                "k": k,
                "overlap": m["overlap"],
                "mean_principal_angle_deg": m["mean_angle_deg"],
                "max_principal_angle_deg": m["max_angle_deg"],
                "subspace_affinity": m["affinity"],
                "random_null_z": float((m["overlap"] - null_mean) / null_std),
                "random_null_mean_overlap": null_mean,
                "random_null_std_overlap": null_std,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


def residual_projection_metrics(store: SegmentStore, args, transport_mean: np.ndarray, transport_basis: np.ndarray, jvp_basis: np.ndarray) -> pd.DataFrame:
    rows = []
    residual_basis = orth_residual_basis(transport_basis, jvp_basis, max(args.k_list), max(args.k_list))
    for residual_k in args.k_list:
        for source in parse_csv(args.transport_sources):
            sub, x = store.rows_for(args.model, source, args.layer)
            if sub.empty:
                continue
            test = store.split_mask(sub, "test")
            if not test.any():
                continue
            xr = residualize(x, jvp_basis, residual_k)
            score_raw = projection_energy(x, transport_mean, transport_basis, residual_k)
            score_resid = projection_energy(xr, np.zeros((1, xr.shape[1]), dtype=np.float32), residual_basis, residual_k)
            for score_name, scores in [("transport_energy", score_raw), ("residual_transport_energy", score_resid)]:
                g = sub.loc[test].copy()
                y = g.final_success.to_numpy(dtype=int)
                rows.append(
                    {
                        "source": source,
                        "layer": args.layer,
                        "residual_k": residual_k,
                        "score": score_name,
                        "comparison": "success_vs_failed_same_optimizer",
                        "n_segments": int(test.sum()),
                        "n_success_segments": int(y.sum()),
                        "n_failed_segments": int((1 - y).sum()),
                        "auroc": safe_auroc(y, scores[test]),
                        "auprc": safe_auprc(y, scores[test]),
                    }
                )
    return pd.DataFrame(rows)


def within_image_z(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[f"{c}_z"] = out.groupby("image_ord")[c].transform(
            lambda s: (s - s.mean()) / max(float(s.std(ddof=0)), 1e-12)
        )
    return out


def split_candidate_images(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    images = np.array(sorted(df.image_ord.unique()))
    train_images = set(images[::2])
    train = df.image_ord.isin(train_images).to_numpy()
    return train, ~train


def train_candidate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "probe_margin_drop",
        "ce_grad_cos",
        "neg_margin_grad_cos",
        "jvp_gain",
        "transport_energy",
        "residual_transport_energy",
    ]
    dfz = within_image_z(df, cols)
    specs = {
        "M1_margin": ["probe_margin_drop_z"],
        "M2_margin_grad": ["probe_margin_drop_z", "ce_grad_cos_z", "neg_margin_grad_cos_z"],
        "M3_margin_grad_jvp": ["probe_margin_drop_z", "ce_grad_cos_z", "neg_margin_grad_cos_z", "jvp_gain_z"],
        "M4_plus_transport": [
            "probe_margin_drop_z",
            "ce_grad_cos_z",
            "neg_margin_grad_cos_z",
            "jvp_gain_z",
            "transport_energy_z",
        ],
        "M5_plus_residual_transport": [
            "probe_margin_drop_z",
            "ce_grad_cos_z",
            "neg_margin_grad_cos_z",
            "jvp_gain_z",
            "residual_transport_energy_z",
        ],
        "M6_plus_both_transport_scores": [
            "probe_margin_drop_z",
            "ce_grad_cos_z",
            "neg_margin_grad_cos_z",
            "jvp_gain_z",
            "transport_energy_z",
            "residual_transport_energy_z",
        ],
    }
    metrics = []
    score_frames = []
    for (eps, alpha), g in dfz.groupby(["eps_over_255", "alpha"], sort=True):
        train, test = split_candidate_images(g)
        y = g.full_success.to_numpy(dtype=int)
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        for name, features in specs.items():
            x = g[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"))
            clf.fit(x[train], y[train])
            scores = np.full(len(g), np.nan, dtype=np.float32)
            scores[train] = clf.predict_proba(x[train])[:, 1]
            scores[test] = clf.predict_proba(x[test])[:, 1]
            metrics.append(
                {
                    "eps_over_255": float(eps),
                    "alpha": float(alpha),
                    "model_name": name,
                    "features": ",".join(features),
                    "test_auc": safe_auroc(y[test], scores[test]),
                    "test_auprc": safe_auprc(y[test], scores[test]),
                    "n_train_candidates": int(train.sum()),
                    "n_test_candidates": int(test.sum()),
                }
            )
            sf = g[["image_ord", "direction_id", "eps_over_255", "alpha", "full_success", "full_margin_drop"]].copy()
            sf["selector"] = f"learned_{name}"
            sf["score"] = scores
            sf["is_test_score"] = test
            score_frames.append(sf)
    return pd.DataFrame(metrics), pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()


def raw_candidate_scores(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "random": None,
        "margin_drop": "probe_margin_drop",
        "jvp_gain": "jvp_gain",
        "transport_energy": "transport_energy",
        "residual_transport_energy": "residual_transport_energy",
        "margin_x_jvp": "score_margin_x_jvp",
        "margin_x_transport": "score_margin_x_transport",
        "margin_x_residual_transport": "score_margin_x_residual_transport",
    }
    rng = np.random.default_rng(0)
    frames = []
    for name, col in mapping.items():
        sf = df[["image_ord", "direction_id", "eps_over_255", "alpha", "full_success", "full_margin_drop"]].copy()
        sf["selector"] = name
        sf["score"] = rng.random(len(sf)) if col is None else df[col].to_numpy(dtype=float)
        sf["is_test_score"] = True
        frames.append(sf)
    return pd.concat(frames, ignore_index=True)


def summarize_topk(score_df: pd.DataFrame, top_ks: list[int], seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed + 404)
    score_df = score_df[score_df.is_test_score.astype(bool)].copy()
    for (eps, alpha, selector, image_ord), g in score_df.groupby(["eps_over_255", "alpha", "selector", "image_ord"], sort=False):
        for top_k in top_ks:
            kk = min(top_k, len(g))
            if selector == "random":
                chosen = g.sample(n=kk, random_state=int(rng.integers(0, 2**31 - 1)))
            else:
                chosen = g.sort_values("score", ascending=False).head(kk)
            rows.append(
                {
                    "eps_over_255": float(eps),
                    "alpha": float(alpha),
                    "selector": selector,
                    "image_ord": int(image_ord),
                    "top_k": int(top_k),
                    "topk_any_success": int(chosen.full_success.astype(int).max()),
                    "topk_precision": float(chosen.full_success.astype(float).mean()),
                    "best_full_margin_drop": float(chosen.full_margin_drop.max()),
                }
            )
    per_image = pd.DataFrame(rows)
    if per_image.empty:
        return per_image
    return (
        per_image.groupby(["eps_over_255", "alpha", "selector", "top_k"], dropna=False)
        .agg(
            n_images=("image_ord", "nunique"),
            topk_asr=("topk_any_success", "mean"),
            topk_precision=("topk_precision", "mean"),
            mean_best_full_margin_drop=("best_full_margin_drop", "mean"),
        )
        .reset_index()
    )


def recompute_candidate_residual_energy(args, wrapper, dataset, candidate_csv: Path, residual_basis: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv(candidate_csv)
    if args.max_candidate_rows and len(df) > args.max_candidate_rows:
        df = df.head(args.max_candidate_rows).copy()
    device = args.device_obj
    out_frames = []
    for (eps, alpha, image_ord), g in df.groupby(["eps_over_255", "alpha", "image_ord"], sort=True):
        first = g.iloc[0]
        x_cpu, _ = dataset[int(first.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        with torch.no_grad():
            h0 = feature_tensor(wrapper, x0, args.layer).detach()
        gen = torch.Generator(device=device).manual_seed(
            args.seed + int(first.dataset_idx) * 1009 + int(float(eps) * 100) * 917 + int(float(alpha) * 1000)
        )
        n_dirs = int(g.direction_id.max()) + 1
        all_scores = np.full(n_dirs, np.nan, dtype=np.float32)
        remaining = n_dirs
        cursor = 0
        probe_eps = float(eps) * float(alpha) / 255.0
        while remaining > 0:
            bs = min(args.candidate_batch_size, remaining)
            signs = torch.where(
                torch.rand((bs,) + tuple(x0.shape[1:]), generator=gen, device=device) < 0.5,
                -torch.ones((bs,) + tuple(x0.shape[1:]), device=device),
                torch.ones((bs,) + tuple(x0.shape[1:]), device=device),
            )
            with torch.no_grad():
                x_probe = (x0.repeat(bs, 1, 1, 1) + probe_eps * signs).clamp(0, 1)
                h_probe = feature_tensor(wrapper, x_probe, args.layer).detach()
            fd = h_probe - h0
            fd_np = fd.cpu().numpy().astype(np.float32)
            all_scores[cursor : cursor + bs] = projection_energy(
                fd_np, np.zeros((1, fd_np.shape[1]), dtype=np.float32), residual_basis, min(args.residual_eval_k, residual_basis.shape[0])
            )
            cursor += bs
            remaining -= bs
        gg = g.copy()
        gg["residual_transport_energy"] = [float(all_scores[int(i)]) for i in gg.direction_id]
        gg["score_margin_x_residual_transport"] = np.maximum(gg.probe_margin_drop.to_numpy(dtype=float), 0.0) * gg[
            "residual_transport_energy"
        ].to_numpy(dtype=float)
        out_frames.append(gg)
        if len(out_frames) % max(1, args.progress_every) == 0:
            print(f"[candidate-residual] groups={len(out_frames)} latest eps={eps} alpha={alpha} image={image_ord}", flush=True)
    return pd.concat(out_frames, ignore_index=True)


def plot_outputs(overlap: pd.DataFrame, pred: pd.DataFrame, topk: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), dpi=180)
    sub = overlap[overlap.comparison.str.contains("jvp_sketch|failed_attack|random_feasible", regex=True, na=False)]
    for comp, g in sub.groupby("comparison"):
        axes[0].plot(g.k, g.overlap, marker="o", label=comp.replace("transport_vs_", ""))
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Projection overlap")
    axes[0].set_title("Transport-basis overlap")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)

    if not pred.empty:
        keep = pred[pred.model_name.isin(["M3_margin_grad_jvp", "M4_plus_transport", "M5_plus_residual_transport"])]
        for name, g in keep.groupby("model_name"):
            axes[1].plot(g.eps_over_255 + 0.03 * g.alpha, g.test_auprc, marker="o", label=name)
    axes[1].set_xlabel("eps/255")
    axes[1].set_ylabel("Held-out AUPRC")
    axes[1].set_title("Residual predictive value")
    axes[1].grid(alpha=0.25)
    if axes[1].has_data():
        axes[1].legend(frameon=False, fontsize=7)

    if not topk.empty:
        keep = topk[(topk.top_k == 10) & (topk.selector.isin(["margin_drop", "margin_x_jvp", "margin_x_transport", "margin_x_residual_transport"]))]
        for selector, g in keep.groupby("selector"):
            axes[2].plot(g.eps_over_255 + 0.03 * g.alpha, g.topk_asr, marker="o", label=selector)
    axes[2].set_xlabel("eps/255")
    axes[2].set_ylabel("Top-10 ASR")
    axes[2].set_title("Residual selector value")
    axes[2].grid(alpha=0.25)
    if axes[2].has_data():
        axes[2].legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "jacobian_basis_residual_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "jacobian_basis_residual_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def write_findings(out_dir: Path, overlap: pd.DataFrame, pred: pd.DataFrame, topk: pd.DataFrame, meta: dict) -> None:
    lines = [
        "# Jacobian-Basis and Residual Transport Findings",
        "",
        f"- Model: `{meta['model']}`",
        f"- Layer: `{meta['layer']}`",
        f"- JVP-sketch train images: {meta['n_jvp_train_images']}",
        f"- JVP directions per image: {meta['n_jvp_dirs']}",
        f"- Vector normalization before PCA: {meta['normalize_vectors']}",
        "",
        "## Basis-Level Overlap",
        "",
    ]
    for r in overlap.sort_values(["k", "comparison"]).itertuples(index=False):
        if "jvp_sketch" in r.comparison or "random_orthogonal" in r.comparison:
            lines.append(
                f"- {r.comparison}, k={r.k}: overlap={r.overlap:.3f}, "
                f"mean angle={r.mean_principal_angle_deg:.1f} deg, random-null z={r.random_null_z:.1f}."
            )
    lines += ["", "## Residual Candidate Prediction", ""]
    if pred.empty:
        lines.append("- Candidate residual prediction was skipped or could not be fit.")
    else:
        base = pred[pred.model_name == "M3_margin_grad_jvp"][["eps_over_255", "alpha", "test_auprc"]].rename(
            columns={"test_auprc": "m3_auprc"}
        )
        res = pred[pred.model_name == "M5_plus_residual_transport"][["eps_over_255", "alpha", "test_auprc"]].rename(
            columns={"test_auprc": "m5_auprc"}
        )
        merged = base.merge(res, on=["eps_over_255", "alpha"], how="inner")
        merged["delta"] = merged.m5_auprc - merged.m3_auprc
        for r in merged.itertuples(index=False):
            lines.append(f"- eps={r.eps_over_255:g}, alpha={r.alpha:g}: residual transport AUPRC delta after M3 = {r.delta:+.3f}.")
    lines += [
        "",
        "## Decision Template",
        "",
        "If transport-vs-JVP overlap is high and residual transport adds little after M3, the paper should be written as a hidden-Jacobian-gain proposal plus margin/gradient-selection story.",
        "If overlap is partial and residual transport adds predictive or intervention value, the paper can claim finite-budget success-conditioned residual structure beyond local hidden-Jacobian gain.",
    ]
    (out_dir / "jacobian_basis_residual_findings.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("analysis_outputs/hidden_jacobian_routing/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto"))
    p.add_argument("--candidate-csv", type=Path, default=Path("analysis_outputs/hidden_jacobian_routing/jacobian_null_response/mobility_vs_jacobian_gain_bbb_resnet50_d64/candidate_level_jvp.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/hidden_jacobian_routing/jacobian_null_response/jacobian_basis_residual_bbb_resnet50"))
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--n-jvp-dirs", type=int, default=64)
    p.add_argument("--jvp-batch-size", type=int, default=8)
    p.add_argument("--candidate-batch-size", type=int, default=64)
    p.add_argument("--k-list", default="5,10,20")
    p.add_argument("--transport-sources", default="pgd,square")
    p.add_argument("--random-sources", default="random_sign_walk_pgd_budget,random_sign_walk_square_budget,correlated_random_walk_pgd_budget,correlated_random_walk_square_budget")
    p.add_argument("--clean-sources", default="")
    p.add_argument("--top-ks", default="1,5,10")
    p.add_argument("--random-nulls", type=int, default=1000)
    p.add_argument("--residual-eval-k", type=int, default=20)
    p.add_argument("--normalize-vectors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--unit-l2-jvp-dirs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-candidate-rows", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress-every", type=int, default=10)
    args = p.parse_args()

    set_seed(args.seed)
    args.device_obj = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.k_list = parse_int_csv(args.k_list)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    store = SegmentStore(args.input_dir)
    _transport_rows, transport_x = collect_vectors(
        store, args.model, parse_csv(args.transport_sources), args.layer, split="train", final_success=1
    )
    if transport_x.size == 0:
        raise RuntimeError("No train-split successful transport vectors found.")
    transport_mean, transport_basis, _ = pca_basis(transport_x, max(args.k_list), args.seed, args.normalize_vectors)

    wrapper = load_model(args.model, args.device_obj).eval()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    train_images = load_images_for_split(args.input_dir, args.model, "train", args.images)
    if train_images.empty:
        raise RuntimeError("No train-split images available for JVP sketch.")
    _jvp_vectors, jvp_basis = compute_jvp_sketch_basis(args, wrapper, dataset, train_images, args.output_dir)

    basis_by_name = fit_named_bases(store, args, transport_basis)
    basis_by_name["jvp_sketch"] = jvp_basis
    overlap, nulls = overlap_tables(basis_by_name, args.k_list, args.random_nulls, args.seed)
    overlap.to_csv(args.output_dir / "summary_subspace_overlap.csv", index=False)
    nulls.to_csv(args.output_dir / "summary_random_subspace_null.csv", index=False)

    residual_metrics = residual_projection_metrics(store, args, transport_mean, transport_basis, jvp_basis)
    residual_metrics.to_csv(args.output_dir / "summary_residual_projection_metrics.csv", index=False)

    residual_basis = orth_residual_basis(transport_basis, jvp_basis, max(args.k_list), max(args.k_list))
    np.savez_compressed(
        args.output_dir / "basis_vectors.npz",
        transport_mean=transport_mean,
        transport_basis=transport_basis,
        jvp_sketch_basis=jvp_basis,
        residual_transport_basis=residual_basis,
    )

    candidate_pred = pd.DataFrame()
    topk = pd.DataFrame()
    candidate_aug = pd.DataFrame()
    if args.candidate_csv.exists():
        candidate_aug = recompute_candidate_residual_energy(args, wrapper, dataset, args.candidate_csv, residual_basis)
        candidate_aug.to_csv(args.output_dir / "candidate_level_with_residual_transport.csv", index=False)
        candidate_pred, learned_scores = train_candidate_models(candidate_aug)
        candidate_pred.to_csv(args.output_dir / "summary_residual_predictive.csv", index=False)
        score_df = pd.concat([raw_candidate_scores(candidate_aug), learned_scores], ignore_index=True)
        topk = summarize_topk(score_df, parse_int_csv(args.top_ks), args.seed)
        topk.to_csv(args.output_dir / "summary_residual_topk.csv", index=False)
    else:
        print(f"[warn] candidate CSV not found: {args.candidate_csv}; skipping candidate residual prediction", flush=True)

    if hasattr(wrapper, "close"):
        wrapper.close()

    plot_outputs(overlap, candidate_pred, topk, args.output_dir)
    meta = {
        "model": args.model,
        "layer": args.layer,
        "input_dir": str(args.input_dir),
        "candidate_csv": str(args.candidate_csv),
        "n_jvp_train_images": int(len(train_images)),
        "n_jvp_dirs": args.n_jvp_dirs,
        "k_list": args.k_list,
        "normalize_vectors": args.normalize_vectors,
        "unit_l2_jvp_dirs": args.unit_l2_jvp_dirs,
        "random_nulls": args.random_nulls,
        "max_candidate_rows": args.max_candidate_rows,
        "outputs": [
            "jvp_sketch_vectors.npz",
            "basis_vectors.npz",
            "summary_subspace_overlap.csv",
            "summary_random_subspace_null.csv",
            "summary_residual_projection_metrics.csv",
            "candidate_level_with_residual_transport.csv",
            "summary_residual_predictive.csv",
            "summary_residual_topk.csv",
            "jacobian_basis_residual_summary.png",
            "jacobian_basis_residual_findings.md",
        ],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    write_findings(args.output_dir, overlap, candidate_pred, topk, meta)

    print(overlap.to_string(index=False))
    if not candidate_pred.empty:
        print(candidate_pred.to_string(index=False))
    if not topk.empty:
        print(topk.to_string(index=False))


if __name__ == "__main__":
    main()
