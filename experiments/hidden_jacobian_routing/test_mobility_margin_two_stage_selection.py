#!/usr/bin/env python3
"""Two-stage mobility/margin selection diagnostic.

This experiment tests the current refined theory:

    representation mobility proposes easy hidden-space directions, while
    margin reduction selects the adversarially useful subset under budget.

For each clean image, we sample many L_inf sign directions without using an
adversarial objective.  Each direction is scored at a small probe radius, then
the same direction is evaluated at the full attack budget.  This avoids the
tautology where a selector observes the final margin used to define success.

The key question is whether probe-time mobility adds useful information beyond
probe-time margin drop and gradient alignment when predicting full-budget
adversarial success.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import load_model  # noqa: E402
from experiments.hidden_jacobian_routing.common import margin  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    y = y[ok].astype(int)
    s = s[ok].astype(float)
    if len(y) < 4 or len(np.unique(y)) < 2 or np.nanstd(s) < 1e-12:
        return np.nan
    return float(roc_auc_score(y, s))


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    y = y[ok].astype(int)
    s = s[ok].astype(float)
    if len(y) < 4 or len(np.unique(y)) < 2 or np.nanstd(s) < 1e-12:
        return np.nan
    return float(average_precision_score(y, s))


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def pca_basis(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    xc = x.astype(np.float32) - mean
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return mean, vt[: min(k, vt.shape[0])].astype(np.float32)


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    xc = x.astype(np.float32) - mean.astype(np.float32)
    kk = min(k, basis.shape[0])
    coeff = xc @ basis[:kk].T
    denom = np.sum(xc * xc, axis=1)
    return np.sum(coeff * coeff, axis=1) / np.clip(denom, 1e-12, None)


def vector_key(model: str, source: str, layer: str) -> str:
    return f"{model}__{source}__{layer}"


def fit_segment_basis(
    input_dir: Path,
    model: str,
    layer: str,
    sources: list[str],
    k: int,
    success_only: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    meta = pd.read_csv(input_dir / "segment_metadata.csv")
    splits = pd.read_csv(input_dir / "image_splits.csv")
    split_map = dict(zip(splits.image_ord.astype(int), splits.split.astype(str)))
    arrays = np.load(input_dir / "segment_vectors.npz")

    chunks = []
    n_rows = 0
    for source in sources:
        key = vector_key(model, source, layer)
        if key not in arrays.files:
            continue
        sub = meta[(meta.model == model) & (meta.source == source) & (meta.layer == layer)].copy()
        if sub.empty:
            continue
        sub["split"] = sub.image_ord.astype(int).map(split_map)
        sub = sub[sub.split == "train"]
        if success_only:
            sub = sub[sub.final_success.astype(int) == 1]
        if sub.empty:
            continue
        chunks.append(arrays[key][sub.vector_idx.to_numpy(dtype=int)].astype(np.float32))
        n_rows += len(sub)
    if not chunks:
        raise RuntimeError(f"No vectors for basis: sources={sources}, layer={layer}")
    x = np.concatenate(chunks, axis=0)
    return (*pca_basis(x, k), n_rows)


def load_eval_images(input_dir: Path, model: str, max_images: int) -> pd.DataFrame:
    outcomes = pd.read_csv(input_dir / "image_outcomes.csv")
    splits = pd.read_csv(input_dir / "image_splits.csv")
    base = outcomes[(outcomes.model == model) & (outcomes.source == "pgd")][
        ["image_ord", "dataset_idx", "label", "clean_pred", "clean_margin"]
    ].drop_duplicates()
    base = base.merge(splits, on="image_ord", how="left")
    sub = base[base.split == "test"].sort_values("image_ord").reset_index(drop=True)
    if max_images > 0:
        sub = sub.head(max_images)
    return sub


def feature_numpy(wrapper, x: torch.Tensor, layer: str) -> np.ndarray:
    _logits, feats, _raw = wrapper.forward_with_features(x)
    if layer not in feats:
        raise RuntimeError(f"Layer {layer} not captured.")
    return feats[layer].detach().cpu().numpy().astype(np.float32)


def clean_grads(wrapper, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probe = x.detach().requires_grad_(True)
    logits = wrapper(probe)
    ce = F.cross_entropy(logits, y)
    ce_grad = torch.autograd.grad(ce, probe, retain_graph=True)[0].detach()
    m = margin(logits, y).sum()
    margin_grad = torch.autograd.grad(m, probe)[0].detach()
    return ce_grad, margin_grad


def cosine_with_batch(signs: torch.Tensor, direction: torch.Tensor) -> np.ndarray:
    s = signs.flatten(1).float()
    d = direction.flatten(1).float()
    d_norm = d.norm(dim=1).clamp_min(1e-12)
    s_norm = s.norm(dim=1).clamp_min(1e-12)
    return ((s * d).sum(dim=1) / (s_norm * d_norm)).detach().cpu().numpy().astype(np.float32)


def evaluate_candidates(args) -> tuple[pd.DataFrame, dict]:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)

    wrapper = load_model(args.model, device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    images = load_eval_images(input_dir, args.model, args.images)

    flow_sources = parse_csv(args.flow_basis_sources)
    flow_mean, flow_basis, n_flow = fit_segment_basis(input_dir, args.model, args.layer, flow_sources, args.k, True)
    mobility_sources = parse_csv(args.mobility_basis_sources)
    mob_mean, mob_basis, n_mob = fit_segment_basis(input_dir, args.model, args.layer, mobility_sources, args.k, False)

    probe_eps = args.probe_eps / 255.0
    attack_eps = args.attack_eps / 255.0
    rows = []

    for image_i, row in enumerate(images.itertuples(index=False), start=1):
        x_cpu, _ = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        with torch.no_grad():
            logits0 = wrapper(x0)
            clean_pred = int(logits0.argmax(1).item())
            clean_margin = float(margin(logits0, y).item())
            clean_py = float(torch.softmax(logits0, dim=1)[0, int(row.label)].item())
            h0 = feature_numpy(wrapper, x0, args.layer)[0]
        if clean_pred != int(row.label):
            continue
        ce_grad, margin_grad = clean_grads(wrapper, x0, y)
        gen = torch.Generator(device=device).manual_seed(args.seed + int(row.dataset_idx) * 1009)
        remaining = args.directions_per_image
        dir_offset = 0
        while remaining > 0:
            bs = min(args.batch_size, remaining)
            signs = torch.where(
                torch.rand((bs,) + tuple(x0.shape[1:]), generator=gen, device=device) < 0.5,
                -torch.ones((bs,) + tuple(x0.shape[1:]), device=device),
                torch.ones((bs,) + tuple(x0.shape[1:]), device=device),
            )
            x_probe = (x0 + probe_eps * signs).clamp(0, 1)
            x_full = (x0 + attack_eps * signs).clamp(0, 1)
            with torch.no_grad():
                logits_probe = wrapper(x_probe)
                logits_full = wrapper(x_full)
                h_probe = feature_numpy(wrapper, x_probe, args.layer)
            probe_margin = margin(logits_probe, y.expand(bs)).detach().cpu().numpy().astype(np.float32)
            full_margin = margin(logits_full, y.expand(bs)).detach().cpu().numpy().astype(np.float32)
            probe_py = torch.softmax(logits_probe, dim=1)[:, int(row.label)].detach().cpu().numpy().astype(np.float32)
            full_py = torch.softmax(logits_full, dim=1)[:, int(row.label)].detach().cpu().numpy().astype(np.float32)
            pred_full = logits_full.argmax(1).detach().cpu().numpy().astype(np.int64)
            disp_probe = h_probe - h0[None, :]
            mobility = np.linalg.norm(disp_probe, axis=1).astype(np.float32)
            flow_energy = projection_energy(disp_probe, flow_mean, flow_basis, args.k).astype(np.float32)
            mobility_basis_energy = projection_energy(disp_probe, mob_mean, mob_basis, args.k).astype(np.float32)
            ce_cos = cosine_with_batch(signs, ce_grad.expand_as(signs))
            neg_margin_cos = cosine_with_batch(signs, (-margin_grad).expand_as(signs))
            probe_margin_drop = (clean_margin - probe_margin).astype(np.float32)
            full_margin_drop = (clean_margin - full_margin).astype(np.float32)
            probe_py_drop = (clean_py - probe_py).astype(np.float32)
            full_py_drop = (clean_py - full_py).astype(np.float32)
            for j in range(bs):
                rows.append(
                    {
                        "image_ord": int(row.image_ord),
                        "dataset_idx": int(row.dataset_idx),
                        "label": int(row.label),
                        "direction_id": int(dir_offset + j),
                        "clean_margin": clean_margin,
                        "clean_p_y": clean_py,
                        "probe_margin": float(probe_margin[j]),
                        "full_margin": float(full_margin[j]),
                        "probe_margin_drop": float(probe_margin_drop[j]),
                        "full_margin_drop": float(full_margin_drop[j]),
                        "probe_p_y_drop": float(probe_py_drop[j]),
                        "full_p_y_drop": float(full_py_drop[j]),
                        "full_pred": int(pred_full[j]),
                        "full_success": int(pred_full[j] != int(row.label)),
                        "probe_mobility": float(mobility[j]),
                        "probe_flow_energy": float(flow_energy[j]),
                        "probe_mobility_basis_energy": float(mobility_basis_energy[j]),
                        "ce_grad_cos": float(ce_cos[j]),
                        "neg_margin_grad_cos": float(neg_margin_cos[j]),
                        "score_mobility_x_margin": float(mobility[j] * max(float(probe_margin_drop[j]), 0.0)),
                        "score_flow_x_margin": float(flow_energy[j] * max(float(probe_margin_drop[j]), 0.0)),
                        "score_mobbasis_x_margin": float(mobility_basis_energy[j] * max(float(probe_margin_drop[j]), 0.0)),
                    }
                )
            remaining -= bs
            dir_offset += bs
        if image_i % max(1, args.progress_every) == 0:
            partial = pd.DataFrame(rows)
            partial.to_csv(out_dir / "partial_two_stage_candidate_scores.csv", index=False)
            print(f"[progress] {image_i}/{len(images)} images, {len(rows)} candidates", flush=True)

    if hasattr(wrapper, "close"):
        wrapper.close()
    df = pd.DataFrame(rows)
    meta = {
        "model": args.model,
        "layer": args.layer,
        "images_requested": args.images,
        "images_evaluated": int(df.image_ord.nunique()) if not df.empty else 0,
        "directions_per_image": args.directions_per_image,
        "probe_eps_over_255": args.probe_eps,
        "attack_eps_over_255": args.attack_eps,
        "k": args.k,
        "flow_basis_sources": flow_sources,
        "mobility_basis_sources": mobility_sources,
        "n_flow_basis_segments": n_flow,
        "n_mobility_basis_segments": n_mob,
        "seed": args.seed,
    }
    return df, meta


def summarize_selectors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selectors = {
        "random_direction": None,
        "probe_mobility": "probe_mobility",
        "probe_margin_drop": "probe_margin_drop",
        "probe_flow_energy": "probe_flow_energy",
        "mobility_x_margin": "score_mobility_x_margin",
        "flow_x_margin": "score_flow_x_margin",
        "mobility_basis_energy": "probe_mobility_basis_energy",
        "mobbasis_x_margin": "score_mobbasis_x_margin",
        "ce_grad_cos": "ce_grad_cos",
        "neg_margin_grad_cos": "neg_margin_grad_cos",
    }
    per_image = []
    rng = np.random.default_rng(0)
    for image_ord, g in df.groupby("image_ord", sort=True):
        for name, col in selectors.items():
            if col is None:
                chosen = g.iloc[int(rng.integers(0, len(g)))]
            else:
                chosen = g.loc[g[col].astype(float).idxmax()]
            row = chosen.to_dict()
            row["selector"] = name
            per_image.append(row)
    per_image_df = pd.DataFrame(per_image)
    summary = (
        per_image_df.groupby("selector", dropna=False)
        .agg(
            n_images=("image_ord", "nunique"),
            asr=("full_success", "mean"),
            mean_full_margin_drop=("full_margin_drop", "mean"),
            median_full_margin_drop=("full_margin_drop", "median"),
            mean_probe_margin_drop=("probe_margin_drop", "mean"),
            mean_probe_mobility=("probe_mobility", "mean"),
            mean_probe_flow_energy=("probe_flow_energy", "mean"),
            mean_ce_grad_cos=("ce_grad_cos", "mean"),
            mean_neg_margin_grad_cos=("neg_margin_grad_cos", "mean"),
        )
        .reset_index()
    )
    order = [
        "random_direction",
        "probe_mobility",
        "probe_flow_energy",
        "mobility_basis_energy",
        "ce_grad_cos",
        "neg_margin_grad_cos",
        "probe_margin_drop",
        "mobility_x_margin",
        "flow_x_margin",
        "mobbasis_x_margin",
    ]
    summary["selector"] = pd.Categorical(summary.selector, categories=order, ordered=True)
    summary = summary.sort_values("selector").reset_index(drop=True)
    return per_image_df, summary


def selector_columns() -> dict[str, str | None]:
    return {
        "random_direction": None,
        "probe_mobility": "probe_mobility",
        "probe_margin_drop": "probe_margin_drop",
        "probe_flow_energy": "probe_flow_energy",
        "mobility_x_margin": "score_mobility_x_margin",
        "flow_x_margin": "score_flow_x_margin",
        "mobility_basis_energy": "probe_mobility_basis_energy",
        "mobbasis_x_margin": "score_mobbasis_x_margin",
        "ce_grad_cos": "ce_grad_cos",
        "neg_margin_grad_cos": "neg_margin_grad_cos",
    }


def summarize_topk_selectors(df: pd.DataFrame, top_ks: list[int], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    base_rng = np.random.default_rng(seed + 1701)
    for image_ord, g in df.groupby("image_ord", sort=True):
        for selector, col in selector_columns().items():
            for k in top_ks:
                kk = min(int(k), len(g))
                if col is None:
                    chosen = g.sample(n=kk, random_state=int(base_rng.integers(0, 2**31 - 1)))
                else:
                    chosen = g.sort_values(col, ascending=False).head(kk)
                rows.append(
                    {
                        "image_ord": int(image_ord),
                        "selector": selector,
                        "top_k": int(k),
                        "n_candidates_selected": int(len(chosen)),
                        "topk_any_success": int(chosen.full_success.astype(int).max()),
                        "topk_success_count": int(chosen.full_success.astype(int).sum()),
                        "topk_precision": float(chosen.full_success.astype(float).mean()),
                        "best_full_margin_drop": float(chosen.full_margin_drop.max()),
                        "mean_full_margin_drop": float(chosen.full_margin_drop.mean()),
                        "best_probe_margin_drop": float(chosen.probe_margin_drop.max()),
                        "mean_probe_mobility": float(chosen.probe_mobility.mean()),
                        "mean_probe_flow_energy": float(chosen.probe_flow_energy.mean()),
                    }
                )
    per_image = pd.DataFrame(rows)
    summary = (
        per_image.groupby(["selector", "top_k"], dropna=False)
        .agg(
            n_images=("image_ord", "nunique"),
            topk_asr=("topk_any_success", "mean"),
            topk_precision=("topk_precision", "mean"),
            mean_best_full_margin_drop=("best_full_margin_drop", "mean"),
            mean_full_margin_drop=("mean_full_margin_drop", "mean"),
            mean_probe_mobility=("mean_probe_mobility", "mean"),
            mean_probe_flow_energy=("mean_probe_flow_energy", "mean"),
        )
        .reset_index()
    )
    return per_image, summary


def bootstrap_topk_summary(per_image: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    if per_image.empty or n_boot <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed + 2909)
    image_ids = np.array(sorted(per_image.image_ord.unique()))
    rows = []
    for (selector, top_k), g in per_image.groupby(["selector", "top_k"], sort=False):
        by_image = g.set_index("image_ord")
        values = []
        for _ in range(n_boot):
            sample = rng.choice(image_ids, size=len(image_ids), replace=True)
            boot = by_image.loc[sample]
            values.append(float(boot.topk_any_success.mean()))
        lo, hi = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "selector": selector,
                "top_k": int(top_k),
                "topk_asr_ci_low": float(lo),
                "topk_asr_ci_high": float(hi),
                "n_boot": int(n_boot),
            }
        )
    return pd.DataFrame(rows)


def summarize_predictors(df: pd.DataFrame) -> pd.DataFrame:
    feature_sets: dict[str, list[str]] = {
        "mobility_only": ["probe_mobility"],
        "flow_energy_only": ["probe_flow_energy"],
        "mobility_basis_energy_only": ["probe_mobility_basis_energy"],
        "probe_margin_only": ["probe_margin_drop"],
        "ce_grad_cos_only": ["ce_grad_cos"],
        "neg_margin_grad_cos_only": ["neg_margin_grad_cos"],
        "margin_plus_mobility": ["probe_margin_drop", "probe_mobility"],
        "margin_plus_flow": ["probe_margin_drop", "probe_flow_energy"],
        "margin_plus_mobbasis": ["probe_margin_drop", "probe_mobility_basis_energy"],
        "margin_plus_grad": ["probe_margin_drop", "ce_grad_cos", "neg_margin_grad_cos"],
        "all_features": [
            "probe_margin_drop",
            "probe_mobility",
            "probe_flow_energy",
            "probe_mobility_basis_energy",
            "ce_grad_cos",
            "neg_margin_grad_cos",
        ],
    }
    train_images = set(sorted(df.image_ord.unique())[::2])
    train = df.image_ord.isin(train_images).to_numpy()
    y = df.full_success.to_numpy(dtype=int)
    rows = []
    for name, cols in feature_sets.items():
        x = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        # Single-feature AUROC on held-out image split.
        if len(cols) == 1:
            score = x[:, 0]
            test_auc = safe_auroc(y[~train], score[~train])
            train_auc = safe_auroc(y[train], score[train])
            rows.append(
                {
                    "model_name": name,
                    "features": ",".join(cols),
                    "train_auc": train_auc,
                    "test_auc": test_auc,
                    "train_auprc": safe_auprc(y[train], score[train]),
                    "test_auprc": safe_auprc(y[~train], score[~train]),
                    "mode": "score",
                }
            )
        if len(np.unique(y[train])) < 2 or len(np.unique(y[~train])) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
        )
        clf.fit(x[train], y[train])
        proba_train = clf.predict_proba(x[train])[:, 1]
        proba_test = clf.predict_proba(x[~train])[:, 1]
        rows.append(
            {
                "model_name": name,
                "features": ",".join(cols),
                "train_auc": safe_auroc(y[train], proba_train),
                "test_auc": safe_auroc(y[~train], proba_test),
                "train_auprc": safe_auprc(y[train], proba_train),
                "test_auprc": safe_auprc(y[~train], proba_test),
                "mode": "logistic",
            }
        )
    return pd.DataFrame(rows).sort_values(["mode", "test_auc"], ascending=[True, False]).reset_index(drop=True)


def conditional_bins(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    rows = []
    work = df.copy()
    work["margin_bin"] = work.groupby("image_ord")["probe_margin_drop"].transform(
        lambda s: pd.qcut(s.rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    )
    for mb, g in work.groupby("margin_bin", dropna=True):
        if len(g) < 10:
            continue
        q = g.probe_mobility.quantile([0.2, 0.8])
        low = g[g.probe_mobility <= q.loc[0.2]]
        high = g[g.probe_mobility >= q.loc[0.8]]
        rows.append(
            {
                "margin_bin": int(mb),
                "n": int(len(g)),
                "low_mobility_asr": float(low.full_success.mean()),
                "high_mobility_asr": float(high.full_success.mean()),
                "asr_delta_high_minus_low": float(high.full_success.mean() - low.full_success.mean()),
                "low_mobility_margin_drop": float(low.full_margin_drop.mean()),
                "high_mobility_margin_drop": float(high.full_margin_drop.mean()),
                "margin_drop_delta_high_minus_low": float(high.full_margin_drop.mean() - low.full_margin_drop.mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_results(
    selector_summary: pd.DataFrame,
    pred: pd.DataFrame,
    cond: pd.DataFrame,
    out_dir: Path,
    topk_summary: pd.DataFrame | None = None,
) -> None:
    labels = {
        "random_direction": "Random",
        "probe_mobility": "Mobility",
        "probe_flow_energy": "Flow energy",
        "mobility_basis_energy": "Mobility-basis",
        "ce_grad_cos": "CE-grad cos",
        "neg_margin_grad_cos": "-margin-grad cos",
        "probe_margin_drop": "Margin drop",
        "mobility_x_margin": "Mobility x margin",
        "flow_x_margin": "Flow x margin",
        "mobbasis_x_margin": "Mob-basis x margin",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), dpi=180)
    s = selector_summary.copy()
    s["label"] = [labels.get(str(v), str(v)) for v in s.selector.astype(str)]
    axes[0].bar(s.label, s.asr, color="#4c78a8")
    axes[0].set_ylabel("Full-budget ASR")
    axes[0].set_title("Select one direction per image")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.25)

    p = pred[pred["mode"] == "logistic"].copy()
    axes[1].bar(p.model_name, p.test_auc, color="#59a14f")
    axes[1].axhline(0.5, color="black", lw=1, ls="--", alpha=0.55)
    axes[1].set_ylabel("Held-out AUROC")
    axes[1].set_title("Predict full-budget success")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.25)

    if not cond.empty:
        axes[2].plot(cond.margin_bin, cond.low_mobility_asr, marker="o", label="low mobility", color="#9ca3af")
        axes[2].plot(cond.margin_bin, cond.high_mobility_asr, marker="o", label="high mobility", color="#dc2626")
        axes[2].set_xlabel("Probe margin-drop bin")
        axes[2].set_ylabel("Full-budget ASR")
        axes[2].set_title("Mobility within margin bins")
        axes[2].legend(frameon=False)
        axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "two_stage_mobility_margin_selection.png", bbox_inches="tight")
    fig.savefig(out_dir / "two_stage_mobility_margin_selection.pdf", bbox_inches="tight")
    plt.close(fig)

    if topk_summary is not None and not topk_summary.empty:
        keep = [
            "random_direction",
            "probe_mobility",
            "probe_flow_energy",
            "ce_grad_cos",
            "neg_margin_grad_cos",
            "probe_margin_drop",
            "mobility_x_margin",
            "flow_x_margin",
        ]
        t = topk_summary[topk_summary.selector.isin(keep)].copy()
        t["label"] = [labels.get(str(v), str(v)) for v in t.selector.astype(str)]
        fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
        for selector, g in t.groupby("label", sort=False):
            ax.plot(g.top_k, g.topk_asr, marker="o", lw=2, label=selector)
        ax.set_xscale("log")
        ax.set_xticks(sorted(t.top_k.unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Top-K candidates allowed per image")
        ax.set_ylabel("Top-K ASR")
        ax.set_title("Selector quality under small candidate budgets")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "two_stage_topk_selector_curves.png", bbox_inches="tight")
        fig.savefig(out_dir / "two_stage_topk_selector_curves.pdf", bbox_inches="tight")
        plt.close(fig)


def write_note(
    out_dir: Path,
    meta: dict,
    selector_summary: pd.DataFrame,
    pred: pd.DataFrame,
    cond: pd.DataFrame,
    topk_summary: pd.DataFrame,
) -> None:
    def val(selector: str, col: str) -> float:
        sub = selector_summary[selector_summary.selector.astype(str) == selector]
        return float(sub[col].iloc[0]) if len(sub) else float("nan")

    margin_auc = pred[(pred.model_name == "probe_margin_only") & (pred["mode"] == "logistic")]
    combo_auc = pred[(pred.model_name == "margin_plus_mobility") & (pred["mode"] == "logistic")]
    all_auc = pred[(pred.model_name == "all_features") & (pred["mode"] == "logistic")]
    lines = [
        "# Two-Stage Mobility/Margin Selection Findings",
        "",
        "## Setup",
        "",
        f"- Model: `{meta['model']}`.",
        f"- Layer: `{meta['layer']}`.",
        f"- Images: {meta['images_evaluated']} held-out clean-correct CIFAR-10 images.",
        f"- Directions per image: {meta['directions_per_image']}.",
        f"- Probe radius: {meta['probe_eps_over_255']}/255.",
        f"- Full budget: {meta['attack_eps_over_255']}/255.",
        "",
        "Each direction is scored at the probe radius and evaluated at the full budget. "
        "This tests whether local mobility helps select directions that become adversarial, "
        "without using the final margin as the selector.",
        "",
        "## Selector Results",
        "",
        f"- Random direction ASR: {val('random_direction', 'asr'):.3f}.",
        f"- Mobility-only selector ASR: {val('probe_mobility', 'asr'):.3f}.",
        f"- Probe margin-drop selector ASR: {val('probe_margin_drop', 'asr'):.3f}.",
        f"- Mobility x margin selector ASR: {val('mobility_x_margin', 'asr'):.3f}.",
        f"- Flow-energy selector ASR: {val('probe_flow_energy', 'asr'):.3f}.",
        f"- Flow-energy x margin selector ASR: {val('flow_x_margin', 'asr'):.3f}.",
        "",
        "## Predictive Added Value",
        "",
    ]
    if len(margin_auc) and len(combo_auc):
        lines.append(f"- Margin-only held-out logistic AUROC: {float(margin_auc.test_auc.iloc[0]):.3f}.")
        lines.append(f"- Margin-only held-out logistic AUPRC: {float(margin_auc.test_auprc.iloc[0]):.3f}.")
        lines.append(f"- Margin + mobility held-out logistic AUROC: {float(combo_auc.test_auc.iloc[0]):.3f}.")
        lines.append(f"- Margin + mobility held-out logistic AUPRC: {float(combo_auc.test_auprc.iloc[0]):.3f}.")
    if len(all_auc):
        lines.append(f"- All-feature held-out logistic AUROC: {float(all_auc.test_auc.iloc[0]):.3f}.")
        lines.append(f"- All-feature held-out logistic AUPRC: {float(all_auc.test_auprc.iloc[0]):.3f}.")
    if not topk_summary.empty:
        lines += ["", "## Top-K Selector Results", ""]
        for selector in ["random_direction", "probe_mobility", "probe_margin_drop", "mobility_x_margin", "flow_x_margin"]:
            sub = topk_summary[topk_summary.selector == selector].sort_values("top_k")
            if sub.empty:
                continue
            vals = ", ".join(f"K={int(r.top_k)}: {float(r.topk_asr):.3f}" for r in sub.itertuples())
            lines.append(f"- {selector}: {vals}.")
    if not cond.empty:
        delta = cond.asr_delta_high_minus_low.mean()
        lines += [
            "",
            "## Margin-Matched Mobility Check",
            "",
            f"Across probe margin-drop bins, high-mobility directions exceed low-mobility directions by "
            f"{delta:.3f} mean ASR on average.",
        ]
    lines += [
        "",
        "## Interpretation Template",
        "",
        "If mobility x margin substantially exceeds margin-only, the two-stage proposal/selection "
        "claim is strengthened. If it does not, the safer interpretation is that mobility enriches "
        "candidate directions but margin/gradient information is the dominant selector.",
    ]
    (out_dir / "two_stage_mobility_margin_findings.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="analysis_outputs/hidden_jacobian_routing/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/jacobian_null_response/two_stage_mobility_margin_selection_bbb_resnet50_c200")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--directions-per-image", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--probe-eps", type=float, default=0.5, help="Probe radius in /255 units.")
    p.add_argument("--attack-eps", type=float, default=2.0, help="Full evaluation budget in /255 units.")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--flow-basis-sources", default="pgd,square")
    p.add_argument("--mobility-basis-sources", default="mobility_top_walk_square_budget")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--top-ks", default="1,5,10")
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, meta = evaluate_candidates(args)
    df.to_csv(out_dir / "two_stage_candidate_scores.csv", index=False)
    per_image, selector_summary = summarize_selectors(df)
    per_image.to_csv(out_dir / "two_stage_selector_per_image.csv", index=False)
    selector_summary.to_csv(out_dir / "two_stage_selector_summary.csv", index=False)
    topk_per_image, topk_summary = summarize_topk_selectors(df, parse_int_csv(args.top_ks), args.seed)
    topk_per_image.to_csv(out_dir / "two_stage_topk_selector_per_image.csv", index=False)
    topk_summary.to_csv(out_dir / "two_stage_topk_selector_summary.csv", index=False)
    topk_ci = bootstrap_topk_summary(topk_per_image, args.bootstrap, args.seed)
    topk_ci.to_csv(out_dir / "two_stage_topk_selector_bootstrap_ci.csv", index=False)
    pred = summarize_predictors(df)
    pred.to_csv(out_dir / "two_stage_predictive_models.csv", index=False)
    cond = conditional_bins(df, n_bins=5)
    cond.to_csv(out_dir / "two_stage_margin_matched_mobility_bins.csv", index=False)
    plot_results(selector_summary, pred, cond, out_dir, topk_summary)
    meta.update(
        {
            "n_candidates": int(len(df)),
            "random_direction_asr": float(df.full_success.mean()) if len(df) else None,
            "outputs": [
                "two_stage_candidate_scores.csv",
                "two_stage_selector_per_image.csv",
                "two_stage_selector_summary.csv",
                "two_stage_topk_selector_per_image.csv",
                "two_stage_topk_selector_summary.csv",
                "two_stage_topk_selector_bootstrap_ci.csv",
                "two_stage_predictive_models.csv",
                "two_stage_margin_matched_mobility_bins.csv",
                "two_stage_mobility_margin_selection.png",
                "two_stage_topk_selector_curves.png",
                "two_stage_mobility_margin_findings.md",
            ],
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    write_note(out_dir, meta, selector_summary, pred, cond, topk_summary)
    print(selector_summary.to_string(index=False))
    print(topk_summary.to_string(index=False))
    print(pred.to_string(index=False))


if __name__ == "__main__":
    main()
