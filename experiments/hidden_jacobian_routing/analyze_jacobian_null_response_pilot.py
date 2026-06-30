#!/usr/bin/env python3
"""Pilot tests for the hidden-Jacobian / realizable-control null hypothesis.

This script is intentionally confirmatory rather than pretty.  It asks whether
the current "adversarial success-flow" signal survives the strongest immediate
criticisms:

1. ambient feature-space random vectors are a weak null;
2. high-gain hidden-Jacobian directions may explain the learned basis;
3. success-only optimizer comparisons may be selection-biased;
4. step pooling may inflate rank and AUROC estimates.

The pilot runs on one CIFAR-10 model by default and writes a decision summary
plus raw-enough CSV/NPZ artifacts to support follow-up analyses.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import (  # noqa: E402
    load_model,
    margin,
    project_linf,
    select_clean_correct,
    square_trajectory,
)


LAYER_ORDER = {
    "bbb_resnet50": ["layer1", "layer2", "layer3", "layer4", "avgpool", "logits"],
    "bbb_vgg19_bn": ["block1", "block2", "block3", "block4", "block5", "penultimate", "logits"],
    "bbb_densenet": ["denseblock1", "denseblock2", "denseblock3", "penultimate", "logits"],
    "bbb_inception_v3": ["mixed5", "mixed6", "mixed7", "penultimate", "logits"],
}

DEFAULT_LAYERS = {
    "bbb_resnet50": "layer4,avgpool",
    "bbb_vgg19_bn": "block5,penultimate",
    "bbb_densenet": "denseblock3,penultimate",
    "bbb_inception_v3": "mixed6,penultimate",
}


def parse_csv(s: str, typ=str) -> list:
    return [typ(x.strip()) for x in s.split(",") if x.strip()]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward_features(wrapper, x: torch.Tensor):
    with torch.no_grad():
        logits, feats, _raw = wrapper.forward_with_features(x)
    return logits.detach(), {k: v.detach().cpu().numpy().astype(np.float32) for k, v in feats.items()}


def true_prob(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1).gather(1, y.view(-1, 1)).squeeze(1)


def pca_basis(x: np.ndarray, max_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x) < 2:
        raise ValueError("Need at least two vectors for PCA.")
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    xc = x - mean
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s * s
    ratio = var / np.clip(var.sum(), 1e-12, None)
    return mean, vt[: min(max_k, vt.shape[0])].astype(np.float32), ratio.astype(np.float64)


def pca_stats(x: np.ndarray) -> dict:
    if len(x) < 2:
        return {"pc1_var": np.nan, "pc10_cum_var": np.nan, "dim80": np.nan, "dim90": np.nan, "effective_rank": np.nan}
    _mean, _basis, ratio = pca_basis(x, min(len(x), x.shape[1]))
    csum = np.cumsum(ratio)
    entropy = -float(np.sum(ratio[ratio > 0] * np.log(ratio[ratio > 0])))
    return {
        "pc1_var": float(ratio[0]) if len(ratio) else np.nan,
        "pc10_cum_var": float(csum[min(9, len(csum) - 1)]) if len(csum) else np.nan,
        "dim80": int(np.searchsorted(csum, 0.8) + 1) if len(csum) else np.nan,
        "dim90": int(np.searchsorted(csum, 0.9) + 1) if len(csum) else np.nan,
        "effective_rank": float(np.exp(entropy)),
    }


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, basis.shape[0], x.shape[1])
    xc = x - mean
    coeff = xc @ basis[:kk].T
    return np.sum(coeff * coeff, axis=1) / np.clip(np.sum(xc * xc, axis=1), 1e-12, None)


def coeff_energy_profile(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, basis.shape[0], x.shape[1])
    xc = x - mean
    denom = np.clip(np.sum(xc * xc, axis=1, keepdims=True), 1e-12, None)
    coeff = xc @ basis[:kk].T
    prof = np.mean((coeff * coeff) / denom, axis=0)
    return prof / np.clip(np.sum(prof), 1e-12, None)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def safe_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    if np.nanstd(s) < 1e-12:
        return np.nan
    return float(roc_auc_score(y, s))


def subspace_overlap(a: np.ndarray, b: np.ndarray, k: int) -> dict:
    kk = min(k, a.shape[0], b.shape[0])
    if kk < 1:
        return {"projection_overlap": np.nan, "mean_principal_angle_deg": np.nan, "subspace_affinity": np.nan}
    s = np.linalg.svd(a[:kk] @ b[:kk].T, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return {
        "projection_overlap": float(np.sum(s * s) / kk),
        "mean_principal_angle_deg": float(np.mean(angles)),
        "subspace_affinity": float(np.linalg.norm(s) / math.sqrt(kk)),
    }


def bootstrap_auc(pos_df: pd.DataFrame, neg_df: pd.DataFrame, score: str, seed: int, reps: int) -> tuple[float, float, float]:
    if pos_df.empty or neg_df.empty:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    pos_imgs = pos_df["image_ord"].drop_duplicates().to_numpy()
    neg_imgs = neg_df["image_ord"].drop_duplicates().to_numpy()
    if len(pos_imgs) < 2 or len(neg_imgs) < 2:
        return np.nan, np.nan, np.nan
    vals = []
    for _ in range(reps):
        ps = rng.choice(pos_imgs, size=len(pos_imgs), replace=True)
        ns = rng.choice(neg_imgs, size=len(neg_imgs), replace=True)
        p = pos_df[pos_df.image_ord.isin(ps)].groupby("image_ord")[score].mean().to_numpy()
        n = neg_df[neg_df.image_ord.isin(ns)].groupby("image_ord")[score].mean().to_numpy()
        vals.append(safe_auc(p, n))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(vals)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def states_to_layer_vectors(wrapper, states: list[torch.Tensor], y: torch.Tensor, layers: list[str]):
    state_meta = []
    feats_by_step = []
    for step, state in enumerate(states):
        logits, feats = forward_features(wrapper, state)
        pred = int(logits.argmax(1).item())
        state_meta.append(
            {
                "state_step": int(step),
                "pred": pred,
                "step_success": int(pred != int(y.item())),
                "margin": float(margin(logits, y).item()),
                "true_prob": float(true_prob(logits, y).item()),
            }
        )
        feats_by_step.append(feats)

    vectors = {layer: [] for layer in layers}
    rows = {layer: [] for layer in layers}
    for step in range(len(feats_by_step) - 1):
        for layer in layers:
            if layer not in feats_by_step[step] or layer not in feats_by_step[step + 1]:
                continue
            v = feats_by_step[step + 1][layer][0] - feats_by_step[step][layer][0]
            vectors[layer].append(v.astype(np.float32))
            rows[layer].append(
                {
                    "step": int(step),
                    "pred_before": int(state_meta[step]["pred"]),
                    "pred_after": int(state_meta[step + 1]["pred"]),
                    "step_success_before": int(state_meta[step]["step_success"]),
                    "step_success_after": int(state_meta[step + 1]["step_success"]),
                    "margin_before": float(state_meta[step]["margin"]),
                    "margin_after": float(state_meta[step + 1]["margin"]),
                    "true_prob_before": float(state_meta[step]["true_prob"]),
                    "true_prob_after": float(state_meta[step + 1]["true_prob"]),
                }
            )
    return rows, vectors


def pgd_states(wrapper, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int, step_size: float):
    x0 = x.detach()
    x_adv = x0.clone()
    states = [x_adv.detach().clone()]
    pred_feature_states = []
    for _ in range(steps):
        probe = x_adv.detach().requires_grad_(True)
        logits = wrapper(probe)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, probe)[0]
        x_next = project_linf(x_adv + step_size * grad.sign(), x0, eps)
        pred_feature_states.append((x_adv.detach().clone(), x_next.detach().clone()))
        x_adv = x_next.detach()
        states.append(x_adv.detach().clone())
    return states, pred_feature_states


def random_walk_states(
    x: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    seed: int,
    *,
    correlated: bool = False,
    rho: float = 0.9,
):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    x0 = x.detach()
    x_adv = x0.clone()
    states = [x_adv.detach().clone()]
    direction = torch.zeros_like(x0)
    for _ in range(steps):
        noise = torch.randn(x0.shape, generator=gen, device=x.device)
        if correlated:
            direction = rho * direction + math.sqrt(max(1.0 - rho * rho, 1e-6)) * noise
            step_dir = direction.sign()
        else:
            step_dir = noise.sign()
        x_adv = project_linf(x_adv + step_size * step_dir, x0, eps)
        states.append(x_adv.detach().clone())
    return states


def mobility_top_walk_states(
    wrapper,
    x: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    seed: int,
    primary_layer: str,
    candidates: int,
):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    x0 = x.detach()
    x_adv = x0.clone()
    states = [x_adv.detach().clone()]
    for _ in range(steps):
        with torch.no_grad():
            _logits0, feats0, _raw0 = wrapper.forward_with_features(x_adv)
        h0 = feats0[primary_layer].detach()
        cand_states = []
        cand_scores = []
        for _j in range(candidates):
            noise = torch.randn(x0.shape, generator=gen, device=x.device).sign()
            cand = project_linf(x_adv + step_size * noise, x0, eps)
            with torch.no_grad():
                _logits, feats, _raw = wrapper.forward_with_features(cand)
            score = torch.norm((feats[primary_layer] - h0).flatten(1), dim=1).item()
            cand_states.append(cand.detach())
            cand_scores.append(score)
        x_adv = cand_states[int(np.argmax(cand_scores))]
        states.append(x_adv.detach().clone())
    return states


def jacobian_probe_vectors(
    wrapper,
    x: torch.Tensor,
    layers: list[str],
    eps_probe: float,
    n_probes: int,
    seed: int,
    batch_size: int,
):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    with torch.no_grad():
        _logits0, feats0, _raw0 = wrapper.forward_with_features(x)
    vals = {layer: [] for layer in layers}
    mobility = {layer: [] for layer in layers}
    remaining = n_probes
    while remaining > 0:
        bs = min(batch_size, remaining)
        signs = torch.randn((bs,) + tuple(x.shape[1:]), generator=gen, device=x.device).sign()
        xb = (x + eps_probe * signs).clamp(0, 1)
        with torch.no_grad():
            _logits, feats, _raw = wrapper.forward_with_features(xb)
        for layer in layers:
            if layer not in feats or layer not in feats0:
                continue
            disp = feats[layer].detach().cpu().numpy().astype(np.float32) - feats0[layer].detach().cpu().numpy().astype(np.float32)
            vals[layer].append(disp)
            mobility[layer].append(np.linalg.norm(disp, axis=1))
        remaining -= bs
    out = {}
    out_mob = {}
    for layer in layers:
        if vals[layer]:
            out[layer] = np.concatenate(vals[layer], axis=0)
            out_mob[layer] = np.concatenate(mobility[layer], axis=0)
    return out, out_mob


def select_train_val_test(image_ords: list[int], seed: int):
    rng = np.random.default_rng(seed)
    ids = np.asarray(sorted(set(image_ords)), dtype=int)
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(round(0.5 * n)))
    n_val = max(1, int(round(0.25 * n)))
    train = set(ids[:n_train].tolist())
    val = set(ids[n_train : n_train + n_val].tolist())
    test = set(ids[n_train + n_val :].tolist())
    return train, val, test


def append_segment(
    *,
    rows: list[dict],
    arrays: dict[str, list[np.ndarray]],
    model: str,
    source: str,
    layer: str,
    image_ord: int,
    dataset_idx: int,
    label: int,
    final_success: int,
    final_pred: int,
    row_meta: dict,
    vec: np.ndarray,
):
    key = f"{model}__{source}__{layer}"
    idx = len(arrays[key])
    arrays[key].append(vec.astype(np.float32))
    row = {
        "model": model,
        "source": source,
        "layer": layer,
        "image_ord": int(image_ord),
        "dataset_idx": int(dataset_idx),
        "label": int(label),
        "final_success": int(final_success),
        "final_pred": int(final_pred),
        "vector_key": key,
        "vector_idx": int(idx),
    }
    row.update(row_meta)
    rows.append(row)


def collect_trajectories(args, wrapper, dataset, selected, layers, device):
    pgd_eps_255 = args.pgd_eps if args.pgd_eps > 0 else args.eps
    square_eps_255 = args.square_eps if args.square_eps > 0 else args.eps
    pgd_eps = pgd_eps_255 / 255.0
    square_eps = square_eps_255 / 255.0
    step_size = args.step_size / 255.0
    eps_probe = args.probe_eps / 255.0
    primary_layer = args.primary_layer or layers[0]
    rows: list[dict] = []
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    image_rows = []

    for image_ord, (dataset_idx, label) in enumerate(selected):
        x_cpu, _ = dataset[dataset_idx]
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)

        with torch.no_grad():
            clean_logits = wrapper(x)
            clean_pred = int(clean_logits.argmax(1).item())
            clean_margin = float(margin(clean_logits, y).item())

        # Adversarial optimizers.
        pgd, pgd_pred_pairs = pgd_states(wrapper, x, y, pgd_eps, args.pgd_steps, step_size)
        sq = square_trajectory(
            wrapper,
            x,
            y,
            square_eps,
            args.square_steps,
            args.seed + image_ord * 1009 + 17,
            args.square_p_init,
            args.square_init_epochs,
            args.square_checkpoints,
        )
        attack_states = {"pgd": pgd, "square": sq}
        with torch.no_grad():
            pgd_final_logits = wrapper(pgd[-1])
        pgd_final_pred = int(pgd_final_logits.argmax(1).item())
        pgd_final_success = int(pgd_final_pred != int(label))

        # Realizable non-adversarial / objective-neutral trajectories.
        control_budgets = [
            ("pgd_budget", pgd_eps, args.pgd_steps, step_size, 23),
        ]
        square_control_steps = max(args.square_checkpoints - 1, 1)
        if abs(square_eps - pgd_eps) > 1e-12 or square_control_steps != args.pgd_steps:
            control_budgets.append(("square_budget", square_eps, square_control_steps, min(step_size, square_eps), 53))
        for budget_name, budget_eps, budget_steps, budget_step_size, seed_offset in control_budgets:
            attack_states[f"random_sign_walk_{budget_name}"] = random_walk_states(
                x,
                budget_eps,
                budget_steps,
                budget_step_size,
                args.seed + image_ord * 1009 + seed_offset,
                correlated=False,
            )
            attack_states[f"correlated_random_walk_{budget_name}"] = random_walk_states(
                x,
                budget_eps,
                budget_steps,
                budget_step_size,
                args.seed + image_ord * 1009 + seed_offset + 6,
                correlated=True,
                rho=args.random_walk_rho,
            )
            attack_states[f"mobility_top_walk_{budget_name}"] = mobility_top_walk_states(
                wrapper,
                x,
                budget_eps,
                budget_steps,
                budget_step_size,
                args.seed + image_ord * 1009 + seed_offset + 8,
                primary_layer,
                args.mobility_candidates,
            )

        for source, states in attack_states.items():
            with torch.no_grad():
                final_logits = wrapper(states[-1])
            final_pred = int(final_logits.argmax(1).item())
            final_success = int(final_pred != int(label))
            image_rows.append(
                {
                    "model": args.model,
                    "source": source,
                    "image_ord": int(image_ord),
                    "dataset_idx": int(dataset_idx),
                    "label": int(label),
                    "clean_pred": clean_pred,
                    "clean_margin": clean_margin,
                    "final_pred": final_pred,
                    "final_success": final_success,
                    "n_states": int(len(states)),
                }
            )
            layer_rows, layer_vecs = states_to_layer_vectors(wrapper, states, y, layers)
            for layer in layers:
                for meta, vec in zip(layer_rows.get(layer, []), layer_vecs.get(layer, [])):
                    append_segment(
                        rows=rows,
                        arrays=arrays,
                        model=args.model,
                        source=source,
                        layer=layer,
                        image_ord=image_ord,
                        dataset_idx=dataset_idx,
                        label=label,
                        final_success=final_success,
                        final_pred=final_pred,
                        row_meta=meta,
                        vec=vec,
                    )

        # Actual PGD gradient-induced feature motion. For deterministic PGD this
        # should closely match the logged PGD step; it is kept as an explicit null.
        for step, (xa, xb) in enumerate(pgd_pred_pairs):
            row_meta = {
                "step": int(step),
                "pred_before": -1,
                "pred_after": -1,
                "step_success_before": -1,
                "step_success_after": -1,
                "margin_before": np.nan,
                "margin_after": np.nan,
                "true_prob_before": np.nan,
                "true_prob_after": np.nan,
            }
            _logits_a, feats_a = forward_features(wrapper, xa)
            _logits_b, feats_b = forward_features(wrapper, xb)
            for layer in layers:
                if layer in feats_a and layer in feats_b:
                    append_segment(
                        rows=rows,
                        arrays=arrays,
                        model=args.model,
                        source="pgd_predicted_gradient_step",
                        layer=layer,
                        image_ord=image_ord,
                        dataset_idx=dataset_idx,
                        label=label,
                        final_success=pgd_final_success,
                        final_pred=pgd_final_pred,
                        row_meta=row_meta,
                        vec=(feats_b[layer][0] - feats_a[layer][0]).astype(np.float32),
                    )

        # Finite-difference Jacobian probe vectors at the clean image.
        probes, probe_mob = jacobian_probe_vectors(
            wrapper,
            x,
            layers,
            eps_probe,
            args.jacobian_probes,
            args.seed + image_ord * 1009 + 37,
            args.probe_batch_size,
        )
        for layer, arr in probes.items():
            mob = probe_mob[layer]
            q = np.quantile(mob, args.high_mobility_quantile)
            for j, vec in enumerate(arr):
                append_segment(
                    rows=rows,
                    arrays=arrays,
                    model=args.model,
                    source="jacobian_probe_all",
                    layer=layer,
                    image_ord=image_ord,
                    dataset_idx=dataset_idx,
                    label=label,
                    final_success=0,
                    final_pred=-1,
                    row_meta={
                        "step": int(j),
                        "pred_before": clean_pred,
                        "pred_after": -1,
                        "step_success_before": 0,
                        "step_success_after": -1,
                        "margin_before": clean_margin,
                        "margin_after": np.nan,
                        "true_prob_before": np.nan,
                        "true_prob_after": np.nan,
                    },
                    vec=vec,
                )
                if mob[j] >= q:
                    append_segment(
                        rows=rows,
                        arrays=arrays,
                        model=args.model,
                        source="jacobian_probe_top_mobility",
                        layer=layer,
                        image_ord=image_ord,
                        dataset_idx=dataset_idx,
                        label=label,
                        final_success=0,
                        final_pred=-1,
                        row_meta={
                            "step": int(j),
                            "pred_before": clean_pred,
                            "pred_after": -1,
                            "step_success_before": 0,
                            "step_success_after": -1,
                            "margin_before": clean_margin,
                            "margin_after": np.nan,
                            "true_prob_before": np.nan,
                            "true_prob_after": np.nan,
                        },
                        vec=vec,
                    )

        if (image_ord + 1) % max(args.partial_every, 1) == 0:
            print(f"[COLLECT] {image_ord + 1}/{len(selected)} images", flush=True)

    return pd.DataFrame(rows), {k: np.stack(v).astype(np.float32) for k, v in arrays.items() if v}, pd.DataFrame(image_rows)


def save_vectors_npz(out_dir: Path, arrays: dict[str, np.ndarray]):
    safe = {k.replace("/", "_"): v for k, v in arrays.items()}
    np.savez_compressed(out_dir / "segment_vectors.npz", **safe)


def vector_matrix(rows: pd.DataFrame, arrays: dict[str, np.ndarray]) -> np.ndarray:
    mats = []
    for key, group in rows.groupby("vector_key", sort=False):
        arr = arrays[key]
        mats.append(arr[group["vector_idx"].to_numpy(dtype=int)])
    if not mats:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(mats, axis=0)


def rows_for(rows: pd.DataFrame, arrays: dict[str, np.ndarray], model: str, source: str, layer: str):
    sub = rows[(rows.model == model) & (rows.source == source) & (rows.layer == layer)].copy()
    if sub.empty:
        return sub, np.zeros((0, 0), dtype=np.float32)
    key = f"{model}__{source}__{layer}"
    return sub, arrays[key][sub["vector_idx"].to_numpy(dtype=int)]


def fit_success_bases(rows: pd.DataFrame, arrays: dict[str, np.ndarray], layers: list[str], train_images: set[int], args):
    bases = {}
    basis_sources = ["pgd", "square"]
    for source in basis_sources:
        for layer in layers:
            sub, x = rows_for(rows, arrays, args.model, source, layer)
            if sub.empty:
                continue
            mask = sub["image_ord"].isin(train_images).to_numpy() & (sub["final_success"].to_numpy(dtype=int) == 1)
            if mask.sum() < max(args.k + 2, 8):
                continue
            mean, basis, ratio = pca_basis(x[mask], args.k)
            bases[(source, layer)] = {"mean": mean, "basis": basis, "ratio": ratio, "n": int(mask.sum())}
    return bases


def fit_source_basis(rows: pd.DataFrame, arrays: dict[str, np.ndarray], args, source: str, layer: str, train_images: set[int]):
    sub, x = rows_for(rows, arrays, args.model, source, layer)
    if sub.empty:
        return None
    mask = sub["image_ord"].isin(train_images).to_numpy()
    if source in {"pgd", "square"}:
        mask &= sub["final_success"].to_numpy(dtype=int) == 1
    if mask.sum() < max(args.k + 2, 8):
        return None
    mean, basis, ratio = pca_basis(x[mask], args.k)
    return {"mean": mean, "basis": basis, "ratio": ratio, "n": int(mask.sum())}


def analyze_projection_and_overlap(rows, arrays, image_rows, layers, train, test, args):
    bases = fit_success_bases(rows, arrays, layers, train, args)
    metric_rows = []
    overlap_rows = []
    score_rows = []
    all_sources = sorted(rows["source"].unique())
    strong_nulls = [
        "random_sign_walk",
        "random_sign_walk_pgd_budget",
        "random_sign_walk_square_budget",
        "correlated_random_walk",
        "correlated_random_walk_pgd_budget",
        "correlated_random_walk_square_budget",
        "mobility_top_walk",
        "mobility_top_walk_pgd_budget",
        "mobility_top_walk_square_budget",
        "jacobian_probe_all",
        "jacobian_probe_top_mobility",
        "pgd_predicted_gradient_step",
    ]

    for (basis_source, layer), basis_obj in bases.items():
        basis = basis_obj["basis"]
        mean = basis_obj["mean"]
        success_sub, success_x = rows_for(rows, arrays, args.model, basis_source, layer)
        success_test = success_sub["image_ord"].isin(test).to_numpy() & (success_sub["final_success"].to_numpy(dtype=int) == 1)
        fail_test = success_sub["image_ord"].isin(test).to_numpy() & (success_sub["final_success"].to_numpy(dtype=int) == 0)
        success_scores = projection_energy(success_x[success_test], mean, basis, args.k) if success_test.any() else np.array([])
        fail_scores = projection_energy(success_x[fail_test], mean, basis, args.k) if fail_test.any() else np.array([])

        for source in all_sources:
            sub, x = rows_for(rows, arrays, args.model, source, layer)
            if sub.empty:
                continue
            test_mask = sub["image_ord"].isin(test).to_numpy()
            if not test_mask.any():
                continue
            scores = projection_energy(x[test_mask], mean, basis, args.k)
            sub_test = sub[test_mask].copy()
            sub_test["projection_energy"] = scores
            sub_test["basis_source"] = basis_source
            score_rows.append(sub_test)

            if source == basis_source:
                pos = success_scores
                neg = fail_scores
                comparison = "success_vs_failed_same_optimizer"
            else:
                pos = success_scores
                neg = scores
                comparison = f"{basis_source}_success_vs_{source}"

            metric_rows.append(
                {
                    "model": args.model,
                    "basis_source": basis_source,
                    "comparison_source": source,
                    "comparison": comparison,
                    "layer": layer,
                    "k": args.k,
                    "n_basis_train_success": int(basis_obj["n"]),
                    "n_pos_success_segments": int(len(pos)),
                    "n_neg_segments": int(len(neg)),
                    "auroc": safe_auc(pos, neg),
                    "pos_mean_energy": float(np.mean(pos)) if len(pos) else np.nan,
                    "neg_mean_energy": float(np.mean(neg)) if len(neg) else np.nan,
                    "is_strong_null": int(source in strong_nulls),
                }
            )

        for null_source in strong_nulls + ["pgd", "square"]:
            null_basis = fit_source_basis(rows, arrays, args, null_source, layer, train)
            if null_basis is None:
                continue
            ov = subspace_overlap(basis, null_basis["basis"], args.k)
            overlap_rows.append(
                {
                    "model": args.model,
                    "basis_source": basis_source,
                    "other_source": null_source,
                    "layer": layer,
                    "k": args.k,
                    "basis_n": int(basis_obj["n"]),
                    "other_n": int(null_basis["n"]),
                    **ov,
                }
            )

    scores_df = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    return pd.DataFrame(metric_rows), pd.DataFrame(overlap_rows), scores_df


def analyze_dimensionality(rows, arrays, layers, train, args):
    out = []
    for source in sorted(rows.source.unique()):
        for layer in layers:
            sub, x = rows_for(rows, arrays, args.model, source, layer)
            if sub.empty:
                continue
            mask = sub["image_ord"].isin(train).to_numpy()
            if source in {"pgd", "square"}:
                mask &= sub["final_success"].to_numpy(dtype=int) == 1
            if mask.sum() < 8:
                continue
            stats = pca_stats(x[mask])
            out.append({"model": args.model, "source": source, "layer": layer, "n": int(mask.sum()), **stats})
    return pd.DataFrame(out)


def analyze_all_run_similarity(rows, arrays, layers, train, test, args):
    bases = fit_success_bases(rows, arrays, layers, train, args)
    out = []
    for layer in layers:
        basis_obj = bases.get(("pgd", layer)) or bases.get(("square", layer))
        if basis_obj is None:
            continue
        mean, basis = basis_obj["mean"], basis_obj["basis"]
        profiles = {}
        for source in ["pgd", "square", "random_sign_walk", "mobility_top_walk"]:
            sub, x = rows_for(rows, arrays, args.model, source, layer)
            if sub.empty:
                continue
            test_mask = sub["image_ord"].isin(test).to_numpy()
            if not test_mask.any():
                continue
            for mode, mask_extra in {
                "all_runs": np.ones(len(sub), dtype=bool),
                "success_only": sub["final_success"].to_numpy(dtype=int) == 1,
                "failed_only": sub["final_success"].to_numpy(dtype=int) == 0,
            }.items():
                mask = test_mask & mask_extra
                if mask.sum() < 2:
                    continue
                profiles[(source, mode)] = coeff_energy_profile(x[mask], mean, basis, args.signature_k)

        keys = sorted(profiles)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                out.append(
                    {
                        "model": args.model,
                        "layer": layer,
                        "basis_source": "pgd" if ("pgd", layer) in bases else "square",
                        "source_a": a[0],
                        "mode_a": a[1],
                        "source_b": b[0],
                        "mode_b": b[1],
                        "signature_k": args.signature_k,
                        "cosine": cosine(profiles[a], profiles[b]),
                    }
                )
    return pd.DataFrame(out)


def image_level_ci(scores_df: pd.DataFrame, args):
    rows = []
    if scores_df.empty:
        return pd.DataFrame()
    for (basis_source, layer), g in scores_df.groupby(["basis_source", "layer"]):
        pos = g[(g.source == basis_source) & (g.final_success == 1)]
        for source, neg in g.groupby("source"):
            if source == basis_source:
                neg = g[(g.source == source) & (g.final_success == 0)]
                comparison = "success_vs_failed_same_optimizer"
            else:
                comparison = f"{basis_source}_success_vs_{source}"
            mean, lo, hi = bootstrap_auc(pos, neg, "projection_energy", args.seed, args.bootstrap_reps)
            rows.append(
                {
                    "basis_source": basis_source,
                    "layer": layer,
                    "comparison_source": source,
                    "comparison": comparison,
                    "image_bootstrap_auroc_mean": mean,
                    "image_bootstrap_auroc_lo": lo,
                    "image_bootstrap_auroc_hi": hi,
                    "pos_images": int(pos["image_ord"].nunique()),
                    "neg_images": int(neg["image_ord"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def write_decision_summary(out_dir: Path, metrics: pd.DataFrame, overlaps: pd.DataFrame, sim: pd.DataFrame, ci: pd.DataFrame):
    lines = [
        "# Pilot Decision Summary",
        "",
        "This pilot treats the rejection critique as true until disproven.",
        "",
    ]
    flags = []
    if metrics.empty:
        flags.append("No projection metrics were produced.")
    else:
        strong = metrics[metrics["is_strong_null"] == 1].copy()
        if not strong.empty:
            max_null = strong.groupby(["basis_source", "layer"])["auroc"].max().dropna()
            lines.append("## Strong-null AUROC snapshot")
            lines.append("")
            for (basis_source, layer), val in max_null.items():
                lines.append(f"- {basis_source} basis, {layer}: max success-vs-null AUROC = {val:.3f}")
            if (max_null < 0.65).any():
                flags.append("At least one basis/layer has weak separation from strong realizable/Jacobian controls.")
        sf = metrics[metrics["comparison"] == "success_vs_failed_same_optimizer"]
        if not sf.empty:
            lines.append("")
            lines.append("## Success-vs-failed snapshot")
            lines.append("")
            for r in sf.itertuples():
                val = "nan" if pd.isna(r.auroc) else f"{r.auroc:.3f}"
                lines.append(f"- {r.basis_source} basis, {r.layer}: AUROC = {val}")
    if not overlaps.empty:
        jac = overlaps[overlaps["other_source"].str.contains("jacobian", na=False)]
        if not jac.empty:
            lines.append("")
            lines.append("## Jacobian-overlap snapshot")
            lines.append("")
            for r in jac.itertuples():
                lines.append(
                    f"- {r.basis_source} vs {r.other_source}, {r.layer}: overlap={r.projection_overlap:.3f}, angle={r.mean_principal_angle_deg:.1f} deg"
                )
            if (jac["projection_overlap"] > 0.5).any():
                flags.append("Success-flow has high overlap with at least one Jacobian probe basis.")
    if not sim.empty:
        ps = sim[(sim.source_a == "pgd") & (sim.source_b == "square")]
        if ps.empty:
            ps = sim[(sim.source_a == "square") & (sim.source_b == "pgd")]
        if not ps.empty:
            lines.append("")
            lines.append("## PGD/Square all-run signature snapshot")
            lines.append("")
            for r in ps.itertuples():
                lines.append(f"- {r.layer}, {r.mode_a} vs {r.mode_b}: cosine={r.cosine:.3f}")
    lines.append("")
    lines.append("## Preliminary Read")
    lines.append("")
    if flags:
        lines.append("Potential problems detected:")
        for f in flags:
            lines.append(f"- {f}")
        lines.append("")
        lines.append("Do not scale before inspecting these failures. A pivot toward hidden-Jacobian high-gain geometry may be needed.")
    else:
        lines.append("No immediate fatal pilot failure detected. Scale-up is plausible, but inspect the CSVs before changing the manuscript.")
    (out_dir / "pilot_decision_summary.md").write_text("\n".join(lines) + "\n")


def run(args):
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    wrapper = load_model(args.model, device).eval()
    layers = parse_csv(args.layers or DEFAULT_LAYERS.get(args.model, "logits"))
    missing = [l for l in layers if l not in LAYER_ORDER.get(args.model, layers)]
    if missing:
        print(f"[WARN] Requested layers not in registry for {args.model}: {missing}", flush=True)

    selected = select_clean_correct(dataset, {args.model: wrapper}, argparse.Namespace(models=[args.model], images=args.images), device)
    print(f"[PILOT] model={args.model} clean_correct={len(selected)} layers={layers} device={device}", flush=True)
    rows, arrays, image_rows = collect_trajectories(args, wrapper, dataset, selected, layers, device)
    rows.to_csv(out_dir / "segment_metadata.csv", index=False)
    image_rows.to_csv(out_dir / "image_outcomes.csv", index=False)
    save_vectors_npz(out_dir, arrays)

    train, val, test = select_train_val_test(rows["image_ord"].unique().tolist(), args.seed)
    split = pd.DataFrame(
        [{"image_ord": i, "split": "train"} for i in sorted(train)]
        + [{"image_ord": i, "split": "val"} for i in sorted(val)]
        + [{"image_ord": i, "split": "test"} for i in sorted(test)]
    )
    split.to_csv(out_dir / "image_splits.csv", index=False)

    metrics, overlaps, scores = analyze_projection_and_overlap(rows, arrays, image_rows, layers, train, test, args)
    dim = analyze_dimensionality(rows, arrays, layers, train, args)
    sim = analyze_all_run_similarity(rows, arrays, layers, train, test, args)
    ci = image_level_ci(scores, args)

    metrics.to_csv(out_dir / "pilot_core_metrics.csv", index=False)
    if "is_strong_null" in metrics.columns:
        metrics[metrics["is_strong_null"] == 1].to_csv(out_dir / "pilot_realisable_random_metrics.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "pilot_realisable_random_metrics.csv", index=False)
    overlaps.to_csv(out_dir / "pilot_jacobian_overlap.csv", index=False)
    dim.to_csv(out_dir / "pilot_dimensionality.csv", index=False)
    sim.to_csv(out_dir / "pilot_all_run_optimizer_metrics.csv", index=False)
    ci.to_csv(out_dir / "pilot_image_level_ci.csv", index=False)
    if not scores.empty:
        scores.to_csv(out_dir / "pilot_projection_scores.csv", index=False)

    meta = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "output_dir": str(out_dir),
        "model": args.model,
        "images_requested": args.images,
        "images_selected": len(selected),
        "layers": layers,
        "eps": args.eps,
        "pgd_eps": args.pgd_eps if args.pgd_eps > 0 else args.eps,
        "square_eps": args.square_eps if args.square_eps > 0 else args.eps,
        "pgd_steps": args.pgd_steps,
        "step_size": args.step_size,
        "square_steps": args.square_steps,
        "square_checkpoints": args.square_checkpoints,
        "jacobian_probes": args.jacobian_probes,
        "mobility_candidates": args.mobility_candidates,
        "k": args.k,
        "signature_k": args.signature_k,
        "seed": args.seed,
        "device": str(device),
        "outputs": {
            "segment_metadata": "segment_metadata.csv",
            "segment_vectors": "segment_vectors.npz",
            "core_metrics": "pilot_core_metrics.csv",
            "jacobian_overlap": "pilot_jacobian_overlap.csv",
            "realizable_random_metrics": "pilot_realisable_random_metrics.csv",
            "all_run_optimizer_metrics": "pilot_all_run_optimizer_metrics.csv",
            "image_level_ci": "pilot_image_level_ci.csv",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    write_decision_summary(out_dir, metrics, overlaps, sim, ci)
    print(f"[DONE] wrote {out_dir}", flush=True)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/jacobian_null_response/pilot_bbb_resnet50_c100")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layers", default="", help="Comma-separated layer list. Default uses model-specific hidden+penultimate.")
    p.add_argument("--primary-layer", default="", help="Layer used to select mobility-top random walk candidates.")
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--eps", type=float, default=8.0, help="Linf epsilon in /255 units.")
    p.add_argument("--pgd-eps", type=float, default=-1.0, help="Optional PGD epsilon in /255 units; defaults to --eps.")
    p.add_argument("--square-eps", type=float, default=-1.0, help="Optional Square epsilon in /255 units; defaults to --eps.")
    p.add_argument("--step-size", type=float, default=2.0, help="Step size in /255 units.")
    p.add_argument("--pgd-steps", type=int, default=20)
    p.add_argument("--square-steps", type=int, default=1000)
    p.add_argument("--square-checkpoints", type=int, default=21)
    p.add_argument("--square-p-init", type=float, default=0.8)
    p.add_argument("--square-init-epochs", type=int, default=0)
    p.add_argument("--random-walk-rho", type=float, default=0.9)
    p.add_argument("--mobility-candidates", type=int, default=8)
    p.add_argument("--jacobian-probes", type=int, default=64)
    p.add_argument("--probe-eps", type=float, default=2.0, help="Finite-difference probe radius in /255 units.")
    p.add_argument("--probe-batch-size", type=int, default=32)
    p.add_argument("--high-mobility-quantile", type=float, default=0.8)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--signature-k", type=int, default=5)
    p.add_argument("--bootstrap-reps", type=int, default=200)
    p.add_argument("--partial-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
