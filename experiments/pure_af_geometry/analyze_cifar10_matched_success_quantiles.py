#!/usr/bin/env python3
"""Matched-margin CIFAR-10 success-flow geometry diagnostic.

This tests whether success-flow separability in robust models survives when
successful and failed trajectory segments are compared at matched margin
quantiles, rather than comparing successful attacks only against generic clean
motions or random directions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.load_models import load_cifar_model


KS = [5, 10, 20, 50, 100]
QUANTILE_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


class MultiFeatureWrapper(nn.Module):
    def __init__(self, name: str, model: nn.Module, model_group: str):
        super().__init__()
        self.name = name
        self.model = model
        self.model_group = model_group
        self.enabled = False
        self.captures: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.handles = []
        self.labels = self._register_hooks()

    def _register_label(self, module_name: str, label: str) -> bool:
        modules = dict(self.model.named_modules())
        if module_name not in modules:
            return False
        self.handles.append(modules[module_name].register_forward_hook(self._make_hook(label)))
        return True

    def _make_hook(self, label: str):
        def hook(_module, _inp, out):
            if self.enabled and torch.is_tensor(out):
                self.captures[label].append(out)

        return hook

    def _register_hooks(self) -> list[str]:
        labels = []
        names = set(dict(self.model.named_modules()))
        if "bbb_resnet50" in self.name:
            for label, module_name in [
                ("layer2", "1.layer2"),
                ("layer3", "1.layer3"),
                ("layer4", "1.layer4"),
                ("avgpool", "1.layer4"),
            ]:
                if self._register_label(module_name, label):
                    labels.append(label)
        elif "bbb_vgg19_bn" in self.name:
            for label, module_name in [
                ("final_conv_block", "1.features.51"),
                ("penultimate_feature", "1.features"),
            ]:
                if self._register_label(module_name, label):
                    labels.append(label)
        elif any(n.endswith("layer4") for n in names):
            prefixes = ["", "branch1.", "branch2.", "branch3.", "branch4."]
            for label in ["layer2", "layer3", "layer4"]:
                ok = False
                for prefix in prefixes:
                    ok = self._register_label(f"{prefix}{label}", label) or ok
                if ok:
                    labels.append(label)
            if "layer4" in labels:
                for prefix in prefixes:
                    self._register_label(f"{prefix}layer4", "avgpool")
                labels.append("avgpool")
        if not labels:
            # Fallback: final convolutional output. This should rarely be used,
            # but keeps the script robust to new CIFAR checkpoints.
            last_conv_name = None
            for module_name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    last_conv_name = module_name
            if last_conv_name is None:
                raise RuntimeError(f"No hookable feature module found for {self.name}")
            self._register_label(last_conv_name, "final_conv")
            labels.append("final_conv")
        return labels

    @staticmethod
    def _pool_feature(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x.flatten(1)

    def _aggregate_label(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        pooled = [self._pool_feature(out) for out in tensors]
        if len(pooled) == 1:
            return pooled[0]
        if all(p.shape == pooled[0].shape for p in pooled):
            return torch.stack(pooled, dim=0).mean(dim=0)
        return torch.cat(pooled, dim=1)

    def forward(self, x):
        return self.model(x)

    def forward_with_features(self, x):
        self.captures = defaultdict(list)
        self.enabled = True
        try:
            logits = self.model(x)
        finally:
            self.enabled = False
        feats = {}
        for label in self.labels:
            outs = self.captures.get(label, [])
            if not outs:
                continue
            feats[label] = self._aggregate_label(outs)
        if not feats:
            raise RuntimeError(f"No features captured for {self.name}")
        return logits, feats, dict(self.captures)

    def aggregate_grads(self, raw_by_label: dict[str, list[torch.Tensor]], raw_grads: list[torch.Tensor | None]):
        out = {}
        cursor = 0
        for label in self.labels:
            raws = raw_by_label.get(label, [])
            if not raws:
                continue
            grads = []
            for raw in raws:
                g = raw_grads[cursor]
                cursor += 1
                if g is None:
                    g = torch.zeros_like(raw)
                grads.append(g)
            out[label] = self._aggregate_label(grads)
        return out

    def close(self):
        for handle in self.handles:
            handle.remove()


def load_model(spec: str, device):
    if spec.startswith("robustbench:"):
        rb_name = spec.split(":", 1)[1]
        from robustbench.utils import load_model as rb_load_model

        model = rb_load_model(rb_name, model_dir="checkpoints/robustbench_cifar10", norm="Linf").to(device).eval()
        return MultiFeatureWrapper(rb_name, model, "robust").to(device).eval()
    model = load_cifar_model(spec).to(device).eval()
    return MultiFeatureWrapper(spec, model, "standard").to(device).eval()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def project_linf(x: torch.Tensor, clean: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x, clean + eps), clean - eps).clamp(0, 1)


def margin_and_py(logits: torch.Tensor, y: torch.Tensor):
    probs = F.softmax(logits, dim=1)
    true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.view(-1, 1), -1e9)
    other = masked.max(dim=1).values
    return (true_logits - other).detach(), probs.gather(1, y.view(-1, 1)).squeeze(1).detach()


def clean_motion_variants(x: torch.Tensor, gen: torch.Generator):
    return [
        ("crop", TF.resized_crop(x, 2, 2, 28, 28, [32, 32], antialias=True)),
        ("color", TF.adjust_contrast(TF.adjust_brightness(x, 1.2), 0.85).clamp(0, 1)),
        ("blur", TF.gaussian_blur(x, [5, 5], [0.8, 0.8])),
        ("noise", (x + torch.randn(x.shape, generator=gen) * 0.03).clamp(0, 1)),
    ]


def pca_basis(x: np.ndarray, max_k: int):
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return mean, vt[: min(max_k, vt.shape[0])]


def pca_stats(x: np.ndarray) -> dict:
    xc = x - x.mean(axis=0, keepdims=True)
    _u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    var = s**2
    ratios = var / np.clip(var.sum(), 1e-12, None)
    csum = np.cumsum(ratios)
    entropy = -float(np.sum(ratios[ratios > 0] * np.log(ratios[ratios > 0])))
    return {
        "n_segments": int(len(x)),
        "d": int(x.shape[1]),
        "pc1_var": float(ratios[0]),
        "pc10_cum_var": float(csum[min(9, len(csum) - 1)]),
        "dim80": int(np.searchsorted(csum, 0.8) + 1),
        "dim90": int(np.searchsorted(csum, 0.9) + 1),
        "effective_rank": float(np.exp(entropy)),
    }


def orthogonalize(vectors: np.ndarray, grads: np.ndarray) -> np.ndarray:
    v = normalize_rows(vectors)
    g = normalize_rows(grads)
    residual = v - np.sum(v * g, axis=1, keepdims=True) * g
    keep = np.linalg.norm(residual, axis=1) > 1e-12
    return normalize_rows(residual[keep]), keep


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> pd.DataFrame:
    xc = x - mean
    denom = np.sum(xc * xc, axis=1)
    coeff = xc @ basis.T
    out = {}
    for k in KS:
        kk = min(k, basis.shape[0])
        out[f"energy_k{k}"] = np.sum(coeff[:, :kk] ** 2, axis=1) / np.clip(denom, 1e-12, None)
    return pd.DataFrame(out)


def bootstrap_auc_ci(pos: np.ndarray, neg: np.ndarray, seed: int, n_boot: int):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan, np.nan
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    point = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pi = rng.integers(0, len(pos), len(pos))
        ni = rng.integers(0, len(neg), len(neg))
        yy = np.r_[np.ones(len(pi)), np.zeros(len(ni))]
        ss = np.r_[pos[pi], neg[ni]]
        vals.append(roc_auc_score(yy, ss))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def collect_trajectories(dataset, indices, wrapper, device, args):
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=2)
    segment_rows = []
    vectors = defaultdict(list)
    grads = defaultdict(list)
    clean_rows = []
    clean_vectors = defaultdict(list)
    image_rows = []
    gen = torch.Generator().manual_seed(args.seed + 919)
    clean_correct = 0
    pgd_success = 0
    offset = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        batch_indices = [int(indices[offset + j]) for j in range(len(x))]
        with torch.no_grad():
            clean_pred = wrapper(x).argmax(1)
        clean_ok = clean_pred.eq(y)
        clean_correct += int(clean_ok.sum().item())

        adv = x.clone()
        feat_steps = []
        grad_steps = []
        margin_steps = []
        py_steps = []
        for step in range(args.attack_steps + 1):
            probe = adv.detach().requires_grad_(True)
            logits, feats, raw_by_label = wrapper.forward_with_features(probe)
            margin, py = margin_and_py(logits, y)
            logp = F.log_softmax(logits, dim=1).gather(1, y.view(-1, 1)).sum()
            raw_tensors = [raw for label in wrapper.labels for raw in raw_by_label.get(label, [])]
            raw_grads = torch.autograd.grad(logp, raw_tensors, retain_graph=False, allow_unused=True)
            grad_feats = wrapper.aggregate_grads(raw_by_label, list(raw_grads))
            feat_steps.append({k: v.detach().cpu().numpy() for k, v in feats.items()})
            grad_steps.append({
                k: grad_feats[k].detach().cpu().numpy()
                for k in feats.keys()
            })
            margin_steps.append(margin.cpu().numpy())
            py_steps.append(py.cpu().numpy())
            if step == args.attack_steps:
                break
            adv.requires_grad_(True)
            loss = F.cross_entropy(wrapper(adv), y)
            grad_x = torch.autograd.grad(loss, adv)[0]
            adv = project_linf(adv.detach() + args.step_size * grad_x.sign(), x, args.eps)

        with torch.no_grad():
            final_logits = wrapper(adv)
            final_pred = final_logits.argmax(1)
        final_success = final_pred.ne(y) & clean_ok
        pgd_success += int(final_success.sum().item())

        for j in range(len(x)):
            image_rows.append({
                "model": wrapper.name,
                "model_group": wrapper.model_group,
                "dataset_idx": batch_indices[j],
                "label": int(y[j].item()),
                "clean_correct": int(clean_ok[j].item()),
                "pgd_success": int(final_success[j].item()),
                "final_pred": int(final_pred[j].item()),
            })
            if not bool(clean_ok[j].item()):
                continue
            for step in range(args.attack_steps):
                for layer in feat_steps[step].keys():
                    v = feat_steps[step + 1][layer][j] - feat_steps[step][layer][j]
                    if np.linalg.norm(v) <= 1e-12:
                        continue
                    vectors[layer].append(v)
                    grads[layer].append(grad_steps[step][layer][j])
                    segment_rows.append({
                        "model": wrapper.name,
                        "model_group": wrapper.model_group,
                        "dataset_idx": batch_indices[j],
                        "label": int(y[j].item()),
                        "layer": layer,
                        "step": step,
                        "pgd_success": int(final_success[j].item()),
                        "current_margin": float(margin_steps[step][j]),
                        "current_py": float(py_steps[step][j]),
                    })

        # Clean class-preserving motions from clean-correct images.
        for j in range(len(x)):
            if not bool(clean_ok[j].item()):
                continue
            with torch.no_grad():
                _, h0, _ = wrapper.forward_with_features(x[j : j + 1])
            x_cpu = x[j].detach().cpu()
            for motion_name, xv_cpu in clean_motion_variants(x_cpu, gen):
                xv = xv_cpu.unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = wrapper(xv).argmax(1).item()
                    if pred != int(y[j].item()):
                        continue
                    _, hv, _ = wrapper.forward_with_features(xv)
                for layer, h_start in h0.items():
                    if layer not in hv:
                        continue
                    v = hv[layer].detach().cpu().numpy()[0] - h_start.detach().cpu().numpy()[0]
                    if np.linalg.norm(v) <= 1e-12:
                        continue
                    clean_vectors[layer].append(v)
                    clean_rows.append({
                        "model": wrapper.name,
                        "model_group": wrapper.model_group,
                        "dataset_idx": batch_indices[j],
                        "label": int(y[j].item()),
                        "layer": layer,
                        "motion": motion_name,
                    })
        offset += len(x)

    return {
        "images": pd.DataFrame(image_rows),
        "segments": pd.DataFrame(segment_rows),
        "vectors": {k: np.stack(v).astype(np.float32) if v else np.empty((0, 1), dtype=np.float32) for k, v in vectors.items()},
        "grads": {k: np.stack(v).astype(np.float32) if v else np.empty((0, 1), dtype=np.float32) for k, v in grads.items()},
        "clean_rows": pd.DataFrame(clean_rows),
        "clean_vectors": {k: np.stack(v).astype(np.float32) if v else np.empty((0, 1), dtype=np.float32) for k, v in clean_vectors.items()},
        "clean_accuracy": clean_correct / max(len(indices), 1),
        "pgd_asr": pgd_success / max(clean_correct, 1),
        "n_clean_correct": clean_correct,
    }


def sample_equal(pos_idx, neg_idx, rng):
    n = min(len(pos_idx), len(neg_idx))
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return rng.choice(pos_idx, size=n, replace=False), rng.choice(neg_idx, size=n, replace=False)


def matched_indices(
    seg: pd.DataFrame,
    success_eval_mask: np.ndarray,
    failed_eval_mask: np.ndarray,
    comparison: str,
    seed: int,
):
    rng = np.random.default_rng(seed)
    success = seg["pgd_success"].to_numpy() == 1
    failed = ~success
    success_eval = np.flatnonzero(success_eval_mask & success)
    failed_eval = np.flatnonzero(failed_eval_mask & failed)
    if len(success_eval) == 0 or len(failed_eval) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), {}

    margins_success_all = seg.loc[success, "current_margin"].to_numpy()
    if len(margins_success_all) < 5:
        return np.array([], dtype=int), np.array([], dtype=int), {}
    edges = np.quantile(margins_success_all, QUANTILE_EDGES)
    edges = np.maximum.accumulate(edges)
    details = {"q_edges": [float(x) for x in edges]}

    if comparison == "failed_near":
        pos_all, neg_all = [], []
        margins = seg["current_margin"].to_numpy()
        for bi in range(len(edges) - 1):
            lo, hi = edges[bi], edges[bi + 1]
            if bi == len(edges) - 2:
                in_bin = (margins >= lo) & (margins <= hi)
            else:
                in_bin = (margins >= lo) & (margins < hi)
            p, n = sample_equal(
                np.intersect1d(success_eval, np.flatnonzero(in_bin), assume_unique=False),
                np.intersect1d(failed_eval, np.flatnonzero(in_bin), assume_unique=False),
                rng,
            )
            pos_all.append(p)
            neg_all.append(n)
        return np.concatenate(pos_all), np.concatenate(neg_all), details
    if comparison == "failed_far":
        margins = seg["current_margin"].to_numpy()
        high = margins >= edges[-2]
        return sample_equal(success_eval, np.intersect1d(failed_eval, np.flatnonzero(high), assume_unique=False), rng) + (details,)
    raise ValueError(comparison)


def evaluate_layer(model_name, model_group, layer, seg, x_raw, x_orth, clean_rows, clean_raw, args):
    rng = np.random.default_rng(args.seed)
    rows = []
    log_rows = []
    ci_rows = []
    dist_frames = []

    success_images = sorted(seg.loc[seg.pgd_success == 1, "dataset_idx"].unique().tolist())
    if len(success_images) < 4:
        return rows, log_rows, ci_rows, dist_frames
    train_imgs, test_imgs = train_test_split(success_images, test_size=0.35, random_state=args.seed)
    train_imgs = set(train_imgs)
    test_imgs = set(test_imgs)
    train_success_mask = seg["dataset_idx"].isin(train_imgs).to_numpy() & (seg["pgd_success"].to_numpy() == 1)
    eval_mask = seg["dataset_idx"].isin(test_imgs).to_numpy()

    for variant, x in [("raw", x_raw), ("gradient_orthogonalized", x_orth)]:
        variant_seg = seg
        if len(x) != len(seg):
            variant_seg = seg.iloc[: len(x)].reset_index(drop=True)
            train_success_mask_v = train_success_mask[: len(x)]
            eval_mask_v = eval_mask[: len(x)]
        else:
            train_success_mask_v = train_success_mask
            eval_mask_v = eval_mask

        train_success = x[train_success_mask_v]
        if len(train_success) < 10:
            continue
        stats = pca_stats(train_success)
        stats.update({"model": model_name, "model_group": model_group, "layer": layer, "variant": variant})
        rows.append(stats)
        mean, basis = pca_basis(train_success, max(KS))

        success_eval_idx = np.flatnonzero(eval_mask_v & (variant_seg["pgd_success"].to_numpy() == 1))
        comparisons = []
        success_eval_mask = eval_mask_v & (variant_seg["pgd_success"].to_numpy() == 1)
        # Failed images never contribute to the success-only PCA basis, so they
        # can all be used as held-out negatives without leakage.
        failed_eval_mask = variant_seg["pgd_success"].to_numpy() == 0
        p_near, n_near, near_details = matched_indices(
            variant_seg,
            success_eval_mask,
            failed_eval_mask,
            "failed_near",
            args.seed + 11,
        )
        comparisons.append(("success_vs_failed_near", p_near, n_near, near_details))
        p_far, n_far, far_details = matched_indices(
            variant_seg,
            success_eval_mask,
            failed_eval_mask,
            "failed_far",
            args.seed + 17,
        )
        comparisons.append(("success_vs_failed_far", p_far, n_far, far_details))

        clean_eval = clean_raw
        if variant == "gradient_orthogonalized":
            # Clean controls have no attack gradient, so evaluate them against the
            # orthogonalized success basis without altering the clean directions.
            clean_eval = clean_raw
        if len(clean_eval):
            clean_mask = clean_rows["dataset_idx"].isin(test_imgs).to_numpy() if len(clean_rows) else np.zeros(0, dtype=bool)
            clean_idx = np.flatnonzero(clean_mask)
            if len(success_eval_idx) and len(clean_idx):
                n = min(len(success_eval_idx), len(clean_idx))
                comparisons.append((
                    "success_vs_clean",
                    rng.choice(success_eval_idx, size=n, replace=False),
                    rng.choice(clean_idx, size=n, replace=False),
                    {},
                ))

        for comparison, pos_idx, neg_idx, details in comparisons:
            if len(pos_idx) < 5 or len(neg_idx) < 5:
                continue
            pos_x = x[pos_idx]
            if comparison == "success_vs_clean":
                neg_x = clean_eval[neg_idx]
            else:
                neg_x = x[neg_idx]
            pos_e = projection_energy(pos_x, mean, basis)
            neg_e = projection_energy(neg_x, mean, basis)
            for k in KS:
                col = f"energy_k{k}"
                y = np.r_[np.ones(len(pos_e)), np.zeros(len(neg_e))]
                score = np.r_[pos_e[col].to_numpy(), neg_e[col].to_numpy()]
                try:
                    auroc = float(roc_auc_score(y, score))
                except ValueError:
                    auroc = np.nan
                try:
                    u_p = float(mannwhitneyu(pos_e[col], neg_e[col], alternative="two-sided").pvalue)
                except ValueError:
                    u_p = np.nan
                b_auc, lo, hi = bootstrap_auc_ci(pos_e[col].to_numpy(), neg_e[col].to_numpy(), args.seed + k, args.bootstrap)
                log_df = pd.DataFrame({
                    "label": y.astype(int),
                    "energy": score,
                })
                log_df["energy_k"] = score
                ci_rows.append({
                    "model": model_name,
                    "model_group": model_group,
                    "layer": layer,
                    "variant": variant,
                    "comparison": comparison,
                    "k": k,
                    "auroc": b_auc,
                    "auroc_ci_low": lo,
                    "auroc_ci_high": hi,
                    "n_pos": len(pos_e),
                    "n_neg": len(neg_e),
                })
                log_rows.append({
                    "model": model_name,
                    "model_group": model_group,
                    "layer": layer,
                    "variant": variant,
                    "comparison": comparison,
                    "k": k,
                    "auroc": auroc,
                    "mannwhitney_p": u_p,
                    "success_mean_energy": float(pos_e[col].mean()),
                    "negative_mean_energy": float(neg_e[col].mean()),
                    "n_pos": len(pos_e),
                    "n_neg": len(neg_e),
                    "details": json.dumps(details),
                })
                if k in (20, 100):
                    dist_frames.append(pd.DataFrame({
                        "model": model_name,
                        "model_group": model_group,
                        "layer": layer,
                        "variant": variant,
                        "comparison": comparison,
                        "k": k,
                        "label": ["success"] * len(pos_e) + ["negative"] * len(neg_e),
                        "energy": score,
                    }))
            features = [f"energy_k{k}" for k in KS]
            all_e = pd.concat([pos_e, neg_e], ignore_index=True)
            all_y = np.r_[np.ones(len(pos_e)), np.zeros(len(neg_e))].astype(int)
            if len(np.unique(all_y)) == 2 and min(np.bincount(all_y)) >= 5:
                xtr, xte, ytr, yte = train_test_split(
                    all_e[features].to_numpy(), all_y, test_size=0.35, stratify=all_y, random_state=args.seed
                )
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
                clf.fit(xtr, ytr)
                pred = clf.predict(xte)
                log_rows.append({
                    "model": model_name,
                    "model_group": model_group,
                    "layer": layer,
                    "variant": variant,
                    "comparison": comparison,
                    "k": "all",
                    "auroc": np.nan,
                    "mannwhitney_p": np.nan,
                    "success_mean_energy": np.nan,
                    "negative_mean_energy": np.nan,
                    "n_pos": len(pos_e),
                    "n_neg": len(neg_e),
                    "details": json.dumps({"logistic_accuracy": float(accuracy_score(yte, pred))}),
                })
    return rows, log_rows, ci_rows, dist_frames


def plot_outputs(dist_df: pd.DataFrame, dim_df: pd.DataFrame, out_dir: Path):
    if len(dist_df):
        subset = dist_df[(dist_df["k"] == 100) & (dist_df["comparison"].isin(["success_vs_failed_near", "success_vs_clean"]))]
        if len(subset):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
            for ax, comparison in zip(axes, ["success_vs_failed_near", "success_vs_clean"]):
                data = subset[subset["comparison"] == comparison]
                labels = []
                vals = []
                for (model, variant, label), g in data.groupby(["model", "variant", "label"]):
                    labels.append(f"{model}\n{variant[:4]}\n{label[:3]}")
                    vals.append(g["energy"].to_numpy())
                if vals:
                    ax.boxplot(vals, labels=labels, showfliers=False)
                    ax.tick_params(axis="x", rotation=80, labelsize=7)
                ax.set_title(comparison)
                ax.set_ylabel("Projection energy k=100")
            fig.savefig(out_dir / "matched_margin_projection_distributions.png", dpi=180)
            plt.close(fig)
    if len(dim_df):
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        raw = dim_df[dim_df["variant"] == "raw"]
        labels = [f"{r.model}\n{r.layer}" for r in raw.itertuples()]
        ax.bar(np.arange(len(raw)), raw["dim80"].to_numpy())
        ax.set_xticks(np.arange(len(raw)))
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
        ax.set_ylabel("dim80")
        ax.set_title("Matched experiment train-success dimensionality")
        fig.savefig(out_dir / "matched_margin_layerwise_summary.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar10_matched_success_quantiles/c500_s20")
    parser.add_argument("--models", default="bbb_resnet50,bbb_vgg19_bn,robustbench:Engstrom2019Robustness,robustbench:Chen2020Adversarial")
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=2 / 255)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    indices = list(range(min(args.max_images, len(dataset))))
    all_dim, all_proj, all_ci, all_dist = [], [], [], []
    all_images, all_segments = [], []
    loaded = []

    for spec in [m.strip() for m in args.models.split(",") if m.strip()]:
        wrapper = load_model(spec, device)
        loaded.append({"spec": spec, "name": wrapper.name, "group": wrapper.model_group, "layers": wrapper.labels})
        print(f"[RUN] {wrapper.name} ({wrapper.model_group}) layers={wrapper.labels}", flush=True)
        data = collect_trajectories(dataset, indices, wrapper, device, args)
        image_df = data["images"]
        image_df["clean_accuracy"] = data["clean_accuracy"]
        image_df["pgd_asr"] = data["pgd_asr"]
        image_df["n_clean_correct"] = data["n_clean_correct"]
        all_images.append(image_df)
        seg_all = data["segments"]
        all_segments.append(seg_all)
        for layer in wrapper.labels:
            seg = seg_all[seg_all["layer"] == layer].reset_index(drop=True)
            if len(seg) == 0 or layer not in data["vectors"]:
                continue
            x_raw = normalize_rows(data["vectors"][layer])
            g_raw = data["grads"][layer]
            x_orth, keep = orthogonalize(data["vectors"][layer], g_raw)
            seg_orth = seg.iloc[np.flatnonzero(keep)].reset_index(drop=True)
            clean_rows = data["clean_rows"][data["clean_rows"]["layer"] == layer].reset_index(drop=True)
            clean_raw = normalize_rows(data["clean_vectors"][layer]) if layer in data["clean_vectors"] and len(data["clean_vectors"][layer]) else np.empty((0, x_raw.shape[1]))
            dim, proj, ci, dist = evaluate_layer(wrapper.name, wrapper.model_group, layer, seg, x_raw, x_orth, clean_rows, clean_raw, args)
            # evaluate_layer assumes orth rows align with the beginning if lengths differ;
            # rerun orth with its matching rows for correctness when orth filtering removed rows.
            if len(seg_orth) != len(seg):
                dim2, proj2, ci2, dist2 = evaluate_layer(wrapper.name, wrapper.model_group, layer, seg_orth, x_orth, x_orth, clean_rows, clean_raw, args)
                dim = [r for r in dim if r["variant"] == "raw"] + [r for r in dim2 if r["variant"] == "gradient_orthogonalized"]
                proj = [r for r in proj if r["variant"] == "raw"] + [r for r in proj2 if r["variant"] == "gradient_orthogonalized"]
                ci = [r for r in ci if r["variant"] == "raw"] + [r for r in ci2 if r["variant"] == "gradient_orthogonalized"]
                dist = [d for d in dist if (len(d) and d["variant"].iloc[0] == "raw")] + [
                    d for d in dist2 if len(d) and d["variant"].iloc[0] == "gradient_orthogonalized"
                ]
            all_dim.extend(dim)
            all_proj.extend(proj)
            all_ci.extend(ci)
            all_dist.extend(dist)
        wrapper.close()
        del wrapper
        torch.cuda.empty_cache()

    images = pd.concat(all_images, ignore_index=True) if all_images else pd.DataFrame()
    segments = pd.concat(all_segments, ignore_index=True) if all_segments else pd.DataFrame()
    dim_df = pd.DataFrame(all_dim)
    proj_df = pd.DataFrame(all_proj)
    ci_df = pd.DataFrame(all_ci)
    dist_df = pd.concat(all_dist, ignore_index=True) if all_dist else pd.DataFrame()

    images.to_csv(out_dir / "matched_margin_image_outcomes.csv", index=False)
    segments.to_csv(out_dir / "matched_margin_segment_metadata.csv", index=False)
    dim_df.to_csv(out_dir / "matched_margin_dimensionality.csv", index=False)
    proj_df.to_csv(out_dir / "matched_margin_projection_metrics.csv", index=False)
    ci_df.to_csv(out_dir / "matched_margin_bootstrap_ci.csv", index=False)
    # Logistic rows are embedded as k == "all" in projection metrics for compactness;
    # also write the requested filename with only those rows.  Tiny smoke runs can
    # produce no projection rows, so preserve the output path without crashing.
    if "k" in proj_df.columns:
        proj_df[proj_df["k"].astype(str) == "all"].to_csv(out_dir / "matched_margin_logistic_regression.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "matched_margin_logistic_regression.csv", index=False)
    if len(dist_df):
        dist_df.to_csv(out_dir / "matched_margin_projection_distributions.csv", index=False)
    plot_outputs(dist_df, dim_df, out_dir)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"args": vars(args), "loaded_models": loaded}, f, indent=2)

    print("\nIMAGE OUTCOMES")
    if len(images):
        print(images.groupby(["model", "model_group"]).agg(
            clean_accuracy=("clean_accuracy", "first"),
            pgd_asr=("pgd_asr", "first"),
            n_clean_correct=("n_clean_correct", "first"),
            n_images=("dataset_idx", "count"),
        ).reset_index().to_string(index=False))
    print("\nMATCHED NEAR AUROC k100")
    if len(proj_df):
        view = proj_df[(proj_df["comparison"] == "success_vs_failed_near") & (proj_df["k"].astype(str) == "100")]
        cols = ["model", "model_group", "layer", "variant", "auroc", "n_pos", "n_neg", "success_mean_energy", "negative_mean_energy"]
        print(view[cols].to_string(index=False))
    print(f"\n[SAVED] {out_dir}")


if __name__ == "__main__":
    main()
