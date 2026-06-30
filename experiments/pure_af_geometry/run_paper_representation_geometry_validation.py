#!/usr/bin/env python3
"""Paper-grade validation of adversarial representation transport geometry.

This script unifies trajectory collection and analysis for the representation
geometry paper direction:

* multi-attack validation,
* temporal emergence,
* cross-architecture comparison.

Large segment vectors are stored in NPZ files; CSVs store metadata and metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import copy
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
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


KS = [5, 10, 20, 50, 100]
TIME_BINS = ["0-20", "20-40", "40-60", "60-80", "80-100"]
MODEL_LAYER_ROLES = {
    "resnet18": {"layer4": "final_conv", "avgpool": "penultimate", "logits": "logits"},
    "densenet121": {"final_conv": "final_conv", "avgpool": "penultimate", "logits": "logits"},
    "vgg16": {"final_conv": "final_conv", "penultimate": "penultimate", "logits": "logits"},
}


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class FeatureModel(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.normalize = Normalize()
        if name == "resnet18":
            self.base = models.resnet18(pretrained=True)
        elif name == "densenet121":
            self.base = models.densenet121(pretrained=True)
        elif name == "vgg16":
            self.base = models.vgg16_bn(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {name}")
        self.layers = list(MODEL_LAYER_ROLES[name])

    def forward(self, x):
        return self.forward_with_features(x)[0]

    def forward_with_features(self, x):
        z = self.normalize(x)
        feats = {}
        if self.name == "resnet18":
            z = self.base.conv1(z)
            z = self.base.bn1(z)
            z = self.base.relu(z)
            z = self.base.maxpool(z)
            z = self.base.layer1(z)
            z = self.base.layer2(z)
            z = self.base.layer3(z)
            z = self.base.layer4(z)
            pooled = self.base.avgpool(z).flatten(1)
            logits = self.base.fc(pooled)
            feats["layer4"] = F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
            feats["avgpool"] = pooled
            feats["logits"] = logits
        elif self.name == "densenet121":
            z = self.base.features(z)
            z = F.relu(z, inplace=False)
            pooled = F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
            logits = self.base.classifier(pooled)
            feats["final_conv"] = pooled
            feats["avgpool"] = pooled
            feats["logits"] = logits
        else:
            z = self.base.features(z)
            conv_flat = torch.flatten(self.base.avgpool(z), 1)
            penultimate = self.base.classifier[:-1](conv_flat)
            logits = self.base.classifier[-1](penultimate)
            feats["final_conv"] = conv_flat
            feats["penultimate"] = penultimate
            feats["logits"] = logits
        return logits, feats


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def project_linf(x: torch.Tensor, clean: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x, clean + eps), clean - eps).clamp(0, 1)


def margin_and_py(logits: torch.Tensor, y: torch.Tensor):
    probs = F.softmax(logits, dim=1)
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.view(-1, 1), -1e9)
    margin = true - masked.max(1).values
    py = probs.gather(1, y.view(-1, 1)).squeeze(1)
    return margin.detach(), py.detach()


def time_bin(step: int, total_steps: int) -> str:
    frac = (step + 1) / max(total_steps, 1)
    idx = min(4, int(frac * 5))
    return TIME_BINS[idx]


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
        "pc1_var": float(ratios[0]) if len(ratios) else np.nan,
        "pc10_cum_var": float(csum[min(9, len(csum) - 1)]) if len(csum) else np.nan,
        "dim80": int(np.searchsorted(csum, 0.8) + 1) if len(csum) else 0,
        "dim90": int(np.searchsorted(csum, 0.9) + 1) if len(csum) else 0,
        "effective_rank": float(np.exp(entropy)) if len(ratios) else np.nan,
    }


def fit_basis(x: np.ndarray, max_k: int):
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return mean, vt[: min(max_k, vt.shape[0])]


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> pd.DataFrame:
    xc = x - mean
    denom = np.sum(xc * xc, axis=1)
    coeff = xc @ basis.T
    out = {}
    for k in KS:
        kk = min(k, basis.shape[0])
        out[f"energy_k{k}"] = np.sum(coeff[:, :kk] ** 2, axis=1) / np.clip(denom, 1e-12, None)
    return pd.DataFrame(out)


def orthogonalize(raw: np.ndarray, grads: np.ndarray):
    v = normalize_rows(raw)
    g = normalize_rows(grads)
    out = v - np.sum(v * g, axis=1, keepdims=True) * g
    keep = np.linalg.norm(out, axis=1) > 1e-12
    return normalize_rows(out[keep]), keep


def clean_motion_variants(x: torch.Tensor, gen: torch.Generator):
    return [
        ("crop", TF.resized_crop(x, 14, 14, 196, 196, [224, 224], antialias=True)),
        ("color", TF.adjust_contrast(TF.adjust_brightness(x, 1.12), 0.9).clamp(0, 1)),
        ("blur", TF.gaussian_blur(x, [5, 5], [1.0, 1.0])),
        ("noise", (x + torch.randn(x.shape, generator=gen) * 0.015).clamp(0, 1)),
    ]


def load_indices(args) -> list[int]:
    if args.indices_metadata and Path(args.indices_metadata).exists():
        data = json.loads(Path(args.indices_metadata).read_text())
        return [int(x) for x in data["indices"][: args.max_images]]
    return list(range(args.max_images))


def select_clean_correct(dataset, model: FeatureModel, candidate_indices, max_images, batch_size, device):
    loader = DataLoader(Subset(dataset, candidate_indices), batch_size=batch_size, shuffle=False, num_workers=2)
    out = []
    offset = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(1)
            for j, ok in enumerate(pred.eq(y).detach().cpu().tolist()):
                if ok:
                    out.append(int(candidate_indices[offset + j]))
                    if len(out) >= max_images:
                        return out
            offset += len(x)
    return out


def append_segment(rows, arrays, model_name, attack_family, dataset_idx, label, layer, role, step, total_steps, success, margin, py, v, g):
    if np.linalg.norm(v) <= 1e-12:
        return
    key = (model_name, attack_family, layer)
    vector_idx = len(arrays["vectors"][key])
    arrays["vectors"][key].append(v.astype(np.float32))
    arrays["grads"][key].append(g.astype(np.float32))
    rows.append({
        "attack_family": attack_family,
        "model": model_name,
        "image_id": int(dataset_idx),
        "label": int(label),
        "layer": layer,
        "layer_role": role,
        "step": int(step),
        "total_steps": int(total_steps),
        "final_success": int(success),
        "current_margin": float(margin),
        "current_p_y": float(py),
        "time_bin": time_bin(step, total_steps),
        "vector_key": f"{model_name}|{attack_family}|{layer}",
        "vector_idx": int(vector_idx),
    })


def collect_attack_trajectories(dataset, indices, model: FeatureModel, attack_family: str, args, device):
    rows = []
    arrays = {"vectors": defaultdict(list), "grads": defaultdict(list)}
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=2)
    offset = 0
    gen = torch.Generator(device=device).manual_seed(args.seed + 3301)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        batch_indices = [int(indices[offset + j]) for j in range(len(x))]
        with torch.no_grad():
            clean_ok = model(x).argmax(1).eq(y)
        adv = x.clone()
        feat_steps, grad_steps, margin_steps, py_steps = [], [], [], []
        loss_best = None
        for step in range(args.attack_steps + 1):
            probe = adv.detach().requires_grad_(True)
            logits, feats = model.forward_with_features(probe)
            margin, py = margin_and_py(logits, y)
            logp = F.log_softmax(logits, dim=1).gather(1, y.view(-1, 1)).sum()
            grads = torch.autograd.grad(logp, list(feats.values()), retain_graph=False, allow_unused=True)
            grad_map = {}
            for (layer, feat), grad in zip(feats.items(), grads):
                grad_map[layer] = torch.zeros_like(feat) if grad is None else grad
            feat_steps.append({k: v.detach().cpu().numpy() for k, v in feats.items()})
            grad_steps.append({k: v.detach().cpu().numpy() for k, v in grad_map.items()})
            margin_steps.append(margin.cpu().numpy())
            py_steps.append(py.cpu().numpy())
            if step == args.attack_steps:
                break
            if attack_family == "pgd":
                adv.requires_grad_(True)
                loss = F.cross_entropy(model(adv), y)
                grad_x = torch.autograd.grad(loss, adv)[0]
                adv = project_linf(adv.detach() + args.step_size * grad_x.sign(), x, args.eps)
            elif attack_family == "square":
                with torch.no_grad():
                    loss_now = F.cross_entropy(model(adv), y, reduction="none")
                if loss_best is None:
                    loss_best = loss_now
                candidate = adv.detach().clone()
                side = max(args.square_min, int(round(args.square_frac * x.shape[-1] * (1 - step / max(args.attack_steps, 1)))))
                side = min(side, x.shape[-1])
                for j in range(len(x)):
                    top = int(torch.randint(0, x.shape[-2] - side + 1, (1,), generator=gen, device=device).item())
                    left = int(torch.randint(0, x.shape[-1] - side + 1, (1,), generator=gen, device=device).item())
                    patch = torch.rand((x.shape[1], side, side), generator=gen, device=device) * (2 * args.eps) - args.eps
                    candidate[j, :, top:top + side, left:left + side] = (x[j, :, top:top + side, left:left + side] + patch).clamp(0, 1)
                candidate = project_linf(candidate, x, args.eps)
                with torch.no_grad():
                    cand_loss = F.cross_entropy(model(candidate), y, reduction="none")
                accept = cand_loss >= loss_best
                adv[accept] = candidate[accept]
                loss_best[accept] = cand_loss[accept]
            else:
                raise ValueError(attack_family)
        with torch.no_grad():
            final_pred = model(adv).argmax(1)
        success = final_pred.ne(y) & clean_ok
        for j in range(len(x)):
            if not bool(clean_ok[j].item()):
                continue
            for step in range(args.attack_steps):
                for layer in model.layers:
                    v = feat_steps[step + 1][layer][j] - feat_steps[step][layer][j]
                    g = grad_steps[step][layer][j]
                    append_segment(
                        rows,
                        arrays,
                        model.name,
                        attack_family,
                        batch_indices[j],
                        int(y[j].item()),
                        layer,
                        MODEL_LAYER_ROLES[model.name][layer],
                        step,
                        args.attack_steps,
                        int(success[j].item()),
                        margin_steps[step][j],
                        py_steps[step][j],
                        v,
                        g,
                    )
        offset += len(x)
    return rows, arrays


def attack_args(base_args, attack_family: str):
    out = copy(base_args)
    if attack_family == "pgd":
        if base_args.pgd_eps is not None:
            out.eps = base_args.pgd_eps
        if base_args.pgd_steps is not None:
            out.attack_steps = base_args.pgd_steps
        if base_args.pgd_step_size is not None:
            out.step_size = base_args.pgd_step_size
    elif attack_family == "square":
        if base_args.square_eps is not None:
            out.eps = base_args.square_eps
        if base_args.square_steps is not None:
            out.attack_steps = base_args.square_steps
    return out


def collect_clean_motion(dataset, indices, model: FeatureModel, args, device):
    rows = []
    arrays = defaultdict(list)
    gen = torch.Generator().manual_seed(args.seed + 881)
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=1)
    for offset, (x, y) in enumerate(loader):
        dataset_idx = int(indices[offset])
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            pred, feats0 = model.forward_with_features(x)
            if pred.argmax(1).item() != y.item():
                continue
            feats0_np = {k: v.detach().cpu().numpy()[0] for k, v in feats0.items()}
        for motion_name, xv_cpu in clean_motion_variants(x.detach().cpu()[0], gen):
            xv = xv_cpu.unsqueeze(0).to(device)
            with torch.no_grad():
                logits, feats = model.forward_with_features(xv)
                if logits.argmax(1).item() != y.item():
                    continue
            for layer in model.layers:
                v = feats[layer].detach().cpu().numpy()[0] - feats0_np[layer]
                if np.linalg.norm(v) <= 1e-12:
                    continue
                key = (model.name, layer)
                vector_idx = len(arrays[key])
                arrays[key].append(v.astype(np.float32))
                rows.append({
                    "model": model.name,
                    "image_id": dataset_idx,
                    "label": int(y.item()),
                    "layer": layer,
                    "layer_role": MODEL_LAYER_ROLES[model.name][layer],
                    "motion": motion_name,
                    "vector_key": f"{model.name}|clean|{layer}",
                    "vector_idx": int(vector_idx),
                })
    return rows, arrays


def collect_ga_segments(manifest_path: str, args):
    rows = []
    arrays = {"vectors": defaultdict(list), "grads": defaultdict(list)}
    path = Path(manifest_path)
    if not path.exists():
        return rows, arrays
    manifest = pd.read_csv(path)
    manifest = manifest[(manifest["trajectory_features_npz"].notna()) & (manifest["success"].astype(int) == 1)].copy()
    layer_map = {"clf_layer4": ("layer4", "final_conv"), "clf_avgpool": ("avgpool", "penultimate"), "clf_logits": ("logits", "logits")}
    for _, row in manifest.iterrows():
        z = np.load(row["trajectory_features_npz"])
        image_id = int(row.get("seed", 0)) + 10_000 * int(row.get("target_class", 0))
        label = int(row.get("target_class", 0))
        for src_layer, (layer, role) in layer_map.items():
            if src_layer not in z:
                continue
            feats = z[src_layer].astype(np.float32)
            total = len(feats) - 1
            for step in range(total):
                v = feats[step + 1] - feats[step]
                # GA manifest does not store analytic feature gradients; zeros mark unavailable.
                g = np.zeros_like(v)
                append_segment(
                    rows, arrays, "resnet18", "ga_noise_pure", image_id, label, layer, role,
                    step, total, 1, np.nan, np.nan, v, g
                )
    return rows, arrays


def merge_arrays(dst, src):
    offset = {}
    for family in ["vectors", "grads"]:
        for key, vals in src[family].items():
            offset[key] = len(dst[family][key])
            dst[family][key].extend(vals)
    return offset


def arrays_to_npz(arrays, out_path: Path):
    packed = {}
    for family in ["vectors", "grads"]:
        for key, vals in arrays[family].items():
            name = f"{family}__{'__'.join(key)}"
            packed[name] = np.stack(vals).astype(np.float32) if vals else np.empty((0, 1), dtype=np.float32)
    np.savez_compressed(out_path, **packed)


def clean_arrays_to_npz(arrays, out_path: Path):
    packed = {}
    for key, vals in arrays.items():
        name = f"clean__{'__'.join(key)}"
        packed[name] = np.stack(vals).astype(np.float32) if vals else np.empty((0, 1), dtype=np.float32)
    np.savez_compressed(out_path, **packed)


def get_array(npz, family, model, attack, layer):
    name = f"{family}__{model}__{attack}__{layer}"
    if name not in npz:
        return np.empty((0, 1), dtype=np.float32)
    return npz[name]


def get_clean_array(npz, model, layer):
    name = f"clean__{model}__{layer}"
    if name not in npz:
        return np.empty((0, 1), dtype=np.float32)
    return npz[name]


def compare_projection(pos, neg, mean, basis, model, attack, layer, variant, comparison, extra=None):
    rows = []
    if len(pos) < 5 or len(neg) < 5:
        return rows
    pos_e = projection_energy(pos, mean, basis)
    neg_e = projection_energy(neg, mean, basis)
    for k in KS:
        col = f"energy_k{k}"
        y = np.r_[np.ones(len(pos_e)), np.zeros(len(neg_e))]
        score = np.r_[pos_e[col].to_numpy(), neg_e[col].to_numpy()]
        try:
            auroc = float(roc_auc_score(y, score))
        except ValueError:
            auroc = np.nan
        try:
            pval = float(mannwhitneyu(pos_e[col], neg_e[col]).pvalue)
        except ValueError:
            pval = np.nan
        row = {
            "model": model,
            "attack_family": attack,
            "layer": layer,
            "variant": variant,
            "comparison": comparison,
            "k": k,
            "auroc": auroc,
            "mannwhitney_p": pval,
            "n_pos": len(pos),
            "n_neg": len(neg),
            "success_mean_energy": float(pos_e[col].mean()),
            "negative_mean_energy": float(neg_e[col].mean()),
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def logistic_row(pos, neg, mean, basis, model, attack, layer, variant, comparison, extra=None):
    if len(pos) < 10 or len(neg) < 10:
        return None
    pos_e = projection_energy(pos, mean, basis)
    neg_e = projection_energy(neg, mean, basis)
    x = pd.concat([pos_e, neg_e], ignore_index=True)[[f"energy_k{k}" for k in KS]].to_numpy()
    y = np.r_[np.ones(len(pos_e)), np.zeros(len(neg_e))].astype(int)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.35, stratify=y, random_state=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(xtr, ytr)
    out = {
        "model": model,
        "attack_family": attack,
        "layer": layer,
        "variant": variant,
        "comparison": comparison,
        "accuracy": float(accuracy_score(yte, clf.predict(xte))),
        "n_pos": len(pos),
        "n_neg": len(neg),
    }
    if extra:
        out.update(extra)
    return out


def matched_failed_indices(meta: pd.DataFrame, success_eval_mask, failed_mask, seed):
    rng = np.random.default_rng(seed)
    success_idx = np.flatnonzero(success_eval_mask)
    failed_idx = np.flatnonzero(failed_mask)
    if len(success_idx) == 0 or len(failed_idx) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    success_margins = meta.loc[success_idx, "current_margin"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(success_margins) < 5:
        return np.array([], dtype=int), np.array([], dtype=int)
    edges = np.quantile(success_margins.to_numpy(), [0, .2, .4, .6, .8, 1])
    pos_all, neg_all = [], []
    margins = meta["current_margin"].to_numpy()
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (margins >= lo) & (margins <= hi if i == 4 else margins < hi)
        p = np.intersect1d(success_idx, np.flatnonzero(in_bin), assume_unique=False)
        n = np.intersect1d(failed_idx, np.flatnonzero(in_bin), assume_unique=False)
        m = min(len(p), len(n))
        if m:
            pos_all.append(rng.choice(p, size=m, replace=False))
            neg_all.append(rng.choice(n, size=m, replace=False))
    if not pos_all:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(pos_all), np.concatenate(neg_all)


def analyze_multi_attack(meta, clean_meta, vectors_npz, clean_npz, args, out_dir):
    dim_rows, metric_rows, orth_metric_rows, log_rows = [], [], [], []
    bases = {}
    rng = np.random.default_rng(args.seed)
    for (model, attack, layer), group in meta.groupby(["model", "attack_family", "layer"]):
        raw = get_array(vectors_npz, "vectors", model, attack, layer)
        grads = get_array(vectors_npz, "grads", model, attack, layer)
        if len(raw) != len(group) or len(raw) < 10:
            continue
        group = group.reset_index(drop=True)
        success = group["final_success"].to_numpy(dtype=int) == 1
        success_images = sorted(group.loc[success, "image_id"].unique())
        if len(success_images) < 3:
            continue
        train_imgs, test_imgs = train_test_split(success_images, test_size=0.35, random_state=args.seed)
        train_mask = group["image_id"].isin(train_imgs).to_numpy() & success
        test_success_mask = group["image_id"].isin(test_imgs).to_numpy() & success
        failed_mask = ~success
        clean = get_clean_array(clean_npz, model, layer)
        clean_layer_meta = clean_meta[(clean_meta["model"] == model) & (clean_meta["layer"] == layer)]
        for variant, x, local_group, train_local, test_local, failed_local in [
            ("raw", normalize_rows(raw), group, train_mask, test_success_mask, failed_mask),
        ]:
            if train_local.sum() < 10:
                continue
            train_x = x[train_local]
            stats = pca_stats(train_x)
            stats.update({"model": model, "attack_family": attack, "layer": layer, "variant": variant})
            dim_rows.append(stats)
            mean, basis = fit_basis(train_x, max(KS))
            bases[(model, attack, layer, variant)] = (mean, basis)
            pos = x[test_local]
            random_x = normalize_rows(rng.normal(size=(max(len(pos), 50), x.shape[1])))
            for comparison, neg in [("success_vs_random", random_x), ("success_vs_failed", x[failed_local])]:
                metric_rows.extend(compare_projection(pos, neg, mean, basis, model, attack, layer, variant, comparison))
                lr = logistic_row(pos, neg, mean, basis, model, attack, layer, variant, comparison)
                if lr:
                    log_rows.append(lr)
            if len(clean):
                clean_eval = normalize_rows(clean)
                metric_rows.extend(compare_projection(pos, clean_eval, mean, basis, model, attack, layer, variant, "success_vs_clean"))
                lr = logistic_row(pos, clean_eval, mean, basis, model, attack, layer, variant, "success_vs_clean")
                if lr:
                    log_rows.append(lr)
            pidx, nidx = matched_failed_indices(local_group, test_local, failed_local, args.seed)
            if len(pidx):
                metric_rows.extend(compare_projection(x[pidx], x[nidx], mean, basis, model, attack, layer, variant, "success_vs_failed_near"))
        if len(grads) == len(raw):
            orth, keep = orthogonalize(raw, grads)
            orth_group = group.iloc[np.flatnonzero(keep)].reset_index(drop=True)
            success_o = orth_group["final_success"].to_numpy(dtype=int) == 1
            train_o = orth_group["image_id"].isin(train_imgs).to_numpy() & success_o
            test_o = orth_group["image_id"].isin(test_imgs).to_numpy() & success_o
            failed_o = ~success_o
            if train_o.sum() >= 10:
                train_x = orth[train_o]
                stats = pca_stats(train_x)
                stats.update({"model": model, "attack_family": attack, "layer": layer, "variant": "gradient_orthogonalized"})
                dim_rows.append(stats)
                mean, basis = fit_basis(train_x, max(KS))
                bases[(model, attack, layer, "gradient_orthogonalized")] = (mean, basis)
                pos = orth[test_o]
                random_x = normalize_rows(rng.normal(size=(max(len(pos), 50), orth.shape[1])))
                for comparison, neg in [("success_vs_random", random_x), ("success_vs_failed", orth[failed_o])]:
                    rows = compare_projection(pos, neg, mean, basis, model, attack, layer, "gradient_orthogonalized", comparison)
                    orth_metric_rows.extend(rows)
                    metric_rows.extend(rows)
                if len(clean):
                    rows = compare_projection(pos, normalize_rows(clean), mean, basis, model, attack, layer, "gradient_orthogonalized", "success_vs_clean")
                    orth_metric_rows.extend(rows)
                    metric_rows.extend(rows)
                pidx, nidx = matched_failed_indices(orth_group, test_o, failed_o, args.seed)
                if len(pidx):
                    rows = compare_projection(orth[pidx], orth[nidx], mean, basis, model, attack, layer, "gradient_orthogonalized", "success_vs_failed_near")
                    orth_metric_rows.extend(rows)
                    metric_rows.extend(rows)
    pd.DataFrame(dim_rows).to_csv(out_dir / "attack_family_dimensionality.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "attack_family_projection_metrics.csv", index=False)
    pd.DataFrame(orth_metric_rows).to_csv(out_dir / "attack_family_gradient_orthogonalized_metrics.csv", index=False)
    pd.DataFrame(log_rows).to_csv(out_dir / "attack_family_logistic_regression.csv", index=False)
    return pd.DataFrame(dim_rows), pd.DataFrame(metric_rows), bases


def analyze_temporal(meta, clean_meta, vectors_npz, clean_npz, args, out_dir):
    metric_rows, log_rows = [], []
    rng = np.random.default_rng(args.seed + 91)
    for (model, attack, layer, tb), group in meta.groupby(["model", "attack_family", "layer", "time_bin"]):
        raw = get_array(vectors_npz, "vectors", model, attack, layer)
        grads = get_array(vectors_npz, "grads", model, attack, layer)
        if len(raw) != len(meta[(meta["model"] == model) & (meta["attack_family"] == attack) & (meta["layer"] == layer)]):
            continue
        base_group = meta[(meta["model"] == model) & (meta["attack_family"] == attack) & (meta["layer"] == layer)].reset_index(drop=True)
        idx = group.index.to_numpy()
        local_pos = base_group.index[base_group["time_bin"].eq(tb)].to_numpy()
        x_all = normalize_rows(raw)
        g_all = grads
        for variant, x, local_group in [("raw", x_all, base_group)]:
            success = local_group["final_success"].to_numpy(dtype=int) == 1
            success_images = sorted(local_group.loc[success & local_group["time_bin"].eq(tb), "image_id"].unique())
            if len(success_images) < 3:
                continue
            train_imgs, test_imgs = train_test_split(success_images, test_size=0.35, random_state=args.seed)
            in_bin = local_group["time_bin"].eq(tb).to_numpy()
            train = in_bin & local_group["image_id"].isin(train_imgs).to_numpy() & success
            test = in_bin & local_group["image_id"].isin(test_imgs).to_numpy() & success
            failed = in_bin & (~success)
            if train.sum() < 10 or test.sum() < 5:
                continue
            mean, basis = fit_basis(x[train], max(KS))
            pidx, nidx = matched_failed_indices(local_group, test, failed, args.seed)
            if len(pidx):
                metric_rows.extend(compare_projection(x[pidx], x[nidx], mean, basis, model, attack, layer, variant, "success_vs_failed_near", {"time_bin": tb}))
            clean = get_clean_array(clean_npz, model, layer)
            if len(clean):
                metric_rows.extend(compare_projection(x[test], normalize_rows(clean), mean, basis, model, attack, layer, variant, "success_vs_clean", {"time_bin": tb}))
        if len(g_all) == len(x_all):
            orth, keep = orthogonalize(raw, g_all)
            og = base_group.iloc[np.flatnonzero(keep)].reset_index(drop=True)
            success = og["final_success"].to_numpy(dtype=int) == 1
            success_images = sorted(og.loc[success & og["time_bin"].eq(tb), "image_id"].unique())
            if len(success_images) >= 3:
                train_imgs, test_imgs = train_test_split(success_images, test_size=0.35, random_state=args.seed)
                in_bin = og["time_bin"].eq(tb).to_numpy()
                train = in_bin & og["image_id"].isin(train_imgs).to_numpy() & success
                test = in_bin & og["image_id"].isin(test_imgs).to_numpy() & success
                failed = in_bin & (~success)
                if train.sum() >= 10 and test.sum() >= 5:
                    mean, basis = fit_basis(orth[train], max(KS))
                    pidx, nidx = matched_failed_indices(og, test, failed, args.seed)
                    if len(pidx):
                        metric_rows.extend(compare_projection(orth[pidx], orth[nidx], mean, basis, model, attack, layer, "gradient_orthogonalized", "success_vs_failed_near", {"time_bin": tb}))
                    clean = get_clean_array(clean_npz, model, layer)
                    if len(clean):
                        metric_rows.extend(compare_projection(orth[test], normalize_rows(clean), mean, basis, model, attack, layer, "gradient_orthogonalized", "success_vs_clean", {"time_bin": tb}))
    df = pd.DataFrame(metric_rows)
    df.to_csv(out_dir / "temporal_projection_metrics.csv", index=False)
    # Bootstrap CI placeholder: exact metric table is deterministic; CI can be added from saved distributions if needed.
    df.to_csv(out_dir / "temporal_bootstrap_ci.csv", index=False)
    pd.DataFrame(log_rows).to_csv(out_dir / "temporal_logistic_regression.csv", index=False)
    if len(df):
        plot = df[(df["comparison"] == "success_vs_failed_near") & (df["k"] == 100)]
        title_comparison = "success vs failed-near"
        if len(plot) == 0:
            plot = df[(df["comparison"] == "success_vs_clean") & (df["k"] == 100)]
            title_comparison = "success vs clean"
        if len(plot):
            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            for (model, attack, layer, variant), g in plot.groupby(["model", "attack_family", "layer", "variant"]):
                g = g.copy()
                g["time_order"] = g["time_bin"].map({b: i for i, b in enumerate(TIME_BINS)})
                g = g.sort_values("time_order")
                ax.plot(g["time_bin"], g["auroc"], marker="o", label=f"{model}/{attack}/{layer}/{variant[:4]}")
            ax.axhline(0.5, color="black", lw=1, ls="--")
            ax.set_ylabel("AUROC")
            ax.set_title(f"Temporal emergence: {title_comparison}, k=100")
            ax.legend(fontsize=6, ncol=2)
            fig.savefig(out_dir / "temporal_emergence_curves.png", dpi=180)
            plt.close(fig)


def subspace_from_rows(x: np.ndarray, max_k: int):
    x = normalize_rows(x)
    xc = x - x.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    return u[:, : min(max_k, u.shape[1])], s


def feature_basis_from_rows(x: np.ndarray, max_k: int):
    x = normalize_rows(x)
    xc = x - x.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    return vt[: min(max_k, vt.shape[0])].T, s


def subspace_metrics(u_a, u_b, k):
    kk = min(k, u_a.shape[1], u_b.shape[1])
    if kk == 0:
        return None
    s = np.linalg.svd(u_a[:, :kk].T @ u_b[:, :kk], compute_uv=False)
    s = np.clip(s, 0, 1)
    angles = np.arccos(s)
    return {
        "k": k,
        "mean_principal_angle_deg": float(np.degrees(angles).mean()),
        "max_principal_angle_deg": float(np.degrees(angles).max()),
        "projection_overlap": float(np.sum(s**2) / kk),
        "grassmann_distance": float(np.linalg.norm(angles)),
        "subspace_affinity": float(np.sqrt(np.sum(s**2) / kk)),
    }


def analyze_overlaps(meta, vectors_npz, dim_df, metric_df, args, out_multi, out_cross):
    rows_attack, rows_arch = [], []
    for (model, layer), group in meta.groupby(["model", "layer"]):
        attacks = sorted(group["attack_family"].unique())
        for i, a in enumerate(attacks):
            for b in attacks[i + 1:]:
                ga = group[(group["attack_family"] == a) & (group["final_success"] == 1)]
                gb = group[(group["attack_family"] == b) & (group["final_success"] == 1)]
                xa = get_array(vectors_npz, "vectors", model, a, layer)
                xb = get_array(vectors_npz, "vectors", model, b, layer)
                if len(ga) < 10 or len(gb) < 10:
                    continue
                ua, _ = feature_basis_from_rows(xa[ga["vector_idx"].to_numpy()], max(KS))
                ub, _ = feature_basis_from_rows(xb[gb["vector_idx"].to_numpy()], max(KS))
                for k in [20, 50, 100]:
                    m = subspace_metrics(ua, ub, k)
                    if m:
                        m.update({"model": model, "layer": layer, "attack_a": a, "attack_b": b, "variant": "raw"})
                        rows_attack.append(m)
    role_groups = meta.groupby(["attack_family", "layer_role"])
    for (attack, role), group in role_groups:
        models_here = sorted(group["model"].unique())
        for i, ma in enumerate(models_here):
            for mb in models_here[i + 1:]:
                ga = group[(group["model"] == ma) & (group["final_success"] == 1)]
                gb = group[(group["model"] == mb) & (group["final_success"] == 1)]
                common = sorted(set(zip(ga["image_id"], ga["step"])).intersection(set(zip(gb["image_id"], gb["step"]))))
                if len(common) < 10:
                    continue
                la = ga["layer"].iloc[0]
                lb = gb["layer"].iloc[0]
                xa = get_array(vectors_npz, "vectors", ma, attack, la)
                xb = get_array(vectors_npz, "vectors", mb, attack, lb)
                ga_map = {(r.image_id, r.step): int(r.vector_idx) for r in ga.itertuples()}
                gb_map = {(r.image_id, r.step): int(r.vector_idx) for r in gb.itertuples()}
                xa_common = np.stack([xa[ga_map[c]] for c in common])
                xb_common = np.stack([xb[gb_map[c]] for c in common])
                ua, _ = subspace_from_rows(xa_common, max(KS))
                ub, _ = subspace_from_rows(xb_common, max(KS))
                for k in [20, 50, 100]:
                    m = subspace_metrics(ua, ub, k)
                    if m:
                        m.update({"attack_family": attack, "layer_role": role, "model_a": ma, "model_b": mb, "layer_a": la, "layer_b": lb, "n_common_segments": len(common), "variant": "raw"})
                        rows_arch.append(m)
    pd.DataFrame(rows_attack).to_csv(out_multi / "attack_family_subspace_overlap.csv", index=False)
    pd.DataFrame(rows_arch).to_csv(out_cross / "cross_arch_subspace_overlap.csv", index=False)
    dim_df.to_csv(out_cross / "cross_arch_dimensionality.csv", index=False)
    metric_df.to_csv(out_cross / "cross_arch_predictiveness.csv", index=False)
    if len(rows_arch):
        df = pd.DataFrame(rows_arch)
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        sub = df[df["k"] == 20]
        labels = [f"{r.attack_family}/{r.layer_role}\n{r.model_a}-{r.model_b}" for r in sub.itertuples()]
        ax.bar(np.arange(len(sub)), sub["projection_overlap"])
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("Projection overlap")
        ax.set_title("Cross-architecture success-flow overlap")
        fig.savefig(out_cross / "cross_arch_summary.png", dpi=180)
        plt.close(fig)


def analyze_cross_attack_basis_transfer(meta, clean_meta, vectors_npz, clean_npz, args, out_multi):
    rows = []
    rng = np.random.default_rng(args.seed + 177)
    for (model, layer), group in meta.groupby(["model", "layer"]):
        attacks = sorted(group["attack_family"].unique())
        clean = get_clean_array(clean_npz, model, layer)
        clean_eval = normalize_rows(clean) if len(clean) else np.empty((0, 1), dtype=np.float32)
        for train_attack in attacks:
            train_group = group[(group["attack_family"] == train_attack) & (group["final_success"] == 1)].reset_index(drop=True)
            train_x_all = get_array(vectors_npz, "vectors", model, train_attack, layer)
            if len(train_group) < 20 or len(train_x_all) == 0:
                continue
            success_images = sorted(train_group["image_id"].unique())
            if len(success_images) < 3:
                continue
            train_imgs, _holdout_imgs = train_test_split(success_images, test_size=0.35, random_state=args.seed)
            train_mask = train_group["image_id"].isin(train_imgs).to_numpy()
            train_x = normalize_rows(train_x_all[train_group.loc[train_mask, "vector_idx"].to_numpy()])
            if len(train_x) < 10:
                continue
            mean, basis = fit_basis(train_x, max(KS))
            for test_attack in attacks:
                test_group_all = group[group["attack_family"] == test_attack].reset_index(drop=True)
                test_x_all = get_array(vectors_npz, "vectors", model, test_attack, layer)
                if len(test_group_all) == 0 or len(test_x_all) == 0:
                    continue
                success_mask = test_group_all["final_success"].to_numpy(dtype=int) == 1
                pos_idx = test_group_all.loc[success_mask, "vector_idx"].to_numpy()
                if len(pos_idx) < 5:
                    continue
                pos = normalize_rows(test_x_all[pos_idx])
                random_x = normalize_rows(rng.normal(size=(max(len(pos), 50), pos.shape[1])))
                for comparison, neg in [("transfer_success_vs_random", random_x)]:
                    rows.extend(compare_projection(pos, neg, mean, basis, model, train_attack, layer, "raw", comparison, {"test_attack_family": test_attack}))
                failed_idx = test_group_all.loc[~success_mask, "vector_idx"].to_numpy()
                if len(failed_idx) >= 5:
                    neg = normalize_rows(test_x_all[failed_idx])
                    rows.extend(compare_projection(pos, neg, mean, basis, model, train_attack, layer, "raw", "transfer_success_vs_failed", {"test_attack_family": test_attack}))
                    pidx, nidx = matched_failed_indices(test_group_all, success_mask, ~success_mask, args.seed)
                    if len(pidx):
                        pos_near = normalize_rows(test_x_all[test_group_all.loc[pidx, "vector_idx"].to_numpy()])
                        neg_near = normalize_rows(test_x_all[test_group_all.loc[nidx, "vector_idx"].to_numpy()])
                        rows.extend(compare_projection(pos_near, neg_near, mean, basis, model, train_attack, layer, "raw", "transfer_success_vs_failed_near", {"test_attack_family": test_attack}))
                if len(clean_eval) and clean_eval.shape[1] == pos.shape[1]:
                    rows.extend(compare_projection(pos, clean_eval, mean, basis, model, train_attack, layer, "raw", "transfer_success_vs_clean", {"test_attack_family": test_attack}))
    pd.DataFrame(rows).to_csv(out_multi / "attack_family_basis_transfer.csv", index=False)


def write_summary_plot(dim_df, metric_df, out_dir):
    if len(dim_df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    raw = dim_df[dim_df["variant"] == "raw"]
    labels = [f"{r.model}\n{r.attack_family}\n{r.layer}" for r in raw.itertuples()]
    axes[0].bar(np.arange(len(raw)), raw["dim80"])
    axes[0].set_xticks(np.arange(len(raw)))
    axes[0].set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    axes[0].set_ylabel("dim80")
    axes[0].set_title("Success-flow dimensionality")
    if len(metric_df):
        met = metric_df[(metric_df["comparison"] == "success_vs_random") & (metric_df["k"] == 100) & (metric_df["variant"] == "raw")]
        labels = [f"{r.model}\n{r.attack_family}\n{r.layer}" for r in met.itertuples()]
        axes[1].bar(np.arange(len(met)), met["auroc"])
        axes[1].axhline(0.5, color="black", lw=1, ls="--")
        axes[1].set_xticks(np.arange(len(met)))
        axes[1].set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        axes[1].set_ylabel("AUROC")
        axes[1].set_title("Success vs random, k=100")
    fig.savefig(out_dir / "attack_family_summary.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    p.add_argument("--indices-metadata", default="analysis_outputs/pure_af_geometry/away_from_true_attractor_resnet18_all_available_c10.csv.metadata.json")
    p.add_argument("--ga-manifest", default="analysis_outputs/pure_af_geometry/random_ga_classifier_trajectories_resnet18_c10_s5/manifest.csv")
    p.add_argument("--output-root", default="analysis_outputs/pure_af_geometry")
    p.add_argument("--models", default="resnet18,densenet121,vgg16")
    p.add_argument("--attacks", default="pgd,square,ga")
    p.add_argument("--max-images", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--attack-steps", type=int, default=20)
    p.add_argument("--step-size", type=float, default=0.005)
    p.add_argument("--pgd-eps", type=float, default=None)
    p.add_argument("--pgd-steps", type=int, default=None)
    p.add_argument("--pgd-step-size", type=float, default=None)
    p.add_argument("--square-eps", type=float, default=None)
    p.add_argument("--square-steps", type=int, default=None)
    p.add_argument("--square-frac", type=float, default=0.35)
    p.add_argument("--square-min", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reuse-trajectories", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.output_root)
    out_multi = root / "paper_multi_attack"
    out_temporal = root / "paper_temporal"
    out_cross = root / "paper_cross_arch"
    for d in [out_multi, out_temporal, out_cross]:
        d.mkdir(parents=True, exist_ok=True)

    traj_meta_path = out_multi / "standardized_segment_metadata.csv"
    vector_npz_path = out_multi / "standardized_segment_vectors.npz"
    clean_meta_path = out_multi / "clean_motion_metadata.csv"
    clean_npz_path = out_multi / "clean_motion_vectors.npz"

    if not args.reuse_trajectories or not (traj_meta_path.exists() and vector_npz_path.exists() and clean_meta_path.exists() and clean_npz_path.exists()):
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
        candidate_indices = load_indices(args)
        all_rows = []
        all_arrays = {"vectors": defaultdict(list), "grads": defaultdict(list)}
        clean_rows_all = []
        clean_arrays_all = defaultdict(list)
        attacks = [x.strip() for x in args.attacks.split(",") if x.strip()]
        for model_name in [x.strip() for x in args.models.split(",") if x.strip()]:
            model = FeatureModel(model_name).to(device).eval()
            indices = select_clean_correct(dataset, model, candidate_indices, args.max_images, args.batch_size, device)
            print(f"[model] {model_name} clean_correct={len(indices)} attacks={attacks}", flush=True)
            clean_rows, clean_arrays = collect_clean_motion(dataset, indices, model, args, device)
            clean_rows_all.extend(clean_rows)
            for key, vals in clean_arrays.items():
                clean_arrays_all[key].extend(vals)
            for attack in attacks:
                if attack == "ga":
                    continue
                local_args = attack_args(args, attack)
                rows, arrays = collect_attack_trajectories(dataset, indices, model, attack, local_args, device)
                all_rows.extend(rows)
                for fam in ["vectors", "grads"]:
                    for key, vals in arrays[fam].items():
                        all_arrays[fam][key].extend(vals)
                print(f"[collected] {model_name}/{attack} segments={len(rows)}", flush=True)
            del model
            torch.cuda.empty_cache()
        if "ga" in attacks:
            rows, arrays = collect_ga_segments(args.ga_manifest, args)
            all_rows.extend(rows)
            for fam in ["vectors", "grads"]:
                for key, vals in arrays[fam].items():
                    all_arrays[fam][key].extend(vals)
            print(f"[collected] ga_noise_pure segments={len(rows)}", flush=True)
        pd.DataFrame(all_rows).to_csv(traj_meta_path, index=False)
        arrays_to_npz(all_arrays, vector_npz_path)
        pd.DataFrame(clean_rows_all).to_csv(clean_meta_path, index=False)
        clean_arrays_to_npz(clean_arrays_all, clean_npz_path)
        with open(out_multi / "metadata.json", "w") as f:
            json.dump({"args": vars(args), "segment_metadata": str(traj_meta_path), "segment_vectors": str(vector_npz_path), "clean_metadata": str(clean_meta_path), "clean_vectors": str(clean_npz_path)}, f, indent=2)

    meta = pd.read_csv(traj_meta_path)
    clean_meta = pd.read_csv(clean_meta_path) if clean_meta_path.exists() else pd.DataFrame()
    vectors_npz = np.load(vector_npz_path)
    clean_npz = np.load(clean_npz_path)
    dim_df, metric_df, _bases = analyze_multi_attack(meta, clean_meta, vectors_npz, clean_npz, args, out_multi)
    write_summary_plot(dim_df, metric_df, out_multi)
    analyze_temporal(meta, clean_meta, vectors_npz, clean_npz, args, out_temporal)
    analyze_cross_attack_basis_transfer(meta, clean_meta, vectors_npz, clean_npz, args, out_multi)
    analyze_overlaps(meta, vectors_npz, dim_df, metric_df, args, out_multi, out_cross)
    print("\nSEGMENTS")
    print(meta.groupby(["model", "attack_family", "layer", "final_success"]).size().reset_index(name="n").to_string(index=False))
    print(f"\n[SAVED] {out_multi}\n[SAVED] {out_temporal}\n[SAVED] {out_cross}")


if __name__ == "__main__":
    main()
