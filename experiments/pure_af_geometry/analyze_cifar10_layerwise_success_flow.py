#!/usr/bin/env python3
"""Layerwise CIFAR-10 GA success-flow emergence analysis.

This experiment asks where random-noise-to-pure GA success-flow geometry appears
inside each BlackboxBench CIFAR-10 architecture. It records all major stages,
compares successful trajectory segments against random and clean class-preserving
motion, and repeats the analysis after removing local feature-gradient components.
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


class CIFARFeatureWrapper(nn.Module):
    def __init__(self, spec: str, model: nn.Module):
        super().__init__()
        self.spec = spec
        self.name = spec.replace(":", "_")
        self.model = model
        self.enabled = False
        self.captures: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.handles = []
        self.labels = self._register_hooks()

    def _make_hook(self, label: str):
        def hook(_module, _inp, out):
            if self.enabled and torch.is_tensor(out):
                self.captures[label].append(out)
        return hook

    def _register(self, module_name: str, label: str) -> bool:
        modules = dict(self.model.named_modules())
        if module_name not in modules:
            return False
        self.handles.append(modules[module_name].register_forward_hook(self._make_hook(label)))
        return True

    def _register_hooks(self) -> list[str]:
        labels = []
        if self.spec == "bbb_resnet50":
            for label, module_name in [
                ("layer1", "1.layer1"),
                ("layer2", "1.layer2"),
                ("layer3", "1.layer3"),
                ("layer4", "1.layer4"),
                # The local CIFAR ResNet exposes no explicit avgpool module;
                # use pooled layer4 activations as the penultimate representation.
                ("avgpool", "1.layer4"),
            ]:
                if self._register(module_name, label):
                    labels.append(label)
        elif self.spec == "bbb_vgg19_bn":
            for label, module_name in [
                ("block1", "1.features.5"),
                ("block2", "1.features.12"),
                ("block3", "1.features.25"),
                ("block4", "1.features.38"),
                ("block5", "1.features.51"),
                ("penultimate", "1.features.52"),
            ]:
                if self._register(module_name, label):
                    labels.append(label)
        elif self.spec == "bbb_densenet":
            for label, module_name in [
                ("denseblock1", "1.dense1"),
                ("denseblock2", "1.dense2"),
                ("denseblock3", "1.dense3"),
                ("penultimate", "1.avgpool"),
            ]:
                if self._register(module_name, label):
                    labels.append(label)
        elif self.spec == "bbb_inception_v3":
            for label, module_name in [
                ("mixed5", "1.Mixed_5d"),
                ("mixed6", "1.Mixed_6e"),
                ("mixed7", "1.Mixed_7c"),
                ("penultimate", "1.avgpool"),
            ]:
                if self._register(module_name, label):
                    labels.append(label)
        if not labels:
            last_conv = None
            for module_name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module_name
            if last_conv is None:
                raise RuntimeError(f"No Conv2d feature found for {self.spec}")
            self._register(last_conv, "final_conv")
            labels.append("final_conv")
        labels.append("logits")
        return labels

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x.flatten(1)

    def _aggregate(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        pooled = [self._pool(x) for x in tensors]
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
        raw = {}
        for label in self.labels:
            if label == "logits":
                feats[label] = logits
                raw[label] = [logits]
            else:
                outs = self.captures.get(label, [])
                if not outs:
                    continue
                feats[label] = self._aggregate(outs)
                raw[label] = outs
        return logits, feats, raw

    def aggregate_grads(self, raw_by_label: dict[str, list[torch.Tensor]], raw_grads: list[torch.Tensor | None]):
        out = {}
        cursor = 0
        for label in self.labels:
            raws = raw_by_label.get(label, [])
            grads = []
            for raw in raws:
                g = raw_grads[cursor]
                cursor += 1
                grads.append(torch.zeros_like(raw) if g is None else g)
            if grads:
                out[label] = self._aggregate(grads)
        return out


def load_model(spec: str, device):
    model = load_cifar_model(spec).to(device).eval()
    return CIFARFeatureWrapper(spec, model).to(device).eval()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


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


def orthogonalize(vectors: np.ndarray, grads: np.ndarray):
    v = normalize_rows(vectors)
    g = normalize_rows(grads)
    residual = v - np.sum(v * g, axis=1, keepdims=True) * g
    keep = np.linalg.norm(residual, axis=1) > 1e-12
    return normalize_rows(residual[keep]), keep


def make_children(parents: torch.Tensor, n_children: int, crossover: float, gen: torch.Generator):
    n_parent = parents.shape[0]
    a = parents[torch.randint(0, n_parent, (n_children,), generator=gen, device=parents.device)]
    b = parents[torch.randint(0, n_parent, (n_children,), generator=gen, device=parents.device)]
    mask = torch.rand((n_children, 1, parents.shape[2], parents.shape[3]), generator=gen, device=parents.device) < crossover
    return torch.where(mask, a, b)


def mutate(x: torch.Tensor, args, gen: torch.Generator):
    out = x.clone()
    pix = torch.rand(out.shape, generator=gen, device=out.device) < args.pixel_rate
    out = torch.where(pix, out + torch.randn(out.shape, generator=gen, device=out.device) * args.pixel_sigma, out)
    if args.block_rate > 0:
        for i in range(len(out)):
            if float(torch.rand((), generator=gen, device=out.device).item()) < args.block_rate:
                side = min(args.block_size, out.shape[-1])
                top = int(torch.randint(0, out.shape[-2] - side + 1, (1,), generator=gen, device=out.device).item())
                left = int(torch.randint(0, out.shape[-1] - side + 1, (1,), generator=gen, device=out.device).item())
                out[i, :, top:top + side, left:left + side] += torch.randn((out.shape[1], side, side), generator=gen, device=out.device) * args.pixel_sigma * 2
    return out.clamp(0, 1)


def eval_population(model, pop, target, batch_size):
    logits_all = []
    for i in range(0, len(pop), batch_size):
        with torch.no_grad():
            logits_all.append(model(pop[i:i + batch_size]))
    logits = torch.cat(logits_all, dim=0)
    probs = F.softmax(logits, dim=1)
    target_logit = logits[:, target]
    masked = logits.clone()
    masked[:, target] = -1e9
    margin = target_logit - masked.max(1).values
    return {
        "fitness": target_logit,
        "margin": margin,
        "prob": probs[:, target],
        "pred": logits.argmax(1),
    }


def record_state(wrapper, image, target, rows, features, grads, run_id, generation, success_final, device):
    probe = image.detach().requires_grad_(True)
    logits, feats, raw = wrapper.forward_with_features(probe)
    logp = F.log_softmax(logits, dim=1)[:, target].sum()
    raw_tensors = [r for label in wrapper.labels for r in raw.get(label, [])]
    raw_grads = torch.autograd.grad(logp, raw_tensors, retain_graph=False, allow_unused=True)
    grad_feats = wrapper.aggregate_grads(raw, list(raw_grads))
    probs = F.softmax(logits, dim=1)
    margin = logits[0, target] - torch.cat([logits[0, :target], logits[0, target + 1:]]).max()
    for layer, h in feats.items():
        key = (wrapper.name, layer)
        features[key].append(h.detach().cpu().numpy()[0].astype(np.float32))
        grads[key].append(grad_feats[layer].detach().cpu().numpy()[0].astype(np.float32))
        rows.append({
            "model": wrapper.name,
            "run_id": run_id,
            "target_class": int(target),
            "generation": int(generation),
            "layer": layer,
            "success_final": int(success_final),
            "prob": float(probs[0, target].item()),
            "margin": float(margin.item()),
            "pred": int(logits.argmax(1).item()),
            "feature_idx": len(features[key]) - 1,
        })


def run_ga(wrapper, target: int, seed: int, args, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    pop = torch.rand((args.population, 3, 32, 32), generator=gen, device=device)
    save_every = max(1, args.save_every)
    saved_images = []
    saved_generations = []
    best = None
    success_gen = None
    for generation in range(args.generations + 1):
        stats = eval_population(wrapper, pop, target, args.eval_batch_size)
        order = torch.argsort(stats["fitness"], descending=True)
        pop = pop[order]
        for k in stats:
            stats[k] = stats[k][order]
        cur = {
            "image": pop[:1].detach().clone(),
            "fitness": float(stats["fitness"][0].item()),
            "prob": float(stats["prob"][0].item()),
            "pred": int(stats["pred"][0].item()),
        }
        if best is None or cur["fitness"] > best["fitness"]:
            best = cur
        if generation % save_every == 0 or generation == args.generations:
            saved_images.append(cur["image"])
            saved_generations.append(generation)
        if success_gen is None and cur["prob"] >= args.prob_threshold and cur["pred"] == target:
            success_gen = generation
            saved_images.append(cur["image"])
            saved_generations.append(generation)
            if args.stop_on_success:
                break
        if generation == args.generations:
            break
        parents = pop[: args.parents]
        elite = pop[: args.elite]
        children = make_children(parents, args.population - args.elite, args.crossover, gen)
        children = mutate(children, args, gen)
        pop = torch.cat([elite, children], dim=0)
    final_success = int(best is not None and best["prob"] >= args.prob_threshold and best["pred"] == target)
    return saved_generations, saved_images, final_success, success_gen


def clean_motion_variants(x, gen):
    return [
        ("crop", TF.resized_crop(x, 2, 2, 28, 28, [32, 32], antialias=True)),
        ("color", TF.adjust_contrast(TF.adjust_brightness(x, 1.2), 0.85).clamp(0, 1)),
        ("blur", TF.gaussian_blur(x, [5, 5], [0.8, 0.8])),
        ("noise", (x + torch.randn(x.shape, generator=gen) * 0.03).clamp(0, 1)),
    ]


def collect_clean_motion(dataset, indices, wrapper, args, device):
    rows = []
    vectors = defaultdict(list)
    gen = torch.Generator().manual_seed(args.seed + 333)
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=1)
    for offset, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            logits, feats0, _raw = wrapper.forward_with_features(x)
            if logits.argmax(1).item() != y.item():
                continue
            feats0 = {k: v.detach().cpu().numpy()[0] for k, v in feats0.items()}
        for motion_name, xv_cpu in clean_motion_variants(x.detach().cpu()[0], gen):
            xv = xv_cpu.unsqueeze(0).to(device)
            with torch.no_grad():
                logits, feats, _raw = wrapper.forward_with_features(xv)
                if logits.argmax(1).item() != y.item():
                    continue
            for layer, h in feats.items():
                key = (wrapper.name, layer)
                v = h.detach().cpu().numpy()[0] - feats0[layer]
                if np.linalg.norm(v) <= 1e-12:
                    continue
                vectors[key].append(v.astype(np.float32))
                rows.append({"model": wrapper.name, "layer": layer, "motion": motion_name, "vector_idx": len(vectors[key]) - 1})
    return rows, vectors


def collect_ga_dataset(args, dataset, device):
    rows = []
    features = defaultdict(list)
    grads = defaultdict(list)
    clean_rows = []
    clean_vectors = defaultdict(list)
    run_rows = []
    for spec in [x.strip() for x in args.models.split(",") if x.strip()]:
        wrapper = load_model(spec, device)
        indices = list(range(min(args.clean_motion_images, len(dataset))))
        cr, cv = collect_clean_motion(dataset, indices, wrapper, args, device)
        clean_rows.extend(cr)
        for k, vals in cv.items():
            clean_vectors[k].extend(vals)
        for target_s in args.classes.split(","):
            target = int(target_s)
            for seed in range(args.seeds):
                run_id = f"{wrapper.name}_class{target}_seed{seed}"
                generations, images, final_success, success_gen = run_ga(wrapper, target, args.seed + seed + target * 1000, args, device)
                for generation, image in zip(generations, images):
                    record_state(wrapper, image, target, rows, features, grads, run_id, generation, final_success, device)
                run_rows.append({
                    "model": wrapper.name,
                    "target_class": target,
                    "seed": seed,
                    "run_id": run_id,
                    "success": final_success,
                    "success_generation": -1 if success_gen is None else int(success_gen),
                    "n_saved": len(images),
                })
        print(f"[COLLECTED] {wrapper.name}", flush=True)
        del wrapper
        torch.cuda.empty_cache()
    return pd.DataFrame(rows), features, grads, pd.DataFrame(clean_rows), clean_vectors, pd.DataFrame(run_rows)


def make_segments(point_rows: pd.DataFrame, features, grads):
    seg_rows = []
    seg_vectors = defaultdict(list)
    seg_grads = defaultdict(list)
    for (model, run_id, layer), group in point_rows.groupby(["model", "run_id", "layer"]):
        group = group.sort_values("generation")
        key = (model, layer)
        feat_arr = features[key]
        grad_arr = grads[key]
        items = list(group.itertuples())
        for a, b in zip(items[:-1], items[1:]):
            v = feat_arr[int(b.feature_idx)] - feat_arr[int(a.feature_idx)]
            if np.linalg.norm(v) <= 1e-12:
                continue
            seg_vectors[key].append(v.astype(np.float32))
            seg_grads[key].append(grad_arr[int(a.feature_idx)].astype(np.float32))
            seg_rows.append({
                "model": model,
                "run_id": run_id,
                "layer": layer,
                "target_class": int(a.target_class),
                "start_generation": int(a.generation),
                "end_generation": int(b.generation),
                "success": int(a.success_final),
                "vector_idx": len(seg_vectors[key]) - 1,
            })
    return pd.DataFrame(seg_rows), seg_vectors, seg_grads


def save_npz(path, prefix, arrays):
    packed = {}
    for key, vals in arrays.items():
        packed[f"{prefix}__{'__'.join(key)}"] = np.stack(vals).astype(np.float32) if vals else np.empty((0, 1), dtype=np.float32)
    np.savez_compressed(path, **packed)


def get_npz(npz, prefix, model, layer):
    key = f"{prefix}__{model}__{layer}"
    return npz[key] if key in npz else np.empty((0, 1), dtype=np.float32)


def compare(pos, neg, mean, basis):
    rows = []
    if len(pos) < 5 or len(neg) < 5:
        return rows
    pe = projection_energy(pos, mean, basis)
    ne = projection_energy(neg, mean, basis)
    for k in KS:
        col = f"energy_k{k}"
        y = np.r_[np.ones(len(pe)), np.zeros(len(ne))]
        score = np.r_[pe[col].to_numpy(), ne[col].to_numpy()]
        rows.append({
            "k": k,
            "auroc": float(roc_auc_score(y, score)),
            "mannwhitney_p": float(mannwhitneyu(pe[col], ne[col]).pvalue),
            "success_mean_energy": float(pe[col].mean()),
            "negative_mean_energy": float(ne[col].mean()),
            "n_pos": len(pe),
            "n_neg": len(ne),
        })
    return rows


def logistic_acc(pos, neg, mean, basis):
    if len(pos) < 10 or len(neg) < 10:
        return np.nan
    pe = projection_energy(pos, mean, basis)
    ne = projection_energy(neg, mean, basis)
    x = pd.concat([pe, ne], ignore_index=True)[[f"energy_k{k}" for k in KS]].to_numpy()
    y = np.r_[np.ones(len(pe)), np.zeros(len(ne))].astype(int)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.35, stratify=y, random_state=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(xtr, ytr)
    return float(accuracy_score(yte, clf.predict(xte)))


def time_bin(start_generation: int, max_generation: int) -> str:
    if max_generation <= 0:
        frac = 0.0
    else:
        frac = min(max(start_generation / max_generation, 0.0), 0.999999)
    lo = int(frac // 0.2) * 20
    return f"{lo}-{lo + 20}%"


def add_energy_rows(rows, pos, neg, mean, basis, meta):
    for row in compare(pos, neg, mean, basis):
        row.update(meta)
        rows.append(row)
    acc = logistic_acc(pos, neg, mean, basis)
    rows.append({
        **meta,
        "k": "all",
        "logistic_accuracy": acc,
        "n_pos": len(pos),
        "n_neg": len(neg),
    })


def feature_basis(x, max_k):
    x = normalize_rows(x)
    xc = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return vt[: min(max_k, vt.shape[0])].T


def subspace_metrics(a, b, k):
    kk = min(k, a.shape[1], b.shape[1])
    if kk == 0 or a.shape[0] != b.shape[0]:
        return None
    s = np.linalg.svd(a[:, :kk].T @ b[:, :kk], compute_uv=False)
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


def analyze(args, out_dir):
    seg = pd.read_csv(out_dir / "segments.csv")
    vec_npz = np.load(out_dir / "segment_vectors.npz")
    grad_npz = np.load(out_dir / "segment_grads.npz")
    clean_npz = np.load(out_dir / "clean_vectors.npz")
    dim_rows, pred_rows, temporal_rows = [], [], []
    rng = np.random.default_rng(args.seed)

    for (model, layer), group in seg.groupby(["model", "layer"]):
        x = get_npz(vec_npz, "vectors", model, layer)
        g = get_npz(grad_npz, "grads", model, layer)
        clean = get_npz(clean_npz, "clean", model, layer)
        if len(x) != len(group) or len(x) < 10:
            continue
        group = group.reset_index(drop=True).copy()
        max_generation = max(int(group["end_generation"].max()), 1)
        group["time_bin"] = [time_bin(int(v), max_generation) for v in group["start_generation"]]

        variants = [("raw", normalize_rows(x), group)]
        if len(g) == len(x):
            orth, keep = orthogonalize(x, g)
            variants.append(("gradient_orthogonalized", orth, group.iloc[np.flatnonzero(keep)].reset_index(drop=True)))

        for variant, vectors, local_group in variants:
            success = local_group["success"].to_numpy(dtype=int) == 1
            success_runs = sorted(local_group.loc[success, "run_id"].unique())
            if len(success_runs) < 3:
                continue
            train_runs, test_runs = train_test_split(success_runs, test_size=0.35, random_state=args.seed)
            train = success & local_group["run_id"].isin(train_runs).to_numpy()
            test = success & local_group["run_id"].isin(test_runs).to_numpy()
            if train.sum() < 10 or test.sum() < 5:
                continue
            stats = pca_stats(vectors[train])
            stats.update({"model": model, "layer": layer, "variant": variant})
            dim_rows.append(stats)
            mean, basis = fit_basis(vectors[train], max(KS))
            pos = vectors[test]
            random_x = normalize_rows(rng.normal(size=(max(len(pos), 50), vectors.shape[1])))
            clean_x = normalize_rows(clean) if len(clean) else np.empty((0, vectors.shape[1]))
            for comp, neg in [("success_vs_random", random_x), ("success_vs_clean", clean_x)]:
                add_energy_rows(pred_rows, pos, neg, mean, basis, {
                    "model": model,
                    "layer": layer,
                    "variant": variant,
                    "comparison": comp,
                })

            for tb in ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]:
                train_bin = train & (local_group["time_bin"].to_numpy() == tb)
                test_bin = test & (local_group["time_bin"].to_numpy() == tb)
                if train_bin.sum() < 5 or test_bin.sum() < 3:
                    continue
                tb_mean, tb_basis = fit_basis(vectors[train_bin], max(KS))
                tb_pos = vectors[test_bin]
                tb_random = normalize_rows(rng.normal(size=(max(len(tb_pos), 50), vectors.shape[1])))
                for comp, neg in [("success_vs_random", tb_random), ("success_vs_clean", clean_x)]:
                    add_energy_rows(temporal_rows, tb_pos, neg, tb_mean, tb_basis, {
                        "model": model,
                        "layer": layer,
                        "variant": variant,
                        "time_bin": tb,
                        "comparison": comp,
                        "mean_start_generation": float(local_group.loc[test_bin, "start_generation"].mean()),
                    })

    dim_df = pd.DataFrame(dim_rows)
    pred_df = pd.DataFrame(pred_rows)
    temporal_df = pd.DataFrame(temporal_rows)
    dim_df.to_csv(out_dir / "layerwise_geometry_metrics.csv", index=False)
    pred_df.to_csv(out_dir / "layerwise_predictiveness_metrics.csv", index=False)
    orth_parts = []
    if len(dim_df):
        orth_parts.append(dim_df[dim_df["variant"] == "gradient_orthogonalized"].assign(metric_table="dimensionality"))
    if len(pred_df):
        orth_parts.append(pred_df[pred_df["variant"] == "gradient_orthogonalized"].assign(metric_table="predictiveness"))
    (pd.concat(orth_parts, ignore_index=True, sort=False) if orth_parts else pd.DataFrame()).to_csv(
        out_dir / "layerwise_gradient_orth_metrics.csv", index=False
    )
    temporal_df.to_csv(out_dir / "layerwise_temporal_metrics.csv", index=False)
    plot_summary(dim_df, pred_df, out_dir)
    plot_temporal(temporal_df, out_dir)


def plot_summary(dim_df, pred_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    if len(dim_df):
        raw = dim_df[dim_df["variant"] == "raw"]
        labels = [f"{r.model}\n{r.layer}" for r in raw.itertuples()]
        axes[0].bar(np.arange(len(raw)), raw["dim80"])
        axes[0].set_xticks(np.arange(len(raw)))
        axes[0].set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        axes[0].set_ylabel("dim80")
        axes[0].set_title("Layerwise success-flow dimensionality")
    if len(pred_df) and "auroc" in pred_df:
        sub = pred_df[(pred_df["k"].astype(str) == "20") & (pred_df["comparison"] == "success_vs_clean") & (pred_df["variant"] == "raw")]
        labels = [f"{r.model}\n{r.layer}" for r in sub.itertuples()]
        axes[1].bar(np.arange(len(sub)), sub["auroc"])
        axes[1].axhline(0.5, color="black", lw=1, ls="--")
        axes[1].set_xticks(np.arange(len(sub)))
        axes[1].set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        axes[1].set_ylabel("AUROC")
        axes[1].set_title("Success-vs-clean predictiveness, k=20")
    fig.savefig(out_dir / "layerwise_summary.png", dpi=180)
    plt.close(fig)


def plot_temporal(temporal_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    if len(temporal_df) and "auroc" in temporal_df:
        sub = temporal_df[
            (temporal_df["k"].astype(str) == "20")
            & (temporal_df["comparison"] == "success_vs_clean")
            & (temporal_df["variant"] == "raw")
        ].copy()
        order = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        for model, g in sub.groupby("model"):
            by_bin = g.groupby("time_bin")["auroc"].mean().reindex(order)
            axes[0].plot(order, by_bin, marker="o", label=model)
        axes[0].axhline(0.5, color="black", lw=1, ls="--")
        axes[0].set_ylabel("Mean AUROC")
        axes[0].set_title("Raw temporal emergence, k=20")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].legend(fontsize=7)

        sub_orth = temporal_df[
            (temporal_df["k"].astype(str) == "20")
            & (temporal_df["comparison"] == "success_vs_clean")
            & (temporal_df["variant"] == "gradient_orthogonalized")
        ].copy()
        for model, g in sub_orth.groupby("model"):
            by_bin = g.groupby("time_bin")["auroc"].mean().reindex(order)
            axes[1].plot(order, by_bin, marker="o", label=model)
        axes[1].axhline(0.5, color="black", lw=1, ls="--")
        axes[1].set_ylabel("Mean AUROC")
        axes[1].set_title("Gradient-orth temporal emergence, k=20")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].legend(fontsize=7)
    fig.savefig(out_dir / "layerwise_temporal_curves.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_layerwise_success_flow")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--models", default="bbb_resnet50,bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--generations", type=int, default=120)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--parents", type=int, default=16)
    p.add_argument("--elite", type=int, default=4)
    p.add_argument("--crossover", type=float, default=0.5)
    p.add_argument("--pixel-sigma", type=float, default=0.08)
    p.add_argument("--pixel-rate", type=float, default=0.08)
    p.add_argument("--block-rate", type=float, default=0.35)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--prob-threshold", type=float, default=0.999)
    p.add_argument("--stop-on-success", action="store_true")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--clean-motion-images", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reuse", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())

    required = [out_dir / "points.csv", out_dir / "segments.csv", out_dir / "segment_vectors.npz", out_dir / "segment_grads.npz", out_dir / "clean_vectors.npz"]
    if not args.reuse or not all(p.exists() for p in required):
        points, features, grads, clean_rows, clean_vectors, runs = collect_ga_dataset(args, dataset, device)
        segments, seg_vectors, seg_grads = make_segments(points, features, grads)
        points.to_csv(out_dir / "points.csv", index=False)
        segments.to_csv(out_dir / "segments.csv", index=False)
        runs.to_csv(out_dir / "runs.csv", index=False)
        clean_rows.to_csv(out_dir / "clean_motion.csv", index=False)
        save_npz(out_dir / "segment_vectors.npz", "vectors", seg_vectors)
        save_npz(out_dir / "segment_grads.npz", "grads", seg_grads)
        save_npz(out_dir / "clean_vectors.npz", "clean", clean_vectors)
        with open(out_dir / "metadata.json", "w") as f:
            json.dump({
                "args": vars(args),
                "layer_notes": {
                    "bbb_resnet50": "avgpool is adaptive-average-pooled layer4 because the local CIFAR ResNet exposes no explicit avgpool module.",
                    "bbb_vgg19_bn": "block labels correspond to the ReLU before each max-pool; penultimate is the final max-pool feature.",
                    "bbb_densenet": "local BlackboxBench DenseNet exposes dense1/dense2/dense3 plus avgpool; no denseblock4 module exists.",
                    "bbb_inception_v3": "mixed5/mixed6/mixed7 use Mixed_5d/Mixed_6e/Mixed_7c; penultimate is avgpool.",
                },
            }, f, indent=2)

    analyze(args, out_dir)
    runs = pd.read_csv(out_dir / "runs.csv")
    print("\nRUN SUCCESS")
    print(runs.groupby("model").agg(n_runs=("success", "size"), success_rate=("success", "mean"), successes=("success", "sum")).reset_index().to_string(index=False))
    print(f"\n[SAVED] {out_dir}")


if __name__ == "__main__":
    main()
