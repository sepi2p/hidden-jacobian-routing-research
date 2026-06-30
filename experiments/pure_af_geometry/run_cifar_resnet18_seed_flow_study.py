#!/usr/bin/env python3
"""Train CIFAR ResNet18 seed models and compare pure/adversarial flow geometry."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surro_models.cifar10_models.resnet import ResNet18  # noqa: E402


LAYERS = ["layer1", "layer2", "layer3", "layer4", "avgpool", "logits"]
KS = [5, 10, 20]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class ResNet18FeatureWrapper(nn.Module):
    def __init__(self, name: str, model: nn.Module):
        super().__init__()
        self.name = name
        self.model = model
        self.enabled = False
        self.captures: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.handles = []
        modules = dict(model.named_modules())
        for layer in ["layer1", "layer2", "layer3", "layer4"]:
            self.handles.append(modules[layer].register_forward_hook(self._hook(layer)))

    def _hook(self, label: str):
        def hook(_module, _inp, out):
            if self.enabled:
                self.captures[label].append(out)
                if label == "layer4":
                    self.captures["avgpool"].append(F.avg_pool2d(out, 4).flatten(1))
        return hook

    @staticmethod
    def pool(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x.flatten(1)

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
        for layer in ["layer1", "layer2", "layer3", "layer4"]:
            vals = self.captures.get(layer, [])
            if vals:
                feats[layer] = self.pool(vals[-1])
                raw[layer] = [vals[-1]]
        vals = self.captures.get("avgpool", [])
        if vals:
            feats["avgpool"] = vals[-1]
            raw["avgpool"] = [vals[-1]]
        feats["logits"] = logits
        raw["logits"] = [logits]
        return logits, feats, raw

    def close(self):
        for h in self.handles:
            h.remove()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def pca_basis(x: np.ndarray, max_k: int):
    x = normalize_rows(x.astype(np.float32))
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s * s
    ratios = var / np.clip(var.sum(), 1e-12, None)
    csum = np.cumsum(ratios)
    entropy = -float(np.sum(ratios[ratios > 0] * np.log(ratios[ratios > 0])))
    return mean, vt[: min(max_k, vt.shape[0])], {
        "n_segments": int(len(x)),
        "d": int(x.shape[1]),
        "pc1_var": float(ratios[0]) if len(ratios) else np.nan,
        "pc5_cum_var": float(csum[min(4, len(csum) - 1)]) if len(csum) else np.nan,
        "pc10_cum_var": float(csum[min(9, len(csum) - 1)]) if len(csum) else np.nan,
        "dim80": int(np.searchsorted(csum, 0.8) + 1) if len(csum) else 0,
        "dim90": int(np.searchsorted(csum, 0.9) + 1) if len(csum) else 0,
        "effective_rank": float(np.exp(entropy)) if len(csum) else np.nan,
    }


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    x = normalize_rows(x.astype(np.float32))
    xc = x - mean
    kk = min(k, basis.shape[0])
    coeff = xc @ basis[:kk].T
    return np.sum(coeff * coeff, axis=1) / np.clip(np.sum(xc * xc, axis=1), 1e-12, None)


def subspace_overlap(a: np.ndarray, b: np.ndarray, k: int):
    kk = min(k, a.shape[0], b.shape[0])
    if kk < 1:
        return None
    s = np.linalg.svd(a[:kk] @ b[:kk].T, compute_uv=False)
    s = np.clip(s, 0, 1)
    angles = np.arccos(s)
    return {
        "k": int(k),
        "projection_overlap": float(np.sum(s * s) / kk),
        "subspace_affinity": float(np.sqrt(np.sum(s * s) / kk)),
        "mean_principal_angle_deg": float(np.degrees(angles).mean()),
        "max_principal_angle_deg": float(np.degrees(angles).max()),
    }


def train_one(seed: int, args, device):
    ckpt = Path(args.checkpoint_dir) / f"resnet18_seed{seed}.pt"
    if ckpt.exists() and not args.retrain:
        return ckpt
    set_seed(seed)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.ToTensor()
    train_set = datasets.CIFAR10(args.dataset_root, train=True, download=False, transform=train_tf)
    test_set = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    model = ResNet18().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_acc = 0.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        acc = evaluate_accuracy(model, test_loader, device, max_batches=args.eval_batches)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"[TRAIN] seed={seed} epoch={epoch+1}/{args.epochs} acc={acc:.4f} best={best_acc:.4f}", flush=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": best_state or model.state_dict(), "seed": seed, "best_acc": best_acc, "epochs": args.epochs}, ckpt)
    del model
    torch.cuda.empty_cache()
    return ckpt


def evaluate_accuracy(model, loader, device, max_batches=0):
    model.eval()
    ok = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            ok += int((pred == y).sum().item())
            total += int(y.numel())
    return ok / max(total, 1)


def load_seed_model(seed: int, args, device):
    ckpt = Path(args.checkpoint_dir) / f"resnet18_seed{seed}.pt"
    model = ResNet18().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["net"] if "net" in state else state)
    model.eval()
    return ResNet18FeatureWrapper(f"resnet18_seed{seed}", model).to(device).eval()


def margin(logits: torch.Tensor, y: torch.Tensor):
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.view(-1, 1), -1e9)
    return true - masked.max(1).values


def record_features(wrapper, image, label: int, device):
    probe = image.detach().requires_grad_(True)
    logits, feats, raw = wrapper.forward_with_features(probe)
    logp = F.log_softmax(logits, dim=1)[0, label]
    raw_items = []
    raw_labels = []
    for layer in LAYERS:
        for item in raw.get(layer, []):
            raw_items.append(item)
            raw_labels.append(layer)
    grads = torch.autograd.grad(logp, raw_items, retain_graph=False, allow_unused=True)
    grad_by_layer = {}
    for layer, item, grad in zip(raw_labels, raw_items, grads):
        grad_by_layer[layer] = torch.zeros_like(item) if grad is None else grad
    out = {}
    for layer, h in feats.items():
        g = grad_by_layer.get(layer)
        if g is None and layer == "avgpool":
            g = torch.zeros_like(h)
        elif g is not None and g.ndim == 4:
            g = F.adaptive_avg_pool2d(g, (1, 1)).flatten(1)
        out[layer] = {
            "feature": h.detach().cpu().numpy()[0].astype(np.float32),
            "grad": (g.detach().cpu().numpy()[0].astype(np.float32) if g is not None else np.zeros_like(h.detach().cpu().numpy()[0], dtype=np.float32)),
        }
    probs = F.softmax(logits, dim=1)
    return out, {
        "pred": int(logits.argmax(1).item()),
        "prob": float(probs[0, label].item()),
        "margin": float(margin(logits, torch.tensor([label], device=device)).item()),
    }


def collect_clean_motion(wrapper, dataset, args, device):
    rows = []
    vectors = defaultdict(list)
    gen = torch.Generator().manual_seed(args.seed + 333)
    for idx in range(min(args.clean_motion_images, len(dataset))):
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            logits, feats0, _ = wrapper.forward_with_features(x)
            if int(logits.argmax(1).item()) != int(y):
                continue
            feats0 = {k: v.detach().cpu().numpy()[0] for k, v in feats0.items()}
        variants = [
            ("crop", TF.resized_crop(x.detach().cpu()[0], 2, 2, 28, 28, [32, 32], antialias=True)),
            ("color", TF.adjust_contrast(TF.adjust_brightness(x.detach().cpu()[0], 1.2), 0.85).clamp(0, 1)),
            ("blur", TF.gaussian_blur(x.detach().cpu()[0], [5, 5], [0.8, 0.8])),
            ("noise", (x.detach().cpu()[0] + torch.randn(x.detach().cpu()[0].shape, generator=gen) * 0.03).clamp(0, 1)),
        ]
        for motion, xv_cpu in variants:
            xv = xv_cpu.unsqueeze(0).to(device)
            with torch.no_grad():
                logits, feats, _ = wrapper.forward_with_features(xv)
                if int(logits.argmax(1).item()) != int(y):
                    continue
            for layer, h in feats.items():
                v = h.detach().cpu().numpy()[0] - feats0[layer]
                if np.linalg.norm(v) > 1e-12:
                    vectors[layer].append(v.astype(np.float32))
                    rows.append({"layer": layer, "motion": motion, "vector_idx": len(vectors[layer]) - 1})
    return pd.DataFrame(rows), vectors


def run_ga(wrapper, target: int, seed: int, args, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    pop = torch.rand((args.ga_population, 3, 32, 32), generator=gen, device=device)
    saved = []
    success = 0
    for generation in range(args.ga_generations + 1):
        with torch.no_grad():
            logits = wrapper(pop)
            probs = F.softmax(logits, dim=1)[:, target]
            order = torch.argsort(logits[:, target], descending=True)
        pop = pop[order]
        logits = logits[order]
        probs = probs[order]
        pred = logits.argmax(1)
        if generation % args.ga_save_every == 0 or generation == args.ga_generations:
            saved.append((generation, pop[:1].detach().clone()))
        if probs[0].item() >= args.pure_threshold and int(pred[0].item()) == target:
            success = 1
            saved.append((generation, pop[:1].detach().clone()))
            if args.stop_on_pure_success:
                break
        if generation == args.ga_generations:
            break
        parents = pop[: args.ga_parents]
        elite = pop[: args.ga_elite]
        a = parents[torch.randint(0, len(parents), (args.ga_population - args.ga_elite,), generator=gen, device=device)]
        b = parents[torch.randint(0, len(parents), (args.ga_population - args.ga_elite,), generator=gen, device=device)]
        mask = torch.rand((len(a), 1, 32, 32), generator=gen, device=device) < 0.5
        children = torch.where(mask, a, b)
        pix = torch.rand(children.shape, generator=gen, device=device) < args.ga_pixel_rate
        children = torch.where(pix, children + torch.randn(children.shape, generator=gen, device=device) * args.ga_pixel_sigma, children).clamp(0, 1)
        pop = torch.cat([elite, children], dim=0)
    return saved, success


def collect_pure_flow(wrapper, model_seed: int, dataset, args, device):
    rows = []
    vectors = defaultdict(list)
    grads = defaultdict(list)
    point_rows = []
    run_rows = []
    for cls in range(10):
        for ga_seed in range(args.ga_seeds_per_class):
            run_id = f"modelseed{model_seed}_class{cls}_gaseed{ga_seed}"
            states, success = run_ga(wrapper, cls, args.seed + model_seed * 10000 + cls * 100 + ga_seed, args, device)
            feat_points = defaultdict(list)
            grad_points = defaultdict(list)
            for generation, image in states:
                rec, meta = record_features(wrapper, image, cls, device)
                for layer, data in rec.items():
                    feat_points[layer].append((generation, data["feature"]))
                    grad_points[layer].append((generation, data["grad"]))
                    point_rows.append({"model_seed": model_seed, "run_id": run_id, "flow": "pure", "target_class": cls, "generation": generation, "layer": layer, "success": success, **meta})
            for layer, vals in feat_points.items():
                vals = sorted(vals, key=lambda item: item[0])
                gvals = dict(grad_points[layer])
                for (ga, fa), (gb, fb) in zip(vals[:-1], vals[1:]):
                    v = fb - fa
                    if np.linalg.norm(v) <= 1e-12:
                        continue
                    vectors[layer].append(v.astype(np.float32))
                    grads[layer].append(gvals[ga].astype(np.float32))
                    rows.append({"model_seed": model_seed, "flow": "pure", "run_id": run_id, "target_class": cls, "layer": layer, "start_step": ga, "end_step": gb, "success": success, "vector_idx": len(vectors[layer]) - 1})
            run_rows.append({"model_seed": model_seed, "flow": "pure", "run_id": run_id, "target_class": cls, "success": success, "n_saved": len(states)})
    return pd.DataFrame(rows), vectors, grads, pd.DataFrame(point_rows), pd.DataFrame(run_rows)


def project_linf(x_adv, x0, eps):
    return torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)


def collect_adv_flow(wrapper, model_seed: int, dataset, args, device):
    selected = []
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        xb = x.unsqueeze(0).to(device)
        with torch.no_grad():
            if int(wrapper(xb).argmax(1).item()) == int(y):
                selected.append((idx, int(y)))
        if len(selected) >= args.adv_images:
            break
    rows = []
    vectors = defaultdict(list)
    grads = defaultdict(list)
    point_rows = []
    run_rows = []
    eps = args.adv_eps / 255.0
    alpha = eps / max(args.adv_steps, 1)
    for image_ord, (idx, label) in enumerate(selected):
        x_cpu, _ = dataset[idx]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        x = x0.clone()
        states = [(0, x.detach().clone())]
        for step in range(1, args.adv_steps + 1):
            probe = x.detach().requires_grad_(True)
            loss = F.cross_entropy(wrapper(probe), y)
            gx = torch.autograd.grad(loss, probe)[0]
            x = project_linf(x + alpha * gx.sign(), x0, eps)
            states.append((step, x.detach().clone()))
        with torch.no_grad():
            final_success = int(wrapper(states[-1][1]).argmax(1).item() != label)
        run_id = f"modelseed{model_seed}_adv_idx{idx}"
        feat_points = defaultdict(list)
        grad_points = defaultdict(list)
        for step, image in states:
            rec, meta = record_features(wrapper, image, label, device)
            for layer, data in rec.items():
                feat_points[layer].append((step, data["feature"]))
                grad_points[layer].append((step, data["grad"]))
                point_rows.append({"model_seed": model_seed, "flow": "adv", "run_id": run_id, "target_class": label, "generation": step, "layer": layer, "success": final_success, **meta})
        for layer, vals in feat_points.items():
            vals = sorted(vals, key=lambda item: item[0])
            gvals = dict(grad_points[layer])
            for (sa, fa), (sb, fb) in zip(vals[:-1], vals[1:]):
                v = fb - fa
                if np.linalg.norm(v) <= 1e-12:
                    continue
                vectors[layer].append(v.astype(np.float32))
                grads[layer].append(gvals[sa].astype(np.float32))
                rows.append({"model_seed": model_seed, "flow": "adv", "run_id": run_id, "target_class": label, "layer": layer, "start_step": sa, "end_step": sb, "success": final_success, "vector_idx": len(vectors[layer]) - 1})
        run_rows.append({"model_seed": model_seed, "flow": "adv", "run_id": run_id, "target_class": label, "success": final_success, "n_saved": len(states)})
    return pd.DataFrame(rows), vectors, grads, pd.DataFrame(point_rows), pd.DataFrame(run_rows)


def analyze_seed(seed: int, flow: str, seg: pd.DataFrame, vectors: dict, clean_vectors: dict, out_dir: Path):
    dim_rows = []
    pred_rows = []
    basis = {}
    class_dirs = {}
    for layer in LAYERS:
        x = np.stack(vectors[layer]).astype(np.float32) if vectors.get(layer) else np.empty((0, 1), dtype=np.float32)
        sub = seg[seg.layer == layer].reset_index(drop=True)
        if len(x) < 10 or len(x) != len(sub):
            continue
        success = sub.success.to_numpy(dtype=int) == 1
        if success.sum() < 5:
            continue
        mean, b, stats = pca_basis(x[success], max(KS))
        basis[layer] = b
        stats.update({"model_seed": seed, "flow": flow, "layer": layer})
        dim_rows.append(stats)
        clean = np.stack(clean_vectors[layer]).astype(np.float32) if clean_vectors.get(layer) else np.empty((0, x.shape[1]), dtype=np.float32)
        if len(clean) >= 5:
            for k in KS:
                pos = projection_energy(x[success], mean, b, k)
                neg = projection_energy(clean, mean, b, k)
                y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
                score = np.r_[pos, neg]
                pred_rows.append({"model_seed": seed, "flow": flow, "layer": layer, "comparison": "flow_vs_clean", "k": k, "auroc": float(roc_auc_score(y, score)), "n_pos": len(pos), "n_neg": len(neg)})
        rng = np.random.default_rng(seed + len(layer))
        random_x = normalize_rows(rng.normal(size=(max(int(success.sum()), 50), x.shape[1])).astype(np.float32))
        for k in KS:
            pos = projection_energy(x[success], mean, b, k)
            neg = projection_energy(random_x, mean, b, k)
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            pred_rows.append({"model_seed": seed, "flow": flow, "layer": layer, "comparison": "flow_vs_random", "k": k, "auroc": float(roc_auc_score(y, np.r_[pos, neg])), "n_pos": len(pos), "n_neg": len(neg)})
        for cls, g in sub[success].groupby("target_class"):
            vals = []
            for run_id, rg in g.groupby("run_id"):
                arr = x[rg.vector_idx.to_numpy(dtype=int)]
                if len(arr):
                    v = arr.sum(axis=0)
                    if np.linalg.norm(v) > 1e-12:
                        vals.append(v / np.linalg.norm(v))
            if vals:
                d = np.mean(vals, axis=0)
                d = d / np.clip(np.linalg.norm(d), 1e-12, None)
                class_dirs[(layer, int(cls))] = d.astype(np.float32)
    return pd.DataFrame(dim_rows), pd.DataFrame(pred_rows), basis, class_dirs


def cosine(a, b):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def compare_across_seeds(seed_results, out_dir: Path):
    dim_parts = [r["dim"] for r in seed_results if not r["dim"].empty]
    pred_parts = [r["pred"] for r in seed_results if not r["pred"].empty]
    dim = pd.concat(dim_parts, ignore_index=True) if dim_parts else pd.DataFrame()
    pred = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    dim.to_csv(out_dir / "seed_flow_dimensionality.csv", index=False)
    pred.to_csv(out_dir / "seed_flow_predictiveness.csv", index=False)
    if pred.empty or "comparison" not in pred.columns:
        dom = pd.DataFrame(columns=["model_seed", "flow", "dominant_layer", "dominant_auroc_k20"])
    else:
        dom = pred[(pred.comparison == "flow_vs_clean") & (pred.k == 20)].sort_values("auroc", ascending=False).groupby(["model_seed", "flow"], as_index=False).first()
        dom = dom.rename(columns={"layer": "dominant_layer", "auroc": "dominant_auroc_k20"})
    dom.to_csv(out_dir / "seed_flow_dominant_layers.csv", index=False)
    overlap_rows = []
    class_rows = []
    for flow in ["pure", "adv"]:
        for layer in LAYERS:
            for ra, rb in combinations(seed_results, 2):
                ba = ra["basis"].get((flow, layer))
                bb = rb["basis"].get((flow, layer))
                if ba is not None and bb is not None:
                    for k in [1, 3, 5]:
                        m = subspace_overlap(ba, bb, k)
                        if m:
                            overlap_rows.append({"flow": flow, "layer": layer, "seed_a": ra["seed"], "seed_b": rb["seed"], **m})
                for cls in range(10):
                    da = ra["class_dirs"].get((flow, layer, cls))
                    db = rb["class_dirs"].get((flow, layer, cls))
                    if da is not None and db is not None:
                        class_rows.append({"flow": flow, "layer": layer, "class": cls, "seed_a": ra["seed"], "seed_b": rb["seed"], "cosine": cosine(da, db)})
    overlap = pd.DataFrame(overlap_rows)
    class_df = pd.DataFrame(class_rows)
    overlap.to_csv(out_dir / "cross_seed_pc_overlap.csv", index=False)
    class_df.to_csv(out_dir / "cross_seed_class_direction_overlap.csv", index=False)

    pure_adv_rows = []
    for r in seed_results:
        for layer in LAYERS:
            bp = r["basis"].get(("pure", layer))
            ba = r["basis"].get(("adv", layer))
            if bp is not None and ba is not None:
                for k in [1, 3, 5]:
                    m = subspace_overlap(bp, ba, k)
                    if m:
                        pure_adv_rows.append({"model_seed": r["seed"], "layer": layer, **m})
    pure_adv = pd.DataFrame(pure_adv_rows)
    pure_adv.to_csv(out_dir / "within_seed_pure_vs_adv_pc_overlap.csv", index=False)
    plot_summary(dom, overlap, class_df, pure_adv, out_dir)
    return dom, overlap, class_df, pure_adv


def plot_summary(dom, overlap, class_df, pure_adv, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    if not dom.empty:
        for flow, g in dom.groupby("flow"):
            axes[0, 0].scatter(g["model_seed"], g["dominant_auroc_k20"], label=flow, s=60)
            for row in g.itertuples():
                axes[0, 0].annotate(row.dominant_layer, (row.model_seed, row.dominant_auroc_k20), fontsize=8)
        axes[0, 0].set_ylim(0.45, 1.02)
        axes[0, 0].set_title("Dominant layer by seed")
        axes[0, 0].set_xlabel("model seed")
        axes[0, 0].set_ylabel("flow-vs-clean AUROC k=20")
        axes[0, 0].legend()
    if not overlap.empty:
        sub = overlap[overlap.k == 3]
        for flow, g in sub.groupby("flow"):
            by = g.groupby("layer")["projection_overlap"].mean().reindex(LAYERS)
            axes[0, 1].plot(LAYERS, by, marker="o", label=flow)
        axes[0, 1].set_title("Cross-seed PC overlap k=3")
        axes[0, 1].tick_params(axis="x", rotation=30)
        axes[0, 1].legend()
    if not class_df.empty:
        for flow, g in class_df.groupby("flow"):
            by = g.groupby("layer")["cosine"].mean().reindex(LAYERS)
            axes[1, 0].plot(LAYERS, by, marker="o", label=flow)
        axes[1, 0].set_title("Cross-seed class direction cosine")
        axes[1, 0].tick_params(axis="x", rotation=30)
        axes[1, 0].legend()
    if not pure_adv.empty:
        sub = pure_adv[pure_adv.k == 3]
        by = sub.groupby("layer")["projection_overlap"].mean().reindex(LAYERS)
        axes[1, 1].plot(LAYERS, by, marker="o")
        axes[1, 1].set_title("Within-seed pure-vs-adv PC overlap k=3")
        axes[1, 1].tick_params(axis="x", rotation=30)
    fig.savefig(out_dir / "resnet18_seed_flow_summary.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.model_seeds.split(",") if s.strip()]
    for seed in seeds:
        train_one(seed, args, device)
    test_set = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    seed_results = []
    all_runs = []
    for seed in seeds:
        wrapper = load_seed_model(seed, args, device)
        clean_rows, clean_vectors = collect_clean_motion(wrapper, test_set, args, device)
        clean_rows.to_csv(out_dir / f"clean_motion_seed{seed}.csv", index=False)
        pseg, pvec, pgrad, ppoints, pruns = collect_pure_flow(wrapper, seed, test_set, args, device)
        aseg, avec, agrad, apoints, aruns = collect_adv_flow(wrapper, seed, test_set, args, device)
        pseg.to_csv(out_dir / f"pure_segments_seed{seed}.csv", index=False)
        aseg.to_csv(out_dir / f"adv_segments_seed{seed}.csv", index=False)
        ppoints.to_csv(out_dir / f"pure_points_seed{seed}.csv", index=False)
        apoints.to_csv(out_dir / f"adv_points_seed{seed}.csv", index=False)
        pd.concat([pruns, aruns], ignore_index=True).to_csv(out_dir / f"runs_seed{seed}.csv", index=False)
        all_runs.append(pd.concat([pruns, aruns], ignore_index=True))
        seed_basis = {}
        seed_dirs = {}
        dim_parts = []
        pred_parts = []
        for flow, seg, vec in [("pure", pseg, pvec), ("adv", aseg, avec)]:
            dim, pred, basis, class_dirs = analyze_seed(seed, flow, seg, vec, clean_vectors, out_dir)
            dim_parts.append(dim)
            pred_parts.append(pred)
            for layer, b in basis.items():
                seed_basis[(flow, layer)] = b
            for (layer, cls), d in class_dirs.items():
                seed_dirs[(flow, layer, cls)] = d
        seed_results.append({
            "seed": seed,
            "dim": pd.concat(dim_parts, ignore_index=True),
            "pred": pd.concat(pred_parts, ignore_index=True),
            "basis": seed_basis,
            "class_dirs": seed_dirs,
        })
        wrapper.close()
        del wrapper
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[DONE SEED] {seed}", flush=True)
    pd.concat(all_runs, ignore_index=True).to_csv(out_dir / "all_runs.csv", index=False)
    dom, overlap, class_df, pure_adv = compare_across_seeds(seed_results, out_dir)
    metadata = {"args": vars(args), "model_seeds": seeds}
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] {out_dir}", flush=True)
    print("\nRUN COUNTS", flush=True)
    print(pd.concat(all_runs, ignore_index=True).groupby(["flow", "model_seed"]).agg(successes=("success", "sum"), runs=("success", "size")).reset_index().to_string(index=False), flush=True)
    print("\nDOMINANT LAYERS", flush=True)
    print(dom.to_string(index=False), flush=True)
    if not overlap.empty:
        print("\nCROSS-SEED PC OVERLAP k=3", flush=True)
        print(overlap[overlap.k == 3].groupby(["flow", "layer"])["projection_overlap"].mean().reset_index().to_string(index=False), flush=True)
    if not class_df.empty:
        print("\nCROSS-SEED CLASS DIRECTION COSINE", flush=True)
        print(class_df.groupby(["flow", "layer"])["cosine"].mean().reset_index().to_string(index=False), flush=True)
    if not pure_adv.empty:
        print("\nWITHIN-SEED PURE-VS-ADV PC OVERLAP k=3", flush=True)
        print(pure_adv[pure_adv.k == 3].groupby("layer")["projection_overlap"].mean().reset_index().to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_resnet18_seed_flow_study")
    p.add_argument("--checkpoint-dir", default="checkpoints/cifar10_resnet18_seed_study")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model-seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--eval-batches", type=int, default=0)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--clean-motion-images", type=int, default=250)
    p.add_argument("--ga-seeds-per-class", type=int, default=1)
    p.add_argument("--ga-generations", type=int, default=80)
    p.add_argument("--ga-population", type=int, default=48)
    p.add_argument("--ga-parents", type=int, default=12)
    p.add_argument("--ga-elite", type=int, default=4)
    p.add_argument("--ga-save-every", type=int, default=5)
    p.add_argument("--ga-pixel-rate", type=float, default=0.08)
    p.add_argument("--ga-pixel-sigma", type=float, default=0.08)
    p.add_argument("--pure-threshold", type=float, default=0.999)
    p.add_argument("--stop-on-pure-success", action="store_true")
    p.add_argument("--adv-images", type=int, default=100)
    p.add_argument("--adv-steps", type=int, default=10)
    p.add_argument("--adv-eps", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
