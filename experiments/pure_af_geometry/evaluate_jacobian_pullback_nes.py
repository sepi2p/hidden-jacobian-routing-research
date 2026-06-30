#!/usr/bin/env python3
"""Jacobian-pullback NES in successful-flow PCA feature subspaces."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, models, transforms


LAYERS = ["clf_logits", "clf_avgpool", "clf_layer4"]


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class ResNet18WithFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.base = models.resnet18(pretrained=True)

    def forward_with_feature(self, x, layer: str):
        z = self.normalize(x)
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
        if layer in {"clf_avgpool", "clf_layer4"}:
            return logits, pooled
        if layer == "clf_logits":
            return logits, logits
        raise ValueError(layer)

    def forward(self, x):
        return self.forward_with_feature(x, "clf_logits")[0]


def load_target(name: str, device):
    if name == "densenet121":
        base = models.densenet121(pretrained=True)
    elif name == "vgg16":
        base = models.vgg16_bn(pretrained=True)
    elif name == "resnet18":
        base = models.resnet18(pretrained=True)
    else:
        raise ValueError(name)
    return nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), base).to(device).eval()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def fit_pca_bases(manifest_path: str, max_k: int, seed: int):
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[(manifest["trajectory_features_npz"].notna()) & (manifest["success"].astype(int) == 1)].copy()
    bases = {}
    for layer in LAYERS:
        segs = []
        for _idx, row in manifest.iterrows():
            z = np.load(row["trajectory_features_npz"])
            feats = z[layer].astype(np.float64)
            for t in range(len(feats) - 1):
                v = feats[t + 1] - feats[t]
                if np.linalg.norm(v) > 1e-12:
                    segs.append(v)
        x = normalize_rows(np.stack(segs))
        # Deterministic shuffle before SVD is not mathematically needed, but keeps train-subset option stable.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(x))
        x = x[order]
        xc = x - x.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
        bases[layer] = torch.from_numpy(vt[:max_k].astype(np.float32))
    return bases


def project_linf(x, clean, eps):
    return torch.max(torch.min(x, clean + eps), clean - eps).clamp(0, 1)


def target_loss(model, x, y):
    return F.cross_entropy(model(x), y, reduction="none")


def make_pullback_dirs(source, clean, layer: str, basis: torch.Tensor):
    clean_req = clean.detach().clone().requires_grad_(True)
    _logits, h = source.forward_with_feature(clean_req, layer)
    dirs = []
    for j in range(basis.shape[0]):
        grad = torch.autograd.grad((h * basis[j:j+1].to(clean.device)).sum(), clean_req, retain_graph=True)[0].detach()
        grad = grad / grad.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        dirs.append(grad)
    return torch.cat(dirs, dim=0)


def coeff_to_adv(clean, alpha, pullback, eps):
    delta = (alpha.view(-1, 1, 1, 1) * pullback).sum(dim=0, keepdim=True)
    return project_linf(clean + delta, clean, eps)


def nes_coeff_attack(target, clean, y, pullback, args, rng, query_curve_steps):
    k = pullback.shape[0]
    alpha = torch.zeros(k, device=clean.device)
    queries = 0
    curve = {}
    with torch.no_grad():
        clean_pred = int(target(clean).argmax(1).item())
    if clean_pred != int(y.item()):
        return True, 0, curve, clean_pred
    best_pred = clean_pred
    for step in range(args.nes_steps):
        z = torch.randn((args.nes_samples, k), generator=rng, device=clean.device)
        losses_pos = []
        losses_neg = []
        preds = []
        with torch.no_grad():
            for i in range(args.nes_samples):
                adv_p = coeff_to_adv(clean, alpha + args.sigma * z[i], pullback, args.eps)
                adv_n = coeff_to_adv(clean, alpha - args.sigma * z[i], pullback, args.eps)
                lp = target_loss(target, adv_p, y)
                ln = target_loss(target, adv_n, y)
                losses_pos.append(lp.item())
                losses_neg.append(ln.item())
                preds.extend([int(target(adv_p).argmax(1).item()), int(target(adv_n).argmax(1).item())])
        queries += 2 * args.nes_samples
        best_pred = preds[-2] if preds else best_pred
        if any(p != int(y.item()) for p in preds):
            for q in query_curve_steps:
                if queries >= q and q not in curve:
                    curve[q] = 1
            return True, queries, curve, next(p for p in preds if p != int(y.item()))
        loss_diff = torch.tensor(losses_pos, device=clean.device) - torch.tensor(losses_neg, device=clean.device)
        grad = (loss_diff.view(-1, 1) * z).mean(dim=0) / (2.0 * args.sigma)
        alpha = alpha + args.lr * grad
        norm = alpha.norm(p=2).clamp_min(1e-12)
        # Conservative coefficient clipping; pixel projection still enforces Linf.
        if norm > args.alpha_l2_clip:
            alpha = alpha / norm * args.alpha_l2_clip
        with torch.no_grad():
            pred = int(target(coeff_to_adv(clean, alpha, pullback, args.eps)).argmax(1).item())
        queries += 1
        best_pred = pred
        for q in query_curve_steps:
            if queries >= q and q not in curve:
                curve[q] = int(pred != int(y.item()))
        if pred != int(y.item()):
            return True, queries, curve, pred
        if queries >= args.max_queries:
            break
    return False, queries, curve, best_pred


def pixel_nes_attack(target, clean, y, args, rng, query_curve_steps):
    pullback = None
    delta = torch.zeros_like(clean)
    queries = 0
    curve = {}
    with torch.no_grad():
        clean_pred = int(target(clean).argmax(1).item())
    if clean_pred != int(y.item()):
        return True, 0, curve, clean_pred
    best_pred = clean_pred
    for _step in range(args.nes_steps):
        z = torch.randn((args.nes_samples,) + tuple(clean.shape[1:]), generator=rng, device=clean.device)
        z = z / z.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        losses_pos = []
        losses_neg = []
        preds = []
        with torch.no_grad():
            for i in range(args.nes_samples):
                adv_p = project_linf(clean + delta + args.sigma * z[i:i+1], clean, args.eps)
                adv_n = project_linf(clean + delta - args.sigma * z[i:i+1], clean, args.eps)
                losses_pos.append(target_loss(target, adv_p, y).item())
                losses_neg.append(target_loss(target, adv_n, y).item())
                preds.extend([int(target(adv_p).argmax(1).item()), int(target(adv_n).argmax(1).item())])
        queries += 2 * args.nes_samples
        if any(p != int(y.item()) for p in preds):
            for q in query_curve_steps:
                if queries >= q and q not in curve:
                    curve[q] = 1
            return True, queries, curve, next(p for p in preds if p != int(y.item()))
        loss_diff = torch.tensor(losses_pos, device=clean.device).view(-1, 1, 1, 1) - torch.tensor(losses_neg, device=clean.device).view(-1, 1, 1, 1)
        grad = (loss_diff * z).mean(dim=0, keepdim=True) / (2.0 * args.sigma)
        delta = (delta + args.lr * grad.sign()).clamp(-args.eps, args.eps)
        adv = (clean + delta).clamp(0, 1)
        with torch.no_grad():
            pred = int(target(adv).argmax(1).item())
        queries += 1
        best_pred = pred
        for q in query_curve_steps:
            if queries >= q and q not in curve:
                curve[q] = int(pred != int(y.item()))
        if pred != int(y.item()):
            return True, queries, curve, pred
        if queries >= args.max_queries:
            break
    return False, queries, curve, best_pred


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    p.add_argument("--trajectory-manifest", default="analysis_outputs/pure_af_geometry/random_ga_classifier_trajectories_resnet18_c10_s5/manifest.csv")
    p.add_argument("--indices-metadata", default="analysis_outputs/pure_af_geometry/away_from_true_attractor_resnet18_all_available_c10.csv.metadata.json")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_pullback_nes")
    p.add_argument("--max-images", type=int, default=20)
    p.add_argument("--target-models", default="densenet121,vgg16")
    p.add_argument("--layers", default="clf_logits,clf_avgpool,clf_layer4")
    p.add_argument("--ks", default="5,10,20,50")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--alpha-l2-clip", type=float, default=5.0)
    p.add_argument("--nes-samples", type=int, default=10)
    p.add_argument("--nes-steps", type=int, default=20)
    p.add_argument("--max-queries", type=int, default=500)
    p.add_argument("--query-curve", default="50,100,200,300,500")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    query_curve = [int(x) for x in args.query_curve.split(",") if x.strip()]
    max_k = max(ks)

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    indices = json.loads(Path(args.indices_metadata).read_text())["indices"][: args.max_images]
    source = ResNet18WithFeatures().to(device).eval()
    targets = {name: load_target(name, device) for name in [x.strip() for x in args.target_models.split(",") if x.strip()]}
    bases = fit_pca_bases(args.trajectory_manifest, max_k, args.seed)

    rows = []
    started = time.time()
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 2026)
    for image_i, idx in enumerate(indices, start=1):
        clean_cpu, label_int = dataset[int(idx)]
        clean = clean_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(label_int)], device=device)
        for target_name, target in targets.items():
            with torch.no_grad():
                clean_target_pred = int(target(clean).argmax(1).item())
            clean_target_correct = int(clean_target_pred == int(y.item()))
            # Pixel NES baseline once per image/target.
            if clean_target_correct:
                success, queries, curve, pred = pixel_nes_attack(target, clean, y, args, gen, query_curve)
            else:
                success, queries, curve, pred = False, 0, {}, clean_target_pred
            base_row = {
                "dataset_idx": int(idx), "source_class": int(label_int), "target_model": target_name,
                "method": "pixel_nes", "layer": "", "k": 0, "success": int(success),
                "queries": int(queries), "final_pred": int(pred), "clean_target_correct": clean_target_correct,
                "clean_target_pred": clean_target_pred,
            }
            for q in query_curve:
                base_row[f"success_at_q{q}"] = int(curve.get(q, int(success and queries <= q)))
            rows.append(base_row)
            for layer in layers:
                pullback_full = make_pullback_dirs(source, clean, layer, bases[layer].to(device))
                for k in ks:
                    if clean_target_correct:
                        success, queries, curve, pred = nes_coeff_attack(target, clean, y, pullback_full[:k], args, gen, query_curve)
                    else:
                        success, queries, curve, pred = False, 0, {}, clean_target_pred
                    row = {
                        "dataset_idx": int(idx), "source_class": int(label_int), "target_model": target_name,
                        "method": "pullback_nes", "layer": layer, "k": int(k), "success": int(success),
                        "queries": int(queries), "final_pred": int(pred), "clean_target_correct": clean_target_correct,
                        "clean_target_pred": clean_target_pred,
                    }
                    for q in query_curve:
                        row[f"success_at_q{q}"] = int(curve.get(q, int(success and queries <= q)))
                    rows.append(row)
        if image_i % 5 == 0:
            print(f"[progress] images={image_i}/{len(indices)} elapsed={time.time()-started:.1f}s", flush=True)

    csv_path = out_dir / "jacobian_pullback_nes_results.csv"
    fieldnames = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    df = pd.DataFrame(rows)
    eval_df = df[df["clean_target_correct"].astype(int) == 1].copy()
    summary = eval_df.groupby(["target_model", "method", "layer", "k"]).agg(
        n=("success", "size"),
        asr=("success", "mean"),
        mean_queries=("queries", "mean"),
        median_queries=("queries", "median"),
        **{f"asr_at_q{q}": (f"success_at_q{q}", "mean") for q in query_curve},
    ).reset_index()
    summary_path = out_dir / "jacobian_pullback_nes_summary.csv"
    summary.to_csv(summary_path, index=False)
    (out_dir / "metadata.json").write_text(json.dumps({
        "args": vars(args),
        "indices": indices,
        "outputs": [str(csv_path), str(summary_path)],
        "elapsed_sec": time.time() - started,
        "note": "Target ASR is untargeted pred != true label. Pullback uses source ResNet18 feature Jacobian only.",
    }, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
