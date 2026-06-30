#!/usr/bin/env python3
"""Evaluate Jacobian-pullback success-subspace initializations for Square attack."""

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
from scipy.stats import wilcoxon
from torch import nn
from torchvision import datasets, models, transforms


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


def load_target(name: str, device):
    if name == "densenet121":
        base = models.densenet121(pretrained=True)
    elif name == "vgg16":
        base = models.vgg16_bn(pretrained=True)
    else:
        raise ValueError(name)
    return nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), base).to(device).eval()


def normalize_rows(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def fit_pca_bases(manifest_path: str, max_k: int, seed: int):
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[(manifest["trajectory_features_npz"].notna()) & (manifest["success"].astype(int) == 1)]
    bases = {}
    for layer in ["clf_logits", "clf_avgpool", "clf_layer4"]:
        segs = []
        for _idx, row in manifest.iterrows():
            z = np.load(row["trajectory_features_npz"])
            feats = z[layer].astype(np.float64)
            for t in range(len(feats) - 1):
                v = feats[t + 1] - feats[t]
                if np.linalg.norm(v) > 1e-12:
                    segs.append(v)
        x = normalize_rows(np.stack(segs))
        rng = np.random.default_rng(seed)
        x = x[rng.permutation(len(x))]
        x = x - x.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(x, full_matrices=False)
        bases[layer] = torch.from_numpy(vt[:max_k].astype(np.float32))
    return bases


def make_pullback_dirs(source, clean, layer: str, basis: torch.Tensor):
    clean_req = clean.detach().clone().requires_grad_(True)
    _logits, h = source.forward_with_feature(clean_req, layer)
    dirs = []
    for j in range(basis.shape[0]):
        grad = torch.autograd.grad((h * basis[j:j + 1].to(clean.device)).sum(), clean_req, retain_graph=True)[0].detach()
        grad = grad / grad.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        dirs.append(grad)
    return torch.cat(dirs, dim=0)


def project_delta(delta, eps):
    return delta.clamp(-eps, eps)


def to_eps_boundary(delta, eps):
    maxabs = delta.abs().max().clamp_min(1e-12)
    return (delta / maxabs * eps).clamp(-eps, eps)


def loss_value(model, x, y):
    with torch.no_grad():
        logits = model(x)
        return float(F.cross_entropy(logits, y).item()), int(logits.argmax(1).item())


def square_attack_from_delta(model, clean, y, init_delta, args, gen, milestones):
    delta = project_delta(init_delta.clone(), args.eps)
    clean_pred = int(model(clean).argmax(1).item())
    if clean_pred != int(y.item()):
        return {"clean_correct": 0, "success": 0, "queries": 0, "final_pred": clean_pred, **{f"success_at_q{q}": 0 for q in milestones}}
    best_loss, pred = loss_value(model, (clean + delta).clamp(0, 1), y)
    queries = 1
    first_success_q = queries if pred != int(y.item()) else -1
    curve = {q: int(first_success_q > 0 and first_success_q <= q) for q in milestones}
    c, h, w = delta.shape[1:]
    for q in range(2, args.max_queries + 1):
        side = max(args.min_square, int(round(args.square_frac * h * (1 - (q / max(args.max_queries, 1))) + args.min_square)))
        side = min(side, h, w)
        top = int(torch.randint(0, h - side + 1, (1,), generator=gen, device=clean.device).item())
        left = int(torch.randint(0, w - side + 1, (1,), generator=gen, device=clean.device).item())
        candidate = delta.clone()
        patch = torch.rand((1, c, side, side), generator=gen, device=clean.device) * (2.0 * args.eps) - args.eps
        candidate[:, :, top:top + side, left:left + side] = patch
        candidate = project_delta(candidate, args.eps)
        cand_loss, cand_pred = loss_value(model, (clean + candidate).clamp(0, 1), y)
        queries = q
        if cand_loss >= best_loss:
            delta = candidate
            best_loss = cand_loss
            pred = cand_pred
        if pred != int(y.item()) and first_success_q < 0:
            first_success_q = queries
        for m in milestones:
            if queries >= m and curve[m] == 0:
                curve[m] = int(first_success_q > 0 and first_success_q <= m)
        if first_success_q > 0:
            break
    return {
        "clean_correct": 1,
        "success": int(first_success_q > 0),
        "queries": int(first_success_q if first_success_q > 0 else args.max_queries),
        "final_pred": int(pred),
        **{f"success_at_q{q}": int(first_success_q > 0 and first_success_q <= q) for q in milestones},
    }


def bootstrap_ci(values, seed=0, reps=5000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(reps)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    p.add_argument("--trajectory-manifest", default="analysis_outputs/pure_af_geometry/random_ga_classifier_trajectories_resnet18_c10_s5/manifest.csv")
    p.add_argument("--indices-metadata", default="analysis_outputs/pure_af_geometry/away_from_true_attractor_resnet18_all_available_c10.csv.metadata.json")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_pullback_nes/init_square_a_vs_b")
    p.add_argument("--target-model", default="densenet121")
    p.add_argument("--max-images", type=int, default=50)
    p.add_argument("--layers", default="clf_logits,clf_avgpool")
    p.add_argument("--ks", default="10,50")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--max-queries", type=int, default=100)
    p.add_argument("--milestones", default="1,10,50,100")
    p.add_argument("--square-frac", type=float, default=0.35)
    p.add_argument("--min-square", type=int, default=8)
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
    milestones = [int(x) for x in args.milestones.split(",") if x.strip()]
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    indices = json.loads(Path(args.indices_metadata).read_text())["indices"][: args.max_images]
    source = ResNet18WithFeatures().to(device).eval()
    target = load_target(args.target_model, device)
    bases = fit_pca_bases(args.trajectory_manifest, max(ks), args.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 4242)

    rows = []
    started = time.time()
    for image_i, idx in enumerate(indices, start=1):
        clean_cpu, label_int = dataset[int(idx)]
        clean = clean_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(label_int)], device=device)
        rand_delta = to_eps_boundary(torch.randn(clean.shape, generator=gen, device=device), args.eps)
        res = square_attack_from_delta(target, clean, y, rand_delta, args, gen, milestones)
        rows.append({"dataset_idx": int(idx), "source_class": int(label_int), "method": "random_init", "layer": "", "k": 0, **res})
        for layer in layers:
            pull = make_pullback_dirs(source, clean, layer, bases[layer].to(device))
            for k in ks:
                alpha = torch.randn(k, generator=gen, device=device)
                delta = (alpha.view(-1, 1, 1, 1) * pull[:k]).sum(dim=0, keepdim=True)
                delta = to_eps_boundary(delta, args.eps)
                res = square_attack_from_delta(target, clean, y, delta, args, gen, milestones)
                rows.append({"dataset_idx": int(idx), "source_class": int(label_int), "method": "pullback_init", "layer": layer, "k": k, **res})
        if image_i % 10 == 0:
            print(f"[progress] images={image_i}/{len(indices)} elapsed={time.time() - started:.1f}s", flush=True)

    csv_path = out_dir / "pullback_init_square_results.csv"
    fieldnames = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    df = pd.DataFrame(rows)
    df_eval = df[df["clean_correct"].astype(int) == 1].copy()
    summary = df_eval.groupby(["method", "layer", "k"]).agg(
        n=("success", "size"),
        asr=("success", "mean"),
        mean_queries=("queries", "mean"),
        median_queries=("queries", "median"),
        **{f"asr_at_q{m}": (f"success_at_q{m}", "mean") for m in milestones},
    ).reset_index()
    summary_path = out_dir / "pullback_init_square_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Paired tests B vs A.
    tests = []
    base = df_eval[df_eval["method"] == "random_init"].set_index("dataset_idx")
    success_metrics = ["success"] + [f"success_at_q{m}" for m in milestones if f"success_at_q{m}" in df_eval.columns]
    for (_method, layer, k), group in df_eval[df_eval["method"] == "pullback_init"].groupby(["method", "layer", "k"]):
        common = base.index.intersection(group["dataset_idx"])
        g = group.set_index("dataset_idx").loc[common]
        b = base.loc[g.index]
        for metric in success_metrics:
            diff = g[metric].to_numpy(dtype=float) - b[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(diff, args.seed)
            pval = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            tests.append({"layer": layer, "k": int(k), "metric": metric, "mean_diff": float(diff.mean()), "ci_low": lo, "ci_high": hi, "wilcoxon_p": pval})
        succ_mask = (g["success"].astype(int) == 1) & (b["success"].astype(int) == 1)
        if succ_mask.any():
            diff = g.loc[succ_mask, "queries"].to_numpy(dtype=float) - b.loc[succ_mask, "queries"].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(diff, args.seed)
            pval = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            tests.append({"layer": layer, "k": int(k), "metric": "queries_success_only", "mean_diff": float(diff.mean()), "ci_low": lo, "ci_high": hi, "wilcoxon_p": pval})
    tests_path = out_dir / "pullback_init_square_paired_tests.csv"
    pd.DataFrame(tests).to_csv(tests_path, index=False)
    (out_dir / "metadata.json").write_text(json.dumps({"args": vars(args), "indices": indices, "elapsed_sec": time.time() - started}, indent=2))
    print(summary.to_string(index=False))
    print("\nPAIRED TESTS")
    print(pd.DataFrame(tests).to_string(index=False))


if __name__ == "__main__":
    main()
