#!/usr/bin/env python3
"""Geometry-guided black-box evolutionary attack with Jacobian-pullback mutations."""

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


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def softmax_np(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64) - float(np.max(x))
    e = np.exp(z)
    return e / float(e.sum())


def logp_field(layer: str, logits: np.ndarray, target: int, fc_weight: np.ndarray) -> np.ndarray:
    probs = softmax_np(logits)
    g_logits = -probs
    g_logits[int(target)] += 1.0
    if layer == "clf_logits":
        return g_logits
    if layer in {"clf_avgpool", "clf_layer4"}:
        return np.matmul(g_logits, fc_weight)
    raise ValueError(layer)


def fit_success_bases(manifest_path: str, layers: list[str], max_k: int, seed: int, fc_weight: np.ndarray):
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[(manifest["trajectory_features_npz"].notna()) & (manifest["success"].astype(int) == 1)]
    rng = np.random.default_rng(seed)
    bases = {"success_raw": {}, "success_orth": {}}
    for layer in layers:
        raw = []
        orth = []
        for _idx, row in manifest.iterrows():
            z = np.load(row["trajectory_features_npz"])
            feats = z[layer].astype(np.float64)
            logits = z["clf_logits"].astype(np.float64)
            target = int(row["target_class"])
            for t in range(len(feats) - 1):
                v = feats[t + 1] - feats[t]
                if np.linalg.norm(v) <= 1e-12:
                    continue
                raw.append(v)
                vn = v / np.linalg.norm(v)
                g = logp_field(layer, logits[t], target, fc_weight)
                g = g / np.clip(np.linalg.norm(g), 1e-12, None)
                r = vn - float(np.dot(vn, g)) * g
                if np.linalg.norm(r) > 1e-12:
                    orth.append(r)
        for kind, vectors in [("success_raw", raw), ("success_orth", orth)]:
            x = normalize_rows(np.stack(vectors))
            x = x[rng.permutation(len(x))]
            xc = x - x.mean(axis=0, keepdims=True)
            _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
            bases[kind][layer] = torch.from_numpy(vt[:max_k].astype(np.float32))
    return bases


def random_basis(dim: int, k: int, rng: np.random.Generator):
    x = rng.normal(size=(dim, k))
    q, _r = np.linalg.qr(x)
    return torch.from_numpy(q[:, :k].T.astype(np.float32))


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


def eval_population(model, clean, y, population):
    with torch.no_grad():
        adv = (clean + population).clamp(0, 1)
        logits = model(adv)
        true = logits[:, int(y)]
        masked = logits.clone()
        masked[:, int(y)] = -1e9
        other = masked.max(dim=1).values
        margin = true - other
        pred = logits.argmax(dim=1)
    return margin.detach(), pred.detach()


def bootstrap_ci(values, seed=0, reps=5000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(reps)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mutate_pixel(parent, args, gen):
    noise = torch.randn(parent.shape, generator=gen, device=parent.device)
    return parent + args.pixel_sigma * noise.sign()


def mutate_geometry(parent, pullback, args, gen):
    k = pullback.shape[0]
    alpha = torch.randn(k, generator=gen, device=parent.device) * args.geom_sigma
    step = (alpha.view(-1, 1, 1, 1) * pullback).sum(dim=0, keepdim=True)
    # Keep geometry and pixel mutation magnitudes in the same rough Linf scale.
    maxabs = step.abs().max().clamp_min(1e-12)
    step = step / maxabs * args.pixel_sigma
    return parent + step


def evolutionary_attack(model, clean, y_int: int, args, gen, milestones, pullback=None, p_geometry=0.0):
    with torch.no_grad():
        clean_pred = int(model(clean).argmax(1).item())
    if clean_pred != y_int:
        return {"clean_correct": 0, "success": 0, "queries": 0, "first_success_query": 0, "final_pred": clean_pred, "best_margin": np.nan, **{f"success_at_q{m}": 0 for m in milestones}, **{f"best_margin_at_q{m}": np.nan for m in milestones}}

    population = (torch.rand((args.population, *clean.shape[1:]), generator=gen, device=clean.device) * 2.0 - 1.0) * args.init_scale
    population[0:1].zero_()
    population = project_delta(population, args.eps)
    margins, preds = eval_population(model, clean, y_int, population)
    queries = args.population
    best_idx = int(torch.argmin(margins).item())
    best_margin = float(margins[best_idx].item())
    best_pred = int(preds[best_idx].item())
    first_success = queries if bool((preds != y_int).any().item()) else -1
    curve_success = {m: int(first_success > 0 and first_success <= m) for m in milestones}
    curve_margin = {m: (best_margin if queries >= m else np.nan) for m in milestones}

    elite_count = max(1, int(round(args.population * args.elite_frac)))
    for _gen_i in range(args.generations):
        if first_success > 0 or queries >= args.max_queries:
            break
        order = torch.argsort(margins)
        elites = population[order[:elite_count]].clone()
        children = [elites[i : i + 1] for i in range(elite_count)]
        while len(children) < args.population:
            parent = elites[int(torch.randint(0, elite_count, (1,), generator=gen, device=clean.device).item()) :][:1]
            use_geom = pullback is not None and float(torch.rand((), generator=gen, device=clean.device).item()) < p_geometry
            child = mutate_geometry(parent, pullback, args, gen) if use_geom else mutate_pixel(parent, args, gen)
            child = project_delta(child, args.eps)
            children.append(child)
        population = torch.cat(children[: args.population], dim=0)
        margins, preds = eval_population(model, clean, y_int, population)
        queries += args.population
        idx = int(torch.argmin(margins).item())
        if float(margins[idx].item()) < best_margin:
            best_margin = float(margins[idx].item())
            best_pred = int(preds[idx].item())
        if bool((preds != y_int).any().item()):
            first_success = queries
            best_pred = int(preds[int((preds != y_int).nonzero()[0].item())].item())
        for m in milestones:
            if queries >= m:
                curve_success[m] = int(first_success > 0 and first_success <= m)
                if np.isnan(curve_margin[m]):
                    curve_margin[m] = best_margin

    return {
        "clean_correct": 1,
        "success": int(first_success > 0),
        "queries": int(first_success if first_success > 0 else min(queries, args.max_queries)),
        "first_success_query": int(first_success if first_success > 0 else -1),
        "final_pred": int(best_pred),
        "best_margin": float(best_margin),
        **{f"success_at_q{m}": int(first_success > 0 and first_success <= m) for m in milestones},
        **{f"best_margin_at_q{m}": float(curve_margin[m] if not np.isnan(curve_margin[m]) else best_margin) for m in milestones},
    }


def summarize_and_test(df, milestones, seed):
    df_eval = df[df["clean_correct"].astype(int) == 1].copy()
    summary = df_eval.groupby(["target_model", "method", "basis_kind", "layer", "k", "p_geometry"]).agg(
        n=("success", "size"),
        asr=("success", "mean"),
        mean_queries=("queries", "mean"),
        median_queries=("queries", "median"),
        mean_best_margin=("best_margin", "mean"),
        **{f"asr_at_q{m}": (f"success_at_q{m}", "mean") for m in milestones},
        **{f"mean_best_margin_at_q{m}": (f"best_margin_at_q{m}", "mean") for m in milestones},
    ).reset_index()

    tests = []
    metric_cols = ["success"] + [f"success_at_q{m}" for m in milestones] + ["best_margin"]
    for target, target_df in df_eval.groupby("target_model"):
        random_rows = target_df[target_df["basis_kind"] == "random_pullback"]
        success_rows = target_df[target_df["basis_kind"].isin(["success_raw", "success_orth"])]
        for (_method, basis_kind, layer, k, p_geom), group in success_rows.groupby(["method", "basis_kind", "layer", "k", "p_geometry"]):
            baseline = random_rows[
                (random_rows["layer"] == layer)
                & (random_rows["k"].astype(int) == int(k))
                & (np.isclose(random_rows["p_geometry"].astype(float), float(p_geom)))
            ].set_index("dataset_idx")
            if baseline.empty:
                continue
            g = group.set_index("dataset_idx")
            common = baseline.index.intersection(g.index)
            g = g.loc[common]
            b = baseline.loc[common]
            for metric in metric_cols:
                sign = -1.0 if metric == "best_margin" else 1.0
                diff = sign * (g[metric].to_numpy(dtype=float) - b[metric].to_numpy(dtype=float))
                lo, hi = bootstrap_ci(diff, seed)
                pval = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                tests.append({
                    "target_model": target,
                    "comparison": f"{basis_kind}_vs_random_pullback",
                    "basis_kind": basis_kind,
                    "layer": layer,
                    "k": int(k),
                    "p_geometry": float(p_geom),
                    "metric": metric,
                    "mean_improvement": float(diff.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "wilcoxon_p": pval,
                })
        # D vs C: orth vs raw.
        raw_rows = target_df[target_df["basis_kind"] == "success_raw"]
        orth_rows = target_df[target_df["basis_kind"] == "success_orth"]
        for (_method, _basis, layer, k, p_geom), group in orth_rows.groupby(["method", "basis_kind", "layer", "k", "p_geometry"]):
            baseline = raw_rows[
                (raw_rows["layer"] == layer)
                & (raw_rows["k"].astype(int) == int(k))
                & (np.isclose(raw_rows["p_geometry"].astype(float), float(p_geom)))
            ].set_index("dataset_idx")
            if baseline.empty:
                continue
            g = group.set_index("dataset_idx")
            common = baseline.index.intersection(g.index)
            g = g.loc[common]
            b = baseline.loc[common]
            for metric in metric_cols:
                sign = -1.0 if metric == "best_margin" else 1.0
                diff = sign * (g[metric].to_numpy(dtype=float) - b[metric].to_numpy(dtype=float))
                lo, hi = bootstrap_ci(diff, seed)
                pval = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                tests.append({
                    "target_model": target,
                    "comparison": "success_orth_vs_success_raw",
                    "basis_kind": "success_orth",
                    "layer": layer,
                    "k": int(k),
                    "p_geometry": float(p_geom),
                    "metric": metric,
                    "mean_improvement": float(diff.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "wilcoxon_p": pval,
                })
    return summary, pd.DataFrame(tests)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    p.add_argument("--trajectory-manifest", default="analysis_outputs/pure_af_geometry/random_ga_classifier_trajectories_resnet18_c10_s5/manifest.csv")
    p.add_argument("--indices-metadata", default="analysis_outputs/pure_af_geometry/away_from_true_attractor_resnet18_all_available_c10.csv.metadata.json")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/geometry_guided_ga")
    p.add_argument("--target-models", default="densenet121,vgg16")
    p.add_argument("--max-images", type=int, default=10)
    p.add_argument("--clean-correct-images", type=int, default=0)
    p.add_argument("--layers", default="clf_avgpool,clf_layer4")
    p.add_argument("--ks", default="10,20,50")
    p.add_argument("--p-geometries", default="0.25,0.50,0.75,1.00")
    p.add_argument("--basis-kinds", default="random_pullback,success_raw,success_orth")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--population", type=int, default=12)
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--max-queries", type=int, default=500)
    p.add_argument("--elite-frac", type=float, default=0.25)
    p.add_argument("--init-scale", type=float, default=0.01)
    p.add_argument("--pixel-sigma", type=float, default=0.01)
    p.add_argument("--geom-sigma", type=float, default=1.0)
    p.add_argument("--milestones", default="50,100,250,500")
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
    p_geometries = [float(x) for x in args.p_geometries.split(",") if x.strip()]
    basis_kinds = [x.strip() for x in args.basis_kinds.split(",") if x.strip()]
    target_names = [x.strip() for x in args.target_models.split(",") if x.strip()]
    milestones = [int(x) for x in args.milestones.split(",") if x.strip()]

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    indices = json.loads(Path(args.indices_metadata).read_text())["indices"][: args.max_images]
    source = ResNet18WithFeatures().to(device).eval()
    fc_weight = source.base.fc.weight.detach().cpu().numpy().astype(np.float64)
    success_bases = fit_success_bases(args.trajectory_manifest, layers, max(ks), args.seed, fc_weight)
    rng_np = np.random.default_rng(args.seed + 31337)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 424242)

    rows = []
    started = time.time()
    for target_name in target_names:
        target = load_target(target_name, device)
        target_indices = indices
        if args.clean_correct_images > 0:
            target_indices = []
            for idx in indices:
                clean_cpu, label_int = dataset[int(idx)]
                clean = clean_cpu.unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = int(target(clean).argmax(1).item())
                if pred == int(label_int):
                    target_indices.append(int(idx))
                if len(target_indices) >= args.clean_correct_images:
                    break
            if len(target_indices) < args.clean_correct_images:
                print(f"[warn] {target_name}: requested {args.clean_correct_images} clean-correct images, found {len(target_indices)} in max-images pool", flush=True)
        for image_i, idx in enumerate(target_indices, start=1):
            clean_cpu, label_int = dataset[int(idx)]
            clean = clean_cpu.unsqueeze(0).to(device)
            y_int = int(label_int)
            # Pixel baseline once per target/image.
            res = evolutionary_attack(target, clean, y_int, args, gen, milestones, pullback=None, p_geometry=0.0)
            rows.append({
                "target_model": target_name,
                "dataset_idx": int(idx),
                "source_class": y_int,
                "method": "pixel_ga",
                "basis_kind": "pixel",
                "layer": "",
                "k": 0,
                "p_geometry": 0.0,
                **res,
            })
            for layer in layers:
                with torch.no_grad():
                    _logits, h = source.forward_with_feature(clean, layer)
                dim = int(h.shape[1])
                basis_cache = {}
                for k in ks:
                    basis_cache[("random_pullback", k)] = random_basis(dim, k, rng_np).to(device)
                    for kind in ["success_raw", "success_orth"]:
                        basis_cache[(kind, k)] = success_bases[kind][layer][:k].to(device)
                pullback_cache = {}
                for kind in basis_kinds:
                    for k in ks:
                        pullback_cache[(kind, k)] = make_pullback_dirs(source, clean, layer, basis_cache[(kind, k)])
                for kind in basis_kinds:
                    for k in ks:
                        pullback = pullback_cache[(kind, k)]
                        for p_geom in p_geometries:
                            res = evolutionary_attack(target, clean, y_int, args, gen, milestones, pullback=pullback, p_geometry=p_geom)
                            rows.append({
                                "target_model": target_name,
                                "dataset_idx": int(idx),
                                "source_class": y_int,
                                "method": "geometry_ga",
                                "basis_kind": kind,
                                "layer": layer,
                                "k": int(k),
                                "p_geometry": float(p_geom),
                                **res,
                            })
            if image_i % 2 == 0:
                print(f"[{target_name}] images={image_i}/{len(target_indices)} elapsed={time.time() - started:.1f}s", flush=True)
        del target
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = out_dir / "geometry_guided_ga_results.csv"
    fieldnames = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    df = pd.DataFrame(rows)
    summary, tests = summarize_and_test(df, milestones, args.seed)
    summary_path = out_dir / "geometry_guided_ga_summary.csv"
    tests_path = out_dir / "geometry_guided_ga_paired_tests.csv"
    summary.to_csv(summary_path, index=False)
    tests.to_csv(tests_path, index=False)
    (out_dir / "metadata.json").write_text(json.dumps({
        "args": vars(args),
        "indices": indices,
        "elapsed_sec": time.time() - started,
        "selection": "target-model black-box margin only; source ResNet18 used only to build fixed pullback mutation directions",
    }, indent=2))
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nPAIRED TESTS")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
