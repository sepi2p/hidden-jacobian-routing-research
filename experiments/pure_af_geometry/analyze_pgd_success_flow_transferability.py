#!/usr/bin/env python3
"""PGD success-flow geometry vs transferability across ImageNet models."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


LAYERS = ["avgpool", "layer4"]
KS = [20, 50, 100]


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
            raise ValueError(name)

    def forward(self, x):
        return self.forward_with_feature(x, "avgpool")[0]

    def forward_with_feature(self, x, layer: str):
        z = self.normalize(x)
        if self.name == "resnet18":
            z = self.base.conv1(z); z = self.base.bn1(z); z = self.base.relu(z); z = self.base.maxpool(z)
            z = self.base.layer1(z); z = self.base.layer2(z); z = self.base.layer3(z); z = self.base.layer4(z)
            pooled = self.base.avgpool(z).flatten(1)
            logits = self.base.fc(pooled)
            h = pooled
        elif self.name == "densenet121":
            z = self.base.features(z)
            z = F.relu(z, inplace=False)
            h = F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
            logits = self.base.classifier(h)
        else:
            z = self.base.features(z)
            # Hidden geometry uses pooled conv features for both requested hidden layers.
            h = F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
            logits = self.base.classifier(torch.flatten(self.base.avgpool(z), 1))
        if layer in {"avgpool", "layer4"}:
            return logits, h
        raise ValueError(layer)


def normalize_rows(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def project_linf(x, clean, eps):
    return torch.max(torch.min(x, clean + eps), clean - eps).clamp(0, 1)


def true_margin(logits, y):
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.view(-1, 1), -1e9)
    return true - masked.max(1).values


def select_common_correct(dataset, models_map, device, candidate_indices, max_images, batch_size):
    out = []
    loader = DataLoader(Subset(dataset, candidate_indices), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    offset = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            ok = torch.ones(len(x), dtype=torch.bool, device=device)
            for model in models_map.values():
                ok &= model(x).argmax(1).eq(y)
            for j, good in enumerate(ok.detach().cpu().tolist()):
                if good:
                    out.append(int(candidate_indices[offset + j]))
                if len(out) >= max_images:
                    return out
            offset += len(x)
    return out


def collect_pgd_flows(dataset, indices, model_name, model, device, args):
    rows = []
    feats = {layer: {} for layer in LAYERS}
    mu = {layer: {} for layer in LAYERS}
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    offset = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        adv = x.clone()
        saved = {layer: [] for layer in LAYERS}
        for step in range(args.attack_steps + 1):
            with torch.no_grad():
                for layer in LAYERS:
                    _logits, h = model.forward_with_feature(adv, layer)
                    saved[layer].append(h.detach().cpu().numpy())
            if step == args.attack_steps:
                break
            adv.requires_grad_(True)
            loss = F.cross_entropy(model(adv), y)
            grad = torch.autograd.grad(loss, adv)[0]
            adv = project_linf(adv.detach() + args.step_size * grad.sign(), x, args.eps)
        with torch.no_grad():
            final_logits = model(adv)
            success = final_logits.argmax(1).ne(y)
        for j in range(len(x)):
            idx = int(indices[offset + j])
            rows.append({"model": model_name, "dataset_idx": idx, "label": int(y[j].item()), "source_success": int(success[j].item())})
            if not bool(success[j].item()):
                continue
            for layer in LAYERS:
                arr = np.stack([s[j] for s in saved[layer]], axis=0)
                mu[layer].setdefault(int(y[j].item()), []).append(arr[-1] - arr[0])
                for t in range(len(arr) - 1):
                    v = arr[t + 1] - arr[t]
                    if np.linalg.norm(v) <= 1e-12:
                        continue
                    feats[layer][(idx, t)] = v.astype(np.float64)
        offset += len(x)
    mu_out = {}
    for layer in LAYERS:
        mu_out[layer] = {}
        for cls, vals in mu[layer].items():
            vec = normalize_rows(np.stack(vals)).mean(axis=0)
            mu_out[layer][cls] = vec / np.clip(np.linalg.norm(vec), 1e-12, None)
    return pd.DataFrame(rows), feats, mu_out


def basis_from_segments(seg_dict, ids):
    x = normalize_rows(np.stack([seg_dict[i] for i in ids]))
    xc = x - x.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    var = s ** 2
    csum = np.cumsum(var / np.clip(var.sum(), 1e-12, None))
    return u, {
        "n": int(len(x)),
        "d": int(x.shape[1]),
        "dim80": int(np.searchsorted(csum, 0.8) + 1),
        "dim90": int(np.searchsorted(csum, 0.9) + 1),
        "effective_rank": float(np.exp(-np.sum((var / np.clip(var.sum(), 1e-12, None)) * np.log(np.clip(var / np.clip(var.sum(), 1e-12, None), 1e-12, None))))),
    }


def subspace_metrics(u_a, u_b, k):
    kk = min(k, u_a.shape[1], u_b.shape[1])
    s = np.linalg.svd(u_a[:, :kk].T @ u_b[:, :kk], compute_uv=False)
    s = np.clip(s, 0, 1)
    angles = np.arccos(s)
    return {
        "k": int(k),
        "mean_principal_angle_deg": float(np.degrees(angles).mean()),
        "max_principal_angle_deg": float(np.degrees(angles).max()),
        "projection_overlap": float(np.sum(s ** 2) / kk),
        "grassmann_distance": float(np.linalg.norm(angles)),
        "subspace_affinity": float(np.sqrt(np.sum(s ** 2) / kk)),
    }


def attack_away(model, layer, mu_by_class, clean, y, args):
    adv = clean.clone()
    with torch.no_grad():
        _logits, h0 = model.forward_with_feature(clean, layer)
    mu = []
    for label in y.detach().cpu().tolist():
        if int(label) not in mu_by_class[layer]:
            mu.append(np.zeros(h0.shape[1], dtype=np.float64))
        else:
            mu.append(mu_by_class[layer][int(label)])
    mu = torch.from_numpy(np.stack(mu)).float().to(clean.device)
    for _ in range(args.attack_steps):
        adv.requires_grad_(True)
        _logits, h = model.forward_with_feature(adv, layer)
        loss = -(F.normalize(h - h0, dim=1) * mu).sum(1).mean()
        grad = torch.autograd.grad(loss, adv)[0]
        adv = project_linf(adv.detach() + args.step_size * grad.sign(), clean, args.eps)
    return adv.detach()


def transfer_eval(dataset, indices, models_map, mu_map, device, args):
    rows = []
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    offset = 0
    for source_name, source in models_map.items():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            advs = {}
            adv = x.clone()
            for _ in range(args.attack_steps):
                adv.requires_grad_(True)
                loss = F.cross_entropy(source(adv), y)
                grad = torch.autograd.grad(loss, adv)[0]
                adv = project_linf(adv.detach() + args.step_size * grad.sign(), x, args.eps)
            advs["ce_pgd"] = adv
            for layer in LAYERS:
                advs[f"away_flow_{layer}"] = attack_away(source, layer, mu_map[source_name], x, y, args)
            with torch.no_grad():
                for target_name, target in models_map.items():
                    if target_name == source_name:
                        continue
                    for attack, adv_x in advs.items():
                        logits = target(adv_x)
                        pred = logits.argmax(1)
                        margin = true_margin(logits, y)
                        for j in range(len(x)):
                            rows.append({
                                "dataset_idx": int(indices[offset + j]),
                                "source_model": source_name,
                                "target_model": target_name,
                                "attack": attack,
                                "target_transfer_success": int(pred[j].item() != int(y[j].item())),
                                "target_margin": float(margin[j].item()),
                            })
            offset += len(x)
        offset = 0
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    p.add_argument("--indices-metadata", default="analysis_outputs/pure_af_geometry/away_from_true_attractor_resnet18_all_available_c10.csv.metadata.json")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/pgd_success_flow_transferability")
    p.add_argument("--models", default="resnet18,densenet121,vgg16")
    p.add_argument("--max-images", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--attack-steps", type=int, default=20)
    p.add_argument("--step-size", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = [x.strip() for x in args.models.split(",") if x.strip()]
    models_map = {n: FeatureModel(n).to(device).eval() for n in names}
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    candidates = json.loads(Path(args.indices_metadata).read_text())["indices"]
    indices = select_common_correct(dataset, models_map, device, candidates, args.max_images, args.batch_size)

    flow_rows = []
    segs = {}
    mu_map = {}
    for name, model in models_map.items():
        fr, fs, mu = collect_pgd_flows(dataset, indices, name, model, device, args)
        flow_rows.append(fr)
        segs[name] = fs
        mu_map[name] = mu

    dim_rows = []
    geom_rows = []
    for layer in LAYERS:
        for name in names:
            ids = sorted(segs[name][layer])
            u, stats = basis_from_segments(segs[name][layer], ids)
            stats.update({"model": name, "layer": layer})
            dim_rows.append(stats)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = sorted(set(segs[a][layer]) & set(segs[b][layer]))
                if len(common) < 3:
                    continue
                ua, _ = basis_from_segments(segs[a][layer], common)
                ub, _ = basis_from_segments(segs[b][layer], common)
                for k in KS:
                    row = subspace_metrics(ua, ub, k)
                    row.update({"model_a": a, "model_b": b, "layer": layer, "variant": "raw", "n_common_segments": len(common)})
                    geom_rows.append(row)
    transfer = transfer_eval(dataset, indices, models_map, mu_map, device, args)
    transfer_matrix = transfer.groupby(["source_model", "target_model", "attack"]).agg(
        n=("target_transfer_success", "size"),
        transfer_asr=("target_transfer_success", "mean"),
        mean_target_margin=("target_margin", "mean"),
    ).reset_index()
    geom = pd.DataFrame(geom_rows)
    dim = pd.DataFrame(dim_rows)
    flow = pd.concat(flow_rows, ignore_index=True)

    corr_rows = []
    pair_asr = transfer_matrix.groupby(["attack", "source_model", "target_model"])["transfer_asr"].mean().reset_index()
    for attack in pair_asr.attack.unique():
        for metric in ["projection_overlap", "subspace_affinity", "grassmann_distance", "mean_principal_angle_deg"]:
            rows = []
            for _idx, g in geom[(geom.layer == "layer4") & (geom.k == 50)].iterrows():
                a, b = g.model_a, g.model_b
                vals = pair_asr[(pair_asr.attack == attack) & (
                    ((pair_asr.source_model == a) & (pair_asr.target_model == b)) |
                    ((pair_asr.source_model == b) & (pair_asr.target_model == a))
                )]["transfer_asr"].tolist()
                if vals:
                    rows.append((float(g[metric]), float(np.mean(vals))))
            if len(rows) >= 3:
                x = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
                pr = float(pearsonr(x, y).statistic)
                sr = float(spearmanr(x, y).statistic)
                coef = np.polyfit(x, y, 1); pred = np.polyval(coef, x)
                r2 = float(1 - np.sum((y - pred) ** 2) / np.clip(np.sum((y - y.mean()) ** 2), 1e-12, None))
            else:
                pr = sr = r2 = np.nan
            corr_rows.append({"attack": attack, "metric": metric, "pearson": pr, "spearman": sr, "linear_r2": r2})
    corr = pd.DataFrame(corr_rows)

    paths = {
        "geometry_overlap_metrics": out / "geometry_overlap_metrics.csv",
        "model_similarity_matrix": out / "model_similarity_matrix.csv",
        "transferability_matrix": out / "transferability_matrix.csv",
        "geometry_vs_transfer_correlation": out / "geometry_vs_transfer_correlation.csv",
        "success_flow_dimensionality": out / "success_flow_dimensionality.csv",
        "flow_success_rows": out / "flow_success_rows.csv",
    }
    geom.to_csv(paths["geometry_overlap_metrics"], index=False)
    geom[(geom.layer == "layer4") & (geom.k == 50)].to_csv(paths["model_similarity_matrix"], index=False)
    transfer_matrix.to_csv(paths["transferability_matrix"], index=False)
    corr.to_csv(paths["geometry_vs_transfer_correlation"], index=False)
    dim.to_csv(paths["success_flow_dimensionality"], index=False)
    flow.to_csv(paths["flow_success_rows"], index=False)
    (out / "metadata.json").write_text(json.dumps({"args": vars(args), "indices": indices, "outputs": {k: str(v) for k, v in paths.items()}, "note": "Hidden-space cross-model principal angles are computed in paired trajectory-index row space, avoiding direct coordinate identification across architectures."}, indent=2))
    print("GEOMETRY")
    print(geom.to_string(index=False))
    print("\nTRANSFER")
    print(transfer_matrix.to_string(index=False))
    print("\nCORRELATION")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
