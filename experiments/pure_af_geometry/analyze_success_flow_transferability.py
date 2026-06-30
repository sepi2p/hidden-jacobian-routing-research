#!/usr/bin/env python3
"""Compare cross-model success-flow geometry with adversarial transferability."""

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
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


LAYERS = ["clf_avgpool", "clf_layer4"]
KS = [20, 50, 100]


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class FeatureModel(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if name == "resnet18":
            self.base = models.resnet18(pretrained=True)
        elif name == "densenet121":
            self.base = models.densenet121(pretrained=True)
        elif name == "vgg16":
            self.base = models.vgg16_bn(pretrained=True)
        else:
            raise ValueError(name)

    def forward(self, x):
        return self.forward_with_feature(x, "clf_logits")[0]

    def forward_with_feature(self, x, layer: str):
        z = self.normalize(x)
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
        elif self.name == "densenet121":
            z = self.base.features.conv0(z)
            z = self.base.features.norm0(z)
            z = self.base.features.relu0(z)
            z = self.base.features.pool0(z)
            z = self.base.features.denseblock1(z)
            z = self.base.features.transition1(z)
            z = self.base.features.denseblock2(z)
            z = self.base.features.transition2(z)
            z = self.base.features.denseblock3(z)
            z = self.base.features.transition3(z)
            z = self.base.features.denseblock4(z)
            z = self.base.features.norm5(z)
            z = F.relu(z, inplace=False)
            pooled = F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
            logits = self.base.classifier(pooled)
        else:
            z = self.base.features(z)
            pooled_map = self.base.avgpool(z)
            flat = torch.flatten(pooled_map, 1)
            logits = self.base.classifier(flat)
            pooled = F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
        if layer in {"clf_avgpool", "clf_layer4"}:
            return logits, pooled
        if layer == "clf_logits":
            return logits, logits
        raise ValueError(layer)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def softmax_np(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64) - float(np.max(x))
    e = np.exp(z)
    return e / float(e.sum())


def logp_field_from_logits(layer: str, logits: np.ndarray, target: int, fc_weight: np.ndarray) -> np.ndarray:
    probs = softmax_np(logits)
    g = -probs
    g[int(target)] += 1.0
    if layer in {"clf_avgpool", "clf_layer4"}:
        return np.matmul(g, fc_weight)
    raise ValueError(layer)


def classifier_weight(model_name: str) -> np.ndarray:
    if model_name == "resnet18":
        return models.resnet18(pretrained=True).eval().fc.weight.detach().cpu().numpy().astype(np.float64)
    if model_name == "densenet121":
        return models.densenet121(pretrained=True).eval().classifier.weight.detach().cpu().numpy().astype(np.float64)
    if model_name == "vgg16":
        # VGG pooled conv layer is 512-d, but classifier is not linear from this pooled representation.
        # Use identity-like fallback: orthogonalization is skipped for VGG hidden layers unless dims match.
        return np.empty((1000, 0), dtype=np.float64)
    raise ValueError(model_name)


def pca_stats(x: np.ndarray) -> dict[str, float]:
    xc = x - x.mean(axis=0, keepdims=True)
    _u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    var = s ** 2
    ratios = var / float(var.sum()) if float(var.sum()) > 0 else np.zeros_like(var)
    csum = np.cumsum(ratios)
    entropy = -float(np.sum(ratios[ratios > 0] * np.log(ratios[ratios > 0])))
    return {
        "pc1_var": float(ratios[0]),
        "dim80": int(np.searchsorted(csum, 0.8) + 1),
        "dim90": int(np.searchsorted(csum, 0.9) + 1),
        "effective_rank": float(np.exp(entropy)),
    }


def collect_segments(manifest_path: str, model_name: str, layer: str, orthogonalize: bool):
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[(manifest["trajectory_features_npz"].notna()) & (manifest["success"].astype(int) == 1)]
    fc_weight = classifier_weight(model_name)
    segs = []
    mus = {}
    for _idx, row in manifest.iterrows():
        z = np.load(row["trajectory_features_npz"])
        feats = z[layer].astype(np.float64)
        logits = z["clf_logits"].astype(np.float64)
        target = int(row["target_class"])
        start = feats[0]
        final = feats[-1]
        mus.setdefault(target, []).append(final - start)
        for t in range(len(feats) - 1):
            v = feats[t + 1] - feats[t]
            if np.linalg.norm(v) <= 1e-12:
                continue
            vn = v / np.linalg.norm(v)
            if orthogonalize and fc_weight.shape[1] == vn.shape[0]:
                g = logp_field_from_logits(layer, logits[t], target, fc_weight)
                g = g / np.clip(np.linalg.norm(g), 1e-12, None)
                vn = vn - float(np.dot(vn, g)) * g
                if np.linalg.norm(vn) <= 1e-12:
                    continue
            segs.append(vn)
    mu_out = {cls: normalize_rows(np.stack(vals)).mean(axis=0) for cls, vals in mus.items()}
    mu_out = {cls: vec / np.clip(np.linalg.norm(vec), 1e-12, None) for cls, vec in mu_out.items()}
    return normalize_rows(np.stack(segs)), mu_out


def pca_basis(x: np.ndarray, max_k: int):
    xc = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return vt[:max_k]


def fit_common_pca(segment_sets: list[np.ndarray], dims: int, seed: int):
    min_n = min(len(x) for x in segment_sets)
    rng = np.random.default_rng(seed)
    sampled = []
    for x in segment_sets:
        idx = rng.choice(len(x), size=min_n, replace=False)
        sampled.append(x[idx])
    all_x = np.concatenate(sampled, axis=0)
    n_components = min(dims, all_x.shape[0] - 1, all_x.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(all_x)
    return pca


def subspace_metrics(a: np.ndarray, b: np.ndarray, k: int):
    qa = a[:k].T
    qb = b[:k].T
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    angles = np.arccos(s)
    affinity = float(np.sqrt(np.sum(s ** 2) / k))
    return {
        "k": int(k),
        "mean_principal_angle_deg": float(np.degrees(angles).mean()),
        "max_principal_angle_deg": float(np.degrees(angles).max()),
        "projection_overlap": float(np.sum(s ** 2) / k),
        "grassmann_distance": float(np.linalg.norm(angles)),
        "subspace_affinity": affinity,
    }


def select_clean_correct_indices(dataset, model, device, candidate_indices, max_images, batch_size):
    out = []
    loader = DataLoader(Subset(dataset, candidate_indices), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    offset = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            ok = (pred == y).detach().cpu().tolist()
            for j, good in enumerate(ok):
                if good:
                    out.append(int(candidate_indices[offset + j]))
                if len(out) >= max_images:
                    return out
            offset += len(ok)
    return out


def project_linf(x, clean, eps):
    return torch.max(torch.min(x, clean + eps), clean - eps).clamp(0, 1)


def true_margin(logits, labels):
    true = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), -1e9)
    return true - masked.max(1).values


def attack_ce(source, clean, labels, eps, steps, step_size):
    adv = clean.clone()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(source(adv), labels)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = project_linf(adv.detach() + step_size * grad.sign(), clean, eps)
    return adv.detach()


def attack_away(source: FeatureModel, layer: str, mu_by_class: dict[int, np.ndarray], clean, labels, eps, steps, step_size):
    adv = clean.clone()
    with torch.no_grad():
        _logits, h0 = source.forward_with_feature(clean, layer)
    mu = torch.stack([
        torch.from_numpy(mu_by_class[int(y.item())]).float().to(clean.device) for y in labels
    ], dim=0)
    for _ in range(steps):
        adv.requires_grad_(True)
        _logits, h = source.forward_with_feature(adv, layer)
        cos_plus = (F.normalize(h - h0, dim=1) * mu).sum(1).mean()
        grad = torch.autograd.grad(-cos_plus, adv)[0]
        adv = project_linf(adv.detach() + step_size * grad.sign(), clean, eps)
    return adv.detach()


def evaluate_transfer(dataset, source_name, target_name, source, target, indices, mu_by_class, args, device):
    rows = []
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    offset = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        attacks = {
            "ce_pgd": attack_ce(source, x, y, args.eps, args.attack_steps, args.step_size),
            "away_flow_layer4": attack_away(source, "clf_layer4", mu_by_class["clf_layer4"], x, y, args.eps, args.attack_steps, args.step_size),
            "away_flow_avgpool": attack_away(source, "clf_avgpool", mu_by_class["clf_avgpool"], x, y, args.eps, args.attack_steps, args.step_size),
        }
        with torch.no_grad():
            clean_source_pred = source(x).argmax(1)
            clean_target_pred = target(x).argmax(1)
            for attack_name, adv in attacks.items():
                src_logits = source(adv)
                tgt_logits = target(adv)
                src_pred = src_logits.argmax(1)
                tgt_pred = tgt_logits.argmax(1)
                src_margin = true_margin(src_logits, y)
                tgt_margin = true_margin(tgt_logits, y)
                for j in range(len(x)):
                    rows.append({
                        "dataset_idx": int(indices[offset + j]),
                        "source_model": source_name,
                        "target_model": target_name,
                        "attack": attack_name,
                        "label": int(y[j].item()),
                        "clean_source_correct": int(clean_source_pred[j].item() == int(y[j].item())),
                        "clean_target_correct": int(clean_target_pred[j].item() == int(y[j].item())),
                        "source_success": int(src_pred[j].item() != int(y[j].item())),
                        "target_transfer_success": int(tgt_pred[j].item() != int(y[j].item())),
                        "source_margin": float(src_margin[j].item()),
                        "target_margin": float(tgt_margin[j].item()),
                    })
        offset += len(x)
    return rows


def corr_rows(geom: pd.DataFrame, transfer: pd.DataFrame):
    rows = []
    trans_pair = transfer.groupby(["source_model", "target_model", "attack"]).agg(transfer_asr=("target_transfer_success", "mean")).reset_index()
    pair_asr = {}
    for _idx, row in trans_pair.iterrows():
        key = tuple(sorted([row["source_model"], row["target_model"]]))
        pair_asr.setdefault((key[0], key[1], row["attack"]), []).append(float(row["transfer_asr"]))
    pair_mean = {k: float(np.mean(v)) for k, v in pair_asr.items()}
    for attack in sorted(trans_pair["attack"].unique()):
        g = geom.copy()
        g["transfer_asr"] = [
            pair_mean.get((min(a, b), max(a, b), attack), np.nan) for a, b in zip(g.model_a, g.model_b)
        ]
        g = g.dropna(subset=["transfer_asr"])
        for metric in ["projection_overlap", "subspace_affinity", "grassmann_distance", "mean_principal_angle_deg"]:
            if len(g) < 3:
                pear = spear = r2 = np.nan
            else:
                x = g[metric].to_numpy(dtype=float)
                y = g["transfer_asr"].to_numpy(dtype=float)
                pear = float(pearsonr(x, y).statistic)
                spear = float(spearmanr(x, y).statistic)
                coef = np.polyfit(x, y, deg=1)
                pred = np.polyval(coef, x)
                r2 = float(1.0 - np.sum((y - pred) ** 2) / np.clip(np.sum((y - y.mean()) ** 2), 1e-12, None))
            rows.append({"attack": attack, "metric": metric, "pearson": pear, "spearman": spear, "linear_r2": r2})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifests", required=True, help="Comma list model=manifest.csv")
    p.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    p.add_argument("--indices-metadata", default="analysis_outputs/pure_af_geometry/away_from_true_attractor_resnet18_all_available_c10.csv.metadata.json")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/success_flow_transferability")
    p.add_argument("--max-images", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--attack-steps", type=int, default=40)
    p.add_argument("--step-size", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_map = dict(item.split("=", 1) for item in args.manifests.split(",") if item.strip())
    models_order = list(manifest_map)

    segment_data = {}
    dim_rows = []
    for model_name, manifest in manifest_map.items():
        segment_data[model_name] = {"raw": {}, "orth": {}, "mu": {}}
        for layer in LAYERS:
            raw, mu = collect_segments(manifest, model_name, layer, orthogonalize=False)
            orth, _ = collect_segments(manifest, model_name, layer, orthogonalize=True)
            segment_data[model_name]["raw"][layer] = raw
            segment_data[model_name]["orth"][layer] = orth
            segment_data[model_name]["mu"][layer] = mu
            for variant, x in [("raw", raw), ("orth", orth)]:
                stats = pca_stats(x)
                stats.update({"model": model_name, "layer": layer, "variant": variant, "n": int(len(x)), "d_original": int(x.shape[1])})
                dim_rows.append(stats)

    overlap_rows = []
    for layer in LAYERS:
        for variant in ["raw", "orth"]:
            sets = [segment_data[m][variant][layer] for m in models_order]
            pca = fit_common_pca(sets, dims=256, seed=args.seed)
            bases = {}
            for model_name, x in zip(models_order, sets):
                projected = pca.transform(x)
                bases[model_name] = pca_basis(normalize_rows(projected), max(KS))
            for i, a in enumerate(models_order):
                for b in models_order[i + 1:]:
                    for k in KS:
                        row = subspace_metrics(bases[a], bases[b], k)
                        row.update({"model_a": a, "model_b": b, "layer": layer, "variant": variant})
                        overlap_rows.append(row)

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    candidates = json.loads(Path(args.indices_metadata).read_text())["indices"]
    model_objs = {name: FeatureModel(name).to(device).eval() for name in models_order}
    transfer_rows = []
    pair_indices = {}
    for source_name in models_order:
        for target_name in models_order:
            if source_name == target_name:
                continue
            # Filter on both models to make source->target transfer rates comparable.
            source = model_objs[source_name]
            target = model_objs[target_name]
            src_ok = select_clean_correct_indices(dataset, source, device, candidates, args.max_images * 2, args.batch_size)
            both_ok = select_clean_correct_indices(dataset, target, device, src_ok, args.max_images, args.batch_size)
            pair_indices[f"{source_name}->{target_name}"] = both_ok
            mu = segment_data[source_name]["mu"]
            transfer_rows.extend(evaluate_transfer(dataset, source_name, target_name, source, target, both_ok, mu, args, device))

    dim = pd.DataFrame(dim_rows)
    geom = pd.DataFrame(overlap_rows)
    transfer = pd.DataFrame(transfer_rows)
    transfer_matrix = transfer.groupby(["source_model", "target_model", "attack"]).agg(
        n=("target_transfer_success", "size"),
        source_asr=("source_success", "mean"),
        transfer_asr=("target_transfer_success", "mean"),
        mean_target_margin=("target_margin", "mean"),
    ).reset_index()
    corr = corr_rows(geom, transfer)

    paths = {
        "model_similarity_matrix": out_dir / "model_similarity_matrix.csv",
        "geometry_overlap_metrics": out_dir / "geometry_overlap_metrics.csv",
        "dimensionality": out_dir / "success_flow_dimensionality.csv",
        "transferability_matrix": out_dir / "transferability_matrix.csv",
        "transferability_rows": out_dir / "transferability_rows.csv",
        "correlation": out_dir / "geometry_vs_transfer_correlation.csv",
    }
    # For compatibility with requested name, model_similarity_matrix is the k=50 layer4 raw slice.
    geom[(geom["layer"] == "clf_layer4") & (geom["variant"] == "raw") & (geom["k"] == 50)].to_csv(paths["model_similarity_matrix"], index=False)
    geom.to_csv(paths["geometry_overlap_metrics"], index=False)
    dim.to_csv(paths["dimensionality"], index=False)
    transfer_matrix.to_csv(paths["transferability_matrix"], index=False)
    transfer.to_csv(paths["transferability_rows"], index=False)
    corr.to_csv(paths["correlation"], index=False)
    (out_dir / "metadata.json").write_text(json.dumps({
        "args": vars(args),
        "manifests": manifest_map,
        "pair_indices": pair_indices,
        "note": "Cross-model subspace metrics use PCA projection to a shared 256-D coordinate system because hidden dimensions differ across architectures.",
        "outputs": {k: str(v) for k, v in paths.items()},
    }, indent=2))
    print("DIMENSIONALITY")
    print(dim.to_string(index=False))
    print("\nGEOMETRY k=50 layer4 raw")
    print(geom[(geom["layer"] == "clf_layer4") & (geom["variant"] == "raw") & (geom["k"] == 50)].to_string(index=False))
    print("\nTRANSFER")
    print(transfer_matrix.to_string(index=False))
    print("\nCORRELATION")
    print(corr.to_string(index=False))


if __name__ == "__main__":
    main()
