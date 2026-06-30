#!/usr/bin/env python3
"""Compare GA successful-flow geometry against clean class-preserving motion."""

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
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as TF


LAYERS = ["clf_avgpool", "clf_layer4", "clf_logits"]
KS = [5, 10, 20, 50, 100]


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

    def forward_features(self, x):
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
        return {"clf_avgpool": pooled, "clf_layer4": pooled, "clf_logits": logits, "logits": logits}


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


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


def orthogonalize_local(vectors: np.ndarray, grads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = normalize_rows(vectors)
    g = normalize_rows(grads)
    coeff = np.sum(v * g, axis=1, keepdims=True)
    residual = v - coeff * g
    residual_norm = np.linalg.norm(residual, axis=1)
    keep = residual_norm > 1e-12
    return normalize_rows(residual[keep]), np.column_stack([coeff[keep, 0], residual_norm[keep]])


def pca_stats(x: np.ndarray) -> dict[str, float]:
    n, d = x.shape
    if n < 2:
        return {"n": n, "d": d, "pc1_var": np.nan, "pc2_var": np.nan, "pc5_cum_var": np.nan,
                "pc10_cum_var": np.nan, "dim80": np.nan, "dim90": np.nan, "dim95": np.nan,
                "effective_rank": np.nan}
    xc = x - x.mean(axis=0, keepdims=True)
    _u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    var = s ** 2
    ratios = var / float(var.sum()) if float(var.sum()) > 0 else np.zeros_like(var)
    csum = np.cumsum(ratios)
    entropy = -float(np.sum(ratios[ratios > 0] * np.log(ratios[ratios > 0])))
    return {
        "n": int(n),
        "d": int(d),
        "pc1_var": float(ratios[0]) if len(ratios) else np.nan,
        "pc2_var": float(ratios[1]) if len(ratios) > 1 else np.nan,
        "pc5_cum_var": float(csum[min(4, len(csum) - 1)]) if len(csum) else np.nan,
        "pc10_cum_var": float(csum[min(9, len(csum) - 1)]) if len(csum) else np.nan,
        "dim80": int(np.searchsorted(csum, 0.80) + 1) if len(csum) else np.nan,
        "dim90": int(np.searchsorted(csum, 0.90) + 1) if len(csum) else np.nan,
        "dim95": int(np.searchsorted(csum, 0.95) + 1) if len(csum) else np.nan,
        "effective_rank": float(np.exp(entropy)),
    }


def fit_pca_basis(x_train: np.ndarray, max_k: int):
    mean = x_train.mean(axis=0, keepdims=True)
    xc = x_train - mean
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return mean, vt[:max_k]


def projection_energies(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, ks: list[int]) -> pd.DataFrame:
    xc = x - mean
    denom = np.sum(xc * xc, axis=1)
    coeff = xc @ basis.T
    rows = {}
    for k in ks:
        kk = min(k, basis.shape[0])
        rows[f"energy_k{k}"] = np.sum(coeff[:, :kk] ** 2, axis=1) / np.clip(denom, 1e-12, None)
    return pd.DataFrame(rows)


def auroc_safe(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def mwu_safe(pos: np.ndarray, neg: np.ndarray):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
    auc = float(stat / (len(pos) * len(neg)))
    return float(p), float(2 * auc - 1)


def logistic_cv(rows: pd.DataFrame, label_col: str, feature_cols: list[str], seed: int):
    y = rows[label_col].to_numpy(dtype=int)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        return {"cv_accuracy_mean": np.nan, "cv_accuracy_std": np.nan}
    n_splits = min(5, int(min(np.bincount(y))))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, rows[feature_cols].to_numpy(), y, cv=cv, scoring="accuracy")
    return {"cv_accuracy_mean": float(scores.mean()), "cv_accuracy_std": float(scores.std())}


def collect_success(manifest: pd.DataFrame, layer: str, fc_weight: np.ndarray):
    vectors = []
    grads = []
    meta = []
    for _idx, row in manifest.iterrows():
        if int(row.get("success", 0)) != 1:
            continue
        z = np.load(row["trajectory_features_npz"])
        feats = z[layer].astype(np.float64)
        logits = z["clf_logits"].astype(np.float64)
        target = int(row["target_class"])
        for t in range(len(feats) - 1):
            seg = feats[t + 1] - feats[t]
            if np.linalg.norm(seg) <= 1e-12:
                continue
            vectors.append(seg)
            grads.append(logp_field(layer, logits[t], target, fc_weight))
            meta.append({"target_class": target, "run_name": row["run_name"], "segment_index": t})
    return np.stack(vectors), np.stack(grads), meta


def random_crop_tensor(x: torch.Tensor, gen: torch.Generator):
    scale = float((torch.rand((), generator=gen) * (1.0 - 0.72) + 0.72).item())
    ratio = float((torch.rand((), generator=gen) * (1.15 - 0.85) + 0.85).item())
    _, h, w = x.shape
    area = h * w
    crop_h = int(round((area * scale / ratio) ** 0.5))
    crop_w = int(round((area * scale * ratio) ** 0.5))
    crop_h = min(max(crop_h, 16), h)
    crop_w = min(max(crop_w, 16), w)
    top = int(torch.randint(0, h - crop_h + 1, (1,), generator=gen).item())
    left = int(torch.randint(0, w - crop_w + 1, (1,), generator=gen).item())
    return TF.resized_crop(x, top, left, crop_h, crop_w, [h, w], antialias=True)


def color_jitter_tensor(x: torch.Tensor, gen: torch.Generator):
    b = float((torch.rand((), generator=gen) * 0.5 + 0.75).item())
    c = float((torch.rand((), generator=gen) * 0.5 + 0.75).item())
    s = float((torch.rand((), generator=gen) * 0.5 + 0.75).item())
    out = TF.adjust_brightness(x, b)
    out = TF.adjust_contrast(out, c)
    out = TF.adjust_saturation(out, s)
    return out.clamp(0, 1)


def blur_tensor(x: torch.Tensor):
    return TF.gaussian_blur(x, kernel_size=[9, 9], sigma=[1.0, 1.0])


def noise_tensor(x: torch.Tensor, gen: torch.Generator, sigma: float):
    return (x + torch.randn(x.shape, generator=gen) * sigma).clamp(0, 1)


def build_clean_segments(args, source: ResNet18WithFeatures, dataset, indices: list[int], device):
    gen = torch.Generator()
    gen.manual_seed(args.seed + 9090)
    rows = []
    by_class: dict[int, list[tuple[int, torch.Tensor]]] = {}
    start_cache = {}
    variants_by_idx = {}

    source.eval()
    with torch.no_grad():
        for idx in indices:
            x_cpu, label = dataset[int(idx)]
            x = x_cpu.unsqueeze(0).to(device)
            feats = source.forward_features(x)
            pred = int(feats["logits"].argmax(1).item())
            if pred != int(label):
                continue
            start_cache[int(idx)] = {k: v.detach().cpu().numpy()[0].astype(np.float64) for k, v in feats.items()}
            by_class.setdefault(int(label), []).append((int(idx), x_cpu))

    for label, items in by_class.items():
        for local_i, (idx, x_cpu) in enumerate(items):
            variants = []
            for r in range(args.crop_reps):
                variants.append((f"crop{r}", random_crop_tensor(x_cpu, gen)))
            for r in range(args.color_reps):
                variants.append((f"color{r}", color_jitter_tensor(x_cpu, gen)))
            for r in range(args.blur_reps):
                variants.append((f"blur{r}", blur_tensor(x_cpu)))
            for r in range(args.noise_reps):
                variants.append((f"noise{r}", noise_tensor(x_cpu, gen, args.noise_sigma)))
            if len(items) > 1:
                for r, lam in enumerate(args.interp_lambdas):
                    other_idx, other = items[(local_i + r + 1) % len(items)]
                    variants.append((f"interp{r}_with_{other_idx}_lam{lam:.2f}", ((1.0 - lam) * x_cpu + lam * other).clamp(0, 1)))
            variants_by_idx[idx] = variants

    clean_vectors = {layer: [] for layer in LAYERS}
    clean_grads = {layer: [] for layer in LAYERS}
    clean_meta = []
    fc_weight = source.base.fc.weight.detach().cpu().numpy().astype(np.float64)

    with torch.no_grad():
        for idx, variants in variants_by_idx.items():
            label = int(dataset[int(idx)][1])
            start = start_cache[idx]
            for variant_name, x_var_cpu in variants:
                x_var = x_var_cpu.unsqueeze(0).to(device)
                feats = source.forward_features(x_var)
                pred = int(feats["logits"].argmax(1).item())
                if pred != label:
                    continue
                for layer in LAYERS:
                    end = feats[layer].detach().cpu().numpy()[0].astype(np.float64)
                    seg = end - start[layer]
                    if np.linalg.norm(seg) <= 1e-12:
                        continue
                    clean_vectors[layer].append(seg)
                    clean_grads[layer].append(logp_field(layer, start["logits"], label, fc_weight))
                clean_meta.append({"dataset_idx": idx, "label": label, "variant": variant_name})
                rows.append({"dataset_idx": idx, "label": label, "variant": variant_name, "pred": pred})

    return clean_vectors, clean_grads, pd.DataFrame(rows)


def compare_projection(source_name: str, source_vectors: np.ndarray, target_name: str, target_vectors: np.ndarray, layer: str, variant: str, seed: int):
    idx_source = np.arange(len(source_vectors))
    idx_target = np.arange(len(target_vectors))
    source_train, source_test = train_test_split(idx_source, test_size=0.3, random_state=seed)
    _target_train, target_test = train_test_split(idx_target, test_size=0.3, random_state=seed)
    max_k = min(max(KS), len(source_train) - 1, source_vectors.shape[1])
    mean, basis = fit_pca_basis(source_vectors[source_train], max_k)
    pos = projection_energies(source_vectors[source_test], mean, basis, KS)
    neg = projection_energies(target_vectors[target_test], mean, basis, KS)
    metric_rows = []
    score_rows = []
    pos["label"] = 1
    neg["label"] = 0
    pos["group"] = source_name
    neg["group"] = target_name
    for frame in [pos, neg]:
        frame["basis_source"] = source_name
        frame["comparison_target"] = target_name
        frame["layer"] = layer
        frame["variant"] = variant
        score_rows.append(frame)
    for k in KS:
        col = f"energy_k{k}"
        p = pos[col].to_numpy()
        n = neg[col].to_numpy()
        y = np.r_[np.ones(len(p), dtype=int), np.zeros(len(n), dtype=int)]
        score = np.r_[p, n]
        p_value, rank_biserial = mwu_safe(p, n)
        metric_rows.append({
            "layer": layer,
            "variant": variant,
            "basis_source": source_name,
            "positive_group": source_name,
            "negative_group": target_name,
            "k": k,
            "n_positive": len(p),
            "n_negative": len(n),
            "positive_mean_energy": float(np.mean(p)),
            "negative_mean_energy": float(np.mean(n)),
            "mean_difference": float(np.mean(p) - np.mean(n)),
            "auroc": auroc_safe(y, score),
            "mannwhitney_p": p_value,
            "rank_biserial": rank_biserial,
        })
    rows = pd.concat([pos, neg], ignore_index=True)
    lr = logistic_cv(rows, "label", [f"energy_k{k}" for k in KS], seed)
    lr.update({"layer": layer, "variant": variant, "basis_source": source_name, "comparison": f"{source_name}_vs_{target_name}", "n": len(rows)})
    return pd.DataFrame(metric_rows), pd.concat(score_rows, ignore_index=True), lr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--trajectory-manifest", default="analysis_outputs/pure_af_geometry/random_ga_classifier_trajectories_resnet18_c10_s5/manifest.csv")
    parser.add_argument("--indices-metadata", default="analysis_outputs/pure_af_geometry/away_from_true_attractor_resnet18_all_available_c10.csv.metadata.json")
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/clean_motion_vs_success_flow")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--crop-reps", type=int, default=2)
    parser.add_argument("--color-reps", type=int, default=2)
    parser.add_argument("--blur-reps", type=int, default=1)
    parser.add_argument("--noise-reps", type=int, default=2)
    parser.add_argument("--noise-sigma", type=float, default=0.03)
    parser.add_argument("--interp-lambdas", default="0.25,0.50,0.75")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.interp_lambdas = [float(x) for x in str(args.interp_lambdas).split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    indices = json.loads(Path(args.indices_metadata).read_text())["indices"][: args.max_images]
    source = ResNet18WithFeatures().to(device).eval()
    fc_weight = source.base.fc.weight.detach().cpu().numpy().astype(np.float64)

    manifest = pd.read_csv(args.trajectory_manifest)
    manifest = manifest[manifest["trajectory_features_npz"].notna()].copy()
    clean_vectors, clean_grads, clean_rows = build_clean_segments(args, source, dataset, indices, device)
    clean_rows.to_csv(out_dir / "clean_motion_segments_manifest.csv", index=False)

    dim_rows = []
    diag_rows = []
    metric_rows = []
    score_rows = []
    logreg_rows = []

    for layer in LAYERS:
        success_raw, success_grads, _success_meta = collect_success(manifest, layer, fc_weight)
        clean_raw = np.stack(clean_vectors[layer])
        clean_grad = np.stack(clean_grads[layer])
        success_raw_n = normalize_rows(success_raw)
        clean_raw_n = normalize_rows(clean_raw)
        success_orth, success_diag = orthogonalize_local(success_raw, success_grads)
        clean_orth, clean_diag = orthogonalize_local(clean_raw, clean_grad)

        datasets_by_variant = {
            "raw": {"success": success_raw_n, "clean": clean_raw_n},
            "gradient_orthogonalized": {"success": success_orth, "clean": clean_orth},
        }
        for variant, groups in datasets_by_variant.items():
            for name, x in groups.items():
                stats = pca_stats(x)
                stats.update({"layer": layer, "variant": variant, "set": name})
                dim_rows.append(stats)
            for source_name, target_name in [("success", "clean"), ("clean", "success")]:
                metrics, scores, lr = compare_projection(
                    source_name,
                    groups[source_name],
                    target_name,
                    groups[target_name],
                    layer,
                    variant,
                    args.seed,
                )
                metric_rows.append(metrics)
                score_rows.append(scores)
                logreg_rows.append(lr)
        for name, diag in [("success", success_diag), ("clean", clean_diag)]:
            diag_rows.append({
                "layer": layer,
                "set": name,
                "n": int(len(diag)),
                "mean_cos_with_local_grad_before_removal": float(np.mean(diag[:, 0])),
                "median_cos_with_local_grad_before_removal": float(np.median(diag[:, 0])),
                "mean_residual_norm_after_removal": float(np.mean(diag[:, 1])),
                "median_residual_norm_after_removal": float(np.median(diag[:, 1])),
            })

    dim = pd.DataFrame(dim_rows)
    diag = pd.DataFrame(diag_rows)
    metrics = pd.concat(metric_rows, ignore_index=True)
    scores = pd.concat(score_rows, ignore_index=True)
    logreg = pd.DataFrame(logreg_rows)

    paths = {
        "dimensionality": out_dir / "clean_vs_success_dimensionality.csv",
        "diagnostics": out_dir / "clean_vs_success_gradient_diagnostics.csv",
        "projection_metrics": out_dir / "clean_vs_success_projection_metrics.csv",
        "projection_scores": out_dir / "clean_vs_success_projection_scores.csv",
        "logistic": out_dir / "clean_vs_success_logistic_regression.csv",
    }
    dim.to_csv(paths["dimensionality"], index=False)
    diag.to_csv(paths["diagnostics"], index=False)
    metrics.to_csv(paths["projection_metrics"], index=False)
    scores.to_csv(paths["projection_scores"], index=False)
    logreg.to_csv(paths["logistic"], index=False)
    meta = {
        "args": vars(args),
        "layers": LAYERS,
        "ks": KS,
        "clean_motion": "random crop, color jitter, blur, Gaussian noise, same-class interpolation; all endpoints filtered to preserve ResNet18 top-1 class",
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2))
    print("\nDIMENSIONALITY")
    print(dim.sort_values(["layer", "variant", "set"]).to_string(index=False))
    print("\nPROJECTION METRICS")
    print(metrics.sort_values(["layer", "variant", "basis_source", "k"]).to_string(index=False))
    print("\nLOGISTIC")
    print(logreg.sort_values(["layer", "variant", "basis_source"]).to_string(index=False))


if __name__ == "__main__":
    main()
