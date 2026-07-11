#!/usr/bin/env python3
"""Function-preserving coordinate rescaling stress test.

This script tests a reviewer concern: hidden-space PCA measurements can depend
on the coordinate system.  For VGG19-BN and ResNet50 on CIFAR-10, the pooled
penultimate representation feeds a single linear classifier, so positive
diagonal rescalings of that representation can be exactly compensated by
inversely rescaling the classifier weights.  The classifier function is
unchanged, but raw Euclidean PCA statistics in those coordinates may change.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.load_models import load_cifar_model


def effective_rank(eigvals: np.ndarray) -> float:
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvals = eigvals[eigvals > 0]
    if eigvals.size == 0:
        return 0.0
    p = eigvals / eigvals.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def dim_for_fraction(eigvals: np.ndarray, frac: float) -> int:
    eigvals = np.asarray(eigvals, dtype=np.float64)
    total = eigvals.sum()
    if total <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(eigvals) / total, frac) + 1)


def fit_pca_basis(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = x_train.mean(axis=0, keepdims=True)
    x_centered = x_train - center
    _, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    eigvals = (s**2) / max(1, x_centered.shape[0] - 1)
    return center.astype(np.float32), vt.astype(np.float32), eigvals.astype(np.float64)


def projection_energy(x: np.ndarray, center: np.ndarray, components: np.ndarray, k: int) -> np.ndarray:
    xc = x - center
    coeff = xc @ components[:k].T
    numerator = np.sum(coeff * coeff, axis=1)
    denominator = np.sum(xc * xc, axis=1) + 1e-12
    return numerator / denominator


class VggPenultimateRescaled(torch.nn.Module):
    def __init__(self, base: torch.nn.Sequential, scale: torch.Tensor):
        super().__init__()
        self.normalize = copy.deepcopy(base[0])
        self.vgg = copy.deepcopy(base[1])
        self.register_buffer("scale", scale.view(1, -1))
        if not isinstance(self.vgg.classifier, torch.nn.Linear):
            raise TypeError(f"Expected a single Linear classifier, got {type(self.vgg.classifier)}")
        if self.vgg.classifier.in_features != self.scale.numel():
            raise ValueError(
                f"Scale dimension {self.scale.numel()} does not match classifier input "
                f"{self.vgg.classifier.in_features}"
            )
        with torch.no_grad():
            self.vgg.classifier.weight.div_(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.normalize(x)
        z = self.vgg.features(z)
        z = torch.flatten(z, 1)
        z = z * self.scale
        return self.vgg.classifier(z)


class ResNetPooledRescaled(torch.nn.Module):
    def __init__(self, base: torch.nn.Sequential, scale: torch.Tensor):
        super().__init__()
        self.normalize = copy.deepcopy(base[0])
        self.resnet = copy.deepcopy(base[1])
        self.register_buffer("scale", scale.view(1, -1))
        if not isinstance(self.resnet.linear, torch.nn.Linear):
            raise TypeError(f"Expected a single Linear classifier, got {type(self.resnet.linear)}")
        if self.resnet.linear.in_features != self.scale.numel():
            raise ValueError(
                f"Scale dimension {self.scale.numel()} does not match classifier input "
                f"{self.resnet.linear.in_features}"
            )
        with torch.no_grad():
            self.resnet.linear.weight.div_(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.normalize(x)
        z = self.resnet.conv1(z)
        z = self.resnet.bn1(z)
        z = torch.relu(z)
        z = self.resnet.layer1(z)
        z = self.resnet.layer2(z)
        z = self.resnet.layer3(z)
        z = self.resnet.layer4(z)
        z = torch.nn.functional.avg_pool2d(z, 4)
        z = z.view(z.size(0), -1)
        z = z * self.scale
        return self.resnet.linear(z)


def make_lognormal_scale(dim: int, sigma: float, seed: int) -> np.ndarray:
    if sigma == 0:
        return np.ones(dim, dtype=np.float32)
    rng = np.random.default_rng(seed)
    log_scale = rng.normal(loc=0.0, scale=sigma, size=dim)
    log_scale = log_scale - log_scale.mean()
    return np.exp(log_scale).astype(np.float32)


def load_layer_vectors(nested_dir: Path, layer: str) -> tuple[np.ndarray, pd.DataFrame]:
    z = np.load(nested_dir / "nested_layer_segment_vectors.npz")
    vectors = z[layer].astype(np.float32)
    meta = pd.read_csv(nested_dir / "nested_layer_segment_metadata.csv")
    meta = meta[meta["layer"].eq(layer)].copy()
    meta = meta.sort_values("vector_idx")
    if len(meta) != len(vectors):
        raise ValueError(f"Metadata rows {len(meta)} do not match vectors {len(vectors)}")
    if not np.array_equal(meta["vector_idx"].to_numpy(), np.arange(len(vectors))):
        raise ValueError(f"{layer} metadata vector_idx is not contiguous")
    return vectors, meta.reset_index(drop=True)


def evaluate_scaled_vectors(vectors: np.ndarray, meta: pd.DataFrame, scale: np.ndarray, k: int) -> dict:
    scaled = vectors * scale.reshape(1, -1)
    train_mask = meta["split"].eq("basis_fit").to_numpy() & meta["success"].eq(1).to_numpy()
    test_mask = meta["split"].eq("final_test").to_numpy()
    center, components, eigvals = fit_pca_basis(scaled[train_mask])
    scores = projection_energy(scaled[test_mask], center, components, min(k, components.shape[0]))
    test_meta = meta.loc[test_mask, ["image_ord", "success"]].copy()
    test_meta["score"] = scores
    segment_auroc = float(roc_auc_score(test_meta["success"], test_meta["score"]))
    image_scores = test_meta.groupby("image_ord").agg(success=("success", "max"), score=("score", "mean"))
    image_auroc = float(roc_auc_score(image_scores["success"], image_scores["score"]))
    return {
        "segment_auroc": segment_auroc,
        "image_auroc": image_auroc,
        "projection_energy_success_mean": float(test_meta.loc[test_meta.success.eq(1), "score"].mean()),
        "projection_energy_failed_mean": float(test_meta.loc[test_meta.success.eq(0), "score"].mean()),
        "dim80": dim_for_fraction(eigvals, 0.80),
        "dim90": dim_for_fraction(eigvals, 0.90),
        "effective_rank": effective_rank(eigvals),
        "pc1_var": float(eigvals[0] / eigvals.sum()),
        "n_train_success_segments": int(train_mask.sum()),
        "n_test_segments": int(test_mask.sum()),
        "n_test_images": int(image_scores.shape[0]),
    }


def verify_logit_equivalence(
    scale: np.ndarray,
    split_csv: Path,
    dataset_root: str,
    model_name: str,
    split_seed: int,
    n_images: int,
    device: torch.device,
) -> dict:
    base = load_cifar_model(model_name).to(device).eval()
    if model_name == "bbb_vgg19_bn":
        scaled = VggPenultimateRescaled(base, torch.from_numpy(scale).to(device)).to(device).eval()
    elif model_name == "bbb_resnet50":
        scaled = ResNetPooledRescaled(base, torch.from_numpy(scale).to(device)).to(device).eval()
    else:
        raise ValueError(f"Function-preserving wrapper is not implemented for {model_name}")
    split_df = pd.read_csv(split_csv)
    rows = split_df[
        split_df["model"].eq(model_name)
        & split_df["split_seed"].eq(split_seed)
        & split_df["split"].eq("final_test")
    ].head(n_images)
    dataset = datasets.CIFAR10(dataset_root, train=False, download=False, transform=transforms.ToTensor())
    max_abs = 0.0
    max_rel = 0.0
    n = 0
    with torch.no_grad():
        for idx in rows["dataset_idx"].astype(int).tolist():
            x, _ = dataset[idx]
            x = x.unsqueeze(0).to(device)
            y0 = base(x)
            y1 = scaled(x)
            diff = (y0 - y1).abs()
            max_abs = max(max_abs, float(diff.max().item()))
            max_rel = max(max_rel, float((diff / (y0.abs() + 1e-8)).max().item()))
            n += 1
    return {"n_images": n, "max_abs_logit_diff": max_abs, "max_rel_logit_diff": max_rel}


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    display = df.copy()
    display["Transform"] = display["sigma"].map(lambda s: "Identity" if s == 0 else f"Diag. scale $\\sigma={s:g}$")
    display["dim80"] = display["dim80"].astype(int)
    display["dim90"] = display["dim90"].astype(int)
    display["Eff. rank"] = display["effective_rank"].map(lambda x: f"{x:.1f}")
    display["AUROC"] = display["image_auroc"].map(lambda x: f"{x:.3f}")
    base_auc = float(df.loc[df["sigma"].eq(0), "image_auroc"].iloc[0])
    display["$\\Delta$ AUROC"] = df["image_auroc"].map(lambda x: f"{x - base_auc:+.3f}")
    display["max $|\\Delta f|$"] = display["max_abs_logit_diff"].map(lambda x: f"{x:.2e}")
    out = display[["Transform", "dim80", "dim90", "Eff. rank", "AUROC", "$\\Delta$ AUROC", "max $|\\Delta f|$"]]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Transform & dim80 & dim90 & Eff. rank & AUROC & $\Delta$ AUROC & max $|\Delta f|$ \\",
        r"\midrule",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"{row['Transform']} & {row['dim80']} & {row['dim90']} & {row['Eff. rank']} & "
            f"{row['AUROC']} & {row['$\\Delta$ AUROC']} & {row['max $|\\Delta f|$']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nested-root", default="analysis_outputs/hidden_jacobian_routing/exact_protocol/phase1_nested_layer_selection")
    p.add_argument("--split-csv", default="analysis_outputs/hidden_jacobian_routing/exact_protocol/cifar_splits/cifar10_exact_splits.csv")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/function_preserving_coordinate_rescaling_vgg_penultimate")
    p.add_argument("--model", default="bbb_vgg19_bn")
    p.add_argument("--layer", default="penultimate")
    p.add_argument("--split-seed", type=int, default=1001)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--sigmas", default="0,0.5,1,2")
    p.add_argument("--seed", type=int, default=31415)
    p.add_argument("--logit-check-images", type=int, default=100)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nested_dir = Path(args.nested_root) / args.model / f"split_seed_{args.split_seed}"
    vectors, meta = load_layer_vectors(nested_dir, args.layer)
    dim = vectors.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric_rows = []
    logit_rows = []
    for sigma in [float(s) for s in args.sigmas.split(",")]:
        scale = make_lognormal_scale(dim, sigma, args.seed + int(round(sigma * 1000)))
        metrics = evaluate_scaled_vectors(vectors, meta, scale, args.k)
        logit = verify_logit_equivalence(
            scale=scale,
            split_csv=Path(args.split_csv),
            dataset_root=args.dataset_root,
            model_name=args.model,
            split_seed=args.split_seed,
            n_images=args.logit_check_images,
            device=device,
        )
        row = {
            "model": args.model,
            "layer": args.layer,
            "split_seed": args.split_seed,
            "k": args.k,
            "sigma": sigma,
            "scale_min": float(scale.min()),
            "scale_median": float(np.median(scale)),
            "scale_max": float(scale.max()),
            **metrics,
            **logit,
        }
        metric_rows.append(row)
        logit_rows.append({k: row[k] for k in ["model", "split_seed", "sigma", "n_images", "max_abs_logit_diff", "max_rel_logit_diff"]})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "function_preserving_projection_metrics.csv", index=False)
    pd.DataFrame(logit_rows).to_csv(out_dir / "function_preserving_logit_equivalence.csv", index=False)
    np.savez_compressed(
        out_dir / "rescaling_test_metadata_arrays.npz",
        sigmas=metrics_df["sigma"].to_numpy(),
        image_auroc=metrics_df["image_auroc"].to_numpy(),
        dim80=metrics_df["dim80"].to_numpy(),
        effective_rank=metrics_df["effective_rank"].to_numpy(),
    )
    write_latex_table(metrics_df, out_dir / "function_preserving_projection_metrics.tex")
    metadata = {
        "purpose": "Function-preserving pooled-feature coordinate diagonal rescaling stress test.",
        "interpretation": "Classifier logits are unchanged up to numerical precision; raw PCA dimensionality/AUROC changes measure coordinate dependence of raw hidden Euclidean statistics.",
        "model": args.model,
        "layer": args.layer,
        "split_seed": args.split_seed,
        "k": args.k,
        "sigmas": [float(s) for s in args.sigmas.split(",")],
        "nested_artifact": str(nested_dir),
        "split_csv": args.split_csv,
        "output_dir": str(out_dir),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(metrics_df.to_string(index=False))
    print(out_dir)


if __name__ == "__main__":
    main()
