#!/usr/bin/env python3
"""Clean-whitened hidden-Jacobian mobility check.

This script tests whether the hidden-Jacobian mobility result is an artifact of
raw hidden-coordinate scaling.  It estimates a whitening transform from clean
class-correct features, then recomputes finite-difference mobility, exact JVP
gain, transport/JVP basis overlap, and residual candidate scores in whitened
coordinates.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import margin  # noqa: E402
from experiments.pure_af_geometry.test_jacobian_basis_and_residual_transport import (  # noqa: E402
    SegmentStore,
    collect_vectors,
    feature_tensor,
    load_images_for_split,
    normalize_rows,
    orth_residual_basis,
    pca_basis,
    projection_energy,
    subspace_metrics,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv(text: str, typ=str) -> list:
    return [typ(x.strip()) for x in text.split(",") if x.strip()]


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    ok = np.isfinite(score)
    if ok.sum() < 4 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], score[ok]))


def safe_auprc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    ok = np.isfinite(score)
    if ok.sum() < 4 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(average_precision_score(y[ok], score[ok]))


def row_apply_whiten(x: np.ndarray, mean: np.ndarray, whiten: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mean.astype(np.float32)) @ whiten.T.astype(np.float32)).astype(np.float32)


def vec_apply_whiten(x: np.ndarray, whiten: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) @ whiten.T.astype(np.float32)).astype(np.float32)


def collect_clean_features(args, wrapper, dataset, device: torch.device) -> np.ndarray:
    rows = []
    scanned = 0
    for idx in range(len(dataset)):
        x_cpu, y_int = dataset[idx]
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(y_int)], device=device)
        with torch.no_grad():
            logits = wrapper(x)
            if int(logits.argmax(1).item()) != int(y.item()):
                continue
            h = feature_tensor(wrapper, x, args.layer).detach().cpu().numpy()[0].astype(np.float32)
        rows.append(h)
        scanned += 1
        if len(rows) >= args.clean_features:
            break
        if scanned % max(1, args.progress_every) == 0:
            print(f"[clean] collected={len(rows)} scanned={idx + 1}", flush=True)
    if len(rows) < 4:
        raise RuntimeError(f"Too few clean-correct features for whitening: {len(rows)}")
    return np.stack(rows, axis=0).astype(np.float32)


def fit_whitener(clean: np.ndarray, shrinkage: float, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = clean.mean(axis=0).astype(np.float32)
    centered = clean - mean
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    trace = float(np.trace(cov))
    isotropic = trace / max(cov.shape[0], 1)
    cov = (1.0 - shrinkage) * cov + shrinkage * isotropic * np.eye(cov.shape[0], dtype=np.float32)
    vals, vecs = np.linalg.eigh(cov.astype(np.float64))
    vals = np.clip(vals, eps, None)
    whiten = (vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T).astype(np.float32)
    return mean, whiten, vals.astype(np.float64)


def fit_whitened_transport_basis(args, store: SegmentStore, mean: np.ndarray, whiten: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _rows, x = collect_vectors(
        store,
        args.model,
        parse_csv(args.transport_sources),
        args.layer,
        split="train",
        final_success=1,
    )
    if len(x) < 2:
        raise RuntimeError("Too few successful transport vectors for whitened PCA basis.")
    xw = vec_apply_whiten(x, whiten)
    basis_input = normalize_rows(xw) if args.normalize_vectors else xw
    t_mean, t_basis, _kk = pca_basis(basis_input, max(args.k_list), args.seed + 17, False)
    return t_mean, t_basis


def compute_whitened_jvp_basis(args, wrapper, dataset, images: pd.DataFrame, whiten: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    device = args.device_obj
    for image_i, row in enumerate(images.itertuples(index=False), start=1):
        x_cpu, _ = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        gen = torch.Generator(device=device).manual_seed(args.seed + int(row.dataset_idx) * 1543 + 97)
        remaining = args.n_jvp_dirs
        while remaining > 0:
            bs = min(args.jvp_batch_size, remaining)
            signs = torch.where(
                torch.rand((bs,) + tuple(x0.shape[1:]), generator=gen, device=device) < 0.5,
                -torch.ones((bs,) + tuple(x0.shape[1:]), device=device),
                torch.ones((bs,) + tuple(x0.shape[1:]), device=device),
            )
            if args.unit_l2_jvp_dirs:
                signs = signs / signs.flatten(1).norm(dim=1).view(bs, 1, 1, 1).clamp_min(1e-12)
            x_batch = x0.repeat(bs, 1, 1, 1)

            def feat(inp: torch.Tensor) -> torch.Tensor:
                return feature_tensor(wrapper, inp, args.layer)

            _val, jvp = torch.autograd.functional.jvp(feat, x_batch, signs, create_graph=False, strict=False)
            vectors.append(vec_apply_whiten(jvp.detach().cpu().numpy().astype(np.float32), whiten))
            remaining -= bs
        if image_i % max(1, args.progress_every) == 0:
            print(f"[jvp-basis] {image_i}/{len(images)} train images", flush=True)
    z = np.concatenate(vectors, axis=0).astype(np.float32)
    basis_input = normalize_rows(z) if args.normalize_vectors else z
    _mean, basis, _kk = pca_basis(basis_input, max(args.k_list), args.seed + 101, False)
    return z, basis


def evaluate_candidates(args, wrapper, dataset, images: pd.DataFrame, whiten: np.ndarray, transport_mean: np.ndarray, transport_basis: np.ndarray, jvp_basis: np.ndarray) -> pd.DataFrame:
    rows = []
    device = args.device_obj
    residual_basis = orth_residual_basis(transport_basis, jvp_basis, max(args.k_list), max(args.k_list))
    for eps in parse_csv(args.eps_list, float):
        for alpha in parse_csv(args.alpha_list, float):
            probe_eps = eps * alpha / 255.0
            attack_eps = eps / 255.0
            for image_i, row in enumerate(images.itertuples(index=False), start=1):
                x_cpu, _ = dataset[int(row.dataset_idx)]
                x0 = x_cpu.unsqueeze(0).to(device)
                y = torch.tensor([int(row.label)], device=device)
                with torch.no_grad():
                    logits0 = wrapper(x0)
                    if int(logits0.argmax(1).item()) != int(row.label):
                        continue
                    clean_margin = float(margin(logits0, y).item())
                    h0_t = feature_tensor(wrapper, x0, args.layer).detach()
                gen = torch.Generator(device=device).manual_seed(args.seed + int(row.dataset_idx) * 1009 + int(eps * 100) * 917 + int(alpha * 1000))
                remaining = args.directions_per_image
                direction_id = 0
                while remaining > 0:
                    bs = min(args.jvp_batch_size, remaining)
                    signs = torch.where(
                        torch.rand((bs,) + tuple(x0.shape[1:]), generator=gen, device=device) < 0.5,
                        -torch.ones((bs,) + tuple(x0.shape[1:]), device=device),
                        torch.ones((bs,) + tuple(x0.shape[1:]), device=device),
                    )
                    delta_probe = probe_eps * signs
                    x_batch = x0.repeat(bs, 1, 1, 1)

                    def feat(inp: torch.Tensor) -> torch.Tensor:
                        return feature_tensor(wrapper, inp, args.layer)

                    with torch.no_grad():
                        x_probe = (x_batch + delta_probe).clamp(0, 1)
                        x_full = (x_batch + attack_eps * signs).clamp(0, 1)
                        logits_probe = wrapper(x_probe)
                        logits_full = wrapper(x_full)
                        h_probe = feature_tensor(wrapper, x_probe, args.layer).detach()
                    _val, jvp = torch.autograd.functional.jvp(feat, x_batch, delta_probe, create_graph=False, strict=False)
                    fd_w = vec_apply_whiten((h_probe - h0_t).detach().cpu().numpy().astype(np.float32), whiten)
                    jvp_w = vec_apply_whiten(jvp.detach().cpu().numpy().astype(np.float32), whiten)
                    fd_norm = np.linalg.norm(fd_w, axis=1)
                    jvp_norm = np.linalg.norm(jvp_w, axis=1)
                    dot = np.sum(fd_w * jvp_w, axis=1)
                    fd_jvp_cos = dot / np.clip(fd_norm * jvp_norm, 1e-12, None)
                    nonlinear_ratio = np.linalg.norm(fd_w - jvp_w, axis=1) / np.clip(fd_norm, 1e-12, None)
                    basis_input = normalize_rows(fd_w) if args.normalize_vectors else fd_w
                    transport_energy = projection_energy(basis_input, transport_mean, transport_basis, max(args.k_list)).astype(np.float32)
                    residual_energy = projection_energy(basis_input, np.zeros_like(transport_mean), residual_basis, max(args.k_list)).astype(np.float32)
                    probe_margin = margin(logits_probe, y.expand(bs)).detach().cpu().numpy().astype(np.float32)
                    full_margin = margin(logits_full, y.expand(bs)).detach().cpu().numpy().astype(np.float32)
                    pred_full = logits_full.argmax(1).detach().cpu().numpy().astype(np.int64)
                    for j in range(bs):
                        rows.append(
                            {
                                "image_ord": int(row.image_ord),
                                "dataset_idx": int(row.dataset_idx),
                                "label": int(row.label),
                                "eps_over_255": float(eps),
                                "alpha": float(alpha),
                                "direction_id": int(direction_id + j),
                                "clean_margin": clean_margin,
                                "probe_margin_drop": float(clean_margin - probe_margin[j]),
                                "full_margin_drop": float(clean_margin - full_margin[j]),
                                "full_pred": int(pred_full[j]),
                                "full_success": int(pred_full[j] != int(row.label)),
                                "mobility_fd_whitened": float(fd_norm[j]),
                                "jvp_gain_whitened": float(jvp_norm[j]),
                                "fd_jvp_cos_whitened": float(fd_jvp_cos[j]),
                                "nonlinear_ratio_whitened": float(nonlinear_ratio[j]),
                                "transport_energy_whitened": float(transport_energy[j]),
                                "residual_transport_energy_whitened": float(residual_energy[j]),
                            }
                        )
                    remaining -= bs
                    direction_id += bs
                if image_i % max(1, args.progress_every) == 0:
                    print(f"[candidate] eps={eps} alpha={alpha} {image_i}/{len(images)} images", flush=True)
    return pd.DataFrame(rows)


def summarize_fd(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (eps, alpha), g in df.groupby(["eps_over_255", "alpha"], sort=True):
        rows.append(
            {
                "eps_over_255": eps,
                "alpha": alpha,
                "n_candidates": int(len(g)),
                "random_candidate_asr": float(g.full_success.mean()),
                "spearman_mobility_jvp_whitened": float(g.mobility_fd_whitened.corr(g.jvp_gain_whitened, method="spearman")),
                "pearson_mobility_jvp_whitened": float(g.mobility_fd_whitened.corr(g.jvp_gain_whitened, method="pearson")),
                "median_fd_jvp_cos_whitened": float(g.fd_jvp_cos_whitened.median()),
                "median_nonlinear_ratio_whitened": float(g.nonlinear_ratio_whitened.median()),
                "mobility_success_auroc_whitened": safe_auc(g.full_success.to_numpy(), g.mobility_fd_whitened.to_numpy()),
                "jvp_success_auroc_whitened": safe_auc(g.full_success.to_numpy(), g.jvp_gain_whitened.to_numpy()),
                "transport_success_auroc_whitened": safe_auc(g.full_success.to_numpy(), g.transport_energy_whitened.to_numpy()),
                "residual_transport_success_auroc_whitened": safe_auc(g.full_success.to_numpy(), g.residual_transport_energy_whitened.to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def summarize_residual(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scores = [
        "probe_margin_drop",
        "mobility_fd_whitened",
        "jvp_gain_whitened",
        "transport_energy_whitened",
        "residual_transport_energy_whitened",
    ]
    for (eps, alpha), g in df.groupby(["eps_over_255", "alpha"], sort=True):
        y = g.full_success.to_numpy()
        for score in scores:
            rows.append(
                {
                    "eps_over_255": eps,
                    "alpha": alpha,
                    "score": score,
                    "auroc": safe_auc(y, g[score].to_numpy()),
                    "auprc": safe_auprc(y, g[score].to_numpy()),
                    "n_candidates": int(len(g)),
                    "n_success": int(g.full_success.sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--layer", required=True)
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--clean-features", type=int, default=1000)
    p.add_argument("--clean-train", action="store_true", help="Use CIFAR train split for clean whitening statistics.")
    p.add_argument("--shrinkage", type=float, default=0.05)
    p.add_argument("--eig-eps", type=float, default=1e-5)
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--directions-per-image", type=int, default=64)
    p.add_argument("--n-jvp-dirs", type=int, default=32)
    p.add_argument("--jvp-batch-size", type=int, default=8)
    p.add_argument("--eps-list", default="2,4,8")
    p.add_argument("--alpha-list", default="0.125,0.25,0.5")
    p.add_argument("--k-list", default="5,10,20")
    p.add_argument("--transport-sources", default="pgd,square")
    p.add_argument("--normalize-vectors", action="store_true", default=True)
    p.add_argument("--unit-l2-jvp-dirs", action="store_true")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.k_list = parse_csv(args.k_list, int)
    args.device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapper = load_model(args.model, args.device_obj)
    clean_dataset = datasets.CIFAR10(root=args.dataset_root, train=bool(args.clean_train), download=False, transform=transforms.ToTensor())
    eval_dataset = datasets.CIFAR10(root=args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    store = SegmentStore(Path(args.input_dir))

    clean = collect_clean_features(args, wrapper, clean_dataset, args.device_obj)
    mean, whiten, eigvals = fit_whitener(clean, args.shrinkage, args.eig_eps)
    np.savez_compressed(out_dir / "clean_whitener.npz", mean=mean, whiten=whiten, eigvals=eigvals)

    train_images = load_images_for_split(Path(args.input_dir), args.model, "train", max(args.images, 100))
    test_images = load_images_for_split(Path(args.input_dir), args.model, "test", args.images)
    transport_mean, transport_basis = fit_whitened_transport_basis(args, store, mean, whiten)
    jvp_vectors, jvp_basis = compute_whitened_jvp_basis(args, wrapper, eval_dataset, train_images, whiten)
    np.savez_compressed(out_dir / "whitened_basis_vectors.npz", transport_mean=transport_mean, transport_basis=transport_basis, jvp_basis=jvp_basis, jvp_vectors=jvp_vectors)

    overlap_rows = []
    for k in args.k_list:
        m = subspace_metrics(transport_basis, jvp_basis, k)
        overlap_rows.append(
            {
                "comparison": "transport_vs_jvp_sketch_whitened",
                "k": k,
                "overlap": m["overlap"],
                "mean_principal_angle_deg": m["mean_angle_deg"],
                "max_principal_angle_deg": m["max_angle_deg"],
                "subspace_affinity": m["affinity"],
            }
        )
    pd.DataFrame(overlap_rows).to_csv(out_dir / "whitened_transport_jvp_overlap.csv", index=False)

    candidates = evaluate_candidates(args, wrapper, eval_dataset, test_images, whiten, transport_mean, transport_basis, jvp_basis)
    candidates.to_csv(out_dir / "whitened_candidate_level_jvp.csv", index=False)
    fd_summary = summarize_fd(candidates)
    fd_summary.to_csv(out_dir / "whitened_mobility_jvp_metrics.csv", index=False)
    residual_summary = summarize_residual(candidates)
    residual_summary.to_csv(out_dir / "whitened_residual_prediction.csv", index=False)

    meta = {
        "model": args.model,
        "layer": args.layer,
        "input_dir": str(args.input_dir),
        "clean_features": int(len(clean)),
        "clean_train": bool(args.clean_train),
        "feature_dim": int(clean.shape[1]),
        "shrinkage": float(args.shrinkage),
        "eig_eps": float(args.eig_eps),
        "eval_images": int(len(test_images)),
        "directions_per_image": int(args.directions_per_image),
        "n_jvp_dirs": int(args.n_jvp_dirs),
        "eps_list": parse_csv(args.eps_list, float),
        "alpha_list": parse_csv(args.alpha_list, float),
        "k_list": args.k_list,
    }
    (out_dir / "clean_covariance_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "# Clean-Whitened Mobility/JVP Findings",
        "",
        f"- Model: `{args.model}`",
        f"- Layer: `{args.layer}`",
        f"- Clean features used for whitening: {len(clean)}",
        f"- Feature dimension: {clean.shape[1]}",
        "",
        "## Small-Probe Whitened Mobility Versus JVP",
    ]
    small = fd_summary[(fd_summary.eps_over_255 == 2) & (fd_summary.alpha == 0.125)]
    if len(small):
        r = small.iloc[0]
        lines.append(f"- Spearman={r.spearman_mobility_jvp_whitened:.3f}; median cosine={r.median_fd_jvp_cos_whitened:.3f}; median nonlinear ratio={r.median_nonlinear_ratio_whitened:.3f}.")
    lines += ["", "## Whitened Transport/JVP Overlap"]
    for r in overlap_rows:
        lines.append(f"- k={r['k']}: overlap={r['overlap']:.3f}, mean angle={r['mean_principal_angle_deg']:.1f} deg.")
    (out_dir / "clean_whitened_mobility_jvp_findings.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
