#!/usr/bin/env python3
"""Static single-image adversarial-road anomaly detector pilot.

The trajectory work treats roads as directions of hidden motion.  This script
tests the static analogue: whether an image's hidden displacement from clean
class centroids has anomalous coordinates in an adversarial-road basis.

Quick intended use:

  python experiments/pure_af_geometry/test_single_image_road_detector.py \
    --output-dir analysis_outputs/pure_af_geometry/single_image_road_detector_resnet50_quick \
    --n-calib 500 --n-eval 200 --attacks pgd20,square
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.evaluate_road_damping_defense import (  # noqa: E402
    DampedResNet50,
    fit_candidate_bases,
    margin,
    project_linf,
)
from utils.load_models import load_cifar_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(device: torch.device) -> DampedResNet50:
    seq = load_cifar_model("bbb_resnet50").to(device).eval()
    return DampedResNet50(seq, None).to(device).eval()


def select_clean_correct(dataset, model, n: int, device: torch.device, max_scan: int = 50000) -> pd.DataFrame:
    rows = []
    for idx in range(min(len(dataset), max_scan)):
        x, y0 = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([int(y0)], device=device)
        with torch.no_grad():
            logits = model(x)
        if int(logits.argmax(1).item()) == int(y0):
            rows.append({"dataset_idx": idx, "label": int(y0), "clean_margin": float(margin(logits, y).item())})
        if len(rows) >= n:
            break
    if len(rows) < n:
        print(f"[WARN] requested {n} clean-correct, found {len(rows)}", flush=True)
    return pd.DataFrame(rows)


def feature(model: DampedResNet50, x: torch.Tensor) -> torch.Tensor:
    return model.pooled_layer4(x).flatten(1)


def logits_margin_conf(model, x: torch.Tensor, y: torch.Tensor):
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
    pred = logits.argmax(1)
    conf = probs.max(1).values
    m_true = margin(logits, y)
    # Predicted-class margin for deployable confidence/margin baseline.
    pred_true = logits.gather(1, pred[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, pred[:, None], -1e9)
    pred_margin = pred_true - masked.max(1).values
    return logits, pred, conf, m_true, pred_margin


def batch_features(dataset, rows: pd.DataFrame, model, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start : start + batch_size]
        xs = torch.stack([dataset[int(r.dataset_idx)][0] for r in batch.itertuples(index=False)]).to(device)
        with torch.no_grad():
            feats.append(feature(model, xs).cpu().numpy())
        labels.extend([int(r.label) for r in batch.itertuples(index=False)])
    return np.concatenate(feats, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)


def orthonormal_rows(x: np.ndarray, k: int) -> np.ndarray:
    q, _ = np.linalg.qr(x[:k].T)
    return q[:, : min(k, q.shape[1])].T.astype(np.float32)


def fit_clean_stats(feats: np.ndarray, labels: np.ndarray, basis: np.ndarray, ridge: float) -> dict:
    u = basis.T.astype(np.float32)
    d = feats.shape[1]
    k = basis.shape[0]
    centroids = np.zeros((10, d), dtype=np.float32)
    coeff_mean = np.zeros((10, k), dtype=np.float32)
    inv_cov = np.zeros((10, k, k), dtype=np.float32)
    for c in range(10):
        fc = feats[labels == c]
        if len(fc) == 0:
            raise RuntimeError(f"No calibration features for class {c}.")
        centroids[c] = fc.mean(axis=0)
        a = (fc - centroids[c][None, :]) @ u
        coeff_mean[c] = a.mean(axis=0)
        if len(a) <= 1:
            cov = np.eye(k, dtype=np.float32)
        else:
            cov = np.cov(a, rowvar=False).astype(np.float32)
        scale = float(np.trace(cov) / max(k, 1))
        cov = cov + (ridge * scale + 1e-5) * np.eye(k, dtype=np.float32)
        inv_cov[c] = np.linalg.pinv(cov).astype(np.float32)
    return {"centroids": centroids, "coeff_mean": coeff_mean, "inv_cov": inv_cov, "basis": basis.astype(np.float32)}


def score_with_stats(h: np.ndarray, pred: int, true_label: int, stats: dict, prefix: str) -> dict:
    basis = stats["basis"]
    u = basis.T
    centroids = stats["centroids"]
    coeff_mean = stats["coeff_mean"]
    inv_cov = stats["inv_cov"]
    energies, mags, mahas = [], [], []
    for c in range(10):
        v = h - centroids[c]
        a = v @ u
        energy = float(np.sum(a * a) / max(float(np.sum(v * v)), 1e-12))
        mag = float(np.linalg.norm(a))
        diff = a - coeff_mean[c]
        maha = float(diff @ inv_cov[c] @ diff)
        energies.append(energy)
        mags.append(mag)
        mahas.append(maha)
    return {
        f"{prefix}_energy_oracle": energies[true_label],
        f"{prefix}_energy_pred": energies[pred],
        f"{prefix}_energy_min": float(np.min(energies)),
        f"{prefix}_mag_oracle": mags[true_label],
        f"{prefix}_mag_pred": mags[pred],
        f"{prefix}_mag_min": float(np.min(mags)),
        f"{prefix}_maha_oracle": mahas[true_label],
        f"{prefix}_maha_pred": mahas[pred],
        f"{prefix}_maha_min": float(np.min(mahas)),
    }


def full_hidden_maha_stats(feats: np.ndarray, labels: np.ndarray, k: int, ridge: float) -> dict:
    # Use PCA-whitened hidden coordinates to avoid inverting 2048-d covariance.
    pca = PCA(n_components=min(k, feats.shape[0] - 1, feats.shape[1]), svd_solver="randomized", random_state=0)
    z = pca.fit_transform(feats)
    stats = fit_clean_stats(z.astype(np.float32), labels, np.eye(z.shape[1], dtype=np.float32), ridge)
    stats["pca_mean"] = pca.mean_.astype(np.float32)
    stats["pca_components"] = pca.components_.astype(np.float32)
    return stats


def score_full_hidden(h: np.ndarray, pred: int, true_label: int, stats: dict) -> dict:
    z = (h - stats["pca_mean"]) @ stats["pca_components"].T
    return score_with_stats(z.astype(np.float32), pred, true_label, stats, "full_hidden")


def pgd_attack(model, x0, y, eps: float, steps: int, step_size: float, seed: int):
    torch.manual_seed(seed)
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
    for _ in range(steps):
        x.requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        grad = torch.autograd.grad(loss, x)[0]
        x = project_linf(x.detach() + step_size * grad.detach().sign(), x0, eps)
    return x.detach()


def square_attack(model, x0, y, eps: float, queries: int, seed: int):
    rng = np.random.default_rng(seed)
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
    with torch.no_grad():
        best_loss = float(F.cross_entropy(model(x), y).item())
    for q in range(queries):
        size = max(1, int(round(32 * (1 - q / max(queries, 1)) ** 0.5)))
        i = int(rng.integers(0, 33 - size))
        j = int(rng.integers(0, 33 - size))
        cand = x.clone()
        sign = -1.0 if rng.random() < 0.5 else 1.0
        cand[:, :, i : i + size, j : j + size] = x0[:, :, i : i + size, j : j + size] + sign * eps
        cand = project_linf(cand, x0, eps)
        with torch.no_grad():
            logits = model(cand)
            loss = float(F.cross_entropy(logits, y).item())
            success = int(logits.argmax(1).item()) != int(y.item())
        if success:
            return cand.detach(), 1
        if loss > best_loss:
            x = cand
            best_loss = loss
    with torch.no_grad():
        success = int(model(x).argmax(1).item()) != int(y.item())
    return x.detach(), int(success)


def benign_aug(x: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=g, device=x.device) * (2 / 255.0)
    scale = 0.9 + 0.2 * torch.rand((x.shape[0], 1, 1, 1), generator=g, device=x.device)
    return (x * scale + noise).clamp(0, 1)


def metric_at_fpr(y_true: np.ndarray, score: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, score)
    ok = fpr <= fpr_target
    if not ok.any():
        return 0.0
    return float(tpr[ok].max())


def safe_auroc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def safe_auprc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, score))


def summarize_detection(df: pd.DataFrame, scores: list[str]) -> pd.DataFrame:
    rows = []
    clean = df[df.kind == "clean"].copy()
    hard_cut = clean.clean_margin.quantile(0.10)
    hard = clean[clean.clean_margin <= hard_cut]
    benign = df[df.kind == "benign_aug"]
    for attack in sorted(k for k in df.kind.unique() if k.startswith("adv_")):
        sub = pd.concat([clean, df[(df.kind == attack) & (df.attack_success == 1)]], ignore_index=True)
        if sub.kind.nunique() < 2:
            continue
        y = (sub.kind == attack).astype(int).to_numpy()
        for s in scores:
            vals = sub[s].to_numpy(dtype=float)
            clean_vals = clean[s].to_numpy(dtype=float)
            thresh_1 = np.quantile(clean_vals, 0.99)
            thresh_5 = np.quantile(clean_vals, 0.95)
            rows.append(
                {
                    "score": s,
                    "attack": attack,
                    "n_clean": int(len(clean)),
                    "n_adv_success": int((sub.kind == attack).sum()),
                    "AUROC": safe_auroc(y, vals),
                    "AUPRC": safe_auprc(y, vals),
                    "TPR@1FPR": metric_at_fpr(y, vals, 0.01),
                    "TPR@5FPR": metric_at_fpr(y, vals, 0.05),
                    "FPR_hard_clean_at_5FPR": float((hard[s] > thresh_5).mean()) if len(hard) else np.nan,
                    "FPR_benign_at_5FPR": float((benign[s] > thresh_5).mean()) if len(benign) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_margin_matched(df: pd.DataFrame, scores: list[str], tol_quantile: float = 0.05) -> pd.DataFrame:
    rows = []
    clean = df[df.kind == "clean"].copy()
    margins = clean.current_margin.to_numpy()
    tol = float(max(np.quantile(np.abs(np.diff(np.sort(margins))) + 1e-6, 0.9), clean.current_margin.std() * tol_quantile))
    for attack in sorted(k for k in df.kind.unique() if k.startswith("adv_")):
        adv = df[(df.kind == attack) & (df.attack_success == 1)].copy()
        if adv.empty:
            continue
        clean_used, adv_used = [], []
        for _, a in adv.iterrows():
            j = (clean.current_margin - a.current_margin).abs().idxmin()
            if abs(float(clean.loc[j, "current_margin"] - a.current_margin)) <= tol:
                adv_used.append(a)
                clean_used.append(clean.loc[j])
        if not adv_used:
            continue
        cdf = pd.DataFrame(clean_used)
        adf = pd.DataFrame(adv_used)
        for s in scores:
            y = np.r_[np.zeros(len(cdf)), np.ones(len(adf))]
            vals = np.r_[cdf[s].to_numpy(float), adf[s].to_numpy(float)]
            rows.append(
                {
                    "score": s,
                    "attack": attack,
                    "matched_pairs": int(len(adf)),
                    "margin_tolerance": tol,
                    "adv_mean": float(adf[s].mean()),
                    "clean_mean": float(cdf[s].mean()),
                    "AUROC_matched": safe_auroc(y, vals),
                }
            )
    return pd.DataFrame(rows)


def plot_outputs(df: pd.DataFrame, summary: pd.DataFrame, out: Path, key_scores: list[str]) -> None:
    fig, axes = plt.subplots(1, len(key_scores), figsize=(4.5 * len(key_scores), 3.2), constrained_layout=True)
    if len(key_scores) == 1:
        axes = [axes]
    for ax, s in zip(axes, key_scores):
        for kind, color in [("clean", "#4C78A8"), ("adv_pgd20", "#E45756"), ("adv_square", "#F58518"), ("benign_aug", "#54A24B")]:
            sub = df[df.kind == kind]
            if not sub.empty:
                ax.hist(sub[s], bins=35, alpha=0.45, density=True, label=kind, color=color)
        ax.set_title(s)
        ax.set_ylabel("density")
        ax.legend(fontsize=7)
    fig.savefig(out / "score_histograms.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    clean = df[df.kind == "clean"]
    for attack, color in [("adv_pgd20", "#E45756"), ("adv_square", "#F58518")]:
        adv = df[(df.kind == attack) & (df.attack_success == 1)]
        sub = pd.concat([clean, adv], ignore_index=True)
        if sub.kind.nunique() < 2:
            continue
        y = (sub.kind == attack).astype(int).to_numpy()
        for s, ls in [(key_scores[0], "-"), ("neg_pred_margin", "--")]:
            fpr, tpr, _ = roc_curve(y, sub[s].to_numpy(float))
            ax.plot(fpr, tpr, ls=ls, color=color, label=f"{attack} {s}")
    ax.plot([0, 1], [0, 1], color="0.6", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=7)
    fig.savefig(out / "roc_curves.png", dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/single_image_road_detector_resnet50_quick")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--n-calib", type=int, default=500)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--basis-k", type=int, default=20)
    p.add_argument("--calib-dirs", type=int, default=96)
    p.add_argument("--basis-select", type=int, default=3000)
    p.add_argument("--ridge", type=float, default=0.1)
    p.add_argument("--attacks", default="pgd20,square")
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--pgd-step", type=float, default=2.0)
    p.add_argument("--square-queries", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    train = datasets.CIFAR10(args.dataset_root, train=True, download=False, transform=transforms.ToTensor())
    test = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    calib_rows = select_clean_correct(train, model, args.n_calib, device)
    eval_rows = select_clean_correct(test, model, args.n_eval, device)
    calib_rows.to_csv(out / "calib_clean_correct.csv", index=False)
    eval_rows.to_csv(out / "eval_clean_correct.csv", index=False)

    calib_feats, calib_labels = batch_features(train, calib_rows, model, device, args.batch_size)
    road_args = argparse.Namespace(
        calib_eps=args.eps,
        calib_dirs=args.calib_dirs,
        batch_size=args.batch_size,
        seed=args.seed,
        basis_select=args.basis_select,
        k=args.basis_k,
    )
    bases = fit_candidate_bases(model, train, calib_rows, np.zeros(calib_feats.shape[1], dtype=np.float32), road_args, device)
    adv_basis = orthonormal_rows(bases["adv_road"], args.basis_k)
    clean_basis = orthonormal_rows(bases.get("clean_motion", bases["mobility_only"]), args.basis_k)
    rng = np.random.default_rng(args.seed + 44)
    random_basis = orthonormal_rows(rng.normal(size=(args.basis_k, calib_feats.shape[1])).astype(np.float32), args.basis_k)
    np.savez(out / "detector_bases.npz", adv_road=adv_basis, clean_motion=clean_basis, random=random_basis)

    stats = {
        "road": fit_clean_stats(calib_feats, calib_labels, adv_basis, args.ridge),
        "random": fit_clean_stats(calib_feats, calib_labels, random_basis, args.ridge),
        "clean_basis": fit_clean_stats(calib_feats, calib_labels, clean_basis, args.ridge),
        "full_hidden": full_hidden_maha_stats(calib_feats, calib_labels, min(50, len(calib_feats) - 1), args.ridge),
    }

    records = []
    attacks = {a.strip() for a in args.attacks.split(",") if a.strip()}
    eps = args.eps / 255.0
    step = args.pgd_step / 255.0
    for row in eval_rows.itertuples(index=False):
        x0, _ = test[int(row.dataset_idx)]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        variants = [("clean", x0.detach(), 0)]
        variants.append(("benign_aug", benign_aug(x0, args.seed + int(row.dataset_idx)), 0))
        if "pgd20" in attacks:
            x_adv = pgd_attack(model, x0, y, eps, 20, step, args.seed + int(row.dataset_idx) * 7)
            with torch.no_grad():
                succ = int(model(x_adv).argmax(1).item() != int(row.label))
            variants.append(("adv_pgd20", x_adv, succ))
        if "pgd100" in attacks:
            x_adv = pgd_attack(model, x0, y, eps, 100, step, args.seed + int(row.dataset_idx) * 11)
            with torch.no_grad():
                succ = int(model(x_adv).argmax(1).item() != int(row.label))
            variants.append(("adv_pgd100", x_adv, succ))
        if "square" in attacks:
            x_adv, succ = square_attack(model, x0, y, eps, args.square_queries, args.seed + int(row.dataset_idx) * 13)
            variants.append(("adv_square", x_adv, succ))
        for kind, x, succ in variants:
            logits, pred, conf, m_true, pred_margin = logits_margin_conf(model, x, y)
            with torch.no_grad():
                h = feature(model, x).cpu().numpy()[0].astype(np.float32)
            pred_i = int(pred.item())
            label_i = int(row.label)
            rec = {
                "image_id": int(row.dataset_idx),
                "split": "eval",
                "kind": kind,
                "true_label": label_i,
                "pred_label": pred_i,
                "attack_success": int(succ),
                "clean_margin": float(row.clean_margin),
                "current_margin": float(m_true.item()),
                "confidence": float(conf.item()),
                "neg_confidence": float(-conf.item()),
                "neg_true_margin": float(-m_true.item()),
                "neg_pred_margin": float(-pred_margin.item()),
            }
            rec.update(score_with_stats(h, pred_i, label_i, stats["road"], "road"))
            rec.update(score_with_stats(h, pred_i, label_i, stats["random"], "random"))
            rec.update(score_with_stats(h, pred_i, label_i, stats["clean_basis"], "clean_basis"))
            rec.update(score_full_hidden(h, pred_i, label_i, stats["full_hidden"]))
            records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(out / "per_image_scores.csv", index=False)

    score_cols = [
        "road_maha_oracle",
        "road_maha_pred",
        "road_maha_min",
        "road_energy_oracle",
        "road_energy_min",
        "random_maha_min",
        "clean_basis_maha_min",
        "full_hidden_maha_min",
        "neg_confidence",
        "neg_true_margin",
        "neg_pred_margin",
    ]
    det = summarize_detection(df, score_cols)
    det.to_csv(out / "summary_detection.csv", index=False)
    mm = summarize_margin_matched(df, score_cols)
    mm.to_csv(out / "summary_margin_matched.csv", index=False)
    hard_cut = df[df.kind == "clean"].clean_margin.quantile(0.10)
    hard = df[(df.kind == "clean") & (df.clean_margin <= hard_cut)]
    hard[["image_id", "clean_margin", *score_cols]].to_csv(out / "summary_hard_clean.csv", index=False)
    plot_outputs(df, det, out, ["road_maha_min", "road_maha_oracle", "neg_pred_margin"])
    (out / "metadata.json").write_text(json.dumps(vars(args) | {"device": str(device)}, indent=2))

    print("Detection summary (top road/margin scores):", flush=True)
    keep = det[
        det.score.isin(
            ["road_maha_oracle", "road_maha_pred", "road_maha_min", "neg_true_margin", "neg_pred_margin", "random_maha_min"]
        )
    ]
    print(keep.sort_values(["attack", "AUROC"], ascending=[True, False]).to_string(index=False), flush=True)
    print("\nMargin matched:", flush=True)
    if mm.empty or "score" not in mm.columns:
        print("[no margin-matched pairs found]", flush=True)
    else:
        keep_mm = mm[mm.score.isin(["road_maha_oracle", "road_maha_min", "neg_true_margin", "neg_pred_margin", "random_maha_min"])]
        print(keep_mm.sort_values(["attack", "AUROC_matched"], ascending=[True, False]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
