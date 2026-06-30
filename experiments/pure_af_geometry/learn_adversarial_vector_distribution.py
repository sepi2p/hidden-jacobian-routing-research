#!/usr/bin/env python3
"""Learn a distribution of adversarially useful hidden transition vectors.

This pilot tests whether a learned distribution over road-coordinate transition
vectors adds value beyond simple margin-drop or road-energy scores.

Object being learned:

    a_t = U^T (h_l(x_{t+1}) - h_l(x_t))

where U is an adversarial-road basis at pooled ResNet layer4.
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
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


def feature(model: DampedResNet50, x: torch.Tensor) -> torch.Tensor:
    return model.pooled_layer4(x).flatten(1)


def orthonormal_rows(x: np.ndarray, k: int) -> np.ndarray:
    q, _ = np.linalg.qr(x[:k].T)
    return q[:, : min(k, q.shape[1])].T.astype(np.float32)


def select_clean_correct(dataset, model, n: int, device: torch.device) -> pd.DataFrame:
    rows = []
    for idx in range(len(dataset)):
        x, y0 = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([int(y0)], device=device)
        with torch.no_grad():
            logits = model(x)
        if int(logits.argmax(1).item()) == int(y0):
            rows.append({"dataset_idx": idx, "label": int(y0), "clean_margin": float(margin(logits, y).item())})
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)


def state(model, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, float, float, int]:
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        h = feature(model, x)
    return h, float(margin(logits, y).item()), float(probs.max(1).values.item()), int(logits.argmax(1).item() != int(y.item()))


def transition_record(
    model,
    x_prev,
    x_next,
    y,
    u: torch.Tensor,
    kind: str,
    image_id: int,
    label: int,
    step: int,
    final_success: int,
) -> tuple[dict, np.ndarray]:
    h0, m0, c0, s0 = state(model, x_prev, y)
    h1, m1, c1, s1 = state(model, x_next, y)
    dh = h1 - h0
    coeff = (dh @ u).detach().cpu().numpy()[0].astype(np.float32)
    proj = torch.as_tensor(coeff, device=dh.device, dtype=dh.dtype)[None, :] @ u.T
    z = dh.flatten(1)
    road = float(proj.norm(dim=1).item())
    total = float(z.norm(dim=1).item())
    orth = float((z - proj).norm(dim=1).item())
    margin_drop = max(m0 - m1, 0.0)
    rec = {
        "kind": kind,
        "image_id": int(image_id),
        "label": int(label),
        "step": int(step),
        "success_now": int(s1),
        "success_final": int(final_success),
        "margin_t": m0,
        "margin_t1": m1,
        "margin_drop": margin_drop,
        "confidence_drop": max(c0 - c1, 0.0),
        "confidence_t": c0,
        "confidence_t1": c1,
        "road_energy": road * road / max(total * total, 1e-12),
        "road_norm": road,
        "residual_norm": orth,
        "delta_h_norm": total,
        "progress": float(step),
        "adv_useful": int(kind.startswith("attack") and final_success and margin_drop > 0.0 and road > 1e-9),
    }
    return rec, coeff


def pgd_trajectory(model, x0, y, u, args, image_id, label):
    eps = args.eps / 255.0
    alpha = args.pgd_step / 255.0
    torch.manual_seed(args.seed + image_id * 101)
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
    xs = [x.detach()]
    for _ in range(args.steps):
        x.requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        grad = torch.autograd.grad(loss, x)[0]
        x = project_linf(x.detach() + alpha * grad.detach().sign(), x0, eps)
        xs.append(x.detach())
    final_success = state(model, xs[-1], y)[3]
    rows, coeffs = [], []
    for t in range(1, len(xs)):
        rec, c = transition_record(model, xs[t - 1], xs[t], y, u, "attack_pgd", image_id, label, t, final_success)
        rows.append(rec)
        coeffs.append(c)
    return rows, coeffs


def square_trajectory(model, x0, y, u, args, image_id, label):
    eps = args.eps / 255.0
    rng = np.random.default_rng(args.seed + image_id * 103)
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
    accepted = [x.detach()]
    with torch.no_grad():
        best_loss = float(F.cross_entropy(model(x), y).item())
    for q in range(1, args.square_queries + 1):
        size = max(1, int(round(32 * (1 - (q - 1) / max(args.square_queries, 1)) ** 0.5)))
        i = int(rng.integers(0, 33 - size))
        j = int(rng.integers(0, 33 - size))
        cand = x.clone()
        sign = -1.0 if rng.random() < 0.5 else 1.0
        cand[:, :, i : i + size, j : j + size] = x0[:, :, i : i + size, j : j + size] + sign * eps
        cand = project_linf(cand, x0, eps)
        with torch.no_grad():
            logits = model(cand)
            loss = float(F.cross_entropy(logits, y).item())
            succ = int(logits.argmax(1).item()) != int(y.item())
        if loss > best_loss or succ:
            x = cand.detach()
            best_loss = max(best_loss, loss)
            accepted.append(x)
        if succ:
            break
    final_success = state(model, accepted[-1], y)[3]
    rows, coeffs = [], []
    for t in range(1, len(accepted)):
        rec, c = transition_record(model, accepted[t - 1], accepted[t], y, u, "attack_square", image_id, label, t, final_success)
        rows.append(rec)
        coeffs.append(c)
    return rows, coeffs


def random_walk(model, x0, y, u, args, image_id, label, kind: str, basis: np.ndarray | None):
    eps = args.eps / 255.0
    rng = torch.Generator(device=x0.device).manual_seed(args.seed + image_id * (107 if basis is None else 109))
    basis_t = torch.as_tensor(basis.T, dtype=torch.float32, device=x0.device) if basis is not None else None
    x = x0.detach()
    rows, coeffs = [], []
    for t in range(1, args.steps + 1):
        if basis_t is None:
            direction = torch.where(torch.rand(x.shape, generator=rng, device=x.device) < 0.5, -torch.ones_like(x), torch.ones_like(x))
        else:
            w = torch.randn((basis_t.shape[1],), generator=rng, device=x.device)
            target = basis_t @ w
            xr = x.detach().requires_grad_(True)
            objective = (feature(model, xr) * target[None, :]).sum()
            grad = torch.autograd.grad(objective, xr)[0]
            direction = grad.sign()
        x_next = project_linf(x.detach() + (args.pgd_step / 255.0) * direction, x0, eps)
        rec, c = transition_record(model, x, x_next, y, u, kind, image_id, label, t, 0)
        rows.append(rec)
        coeffs.append(c)
        x = x_next.detach()
    return rows, coeffs


def build_feature_matrix(meta: pd.DataFrame, coeffs: np.ndarray, k: int, include_margin: bool) -> tuple[np.ndarray, list[str]]:
    base_cols = ["road_energy", "road_norm", "residual_norm", "delta_h_norm", "confidence_drop", "progress"]
    if include_margin:
        base_cols.append("margin_drop")
    x = [coeffs[:, :k]]
    names = [f"a{i}" for i in range(k)]
    x.append(meta[base_cols].to_numpy(np.float32))
    names += base_cols
    return np.concatenate(x, axis=1), names


def split_by_image(meta: pd.DataFrame, seed: int):
    ids = meta.image_id.drop_duplicates().to_numpy()
    tr, te = train_test_split(ids, test_size=0.4, random_state=seed)
    return meta.image_id.isin(tr).to_numpy(), meta.image_id.isin(te).to_numpy()


def safe_auc(y, s):
    return np.nan if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def safe_ap(y, s):
    return np.nan if len(np.unique(y)) < 2 else float(average_precision_score(y, s))


def tpr_at(y, s, target):
    if len(np.unique(y)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= target].max()) if (fpr <= target).any() else 0.0


def fit_eval_models(meta: pd.DataFrame, coeffs: np.ndarray, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask, test_mask = split_by_image(meta, args.seed)
    # Positives for fitting: adversarially useful transitions. Negatives: benign transitions.
    y_fit = meta.adv_useful.astype(int).to_numpy()
    benign = meta.kind.str.startswith("benign").to_numpy()
    fit_mask = train_mask & ((y_fit == 1) | benign)
    eval_mask = test_mask & (meta.kind.str.startswith("attack") | benign)
    rows = []
    scored = meta.loc[eval_mask].copy().reset_index(drop=True)
    eval_y_attack = scored.kind.str.startswith("attack").astype(int).to_numpy()
    eval_y_useful = scored.adv_useful.astype(int).to_numpy()
    for include_margin in [False, True]:
        x, names = build_feature_matrix(meta, coeffs, args.basis_k, include_margin)
        scaler = StandardScaler().fit(x[fit_mask])
        xz = scaler.transform(x)
        pos = fit_mask & (y_fit == 1)
        neg = fit_mask & benign
        n_pos = int(pos.sum())
        n_neg = int(neg.sum())
        n = min(n_pos, n_neg)
        rng = np.random.default_rng(args.seed + int(include_margin))
        if n < max(args.gmm_components * 2, 10):
            continue
        pos_idx = rng.choice(np.where(pos)[0], size=n, replace=False)
        neg_idx = rng.choice(np.where(neg)[0], size=n, replace=False)
        gmm_pos = GaussianMixture(n_components=args.gmm_components, covariance_type="diag", random_state=args.seed).fit(xz[pos_idx])
        gmm_neg = GaussianMixture(n_components=args.gmm_components, covariance_type="diag", random_state=args.seed + 1).fit(xz[neg_idx])
        llr = gmm_pos.score_samples(xz[eval_mask]) - gmm_neg.score_samples(xz[eval_mask])
        clf_idx = np.r_[pos_idx, neg_idx]
        clf_y = np.r_[np.ones(len(pos_idx)), np.zeros(len(neg_idx))]
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(xz[clf_idx], clf_y)
        prob = clf.predict_proba(xz[eval_mask])[:, 1]
        suffix = "with_margin" if include_margin else "no_margin"
        scored[f"gmm_llr_{suffix}"] = llr
        scored[f"logreg_{suffix}"] = prob
        for target_name, y_eval in [("attack_vs_benign", eval_y_attack), ("useful_vs_other", eval_y_useful)]:
            for score_name, vals in [
                (f"gmm_llr_{suffix}", llr),
                (f"logreg_{suffix}", prob),
            ]:
                rows.append(
                    {
                        "target": target_name,
                        "score": score_name,
                        "include_margin": include_margin,
                        "n_fit_pos": n_pos,
                        "n_fit_neg": n_neg,
                        "AUROC": safe_auc(y_eval, vals),
                        "AUPRC": safe_ap(y_eval, vals),
                        "TPR@1FPR": tpr_at(y_eval, vals, 0.01),
                        "TPR@5FPR": tpr_at(y_eval, vals, 0.05),
                    }
                )
    # Baselines.
    for target_name, y_eval in [("attack_vs_benign", eval_y_attack), ("useful_vs_other", eval_y_useful)]:
        for score_name in ["margin_drop", "road_energy", "road_norm", "score_ERD"]:
            if score_name == "score_ERD":
                vals = (scored.road_energy * scored.road_norm * scored.margin_drop).to_numpy(float)
            else:
                vals = scored[score_name].to_numpy(float)
            rows.append(
                {
                    "target": target_name,
                    "score": score_name,
                    "include_margin": score_name in {"margin_drop", "score_ERD"},
                    "n_fit_pos": np.nan,
                    "n_fit_neg": np.nan,
                    "AUROC": safe_auc(y_eval, vals),
                    "AUPRC": safe_ap(y_eval, vals),
                    "TPR@1FPR": tpr_at(y_eval, vals, 0.01),
                    "TPR@5FPR": tpr_at(y_eval, vals, 0.05),
                }
            )
    return scored, pd.DataFrame(rows)


def flag_before_success(scored: pd.DataFrame, score_cols: list[str], fpr: float) -> pd.DataFrame:
    controls = scored[scored.kind.str.startswith("benign")]
    rows = []
    for score in score_cols:
        if score not in scored.columns:
            continue
        threshold = float(np.quantile(controls[score].dropna(), 1 - fpr))
        for attack in ["attack_pgd", "attack_square"]:
            for image_id, g in scored[scored.kind == attack].sort_values("step").groupby("image_id"):
                hit = g[g[score] > threshold]
                succ = g[g.success_now.astype(int) == 1]
                first_hit = int(hit.step.min()) if not hit.empty else -1
                first_succ = int(succ.step.min()) if not succ.empty else -1
                rows.append(
                    {
                        "score": score,
                        "attack": attack,
                        "fpr": fpr,
                        "image_id": int(image_id),
                        "first_flag_step": first_hit,
                        "first_success_step": first_succ,
                        "flag_before_success": int(first_hit > 0 and (first_succ < 0 or first_hit <= first_succ)),
                    }
                )
    return pd.DataFrame(rows)


def plot(scored: pd.DataFrame, out: Path) -> None:
    cols = [c for c in ["gmm_llr_no_margin", "gmm_llr_with_margin", "logreg_no_margin", "margin_drop"] if c in scored.columns]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 3.2), constrained_layout=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        for kind in ["attack_pgd", "attack_square", "benign_random", "benign_mobility"]:
            sub = scored[scored.kind == kind]
            if len(sub):
                ax.hist(sub[col], bins=50, alpha=0.4, density=True, label=kind)
        ax.set_title(col)
        ax.legend(fontsize=7)
    fig.savefig(out / "learned_vector_distribution_scores.png", dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/adversarial_vector_distribution_resnet50_quick")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--n-calib", type=int, default=300)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--basis-k", type=int, default=20)
    p.add_argument("--calib-dirs", type=int, default=64)
    p.add_argument("--basis-select", type=int, default=2000)
    p.add_argument("--gmm-components", type=int, default=4)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--pgd-step", type=float, default=2.0)
    p.add_argument("--square-queries", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    train = datasets.CIFAR10(args.dataset_root, train=True, download=False, transform=transforms.ToTensor())
    test = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    calib = select_clean_correct(train, model, args.n_calib, device)
    eval_rows = select_clean_correct(test, model, args.n_eval, device)
    calib.to_csv(out / "calib_clean_correct.csv", index=False)
    eval_rows.to_csv(out / "eval_clean_correct.csv", index=False)

    road_args = argparse.Namespace(
        calib_eps=args.eps,
        calib_dirs=args.calib_dirs,
        batch_size=128,
        seed=args.seed,
        basis_select=args.basis_select,
        k=args.basis_k,
    )
    bases = fit_candidate_bases(model, train, calib, np.zeros(2048, dtype=np.float32), road_args, device)
    adv_basis = orthonormal_rows(bases["adv_road"], args.basis_k)
    mobility_basis = orthonormal_rows(bases["mobility_only"], args.basis_k)
    np.savez(out / "adversarial_vector_bases.npz", adv_road=adv_basis, mobility_only=mobility_basis)
    u = torch.as_tensor(adv_basis.T, dtype=torch.float32, device=device)

    meta_rows, coeff_rows = [], []
    for row in eval_rows.itertuples(index=False):
        x0, _ = test[int(row.dataset_idx)]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        for rows, coeffs in [
            pgd_trajectory(model, x0, y, u, args, int(row.dataset_idx), int(row.label)),
            square_trajectory(model, x0, y, u, args, int(row.dataset_idx), int(row.label)),
            random_walk(model, x0, y, u, args, int(row.dataset_idx), int(row.label), "benign_random", None),
            random_walk(model, x0, y, u, args, int(row.dataset_idx), int(row.label), "benign_mobility", mobility_basis),
        ]:
            meta_rows.extend(rows)
            coeff_rows.extend(coeffs)
    meta = pd.DataFrame(meta_rows)
    coeffs = np.stack(coeff_rows).astype(np.float32)
    meta.to_csv(out / "transition_metadata.csv", index=False)
    np.savez_compressed(out / "transition_coefficients.npz", coeffs=coeffs)

    scored, summary = fit_eval_models(meta, coeffs, args)
    scored.to_csv(out / "scored_transitions.csv", index=False)
    summary.to_csv(out / "summary_distribution_detection.csv", index=False)
    flags = pd.concat(
        [
            flag_before_success(scored, ["gmm_llr_no_margin", "gmm_llr_with_margin", "logreg_no_margin", "logreg_with_margin", "margin_drop"], 0.01),
            flag_before_success(scored, ["gmm_llr_no_margin", "gmm_llr_with_margin", "logreg_no_margin", "logreg_with_margin", "margin_drop"], 0.05),
        ],
        ignore_index=True,
    )
    flags.to_csv(out / "summary_flag_before_success.csv", index=False)
    plot(scored, out)
    (out / "metadata.json").write_text(json.dumps(vars(args) | {"device": str(device)}, indent=2))

    print("Distribution detection:", flush=True)
    print(summary.sort_values(["target", "AUROC"], ascending=[True, False]).to_string(index=False), flush=True)
    print("\nFlag before success:", flush=True)
    print(flags.groupby(["attack", "score", "fpr"]).flag_before_success.mean().reset_index().to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
