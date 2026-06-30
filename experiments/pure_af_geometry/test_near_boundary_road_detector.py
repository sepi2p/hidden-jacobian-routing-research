#!/usr/bin/env python3
"""Near-boundary static adversarial-road detector test.

This follows the second detector experiment:

* compare PGD prefix / epsilon-sweep / boundary-bisected / Square-prefix states;
* compare against genuinely hard clean examples;
* test road-specific scores: excess over random bases, percentile among random
  bases, and excess over clean-motion basis.

The script is a pilot diagnostic.  It should not be used as a defense claim
without detector-aware adaptive attacks.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
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
from experiments.pure_af_geometry.test_single_image_road_detector import (  # noqa: E402
    fit_clean_stats,
    full_hidden_maha_stats,
    logits_margin_conf,
    score_full_hidden,
    score_with_stats,
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


def select_clean_correct_scanned(dataset, model, n_needed: int, device: torch.device, max_scan: int) -> pd.DataFrame:
    rows = []
    for idx in range(min(len(dataset), max_scan)):
        x, y0 = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([int(y0)], device=device)
        with torch.no_grad():
            logits = model(x)
        if int(logits.argmax(1).item()) == int(y0):
            rows.append({"dataset_idx": idx, "label": int(y0), "clean_margin": float(margin(logits, y).item())})
        if len(rows) >= n_needed:
            break
    return pd.DataFrame(rows)


def batch_features(dataset, rows: pd.DataFrame, model, device: torch.device, batch_size: int):
    feats, labels = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start : start + batch_size]
        xs = torch.stack([dataset[int(r.dataset_idx)][0] for r in batch.itertuples(index=False)]).to(device)
        with torch.no_grad():
            feats.append(feature(model, xs).cpu().numpy())
        labels.extend([int(r.label) for r in batch.itertuples(index=False)])
    return np.concatenate(feats, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int64)


def pgd_prefixes(model, x0, y, eps: float, steps: int, step_size: float, checkpoints: set[int], seed: int):
    torch.manual_seed(seed)
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
    out = {}
    for t in range(1, steps + 1):
        x.requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        grad = torch.autograd.grad(loss, x)[0]
        x = project_linf(x.detach() + step_size * grad.detach().sign(), x0, eps)
        if t in checkpoints:
            out[t] = x.detach().clone()
    return out


def square_prefixes(model, x0, y, eps: float, queries: int, checkpoints: set[int], seed: int):
    rng = np.random.default_rng(seed)
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
    with torch.no_grad():
        best_loss = float(F.cross_entropy(model(x), y).item())
    out = {}
    for q in range(1, queries + 1):
        size = max(1, int(round(32 * (1 - (q - 1) / max(queries, 1)) ** 0.5)))
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
            x = cand
            best_loss = max(best_loss, loss)
        if q in checkpoints:
            out[q] = x.detach().clone()
    return out


def true_margin(model, x, y) -> float:
    with torch.no_grad():
        return float(margin(model(x), y).item())


def bisect_margin_states(model, x0, xadv, y, targets: list[float], iters: int = 18):
    out = {}
    m0 = true_margin(model, x0, y)
    m1 = true_margin(model, xadv, y)
    if m1 > min(targets):
        return out
    for target in targets:
        if not (m1 <= target <= m0):
            continue
        lo, hi = 0.0, 1.0
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            xm = (x0 + mid * (xadv - x0)).clamp(0, 1)
            mm = true_margin(model, xm, y)
            if mm > target:
                lo = mid
            else:
                hi = mid
        out[target] = (x0 + hi * (xadv - x0)).clamp(0, 1).detach()
    return out


def benign_aug(x: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=g, device=x.device) * (2 / 255.0)
    scale = 0.9 + 0.2 * torch.rand((x.shape[0], 1, 1, 1), generator=g, device=x.device)
    return (x * scale + noise).clamp(0, 1)


def score_row(model, x, y, label_i: int, image_id: int, kind: str, meta: dict, stats: dict, random_stats: list[dict]):
    logits, pred, conf, m_true, pred_margin = logits_margin_conf(model, x, y)
    pred_i = int(pred.item())
    with torch.no_grad():
        h = feature(model, x).cpu().numpy()[0].astype(np.float32)
    rec = {
        "image_id": image_id,
        "kind": kind,
        "true_label": label_i,
        "pred_label": pred_i,
        "attack_success": int(pred_i != label_i and kind != "clean" and kind != "hard_clean" and kind != "benign_aug"),
        "current_margin": float(m_true.item()),
        "confidence": float(conf.item()),
        "neg_confidence": float(-conf.item()),
        "neg_true_margin": float(-m_true.item()),
        "neg_pred_margin": float(-pred_margin.item()),
        **meta,
    }
    rec.update(score_with_stats(h, pred_i, label_i, stats["road"], "road"))
    rec.update(score_with_stats(h, pred_i, label_i, stats["clean_basis"], "clean_basis"))
    rec.update(score_full_hidden(h, pred_i, label_i, stats["full_hidden"]))
    # Many random bases for specificity.
    rand_mahas = []
    for rs in random_stats:
        rand_mahas.append(score_with_stats(h, pred_i, label_i, rs, "rand")["rand_maha_min"])
    rand_mahas = np.asarray(rand_mahas, dtype=np.float32)
    road_min = rec["road_maha_min"]
    rec["random_maha_mean_min"] = float(rand_mahas.mean())
    rec["random_maha_std_min"] = float(rand_mahas.std())
    rec["road_excess_random_min"] = float(road_min - rand_mahas.mean())
    rec["road_percentile_random_min"] = float((road_min > rand_mahas).mean())
    rec["road_excess_clean_min"] = float(road_min - rec["clean_basis_maha_min"])
    return rec


def safe_auroc(y, s):
    return np.nan if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def safe_auprc(y, s):
    return np.nan if len(np.unique(y)) < 2 else float(average_precision_score(y, s))


def tpr_at_fpr(y, s, fpr_target):
    if len(np.unique(y)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def detection_summary(df: pd.DataFrame, controls: pd.DataFrame, kinds: list[str], scores: list[str]) -> pd.DataFrame:
    rows = []
    hard = controls[controls.kind == "hard_clean"]
    benign = controls[controls.kind == "benign_aug"]
    clean = controls[controls.kind.isin(["clean", "hard_clean"])]
    for kind in kinds:
        adv = df[df.kind == kind]
        if adv.empty:
            continue
        sub = pd.concat([clean, adv], ignore_index=True)
        y = (sub.kind == kind).astype(int).to_numpy()
        for s in scores:
            clean_vals = clean[s].to_numpy(float)
            thr5 = np.quantile(clean_vals, 0.95)
            vals = sub[s].to_numpy(float)
            rows.append(
                {
                    "score": s,
                    "kind": kind,
                    "n_control": int(len(clean)),
                    "n_adv": int(len(adv)),
                    "AUROC": safe_auroc(y, vals),
                    "AUPRC": safe_auprc(y, vals),
                    "TPR@1FPR": tpr_at_fpr(y, vals, 0.01),
                    "TPR@5FPR": tpr_at_fpr(y, vals, 0.05),
                    "FPR_hard_clean_at_5FPR": float((hard[s] > thr5).mean()) if len(hard) else np.nan,
                    "FPR_benign_at_5FPR": float((benign[s] > thr5).mean()) if len(benign) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def margin_matched_summary(df: pd.DataFrame, hard_clean: pd.DataFrame, kinds: list[str], scores: list[str]) -> pd.DataFrame:
    rows = []
    for kind in kinds:
        adv = df[df.kind == kind]
        if adv.empty or hard_clean.empty:
            continue
        tol = max(0.5, float(hard_clean.current_margin.std() * 0.25))
        adv_used, clean_used = [], []
        for _, a in adv.iterrows():
            j = (hard_clean.current_margin - a.current_margin).abs().idxmin()
            if abs(float(hard_clean.loc[j, "current_margin"] - a.current_margin)) <= tol:
                adv_used.append(a)
                clean_used.append(hard_clean.loc[j])
        if not adv_used:
            continue
        adf = pd.DataFrame(adv_used)
        cdf = pd.DataFrame(clean_used)
        for s in scores:
            y = np.r_[np.zeros(len(cdf)), np.ones(len(adf))]
            vals = np.r_[cdf[s].to_numpy(float), adf[s].to_numpy(float)]
            rows.append(
                {
                    "score": s,
                    "kind": kind,
                    "matched_pairs": int(len(adf)),
                    "margin_tolerance": tol,
                    "adv_mean": float(adf[s].mean()),
                    "hard_clean_mean": float(cdf[s].mean()),
                    "AUROC_matched": safe_auroc(y, vals),
                }
            )
    return pd.DataFrame(rows)


def logistic_added_value(df: pd.DataFrame, controls: pd.DataFrame, kinds: list[str], out_scores: list[str]) -> pd.DataFrame:
    rows = []
    clean = controls[controls.kind.isin(["clean", "hard_clean"])]
    feature_sets = {
        "margin_only": ["neg_true_margin"],
        "margin_full_hidden": ["neg_true_margin", "full_hidden_maha_min"],
        "margin_full_clean": ["neg_true_margin", "full_hidden_maha_min", "clean_basis_maha_min"],
        "plus_road": ["neg_true_margin", "full_hidden_maha_min", "clean_basis_maha_min", "road_maha_min"],
        "road_excess_only": ["road_excess_random_min", "road_excess_clean_min", "road_percentile_random_min"],
    }
    for kind in kinds:
        adv = df[df.kind == kind]
        if len(adv) < 5:
            continue
        sub = pd.concat([clean, adv], ignore_index=True)
        y = (sub.kind == kind).astype(int).to_numpy()
        try:
            tr, te = train_test_split(np.arange(len(sub)), test_size=0.4, stratify=y, random_state=0)
        except ValueError:
            continue
        for name, feats in feature_sets.items():
            x = sub[feats].to_numpy(float)
            mu, sd = x[tr].mean(axis=0), x[tr].std(axis=0) + 1e-8
            xz = (x - mu) / sd
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(xz[tr], y[tr])
            prob = clf.predict_proba(xz[te])[:, 1]
            rows.append(
                {
                    "model": name,
                    "kind": kind,
                    "AUROC": safe_auroc(y[te], prob),
                    "AUPRC": safe_auprc(y[te], prob),
                    "TPR@5FPR": tpr_at_fpr(y[te], prob, 0.05),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(df: pd.DataFrame, out: Path) -> None:
    scores = ["road_excess_random_min", "road_percentile_random_min", "road_maha_min", "neg_true_margin"]
    kinds = ["hard_clean", "pgd_prefix", "pgd_boundary", "square_prefix"]
    fig, axes = plt.subplots(1, len(scores), figsize=(4.2 * len(scores), 3.2), constrained_layout=True)
    for ax, s in zip(axes, scores):
        for kind in kinds:
            sub = df[df.kind == kind]
            if len(sub):
                ax.hist(sub[s], bins=40, alpha=0.42, density=True, label=kind)
        ax.set_title(s)
        ax.legend(fontsize=7)
    fig.savefig(out / "near_boundary_score_histograms.png", dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/near_boundary_road_detector_resnet50_quick")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--n-calib", type=int, default=500)
    p.add_argument("--n-eval", type=int, default=160)
    p.add_argument("--scan-clean", type=int, default=2000)
    p.add_argument("--hard-clean-quantile", type=float, default=0.10)
    p.add_argument("--basis-k", type=int, default=20)
    p.add_argument("--n-random-bases", type=int, default=50)
    p.add_argument("--calib-dirs", type=int, default=96)
    p.add_argument("--basis-select", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eps-list", default="1,2,4,8")
    p.add_argument("--pgd-prefix-steps", default="1,2,3,5,10,20")
    p.add_argument("--square-prefix-queries", default="10,25,50,100,300")
    p.add_argument("--boundary-margin-targets", default="1.0,0.5,0.25,0.0,-0.25,-0.5,-1.0")
    p.add_argument("--pgd-step", type=float, default=2.0)
    p.add_argument("--ridge", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    train = datasets.CIFAR10(args.dataset_root, train=True, download=False, transform=transforms.ToTensor())
    test = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())

    calib_rows = select_clean_correct_scanned(train, model, args.n_calib, device, max_scan=50000)
    scanned = select_clean_correct_scanned(test, model, max(args.scan_clean, args.n_eval), device, max_scan=10000)
    scanned = scanned.sort_values("clean_margin").reset_index(drop=True)
    n_hard = max(20, int(len(scanned) * args.hard_clean_quantile))
    hard_rows = scanned.head(n_hard).copy().reset_index(drop=True)
    eval_rows = scanned.tail(args.n_eval).sample(n=min(args.n_eval, len(scanned)), random_state=args.seed).reset_index(drop=True)
    calib_rows.to_csv(out / "calib_clean_correct.csv", index=False)
    hard_rows.to_csv(out / "hard_clean_rows.csv", index=False)
    eval_rows.to_csv(out / "eval_rows.csv", index=False)

    calib_feats, calib_labels = batch_features(train, calib_rows, model, device, args.batch_size)
    road_args = argparse.Namespace(
        calib_eps=8.0,
        calib_dirs=args.calib_dirs,
        batch_size=args.batch_size,
        seed=args.seed,
        basis_select=args.basis_select,
        k=args.basis_k,
    )
    bases = fit_candidate_bases(model, train, calib_rows, np.zeros(calib_feats.shape[1], dtype=np.float32), road_args, device)
    adv_basis = orthonormal_rows(bases["adv_road"], args.basis_k)
    clean_basis = orthonormal_rows(bases.get("clean_motion", bases["mobility_only"]), args.basis_k)
    rng = np.random.default_rng(args.seed + 100)
    random_bases = [
        orthonormal_rows(rng.normal(size=(args.basis_k, calib_feats.shape[1])).astype(np.float32), args.basis_k)
        for _ in range(args.n_random_bases)
    ]
    np.savez(out / "detector_bases.npz", adv_road=adv_basis, clean_motion=clean_basis, random_bases=np.stack(random_bases))

    stats = {
        "road": fit_clean_stats(calib_feats, calib_labels, adv_basis, args.ridge),
        "clean_basis": fit_clean_stats(calib_feats, calib_labels, clean_basis, args.ridge),
        "full_hidden": full_hidden_maha_stats(calib_feats, calib_labels, min(50, len(calib_feats) - 1), args.ridge),
    }
    random_stats = [fit_clean_stats(calib_feats, calib_labels, rb, args.ridge) for rb in random_bases]

    records = []
    # Controls: ordinary clean, hard clean, and benign hard-clean transforms.
    for kind, rows in [("clean", eval_rows), ("hard_clean", hard_rows)]:
        for row in rows.itertuples(index=False):
            x, _ = test[int(row.dataset_idx)]
            x = x.unsqueeze(0).to(device)
            y = torch.tensor([int(row.label)], device=device)
            records.append(score_row(model, x, y, int(row.label), int(row.dataset_idx), kind, {"source": "control"}, stats, random_stats))
            if kind == "hard_clean":
                xb = benign_aug(x, args.seed + int(row.dataset_idx))
                records.append(
                    score_row(model, xb, y, int(row.label), int(row.dataset_idx), "benign_aug", {"source": "hard_aug"}, stats, random_stats)
                )

    eps_list = [float(x) / 255.0 for x in args.eps_list.split(",") if x.strip()]
    pgd_steps = {int(x) for x in args.pgd_prefix_steps.split(",") if x.strip()}
    max_pgd = max(pgd_steps)
    sq_steps = {int(x) for x in args.square_prefix_queries.split(",") if x.strip()}
    max_sq = max(sq_steps)
    targets = [float(x) for x in args.boundary_margin_targets.split(",") if x.strip()]
    pgd_step = args.pgd_step / 255.0

    for row in eval_rows.itertuples(index=False):
        x0, _ = test[int(row.dataset_idx)]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        label = int(row.label)
        # PGD prefix and epsilon sweep.
        final_eps8 = None
        for eps in eps_list:
            prefixes = pgd_prefixes(model, x0, y, eps, max_pgd, min(pgd_step, eps), pgd_steps, args.seed + int(row.dataset_idx) * 17 + int(eps * 255))
            for step_num, xp in prefixes.items():
                kind = "pgd_prefix"
                meta = {"source": "pgd", "eps255": eps * 255.0, "checkpoint": step_num}
                records.append(score_row(model, xp, y, label, int(row.dataset_idx), kind, meta, stats, random_stats))
            if abs(eps * 255.0 - 8.0) < 1e-6:
                final_eps8 = prefixes[max_pgd]
        # Boundary-bisected states from PGD20 at eps=8/255.
        if final_eps8 is not None:
            for target, xb in bisect_margin_states(model, x0, final_eps8, y, targets).items():
                records.append(
                    score_row(
                        model,
                        xb,
                        y,
                        label,
                        int(row.dataset_idx),
                        "pgd_boundary",
                        {"source": "pgd_bisect", "eps255": 8.0, "checkpoint": target},
                        stats,
                        random_stats,
                    )
                )
        # Square prefix states.
        sq = square_prefixes(model, x0, y, 8 / 255.0, max_sq, sq_steps, args.seed + int(row.dataset_idx) * 19)
        for q, xs in sq.items():
            records.append(
                score_row(
                    model,
                    xs,
                    y,
                    label,
                    int(row.dataset_idx),
                    "square_prefix",
                    {"source": "square", "eps255": 8.0, "checkpoint": q},
                    stats,
                    random_stats,
                )
            )

    df = pd.DataFrame(records)
    df.to_csv(out / "per_image_scores.csv", index=False)
    scores = [
        "neg_true_margin",
        "neg_confidence",
        "full_hidden_maha_min",
        "clean_basis_maha_min",
        "road_maha_min",
        "road_excess_random_min",
        "road_percentile_random_min",
        "road_excess_clean_min",
    ]
    controls = df[df.kind.isin(["clean", "hard_clean", "benign_aug"])]
    adv_kinds = ["pgd_prefix", "pgd_boundary", "square_prefix"]
    det = detection_summary(df, controls, adv_kinds, scores)
    det.to_csv(out / "summary_detection.csv", index=False)
    mm = margin_matched_summary(df, df[df.kind == "hard_clean"], adv_kinds, scores)
    mm.to_csv(out / "summary_margin_matched.csv", index=False)
    added = logistic_added_value(df, controls, adv_kinds, scores)
    added.to_csv(out / "summary_added_value.csv", index=False)
    plot_summary(df, out)
    (out / "metadata.json").write_text(json.dumps(vars(args) | {"device": str(device)}, indent=2))

    print("Detection summary:", flush=True)
    show = det[det.score.isin(["road_maha_min", "road_excess_random_min", "road_percentile_random_min", "road_excess_clean_min", "full_hidden_maha_min", "neg_true_margin"])]
    print(show.sort_values(["kind", "AUROC"], ascending=[True, False]).to_string(index=False), flush=True)
    print("\nMargin matched:", flush=True)
    print(mm[mm.score.isin(["road_maha_min", "road_excess_random_min", "road_excess_clean_min", "full_hidden_maha_min", "neg_true_margin"])].to_string(index=False), flush=True)
    print("\nAdded value:", flush=True)
    print(added.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
