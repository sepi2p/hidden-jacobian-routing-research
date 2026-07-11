#!/usr/bin/env python3
"""Exact K&O-style clean-start comparator for Q1 reviewer validation.

This is the first promoted implementation of reviewer item 1.  It compares
top hidden-Jacobian right-singular directions against margin/gradient selection
and successful-trajectory transport projection features on the exact CIFAR
splits.

Scope note: this script is the clean-start K&O comparator.  It uses the exact
K=20, 12 JVP/VJP iterations, sign/alpha grid, split seeds, and held-out final
test split.  Successful transport PCs are fitted from PGD20 basis-fit
trajectories.  APGD/Square trajectory transport bases are left for the later
full-trajectory collector; this artifact should therefore be read as Step 1A,
not the entire Step 1 closeout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import load_model, margin, project_linf  # noqa: E402


PRESPECIFIED_PRELOGIT = {
    "bbb_resnet50": "avgpool",
    "bbb_vgg19_bn": "penultimate",
    "bbb_densenet": "penultimate",
    "bbb_inception_v3": "penultimate",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv(text: str, typ=float) -> list:
    return [typ(x.strip()) for x in text.split(",") if x.strip()]


def normalize_l2(v: torch.Tensor) -> torch.Tensor:
    return v / v.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)


def orthogonalize(v: torch.Tensor, basis: list[torch.Tensor]) -> torch.Tensor:
    out = v
    for b in basis:
        coeff = (out.flatten(1) * b.flatten(1)).sum(dim=1).view(-1, 1, 1, 1)
        out = out - coeff * b
    return out


def dlr_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_sorted, ind_sorted = logits.sort(dim=1)
    true = logits.gather(1, y[:, None]).squeeze(1)
    top1 = x_sorted[:, -1]
    top2 = x_sorted[:, -2]
    top3 = x_sorted[:, -3]
    top_other = torch.where(ind_sorted[:, -1] == y, top2, top1)
    return -(true - top_other) / (top1 - top3 + 1e-12)


def pca_basis(x: np.ndarray, max_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    xc = x.astype(np.float32) - mean
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s.astype(np.float64) ** 2
    ratio = var / np.clip(var.sum(), 1e-12, None)
    return mean, vt[: min(max_k, vt.shape[0])].astype(np.float32), ratio


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, basis.shape[0])
    xc = x.astype(np.float32) - mean.astype(np.float32)
    coeff = xc @ basis[:kk].T
    denom = np.sum(xc * xc, axis=1)
    return np.sum(coeff[:, :kk] ** 2, axis=1) / np.clip(denom, 1e-12, None)


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    y = np.asarray(y, dtype=int)[ok]
    s = np.asarray(s, dtype=float)[ok]
    if len(y) < 4 or len(np.unique(y)) < 2 or np.nanstd(s) < 1e-12:
        return np.nan
    return float(roc_auc_score(y, s))


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    y = np.asarray(y, dtype=int)[ok]
    s = np.asarray(s, dtype=float)[ok]
    if len(y) < 4 or len(np.unique(y)) < 2 or np.nanstd(s) < 1e-12:
        return np.nan
    return float(average_precision_score(y, s))


def feature_tensor(wrapper, x: torch.Tensor, layer: str) -> torch.Tensor:
    logits, feats, _raw = wrapper.forward_with_features(x)
    if layer not in feats:
        raise RuntimeError(f"Layer {layer} not captured. Available: {list(feats)}")
    return feats[layer].flatten(1)


def feature_numpy(wrapper, x: torch.Tensor, layer: str) -> tuple[torch.Tensor, np.ndarray]:
    with torch.no_grad():
        logits, feats, _raw = wrapper.forward_with_features(x)
    if layer not in feats:
        raise RuntimeError(f"Layer {layer} not captured.")
    return logits.detach(), feats[layer].detach().cpu().numpy().astype(np.float32)


def estimate_topk_right_singular(
    wrapper,
    x: torch.Tensor,
    layer: str,
    k: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> tuple[list[torch.Tensor], list[float], list[np.ndarray]]:
    """Deflated power iteration for top-k right singular directions of J_l(x)."""

    def f(inp: torch.Tensor) -> torch.Tensor:
        return feature_tensor(wrapper, inp, layer)

    dirs: list[torch.Tensor] = []
    sigmas: list[float] = []
    jvps: list[np.ndarray] = []
    gen = torch.Generator(device=x.device).manual_seed(seed)
    for rank in range(k):
        v = torch.randn(x.shape, generator=gen, device=x.device)
        v = normalize_l2(orthogonalize(v, dirs))
        prev_sigma = None
        sigma = float("nan")
        last_jv = None
        for it in range(max_iter):
            x_req = x.detach().requires_grad_(True)
            _h0, jv = torch.autograd.functional.jvp(f, x_req, v, create_graph=False, strict=False)
            h = f(x_req)
            dot = (h * jv.detach()).sum()
            w = torch.autograd.grad(dot, x_req)[0].detach()
            w = orthogonalize(w, dirs)
            v = normalize_l2(w)
            with torch.no_grad():
                _h1, jv_eval = torch.autograd.functional.jvp(f, x.detach(), v, create_graph=False, strict=False)
                sigma = float(jv_eval.flatten(1).norm(dim=1).item())
                last_jv = jv_eval.detach().cpu().numpy().reshape(-1).astype(np.float32)
            if prev_sigma is not None:
                rel = abs(sigma - prev_sigma) / max(abs(prev_sigma), 1e-12)
                if rel < tol and it >= 1:
                    break
            prev_sigma = sigma
        dirs.append(v.detach())
        sigmas.append(float(sigma))
        assert last_jv is not None
        jvps.append(last_jv)
    return dirs, sigmas, jvps


def input_grads(wrapper, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_req = x.detach().requires_grad_(True)
    logits = wrapper(x_req)
    ce = F.cross_entropy(logits, y)
    ce_grad = torch.autograd.grad(ce, x_req, retain_graph=True)[0].detach()
    dlr = dlr_loss(logits, y).sum()
    dlr_grad = torch.autograd.grad(dlr, x_req)[0].detach()
    return ce_grad, dlr_grad


def cosine_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.flatten().float()
    bb = b.flatten().float()
    return float((aa * bb).sum().item() / max(float(aa.norm().item() * bb.norm().item()), 1e-12))


def pgd20_states(wrapper, x0: torch.Tensor, y: torch.Tensor, eps: float, steps: int, step_size: float) -> list[torch.Tensor]:
    x = x0.detach().clone()
    states = [x.detach().clone()]
    for _ in range(steps):
        x_req = x.detach().requires_grad_(True)
        logits = wrapper(x_req)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_req)[0]
        x = project_linf(x + step_size * grad.sign(), x0, eps).detach()
        states.append(x.detach().clone())
    return states


def fit_pgd20_transport_basis(args, wrapper, dataset, split_df: pd.DataFrame, layer: str, device: torch.device):
    out = Path(args.output_dir)
    if args.basis_cache_dir:
        basis_dir = Path(args.basis_cache_dir)
    elif args.max_basis_images < 0:
        # Candidate seeds share the same model/split/layer PGD20 transport
        # basis.  Store it one level above candidate_seed_* directories so the
        # resumable queue does not refit the same basis five times.
        basis_dir = out.parent / "shared_transport_bases"
    else:
        # Smoke/debug runs should remain self-contained.
        basis_dir = out
    basis_dir.mkdir(parents=True, exist_ok=True)
    out_basis = basis_dir / f"transport_basis_{layer}.npz"
    out_rows = basis_dir / f"transport_basis_{layer}_rows.csv"
    if not out_basis.exists() and args.max_basis_images < 0 and not args.overwrite_basis:
        # If candidate_seed_0 was launched before shared caching was added, it
        # may have written the basis locally.  Promote that sibling basis into
        # the shared cache for later candidate seeds.
        for sibling in sorted(out.parent.glob("candidate_seed_*/" + f"transport_basis_{layer}.npz")):
            if sibling == out_basis:
                continue
            shutil.copy2(sibling, out_basis)
            sibling_rows = sibling.with_name(f"transport_basis_{layer}_rows.csv")
            if sibling_rows.exists():
                shutil.copy2(sibling_rows, out_rows)
            break
    if out_basis.exists() and not args.overwrite_basis:
        z = np.load(out_basis)
        return z["mean"], z["basis"], z["explained_variance"], int(z["n_vectors"])

    eps = args.attack_eps / 255.0
    step_size = args.pgd20_step_size / 255.0
    vectors = []
    basis_rows = []
    basis_df = split_df[split_df.split == "basis_fit"].sort_values(["label", "class_ord"])
    if args.max_basis_images > 0:
        basis_df = basis_df.head(args.max_basis_images)
    for n_done, r in enumerate(basis_df.itertuples(index=False), start=1):
        x_cpu, y0 = dataset[int(r.dataset_idx)]
        label = int(r.label)
        if int(y0) != label:
            raise RuntimeError(f"Label mismatch idx={r.dataset_idx}")
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        states = pgd20_states(wrapper, x0, y, eps, args.pgd20_steps, step_size)
        final_logits, _ = feature_numpy(wrapper, states[-1], layer)
        final_pred = int(final_logits.argmax(1).item())
        success = int(final_pred != label)
        feats = [feature_numpy(wrapper, st, layer)[1][0] for st in states]
        if success:
            for t in range(len(feats) - 1):
                vectors.append((feats[t + 1] - feats[t]).astype(np.float32))
                basis_rows.append(
                    {
                        "image_ord": int(r.image_ord),
                        "dataset_idx": int(r.dataset_idx),
                        "label": label,
                        "step": t,
                        "final_pred": final_pred,
                    }
                )
        if n_done % args.progress_every == 0:
            print(f"[basis {args.model} seed={args.split_seed} layer={layer}] {n_done}/{len(basis_df)}", flush=True)
    if len(vectors) < 2:
        raise RuntimeError(f"Not enough successful PGD20 transport vectors for {args.model} seed={args.split_seed} layer={layer}")
    x = np.stack(vectors).astype(np.float32)
    mean, basis, ratio = pca_basis(x, args.max_k)
    np.savez_compressed(out_basis, mean=mean, basis=basis, explained_variance=ratio[: args.max_k], n_vectors=np.asarray([len(vectors)]))
    pd.DataFrame(basis_rows).to_csv(out_rows, index=False)
    return mean, basis, ratio[: args.max_k], len(vectors)


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def completed_images(path: Path, layer: str) -> set[int]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["layer", "image_ord"])
    except Exception:
        return set()
    return set(df[df.layer == layer].image_ord.astype(int).unique().tolist())


def subspace_overlap(hidden_jvps: list[np.ndarray], basis: np.ndarray, k: int) -> tuple[float, float]:
    if len(hidden_jvps) == 0 or basis.size == 0:
        return np.nan, np.nan
    a = np.stack(hidden_jvps).astype(np.float32)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = basis[: min(k, basis.shape[0])].astype(np.float32)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    kk = min(a.shape[0], b.shape[0])
    s = np.linalg.svd(a[:kk] @ b[:kk].T, compute_uv=False)
    s = np.clip(s, 0, 1)
    return float(np.sum(s * s) / kk), float(np.degrees(np.arccos(s)).mean())


def evaluate_layer(args, wrapper, dataset, split_df: pd.DataFrame, layer: str, layer_rule: str, device: torch.device):
    out = Path(args.output_dir)
    score_path = out / f"ko_candidate_scores_{layer_rule}_{layer}.csv"
    done_path = out / f"ko_candidate_scores_{layer_rule}_{layer}.done"
    if done_path.exists() and not args.overwrite:
        print(f"[SKIP] {done_path}")
        return

    mean, basis, ratio, n_basis_vectors = fit_pgd20_transport_basis(args, wrapper, dataset, split_df, layer, device)
    final_df = split_df[split_df.split == "final_test"].sort_values(["label", "class_ord"]).copy()
    if args.max_images > 0:
        final_df = final_df.head(args.max_images)
    done = completed_images(score_path, layer)
    eps = args.attack_eps / 255.0
    alphas = [a / 255.0 for a in parse_csv(args.alpha_grid, float)]
    k = args.k
    for n_done, r in enumerate(final_df.itertuples(index=False), start=1):
        if int(r.image_ord) in done:
            continue
        x_cpu, y0 = dataset[int(r.dataset_idx)]
        label = int(r.label)
        if int(y0) != label:
            raise RuntimeError(f"Label mismatch idx={r.dataset_idx}")
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        clean_logits, h0_np = feature_numpy(wrapper, x0, layer)
        clean_pred = int(clean_logits.argmax(1).item())
        if clean_pred != label:
            raise RuntimeError(f"Split image not clean-correct idx={r.dataset_idx}")
        clean_margin = float(margin(clean_logits, y).item())
        ce_grad, dlr_grad = input_grads(wrapper, x0, y)
        dirs, sigmas, hidden_jvps = estimate_topk_right_singular(
            wrapper,
            x0,
            layer,
            k,
            args.power_iters,
            args.tol,
            args.candidate_seed + int(r.dataset_idx) * 1009 + args.split_seed,
        )
        overlap, mean_angle = subspace_overlap(hidden_jvps, basis, k)
        rows = []
        for rank, (v, sigma, hidden_jvp) in enumerate(zip(dirs, sigmas, hidden_jvps), start=1):
            signed_base = v.sign()
            for sign in [-1, 1]:
                direction = sign * signed_base
                ce_cos = cosine_torch(direction, ce_grad)
                dlr_cos = cosine_torch(direction, dlr_grad)
                for alpha_255, alpha in zip(parse_csv(args.alpha_grid, float), alphas):
                    cand = project_linf(x0 + alpha * direction, x0, eps).detach()
                    logits, hc_np = feature_numpy(wrapper, cand, layer)
                    pred = int(logits.argmax(1).item())
                    cand_margin = float(margin(logits, y).item())
                    disp = (hc_np[0] - h0_np[0]).astype(np.float32)
                    pe = float(projection_energy(disp[None, :], mean, basis, k)[0])
                    rows.append(
                        {
                            "model": args.model,
                            "split_seed": args.split_seed,
                            "candidate_seed": args.candidate_seed,
                            "layer_rule": layer_rule,
                            "layer": layer,
                            "image_ord": int(r.image_ord),
                            "dataset_idx": int(r.dataset_idx),
                            "label": label,
                            "rank": rank,
                            "sign": sign,
                            "alpha_255": float(alpha_255),
                            "singular_value": float(sigma),
                            "jvp_gain": float(sigma * alpha),
                            "ce_grad_cos": ce_cos,
                            "dlr_grad_cos": dlr_cos,
                            "clean_margin": clean_margin,
                            "candidate_margin": cand_margin,
                            "margin_drop": clean_margin - cand_margin,
                            "candidate_pred": pred,
                            "candidate_success": int(pred != label),
                            "candidate_mobility": float(np.linalg.norm(disp)),
                            "transport_projection_energy": pe,
                            "transport_basis_vectors": int(n_basis_vectors),
                            "singular_transport_overlap": overlap,
                            "singular_transport_mean_angle_deg": mean_angle,
                        }
                    )
        append_rows(score_path, rows)
        if n_done % args.progress_every == 0:
            print(f"[candidates {args.model} seed={args.split_seed} {layer_rule}:{layer}] {n_done}/{len(final_df)}", flush=True)
    done_path.write_text("done\n", encoding="utf-8")


def summarize_scores(args, layers: list[tuple[str, str]]) -> None:
    out = Path(args.output_dir)
    metric_rows = []
    topk_rows = []
    model_rows = []
    overlap_rows = []
    for layer_rule, layer in layers:
        path = out / f"ko_candidate_scores_{layer_rule}_{layer}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        y = df.candidate_success.to_numpy(dtype=int)
        scores = {
            "singular_value": df.singular_value.to_numpy(float),
            "jvp_gain": df.jvp_gain.to_numpy(float),
            "margin_drop": df.margin_drop.to_numpy(float),
            "ce_grad_cos": df.ce_grad_cos.to_numpy(float),
            "dlr_grad_cos": df.dlr_grad_cos.to_numpy(float),
            "transport_projection_energy": df.transport_projection_energy.to_numpy(float),
            "mobility_margin_positive": df.jvp_gain.to_numpy(float) * np.maximum(df.margin_drop.to_numpy(float), 0),
        }
        for name, score in scores.items():
            metric_rows.append(
                {
                    "model": args.model,
                    "split_seed": args.split_seed,
                    "candidate_seed": args.candidate_seed,
                    "layer_rule": layer_rule,
                    "layer": layer,
                    "score": name,
                    "n_candidates": int(len(df)),
                    "n_positive": int(y.sum()),
                    "auroc": safe_auroc(y, score),
                    "auprc": safe_auprc(y, score),
                }
            )
        for selector, score in scores.items():
            tmp = df.copy()
            tmp["_score"] = score
            for topk in [1, 5, 10]:
                per = []
                for _img, g in tmp.groupby("image_ord"):
                    gg = g.sort_values("_score", ascending=False).head(topk)
                    per.append(int(gg.candidate_success.max()))
                topk_rows.append(
                    {
                        "model": args.model,
                        "split_seed": args.split_seed,
                        "candidate_seed": args.candidate_seed,
                        "layer_rule": layer_rule,
                        "layer": layer,
                        "selector": selector,
                        "topk": topk,
                        "n_images": len(per),
                        "topk_candidate_asr": float(np.mean(per)) if per else np.nan,
                    }
                )
        features = [
            ("M0_singular", ["rank", "singular_value"]),
            ("M1_jvp", ["rank", "singular_value", "jvp_gain"]),
            ("M2_margin", ["rank", "singular_value", "jvp_gain", "margin_drop"]),
            ("M3_gradient", ["rank", "singular_value", "jvp_gain", "margin_drop", "ce_grad_cos", "dlr_grad_cos"]),
            ("M4_transport", ["rank", "singular_value", "jvp_gain", "margin_drop", "ce_grad_cos", "dlr_grad_cos", "transport_projection_energy"]),
        ]
        for name, cols in features:
            ok = np.isfinite(df[cols].to_numpy(float)).all(axis=1)
            yy = y[ok]
            if len(yy) < 10 or len(np.unique(yy)) < 2:
                auprc = np.nan
                auroc = np.nan
            else:
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
                clf.fit(df.loc[ok, cols].to_numpy(float), yy)
                prob = clf.predict_proba(df.loc[ok, cols].to_numpy(float))[:, 1]
                auprc = safe_auprc(yy, prob)
                auroc = safe_auroc(yy, prob)
            model_rows.append(
                {
                    "model": args.model,
                    "split_seed": args.split_seed,
                    "candidate_seed": args.candidate_seed,
                    "layer_rule": layer_rule,
                    "layer": layer,
                    "nested_model": name,
                    "features": ",".join(cols),
                    "auroc": auroc,
                    "auprc": auprc,
                }
            )
        overlap_rows.append(
            {
                "model": args.model,
                "split_seed": args.split_seed,
                "candidate_seed": args.candidate_seed,
                "layer_rule": layer_rule,
                "layer": layer,
                "mean_singular_transport_overlap": float(df.groupby("image_ord").singular_transport_overlap.first().mean()),
                "mean_singular_transport_angle_deg": float(df.groupby("image_ord").singular_transport_mean_angle_deg.first().mean()),
            }
        )
    pd.DataFrame(metric_rows).to_csv(out / "ko_candidate_metric_auroc_auprc.csv", index=False)
    pd.DataFrame(topk_rows).to_csv(out / "ko_topk_candidate_asr.csv", index=False)
    pd.DataFrame(model_rows).to_csv(out / "ko_incremental_models.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(out / "ko_singular_transport_overlap.csv", index=False)


def selected_layers(nested_root: Path, model: str, split_seed: int) -> list[tuple[str, str]]:
    summary = nested_root / model / f"split_seed_{split_seed}" / "nested_layer_selection_summary.csv"
    if not summary.exists():
        raise RuntimeError(f"Missing nested selection summary: {summary}")
    df = pd.read_csv(summary)
    selected = df[df.layer_rule == "nested_selected_nonlogit"].reported_layer.iloc[0]
    prelogit = PRESPECIFIED_PRELOGIT[model]
    layers = [("nested_selected_nonlogit", str(selected))]
    if str(prelogit) != str(selected):
        layers.append(("prespecified_prelogit", str(prelogit)))
    return layers


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-csv", default="analysis_outputs/hidden_jacobian_routing/exact_protocol/cifar_splits/cifar10_exact_splits.csv")
    p.add_argument("--nested-root", default="analysis_outputs/hidden_jacobian_routing/exact_protocol/phase1_nested_layer_selection")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split-seed", type=int, required=True)
    p.add_argument("--candidate-seed", type=int, default=0)
    p.add_argument("--attack-eps", type=float, default=8.0)
    p.add_argument("--pgd20-steps", type=int, default=20)
    p.add_argument("--pgd20-step-size", type=float, default=2.0)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--max-k", type=int, default=20)
    p.add_argument("--power-iters", type=int, default=12)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--alpha-grid", default="1,2,4,6,8")
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--max-basis-images", type=int, default=-1, help="Debug/smoke mode only. Do not use for promoted evidence.")
    p.add_argument("--basis-cache-dir", default="", help="Optional shared cache for PGD20 transport bases.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--overwrite-basis", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.candidate_seed + args.split_seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    wrapper = load_model(args.model, device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    splits = pd.read_csv(args.splits_csv)
    split_df = splits[(splits.model == args.model) & (splits.split_seed == args.split_seed)].copy()
    if split_df.empty:
        raise RuntimeError(f"No exact split rows for model={args.model} seed={args.split_seed}")
    layers = selected_layers(Path(args.nested_root), args.model, args.split_seed)
    metadata = {
        "experiment": "q1_exact_ko_cleanstart_comparator_step1a",
        "model": args.model,
        "split_seed": args.split_seed,
        "candidate_seed": args.candidate_seed,
        "layers": layers,
        "k": args.k,
        "power_iters": args.power_iters,
        "tol": args.tol,
        "alpha_grid_255": parse_csv(args.alpha_grid, float),
        "transport_basis": "PGD20 successful basis-fit local steps",
        "promotable_scope": "Reviewer Step 1A clean-start K&O comparator, not full Step 1 closeout",
        "max_images_debug": args.max_images,
        "max_basis_images_debug": args.max_basis_images,
        "promotable": bool(args.max_images < 0 and args.max_basis_images < 0),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for layer_rule, layer in layers:
        evaluate_layer(args, wrapper, dataset, split_df, layer, layer_rule, device)
    summarize_scores(args, layers)
    (out / "DONE").write_text("done\n", encoding="utf-8")
    print(f"[DONE] {args.model} split_seed={args.split_seed} candidate_seed={args.candidate_seed} output={out}")


if __name__ == "__main__":
    main()
