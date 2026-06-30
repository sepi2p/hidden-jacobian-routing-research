#!/usr/bin/env python3
"""Exact JVP control for the mobility proposal signal.

This script asks whether finite-difference representation mobility is just
local hidden-Jacobian gain.  For each feasible random sign direction, it
computes both:

    mobility_fd = ||h(x + delta_probe) - h(x)||_2
    jvp_gain    = ||J_h(x) delta_probe||_2

and then tests whether mobility or learned transport energy add predictive
value beyond probe-margin drop, gradient alignment, and JVP gain.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import load_model  # noqa: E402
from experiments.hidden_jacobian_routing.common import margin  # noqa: E402
from experiments.hidden_jacobian_routing.test_mobility_margin_two_stage_selection import (  # noqa: E402
    fit_segment_basis,
    load_eval_images,
    parse_csv,
    parse_int_csv,
    projection_energy,
    safe_auprc,
    safe_auroc,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature_tensor(wrapper, x: torch.Tensor, layer: str) -> torch.Tensor:
    _logits, feats, _raw = wrapper.forward_with_features(x)
    if layer not in feats:
        raise RuntimeError(f"Layer {layer} not captured.")
    return feats[layer]


def clean_grads(wrapper, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probe = x.detach().requires_grad_(True)
    logits = wrapper(probe)
    ce = F.cross_entropy(logits, y)
    ce_grad = torch.autograd.grad(ce, probe, retain_graph=True)[0].detach()
    m = margin(logits, y).sum()
    margin_grad = torch.autograd.grad(m, probe)[0].detach()
    return ce_grad, margin_grad


def cosine_with_direction(signs: torch.Tensor, direction: torch.Tensor) -> np.ndarray:
    s = signs.flatten(1).float()
    d = direction.flatten(1).float()
    return ((s * d).sum(dim=1) / (s.norm(dim=1).clamp_min(1e-12) * d.norm(dim=1).clamp_min(1e-12))).detach().cpu().numpy()


def corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    sx = pd.Series(x[ok])
    sy = pd.Series(y[ok])
    if sx.nunique() < 2 or sy.nunique() < 2:
        return np.nan
    return float(sx.corr(sy, method=method))


def within_image_z(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        zc = f"{c}_z"
        out[zc] = out.groupby("image_ord")[c].transform(
            lambda s: (s - s.mean()) / max(float(s.std(ddof=0)), 1e-12)
        )
    out["margin_x_mobility_z"] = out["probe_margin_drop_z"] * out["mobility_fd_z"]
    out["margin_x_jvp_z"] = out["probe_margin_drop_z"] * out["jvp_gain_z"]
    out["margin_x_transport_z"] = out["probe_margin_drop_z"] * out["transport_energy_z"]
    out["margin_x_nonlinear_z"] = out["probe_margin_drop_z"] * out["nonlinear_ratio_z"]
    return out


def split_images(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    image_ids = np.array(sorted(df.image_ord.unique()))
    train_images = set(image_ids[::2])
    train = df.image_ord.isin(train_images).to_numpy()
    return train, ~train


def evaluate_config(args, wrapper, dataset, images: pd.DataFrame, eps: float, alpha: float) -> pd.DataFrame:
    input_dir = Path(args.input_dir)
    flow_mean, flow_basis, _n_flow = fit_segment_basis(
        input_dir,
        args.model,
        args.layer,
        parse_csv(args.flow_basis_sources),
        args.k,
        True,
    )
    probe_eps = eps * alpha / 255.0
    attack_eps = eps / 255.0
    rows = []

    for image_i, row in enumerate(images.itertuples(index=False), start=1):
        x_cpu, _ = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(args.device_obj)
        y = torch.tensor([int(row.label)], device=args.device_obj)
        with torch.no_grad():
            logits0 = wrapper(x0)
            if int(logits0.argmax(1).item()) != int(row.label):
                continue
            clean_margin = float(margin(logits0, y).item())
            clean_py = float(torch.softmax(logits0, dim=1)[0, int(row.label)].item())
            h0_t = feature_tensor(wrapper, x0, args.layer).detach()
            h0 = h0_t.cpu().numpy().astype(np.float32)[0]
        ce_grad, margin_grad = clean_grads(wrapper, x0, y)
        gen = torch.Generator(device=args.device_obj).manual_seed(args.seed + int(row.dataset_idx) * 1009 + int(eps * 100) * 917 + int(alpha * 1000))
        remaining = args.directions_per_image
        direction_id = 0
        while remaining > 0:
            bs = min(args.jvp_batch_size, remaining)
            signs = torch.where(
                torch.rand((bs,) + tuple(x0.shape[1:]), generator=gen, device=args.device_obj) < 0.5,
                -torch.ones((bs,) + tuple(x0.shape[1:]), device=args.device_obj),
                torch.ones((bs,) + tuple(x0.shape[1:]), device=args.device_obj),
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
            jvp = jvp.detach()
            fd = h_probe - h0_t
            fd_np = fd.cpu().numpy().astype(np.float32)
            jvp_np = jvp.cpu().numpy().astype(np.float32)
            fd_norm = np.linalg.norm(fd_np, axis=1)
            jvp_norm = np.linalg.norm(jvp_np, axis=1)
            dot = np.sum(fd_np * jvp_np, axis=1)
            fd_jvp_cos = dot / np.clip(fd_norm * jvp_norm, 1e-12, None)
            nonlinear_ratio = np.linalg.norm(fd_np - jvp_np, axis=1) / np.clip(fd_norm, 1e-12, None)
            transport_energy = projection_energy(fd_np, flow_mean, flow_basis, args.k).astype(np.float32)

            probe_margin = margin(logits_probe, y.expand(bs)).detach().cpu().numpy().astype(np.float32)
            full_margin = margin(logits_full, y.expand(bs)).detach().cpu().numpy().astype(np.float32)
            probe_py = torch.softmax(logits_probe, dim=1)[:, int(row.label)].detach().cpu().numpy().astype(np.float32)
            full_py = torch.softmax(logits_full, dim=1)[:, int(row.label)].detach().cpu().numpy().astype(np.float32)
            pred_full = logits_full.argmax(1).detach().cpu().numpy().astype(np.int64)
            ce_cos = cosine_with_direction(signs, ce_grad.expand_as(signs))
            neg_margin_cos = cosine_with_direction(signs, (-margin_grad).expand_as(signs))

            for j in range(bs):
                rows.append(
                    {
                        "image_ord": int(row.image_ord),
                        "dataset_idx": int(row.dataset_idx),
                        "label": int(row.label),
                        "eps_over_255": float(eps),
                        "alpha": float(alpha),
                        "probe_eps_over_255": float(eps * alpha),
                        "direction_id": int(direction_id + j),
                        "clean_margin": clean_margin,
                        "clean_p_y": clean_py,
                        "probe_margin": float(probe_margin[j]),
                        "full_margin": float(full_margin[j]),
                        "probe_margin_drop": float(clean_margin - probe_margin[j]),
                        "full_margin_drop": float(clean_margin - full_margin[j]),
                        "probe_p_y_drop": float(clean_py - probe_py[j]),
                        "full_p_y_drop": float(clean_py - full_py[j]),
                        "full_pred": int(pred_full[j]),
                        "full_success": int(pred_full[j] != int(row.label)),
                        "mobility_fd": float(fd_norm[j]),
                        "jvp_gain": float(jvp_norm[j]),
                        "fd_jvp_cos": float(fd_jvp_cos[j]),
                        "nonlinear_ratio": float(nonlinear_ratio[j]),
                        "transport_energy": float(transport_energy[j]),
                        "ce_grad_cos": float(ce_cos[j]),
                        "neg_margin_grad_cos": float(neg_margin_cos[j]),
                        "score_margin_x_mobility": float(max(clean_margin - probe_margin[j], 0.0) * fd_norm[j]),
                        "score_margin_x_jvp": float(max(clean_margin - probe_margin[j], 0.0) * jvp_norm[j]),
                        "score_margin_x_transport": float(max(clean_margin - probe_margin[j], 0.0) * transport_energy[j]),
                    }
                )
            remaining -= bs
            direction_id += bs
        if image_i % max(1, args.progress_every) == 0:
            print(f"[progress] eps={eps} alpha={alpha} {image_i}/{len(images)} images", flush=True)
    return pd.DataFrame(rows)


def summarize_linearity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (eps, alpha), g in df.groupby(["eps_over_255", "alpha"], sort=True):
        rows.append(
            {
                "eps_over_255": eps,
                "alpha": alpha,
                "n_candidates": int(len(g)),
                "random_candidate_asr": float(g.full_success.mean()),
                "pearson_mobility_jvp": corr(g.mobility_fd.to_numpy(), g.jvp_gain.to_numpy(), "pearson"),
                "spearman_mobility_jvp": corr(g.mobility_fd.to_numpy(), g.jvp_gain.to_numpy(), "spearman"),
                "median_fd_jvp_cos": float(g.fd_jvp_cos.median()),
                "mean_fd_jvp_cos": float(g.fd_jvp_cos.mean()),
                "median_nonlinear_ratio": float(g.nonlinear_ratio.median()),
                "mean_nonlinear_ratio": float(g.nonlinear_ratio.mean()),
                "mobility_success_auroc": safe_auroc(g.full_success.to_numpy(), g.mobility_fd.to_numpy()),
                "jvp_success_auroc": safe_auroc(g.full_success.to_numpy(), g.jvp_gain.to_numpy()),
                "transport_success_auroc": safe_auroc(g.full_success.to_numpy(), g.transport_energy.to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def train_nested_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "probe_margin_drop",
        "ce_grad_cos",
        "neg_margin_grad_cos",
        "jvp_gain",
        "mobility_fd",
        "transport_energy",
        "nonlinear_ratio",
    ]
    dfz = within_image_z(df, cols)
    specs = {
        "M1_margin": ["probe_margin_drop_z"],
        "M2_margin_grad": ["probe_margin_drop_z", "ce_grad_cos_z", "neg_margin_grad_cos_z"],
        "M3_margin_grad_jvp": ["probe_margin_drop_z", "ce_grad_cos_z", "neg_margin_grad_cos_z", "jvp_gain_z"],
        "M4_plus_mobility": ["probe_margin_drop_z", "ce_grad_cos_z", "neg_margin_grad_cos_z", "jvp_gain_z", "mobility_fd_z"],
        "M5_plus_transport": ["probe_margin_drop_z", "ce_grad_cos_z", "neg_margin_grad_cos_z", "jvp_gain_z", "transport_energy_z"],
        "M6_plus_mobility_transport": [
            "probe_margin_drop_z",
            "ce_grad_cos_z",
            "neg_margin_grad_cos_z",
            "jvp_gain_z",
            "mobility_fd_z",
            "transport_energy_z",
        ],
        "M7_plus_interactions": [
            "probe_margin_drop_z",
            "ce_grad_cos_z",
            "neg_margin_grad_cos_z",
            "jvp_gain_z",
            "mobility_fd_z",
            "transport_energy_z",
            "margin_x_mobility_z",
            "margin_x_jvp_z",
            "margin_x_transport_z",
        ],
        "M8_plus_nonlinear": [
            "probe_margin_drop_z",
            "ce_grad_cos_z",
            "neg_margin_grad_cos_z",
            "jvp_gain_z",
            "nonlinear_ratio_z",
        ],
    }
    score_frames = []
    metric_rows = []
    for (eps, alpha), g in dfz.groupby(["eps_over_255", "alpha"], sort=True):
        train, test = split_images(g)
        y = g.full_success.to_numpy(dtype=int)
        for name, features in specs.items():
            x = g[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
            if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
                continue
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
            )
            clf.fit(x[train], y[train])
            score = np.full(len(g), np.nan, dtype=np.float32)
            score[train] = clf.predict_proba(x[train])[:, 1]
            score[test] = clf.predict_proba(x[test])[:, 1]
            metric_rows.append(
                {
                    "eps_over_255": float(eps),
                    "alpha": float(alpha),
                    "model_name": name,
                    "features": ",".join(features),
                    "train_auc": safe_auroc(y[train], score[train]),
                    "test_auc": safe_auroc(y[test], score[test]),
                    "train_auprc": safe_auprc(y[train], score[train]),
                    "test_auprc": safe_auprc(y[test], score[test]),
                    "n_train_candidates": int(train.sum()),
                    "n_test_candidates": int(test.sum()),
                }
            )
            sf = g[["image_ord", "direction_id", "eps_over_255", "alpha", "full_success", "full_margin_drop"]].copy()
            sf["selector"] = f"learned_{name}"
            sf["score"] = score
            sf["is_test_score"] = test
            score_frames.append(sf)
    return pd.DataFrame(metric_rows), pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()


def raw_selector_scores(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "random": None,
        "mobility_fd": "mobility_fd",
        "jvp_gain": "jvp_gain",
        "margin_drop": "probe_margin_drop",
        "transport_energy": "transport_energy",
        "gradient_cos": "ce_grad_cos",
        "margin_x_mobility": "score_margin_x_mobility",
        "margin_x_jvp": "score_margin_x_jvp",
        "margin_x_transport": "score_margin_x_transport",
        "nonlinear_ratio": "nonlinear_ratio",
    }
    frames = []
    rng = np.random.default_rng(0)
    for name, col in mapping.items():
        sf = df[["image_ord", "direction_id", "eps_over_255", "alpha", "full_success", "full_margin_drop"]].copy()
        sf["selector"] = name
        sf["score"] = rng.random(len(sf)) if col is None else df[col].to_numpy(dtype=float)
        sf["is_test_score"] = True
        frames.append(sf)
    return pd.concat(frames, ignore_index=True)


def summarize_topk(score_df: pd.DataFrame, top_ks: list[int], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    rng = np.random.default_rng(seed + 777)
    test_scores = score_df[score_df.is_test_score.astype(bool)].copy()
    for (eps, alpha, selector, image_ord), g in test_scores.groupby(["eps_over_255", "alpha", "selector", "image_ord"], sort=False):
        for k in top_ks:
            kk = min(int(k), len(g))
            if selector == "random":
                chosen = g.sample(n=kk, random_state=int(rng.integers(0, 2**31 - 1)))
            else:
                chosen = g.sort_values("score", ascending=False).head(kk)
            rows.append(
                {
                    "eps_over_255": float(eps),
                    "alpha": float(alpha),
                    "selector": selector,
                    "image_ord": int(image_ord),
                    "top_k": int(k),
                    "topk_any_success": int(chosen.full_success.astype(int).max()),
                    "topk_precision": float(chosen.full_success.astype(float).mean()),
                    "best_full_margin_drop": float(chosen.full_margin_drop.max()),
                }
            )
    per_image = pd.DataFrame(rows)
    summary = (
        per_image.groupby(["eps_over_255", "alpha", "selector", "top_k"], dropna=False)
        .agg(
            n_images=("image_ord", "nunique"),
            topk_asr=("topk_any_success", "mean"),
            topk_precision=("topk_precision", "mean"),
            mean_best_full_margin_drop=("best_full_margin_drop", "mean"),
        )
        .reset_index()
    )
    return per_image, summary


def plot_summary(linearity: pd.DataFrame, topk: pd.DataFrame, pred: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), dpi=180)
    for eps, g in linearity.groupby("eps_over_255"):
        axes[0].plot(g.alpha, g.spearman_mobility_jvp, marker="o", label=f"eps={eps:g}/255")
    axes[0].set_xlabel("probe fraction alpha")
    axes[0].set_ylabel("Spearman(mobility, JVP gain)")
    axes[0].set_title("Finite difference vs exact JVP")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    keep = topk[(topk.top_k == 10) & (topk.selector.isin(["margin_drop", "margin_x_mobility", "margin_x_jvp", "jvp_gain", "mobility_fd"]))]
    for selector, g in keep.groupby("selector"):
        axes[1].plot(g.eps_over_255 + 0.03 * g.alpha, g.topk_asr, marker="o", label=selector)
    axes[1].set_xlabel("full budget eps/255")
    axes[1].set_ylabel("Top-10 ASR")
    axes[1].set_title("Operational selector comparison")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)

    if not pred.empty and "model_name" in pred.columns:
        p = pred[pred.model_name.isin(["M3_margin_grad_jvp", "M4_plus_mobility", "M5_plus_transport", "M6_plus_mobility_transport"])]
        for name, g in p.groupby("model_name"):
            axes[2].plot(g.eps_over_255 + 0.03 * g.alpha, g.test_auprc, marker="o", label=name)
    axes[2].set_xlabel("full budget eps/255")
    axes[2].set_ylabel("Held-out AUPRC")
    axes[2].set_title("Added value after JVP gain")
    axes[2].grid(alpha=0.25)
    if axes[2].has_data():
        axes[2].legend(frameon=False, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "mobility_vs_jacobian_gain_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "mobility_vs_jacobian_gain_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def write_note(out_dir: Path, linearity: pd.DataFrame, pred: pd.DataFrame, topk: pd.DataFrame, meta: dict) -> None:
    if not pred.empty and "model_name" in pred.columns:
        m3 = pred[pred.model_name == "M3_margin_grad_jvp"].copy()
        m4 = pred[pred.model_name == "M4_plus_mobility"].copy()
        m5 = pred[pred.model_name == "M5_plus_transport"].copy()
        merged = m3[["eps_over_255", "alpha", "test_auprc", "test_auc"]].rename(columns={"test_auprc": "m3_auprc", "test_auc": "m3_auc"})
        merged = merged.merge(m4[["eps_over_255", "alpha", "test_auprc", "test_auc"]].rename(columns={"test_auprc": "m4_auprc", "test_auc": "m4_auc"}), on=["eps_over_255", "alpha"], how="left")
        merged = merged.merge(m5[["eps_over_255", "alpha", "test_auprc", "test_auc"]].rename(columns={"test_auprc": "m5_auprc", "test_auc": "m5_auc"}), on=["eps_over_255", "alpha"], how="left")
        merged["delta_m4_m3_auprc"] = merged.m4_auprc - merged.m3_auprc
        merged["delta_m5_m3_auprc"] = merged.m5_auprc - merged.m3_auprc
    else:
        merged = pd.DataFrame()
    lines = [
        "# Mobility Versus Exact Hidden-Jacobian Gain Findings",
        "",
        "## Setup",
        "",
        f"- Model: `{meta['model']}`.",
        f"- Layer: `{meta['layer']}`.",
        f"- Images: {meta['images_evaluated']} held-out images.",
        f"- Directions per image: {meta['directions_per_image']}.",
        f"- Budgets: {meta['eps_list_over_255']}.",
        f"- Probe fractions: {meta['alpha_list']}.",
        "",
        "## Finite-Difference Mobility Versus JVP Gain",
        "",
    ]
    for r in linearity.itertuples(index=False):
        lines.append(
            f"- eps={r.eps_over_255:g}/255, alpha={r.alpha:g}: "
            f"Spearman={r.spearman_mobility_jvp:.3f}, median cos={r.median_fd_jvp_cos:.3f}, "
            f"median nonlinear ratio={r.median_nonlinear_ratio:.3f}."
        )
    lines += ["", "## Added Predictive Value After JVP", ""]
    if merged.empty:
        lines.append("- Predictive models could not be fit because the smoke/small run lacked class diversity.")
    else:
        for r in merged.itertuples(index=False):
            lines.append(
                f"- eps={r.eps_over_255:g}, alpha={r.alpha:g}: "
                f"M4-M3 AUPRC={r.delta_m4_m3_auprc:+.3f}; "
                f"M5-M3 AUPRC={r.delta_m5_m3_auprc:+.3f}."
            )
    lines += [
        "",
        "## Interpretation Template",
        "",
        "If finite-difference mobility and JVP gain are almost identical, the proposal-bias story should be written as a hidden-Jacobian-gain story. If mobility or transport energy add AUPRC/top-K ASR after JVP gain, that residual is the stronger possible contribution.",
    ]
    (out_dir / "mobility_vs_jacobian_gain_findings.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="analysis_outputs/hidden_jacobian_routing/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/jacobian_null_response/mobility_vs_jacobian_gain_bbb_resnet50")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--directions-per-image", type=int, default=128)
    p.add_argument("--jvp-batch-size", type=int, default=32)
    p.add_argument("--eps-list", default="2,4,8")
    p.add_argument("--alpha-list", default="0.125,0.25,0.5")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--flow-basis-sources", default="pgd,square")
    p.add_argument("--top-ks", default="1,5,10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress-every", type=int, default=10)
    args = p.parse_args()

    set_seed(args.seed)
    args.device_obj = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = load_model(args.model, args.device_obj)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    images = load_eval_images(Path(args.input_dir), args.model, args.images)
    eps_list = [float(x) for x in parse_csv(args.eps_list)]
    alpha_list = [float(x) for x in parse_csv(args.alpha_list)]

    frames = []
    for eps in eps_list:
        for alpha in alpha_list:
            print(f"[run] eps={eps}/255 alpha={alpha}", flush=True)
            frames.append(evaluate_config(args, wrapper, dataset, images, eps, alpha))
            pd.concat(frames, ignore_index=True).to_csv(out_dir / "candidate_level_jvp.partial.csv", index=False)
    if hasattr(wrapper, "close"):
        wrapper.close()
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_dir / "candidate_level_jvp.csv", index=False)
    linearity = summarize_linearity(df)
    linearity.to_csv(out_dir / "summary_fd_vs_jvp.csv", index=False)
    pred, learned_scores = train_nested_models(df)
    pred.to_csv(out_dir / "summary_predictive_nested.csv", index=False)
    score_df = pd.concat([raw_selector_scores(df), learned_scores], ignore_index=True)
    topk_per_image, topk = summarize_topk(score_df, parse_int_csv(args.top_ks), args.seed)
    topk_per_image.to_csv(out_dir / "summary_topk_selectors_per_image.csv", index=False)
    topk.to_csv(out_dir / "summary_topk_selectors.csv", index=False)
    plot_summary(linearity, topk, pred, out_dir)
    meta = {
        "model": args.model,
        "layer": args.layer,
        "images_requested": args.images,
        "images_evaluated": int(df.image_ord.nunique()),
        "directions_per_image": args.directions_per_image,
        "eps_list_over_255": eps_list,
        "alpha_list": alpha_list,
        "k": args.k,
        "flow_basis_sources": parse_csv(args.flow_basis_sources),
        "seed": args.seed,
        "outputs": [
            "candidate_level_jvp.csv",
            "summary_fd_vs_jvp.csv",
            "summary_predictive_nested.csv",
            "summary_topk_selectors.csv",
            "mobility_vs_jacobian_gain_summary.png",
            "mobility_vs_jacobian_gain_findings.md",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    write_note(out_dir, linearity, pred, topk, meta)
    print(linearity.to_string(index=False))
    print(pred.to_string(index=False))
    print(topk.to_string(index=False))


if __name__ == "__main__":
    main()
