#!/usr/bin/env python3
"""Actual-trajectory JVP linearization diagnostic.

Candidate-direction JVP controls test random feasible proposals.  This script
tests the recorded attack process itself by regenerating the balanced PGD and
Square trajectories and comparing each observed hidden step

    h_l(x_{t+1}) - h_l(x_t)

with the local linearized feature motion

    J_l(x_t) (x_{t+1} - x_t).

The balanced artifact does not store pixel states, so this script regenerates
states from the protocol metadata and writes a standalone linearization table.
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
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import load_model  # noqa: E402
from experiments.hidden_jacobian_routing.common import square_trajectory  # noqa: E402
from experiments.hidden_jacobian_routing.analyze_jacobian_null_response_pilot import pgd_states  # noqa: E402
from experiments.hidden_jacobian_routing.common import margin  # noqa: E402


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


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    if len(y) < 4 or len(np.unique(y)) < 2 or np.std(s) < 1e-12:
        return np.nan
    return float(roc_auc_score(y, s))


def progress_bin(step: int, n_steps: int) -> str:
    frac = step / max(n_steps, 1)
    bins = [(0.2, "0-20%"), (0.4, "20-40%"), (0.6, "40-60%"), (0.8, "60-80%"), (1.01, "80-100%")]
    for hi, name in bins:
        if frac < hi:
            return name
    return "80-100%"


def load_images(input_dir: Path, model: str, split: str, max_images: int) -> pd.DataFrame:
    outcomes = pd.read_csv(input_dir / "image_outcomes.csv")
    splits = pd.read_csv(input_dir / "image_splits.csv")
    base = outcomes[(outcomes.model == model) & (outcomes.source == "pgd")][
        ["image_ord", "dataset_idx", "label", "clean_pred", "clean_margin"]
    ].drop_duplicates()
    base = base.merge(splits, on="image_ord", how="left")
    if split != "all":
        base = base[base.split == split]
    base = base.sort_values("image_ord").reset_index(drop=True)
    if max_images > 0:
        base = base.head(max_images)
    return base


def pgd_rule_jvp(wrapper, xa: torch.Tensor, y: torch.Tensor, layer: str, step_size: float) -> tuple[float, float]:
    probe = xa.detach().requires_grad_(True)
    logits = wrapper(probe)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, probe)[0].detach()
    rule_delta = step_size * grad.sign()

    def feat(inp: torch.Tensor) -> torch.Tensor:
        return feature_tensor(wrapper, inp, layer)

    _val, jvp = torch.autograd.functional.jvp(feat, xa.detach(), rule_delta, create_graph=False, strict=False)
    return float(jvp.norm().item()), 0.0


def analyze_step(wrapper, xa: torch.Tensor, xb: torch.Tensor, y: torch.Tensor, layer: str) -> dict:
    with torch.no_grad():
        ha = feature_tensor(wrapper, xa, layer).detach()
        hb = feature_tensor(wrapper, xb, layer).detach()
    v = hb - ha
    dx = xb.detach() - xa.detach()

    def feat(inp: torch.Tensor) -> torch.Tensor:
        return feature_tensor(wrapper, inp, layer)

    _val, jvp = torch.autograd.functional.jvp(feat, xa.detach(), dx, create_graph=False, strict=False)
    jvp = jvp.detach()
    v_norm = float(v.norm().item())
    jvp_norm = float(jvp.norm().item())
    dx_l2 = float(dx.flatten(1).norm(dim=1).item())
    dx_linf = float(dx.abs().max().item())
    if v_norm < 1e-12 or jvp_norm < 1e-12:
        cos = np.nan
        nonlinear = np.nan
    else:
        cos = float((v.flatten(1) * jvp.flatten(1)).sum().item() / max(v_norm * jvp_norm, 1e-12))
        nonlinear = float((v - jvp).norm().item() / max(v_norm, 1e-12))
    with torch.no_grad():
        logits_a = wrapper(xa)
        logits_b = wrapper(xb)
        pred_a = int(logits_a.argmax(1).item())
        pred_b = int(logits_b.argmax(1).item())
        margin_a = float(margin(logits_a, y).item())
        margin_b = float(margin(logits_b, y).item())
    return {
        "v_norm": v_norm,
        "jvp_norm": jvp_norm,
        "dx_l2": dx_l2,
        "dx_linf": dx_linf,
        "cos_v_jdx": cos,
        "nonlinear_ratio": nonlinear,
        "pred_before": pred_a,
        "pred_after": pred_b,
        "margin_before": margin_a,
        "margin_after": margin_b,
        "margin_drop": margin_a - margin_b,
    }


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (attack, layer, bin_name), g in df.groupby(["attack", "layer", "progress_bin"], sort=True):
        rows.append(
            {
                "attack": attack,
                "layer": layer,
                "progress_bin": bin_name,
                "n_steps": int(len(g)),
                "n_images": int(g.image_ord.nunique()),
                "success_rate": float(g.final_success.mean()),
                "median_cos_v_jdx": float(g.cos_v_jdx.median()),
                "mean_cos_v_jdx": float(g.cos_v_jdx.mean()),
                "median_nonlinear_ratio": float(g.nonlinear_ratio.median()),
                "mean_nonlinear_ratio": float(g.nonlinear_ratio.mean()),
                "median_margin_drop": float(g.margin_drop.median()),
            }
        )
    by_bin = pd.DataFrame(rows)
    sep_rows = []
    for (attack, layer, bin_name), g in df.groupby(["attack", "layer", "progress_bin"], sort=True):
        y = g.final_success.to_numpy(dtype=int)
        sep_rows.append(
            {
                "attack": attack,
                "layer": layer,
                "progress_bin": bin_name,
                "score": "nonlinear_ratio",
                "success_vs_failed_auroc": safe_auroc(y, g.nonlinear_ratio.to_numpy()),
            }
        )
        sep_rows.append(
            {
                "attack": attack,
                "layer": layer,
                "progress_bin": bin_name,
                "score": "negative_cos_v_jdx",
                "success_vs_failed_auroc": safe_auroc(y, -g.cos_v_jdx.to_numpy()),
            }
        )
    return by_bin, pd.DataFrame(sep_rows)


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    order = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=180)
    for attack, g in summary.groupby("attack"):
        gg = g.copy()
        gg["x"] = gg.progress_bin.map({name: i for i, name in enumerate(order)})
        gg = gg.sort_values("x")
        axes[0].plot(gg.x, gg.median_cos_v_jdx, marker="o", label=attack)
        axes[1].plot(gg.x, gg.median_nonlinear_ratio, marker="o", label=attack)
    for ax in axes:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=20)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Median cos(observed, J dx)")
    axes[0].set_title("Local linear agreement")
    axes[1].set_ylabel("Median nonlinear residual ratio")
    axes[1].set_title("Finite-step nonlinearity")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_jvp_linearity_by_progress.png", bbox_inches="tight")
    fig.savefig(out_dir / "trajectory_jvp_linearity_by_progress.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("analysis_outputs/hidden_jacobian_routing/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/hidden_jacobian_routing/jacobian_null_response/actual_trajectory_jvp_linearization_bbb_resnet50"))
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--images", type=int, default=50)
    p.add_argument("--attacks", default="pgd,square")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress-every", type=int, default=10)
    args = p.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    meta = json.loads((args.input_dir / "metadata.json").read_text())
    pgd_eps = float(meta.get("pgd_eps", meta.get("eps", 2.0))) / 255.0
    square_eps = float(meta.get("square_eps", meta.get("eps", 6.0))) / 255.0
    pgd_steps = int(meta.get("pgd_steps", 2))
    square_steps = int(meta.get("square_steps", 250))
    step_size = float(meta.get("step_size", 1.0)) / 255.0
    square_checkpoints = int(meta.get("square_checkpoints", 21))
    square_p_init = float(meta.get("square_p_init", 0.8))
    square_init_epochs = int(meta.get("square_init_epochs", 1))

    wrapper = load_model(args.model, device).eval()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    images = load_images(args.input_dir, args.model, args.split, args.images)
    attacks = [x.strip() for x in args.attacks.split(",") if x.strip()]
    rows = []
    for image_i, row in enumerate(images.itertuples(index=False), start=1):
        x_cpu, _ = dataset[int(row.dataset_idx)]
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        states_by_attack = {}
        if "pgd" in attacks:
            states_by_attack["pgd"] = pgd_states(wrapper, x, y, pgd_eps, pgd_steps, step_size)[0]
        if "square" in attacks:
            states_by_attack["square"] = square_trajectory(
                wrapper,
                x,
                y,
                square_eps,
                square_steps,
                args.seed + int(row.image_ord) * 1009 + 17,
                square_p_init,
                square_init_epochs,
                square_checkpoints,
            )
        for attack, states in states_by_attack.items():
            with torch.no_grad():
                final_logits = wrapper(states[-1])
            final_pred = int(final_logits.argmax(1).item())
            final_success = int(final_pred != int(row.label))
            n_steps = len(states) - 1
            for step, (xa, xb) in enumerate(zip(states[:-1], states[1:])):
                rec = analyze_step(wrapper, xa, xb, y, args.layer)
                rec.update(
                    {
                        "model": args.model,
                        "attack": attack,
                        "layer": args.layer,
                        "image_ord": int(row.image_ord),
                        "dataset_idx": int(row.dataset_idx),
                        "label": int(row.label),
                        "step": int(step),
                        "n_steps": int(n_steps),
                        "progress": float((step + 1) / max(n_steps, 1)),
                        "progress_bin": progress_bin(step, n_steps),
                        "final_pred": final_pred,
                        "final_success": final_success,
                    }
                )
                if attack == "pgd":
                    rule_norm, _ = pgd_rule_jvp(wrapper, xa, y, args.layer, step_size)
                    rec["pgd_rule_jvp_norm"] = rule_norm
                rows.append(rec)
        if image_i % max(1, args.progress_every) == 0:
            print(f"[trajectory-jvp] {image_i}/{len(images)} images", flush=True)
            pd.DataFrame(rows).to_csv(args.output_dir / "trajectory_step_jvp_linearization.partial.csv", index=False)
    if hasattr(wrapper, "close"):
        wrapper.close()
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "trajectory_step_jvp_linearization.csv", index=False)
    summary, sep = summarize(df)
    summary.to_csv(args.output_dir / "summary_attack_step_linearity.csv", index=False)
    sep.to_csv(args.output_dir / "summary_residual_success_separation.csv", index=False)
    plot_summary(summary, args.output_dir)
    out_meta = {
        "model": args.model,
        "layer": args.layer,
        "split": args.split,
        "images": int(len(images)),
        "attacks": attacks,
        "pgd_eps_over_255": pgd_eps * 255.0,
        "pgd_steps": pgd_steps,
        "square_eps_over_255": square_eps * 255.0,
        "square_steps": square_steps,
        "square_checkpoints": square_checkpoints,
        "note": "Square linearization is over saved best-state checkpoints, not every proposed query update.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(out_meta, indent=2))
    lines = [
        "# Actual-Trajectory JVP Linearization Findings",
        "",
        "Square results are computed over regenerated best-state checkpoints, not every proposed query update.",
        "",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"- {r.attack} {r.progress_bin}: median cos={r.median_cos_v_jdx:.3f}, "
            f"median nonlinear ratio={r.median_nonlinear_ratio:.3f}, n={r.n_steps}."
        )
    (args.output_dir / "actual_trajectory_jvp_linearization_findings.md").write_text("\n".join(lines) + "\n")
    print(summary.to_string(index=False))
    print(sep.to_string(index=False))


if __name__ == "__main__":
    main()
