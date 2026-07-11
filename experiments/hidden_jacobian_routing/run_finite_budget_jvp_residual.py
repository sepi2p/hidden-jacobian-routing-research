#!/usr/bin/env python3
"""Reduced finite-budget trajectory JVP/residual test.

This is the fast highest-priority check before scaling the full reviewer
protocol.  It logs controlled PGD/APGD-style trajectories on exact CIFAR-10
final-test splits and compares each observed hidden step with the local
linearized motion J_h(x_t) Delta x_t.

The APGD-style trajectories are not the official AutoAttack APGD implementation;
they are explicitly labeled as controlled APGD-style projected attacks because
the public AutoAttack API does not expose per-step states.
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
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import load_model, margin, project_linf
from experiments.hidden_jacobian_routing.run_exact_ko_cleanstart_comparator import dlr_loss


DEFAULT_LAYERS = {
    "bbb_resnet50": "avgpool",
    "bbb_vgg19_bn": "block5",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def progress_bin(step: int, n_steps: int) -> str:
    frac = (step + 1) / max(n_steps, 1)
    for hi, name in [(0.2, "0-20%"), (0.4, "20-40%"), (0.6, "40-60%"), (0.8, "60-80%"), (1.01, "80-100%")]:
        if frac <= hi:
            return name
    return "80-100%"


def feature_tensor(wrapper, x: torch.Tensor, layer: str) -> torch.Tensor:
    _logits, feats, _raw = wrapper.forward_with_features(x)
    if layer not in feats:
        raise RuntimeError(f"Layer {layer} not captured; available={list(feats)}")
    return feats[layer]


def attack_objective(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction="none")
    if loss_name == "dlr":
        return dlr_loss(logits, y)
    raise ValueError(f"unknown loss: {loss_name}")


def projected_states(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    loss_name: str,
    adaptive: bool,
) -> list[torch.Tensor]:
    x = x0.detach().clone()
    states = [x.detach().clone()]
    eta = step_size
    best_margin = float(margin(wrapper(x), y).item())
    stale = 0
    for _ in range(steps):
        probe = x.detach().requires_grad_(True)
        logits = wrapper(probe)
        loss = attack_objective(logits, y, loss_name).sum()
        grad = torch.autograd.grad(loss, probe)[0]
        candidate = project_linf(x + eta * grad.sign(), x0, eps).detach()
        if adaptive:
            cand_margin = float(margin(wrapper(candidate), y).item())
            if cand_margin < best_margin:
                best_margin = cand_margin
                stale = 0
            else:
                stale += 1
                if stale >= 5:
                    eta = max(eta * 0.5, eps / 64.0)
                    stale = 0
        x = candidate
        states.append(x.detach().clone())
    return states


def analyze_step(wrapper, xa: torch.Tensor, xb: torch.Tensor, y: torch.Tensor, layer: str) -> dict:
    with torch.no_grad():
        ha = feature_tensor(wrapper, xa, layer).detach()
        hb = feature_tensor(wrapper, xb, layer).detach()
        logits_a = wrapper(xa)
        logits_b = wrapper(xb)
    observed = hb - ha
    dx = xb.detach() - xa.detach()

    def feat(inp: torch.Tensor) -> torch.Tensor:
        return feature_tensor(wrapper, inp, layer)

    _val, jvp = torch.autograd.functional.jvp(feat, xa.detach(), dx, create_graph=False, strict=False)
    jvp = jvp.detach()
    obs_flat = observed.flatten(1)
    jvp_flat = jvp.flatten(1)
    obs_norm = float(obs_flat.norm(dim=1).item())
    jvp_norm = float(jvp_flat.norm(dim=1).item())
    if obs_norm < 1e-12 or jvp_norm < 1e-12:
        cos = np.nan
        residual = np.nan
    else:
        cos = float((obs_flat * jvp_flat).sum().item() / max(obs_norm * jvp_norm, 1e-12))
        residual = float((observed - jvp).flatten(1).norm(dim=1).item() / max(obs_norm, 1e-12))
    ma = float(margin(logits_a, y).item())
    mb = float(margin(logits_b, y).item())
    return {
        "observed_norm": obs_norm,
        "jvp_norm": jvp_norm,
        "dx_linf": float(dx.abs().max().item()),
        "dx_l2": float(dx.flatten(1).norm(dim=1).item()),
        "fd_jvp_cos": cos,
        "residual_ratio": residual,
        "margin_before": ma,
        "margin_after": mb,
        "margin_drop": ma - mb,
        "pred_before": int(logits_a.argmax(1).item()),
        "pred_after": int(logits_b.argmax(1).item()),
    }


def safe_auc(y: np.ndarray, s: np.ndarray, kind: str) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    if len(np.unique(y)) < 2 or len(y) < 4:
        return np.nan
    if kind == "auroc":
        return float(roc_auc_score(y, s))
    return float(average_precision_score(y, s))


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (model, attack, layer, progress), g in df.groupby(["model", "attack", "layer", "progress_bin"], sort=True):
        rows.append(
            {
                "model": model,
                "attack": attack,
                "layer": layer,
                "progress_bin": progress,
                "n_steps": int(len(g)),
                "n_images": int(g.image_ord.nunique()),
                "success_rate": float(g.final_success.mean()),
                "median_fd_jvp_cos": float(g.fd_jvp_cos.median()),
                "mean_fd_jvp_cos": float(g.fd_jvp_cos.mean()),
                "median_residual_ratio": float(g.residual_ratio.median()),
                "mean_residual_ratio": float(g.residual_ratio.mean()),
                "median_margin_drop": float(g.margin_drop.median()),
            }
        )
    summary = pd.DataFrame(rows)
    sep_rows = []
    for (model, attack, layer), g in df.groupby(["model", "attack", "layer"], sort=True):
        y = g.final_success.to_numpy(int)
        sep_rows.extend(
            [
                {
                    "model": model,
                    "attack": attack,
                    "layer": layer,
                    "score": "negative_residual_ratio",
                    "success_vs_failed_auroc": safe_auc(y, -g.residual_ratio.to_numpy(), "auroc"),
                    "success_vs_failed_auprc": safe_auc(y, -g.residual_ratio.to_numpy(), "auprc"),
                },
                {
                    "model": model,
                    "attack": attack,
                    "layer": layer,
                    "score": "fd_jvp_cos",
                    "success_vs_failed_auroc": safe_auc(y, g.fd_jvp_cos.to_numpy(), "auroc"),
                    "success_vs_failed_auprc": safe_auc(y, g.fd_jvp_cos.to_numpy(), "auprc"),
                },
                {
                    "model": model,
                    "attack": attack,
                    "layer": layer,
                    "score": "margin_drop",
                    "success_vs_failed_auroc": safe_auc(y, g.margin_drop.to_numpy(), "auroc"),
                    "success_vs_failed_auprc": safe_auc(y, g.margin_drop.to_numpy(), "auprc"),
                },
            ]
        )
    return summary, pd.DataFrame(sep_rows)


def run_model(args, model_name: str, layer: str) -> None:
    model_dir = args.output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    done = model_dir / "_SUCCESS"
    if done.exists() and not args.force:
        print(f"[skip] {model_name} already complete: {done}", flush=True)
        return
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    wrapper = load_model(model_name, device).eval()
    split_df = pd.read_csv(args.split_csv)
    rows = split_df[
        split_df["model"].eq(model_name)
        & split_df["split_seed"].eq(args.split_seed)
        & split_df["split"].eq("final_test")
    ].sort_values("image_ord")
    if args.images > 0:
        rows = rows.head(args.images)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    eps = args.eps / 255.0
    attack_specs = []
    for attack in [x.strip() for x in args.attacks.split(",") if x.strip()]:
        if attack == "pgd_ce20":
            attack_specs.append((attack, 20, 2.0 / 255.0, "ce", False))
        elif attack == "apgd_style_ce50":
            attack_specs.append((attack, 50, 2.0 / 255.0, "ce", True))
        elif attack == "apgd_style_dlr50":
            attack_specs.append((attack, 50, 2.0 / 255.0, "dlr", True))
        else:
            raise ValueError(f"unknown attack spec: {attack}")

    all_rows = []
    for image_i, row in enumerate(rows.itertuples(index=False), start=1):
        x_cpu, y_dataset = dataset[int(row.dataset_idx)]
        if int(y_dataset) != int(row.label):
            raise RuntimeError(f"Label mismatch for dataset_idx={row.dataset_idx}")
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        for attack_name, steps, step_size, loss_name, adaptive in attack_specs:
            states = projected_states(wrapper, x0, y, eps, steps, step_size, loss_name, adaptive)
            with torch.no_grad():
                final_pred = int(wrapper(states[-1]).argmax(1).item())
            final_success = int(final_pred != int(row.label))
            for step, (xa, xb) in enumerate(zip(states[:-1], states[1:])):
                rec = analyze_step(wrapper, xa, xb, y, layer)
                rec.update(
                    {
                        "model": model_name,
                        "attack": attack_name,
                        "layer": layer,
                        "split_seed": args.split_seed,
                        "image_ord": int(row.image_ord),
                        "dataset_idx": int(row.dataset_idx),
                        "label": int(row.label),
                        "step": int(step),
                        "n_steps": int(steps),
                        "progress_bin": progress_bin(step, steps),
                        "final_pred": final_pred,
                        "final_success": final_success,
                    }
                )
                all_rows.append(rec)
        if image_i % args.progress_every == 0:
            print(f"[{model_name}] {image_i}/{len(rows)} images", flush=True)
            pd.DataFrame(all_rows).to_csv(model_dir / "finite_budget_step_jvp_rows.partial.csv", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(model_dir / "finite_budget_step_jvp_rows.csv", index=False)
    summary, sep = summarize(df)
    summary.to_csv(model_dir / "finite_budget_step_jvp_summary.csv", index=False)
    sep.to_csv(model_dir / "finite_budget_residual_success_separation.csv", index=False)
    metadata = {
        "model": model_name,
        "layer": layer,
        "split_seed": args.split_seed,
        "images": int(len(rows)),
        "eps_over_255": args.eps,
        "attacks": [x[0] for x in attack_specs],
        "note": "APGD-style trajectories are controlled projected attacks with adaptive step reduction; not official AutoAttack APGD trajectories.",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    done.write_text("complete\n")
    print(summary.to_string(index=False), flush=True)


def aggregate_outputs(out_dir: Path) -> None:
    summaries = []
    seps = []
    for model_dir in out_dir.iterdir() if out_dir.exists() else []:
        if not model_dir.is_dir():
            continue
        s = model_dir / "finite_budget_step_jvp_summary.csv"
        e = model_dir / "finite_budget_residual_success_separation.csv"
        if s.exists():
            summaries.append(pd.read_csv(s))
        if e.exists():
            seps.append(pd.read_csv(e))
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(out_dir / "finite_budget_step_jvp_summary_all.csv", index=False)
    if seps:
        pd.concat(seps, ignore_index=True).to_csv(out_dir / "finite_budget_residual_success_separation_all.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/hidden_jacobian_routing/finite_budget_jvp_residual"))
    p.add_argument("--split-csv", type=Path, default=Path("analysis_outputs/hidden_jacobian_routing/exact_protocol/cifar_splits/cifar10_exact_splits.csv"))
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--models", default="bbb_resnet50,bbb_vgg19_bn")
    p.add_argument("--layers", default="", help="Optional comma list aligned with --models; defaults to nested-selected layers.")
    p.add_argument("--split-seed", type=int, default=1001)
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--attacks", default="pgd_ce20,apgd_style_ce50,apgd_style_dlr50")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    if args.layers:
        layers = [x.strip() for x in args.layers.split(",") if x.strip()]
        if len(layers) != len(models):
            raise ValueError("--layers must align with --models")
    else:
        layers = [DEFAULT_LAYERS[m] for m in models]
    for model_name, layer in zip(models, layers):
        run_model(args, model_name, layer)
        aggregate_outputs(args.output_dir)
    aggregate_outputs(args.output_dir)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "models": models,
                "layers": layers,
                "split_seed": args.split_seed,
                "images": args.images,
                "attacks": args.attacks,
                "note": "Reduced finite-budget trajectory JVP/residual gate; scale only if this passes.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
