#!/usr/bin/env python3
"""Mine images that remain correctly classified after stronger PGD sweeps.

The goal is not to define a new attack, but to identify genuinely hard PGD
cases for follow-up routing diagnostics.  The script uses the same CIFAR image
cohort as the balanced Jacobian-null run when an outcome CSV is provided.
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
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import project_linf, select_clean_correct  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_float(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_int(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.view(-1, 1), -1e9)
    return true - masked.max(1).values


@torch.no_grad()
def eval_state(model, x: torch.Tensor, y: torch.Tensor) -> dict:
    logits = model(x)
    pred = int(logits.argmax(1).item())
    return {
        "pred": pred,
        "success": int(pred != int(y.item())),
        "margin": float(margin(logits, y).item()),
        "p_y": float(torch.softmax(logits, dim=1).gather(1, y.view(-1, 1)).item()),
    }


def pgd_restarted(
    model,
    x0: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    restarts: int,
    seed: int,
) -> dict:
    gen = torch.Generator(device=x0.device).manual_seed(seed)
    best = eval_state(model, x0, y)
    best.update({"first_success_step": -1, "restart": -1, "linf": 0.0})
    total_evals = 0
    for restart in range(restarts):
        if restart == 0:
            x = x0.detach()
        else:
            noise = torch.empty(x0.shape, device=x0.device).uniform_(-eps, eps, generator=gen)
            x = project_linf(x0 + noise, x0, eps).detach()
        for step in range(steps):
            probe = x.detach().requires_grad_(True)
            logits = model(probe)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, probe)[0]
            x = project_linf(x + step_size * grad.sign(), x0, eps).detach()
            ev = eval_state(model, x, y)
            total_evals += 1
            if ev["margin"] < best["margin"]:
                best = {
                    **ev,
                    "first_success_step": step + 1 if ev["success"] else -1,
                    "restart": restart,
                    "linf": float((x - x0).abs().max().item()),
                }
            if ev["success"]:
                best.update({"evals": total_evals})
                return best
    best.update({"evals": total_evals})
    return best


def load_cohort(args, dataset, model, device) -> pd.DataFrame:
    if args.cohort_csv:
        df = pd.read_csv(args.cohort_csv)
        if "source" in df.columns:
            df = df[df.source == "pgd"]
        cols = ["image_ord", "dataset_idx", "label"]
        return df[cols].drop_duplicates().sort_values("image_ord").head(args.images).reset_index(drop=True)
    selected = select_clean_correct(dataset, {args.model: model}, argparse.Namespace(models=[args.model], images=args.images), device)
    return pd.DataFrame(
        [
            {"image_ord": i, "dataset_idx": int(dataset_idx), "label": int(label)}
            for i, (dataset_idx, label) in enumerate(selected)
        ]
    )


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        rows.groupby(["eps_255", "steps", "step_size_255", "restarts"])
        .agg(
            n=("success", "size"),
            asr=("success", "mean"),
            n_success=("success", "sum"),
            mean_margin=("margin", "mean"),
            median_margin=("margin", "median"),
            mean_evals=("evals", "mean"),
        )
        .reset_index()
        .sort_values(["eps_255", "asr", "steps", "step_size_255"], ascending=[True, False, True, True])
    )
    hard_rows = []
    for eps_255, sub in rows.groupby("eps_255"):
        hard = (
            sub.groupby(["image_ord", "dataset_idx", "label"])
            .agg(any_success=("success", "max"), best_margin=("margin", "min"), n_configs=("success", "size"))
            .reset_index()
        )
        hard = hard[hard.any_success == 0].copy()
        hard["eps_255"] = eps_255
        hard_rows.append(hard)
    hard_failures = pd.concat(hard_rows, ignore_index=True) if hard_rows else pd.DataFrame()
    return summary, hard_failures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/absolute_pgd_failures_bbb_resnet50_c200")
    p.add_argument("--cohort-csv", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto/image_outcomes.csv")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--eps-list", default="2")
    p.add_argument("--steps-list", default="10,20,50,100")
    p.add_argument(
        "--step-size-multipliers",
        default="0.25,0.5,1.0",
        help="Step size is multiplier * eps. Multiple values reduce tuning sensitivity.",
    )
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args.model, device).eval()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    cohort = load_cohort(args, dataset, model, device)
    eps_values = parse_csv_float(args.eps_list)
    steps_values = parse_csv_int(args.steps_list)
    multipliers = parse_csv_float(args.step_size_multipliers)
    rows = []
    for i, row in enumerate(cohort.itertuples(index=False), start=1):
        x_cpu, _ = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        clean = eval_state(model, x0, y)
        for eps_255 in eps_values:
            eps = eps_255 / 255.0
            for steps in steps_values:
                for mult in multipliers:
                    step_size = eps * mult
                    ev = pgd_restarted(
                        model,
                        x0,
                        y,
                        eps,
                        steps,
                        step_size,
                        args.restarts,
                        args.seed + 100003 * int(row.image_ord) + 1009 * steps + int(round(1000 * mult)),
                    )
                    rows.append(
                        {
                            "image_ord": int(row.image_ord),
                            "dataset_idx": int(row.dataset_idx),
                            "label": int(row.label),
                            "clean_pred": int(clean["pred"]),
                            "clean_margin": float(clean["margin"]),
                            "eps_255": float(eps_255),
                            "steps": int(steps),
                            "step_size_255": float(step_size * 255.0),
                            "step_size_multiplier": float(mult),
                            "restarts": int(args.restarts),
                            "success": int(ev["success"]),
                            "pred": int(ev["pred"]),
                            "margin": float(ev["margin"]),
                            "p_y": float(ev["p_y"]),
                            "first_success_step": int(ev["first_success_step"]),
                            "restart": int(ev["restart"]),
                            "evals": int(ev["evals"]),
                            "linf": float(ev["linf"]),
                        }
                    )
        if i % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(out_dir / "partial_strong_pgd_grid_per_image.csv", index=False)
            print(f"[{i}/{len(cohort)}] rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    summary, hard = summarize(df)
    df.to_csv(out_dir / "strong_pgd_grid_per_image.csv", index=False)
    summary.to_csv(out_dir / "strong_pgd_grid_summary.csv", index=False)
    hard.to_csv(out_dir / "absolute_pgd_hard_failures.csv", index=False)
    metadata = vars(args).copy()
    metadata.update({"device": str(device), "n_images": int(len(cohort))})
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lines = ["# Absolute PGD Failure Mining", "", "## Summary", ""]
    for r in summary.itertuples(index=False):
        lines.append(
            f"- eps={r.eps_255:g}/255, steps={r.steps}, step={r.step_size_255:.3g}/255, "
            f"restarts={r.restarts}: ASR={r.asr:.3f}, failures={int(r.n - r.n_success)}/{int(r.n)}"
        )
    lines += ["", "## Hard Failures Across All Tested Configs", ""]
    for eps_255, sub in hard.groupby("eps_255") if not hard.empty else []:
        lines.append(f"- eps={eps_255:g}/255: {len(sub)} hard failures")
    if hard.empty:
        lines.append("- none")
    (out_dir / "absolute_pgd_failure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print("\nHard failures:")
    print(hard.to_string(index=False) if not hard.empty else "none", flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
