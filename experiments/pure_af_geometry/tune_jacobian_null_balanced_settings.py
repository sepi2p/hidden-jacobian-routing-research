#!/usr/bin/env python3
"""Tune attack strengths for balanced Jacobian-null follow-up runs.

The main Jacobian-null pilot needs nontrivial successes and failures.  This
script is deliberately lightweight: it measures ASR for PGD and Square over a
small grid without saving feature trajectories.  The selected setting can then
be passed to ``analyze_jacobian_null_response_pilot.py`` for the full controls.
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
from experiments.pure_af_geometry.analyze_cifar_benchmark_optimizer_transport import square_trajectory  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import project_linf, select_clean_correct  # noqa: E402


def parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pgd_success(model, x, y, eps: float, steps: int, step_size: float) -> tuple[int, int]:
    x0 = x.detach()
    x_adv = x0.clone()
    first_success = -1
    for step in range(steps):
        probe = x_adv.detach().requires_grad_(True)
        logits = model(probe)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, probe)[0]
        x_adv = project_linf(x_adv + step_size * grad.sign(), x0, eps).detach()
        with torch.no_grad():
            pred = int(model(x_adv).argmax(1).item())
        if pred != int(y.item()) and first_success < 0:
            first_success = step + 1
    return int(first_success >= 0), first_success


def square_success(model, x, y, eps: float, queries: int, seed: int, p_init: float, init_epochs: int) -> tuple[int, int]:
    # Use two checkpoints: clean and final. The imported implementation still
    # performs the same square-update loop, but avoids storing a long trajectory.
    states = square_trajectory(model, x, y, eps, queries, seed, p_init, init_epochs, n_checkpoints=2)
    with torch.no_grad():
        pred = int(model(states[-1]).argmax(1).item())
    return int(pred != int(y.item())), queries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_setting_tune_bbb_resnet50_c200")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--eps-list", default="0.5,1,1.5,2")
    p.add_argument("--pgd-steps-list", default="1,2,3,5,10")
    p.add_argument("--square-eps-list", default="2,4,6,8")
    p.add_argument("--square-query-list", default="100,250,500,1000,2000")
    p.add_argument(
        "--step-size",
        type=float,
        default=2.0,
        help="Maximum PGD step size in /255 units; actual step is min(this, eps/2).",
    )
    p.add_argument("--square-p-init", type=float, default=0.8)
    p.add_argument("--square-init-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--partial-every", type=int, default=25)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, device)
    tfm = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=args.dataset_root, train=False, download=False, transform=tfm)
    selected = select_clean_correct(dataset, {args.model: model}, argparse.Namespace(models=[args.model], images=args.images), device)
    print(f"[TUNE] model={args.model} clean_correct={len(selected)} device={device}", flush=True)

    eps_values = parse_csv_floats(args.eps_list)
    square_eps_values = parse_csv_floats(args.square_eps_list)
    pgd_steps_values = parse_csv_ints(args.pgd_steps_list)
    square_queries_values = parse_csv_ints(args.square_query_list)

    rows = []
    image_rows = []

    # PGD grid.
    for eps_i in eps_values:
        eps = eps_i / 255.0
        step_size = min(args.step_size / 255.0, eps / 2.0)
        for steps in pgd_steps_values:
            succ = []
            first_steps = []
            for image_ord, (dataset_idx, label) in enumerate(selected):
                x_cpu, _ = dataset[dataset_idx]
                x = x_cpu.unsqueeze(0).to(device)
                y = torch.tensor([label], device=device)
                ok, first = pgd_success(model, x, y, eps, steps, step_size)
                succ.append(ok)
                first_steps.append(first)
                image_rows.append(
                    {
                        "attack": "pgd",
                        "eps_255": eps_i,
                        "step_size_255": step_size * 255.0,
                        "steps_or_queries": steps,
                        "image_ord": image_ord,
                        "dataset_idx": int(dataset_idx),
                        "label": int(label),
                        "success": int(ok),
                        "first_success_step": int(first),
                    }
                )
            rows.append(
                {
                    "attack": "pgd",
                    "eps_255": eps_i,
                    "step_size_255": step_size * 255.0,
                    "steps_or_queries": steps,
                    "asr": float(np.mean(succ)),
                    "n_success": int(np.sum(succ)),
                    "n": int(len(succ)),
                    "mean_first_success": float(np.mean([s for s in first_steps if s >= 0])) if any(s >= 0 for s in first_steps) else np.nan,
                }
            )
            print(
                f"[TUNE] PGD eps={eps_i:g}/255 step={step_size * 255.0:g}/255 steps={steps}: ASR={np.mean(succ):.3f}",
                flush=True,
            )

    # Square grid.
    for eps_i in square_eps_values:
        eps = eps_i / 255.0
        for queries in square_queries_values:
            succ = []
            for image_ord, (dataset_idx, label) in enumerate(selected):
                x_cpu, _ = dataset[dataset_idx]
                x = x_cpu.unsqueeze(0).to(device)
                y = torch.tensor([label], device=device)
                ok, used = square_success(
                    model,
                    x,
                    y,
                    eps,
                    queries,
                    int(args.seed + int(round(100000 * eps_i)) + 1009 * image_ord + queries),
                    args.square_p_init,
                    args.square_init_epochs,
                )
                succ.append(ok)
                image_rows.append(
                    {
                        "attack": "square",
                        "eps_255": eps_i,
                        "step_size_255": np.nan,
                        "steps_or_queries": queries,
                        "image_ord": image_ord,
                        "dataset_idx": int(dataset_idx),
                        "label": int(label),
                        "success": int(ok),
                        "first_success_step": int(used if ok else -1),
                    }
                )
            rows.append(
                {
                    "attack": "square",
                    "eps_255": eps_i,
                    "step_size_255": np.nan,
                    "steps_or_queries": queries,
                    "asr": float(np.mean(succ)),
                    "n_success": int(np.sum(succ)),
                    "n": int(len(succ)),
                    "mean_first_success": float(queries) if any(succ) else np.nan,
                }
            )
            print(f"[TUNE] Square eps={eps_i:g}/255 queries={queries}: ASR={np.mean(succ):.3f}", flush=True)

    summary = pd.DataFrame(rows)
    per_image = pd.DataFrame(image_rows)
    summary.to_csv(out_dir / "balanced_setting_grid.csv", index=False)
    per_image.to_csv(out_dir / "balanced_setting_per_image.csv", index=False)

    candidates = summary[(summary["asr"] >= 0.5) & (summary["asr"] <= 0.8)].copy()
    candidates["distance_to_065"] = (candidates["asr"] - 0.65).abs()
    candidates = candidates.sort_values(["distance_to_065", "attack", "eps_255", "steps_or_queries"])
    candidates.to_csv(out_dir / "balanced_setting_candidates.csv", index=False)

    metadata = {
        "script": "experiments/pure_af_geometry/tune_jacobian_null_balanced_settings.py",
        "model": args.model,
        "images": len(selected),
        "eps_list": eps_values,
        "square_eps_list": square_eps_values,
        "pgd_steps_list": pgd_steps_values,
        "square_query_list": square_queries_values,
        "max_step_size_255": args.step_size,
        "pgd_step_rule": "min(max_step_size_255/255, eps/2)",
        "seed": args.seed,
        "device": str(device),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[DONE] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
