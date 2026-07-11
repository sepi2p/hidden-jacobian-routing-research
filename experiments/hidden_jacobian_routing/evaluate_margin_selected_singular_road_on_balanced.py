#!/usr/bin/env python3
"""Evaluate margin-selected top-Jacobian roads on a balanced PGD dataset.

For each image from an existing balanced artifact folder, this script traces a
route that repeatedly estimates the top input-space singular direction of the
hidden-feature Jacobian and chooses the sign that most reduces the true-class
margin. It compares the route against the balanced PGD outcome on the same
images and budget.
"""

from __future__ import annotations

import argparse
import json
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

from experiments.hidden_jacobian_routing.common import margin, project_linf  # noqa: E402
from experiments.hidden_jacobian_routing.trace_jacobian_singular_roads import (  # noqa: E402
    estimate_top_direction,
    load_model,
    set_seed,
)


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def logits_stats(model, x, y):
    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(1).item())
        m = float(margin(logits, y).item())
        ce = float(F.cross_entropy(logits, y).item())
        p_true = float(F.softmax(logits, dim=1)[0, int(y.item())].item())
    return pred, m, ce, p_true


def pgd_states(model, x0, y, eps, steps, step_size):
    x = x0.detach().clone()
    states = [x.detach().clone()]
    rows = []
    for t in range(steps):
        pred0, m0, ce0, p0 = logits_stats(model, x, y)
        probe = x.detach().requires_grad_(True)
        loss = F.cross_entropy(model(probe), y)
        grad = torch.autograd.grad(loss, probe)[0]
        x_next = project_linf(x + step_size * grad.sign(), x0, eps).detach()
        pred1, m1, ce1, p1 = logits_stats(model, x_next, y)
        rows.append(
            {
                "step": t,
                "pred_before": pred0,
                "pred_after": pred1,
                "margin_before": m0,
                "margin_after": m1,
                "ce_before": ce0,
                "ce_after": ce1,
                "true_prob_before": p0,
                "true_prob_after": p1,
            }
        )
        states.append(x_next)
        x = x_next
    return states, rows


def margin_selected_road_states(model, x0, y, eps, steps, step_size, power_iters, seed):
    x = x0.detach().clone()
    states = [x.detach().clone()]
    rows = []
    prev_v = None
    for t in range(steps):
        pred0, m0, ce0, p0 = logits_stats(model, x, y)
        v, sigma = estimate_top_direction(model, x, power_iters, prev_v, seed + t)
        if prev_v is not None and float((v.flatten(1) * prev_v.flatten(1)).sum().item()) < 0:
            v = -v

        step = step_size * v.sign()
        xf = project_linf(x + step, x0, eps).detach()
        xb = project_linf(x - step, x0, eps).detach()
        pf, mf, cef, ptrue_f = logits_stats(model, xf, y)
        pb, mb, ceb, ptrue_b = logits_stats(model, xb, y)
        if mf <= mb:
            x_next = xf
            chosen_sign = 1
            pred1, m1, ce1, p1 = pf, mf, cef, ptrue_f
        else:
            x_next = xb
            chosen_sign = -1
            pred1, m1, ce1, p1 = pb, mb, ceb, ptrue_b
            v = -v
        rows.append(
            {
                "step": t,
                "chosen_sign": chosen_sign,
                "sigma1_est": sigma,
                "pred_before": pred0,
                "pred_after": pred1,
                "margin_before": m0,
                "margin_after": m1,
                "ce_before": ce0,
                "ce_after": ce1,
                "true_prob_before": p0,
                "true_prob_after": p1,
                "forward_pred": pf,
                "forward_margin": mf,
                "backward_pred": pb,
                "backward_margin": mb,
            }
        )
        states.append(x_next)
        x = x_next
        prev_v = v.detach()
    return states, rows


def summarize_overlap(df: pd.DataFrame) -> pd.DataFrame:
    pgd = df["pgd_success"].astype(bool)
    road = df["road_success"].astype(bool)
    return pd.DataFrame(
        [
            {
                "pair": "PGD vs margin-selected road",
                "pgd_only": int((pgd & ~road).sum()),
                "road_only": int((road & ~pgd).sum()),
                "both": int((pgd & road).sum()),
                "neither": int((~pgd & ~road).sum()),
                "n": int(len(df)),
                "pgd_asr": float(pgd.mean()),
                "road_asr": float(road.mean()),
                "union_asr": float((pgd | road).mean()),
            }
        ]
    )


def df_text(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--balanced-dir",
        default="analysis_outputs/hidden_jacobian_routing/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto",
    )
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument(
        "--output-dir",
        default="analysis_outputs/hidden_jacobian_routing/margin_selected_singular_road_balanced_resnet50_c200",
    )
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--power-iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bdir = Path(args.balanced_dir)
    meta = json.loads((bdir / "metadata.json").read_text())
    eps = float(meta["pgd_eps"]) / 255.0
    steps = int(meta["pgd_steps"])
    step_size = float(meta["step_size"]) / 255.0

    image_outcomes = pd.read_csv(bdir / "image_outcomes.csv")
    pgd_ref = (
        image_outcomes[image_outcomes["source"] == "pgd"]
        .drop_duplicates("image_ord")
        .sort_values("image_ord")
        .reset_index(drop=True)
    )
    if args.max_images > 0:
        pgd_ref = pgd_ref.iloc[: args.max_images].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())

    per_image = []
    per_step = []
    for row_idx, r in pgd_ref.iterrows():
        dataset_idx = int(r.dataset_idx)
        label = int(r.label)
        x, y0 = dataset[dataset_idx]
        if int(y0) != label:
            raise RuntimeError(f"Label mismatch at dataset_idx={dataset_idx}: artifact={label}, dataset={y0}")
        x0 = x.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        clean_pred, clean_margin, clean_ce, clean_p = logits_stats(model, x0, y)
        if clean_pred != label:
            raise RuntimeError(f"Image {dataset_idx} is no longer clean-correct: pred={clean_pred}, label={label}")

        pgd_states_x, pgd_rows = pgd_states(model, x0, y, eps, steps, step_size)
        road_states_x, road_rows = margin_selected_road_states(
            model,
            x0,
            y,
            eps,
            steps,
            step_size,
            args.power_iters,
            args.seed + int(r.image_ord) * 1009,
        )
        pgd_final_pred, pgd_final_margin, pgd_final_ce, pgd_final_p = logits_stats(model, pgd_states_x[-1], y)
        road_final_pred, road_final_margin, road_final_ce, road_final_p = logits_stats(model, road_states_x[-1], y)
        pgd_success = int(pgd_final_pred != label)
        road_success = int(road_final_pred != label)

        per_image.append(
            {
                "image_ord": int(r.image_ord),
                "dataset_idx": dataset_idx,
                "label": label,
                "label_class": CIFAR10_CLASSES[label],
                "clean_margin": clean_margin,
                "artifact_pgd_success": int(r.final_success),
                "pgd_success": pgd_success,
                "road_success": road_success,
                "pgd_final_pred": pgd_final_pred,
                "pgd_final_class": CIFAR10_CLASSES[pgd_final_pred],
                "road_final_pred": road_final_pred,
                "road_final_class": CIFAR10_CLASSES[road_final_pred],
                "pgd_margin_drop": clean_margin - pgd_final_margin,
                "road_margin_drop": clean_margin - road_final_margin,
                "pgd_final_margin": pgd_final_margin,
                "road_final_margin": road_final_margin,
                "pgd_final_ce": pgd_final_ce,
                "road_final_ce": road_final_ce,
                "same_final_pred": int(pgd_final_pred == road_final_pred),
            }
        )
        for source, rows in [("pgd", pgd_rows), ("margin_selected_road", road_rows)]:
            for rr in rows:
                per_step.append(
                    {
                        "image_ord": int(r.image_ord),
                        "dataset_idx": dataset_idx,
                        "label": label,
                        "source": source,
                        **rr,
                    }
                )
        if (row_idx + 1) % 25 == 0:
            print(f"[{row_idx + 1}/{len(pgd_ref)}] processed", flush=True)

    per_image_df = pd.DataFrame(per_image)
    per_step_df = pd.DataFrame(per_step)
    summary = pd.DataFrame(
        [
            {
                "n": int(len(per_image_df)),
                "pgd_asr": float(per_image_df["pgd_success"].mean()),
                "artifact_pgd_asr": float(per_image_df["artifact_pgd_success"].mean()),
                "road_asr": float(per_image_df["road_success"].mean()),
                "pgd_mean_margin_drop": float(per_image_df["pgd_margin_drop"].mean()),
                "road_mean_margin_drop": float(per_image_df["road_margin_drop"].mean()),
                "pgd_median_margin_drop": float(per_image_df["pgd_margin_drop"].median()),
                "road_median_margin_drop": float(per_image_df["road_margin_drop"].median()),
                "same_final_pred_rate": float(per_image_df["same_final_pred"].mean()),
                "eps_255": float(meta["pgd_eps"]),
                "steps": steps,
                "step_size_255": float(meta["step_size"]),
                "power_iters": int(args.power_iters),
            }
        ]
    )
    overlap = summarize_overlap(per_image_df)

    per_image_df.to_csv(out / "margin_selected_road_vs_pgd_per_image.csv", index=False)
    per_step_df.to_csv(out / "margin_selected_road_vs_pgd_per_step.csv", index=False)
    summary.to_csv(out / "margin_selected_road_vs_pgd_summary.csv", index=False)
    overlap.to_csv(out / "margin_selected_road_vs_pgd_overlap.csv", index=False)

    readme = [
        "# Margin-Selected Singular Road on Balanced PGD Dataset",
        "",
        f"Balanced artifact: `{args.balanced_dir}`",
        f"Images: {len(per_image_df)}",
        f"Budget: eps={meta['pgd_eps']}/255, steps={steps}, step_size={meta['step_size']}/255",
        f"Power iterations per road step: {args.power_iters}",
        "",
        "## Summary",
        "```text",
        df_text(summary),
        "```",
        "",
        "## Success overlap",
        "```text",
        df_text(overlap),
        "```",
        "",
        "Interpretation: the road method uses the top local hidden-Jacobian mobility axis,",
        "but chooses the sign that gives the stronger true-class margin drop at each step.",
    ]
    (out / "README.md").write_text("\n".join(readme))
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "balanced_dir": args.balanced_dir,
                "dataset_root": args.dataset_root,
                "output_dir": args.output_dir,
                "model": meta.get("model"),
                "eps_255": meta["pgd_eps"],
                "steps": steps,
                "step_size_255": meta["step_size"],
                "power_iters": args.power_iters,
                "seed": args.seed,
                "device": str(device),
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False), flush=True)
    print(overlap.to_string(index=False), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
