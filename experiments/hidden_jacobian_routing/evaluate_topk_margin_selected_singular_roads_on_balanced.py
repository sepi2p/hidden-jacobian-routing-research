#!/usr/bin/env python3
"""Evaluate top-k margin-selected hidden-Jacobian roads on a balanced dataset.

At each step, approximate the top-k input-space right singular directions of
the hidden-feature Jacobian. For every direction and both signs, evaluate the
candidate next state and choose the move that gives the largest true-class
margin drop. This tests whether routing over multiple high-mobility roads is
stronger than using only the top road.
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

from experiments.hidden_jacobian_routing.evaluate_margin_selected_singular_road_on_balanced import (  # noqa: E402
    CIFAR10_CLASSES,
    logits_stats,
    pgd_states,
)
from experiments.hidden_jacobian_routing.common import margin, project_linf  # noqa: E402
from experiments.hidden_jacobian_routing.trace_jacobian_singular_roads import feat, load_model, set_seed  # noqa: E402


def normalize_l2(v: torch.Tensor) -> torch.Tensor:
    return v / v.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)


def orthogonalize(v: torch.Tensor, basis: list[torch.Tensor]) -> torch.Tensor:
    out = v
    for b in basis:
        coeff = (out.flatten(1) * b.flatten(1)).sum(dim=1).view(-1, 1, 1, 1)
        out = out - coeff * b
    return out


def estimate_topk_directions(
    model,
    x: torch.Tensor,
    k: int,
    n_iter: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[float]]:
    """Deflated power iteration for top-k directions of J_h(x)^T J_h(x)."""

    def f(inp):
        return feat(model, inp)

    dirs: list[torch.Tensor] = []
    sigmas: list[float] = []
    gen = torch.Generator(device=x.device).manual_seed(seed)
    for i in range(k):
        v = torch.randn(x.shape, generator=gen, device=x.device)
        v = normalize_l2(orthogonalize(v, dirs))
        for _ in range(n_iter):
            x_req = x.detach().requires_grad_(True)
            _h, jv = torch.autograd.functional.jvp(f, x_req, v, create_graph=False, strict=False)
            h = f(x_req)
            dot = (h * jv.detach()).sum()
            w = torch.autograd.grad(dot, x_req)[0]
            w = orthogonalize(w.detach(), dirs)
            v = normalize_l2(w)
        with torch.no_grad():
            _h, jv = torch.autograd.functional.jvp(f, x.detach(), v, create_graph=False, strict=False)
            sigma = float(jv.flatten(1).norm(dim=1).item())
        dirs.append(v.detach())
        sigmas.append(sigma)
    return dirs, sigmas


def topk_margin_selected_states(model, x0, y, eps, steps, step_size, k, power_iters, seed):
    x = x0.detach().clone()
    states = [x.detach().clone()]
    rows = []
    for t in range(steps):
        pred0, m0, ce0, p0 = logits_stats(model, x, y)
        dirs, sigmas = estimate_topk_directions(model, x, k, power_iters, seed + 9973 * t)

        best = None
        for rank, (v, sigma) in enumerate(zip(dirs, sigmas), start=1):
            for sign in [1, -1]:
                cand = project_linf(x + sign * step_size * v.sign(), x0, eps).detach()
                pred, m, ce, ptrue = logits_stats(model, cand, y)
                item = {
                    "rank": rank,
                    "sign": sign,
                    "sigma": sigma,
                    "x": cand,
                    "pred": pred,
                    "margin": m,
                    "ce": ce,
                    "true_prob": ptrue,
                }
                if best is None or item["margin"] < best["margin"]:
                    best = item
        assert best is not None
        x_next = best["x"]
        rows.append(
            {
                "step": t,
                "k": k,
                "chosen_rank": int(best["rank"]),
                "chosen_sign": int(best["sign"]),
                "chosen_sigma": float(best["sigma"]),
                "pred_before": pred0,
                "pred_after": int(best["pred"]),
                "margin_before": m0,
                "margin_after": float(best["margin"]),
                "ce_before": ce0,
                "ce_after": float(best["ce"]),
                "true_prob_before": p0,
                "true_prob_after": float(best["true_prob"]),
            }
        )
        states.append(x_next)
        x = x_next
    return states, rows


def bootstrap_delta(a: np.ndarray, b: np.ndarray, reps: int, seed: int) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        vals.append(float((a[idx] - b[idx]).mean()))
    lo, med, hi = np.percentile(vals, [2.5, 50, 97.5])
    return float((a - b).mean()), float(lo), float(med), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--balanced-dir",
        default="analysis_outputs/hidden_jacobian_routing/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto",
    )
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/topk_margin_selected_singular_roads_balanced_resnet50_c200")
    p.add_argument("--k-values", default="2,5,10")
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--power-iters", type=int, default=4)
    p.add_argument("--bootstrap-reps", type=int, default=5000)
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
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]

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
        x_raw, y0 = dataset[dataset_idx]
        if int(y0) != label:
            raise RuntimeError(f"Label mismatch at dataset_idx={dataset_idx}: artifact={label}, dataset={y0}")
        x0 = x_raw.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        clean_pred, clean_margin, clean_ce, clean_p = logits_stats(model, x0, y)
        if clean_pred != label:
            raise RuntimeError(f"Image {dataset_idx} is not clean-correct: pred={clean_pred}, label={label}")

        pgd_xs, pgd_rows = pgd_states(model, x0, y, eps, steps, step_size)
        pgd_final_pred, pgd_final_margin, pgd_final_ce, pgd_final_p = logits_stats(model, pgd_xs[-1], y)
        pgd_success = int(pgd_final_pred != label)
        base = {
            "image_ord": int(r.image_ord),
            "dataset_idx": dataset_idx,
            "label": label,
            "label_class": CIFAR10_CLASSES[label],
            "clean_margin": clean_margin,
            "pgd_success": pgd_success,
            "pgd_final_pred": pgd_final_pred,
            "pgd_final_class": CIFAR10_CLASSES[pgd_final_pred],
            "pgd_margin_drop": clean_margin - pgd_final_margin,
            "pgd_final_margin": pgd_final_margin,
        }
        for rr in pgd_rows:
            per_step.append({"image_ord": int(r.image_ord), "dataset_idx": dataset_idx, "label": label, "source": "pgd", **rr})

        for k in k_values:
            road_xs, road_rows = topk_margin_selected_states(
                model,
                x0,
                y,
                eps,
                steps,
                step_size,
                k,
                args.power_iters,
                args.seed + int(r.image_ord) * 1009 + k * 37,
            )
            road_final_pred, road_final_margin, road_final_ce, road_final_p = logits_stats(model, road_xs[-1], y)
            row = dict(base)
            row.update(
                {
                    "k": k,
                    "road_success": int(road_final_pred != label),
                    "road_final_pred": road_final_pred,
                    "road_final_class": CIFAR10_CLASSES[road_final_pred],
                    "road_margin_drop": clean_margin - road_final_margin,
                    "road_final_margin": road_final_margin,
                    "same_final_pred": int(road_final_pred == pgd_final_pred),
                }
            )
            per_image.append(row)
            for rr in road_rows:
                per_step.append(
                    {
                        "image_ord": int(r.image_ord),
                        "dataset_idx": dataset_idx,
                        "label": label,
                        "source": f"top{k}_margin_selected_road",
                        **rr,
                    }
                )

        if (row_idx + 1) % 20 == 0:
            print(f"[{row_idx + 1}/{len(pgd_ref)}] processed", flush=True)

    per_image_df = pd.DataFrame(per_image)
    per_step_df = pd.DataFrame(per_step)

    summary_rows = []
    overlap_rows = []
    ci_rows = []
    for k, g in per_image_df.groupby("k"):
        pgd = g["pgd_success"].to_numpy(dtype=int)
        road = g["road_success"].to_numpy(dtype=int)
        pgdm = g["pgd_margin_drop"].to_numpy(dtype=float)
        roadm = g["road_margin_drop"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "k": int(k),
                "n": int(len(g)),
                "pgd_asr": float(pgd.mean()),
                "road_asr": float(road.mean()),
                "pgd_mean_margin_drop": float(pgdm.mean()),
                "road_mean_margin_drop": float(roadm.mean()),
                "pgd_median_margin_drop": float(np.median(pgdm)),
                "road_median_margin_drop": float(np.median(roadm)),
                "same_final_pred_rate": float(g["same_final_pred"].mean()),
            }
        )
        overlap_rows.append(
            {
                "k": int(k),
                "pgd_only": int(((pgd == 1) & (road == 0)).sum()),
                "road_only": int(((pgd == 0) & (road == 1)).sum()),
                "both": int(((pgd == 1) & (road == 1)).sum()),
                "neither": int(((pgd == 0) & (road == 0)).sum()),
                "union_asr": float(((pgd == 1) | (road == 1)).mean()),
            }
        )
        d, lo, med, hi = bootstrap_delta(road, pgd, args.bootstrap_reps, args.seed + int(k))
        ci_rows.append({"k": int(k), "metric": "ASR road - PGD", "delta": d, "ci_low": lo, "ci_median": med, "ci_high": hi})
        d, lo, med, hi = bootstrap_delta(roadm, pgdm, args.bootstrap_reps, args.seed + 1000 + int(k))
        ci_rows.append({"k": int(k), "metric": "margin_drop road - PGD", "delta": d, "ci_low": lo, "ci_median": med, "ci_high": hi})

    summary = pd.DataFrame(summary_rows).sort_values("k")
    overlap = pd.DataFrame(overlap_rows).sort_values("k")
    ci = pd.DataFrame(ci_rows).sort_values(["k", "metric"])
    chosen = (
        per_step_df[per_step_df["source"].str.contains("top", na=False)]
        .groupby(["source", "chosen_rank"])
        .size()
        .reset_index(name="count")
    )
    chosen["fraction"] = chosen.groupby("source")["count"].transform(lambda x: x / x.sum())

    per_image_df.to_csv(out / "topk_margin_selected_roads_per_image.csv", index=False)
    per_step_df.to_csv(out / "topk_margin_selected_roads_per_step.csv", index=False)
    summary.to_csv(out / "topk_margin_selected_roads_summary.csv", index=False)
    overlap.to_csv(out / "topk_margin_selected_roads_overlap.csv", index=False)
    ci.to_csv(out / "topk_margin_selected_roads_bootstrap_ci.csv", index=False)
    chosen.to_csv(out / "topk_margin_selected_roads_chosen_rank_distribution.csv", index=False)

    readme = [
        "# Top-k Margin-Selected Singular Roads",
        "",
        f"Balanced artifact: `{args.balanced_dir}`",
        f"Budget: eps={meta['pgd_eps']}/255, steps={steps}, step_size={meta['step_size']}/255",
        f"k values: {k_values}",
        f"power iterations: {args.power_iters}",
        "",
        "## Summary",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Overlap",
        "```text",
        overlap.to_string(index=False),
        "```",
        "",
        "## Bootstrap CI",
        "```text",
        ci.to_string(index=False),
        "```",
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
                "k_values": k_values,
                "power_iters": args.power_iters,
                "bootstrap_reps": args.bootstrap_reps,
                "seed": args.seed,
                "device": str(device),
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False), flush=True)
    print(overlap.to_string(index=False), flush=True)
    print(ci.to_string(index=False), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
