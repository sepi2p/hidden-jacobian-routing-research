#!/usr/bin/env python3
"""Efficiency ablation for hidden-Jacobian road routing.

This evaluates whether the expensive top-k hidden-Jacobian road estimate needs
to be recomputed from scratch at every attack step.  The attack recomputes the
top-k local road directions every m steps and reuses them between refreshes.
Both signs of every cached road direction are tested by true-class margin, as in
the original road-routing attack.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.evaluate_k2_margin_selected_roads_by_layer import (  # noqa: E402
    CIFAR10_CLASSES,
    ResNet50LayerFeatures,
    estimate_topk_for_layer,
)
from experiments.hidden_jacobian_routing.evaluate_margin_selected_singular_road_on_balanced import (  # noqa: E402
    logits_stats,
)
from experiments.hidden_jacobian_routing.common import project_linf  # noqa: E402
from experiments.hidden_jacobian_routing.trace_jacobian_singular_roads import set_seed  # noqa: E402
from utils.load_models import load_cifar_model  # noqa: E402


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def bootstrap_delta(a: np.ndarray, b: np.ndarray, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        vals.append(float((a[idx] - b[idx]).mean()))
    lo, med, hi = np.percentile(vals, [2.5, 50, 97.5])
    return float((a - b).mean()), float(lo), float(med), float(hi)


def cached_road_attack(
    model,
    x0: torch.Tensor,
    y: torch.Tensor,
    *,
    layer: str,
    eps: float,
    steps: int,
    step_size: float,
    k: int,
    power_iters: int,
    recompute_every: int,
    seed: int,
):
    x = x0.detach().clone()
    dirs: list[torch.Tensor] = []
    sigmas: list[float] = []
    per_step = []
    recomputes = 0
    for t in range(steps):
        pred0, m0, ce0, p0 = logits_stats(model, x, y)
        if t % recompute_every == 0 or not dirs:
            dirs, sigmas = estimate_topk_for_layer(model, x, layer, k, power_iters, seed + 1009 * t)
            recomputes += 1
        best = None
        for rank, (v, sigma) in enumerate(zip(dirs, sigmas), start=1):
            for sign in (1, -1):
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
        x = best["x"]
        per_step.append(
            {
                "step": t,
                "recomputed": int(t % recompute_every == 0 or t == 0),
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
    return x, per_step, recomputes


def load_balanced_rows(args):
    bdir = Path(args.balanced_dir)
    meta = json.loads((bdir / "metadata.json").read_text())
    image_outcomes = pd.read_csv(bdir / "image_outcomes.csv")
    pgd_ref = (
        image_outcomes[image_outcomes["source"] == "pgd"]
        .drop_duplicates("image_ord")
        .sort_values("image_ord")
        .reset_index(drop=True)
    )
    if args.max_images > 0:
        pgd_ref = pgd_ref.iloc[: args.max_images].copy()
    return pgd_ref, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--balanced-dir",
        default="analysis_outputs/hidden_jacobian_routing/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto",
    )
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/efficient_road_routing_ablation_resnet50_c200")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--steps-list", default="10,20")
    p.add_argument("--power-iters-list", default="1,2,3")
    p.add_argument("--recompute-every-list", default="1,2,4,8")
    p.add_argument("--step-size-linf", type=float, default=1.0)
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pgd_ref, meta = load_balanced_rows(args)
    eps = float(meta["pgd_eps"]) / 255.0
    step_size = float(args.step_size_linf) / 255.0
    steps_list = parse_ints(args.steps_list)
    power_list = parse_ints(args.power_iters_list)
    recompute_list = parse_ints(args.recompute_every_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50LayerFeatures(load_cifar_model("bbb_resnet50").to(device).eval()).to(device).eval()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())

    # Reference PGD success from the balanced artifact on the same rows.
    ref = pgd_ref[["image_ord", "dataset_idx", "label", "final_success"]].rename(columns={"final_success": "pgd_success"})
    all_summary = []
    all_ci = []
    all_overlap = []

    for steps in steps_list:
        for power_iters in power_list:
            for recompute_every in recompute_list:
                tag = f"s{steps}_p{power_iters}_r{recompute_every}"
                per_image_path = out / f"efficient_road_per_image_{tag}.csv"
                per_step_path = out / f"efficient_road_per_step_{tag}.csv"
                if per_image_path.exists() and not args.overwrite:
                    per_image = pd.read_csv(per_image_path)
                    per_step = pd.read_csv(per_step_path) if per_step_path.exists() else pd.DataFrame()
                    print(f"[skip] {tag}", flush=True)
                else:
                    t0 = time.perf_counter()
                    per_image_rows = []
                    per_step_rows = []
                    for row_idx, r in enumerate(pgd_ref.itertuples(index=False), start=1):
                        x_raw, y0 = dataset[int(r.dataset_idx)]
                        if int(y0) != int(r.label):
                            raise RuntimeError(f"label mismatch for dataset_idx={r.dataset_idx}")
                        x0 = x_raw.unsqueeze(0).to(device)
                        y = torch.tensor([int(r.label)], device=device)
                        clean_pred, clean_margin, clean_ce, clean_p = logits_stats(model, x0, y)
                        if clean_pred != int(r.label):
                            raise RuntimeError(f"image {r.dataset_idx} is not clean-correct")
                        x_adv, step_rows, recomputes = cached_road_attack(
                            model,
                            x0,
                            y,
                            layer=args.layer,
                            eps=eps,
                            steps=steps,
                            step_size=step_size,
                            k=args.k,
                            power_iters=power_iters,
                            recompute_every=recompute_every,
                            seed=args.seed + int(r.image_ord) * 1009,
                        )
                        pred, final_margin, final_ce, final_p = logits_stats(model, x_adv, y)
                        per_image_rows.append(
                            {
                                "tag": tag,
                                "image_ord": int(r.image_ord),
                                "dataset_idx": int(r.dataset_idx),
                                "label": int(r.label),
                                "label_class": CIFAR10_CLASSES[int(r.label)],
                                "clean_margin": clean_margin,
                                "final_pred": pred,
                                "final_class": CIFAR10_CLASSES[pred],
                                "success": int(pred != int(r.label)),
                                "final_margin": final_margin,
                                "margin_drop": clean_margin - final_margin,
                                "recomputes": recomputes,
                                "steps": steps,
                                "power_iters": power_iters,
                                "recompute_every": recompute_every,
                            }
                        )
                        for sr in step_rows:
                            per_step_rows.append(
                                {
                                    "tag": tag,
                                    "image_ord": int(r.image_ord),
                                    "dataset_idx": int(r.dataset_idx),
                                    "label": int(r.label),
                                    "steps": steps,
                                    "power_iters": power_iters,
                                    "recompute_every": recompute_every,
                                    **sr,
                                }
                            )
                        if row_idx % 25 == 0:
                            print(f"[{tag}] {row_idx}/{len(pgd_ref)}", flush=True)
                    runtime_s = time.perf_counter() - t0
                    per_image = pd.DataFrame(per_image_rows)
                    per_image["runtime_s_total"] = runtime_s
                    per_step = pd.DataFrame(per_step_rows)
                    per_image.to_csv(per_image_path, index=False)
                    per_step.to_csv(per_step_path, index=False)
                    print(f"[done] {tag} runtime={runtime_s:.1f}s", flush=True)

                merged = per_image.merge(ref, on=["image_ord", "dataset_idx", "label"], how="left")
                road = merged["success"].to_numpy(dtype=int)
                pgd = merged["pgd_success"].to_numpy(dtype=int)
                md = merged["margin_drop"].to_numpy(dtype=float)
                recomputes = int(merged["recomputes"].iloc[0])
                derivative_est = float(2 * args.k * power_iters * 2 * recomputes)
                candidate_forward_est = float(2 * args.k * steps)
                all_summary.append(
                    {
                        "tag": tag,
                        "n": int(len(merged)),
                        "steps": steps,
                        "power_iters": power_iters,
                        "recompute_every": recompute_every,
                        "recomputes": recomputes,
                        "asr": float(road.mean()),
                        "mean_margin_drop": float(md.mean()),
                        "median_margin_drop": float(np.median(md)),
                        "pgd_asr_reference": float(pgd.mean()),
                        "derivative_equiv_est": derivative_est,
                        "candidate_forward_est": candidate_forward_est,
                        "runtime_s_total": float(merged["runtime_s_total"].iloc[0]) if "runtime_s_total" in merged else math.nan,
                    }
                )
                all_overlap.append(
                    {
                        "tag": tag,
                        "pgd_only": int(((pgd == 1) & (road == 0)).sum()),
                        "road_only": int(((pgd == 0) & (road == 1)).sum()),
                        "both": int(((pgd == 1) & (road == 1)).sum()),
                        "neither": int(((pgd == 0) & (road == 0)).sum()),
                    }
                )
                d, lo, med, hi = bootstrap_delta(road, pgd, args.bootstrap_reps, args.seed + steps + 13 * power_iters + recompute_every)
                all_ci.append({"tag": tag, "metric": "ASR road - balanced PGD", "delta": d, "ci_low": lo, "ci_median": med, "ci_high": hi})

    summary = pd.DataFrame(all_summary).sort_values(["steps", "derivative_equiv_est", "asr"], ascending=[True, True, False])
    overlap = pd.DataFrame(all_overlap)
    ci = pd.DataFrame(all_ci)
    summary.to_csv(out / "efficient_road_summary.csv", index=False)
    overlap.to_csv(out / "efficient_road_overlap.csv", index=False)
    ci.to_csv(out / "efficient_road_bootstrap_ci.csv", index=False)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "balanced_dir": args.balanced_dir,
                "dataset_root": args.dataset_root,
                "model": "bbb_resnet50",
                "layer": args.layer,
                "k": args.k,
                "eps_255": float(meta["pgd_eps"]),
                "step_size_255": args.step_size_linf,
                "steps_list": steps_list,
                "power_iters_list": power_list,
                "recompute_every_list": recompute_list,
                "max_images": args.max_images,
                "seed": args.seed,
                "device": str(device),
                "derivative_estimate": "2 * k * power_iters * 2 * recomputes: approximate JVP+VJP derivative ops for each refreshed top-k basis.",
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
