#!/usr/bin/env python3
"""Same-harness routing-efficiency validation.

Runs several attacks/proposal mechanisms on the same clean-correct CIFAR-10
images and logs finite-difference hidden mobility for each applied step or
proposal.  This avoids comparing a true hidden-mobility ratio against older
road-router logs that only stored singular-value proxies.
"""

from __future__ import annotations

import argparse
import json
import math
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

from attacks.square import p_selection  # noqa: E402
from experiments.hidden_jacobian_routing.analyze_attack_road_selection_diagnostic import (  # noqa: E402
    eval_state,
    normalize_l2,
    road_context,
    score_move,
    select_clean_correct,
)
from experiments.hidden_jacobian_routing.common import load_model, project_linf  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pgd_ce(wrapper, x0, y, args, image_ord: int) -> list[dict]:
    eps = args.eps / 255.0
    eta = args.pgd_eta / 255.0
    x = x0.detach().clone()
    rows = []
    for t in range(args.pgd_steps):
        ctx = road_context(
            wrapper,
            x,
            y,
            args.layer,
            k=args.k,
            power_iters=args.power_iters,
            eta=args.road_eta / 255.0,
            eps=eps,
            x0=x0,
            seed=args.seed + image_ord * 1009 + t,
        )
        probe = x.detach().requires_grad_(True)
        loss = F.cross_entropy(wrapper(probe), y)
        grad = torch.autograd.grad(loss, probe)[0]
        cand = project_linf(x + eta * grad.sign(), x0, eps).detach()
        rec = score_move(wrapper, args.layer, x, cand, y, ctx)
        rows.append(
            {
                "attack": "pgd_ce",
                "proposal_type": "applied_step",
                "accepted": 1,
                "helpful": int(rec["margin_drop"] > 0),
                "image_ord": image_ord,
                "step": t,
                "query": t + 1,
                **rec,
            }
        )
        x = cand
        if rec["success_after"] and args.stop_on_success:
            break
    return rows


def road_router(wrapper, x0, y, args, image_ord: int) -> list[dict]:
    eps = args.eps / 255.0
    x = x0.detach().clone()
    rows = []
    eta_grid = [float(v) / 255.0 for v in args.road_eta_grid]
    for t in range(args.road_steps):
        ctx = road_context(
            wrapper,
            x,
            y,
            args.layer,
            k=args.k,
            power_iters=args.power_iters,
            eta=args.road_eta / 255.0,
            eps=eps,
            x0=x0,
            seed=args.seed + 30011 + image_ord * 1009 + t,
        )
        best = None
        for rank, v in enumerate(ctx["dirs"], start=1):
            for sign in (1, -1):
                for eta in eta_grid:
                    cand = project_linf(x + sign * eta * v.sign(), x0, eps).detach()
                    ev = eval_state(wrapper, cand, y)
                    item = {"rank": rank, "sign": sign, "eta255": 255.0 * eta, "cand": cand, "margin": ev["margin"]}
                    if best is None or item["margin"] < best["margin"]:
                        best = item
        assert best is not None
        rec = score_move(wrapper, args.layer, x, best["cand"], y, ctx)
        rows.append(
            {
                "attack": "road_topk_margin",
                "proposal_type": "applied_step",
                "accepted": 1,
                "helpful": int(rec["margin_drop"] > 0),
                "image_ord": image_ord,
                "step": t,
                "query": (t + 1) * args.k * 2 * len(eta_grid),
                "chosen_rank": best["rank"],
                "chosen_sign": best["sign"],
                "chosen_eta255": best["eta255"],
                **rec,
            }
        )
        x = best["cand"]
        if rec["success_after"] and args.stop_on_success:
            break
    return rows


def square(wrapper, x0, y, args, image_ord: int) -> list[dict]:
    eps = args.eps / 255.0
    gen = torch.Generator(device=x0.device).manual_seed(args.seed + 17 + image_ord * 1009)
    _, c, h, w = x0.shape
    x = x0.detach().clone()
    best = eval_state(wrapper, x, y)
    rows = []
    ctx = None
    accepted_state_id = 0
    for q in range(args.square_queries):
        if ctx is None:
            ctx = road_context(
                wrapper,
                x,
                y,
                args.layer,
                k=args.k,
                power_iters=args.power_iters,
                eta=args.road_eta / 255.0,
                eps=eps,
                x0=x0,
                seed=args.seed + 200003 + image_ord * 1009 + accepted_state_id,
            )
        perturbation = x - x0
        p = p_selection(args.square_p_init, q + args.square_init_epochs, args.square_queries)
        side = int(round(np.sqrt(p * c * h * w / c)))
        side = min(max(side, 1), h - 1)
        top = int(torch.randint(0, h - side + 1, (1,), generator=gen, device=x0.device).item())
        left = int(torch.randint(0, w - side + 1, (1,), generator=gen, device=x0.device).item())
        patch = torch.where(
            torch.rand((1, c, 1, 1), generator=gen, device=x0.device) < 0.5,
            -torch.ones((1, c, 1, 1), device=x0.device),
            torch.ones((1, c, 1, 1), device=x0.device),
        ) * eps
        perturbation = perturbation.clone()
        perturbation[:, :, top : top + side, left : left + side] = patch
        cand = (x0 + perturbation).clamp(0, 1).detach()
        rec = score_move(wrapper, args.layer, x, cand, y, ctx)
        accepted = int(rec["margin_after"] < best["margin"])
        rows.append(
            {
                "attack": "square",
                "proposal_type": "proposal",
                "accepted": accepted,
                "helpful": int(rec["margin_drop"] > 0),
                "image_ord": image_ord,
                "step": q,
                "query": q + 1,
                "square_side": side,
                **rec,
            }
        )
        if accepted:
            x = cand
            best = eval_state(wrapper, x, y)
            ctx = None
            accepted_state_id += 1
        if best["success"] and args.stop_on_success:
            break
    return rows


def nes(wrapper, x0, y, args, image_ord: int) -> list[dict]:
    eps = args.eps / 255.0
    sigma = args.nes_sigma / 255.0
    lr = args.nes_lr / 255.0
    gen = torch.Generator(device=x0.device).manual_seed(args.seed + 31 + image_ord * 1009)
    x = x0.detach().clone()
    rows = []
    query = 0
    for t in range(args.nes_steps):
        ctx = road_context(
            wrapper,
            x,
            y,
            args.layer,
            k=args.k,
            power_iters=args.power_iters,
            eta=args.road_eta / 255.0,
            eps=eps,
            x0=x0,
            seed=args.seed + 400009 + image_ord * 1009 + t,
        )
        grad_est = torch.zeros_like(x)
        for s in range(args.nes_samples):
            u = normalize_l2(torch.randn(x.shape, generator=gen, device=x.device))
            vals = []
            for sign in (1, -1):
                cand = project_linf(x + sign * sigma * u.sign(), x0, eps).detach()
                rec = score_move(wrapper, args.layer, x, cand, y, ctx)
                query += 1
                rows.append(
                    {
                        "attack": "nes",
                        "proposal_type": "sample",
                        "accepted": math.nan,
                        "helpful": int(rec["margin_drop"] > 0),
                        "image_ord": image_ord,
                        "step": t,
                        "query": query,
                        "sample_idx": s,
                        "sample_sign": sign,
                        **rec,
                    }
                )
                vals.append(-rec["margin_after"])
            grad_est = grad_est + ((vals[0] - vals[1]) / (2.0 * sigma)) * u
        cand_update = project_linf(x + lr * grad_est.sign(), x0, eps).detach()
        rec = score_move(wrapper, args.layer, x, cand_update, y, ctx)
        rows.append(
            {
                "attack": "nes_update",
                "proposal_type": "applied_step",
                "accepted": 1,
                "helpful": int(rec["margin_drop"] > 0),
                "image_ord": image_ord,
                "step": t,
                "query": query,
                **rec,
            }
        )
        x = cand_update
        if rec["success_after"] and args.stop_on_success:
            break
    return rows


def summarize_paths(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["gain_per_mobility"] = df.margin_drop / df.fd_mobility.clip(lower=1e-8)
    move_summary = (
        df.groupby(["attack", "proposal_type"], dropna=False)
        .agg(
            n=("margin_drop", "size"),
            n_images=("image_ord", "nunique"),
            median_gain_per_mobility=("gain_per_mobility", "median"),
            mean_gain_per_mobility=("gain_per_mobility", "mean"),
            median_margin_drop=("margin_drop", "median"),
            median_mobility=("fd_mobility", "median"),
            median_road_cos=("max_abs_road_cos", "median"),
            helpful_rate=("helpful", "mean"),
        )
        .reset_index()
    )
    path_rows = []
    views = {
        "pgd_ce_applied": df[df.attack.eq("pgd_ce")],
        "road_topk_margin_applied": df[df.attack.eq("road_topk_margin")],
        "nes_update_applied": df[df.attack.eq("nes_update")],
        "nes_all_query_cost": df[df.attack.isin(["nes", "nes_update"])],
        "square_accepted_path": df[df.attack.eq("square") & df.accepted.eq(1)],
        "square_all_query_cost": df[df.attack.eq("square")],
    }
    for view, sub in views.items():
        for image_ord, g in sub.groupby("image_ord"):
            g = g.sort_values(["step", "query"])
            if (g.success_after == 1).any():
                first = g.index[g.success_after.eq(1)][0]
                g = g.loc[:first]
            mobility = g.fd_mobility.clip(lower=1e-8).sum()
            gain = g.margin_drop.sum()
            path_rows.append(
                {
                    "path_view": view,
                    "image_ord": image_ord,
                    "success": int(g.success_after.max()),
                    "n_steps_or_queries": len(g),
                    "total_margin_drop": gain,
                    "total_mobility": mobility,
                    "path_gain_per_mobility": gain / mobility if mobility else np.nan,
                    "median_road_cos": g.max_abs_road_cos.median(),
                }
            )
    path = pd.DataFrame(path_rows)
    path_summary = (
        path.groupby("path_view")
        .agg(
            n_images=("image_ord", "nunique"),
            asr=("success", "mean"),
            median_steps_or_queries=("n_steps_or_queries", "median"),
            median_path_gain_per_mobility=("path_gain_per_mobility", "median"),
            mean_path_gain_per_mobility=("path_gain_per_mobility", "mean"),
            median_total_margin_drop=("total_margin_drop", "median"),
            median_total_mobility=("total_mobility", "median"),
            median_road_cos=("median_road_cos", "median"),
        )
        .reset_index()
    )
    return move_summary, path_summary, path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/same_harness_routing_efficiency_resnet50_n50")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--images", type=int, default=50)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--power-iters", type=int, default=1)
    p.add_argument("--road-eta", type=float, default=2.0)
    p.add_argument("--road-eta-grid", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--pgd-steps", type=int, default=20)
    p.add_argument("--pgd-eta", type=float, default=2.0)
    p.add_argument("--road-steps", type=int, default=10)
    p.add_argument("--square-queries", type=int, default=80)
    p.add_argument("--square-p-init", type=float, default=0.3)
    p.add_argument("--square-init-epochs", type=int, default=500)
    p.add_argument("--nes-steps", type=int, default=12)
    p.add_argument("--nes-samples", type=int, default=6)
    p.add_argument("--nes-sigma", type=float, default=1.0)
    p.add_argument("--nes-lr", type=float, default=2.0)
    p.add_argument("--stop-on-success", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = load_model(args.model, device).eval()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    image_rows = select_clean_correct(dataset, wrapper, args.images, device)
    image_rows.to_csv(out / "image_selection.csv", index=False)

    rows = []
    for row in image_rows.itertuples(index=False):
        x_raw, y0 = dataset[int(row.dataset_idx)]
        x0 = x_raw.unsqueeze(0).to(device)
        y = torch.tensor([int(y0)], dtype=torch.long, device=device)
        image_ord = int(row.image_ord)
        rows.extend(pgd_ce(wrapper, x0, y, args, image_ord))
        rows.extend(road_router(wrapper, x0, y, args, image_ord))
        rows.extend(square(wrapper, x0, y, args, image_ord))
        rows.extend(nes(wrapper, x0, y, args, image_ord))
        if (image_ord + 1) % 5 == 0:
            print(f"[same-harness-eff] {image_ord + 1}/{len(image_rows)}", flush=True)
            pd.DataFrame(rows).to_csv(out / "same_harness_step_proposals.partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "same_harness_step_proposals.csv", index=False)
    move_summary, path_summary, path = summarize_paths(df)
    move_summary.to_csv(out / "same_harness_move_efficiency_summary.csv", index=False)
    path_summary.to_csv(out / "same_harness_path_efficiency_summary.csv", index=False)
    path.to_csv(out / "same_harness_path_efficiency_per_image.csv", index=False)
    corr_rows = []
    for view, g in path.groupby("path_view"):
        corr_rows.append(
            {
                "path_view": view,
                "n": len(g),
                "asr": g.success.mean(),
                "spearman_eff_success": g.path_gain_per_mobility.corr(g.success, method="spearman")
                if g.success.nunique() > 1
                else np.nan,
                "pearson_eff_success": g.path_gain_per_mobility.corr(g.success, method="pearson")
                if g.success.nunique() > 1
                else np.nan,
            }
        )
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(out / "same_harness_efficiency_success_correlations.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(vars(args), indent=2))
    print(path_summary.to_string(index=False), flush=True)
    print(corr.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
