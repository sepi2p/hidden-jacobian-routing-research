#!/usr/bin/env python3
"""Attack step/proposal selection in hidden-Jacobian road coordinates.

This diagnostic asks how attack moves relate to the local high-mobility roads
defined by top hidden-Jacobian singular directions.  It separates two cases:

* gradient-based updates: APGD-style CE steps are compared with the best
  margin-dropping signed road candidate available at the same state;
* query-based proposals: Square proposals are labeled accepted/rejected, while
  NES antithetic samples are labeled helpful/harmful by immediate margin drop.

The goal is mechanism analysis, not a tuned attack benchmark.
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
from experiments.hidden_jacobian_routing.common import load_model, margin, project_linf  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_l2(v: torch.Tensor) -> torch.Tensor:
    return v / v.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)


def orthogonalize(v: torch.Tensor, basis: list[torch.Tensor]) -> torch.Tensor:
    out = v
    for b in basis:
        coeff = (out.flatten(1) * b.flatten(1)).sum(dim=1).view(-1, 1, 1, 1)
        out = out - coeff * b
    return out


def feature_fn(wrapper, layer: str):
    def f(inp: torch.Tensor) -> torch.Tensor:
        _logits, feats, _raw = wrapper.forward_with_features(inp)
        return feats[layer]

    return f


def estimate_topk_roads(wrapper, x: torch.Tensor, layer: str, k: int, power_iters: int, seed: int):
    f = feature_fn(wrapper, layer)
    dirs: list[torch.Tensor] = []
    sigmas: list[float] = []
    gen = torch.Generator(device=x.device).manual_seed(seed)
    for _rank in range(k):
        v = normalize_l2(torch.randn(x.shape, generator=gen, device=x.device))
        v = normalize_l2(orthogonalize(v, dirs))
        for _ in range(power_iters):
            x_req = x.detach().requires_grad_(True)
            _h, jv = torch.autograd.functional.jvp(f, x_req, v, create_graph=False, strict=False)
            h = f(x_req)
            dot = (h * jv.detach()).sum()
            w = torch.autograd.grad(dot, x_req)[0]
            v = normalize_l2(orthogonalize(w.detach(), dirs))
        with torch.no_grad():
            _h, jv = torch.autograd.functional.jvp(f, x.detach(), v, create_graph=False, strict=False)
            sigmas.append(float(jv.flatten(1).norm(dim=1).item()))
        dirs.append(v.detach())
    return dirs, sigmas


def eval_state(wrapper, x: torch.Tensor, y: torch.Tensor) -> dict:
    with torch.no_grad():
        logits = wrapper(x)
        pred = int(logits.argmax(1).item())
        m = float(margin(logits, y).item())
        ce = float(F.cross_entropy(logits, y).item())
    return {"pred": pred, "margin": m, "ce": ce, "success": int(pred != int(y.item()))}


def feature_mobility(wrapper, layer: str, x: torch.Tensor, cand: torch.Tensor) -> float:
    with torch.no_grad():
        _l0, f0, _r0 = wrapper.forward_with_features(x)
        _l1, f1, _r1 = wrapper.forward_with_features(cand)
        return float((f1[layer] - f0[layer]).flatten(1).norm(dim=1).item())


def road_context(
    wrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    layer: str,
    *,
    k: int,
    power_iters: int,
    eta: float,
    eps: float,
    x0: torch.Tensor,
    seed: int,
) -> dict:
    state = eval_state(wrapper, x, y)
    dirs, sigmas = estimate_topk_roads(wrapper, x, layer, k, power_iters, seed)
    candidates = []
    for rank, (v, sigma) in enumerate(zip(dirs, sigmas), start=1):
        signed = v.sign()
        for sign in (1, -1):
            cand = project_linf(x + sign * eta * signed, x0, eps).detach()
            ev = eval_state(wrapper, cand, y)
            candidates.append(
                {
                    "rank": rank,
                    "sign": sign,
                    "sigma": float(sigma),
                    "margin_after": ev["margin"],
                    "margin_drop": state["margin"] - ev["margin"],
                    "success": ev["success"],
                }
            )
    best_margin = max(candidates, key=lambda r: (r["margin_drop"], r["sigma"]))
    top_mobility_best_sign = max([r for r in candidates if r["rank"] == 1], key=lambda r: r["margin_drop"])
    return {
        "state": state,
        "dirs": dirs,
        "sigmas": sigmas,
        "best_road_margin_drop": float(best_margin["margin_drop"]),
        "best_road_rank": int(best_margin["rank"]),
        "best_road_sigma": float(best_margin["sigma"]),
        "best_road_success": int(best_margin["success"]),
        "rank1_best_sign_margin_drop": float(top_mobility_best_sign["margin_drop"]),
        "rank1_sigma": float(sigmas[0]),
    }


def score_move(wrapper, layer: str, x: torch.Tensor, cand: torch.Tensor, y: torch.Tensor, ctx: dict) -> dict:
    before = ctx["state"]
    after = eval_state(wrapper, cand, y)
    delta = (cand - x).detach()
    delta_flat = delta.flatten(1)
    delta_norm = float(delta_flat.norm(dim=1).item())
    delta_linf = float(delta.abs().max().item())
    mobility = feature_mobility(wrapper, layer, x, cand)
    if delta_norm < 1e-12:
        max_abs_cos = 0.0
        signed_cos = 0.0
        aligned_rank = -1
        aligned_sigma = 0.0
    else:
        cosines = []
        for rank, (v, sigma) in enumerate(zip(ctx["dirs"], ctx["sigmas"]), start=1):
            c = float((delta_flat * v.flatten(1)).sum().item() / max(delta_norm, 1e-12))
            cosines.append((abs(c), c, rank, float(sigma)))
        max_abs_cos, signed_cos, aligned_rank, aligned_sigma = max(cosines, key=lambda z: z[0])
    margin_drop = before["margin"] - after["margin"]
    denom = max(ctx["best_road_margin_drop"], 1e-8)
    return {
        "margin_before": before["margin"],
        "margin_after": after["margin"],
        "margin_drop": float(margin_drop),
        "ce_before": before["ce"],
        "ce_after": after["ce"],
        "success_after": int(after["success"]),
        "delta_l2": delta_norm,
        "delta_linf": delta_linf,
        "fd_mobility": mobility,
        "aligned_rank": int(aligned_rank),
        "max_abs_road_cos": float(max_abs_cos),
        "signed_road_cos": float(signed_cos),
        "aligned_sigma": float(aligned_sigma),
        "best_road_margin_drop": ctx["best_road_margin_drop"],
        "best_road_rank": ctx["best_road_rank"],
        "best_road_sigma": ctx["best_road_sigma"],
        "rank1_best_sign_margin_drop": ctx["rank1_best_sign_margin_drop"],
        "rank1_sigma": ctx["rank1_sigma"],
        "actual_over_best_road_gain": float(margin_drop / denom),
    }


def select_clean_correct(dataset, wrapper, n: int, device: torch.device) -> pd.DataFrame:
    rows = []
    for idx in range(len(dataset)):
        x, y0 = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([int(y0)], device=device)
        pred = eval_state(wrapper, x, y)["pred"]
        if pred == int(y0):
            rows.append({"image_ord": len(rows), "dataset_idx": idx, "label": int(y0)})
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise RuntimeError(f"Only found {len(rows)} clean-correct images")
    return pd.DataFrame(rows)


def apgd_style_steps(wrapper, x0, y, args, image_ord: int):
    eps = args.eps / 255.0
    eta = args.apgd_eta / 255.0
    x = x0.detach().clone()
    rows = []
    for t in range(args.apgd_steps):
        ctx = road_context(
            wrapper,
            x,
            y,
            args.layer,
            k=args.k,
            power_iters=args.power_iters,
            eta=eta,
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
                "attack": "apgd_style_ce",
                "proposal_type": "update",
                "accepted": 1,
                "helpful": int(rec["margin_drop"] > 0),
                "image_ord": image_ord,
                "step": t,
                "query": t + 1,
                **rec,
            }
        )
        x = cand
    return rows


def square_proposals(wrapper, x0, y, args, image_ord: int):
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
        if best["success"]:
            break
    return rows


def nes_proposals(wrapper, x0, y, args, image_ord: int):
    eps = args.eps / 255.0
    sigma = args.nes_sigma / 255.0
    lr = args.nes_lr / 255.0
    gen = torch.Generator(device=x0.device).manual_seed(args.seed + 31 + image_ord * 1009)
    x = x0.detach().clone()
    rows = []
    update_rows = []
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
        update_rows.append(
            {
                "attack": "nes_update",
                "proposal_type": "update",
                "accepted": 1,
                "helpful": int(rec["margin_drop"] > 0),
                "image_ord": image_ord,
                "step": t,
                "query": query,
                **rec,
            }
        )
        x = cand_update
        if rec["success_after"]:
            break
    return rows, update_rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["attack", "proposal_type"]
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("margin_drop", "size"),
            n_images=("image_ord", "nunique"),
            mean_margin_drop=("margin_drop", "mean"),
            median_margin_drop=("margin_drop", "median"),
            mean_fd_mobility=("fd_mobility", "mean"),
            median_fd_mobility=("fd_mobility", "median"),
            mean_road_cos=("max_abs_road_cos", "mean"),
            median_road_cos=("max_abs_road_cos", "median"),
            top1_align=("aligned_rank", lambda x: float((x == 1).mean())),
            top5_align=("aligned_rank", lambda x: float(((x >= 1) & (x <= 5)).mean())),
            median_actual_over_best=("actual_over_best_road_gain", "median"),
            helpful_rate=("helpful", "mean"),
            success_after_rate=("success_after", "mean"),
        )
        .reset_index()
    )
    return agg


def acceptance_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attack, sub in df.groupby("attack"):
        if sub["accepted"].notna().sum() == 0:
            continue
        for val, name in [(1, "accepted"), (0, "rejected")]:
            g = sub[sub.accepted == val]
            if len(g) == 0:
                continue
            rows.append(
                {
                    "attack": attack,
                    "group": name,
                    "n": len(g),
                    "mean_margin_drop": float(g.margin_drop.mean()),
                    "median_margin_drop": float(g.margin_drop.median()),
                    "mean_fd_mobility": float(g.fd_mobility.mean()),
                    "median_fd_mobility": float(g.fd_mobility.median()),
                    "mean_road_cos": float(g.max_abs_road_cos.mean()),
                    "median_road_cos": float(g.max_abs_road_cos.median()),
                    "top1_align": float((g.aligned_rank == 1).mean()),
                    "helpful_rate": float(g.helpful.mean()),
                }
            )
    return pd.DataFrame(rows)


def write_findings(out: Path, summary: pd.DataFrame, accept: pd.DataFrame) -> None:
    lines = [
        "# Attack Road-Selection Diagnostic",
        "",
        "This diagnostic compares attack steps/proposals with local top-k hidden-Jacobian roads. APGD-style rows are white-box CE updates; Square rows are accepted/rejected proposals; NES rows are antithetic query samples labeled helpful/harmful by immediate margin drop, plus NES update rows.",
        "",
        "## Overall Step/Proposal Summary",
        "",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"- {r.attack} ({r.proposal_type}): n={int(r.n)}, median margin drop={float(r.median_margin_drop):.4f}, "
            f"median mobility={float(r.median_fd_mobility):.4f}, median road cosine={float(r.median_road_cos):.3f}, "
            f"top-1 alignment={100*float(r.top1_align):.1f}%, median actual/best-road gain={float(r.median_actual_over_best):.3f}."
        )
    if not accept.empty:
        lines += ["", "## Accepted Versus Rejected Query Proposals", ""]
        for r in accept.itertuples(index=False):
            lines.append(
                f"- {r.attack} {r.group}: n={int(r.n)}, median margin drop={float(r.median_margin_drop):.4f}, "
                f"median mobility={float(r.median_fd_mobility):.4f}, median road cosine={float(r.median_road_cos):.3f}, "
                f"helpful rate={100*float(r.helpful_rate):.1f}%."
            )
    lines += [
        "",
        "Interpretation guardrail: Square acceptance is defined by objective improvement, so lower rejected-proposal margin gain is partly algorithmic by definition. The non-tautological question is whether accepted proposals are also enriched in hidden-Jacobian road alignment or mobility.",
    ]
    (out / "attack_road_selection_findings.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/attack_road_selection_diagnostic_resnet50_n30")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--images", type=int, default=30)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--power-iters", type=int, default=1)
    p.add_argument("--road-eta", type=float, default=2.0)
    p.add_argument("--apgd-steps", type=int, default=20)
    p.add_argument("--apgd-eta", type=float, default=2.0)
    p.add_argument("--square-queries", type=int, default=80)
    p.add_argument("--square-p-init", type=float, default=0.3)
    p.add_argument("--square-init-epochs", type=int, default=500)
    p.add_argument("--nes-steps", type=int, default=12)
    p.add_argument("--nes-samples", type=int, default=6)
    p.add_argument("--nes-sigma", type=float, default=1.0)
    p.add_argument("--nes-lr", type=float, default=2.0)
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
        rows.extend(apgd_style_steps(wrapper, x0, y, args, int(row.image_ord)))
        rows.extend(square_proposals(wrapper, x0, y, args, int(row.image_ord)))
        nes_rows, nes_update_rows = nes_proposals(wrapper, x0, y, args, int(row.image_ord))
        rows.extend(nes_rows)
        rows.extend(nes_update_rows)
        if (int(row.image_ord) + 1) % 5 == 0:
            print(f"[road-selection] {int(row.image_ord) + 1}/{len(image_rows)} images", flush=True)
            pd.DataFrame(rows).to_csv(out / "attack_road_selection_step_proposals.partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "attack_road_selection_step_proposals.csv", index=False)
    summary = summarize(df)
    accept = acceptance_summary(df)
    summary.to_csv(out / "attack_road_selection_summary.csv", index=False)
    accept.to_csv(out / "attack_road_selection_accepted_rejected.csv", index=False)
    write_findings(out, summary, accept)
    (out / "metadata.json").write_text(json.dumps(vars(args), indent=2))
    print(summary.to_string(index=False), flush=True)
    print(accept.to_string(index=False), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
