#!/usr/bin/env python3
"""Test image-conditioned highway selection as a Square pre-stage.

The previous highway preconditioning test searched for high-mobility states.
This version searches for the image-specific signed highway route that most
drops the true-class margin, then starts Square from that candidate while
charging the route-selection evaluations against the same query budget.

This is a diagnostic intervention, not a pure black-box claim: highway pullback
directions use the local model Jacobian. Target score evaluations used to choose
among candidate routes are counted as queries.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar_benchmark_optimizer_transport import p_selection  # noqa: E402
from experiments.pure_af_geometry.evaluate_image_conditioned_highway_selector import (  # noqa: E402
    ArtifactStore,
    eval_clean,
    fit_highway_basis,
    margin_pixel_grad,
    project_attack_rows,
    rank_signed_routes,
)
from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import margin, project_linf  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def feature_tensor(wrapper, x: torch.Tensor, layer: str) -> torch.Tensor:
    _logits, feats, _raw = wrapper.forward_with_features(x)
    if layer not in feats:
        raise RuntimeError(f"Layer {layer} not captured.")
    return feats[layer]


def highway_candidate(
    wrapper,
    x0: torch.Tensor,
    layer: str,
    route_direction: torch.Tensor,
    eps: float,
    step_size: float,
) -> torch.Tensor:
    x_probe = x0.detach().requires_grad_(True)
    h = feature_tensor(wrapper, x_probe, layer)
    scalar = (h * route_direction.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, x_probe)[0]
    return project_linf(x0 + step_size * grad.sign(), x0, eps).detach()


def route_pixel_grad_score(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    layer: str,
    route_direction: torch.Tensor,
    margin_grad_x: torch.Tensor,
) -> float:
    x_probe = x0.detach().requires_grad_(True)
    h = feature_tensor(wrapper, x_probe, layer)
    scalar = (h * route_direction.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, x_probe)[0]
    return float((-(margin_grad_x.detach()) * grad.detach().sign()).sum().item())


def ce_candidate(wrapper, x0: torch.Tensor, y: torch.Tensor, eps: float, step_size: float) -> torch.Tensor:
    x_probe = x0.detach().requires_grad_(True)
    logits = wrapper(x_probe)
    loss = torch.nn.functional.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_probe)[0]
    return project_linf(x0 + step_size * grad.sign(), x0, eps).detach()


def select_pre_state(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    label: int,
    layer: str,
    basis_t: torch.Tensor,
    routes: pd.DataFrame,
    mode: str,
    eps: float,
    step_size: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor | None, dict, list[dict]]:
    clean = eval_clean(wrapper, x0, y)
    if mode == "none":
        return None, {"queries": 0, "pred": clean["pred"], "margin": clean["margin"], "success": 0}, []

    candidate_routes = routes.copy()
    selector_kind = "observed_margin"
    if mode.startswith("global_rank1"):
        candidate_routes = routes[routes.global_rank <= 1].copy()
    elif mode.startswith("best_route_top"):
        n = int(mode.replace("best_route_top", "").replace("_pre", ""))
        candidate_routes = routes[routes.global_rank <= n].copy()
    elif mode.startswith("best_route_all"):
        candidate_routes = routes.copy()
    elif mode.startswith("random_route"):
        n = int(mode.replace("random_route", "").replace("_pre", ""))
        candidate_routes = routes.sample(n=min(n, len(routes)), random_state=int(rng.integers(0, 2**31 - 1))).copy()
    elif mode == "pixel_grad_route_pre":
        selector_kind = "pixel_margin_grad"
        candidate_routes = routes.copy()
    elif mode == "ce_step_pre":
        x_cand = ce_candidate(wrapper, x0, y, eps, step_size)
        ev = eval_clean(wrapper, x_cand, y)
        return (
            x_cand,
            {
                "queries": 1,
                "pred": ev["pred"],
                "margin": ev["margin"],
                "success": int(ev["pred"] != label),
                "chosen_route": "",
                "chosen_rank": np.nan,
                "selector_kind": "ce",
                "pre_margin_drop": float(clean["margin"] - ev["margin"]),
            },
            [],
        )
    else:
        raise ValueError(f"Unknown pre-stage mode: {mode}")

    route_rows = []
    margin_grad_x = margin_pixel_grad(wrapper, x0, y) if selector_kind == "pixel_margin_grad" else None
    if selector_kind == "pixel_margin_grad":
        for route in candidate_routes.itertuples():
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            score = route_pixel_grad_score(wrapper, x0, y, layer, direction, margin_grad_x)
            route_rows.append(
                {
                    "route": str(route.route),
                    "pc": int(route.pc),
                    "sign": int(route.sign),
                    "global_rank": int(route.global_rank),
                    "global_score": float(route.train_weighted_margin_drop_score),
                    "selector_score": score,
                    "margin": np.nan,
                    "margin_drop": np.nan,
                    "success": np.nan,
                    "pred": np.nan,
                }
            )
        chosen_meta = max(route_rows, key=lambda r: r["selector_score"])
        route = candidate_routes[candidate_routes.route == chosen_meta["route"]].iloc[0]
        direction = int(route.sign) * basis_t[int(route.pc) - 1]
        x_chosen = highway_candidate(wrapper, x0, layer, direction, eps, step_size)
        ev = eval_clean(wrapper, x_chosen, y)
        chosen_meta.update(
            {
                "margin": ev["margin"],
                "margin_drop": float(clean["margin"] - ev["margin"]),
                "success": int(ev["pred"] != label),
                "pred": ev["pred"],
            }
        )
        queries = 1
    else:
        x_chosen = None
        chosen_meta = None
        for route in candidate_routes.itertuples():
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            x_cand = highway_candidate(wrapper, x0, layer, direction, eps, step_size)
            ev = eval_clean(wrapper, x_cand, y)
            cand = {
                "route": str(route.route),
                "pc": int(route.pc),
                "sign": int(route.sign),
                "global_rank": int(route.global_rank),
                "global_score": float(route.train_weighted_margin_drop_score),
                "selector_score": float(clean["margin"] - ev["margin"]),
                "margin": ev["margin"],
                "margin_drop": float(clean["margin"] - ev["margin"]),
                "success": int(ev["pred"] != label),
                "pred": ev["pred"],
            }
            route_rows.append(cand)
            if chosen_meta is None or cand["margin_drop"] > chosen_meta["margin_drop"]:
                chosen_meta = cand
                x_chosen = x_cand
        queries = int(len(candidate_routes))

    pre_state = {
        "queries": queries,
        "pred": int(chosen_meta["pred"]),
        "margin": float(chosen_meta["margin"]),
        "success": int(chosen_meta["success"]),
        "chosen_route": str(chosen_meta["route"]),
        "chosen_rank": int(chosen_meta["global_rank"]),
        "selector_kind": selector_kind,
        "pre_margin_drop": float(chosen_meta["margin_drop"]),
    }
    return x_chosen.detach(), pre_state, route_rows


def square_from_known_state(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    x_start: torch.Tensor | None,
    eps: float,
    total_queries: int,
    seed: int,
    p_init: float,
    init_epochs: int,
    queries_used: int,
    known_margin: float | None = None,
    known_pred: int | None = None,
):
    gen = torch.Generator(device=x0.device).manual_seed(seed)
    c, h, w = x0.shape[1:]
    if x_start is None:
        stripe = torch.where(
            torch.rand((1, c, 1, w), generator=gen, device=x0.device) < 0.5,
            -torch.ones((1, c, 1, w), device=x0.device),
            torch.ones((1, c, 1, w), device=x0.device),
        ) * eps
        x_adv = (x0 + stripe).clamp(0, 1)
        queries = int(queries_used)
        with torch.no_grad():
            logits = wrapper(x_adv)
            best_margin = margin(logits, y).detach()
            final_pred = int(logits.argmax(1).item())
        queries += 1
    else:
        x_adv = project_linf(x_start, x0, eps).detach()
        queries = int(queries_used)
        if known_margin is None or known_pred is None:
            with torch.no_grad():
                logits = wrapper(x_adv)
                best_margin = margin(logits, y).detach()
                final_pred = int(logits.argmax(1).item())
            queries += 1
        else:
            best_margin = torch.tensor(float(known_margin), device=x0.device)
            final_pred = int(known_pred)
    if final_pred != int(y.item()):
        return x_adv.detach(), queries, queries, float(best_margin.item()), final_pred

    remaining = max(total_queries - queries, 0)
    success_query = math.nan
    for step in range(1, remaining + 1):
        perturbation = (x_adv - x0).detach().clone()
        p = p_selection(p_init, step + init_epochs, max(remaining, 1))
        side = int(round(np.sqrt(p * c * h * w / c)))
        side = min(max(side, 1), h - 1)
        top = int(torch.randint(0, h - side + 1, (1,), generator=gen, device=x0.device).item())
        left = int(torch.randint(0, w - side + 1, (1,), generator=gen, device=x0.device).item())
        patch = torch.where(
            torch.rand((1, c, 1, 1), generator=gen, device=x0.device) < 0.5,
            -torch.ones((1, c, 1, 1), device=x0.device),
            torch.ones((1, c, 1, 1), device=x0.device),
        ) * eps
        perturbation[:, :, top : top + side, left : left + side] = patch
        candidate = (x0 + perturbation).clamp(0, 1)
        with torch.no_grad():
            cand_logits = wrapper(candidate)
            cand_margin = margin(cand_logits, y)
            cand_pred = int(cand_logits.argmax(1).item())
        queries += 1
        if float(cand_margin.item()) < float(best_margin.item()):
            x_adv = candidate.detach()
            best_margin = cand_margin.detach()
            final_pred = cand_pred
        if cand_pred != int(y.item()):
            success_query = queries
            final_pred = cand_pred
            x_adv = candidate.detach()
            best_margin = cand_margin.detach()
            break
    return x_adv.detach(), queries, success_query, float(best_margin.item()), final_pred


def final_state(wrapper, x0: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> dict:
    with torch.no_grad():
        logits = wrapper(x)
        pred = int(logits.argmax(1).item())
        return {
            "pred": pred,
            "success": int(pred != int(y.item())),
            "margin": float(margin(logits, y).item()),
            "linf": float((x - x0).abs().max().item()),
        }


def summarize(df: pd.DataFrame, total_queries: int) -> pd.DataFrame:
    rows = []
    for mode, g in df.groupby("mode"):
        q = pd.to_numeric(g["query_to_success"], errors="coerce")
        q_all = q.fillna(total_queries)
        rows.append(
            {
                "mode": mode,
                "n_images": int(len(g)),
                "asr": float(g["final_success"].mean()),
                "pre_success_rate": float(g["pre_success"].mean()),
                "mean_queries_success_only": float(q.dropna().mean()) if q.notna().any() else np.nan,
                "median_queries_success_only": float(q.dropna().median()) if q.notna().any() else np.nan,
                "mean_queries_all_failures_as_budget": float(q_all.mean()),
                "median_queries_all_failures_as_budget": float(q_all.median()),
                "mean_pre_queries": float(g["pre_queries_used"].mean()),
                "mean_pre_margin_drop": float(g["pre_margin_drop"].mean()),
                "mean_total_margin_drop": float((g["clean_margin"] - g["final_margin"]).mean()),
                "mean_chosen_rank": float(g["chosen_rank"].mean()),
                "median_chosen_rank": float(g["chosen_rank"].median()),
                "max_linf": float(g["final_linf"].max()),
            }
        )
    order = [
        "none",
        "global_rank1_pre",
        "random_route5_pre",
        "random_route10_pre",
        "best_route_top5_pre",
        "best_route_top10_pre",
        "best_route_all_pre",
        "pixel_grad_route_pre",
        "ce_step_pre",
    ]
    out = pd.DataFrame(rows)
    out["mode"] = pd.Categorical(out["mode"], categories=order, ordered=True)
    return out.sort_values("mode").reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    labels = {
        "none": "Square",
        "global_rank1_pre": "Global rank-1",
        "random_route5_pre": "Best random 5",
        "random_route10_pre": "Best random 10",
        "best_route_top5_pre": "Best top-5",
        "best_route_top10_pre": "Best top-10",
        "best_route_all_pre": "Best all highways",
        "pixel_grad_route_pre": "Pixel-grad highway",
        "ce_step_pre": "CE pre-step",
    }
    sub = summary.copy()
    sub["label"] = [labels.get(str(m), str(m)) for m in sub["mode"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.3), dpi=180)
    axes[0].bar(sub["label"], sub["asr"], color="#59a14f")
    axes[0].set_ylabel("ASR")
    axes[0].set_title("Square after image-conditioned highway pre-stage")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(sub["label"], sub["mean_queries_all_failures_as_budget"], color="#4c78a8")
    axes[1].set_ylabel("Mean queries (failures as budget)")
    axes[1].set_title("Query efficiency under same budget")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "image_conditioned_highway_preattack.png", bbox_inches="tight")
    fig.savefig(out_dir / "image_conditioned_highway_preattack.pdf", bbox_inches="tight")
    plt.close(fig)


def paired_deltas(df: pd.DataFrame, total_queries: int) -> pd.DataFrame:
    wide_s = df.pivot(index="image_ord", columns="mode", values="final_success")
    wide_q = df.assign(q_all=pd.to_numeric(df["query_to_success"], errors="coerce").fillna(total_queries)).pivot(
        index="image_ord", columns="mode", values="q_all"
    )
    wide_m = df.assign(total_margin_drop=df["clean_margin"] - df["final_margin"]).pivot(
        index="image_ord", columns="mode", values="total_margin_drop"
    )
    rows = []
    baseline = "none"
    rng = np.random.default_rng(0)
    for mode in [c for c in wide_s.columns if c != baseline]:
        for metric, wide, sign in [("success", wide_s, 1), ("queries_all", wide_q, -1), ("total_margin_drop", wide_m, 1)]:
            if baseline not in wide or mode not in wide:
                continue
            d = (wide[mode] - wide[baseline]).dropna().to_numpy(dtype=float)
            if metric == "queries_all":
                # Negative is better for queries; report mode-baseline delta as well as fraction improved.
                improved = d < 0
            else:
                improved = d > 0
            boots = []
            n = len(d)
            if n:
                for _ in range(5000):
                    idx = rng.integers(0, n, n)
                    boots.append(float(d[idx].mean()))
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "n": n,
                    "mean_delta_vs_square": float(d.mean()) if n else np.nan,
                    "ci_low": float(np.quantile(boots, 0.025)) if boots else np.nan,
                    "ci_high": float(np.quantile(boots, 0.975)) if boots else np.nan,
                    "fraction_improved": float(improved.mean()) if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run(args):
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    store = ArtifactStore(Path(args.input_dir))
    _mean, basis, n_highway_train = fit_highway_basis(store, args.model, args.highway_source, args.layer, args.highway_k, args.seed)
    projected = project_attack_rows(store, args.model, parse_csv(args.rank_sources), args.layer, basis)
    routes = rank_signed_routes(projected, parse_csv(args.rank_sources), args.highway_k)
    images = store.eval_images(args.model, args.split, args.images)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    wrapper = load_model(args.model, device).eval()
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=device)
    eps = args.eps / 255.0
    pre_step_size = args.pre_step_size / 255.0
    modes = parse_csv(args.modes)
    rng = np.random.default_rng(args.seed)
    rows = []
    pre_candidate_rows = []
    for i, row in enumerate(images.itertuples(index=False), start=1):
        x_cpu, _ = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        clean = eval_clean(wrapper, x0, y)
        for mode in modes:
            seed = args.seed + int(row.image_ord) * 1009 + sum(ord(c) for c in mode)
            x_pre, pre_state, route_rows = select_pre_state(
                wrapper,
                x0,
                y,
                int(row.label),
                args.layer,
                basis_t,
                routes,
                mode,
                eps,
                pre_step_size,
                rng,
            )
            x_adv, queries, q_success, final_margin, final_pred = square_from_known_state(
                wrapper,
                x0,
                y,
                x_pre,
                eps,
                args.total_queries,
                seed + 17,
                args.square_p_init,
                args.square_init_epochs,
                int(pre_state["queries"]),
                known_margin=float(pre_state["margin"]) if mode != "none" else None,
                known_pred=int(pre_state["pred"]) if mode != "none" else None,
            )
            final = final_state(wrapper, x0, x_adv, y)
            rows.append(
                {
                    "model": args.model,
                    "layer": args.layer,
                    "image_ord": int(row.image_ord),
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "mode": mode,
                    "total_query_budget": int(args.total_queries),
                    "pre_queries_used": int(pre_state["queries"]),
                    "queries_used": int(queries),
                    "query_to_success": q_success,
                    "clean_margin": float(clean["margin"]),
                    "pre_margin": float(pre_state["margin"]),
                    "final_margin": float(final_margin),
                    "pre_margin_drop": float(pre_state.get("pre_margin_drop", 0.0)),
                    "pre_success": int(pre_state["success"]),
                    "final_success": int(final_pred != int(row.label)),
                    "final_pred": int(final_pred),
                    "chosen_route": str(pre_state.get("chosen_route", "")),
                    "chosen_rank": pre_state.get("chosen_rank", np.nan),
                    "selector_kind": str(pre_state.get("selector_kind", "")),
                    "final_linf": float(final["linf"]),
                }
            )
            for rr in route_rows:
                rr.update(
                    {
                        "model": args.model,
                        "layer": args.layer,
                        "image_ord": int(row.image_ord),
                        "dataset_idx": int(row.dataset_idx),
                        "label": int(row.label),
                        "mode": mode,
                    }
                )
                pre_candidate_rows.append(rr)
        if i % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(out_dir / "partial_image_conditioned_highway_preattack_per_image.csv", index=False)
            print(f"[{i}/{len(images)}] rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    candidates = pd.DataFrame(pre_candidate_rows)
    summary = summarize(df, args.total_queries)
    deltas = paired_deltas(df, args.total_queries)
    routes.to_csv(out_dir / "global_route_ranking.csv", index=False)
    df.to_csv(out_dir / "image_conditioned_highway_preattack_per_image.csv", index=False)
    candidates.to_csv(out_dir / "image_conditioned_highway_preattack_candidates.csv", index=False)
    summary.to_csv(out_dir / "image_conditioned_highway_preattack_summary.csv", index=False)
    deltas.to_csv(out_dir / "image_conditioned_highway_preattack_paired_deltas.csv", index=False)
    plot_summary(summary, out_dir)
    meta = vars(args).copy()
    meta.update({"device": str(device), "highway_train_vectors": int(n_highway_train), "n_eval_images": int(len(images))})
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = [
        "# Image-Conditioned Highway Pre-Attack",
        "",
        "The pre-stage evaluates image-specific signed highway routes and charges those evaluations against the same Square query budget.",
        "",
        "## Summary",
        "",
    ]
    for r in summary.itertuples():
        lines.append(
            f"- `{r.mode}`: ASR={r.asr:.3f}, meanQ(all)={r.mean_queries_all_failures_as_budget:.1f}, "
            f"preQ={r.mean_pre_queries:.1f}, pre_margin_drop={r.mean_pre_margin_drop:.3f}, "
            f"total_margin_drop={r.mean_total_margin_drop:.3f}, mean_rank={r.mean_chosen_rank:.2f}"
        )
    (out_dir / "image_conditioned_highway_preattack_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/image_conditioned_highway_preattack_bbb_resnet50_c50")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--rank-sources", default="pgd,square")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--split", default="test")
    p.add_argument("--images", type=int, default=50)
    p.add_argument("--eps", type=float, default=6.0)
    p.add_argument("--pre-step-size", type=float, default=2.0)
    p.add_argument("--total-queries", type=int, default=250)
    p.add_argument(
        "--modes",
        default="none,global_rank1_pre,random_route5_pre,random_route10_pre,best_route_top5_pre,best_route_top10_pre,best_route_all_pre,pixel_grad_route_pre,ce_step_pre",
    )
    p.add_argument("--square-p-init", type=float, default=0.8)
    p.add_argument("--square-init-epochs", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
