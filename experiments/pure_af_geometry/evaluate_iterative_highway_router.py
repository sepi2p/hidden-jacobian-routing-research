#!/usr/bin/env python3
"""Evaluate iterative state-conditioned highway routing before Square.

This is the "traffic routing" test.  Rather than selecting a highway route once
from the clean image, the router re-selects a signed highway route at each
current state, moves a small projected step, and then gives the remaining query
budget to Square.

All score evaluations used to select among candidate routes are counted against
the same total query budget.  Route construction still uses local feature
pullbacks, so this is a diagnostic intervention rather than a target-only
black-box attack.
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

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.evaluate_image_conditioned_highway_preattack import (  # noqa: E402
    final_state,
    square_from_known_state,
)
from experiments.pure_af_geometry.evaluate_image_conditioned_highway_selector import (  # noqa: E402
    ArtifactStore,
    eval_clean,
    fit_highway_basis,
    margin_pixel_grad,
    project_attack_rows,
    rank_signed_routes,
)
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


def route_candidate(
    wrapper,
    x0: torch.Tensor,
    x_cur: torch.Tensor,
    layer: str,
    route_direction: torch.Tensor,
    eps: float,
    step_size: float,
) -> torch.Tensor:
    x_probe = x_cur.detach().requires_grad_(True)
    h = feature_tensor(wrapper, x_probe, layer)
    scalar = (h * route_direction.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, x_probe)[0]
    return project_linf(x_cur + step_size * grad.sign(), x0, eps).detach()


def ce_candidate(wrapper, x0: torch.Tensor, x_cur: torch.Tensor, y: torch.Tensor, eps: float, step_size: float) -> torch.Tensor:
    x_probe = x_cur.detach().requires_grad_(True)
    logits = wrapper(x_probe)
    loss = torch.nn.functional.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_probe)[0]
    return project_linf(x_cur + step_size * grad.sign(), x0, eps).detach()


def route_pixel_grad_score(
    wrapper,
    x_cur: torch.Tensor,
    y: torch.Tensor,
    layer: str,
    route_direction: torch.Tensor,
    margin_grad_x: torch.Tensor,
) -> float:
    x_probe = x_cur.detach().requires_grad_(True)
    h = feature_tensor(wrapper, x_probe, layer)
    scalar = (h * route_direction.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, x_probe)[0]
    return float((-(margin_grad_x.detach()) * grad.detach().sign()).sum().item())


def choose_routes(routes: pd.DataFrame, mode: str, rng: np.random.Generator) -> tuple[pd.DataFrame, str]:
    if mode == "none":
        return routes.iloc[:0].copy(), "none"
    if mode.startswith("global_rank1"):
        return routes[routes.global_rank <= 1].copy(), "observed_margin"
    if mode.startswith("top"):
        n = int(mode.split("_", 1)[0].replace("top", ""))
        return routes[routes.global_rank <= n].copy(), "observed_margin"
    if mode.startswith("all"):
        return routes.copy(), "observed_margin"
    if mode.startswith("random"):
        n = int(mode.split("_", 1)[0].replace("random", ""))
        return routes.sample(n=min(n, len(routes)), random_state=int(rng.integers(0, 2**31 - 1))).copy(), "observed_margin"
    if mode.startswith("pixelgrad"):
        return routes.copy(), "pixel_grad"
    if mode.startswith("ce"):
        return routes.iloc[:0].copy(), "ce"
    raise ValueError(f"Unknown mode {mode}")


def route_once(
    wrapper,
    x0: torch.Tensor,
    x_cur: torch.Tensor,
    y: torch.Tensor,
    label: int,
    layer: str,
    basis_t: torch.Tensor,
    routes: pd.DataFrame,
    mode: str,
    eps: float,
    step_size: float,
    rng: np.random.Generator,
    current_margin: float,
) -> tuple[torch.Tensor, dict, list[dict]]:
    candidates, selector_kind = choose_routes(routes, mode, rng)
    candidate_rows: list[dict] = []

    if selector_kind == "none":
        return x_cur, {
            "queries": 0,
            "accepted": 0,
            "success": 0,
            "pred": label,
            "margin": current_margin,
            "chosen_route": "",
            "chosen_rank": np.nan,
            "chosen_margin_drop": 0.0,
        }, candidate_rows

    if selector_kind == "ce":
        x_next = ce_candidate(wrapper, x0, x_cur, y, eps, step_size)
        ev = eval_clean(wrapper, x_next, y)
        drop = float(current_margin - ev["margin"])
        return x_next, {
            "queries": 1,
            "accepted": 1,
            "success": int(ev["pred"] != label),
            "pred": int(ev["pred"]),
            "margin": float(ev["margin"]),
            "chosen_route": "ce",
            "chosen_rank": np.nan,
            "chosen_margin_drop": drop,
        }, candidate_rows

    if selector_kind == "pixel_grad":
        margin_grad_x = margin_pixel_grad(wrapper, x_cur, y)
        scored = []
        for route in candidates.itertuples():
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            score = route_pixel_grad_score(wrapper, x_cur, y, layer, direction, margin_grad_x)
            scored.append((score, route))
        _score, chosen_route = max(scored, key=lambda x: x[0])
        direction = int(chosen_route.sign) * basis_t[int(chosen_route.pc) - 1]
        x_next = route_candidate(wrapper, x0, x_cur, layer, direction, eps, step_size)
        ev = eval_clean(wrapper, x_next, y)
        drop = float(current_margin - ev["margin"])
        return x_next, {
            "queries": 1,
            "accepted": 1,
            "success": int(ev["pred"] != label),
            "pred": int(ev["pred"]),
            "margin": float(ev["margin"]),
            "chosen_route": str(chosen_route.route),
            "chosen_rank": int(chosen_route.global_rank),
            "chosen_margin_drop": drop,
        }, candidate_rows

    chosen = None
    chosen_x = None
    for route in candidates.itertuples():
        direction = int(route.sign) * basis_t[int(route.pc) - 1]
        x_cand = route_candidate(wrapper, x0, x_cur, layer, direction, eps, step_size)
        ev = eval_clean(wrapper, x_cand, y)
        drop = float(current_margin - ev["margin"])
        row = {
            "route": str(route.route),
            "pc": int(route.pc),
            "sign": int(route.sign),
            "global_rank": int(route.global_rank),
            "global_score": float(route.train_weighted_margin_drop_score),
            "candidate_margin": float(ev["margin"]),
            "margin_drop": drop,
            "success": int(ev["pred"] != label),
            "pred": int(ev["pred"]),
        }
        candidate_rows.append(row)
        if chosen is None or row["candidate_margin"] < chosen["candidate_margin"]:
            chosen = row
            chosen_x = x_cand

    if chosen is None:
        return x_cur, {
            "queries": 0,
            "accepted": 0,
            "success": 0,
            "pred": label,
            "margin": current_margin,
            "chosen_route": "",
            "chosen_rank": np.nan,
            "chosen_margin_drop": 0.0,
        }, candidate_rows

    accepted = bool(chosen["candidate_margin"] < current_margin)
    x_next = chosen_x.detach() if accepted else x_cur.detach()
    next_margin = float(chosen["candidate_margin"]) if accepted else float(current_margin)
    return x_next, {
        "queries": int(len(candidates)),
        "accepted": int(accepted),
        "success": int(chosen["success"]) if accepted else 0,
        "pred": int(chosen["pred"]) if accepted else label,
        "margin": next_margin,
        "chosen_route": str(chosen["route"]),
        "chosen_rank": int(chosen["global_rank"]),
        "chosen_margin_drop": float(chosen["margin_drop"]),
    }, candidate_rows


def router_stage(
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
    max_steps: int,
    max_pre_queries: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor | None, dict, list[dict]]:
    if mode == "none":
        clean = eval_clean(wrapper, x0, y)
        return None, {
            "queries": 0,
            "steps": 0,
            "accepted_steps": 0,
            "success": 0,
            "pred": int(clean["pred"]),
            "margin": float(clean["margin"]),
            "total_margin_drop": 0.0,
            "mean_chosen_rank": np.nan,
        }, []

    clean = eval_clean(wrapper, x0, y)
    x_cur = x0.detach()
    current_margin = float(clean["margin"])
    used = 0
    accepted = 0
    chosen_ranks = []
    route_rows_all = []
    for step in range(max_steps):
        if used >= max_pre_queries:
            break
        x_next, info, candidate_rows = route_once(
            wrapper,
            x0,
            x_cur,
            y,
            label,
            layer,
            basis_t,
            routes,
            mode,
            eps,
            step_size,
            rng,
            current_margin,
        )
        if used + int(info["queries"]) > max_pre_queries:
            break
        used += int(info["queries"])
        accepted += int(info["accepted"])
        if np.isfinite(info["chosen_rank"]):
            chosen_ranks.append(float(info["chosen_rank"]))
        for row in candidate_rows:
            row.update({"router_step": step, "mode": mode})
            route_rows_all.append(row)
        x_cur = x_next.detach()
        current_margin = float(info["margin"])
        if int(info["success"]):
            return x_cur, {
                "queries": used,
                "steps": step + 1,
                "accepted_steps": accepted,
                "success": 1,
                "pred": int(info["pred"]),
                "margin": current_margin,
                "total_margin_drop": float(clean["margin"] - current_margin),
                "mean_chosen_rank": float(np.mean(chosen_ranks)) if chosen_ranks else np.nan,
            }, route_rows_all
    return x_cur.detach(), {
        "queries": used,
        "steps": step + 1 if max_steps else 0,
        "accepted_steps": accepted,
        "success": 0,
        "pred": label,
        "margin": current_margin,
        "total_margin_drop": float(clean["margin"] - current_margin),
        "mean_chosen_rank": float(np.mean(chosen_ranks)) if chosen_ranks else np.nan,
    }, route_rows_all


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
                "router_success_rate": float(g["router_success"].mean()),
                "mean_queries_success_only": float(q.dropna().mean()) if q.notna().any() else np.nan,
                "median_queries_success_only": float(q.dropna().median()) if q.notna().any() else np.nan,
                "mean_queries_all_failures_as_budget": float(q_all.mean()),
                "median_queries_all_failures_as_budget": float(q_all.median()),
                "mean_router_queries": float(g["router_queries"].mean()),
                "mean_router_steps": float(g["router_steps"].mean()),
                "mean_router_accepted_steps": float(g["router_accepted_steps"].mean()),
                "mean_router_margin_drop": float(g["router_margin_drop"].mean()),
                "mean_total_margin_drop": float((g["clean_margin"] - g["final_margin"]).mean()),
                "mean_chosen_rank": float(g["mean_chosen_rank"].mean()),
                "max_linf": float(g["final_linf"].max()),
            }
        )
    order = [
        "none",
        "global_rank1",
        "random5",
        "random10",
        "top5",
        "top10",
        "all",
        "pixelgrad",
        "ce",
    ]
    out = pd.DataFrame(rows)
    out["mode"] = pd.Categorical(out["mode"], categories=order, ordered=True)
    return out.sort_values("mode").reset_index(drop=True)


def paired_deltas(df: pd.DataFrame, total_queries: int) -> pd.DataFrame:
    wide_s = df.pivot(index="image_ord", columns="mode", values="final_success")
    wide_q = df.assign(q_all=pd.to_numeric(df["query_to_success"], errors="coerce").fillna(total_queries)).pivot(
        index="image_ord", columns="mode", values="q_all"
    )
    wide_m = df.assign(total_margin_drop=df["clean_margin"] - df["final_margin"]).pivot(
        index="image_ord", columns="mode", values="total_margin_drop"
    )
    rows = []
    rng = np.random.default_rng(0)
    baseline = "none"
    for mode in [c for c in wide_s.columns if c != baseline]:
        for metric, wide in [("success", wide_s), ("queries_all", wide_q), ("total_margin_drop", wide_m)]:
            d = (wide[mode] - wide[baseline]).dropna().to_numpy(dtype=float)
            n = len(d)
            boots = []
            for _ in range(5000):
                idx = rng.integers(0, n, n)
                boots.append(float(d[idx].mean()))
            if metric == "queries_all":
                improved = d < 0
            else:
                improved = d > 0
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "n": n,
                    "mean_delta_vs_square": float(d.mean()),
                    "ci_low": float(np.quantile(boots, 0.025)),
                    "ci_high": float(np.quantile(boots, 0.975)),
                    "fraction_improved": float(improved.mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    labels = {
        "none": "Square",
        "global_rank1": "Global",
        "random5": "Rand5",
        "random10": "Rand10",
        "top5": "Top5",
        "top10": "Top10",
        "all": "All highways",
        "pixelgrad": "Pixel-grad",
        "ce": "CE",
    }
    sub = summary.copy()
    sub["label"] = [labels.get(str(m), str(m)) for m in sub["mode"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=180)
    axes[0].bar(sub["label"], sub["asr"], color="#59a14f")
    axes[0].set_ylabel("ASR")
    axes[0].set_title("Iterative highway routing + Square")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(sub["label"], sub["mean_queries_all_failures_as_budget"], color="#4c78a8")
    axes[1].set_ylabel("Mean queries (failures as budget)")
    axes[1].set_title("Same 250-query budget")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "iterative_highway_router_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "iterative_highway_router_summary.pdf", bbox_inches="tight")
    plt.close(fig)


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
    step_size = args.router_step_size / 255.0
    modes = parse_csv(args.modes)
    rng = np.random.default_rng(args.seed)
    rows = []
    route_rows = []
    for i, row in enumerate(images.itertuples(index=False), start=1):
        x_cpu, _ = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        clean = eval_clean(wrapper, x0, y)
        for mode in modes:
            seed = args.seed + int(row.image_ord) * 1009 + sum(ord(c) for c in mode)
            x_start, router_info, candidate_rows = router_stage(
                wrapper,
                x0,
                y,
                int(row.label),
                args.layer,
                basis_t,
                routes,
                mode,
                eps,
                step_size,
                args.router_steps,
                args.max_router_queries,
                rng,
            )
            x_adv, queries, q_success, final_margin, final_pred = square_from_known_state(
                wrapper,
                x0,
                y,
                x_start,
                eps,
                args.total_queries,
                seed + 17,
                args.square_p_init,
                args.square_init_epochs,
                int(router_info["queries"]),
                known_margin=float(router_info["margin"]) if mode != "none" else None,
                known_pred=int(router_info["pred"]) if mode != "none" else None,
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
                    "router_queries": int(router_info["queries"]),
                    "router_steps": int(router_info["steps"]),
                    "router_accepted_steps": int(router_info["accepted_steps"]),
                    "queries_used": int(queries),
                    "query_to_success": q_success,
                    "clean_margin": float(clean["margin"]),
                    "router_margin": float(router_info["margin"]),
                    "final_margin": float(final_margin),
                    "router_margin_drop": float(router_info["total_margin_drop"]),
                    "router_success": int(router_info["success"]),
                    "final_success": int(final_pred != int(row.label)),
                    "final_pred": int(final_pred),
                    "mean_chosen_rank": float(router_info["mean_chosen_rank"]),
                    "final_linf": float(final["linf"]),
                }
            )
            for rr in candidate_rows:
                rr.update({"image_ord": int(row.image_ord), "dataset_idx": int(row.dataset_idx), "label": int(row.label)})
                route_rows.append(rr)
        if i % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(out_dir / "partial_iterative_highway_router_per_image.csv", index=False)
            print(f"[{i}/{len(images)}] rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    route_df = pd.DataFrame(route_rows)
    summary = summarize(df, args.total_queries)
    deltas = paired_deltas(df, args.total_queries)
    routes.to_csv(out_dir / "global_route_ranking.csv", index=False)
    df.to_csv(out_dir / "iterative_highway_router_per_image.csv", index=False)
    route_df.to_csv(out_dir / "iterative_highway_router_candidates.csv", index=False)
    summary.to_csv(out_dir / "iterative_highway_router_summary.csv", index=False)
    deltas.to_csv(out_dir / "iterative_highway_router_paired_deltas.csv", index=False)
    plot_summary(summary, out_dir)
    meta = vars(args).copy()
    meta.update({"device": str(device), "highway_train_vectors": int(n_highway_train), "n_eval_images": int(len(images))})
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = [
        "# Iterative Highway Router",
        "",
        "At each current state, the router re-selects a signed highway route, moves one projected step, then Square uses the remaining budget.",
        "",
        "## Summary",
        "",
    ]
    for r in summary.itertuples():
        lines.append(
            f"- `{r.mode}`: ASR={r.asr:.3f}, meanQ(all)={r.mean_queries_all_failures_as_budget:.1f}, "
            f"routerQ={r.mean_router_queries:.1f}, router_drop={r.mean_router_margin_drop:.3f}, "
            f"total_drop={r.mean_total_margin_drop:.3f}"
        )
    (out_dir / "iterative_highway_router_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/iterative_highway_router_bbb_resnet50_c50")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--rank-sources", default="pgd,square")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--split", default="test")
    p.add_argument("--images", type=int, default=50)
    p.add_argument("--eps", type=float, default=6.0)
    p.add_argument("--router-step-size", type=float, default=2.0)
    p.add_argument("--router-steps", type=int, default=5)
    p.add_argument("--max-router-queries", type=int, default=50)
    p.add_argument("--total-queries", type=int, default=250)
    p.add_argument("--modes", default="none,global_rank1,random5,random10,top5,top10,all,pixelgrad,ce")
    p.add_argument("--square-p-init", type=float, default=0.8)
    p.add_argument("--square-init-epochs", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
