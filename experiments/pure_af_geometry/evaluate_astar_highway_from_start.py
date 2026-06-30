#!/usr/bin/env python3
"""White-box A*/beam-style highway routing from the clean image.

This is the from-start traffic-routing test.  Instead of first running PGD and
then rescuing failures, we compare PGD against planners that search over
state-dependent highway actions from the clean image.

The search space is continuous, so this is an A*-like bounded best-first/beam
planner rather than textbook graph A*.  Each node is an image state inside the
same L_inf ball.  Each edge is a signed highway pullback move.  The priority is
the path depth plus a normalized remaining-margin heuristic.
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
from dataclasses import dataclass, field
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


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def feature_tensor(wrapper, x: torch.Tensor, layer: str) -> torch.Tensor:
    _logits, feats, _raw = wrapper.forward_with_features(x)
    if layer not in feats:
        raise RuntimeError(f"Layer {layer} not captured.")
    return feats[layer]


def eval_state(wrapper, x0: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> dict:
    ev = eval_clean(wrapper, x, y)
    return {
        "pred": int(ev["pred"]),
        "margin": float(ev["margin"]),
        "success": int(ev["pred"] != int(y.item())),
        "linf": float((x - x0).abs().max().item()),
    }


def ce_step(wrapper, x0: torch.Tensor, x_cur: torch.Tensor, y: torch.Tensor, eps: float, step_size: float) -> torch.Tensor:
    probe = x_cur.detach().requires_grad_(True)
    logits = wrapper(probe)
    loss = torch.nn.functional.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, probe)[0]
    return project_linf(x_cur + step_size * grad.sign(), x0, eps).detach()


def pgd_attack(wrapper, x0: torch.Tensor, y: torch.Tensor, eps: float, steps: int, step_size: float):
    x = x0.detach()
    states = []
    for _ in range(steps):
        x = ce_step(wrapper, x0, x, y, eps, step_size)
        states.append(eval_state(wrapper, x0, x, y))
        if states[-1]["success"]:
            break
    final = eval_state(wrapper, x0, x, y)
    return x, final, states


def random_pixel_attack(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    gen: torch.Generator,
):
    x = x0.detach()
    states = []
    for _ in range(steps):
        direction = torch.randn(x.shape, generator=gen, device=x.device).sign()
        x = project_linf(x + step_size * direction, x0, eps).detach()
        states.append(eval_state(wrapper, x0, x, y))
        if states[-1]["success"]:
            break
    final = eval_state(wrapper, x0, x, y)
    return x, final, states


def route_candidate(
    wrapper,
    x0: torch.Tensor,
    x_cur: torch.Tensor,
    layer: str,
    direction: torch.Tensor,
    eps: float,
    step_size: float,
) -> torch.Tensor:
    probe = x_cur.detach().requires_grad_(True)
    h = feature_tensor(wrapper, probe, layer)
    scalar = (h * direction.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, probe)[0]
    return project_linf(x_cur + step_size * grad.sign(), x0, eps).detach()


def route_pixelgrad_score(
    wrapper,
    x_cur: torch.Tensor,
    y: torch.Tensor,
    layer: str,
    direction: torch.Tensor,
    margin_grad_x: torch.Tensor,
) -> float:
    probe = x_cur.detach().requires_grad_(True)
    h = feature_tensor(wrapper, probe, layer)
    scalar = (h * direction.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, probe)[0]
    return float((-(margin_grad_x.detach()) * grad.detach().sign()).sum().item())


def select_routes(
    wrapper,
    x_cur: torch.Tensor,
    y: torch.Tensor,
    layer: str,
    basis_t: torch.Tensor,
    routes: pd.DataFrame,
    action_set: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if action_set.startswith("top"):
        k = int(action_set.replace("top", ""))
        return routes[routes.global_rank <= k].copy()
    if action_set.startswith("random"):
        k = int(action_set.replace("random", ""))
        return routes.sample(n=min(k, len(routes)), random_state=int(rng.integers(0, 2**31 - 1))).copy()
    if action_set == "all":
        return routes.copy()
    if action_set.startswith("pixelgrad"):
        k = int(action_set.replace("pixelgrad", ""))
        grad_x = margin_pixel_grad(wrapper, x_cur, y)
        scored = []
        for route in routes.itertuples():
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            score = route_pixelgrad_score(wrapper, x_cur, y, layer, direction, grad_x)
            scored.append((score, route.Index))
        keep = [idx for _score, idx in sorted(scored, reverse=True)[:k]]
        return routes.loc[keep].copy()
    raise ValueError(f"Unknown action set: {action_set}")


@dataclass(order=True)
class SearchNode:
    priority: float
    tie: int
    x: torch.Tensor = field(compare=False)
    margin: float = field(compare=False)
    depth: int = field(compare=False)
    path: tuple[str, ...] = field(compare=False)
    evals: int = field(compare=False)


def priority_fn(depth: int, margin_value: float, clean_margin: float, heuristic_weight: float) -> float:
    denom = max(abs(clean_margin), 1e-6)
    return float(depth + heuristic_weight * max(margin_value, 0.0) / denom)


def highway_greedy(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    layer: str,
    basis_t: torch.Tensor,
    routes: pd.DataFrame,
    action_set: str,
    eps: float,
    step_size: float,
    steps: int,
    rng: np.random.Generator,
):
    clean = eval_state(wrapper, x0, x0, y)
    x = x0.detach()
    current_margin = float(clean["margin"])
    evals = 0
    path = []
    candidate_rows = []
    for depth in range(steps):
        candidates = select_routes(wrapper, x, y, layer, basis_t, routes, action_set, rng)
        best = None
        best_x = None
        for route in candidates.itertuples():
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            x_cand = route_candidate(wrapper, x0, x, layer, direction, eps, step_size)
            ev = eval_state(wrapper, x0, x_cand, y)
            evals += 1
            row = {
                "depth": depth + 1,
                "route": str(route.route),
                "global_rank": int(route.global_rank),
                "margin": ev["margin"],
                "margin_drop": current_margin - ev["margin"],
                "success": ev["success"],
            }
            candidate_rows.append(row)
            if best is None or ev["margin"] < best["margin"]:
                best = {**row, **ev}
                best_x = x_cand
        x = best_x.detach()
        current_margin = float(best["margin"])
        path.append(str(best["route"]))
        if int(best["success"]):
            break
    final = eval_state(wrapper, x0, x, y)
    return x, {**final, "evals": evals, "depth": len(path), "path": "|".join(path)}, candidate_rows


def highway_astar(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    layer: str,
    basis_t: torch.Tensor,
    routes: pd.DataFrame,
    action_set: str,
    eps: float,
    step_size: float,
    max_depth: int,
    max_expansions: int,
    beam_size: int,
    heuristic_weight: float,
    rng: np.random.Generator,
):
    clean = eval_state(wrapper, x0, x0, y)
    clean_margin = float(clean["margin"])
    frontier: list[SearchNode] = []
    counter = 0
    start = SearchNode(priority_fn(0, clean_margin, clean_margin, heuristic_weight), counter, x0.detach(), clean_margin, 0, tuple(), 0)
    heapq.heappush(frontier, start)
    best_node = start
    expansions = 0
    total_evals = 0
    candidate_rows = []
    while frontier and expansions < max_expansions:
        node = heapq.heappop(frontier)
        if node.margin < best_node.margin:
            best_node = node
        if node.margin < 0:
            break
        if node.depth >= max_depth:
            continue
        candidates = select_routes(wrapper, node.x, y, layer, basis_t, routes, action_set, rng)
        children = []
        for route in candidates.itertuples():
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            x_cand = route_candidate(wrapper, x0, node.x, layer, direction, eps, step_size)
            ev = eval_state(wrapper, x0, x_cand, y)
            total_evals += 1
            counter += 1
            path = node.path + (str(route.route),)
            pr = priority_fn(node.depth + 1, ev["margin"], clean_margin, heuristic_weight)
            child = SearchNode(pr, counter, x_cand.detach(), float(ev["margin"]), node.depth + 1, path, node.evals + 1)
            children.append(child)
            candidate_rows.append(
                {
                    "expanded_depth": node.depth,
                    "child_depth": node.depth + 1,
                    "route": str(route.route),
                    "global_rank": int(route.global_rank),
                    "margin": float(ev["margin"]),
                    "success": int(ev["success"]),
                    "priority": float(pr),
                }
            )
            if ev["margin"] < best_node.margin:
                best_node = child
            if int(ev["success"]):
                frontier = [child]
                break
        expansions += 1
        if best_node.margin < 0:
            break
        frontier.extend(children)
        frontier = heapq.nsmallest(beam_size, frontier)
        heapq.heapify(frontier)
    final = eval_state(wrapper, x0, best_node.x, y)
    return best_node.x.detach(), {
        **final,
        "evals": total_evals,
        "expansions": expansions,
        "depth": best_node.depth,
        "path": "|".join(best_node.path),
        "best_priority": float(best_node.priority),
    }, candidate_rows


def load_images(
    store: ArtifactStore,
    model: str,
    split: str,
    max_images: int,
    image_ord_csv: str = "",
    image_ords: str = "",
) -> pd.DataFrame:
    base = store.outcomes[(store.outcomes.model == model) & (store.outcomes.source == "pgd")][
        ["image_ord", "dataset_idx", "label"]
    ].drop_duplicates()
    keep_ords: set[int] = set()
    if image_ord_csv:
        csv_df = pd.read_csv(image_ord_csv)
        if "image_ord" not in csv_df.columns:
            raise ValueError(f"{image_ord_csv} must contain an image_ord column.")
        keep_ords.update(int(x) for x in csv_df["image_ord"].dropna().unique())
    if image_ords:
        keep_ords.update(int(x.strip()) for x in image_ords.split(",") if x.strip())
    if keep_ords:
        base = base[base.image_ord.isin(sorted(keep_ords))]
    if split != "all":
        base = base.merge(store.splits, on="image_ord", how="left")
        base = base[base.split == split]
    base = base.sort_values("image_ord")
    if max_images > 0:
        base = base.head(max_images)
    return base.reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("method", dropna=False)
        .agg(
            n=("success", "size"),
            asr=("success", "mean"),
            mean_margin=("margin", "mean"),
            median_margin=("margin", "median"),
            mean_margin_drop=("margin_drop", "mean"),
            mean_evals=("evals", "mean"),
            median_evals=("evals", "median"),
            mean_depth=("depth", "mean"),
            max_linf=("linf", "max"),
        )
        .reset_index()
    )


def paired_deltas(df: pd.DataFrame, baselines: list[str]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    wide_s = df.pivot(index="image_ord", columns="method", values="success")
    wide_m = df.pivot(index="image_ord", columns="method", values="margin_drop")
    for baseline in baselines:
        if baseline not in wide_s:
            continue
        for method in [c for c in wide_s.columns if c != baseline]:
            for metric, wide in [("success", wide_s), ("margin_drop", wide_m)]:
                d = (wide[method] - wide[baseline]).dropna().to_numpy(dtype=float)
                if len(d) == 0:
                    continue
                boots = []
                for _ in range(5000):
                    idx = rng.integers(0, len(d), len(d))
                    boots.append(float(d[idx].mean()))
                rows.append(
                    {
                        "baseline": baseline,
                        "method": method,
                        "metric": metric,
                        "n": int(len(d)),
                        "mean_delta": float(d.mean()),
                        "ci_low": float(np.quantile(boots, 0.025)),
                        "ci_high": float(np.quantile(boots, 0.975)),
                        "fraction_better": float((d > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    order = [
        "pgd2",
        "pgd3",
        "pgd5",
        "random_pixel3",
        "greedy_top5",
        "greedy_pixelgrad5",
        "astar_top5",
        "astar_top10",
        "astar_pixelgrad5",
    ]
    sub = summary.copy()
    sub["method"] = pd.Categorical(sub["method"], categories=order, ordered=True)
    sub = sub.sort_values("method")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=180)
    axes[0].bar(sub["method"].astype(str), sub["asr"], color="#59a14f")
    axes[0].set_ylabel("ASR from clean")
    axes[0].set_title("From-start white-box highway planning")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(sub["method"].astype(str), sub["mean_margin_drop"], color="#4c78a8")
    axes[1].set_ylabel("Mean margin drop")
    axes[1].set_title("Progress inside same L_inf ball")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "astar_highway_from_start_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "astar_highway_from_start_summary.pdf", bbox_inches="tight")
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
    images = load_images(store, args.model, args.split, args.images, args.image_ord_csv, args.image_ords)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    wrapper = load_model(args.model, device).eval()
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=device)
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    pgd_steps = parse_int_csv(args.pgd_steps)
    rng = np.random.default_rng(args.seed)
    rows = []
    candidate_rows = []
    for i, row in enumerate(images.itertuples(index=False), start=1):
        x_cpu, _ = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        clean = eval_state(wrapper, x0, x0, y)
        for steps in pgd_steps:
            _x, final, states = pgd_attack(wrapper, x0, y, eps, steps, step_size)
            rows.append(
                {
                    "image_ord": int(row.image_ord),
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "method": f"pgd{steps}",
                    "success": int(final["success"]),
                    "margin": float(final["margin"]),
                    "margin_drop": float(clean["margin"] - final["margin"]),
                    "evals": int(len(states)),
                    "depth": int(len(states)),
                    "linf": float(final["linf"]),
                    "path": "ce",
                }
            )
        gen = torch.Generator(device=device).manual_seed(args.seed + int(row.image_ord) * 1009)
        _x, final, states = random_pixel_attack(wrapper, x0, y, eps, args.max_depth, step_size, gen)
        rows.append(
            {
                "image_ord": int(row.image_ord),
                "dataset_idx": int(row.dataset_idx),
                "label": int(row.label),
                "method": f"random_pixel{args.max_depth}",
                "success": int(final["success"]),
                "margin": float(final["margin"]),
                "margin_drop": float(clean["margin"] - final["margin"]),
                "evals": int(len(states)),
                "depth": int(len(states)),
                "linf": float(final["linf"]),
                "path": "random_pixel",
            }
        )
        for action_set in parse_csv(args.greedy_action_sets):
            _x, final, cand = highway_greedy(wrapper, x0, y, args.layer, basis_t, routes, action_set, eps, step_size, args.max_depth, rng)
            method = f"greedy_{action_set}"
            rows.append(
                {
                    "image_ord": int(row.image_ord),
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "method": method,
                    "success": int(final["success"]),
                    "margin": float(final["margin"]),
                    "margin_drop": float(clean["margin"] - final["margin"]),
                    "evals": int(final["evals"]),
                    "depth": int(final["depth"]),
                    "linf": float(final["linf"]),
                    "path": str(final["path"]),
                }
            )
            for c in cand:
                c.update({"image_ord": int(row.image_ord), "method": method})
            candidate_rows.extend(cand)
        for action_set in parse_csv(args.astar_action_sets):
            _x, final, cand = highway_astar(
                wrapper,
                x0,
                y,
                args.layer,
                basis_t,
                routes,
                action_set,
                eps,
                step_size,
                args.max_depth,
                args.max_expansions,
                args.beam_size,
                args.heuristic_weight,
                rng,
            )
            method = f"astar_{action_set}"
            rows.append(
                {
                    "image_ord": int(row.image_ord),
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "method": method,
                    "success": int(final["success"]),
                    "margin": float(final["margin"]),
                    "margin_drop": float(clean["margin"] - final["margin"]),
                    "evals": int(final["evals"]),
                    "depth": int(final["depth"]),
                    "linf": float(final["linf"]),
                    "path": str(final["path"]),
                }
            )
            for c in cand:
                c.update({"image_ord": int(row.image_ord), "method": method})
            candidate_rows.extend(cand)
        if i % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(out_dir / "partial_astar_highway_from_start_per_image.csv", index=False)
            print(f"[{i}/{len(images)}] rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    candidates = pd.DataFrame(candidate_rows)
    summary = summarize(df)
    deltas = paired_deltas(df, parse_csv(args.delta_baselines))
    routes.to_csv(out_dir / "global_route_ranking.csv", index=False)
    df.to_csv(out_dir / "astar_highway_from_start_per_image.csv", index=False)
    candidates.to_csv(out_dir / "astar_highway_from_start_candidates.csv", index=False)
    summary.to_csv(out_dir / "astar_highway_from_start_summary.csv", index=False)
    deltas.to_csv(out_dir / "astar_highway_from_start_paired_deltas.csv", index=False)
    plot_summary(summary, out_dir)
    meta = vars(args).copy()
    meta.update({"device": str(device), "highway_train_vectors": int(n_highway_train), "n_images": int(len(images))})
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = [
        "# A*-Like Highway Planning From Clean",
        "",
        "Bounded best-first/beam search over signed highway pullback actions from the clean image.",
        "",
        "## Summary",
        "",
    ]
    for r in summary.itertuples():
        lines.append(
            f"- `{r.method}`: ASR={r.asr:.3f}, margin_drop={r.mean_margin_drop:.3f}, "
            f"evals={r.mean_evals:.1f}, depth={r.mean_depth:.2f}"
        )
    (out_dir / "astar_highway_from_start_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/astar_highway_from_start_bbb_resnet50_c200")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--rank-sources", default="pgd,square")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--split", default="all")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--image-ord-csv", default="", help="Optional CSV containing image_ord values to evaluate.")
    p.add_argument("--image-ords", default="", help="Optional comma-separated image_ord list to evaluate.")
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--pgd-steps", default="2,3,5")
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--greedy-action-sets", default="top5,pixelgrad5")
    p.add_argument("--astar-action-sets", default="top5,top10,pixelgrad5")
    p.add_argument("--max-expansions", type=int, default=10)
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--heuristic-weight", type=float, default=1.0)
    p.add_argument("--delta-baselines", default="pgd2,pgd3,pgd5")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
