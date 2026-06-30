#!/usr/bin/env python3
"""Route adversarial motion on an image-free class-highway graph.

This script tests the traffic-routing interpretation directly.  It first uses
an image-free class graph whose directed edges are signed highway routes scored
by classifier-head class-pair margin effect.  Then, for each clean image, it
uses those graph edges as the action set in a bounded state-space planner inside
the same L_inf ball.

No PGD or Square trajectories are used to rank the route actions.  The planner
still uses the true-class margin as the routing objective, so this is a
white-box traffic-routing attack/diagnostic rather than an attack-independent
trajectory statistic.
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
from experiments.pure_af_geometry.evaluate_astar_highway_from_start import (  # noqa: E402
    eval_state,
    feature_tensor,
    load_images,
    pgd_attack,
    random_pixel_attack,
)
from experiments.pure_af_geometry.evaluate_image_conditioned_highway_selector import (  # noqa: E402
    ArtifactStore,
    fit_highway_basis,
)
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import project_linf  # noqa: E402


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


def load_class_graph(pair_gain_path: Path, absolute_gain_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair = pd.read_csv(pair_gain_path)
    abs_gain = pd.read_csv(absolute_gain_path)
    per_source = (
        pair[pair["source_class"].astype(int) >= 0]
        .sort_values(["route", "source_class", "best_class_margin_drop"], ascending=[True, True, False])
        .drop_duplicates(["route", "source_class"])
        .copy()
    )
    keep_cols = ["route", "pc", "sign", "source_class", "best_target_class", "best_class_margin_drop"]
    per_source = per_source[keep_cols]
    per_source["source_class"] = per_source["source_class"].astype(int)
    per_source["best_target_class"] = per_source["best_target_class"].astype(int)
    per_source["class_gain_rank"] = (
        per_source.groupby("source_class")["best_class_margin_drop"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    abs_cols = ["route", "absolute_adv_gain_rank", "mean_best_margin_drop", "logit_effect_l2"]
    per_source = per_source.merge(abs_gain[abs_cols], on="route", how="left")
    global_routes = abs_gain[["route", "pc", "sign", "absolute_adv_gain_rank", "mean_best_margin_drop"]].copy()
    return per_source, global_routes


def route_candidate(wrapper, x0: torch.Tensor, x_cur: torch.Tensor, layer: str, direction: torch.Tensor, eps: float, step_size: float) -> torch.Tensor:
    probe = x_cur.detach().requires_grad_(True)
    h = feature_tensor(wrapper, probe, layer)
    scalar = (h * direction.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, probe)[0]
    return project_linf(x_cur + step_size * grad.sign(), x0, eps).detach()


def route_step(wrapper, x0, x_cur, y, layer, direction, eps, step_size):
    h0 = feature_tensor(wrapper, x_cur, layer).detach()
    x_next = route_candidate(wrapper, x0, x_cur, layer, direction, eps, step_size)
    h1 = feature_tensor(wrapper, x_next, layer).detach()
    dh = h1 - h0
    ev = eval_state(wrapper, x0, x_next, y)
    return x_next, ev, float(torch.norm(dh, dim=1).item())


def select_routes(
    class_graph: pd.DataFrame,
    global_routes: pd.DataFrame,
    true_label: int,
    action_set: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if action_set.startswith("graph_top"):
        k = int(action_set.replace("graph_top", ""))
        pool = class_graph[class_graph.source_class == int(true_label)].copy()
        return pool.sort_values("best_class_margin_drop", ascending=False).head(k).reset_index(drop=True)
    if action_set.startswith("graph_positive_top"):
        k = int(action_set.replace("graph_positive_top", ""))
        pool = class_graph[(class_graph.source_class == int(true_label)) & (class_graph.best_class_margin_drop > 0)].copy()
        return pool.sort_values("best_class_margin_drop", ascending=False).head(k).reset_index(drop=True)
    if action_set.startswith("absolute_top"):
        k = int(action_set.replace("absolute_top", ""))
        return global_routes.sort_values("absolute_adv_gain_rank").head(k).reset_index(drop=True)
    if action_set.startswith("random_graph"):
        k = int(action_set.replace("random_graph", ""))
        pool = class_graph[(class_graph.source_class == int(true_label)) & (class_graph.best_class_margin_drop > 0)].copy()
        if len(pool) == 0:
            pool = class_graph[class_graph.source_class == int(true_label)].copy()
        return pool.sample(n=min(k, len(pool)), random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    if action_set.startswith("random_highway"):
        k = int(action_set.replace("random_highway", ""))
        return global_routes.sample(n=min(k, len(global_routes)), random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    raise ValueError(f"Unknown action set: {action_set}")


@dataclass(order=True)
class Node:
    priority: float
    tie: int
    x: torch.Tensor = field(compare=False)
    margin: float = field(compare=False)
    depth: int = field(compare=False)
    path: tuple[str, ...] = field(compare=False)


def priority(depth: int, margin_value: float, clean_margin: float, heuristic_weight: float) -> float:
    return float(depth + heuristic_weight * max(margin_value, 0.0) / max(abs(clean_margin), 1e-6))


def greedy_route(
    wrapper,
    x0,
    y,
    layer,
    basis_t,
    class_graph,
    global_routes,
    action_set,
    eps,
    step_size,
    max_depth,
    rng,
):
    clean = eval_state(wrapper, x0, x0, y)
    x = x0.detach()
    evals = 0
    path = []
    candidate_rows = []
    current_margin = float(clean["margin"])
    for depth in range(max_depth):
        candidates = select_routes(class_graph, global_routes, int(y.item()), action_set, rng)
        best = None
        best_x = None
        for route in candidates.itertuples(index=False):
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            x_cand, ev, feature_speed = route_step(wrapper, x0, x, y, layer, direction, eps, step_size)
            evals += 1
            row = {
                "depth": depth + 1,
                "route": str(route.route),
                "pc": int(route.pc),
                "sign": int(route.sign),
                "action_set": action_set,
                "margin": float(ev["margin"]),
                "margin_drop": float(current_margin - ev["margin"]),
                "success": int(ev["success"]),
                "feature_speed": feature_speed,
                "class_gain": float(getattr(route, "best_class_margin_drop", np.nan)),
                "absolute_rank": float(getattr(route, "absolute_adv_gain_rank", np.nan)),
                "best_target_class": int(getattr(route, "best_target_class", -1)),
            }
            candidate_rows.append(row)
            if best is None or ev["margin"] < best["margin"]:
                best = {**row, **ev}
                best_x = x_cand
        if best is None:
            break
        x = best_x.detach()
        current_margin = float(best["margin"])
        path.append(str(best["route"]))
        if int(best["success"]):
            break
    final = eval_state(wrapper, x0, x, y)
    return x, {**final, "evals": evals, "depth": len(path), "path": "|".join(path)}, candidate_rows


def astar_route(
    wrapper,
    x0,
    y,
    layer,
    basis_t,
    class_graph,
    global_routes,
    action_set,
    eps,
    step_size,
    max_depth,
    max_expansions,
    beam_size,
    heuristic_weight,
    rng,
):
    clean = eval_state(wrapper, x0, x0, y)
    clean_margin = float(clean["margin"])
    frontier = [Node(priority(0, clean_margin, clean_margin, heuristic_weight), 0, x0.detach(), clean_margin, 0, tuple())]
    best_node = frontier[0]
    counter = 0
    expansions = 0
    evals = 0
    candidate_rows = []
    while frontier and expansions < max_expansions:
        node = heapq.heappop(frontier)
        if node.margin < best_node.margin:
            best_node = node
        if node.margin < 0 or node.depth >= max_depth:
            if node.margin < 0:
                break
            continue
        candidates = select_routes(class_graph, global_routes, int(y.item()), action_set, rng)
        children = []
        for route in candidates.itertuples(index=False):
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            x_cand, ev, feature_speed = route_step(wrapper, x0, node.x, y, layer, direction, eps, step_size)
            evals += 1
            counter += 1
            pr = priority(node.depth + 1, float(ev["margin"]), clean_margin, heuristic_weight)
            path = node.path + (str(route.route),)
            child = Node(pr, counter, x_cand.detach(), float(ev["margin"]), node.depth + 1, path)
            children.append(child)
            candidate_rows.append(
                {
                    "expanded_depth": node.depth,
                    "child_depth": node.depth + 1,
                    "route": str(route.route),
                    "pc": int(route.pc),
                    "sign": int(route.sign),
                    "action_set": action_set,
                    "margin": float(ev["margin"]),
                    "success": int(ev["success"]),
                    "priority": float(pr),
                    "feature_speed": feature_speed,
                    "class_gain": float(getattr(route, "best_class_margin_drop", np.nan)),
                    "absolute_rank": float(getattr(route, "absolute_adv_gain_rank", np.nan)),
                    "best_target_class": int(getattr(route, "best_target_class", -1)),
                }
            )
            if child.margin < best_node.margin:
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
    return best_node.x, {
        **final,
        "evals": evals,
        "expansions": expansions,
        "depth": best_node.depth,
        "path": "|".join(best_node.path),
        "best_priority": float(best_node.priority),
    }, candidate_rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("method", dropna=False)
        .agg(
            n=("success", "size"),
            asr=("success", "mean"),
            mean_margin=("margin", "mean"),
            median_margin=("margin", "median"),
            mean_margin_drop=("margin_drop", "mean"),
            median_margin_drop=("margin_drop", "median"),
            mean_evals=("evals", "mean"),
            median_evals=("evals", "median"),
            mean_depth=("depth", "mean"),
            max_linf=("linf", "max"),
        )
        .reset_index()
        .sort_values(["asr", "mean_margin_drop"], ascending=[False, False])
    )


def paired_deltas(df: pd.DataFrame, baselines: list[str]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    wide_s = df.pivot_table(index="image_ord", columns="method", values="success", aggfunc="first")
    wide_m = df.pivot_table(index="image_ord", columns="method", values="margin_drop", aggfunc="first")
    for baseline in baselines:
        if baseline not in wide_s:
            continue
        for method in [c for c in wide_s.columns if c != baseline]:
            for metric, wide in [("success", wide_s), ("margin_drop", wide_m)]:
                d = (wide[method] - wide[baseline]).dropna().to_numpy(float)
                if len(d) == 0:
                    continue
                boots = []
                for _ in range(3000):
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
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), dpi=180)
    order = summary.sort_values("asr", ascending=False)
    axes[0].bar(order.method, order.asr, color="#4c78a8")
    axes[0].set_ylabel("ASR")
    axes[0].set_title("Graph routing success")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(order.method, order.mean_margin_drop, color="#59a14f")
    axes[1].set_ylabel("Mean margin drop")
    axes[1].set_title("Graph routing progress")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "highway_graph_routing_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "highway_graph_routing_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    store = ArtifactStore(Path(args.input_dir))
    _mean, basis, n_highway_train = fit_highway_basis(store, args.model, args.highway_source, args.layer, args.highway_k, args.seed)
    class_graph, global_routes = load_class_graph(Path(args.class_pair_gains), Path(args.absolute_gains))
    class_graph = class_graph[class_graph.pc <= args.highway_k].copy()
    global_routes = global_routes[global_routes.pc <= args.highway_k].copy()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    images = load_images(store, args.model, args.split, args.images, args.image_ord_csv, args.image_ords)
    wrapper = load_model(args.model, device).eval()
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=device)
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    rng = np.random.default_rng(args.seed)

    rows = []
    cand_rows = []
    for i, row in enumerate(images.itertuples(index=False), start=1):
        x_cpu, _ = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        clean = eval_state(wrapper, x0, x0, y)

        for steps in parse_int_csv(args.pgd_steps):
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

        gen = torch.Generator(device=device).manual_seed(args.seed + int(row.image_ord) * 7919)
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
            _x, final, cand = greedy_route(
                wrapper, x0, y, args.layer, basis_t, class_graph, global_routes,
                action_set, eps, step_size, args.max_depth, rng
            )
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
                c.update({"image_ord": int(row.image_ord), "dataset_idx": int(row.dataset_idx), "label": int(row.label), "method": method})
            cand_rows.extend(cand)

        for action_set in parse_csv(args.astar_action_sets):
            _x, final, cand = astar_route(
                wrapper, x0, y, args.layer, basis_t, class_graph, global_routes,
                action_set, eps, step_size, args.max_depth, args.max_expansions,
                args.beam_size, args.heuristic_weight, rng
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
                c.update({"image_ord": int(row.image_ord), "dataset_idx": int(row.dataset_idx), "label": int(row.label), "method": method})
            cand_rows.extend(cand)

        if i % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(out_dir / "partial_highway_graph_routing_per_image.csv", index=False)
            print(f"[{i}/{len(images)}] rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    cand_df = pd.DataFrame(cand_rows)
    summary = summarize(df)
    deltas = paired_deltas(df, parse_csv(args.delta_baselines))
    class_graph.to_csv(out_dir / "class_highway_graph_edges.csv", index=False)
    global_routes.to_csv(out_dir / "absolute_highway_routes.csv", index=False)
    df.to_csv(out_dir / "highway_graph_routing_per_image.csv", index=False)
    cand_df.to_csv(out_dir / "highway_graph_routing_candidate_edges.csv", index=False)
    summary.to_csv(out_dir / "highway_graph_routing_summary.csv", index=False)
    deltas.to_csv(out_dir / "highway_graph_routing_paired_deltas.csv", index=False)
    plot_summary(summary, out_dir)

    meta = vars(args).copy()
    meta.update({"device": str(device), "n_images": int(len(images)), "highway_train_vectors": int(n_highway_train)})
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Highway Graph Routing",
        "",
        "Routes are ranked by image-free class-pair margin effects from the classifier head, not by PGD/Square trajectory usage.",
        "",
        "## Summary",
        "",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"- `{r.method}`: ASR={r.asr:.3f}, mean_margin_drop={r.mean_margin_drop:.3f}, "
            f"mean_evals={r.mean_evals:.1f}, mean_depth={r.mean_depth:.2f}"
        )
    lines += [
        "",
        "## Interpretation Template",
        "",
        "If graph-ranked routes outperform random-highway routes under the same planner, the class-highway graph contains useful routing information. "
        "If they do not, the graph structure is currently descriptive but not sufficient for routing.",
        "",
    ]
    (out_dir / "highway_graph_routing_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--absolute-gains", default="analysis_outputs/pure_af_geometry/jacobian_null_response/absolute_highway_adv_gain_bbb_resnet50_layer4/absolute_highway_adv_gain.csv")
    p.add_argument("--class-pair-gains", default="analysis_outputs/pure_af_geometry/jacobian_null_response/absolute_highway_adv_gain_bbb_resnet50_layer4/absolute_highway_class_pair_gains.csv")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/highway_graph_routing_bbb_resnet50_c200")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--split", default="all")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--image-ord-csv", default="")
    p.add_argument("--image-ords", default="")
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--pgd-steps", default="2,3,5")
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--greedy-action-sets", default="graph_top5,random_graph5,random_highway5,absolute_top5")
    p.add_argument("--astar-action-sets", default="graph_top5,graph_top10,random_graph5,random_highway5,absolute_top5")
    p.add_argument("--max-expansions", type=int, default=10)
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--heuristic-weight", type=float, default=1.0)
    p.add_argument("--delta-baselines", default="pgd2,pgd3,pgd5,greedy_random_highway5,astar_random_highway5")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
