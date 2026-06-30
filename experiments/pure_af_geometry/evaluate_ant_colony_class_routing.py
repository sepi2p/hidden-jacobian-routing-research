#!/usr/bin/env python3
"""Ant-colony routing from an image to another class region.

This is a deliberately non-attack planner.  Ants start at a clean image and
try to reach a target class "city" represented by clean hidden features.  The
transition graph is implicit: at each state, ants choose among signed
high-mobility representation directions, pull the direction back to pixels
through the local Jacobian, and take an L_inf-projected step.

Planning uses:
  * target class representation centroid/prototype;
  * local Jacobian mobility/progress;
  * ant-colony pheromone over signed road directions.

Planning excludes:
  * CE loss and target loss;
  * true-class or target-class margins;
  * PGD/Square/GA trajectories;
  * adversarial endpoints;
  * class-pair highway gains from the classifier head.
"""

from __future__ import annotations

import argparse
import json
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
from experiments.pure_af_geometry.evaluate_astar_highway_from_start import feature_tensor  # noqa: E402
from experiments.pure_af_geometry.evaluate_class_to_class_road_paths import (  # noqa: E402
    collect_clean_prototypes,
    target_feature_for_source,
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


def parse_targets(label: int, spec: str) -> list[int]:
    if spec == "next":
        return [(label + 1) % 10]
    if spec == "all":
        return [c for c in range(10) if c != label]
    if spec.startswith("fixed:"):
        t = int(spec.split(":", 1)[1])
        return [t] if t != label else []
    return [int(x.strip()) for x in spec.split(",") if x.strip() and int(x.strip()) != label]


def eval_state(wrapper, x0: torch.Tensor, x: torch.Tensor, y: int, target: int, h_goal: torch.Tensor, layer: str) -> dict:
    with torch.no_grad():
        logits, feats, _raw = wrapper.forward_with_features(x)
        pred = int(logits.argmax(1).item())
        h = feats[layer].detach()
        dist = float(torch.norm(h - h_goal, dim=1).item())
        probs = torch.softmax(logits, dim=1)
    return {
        "pred": pred,
        "target_hit": int(pred == int(target)),
        "class_changed": int(pred != int(y)),
        "target_prob": float(probs[0, int(target)].item()),
        "true_prob": float(probs[0, int(y)].item()),
        "target_dist": dist,
        "linf": float((x - x0).abs().max().item()),
    }


def route_step(
    wrapper,
    x0: torch.Tensor,
    x_cur: torch.Tensor,
    layer: str,
    direction: torch.Tensor,
    h_goal: torch.Tensor,
    eps: float,
    step_size: float,
) -> tuple[torch.Tensor, dict]:
    with torch.no_grad():
        h_before = feature_tensor(wrapper, x_cur, layer).detach()
        dist_before = float(torch.norm(h_before - h_goal, dim=1).item())
    probe = x_cur.detach().requires_grad_(True)
    h = feature_tensor(wrapper, probe, layer)
    scalar = (h * direction.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, probe)[0]
    x_next = project_linf(x_cur + step_size * grad.sign(), x0, eps).detach()
    with torch.no_grad():
        h_after = feature_tensor(wrapper, x_next, layer).detach()
        dh = h_after - h_before
        speed = float(torch.norm(dh, dim=1).item())
        dist_after = float(torch.norm(h_after - h_goal, dim=1).item())
        progress = dist_before - dist_after
        align = float((dh * direction.view_as(dh)).sum().item() / max(speed, 1e-12))
    return x_next, {
        "dist_before": dist_before,
        "dist_after": dist_after,
        "progress": float(progress),
        "feature_speed": speed,
        "realized_alignment": align,
    }


def select_clean_correct_images(dataset, wrapper, device, max_per_class: int, max_total: int) -> pd.DataFrame:
    rows = []
    counts = {c: 0 for c in range(10)}
    for idx in range(len(dataset)):
        x_cpu, label = dataset[idx]
        label = int(label)
        if counts[label] >= max_per_class:
            continue
        x = x_cpu.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(wrapper(x).argmax(1).item())
        if pred == label:
            rows.append({"dataset_idx": int(idx), "label": label, "class_ord": counts[label]})
            counts[label] += 1
        if sum(counts.values()) >= max_total or all(v >= max_per_class for v in counts.values()):
            break
    return pd.DataFrame(rows)


def make_routes(basis: np.ndarray, candidate_k: int) -> pd.DataFrame:
    rows = []
    for pc in range(1, min(candidate_k, basis.shape[0]) + 1):
        rows.append({"route_id": len(rows), "route": f"pc{pc}+", "pc": pc, "sign": 1})
        rows.append({"route_id": len(rows), "route": f"pc{pc}-", "pc": pc, "sign": -1})
    return pd.DataFrame(rows)


def score_candidate(stats: dict, mode: str) -> float:
    progress = float(stats["progress"])
    speed = float(stats["feature_speed"])
    if mode == "progress":
        return max(progress, 0.0) + 1e-9
    if mode == "mobility_progress":
        return (max(progress, 0.0) + 1e-9) * (speed + 1e-9)
    if mode == "inverse_distance":
        return 1.0 / (float(stats["dist_after"]) + 1e-6)
    raise ValueError(f"Unknown desirability mode: {mode}")


def run_greedy(
    wrapper,
    x0: torch.Tensor,
    y: int,
    target: int,
    h_goal: torch.Tensor,
    layer: str,
    basis_t: torch.Tensor,
    routes: pd.DataFrame,
    eps: float,
    step_size: float,
    max_depth: int,
    mode: str,
) -> tuple[dict, list[dict]]:
    x = x0.detach()
    path = []
    for depth in range(1, max_depth + 1):
        candidates = []
        for route in routes.itertuples(index=False):
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            x_next, stats = route_step(wrapper, x0, x, layer, direction, h_goal, eps, step_size)
            stats.update({"depth": depth, "route": str(route.route), "route_id": int(route.route_id), "x_next": x_next})
            stats["desirability"] = score_candidate(stats, mode)
            candidates.append(stats)
        best = max(candidates, key=lambda r: r["desirability"])
        x = best.pop("x_next").detach()
        path.append(best)
        ev = eval_state(wrapper, x0, x, y, target, h_goal, layer)
        if ev["target_hit"]:
            break
    final = eval_state(wrapper, x0, x, y, target, h_goal, layer)
    return {**final, "depth": len(path), "evals": len(path) * len(routes), "path": "|".join(r["route"] for r in path)}, path


def run_random(
    wrapper,
    x0: torch.Tensor,
    y: int,
    target: int,
    h_goal: torch.Tensor,
    layer: str,
    basis_t: torch.Tensor,
    routes: pd.DataFrame,
    eps: float,
    step_size: float,
    max_depth: int,
    rng: np.random.Generator,
) -> tuple[dict, list[dict]]:
    x = x0.detach()
    path = []
    for depth in range(1, max_depth + 1):
        route = routes.iloc[int(rng.integers(0, len(routes)))]
        direction = int(route.sign) * basis_t[int(route.pc) - 1]
        x, stats = route_step(wrapper, x0, x, layer, direction, h_goal, eps, step_size)
        stats.update({"depth": depth, "route": str(route.route), "route_id": int(route.route_id)})
        path.append(stats)
        ev = eval_state(wrapper, x0, x, y, target, h_goal, layer)
        if ev["target_hit"]:
            break
    final = eval_state(wrapper, x0, x, y, target, h_goal, layer)
    return {**final, "depth": len(path), "evals": len(path), "path": "|".join(r["route"] for r in path)}, path


def run_aco(
    wrapper,
    x0: torch.Tensor,
    y: int,
    target: int,
    h_goal: torch.Tensor,
    layer: str,
    basis_t: torch.Tensor,
    routes: pd.DataFrame,
    eps: float,
    step_size: float,
    max_depth: int,
    n_ants: int,
    n_iters: int,
    alpha: float,
    beta: float,
    evaporation: float,
    deposit_scale: float,
    mode: str,
    rng: np.random.Generator,
) -> tuple[dict, list[dict]]:
    n_routes = len(routes)
    pheromone = np.ones(n_routes, dtype=np.float64)
    best_final = None
    best_path: list[dict] = []
    best_reward = -np.inf
    clean = eval_state(wrapper, x0, x0, y, target, h_goal, layer)
    start_dist = max(float(clean["target_dist"]), 1e-9)
    trace_rows = []

    for iteration in range(1, n_iters + 1):
        iter_paths = []
        for ant in range(n_ants):
            x = x0.detach()
            path: list[dict] = []
            for depth in range(1, max_depth + 1):
                candidates = []
                for route in routes.itertuples(index=False):
                    direction = int(route.sign) * basis_t[int(route.pc) - 1]
                    x_next, stats = route_step(wrapper, x0, x, layer, direction, h_goal, eps, step_size)
                    desirability = score_candidate(stats, mode)
                    rid = int(route.route_id)
                    score = (pheromone[rid] ** alpha) * (desirability ** beta)
                    candidates.append(
                        {
                            **stats,
                            "route_id": rid,
                            "route": str(route.route),
                            "depth": depth,
                            "score": float(score),
                            "desirability": float(desirability),
                            "x_next": x_next,
                        }
                    )
                probs = np.array([c["score"] for c in candidates], dtype=np.float64)
                if not np.isfinite(probs).all() or probs.sum() <= 0:
                    probs = np.ones(len(candidates), dtype=np.float64) / len(candidates)
                else:
                    probs = probs / probs.sum()
                chosen = candidates[int(rng.choice(len(candidates), p=probs))]
                x = chosen.pop("x_next").detach()
                path.append(chosen)
                ev = eval_state(wrapper, x0, x, y, target, h_goal, layer)
                if ev["target_hit"]:
                    break
            final = eval_state(wrapper, x0, x, y, target, h_goal, layer)
            fraction_closed = (start_dist - float(final["target_dist"])) / start_dist
            reward = float(fraction_closed + 2.0 * final["target_hit"] + 0.25 * final["class_changed"])
            iter_paths.append((reward, path, final))
            if reward > best_reward:
                best_reward = reward
                best_path = path
                best_final = final

        pheromone *= 1.0 - evaporation
        if iter_paths:
            rewards = np.array([r for r, _p, _f in iter_paths], dtype=np.float64)
            # Deposit only from the upper half to reduce noise.
            cutoff = np.quantile(rewards, 0.5)
            for reward, path, _final in iter_paths:
                if reward < cutoff:
                    continue
                dep = deposit_scale * max(reward, 0.0)
                for step in path:
                    pheromone[int(step["route_id"])] += dep
        trace_rows.append(
            {
                "iteration": iteration,
                "best_reward_so_far": float(best_reward),
                "mean_iteration_reward": float(np.mean([r for r, _p, _f in iter_paths])) if iter_paths else np.nan,
                "max_pheromone": float(pheromone.max()),
                "min_pheromone": float(pheromone.min()),
            }
        )

    assert best_final is not None
    return {
        **best_final,
        "depth": len(best_path),
        "evals": int(n_iters * n_ants * max_depth * n_routes),
        "path": "|".join(r["route"] for r in best_path),
        "best_reward": float(best_reward),
    }, [{**r, "selected_by": "best_path"} for r in best_path] + trace_rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["method", "target_mode", "target_spec"], dropna=False)
        .agg(
            n=("image_ord", "size"),
            target_hit_rate=("target_hit", "mean"),
            class_changed_rate=("class_changed", "mean"),
            mean_fraction_closed=("fraction_closed", "mean"),
            median_fraction_closed=("fraction_closed", "median"),
            mean_target_dist=("target_dist", "mean"),
            mean_target_prob=("target_prob", "mean"),
            mean_evals=("evals", "mean"),
            mean_depth=("depth", "mean"),
            mean_linf=("linf", "mean"),
        )
        .reset_index()
        .sort_values(["target_hit_rate", "mean_fraction_closed"], ascending=[False, False])
    )


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    order = summary.sort_values(["target_hit_rate", "mean_fraction_closed"], ascending=False)
    labels = order["method"] + "\n" + order["target_mode"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), dpi=200)
    axes[0].bar(labels, order["target_hit_rate"], color="#4c78a8")
    axes[0].set_ylabel("Target class hit rate")
    axes[0].set_title("Ant-colony class-region routing")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, order["mean_fraction_closed"], color="#59a14f")
    axes[1].set_ylabel("Mean target distance closed")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "ant_colony_class_routing_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "ant_colony_class_routing_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/ant_colony_class_routing_bbb_resnet50_c100_next")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--candidate-k", type=int, default=10)
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--images-per-class", type=int, default=10)
    p.add_argument("--targets", default="next")
    p.add_argument("--target-modes", default="centroid")
    p.add_argument("--prototypes-per-class", type=int, default=80)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--ants", type=int, default=8)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--evaporation", type=float, default=0.35)
    p.add_argument("--deposit-scale", type=float, default=0.25)
    p.add_argument("--desirability", default="mobility_progress")
    p.add_argument("--methods", default="random,greedy,aco")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    wrapper = load_model(args.model, device).eval()
    store = ArtifactStore(Path(args.input_dir))
    _mean, basis, n_highway_train = fit_highway_basis(
        store, args.model, args.highway_source, args.layer, args.highway_k, args.seed
    )
    routes = make_routes(basis, args.candidate_k)
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    images = select_clean_correct_images(dataset, wrapper, device, args.images_per_class, args.images)
    centroids, prototypes, prototype_pixels, prototype_meta = collect_clean_prototypes(
        wrapper, dataset, args.layer, args.prototypes_per_class, args.batch_size, device
    )
    images.to_csv(out_dir / "ant_colony_eval_images.csv", index=False)
    prototype_meta.to_csv(out_dir / "ant_colony_class_prototypes.csv", index=False)
    np.savez_compressed(
        out_dir / "ant_colony_class_prototypes.npz",
        **{f"class_{c}_features": prototypes[c] for c in range(10)},
        **{f"class_{c}_pixels": prototype_pixels[c] for c in range(10)},
        **{f"class_{c}_centroid": centroids[c] for c in range(10)},
    )

    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    rng = np.random.default_rng(args.seed)
    methods = parse_csv(args.methods)
    target_modes = parse_csv(args.target_modes)
    rows = []
    trace_rows = []

    for image_i, image in enumerate(images.itertuples(index=False), start=1):
        x_cpu, _label = dataset[int(image.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        h0 = feature_tensor(wrapper, x0, args.layer).detach().cpu().numpy()[0].astype(np.float32)
        for target_class in parse_targets(int(image.label), args.targets):
            for target_mode in target_modes:
                h_goal_np, proto_idx = target_feature_for_source(h0, target_class, centroids, prototypes, target_mode)
                h_goal = torch.as_tensor(h_goal_np.reshape(1, -1), dtype=torch.float32, device=device)
                clean = eval_state(wrapper, x0, x0, int(image.label), target_class, h_goal, args.layer)
                start_dist = max(float(clean["target_dist"]), 1e-9)
                for method in methods:
                    if method == "random":
                        final, trace = run_random(
                            wrapper, x0, int(image.label), target_class, h_goal, args.layer, basis_t,
                            routes, eps, step_size, args.max_depth, rng
                        )
                    elif method == "greedy":
                        final, trace = run_greedy(
                            wrapper, x0, int(image.label), target_class, h_goal, args.layer, basis_t,
                            routes, eps, step_size, args.max_depth, args.desirability
                        )
                    elif method == "aco":
                        final, trace = run_aco(
                            wrapper, x0, int(image.label), target_class, h_goal, args.layer, basis_t,
                            routes, eps, step_size, args.max_depth, args.ants, args.iterations,
                            args.alpha, args.beta, args.evaporation, args.deposit_scale,
                            args.desirability, rng
                        )
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    fraction_closed = (start_dist - float(final["target_dist"])) / start_dist
                    method_name = f"{method}_{args.desirability}_d{args.max_depth}"
                    rows.append(
                        {
                            "image_ord": image_i - 1,
                            "dataset_idx": int(image.dataset_idx),
                            "source_class": int(image.label),
                            "target_class": int(target_class),
                            "target_spec": args.targets,
                            "target_mode": target_mode,
                            "prototype_idx": int(proto_idx),
                            "method": method_name,
                            "start_target_dist": float(start_dist),
                            "fraction_closed": float(fraction_closed),
                            **final,
                        }
                    )
                    for tr in trace:
                        tr.update(
                            {
                                "image_ord": image_i - 1,
                                "dataset_idx": int(image.dataset_idx),
                                "source_class": int(image.label),
                                "target_class": int(target_class),
                                "target_mode": target_mode,
                                "method": method_name,
                            }
                        )
                        trace_rows.append(tr)
        if image_i % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(out_dir / "partial_ant_colony_class_routing_per_image.csv", index=False)
            print(f"[{image_i}/{len(images)}] rows={len(rows)}", flush=True)

    per = pd.DataFrame(rows)
    trace = pd.DataFrame(trace_rows)
    summary = summarize(per)
    per.to_csv(out_dir / "ant_colony_class_routing_per_image.csv", index=False)
    trace.to_csv(out_dir / "ant_colony_class_routing_trace.csv", index=False)
    routes.to_csv(out_dir / "ant_colony_candidate_routes.csv", index=False)
    summary.to_csv(out_dir / "ant_colony_class_routing_summary.csv", index=False)
    plot_summary(summary, out_dir)
    metadata = vars(args).copy()
    metadata.update(
        {
            "device": str(device),
            "highway_train_vectors": int(n_highway_train),
            "planning_uses": [
                "clean target-class representation region",
                "local Jacobian pullback mobility",
                "ant-colony pheromone over signed high-mobility roads",
            ],
            "planning_excludes": [
                "cross-entropy loss",
                "true/target class margin",
                "PGD/Square/GA trajectories",
                "adversarial endpoints",
                "classifier-head class-pair highway gains",
            ],
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Ant-Colony Class Routing",
        "",
        "Ants start from clean images and route toward clean target-class representation regions using local Jacobian road quality and pheromone. No adversarial loss, margin, attack endpoint, or classifier-head class-pair gain is used during planning.",
        "",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"- `{r.method}`/{r.target_mode}: target_hit={r.target_hit_rate:.3f}, "
            f"class_changed={r.class_changed_rate:.3f}, closed={r.mean_fraction_closed:.3f}, "
            f"evals={r.mean_evals:.1f}"
        )
    (out_dir / "ant_colony_class_routing_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
