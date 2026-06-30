#!/usr/bin/env python3
"""Continuous Jacobian-road paths to fixed representation endpoints.

This is the continuous analogue of the road-graph experiment.  After a target
endpoint is fixed, the planner moves from the clean image toward the endpoint
using only:

* the current representation h(x),
* the target representation h_goal,
* the local Jacobian-induced mobility metric.

No margin, class label objective, or attack success signal is used to choose
steps.  Labels are used only to generate/evaluate PGD endpoints for this first
diagnostic.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.evaluate_astar_highway_from_start import eval_state, feature_tensor  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import project_linf  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def feature(wrapper, x: torch.Tensor, layer: str) -> torch.Tensor:
    with torch.no_grad():
        return feature_tensor(wrapper, x, layer).detach()


def ce_step(wrapper, x0: torch.Tensor, x: torch.Tensor, y: torch.Tensor, eps: float, step_size: float) -> torch.Tensor:
    probe = x.detach().requires_grad_(True)
    logits = wrapper(probe)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, probe)[0]
    return project_linf(x + step_size * grad.sign(), x0, eps).detach()


def pgd_endpoint(wrapper, x0: torch.Tensor, y: torch.Tensor, eps: float, step_size: float, steps: int):
    x = x0.detach()
    for i in range(steps):
        x = ce_step(wrapper, x0, x, y, eps, step_size)
        ev = eval_state(wrapper, x0, x, y)
        if int(ev["success"]):
            ev["pgd_steps_used"] = i + 1
            return x.detach(), ev
    ev = eval_state(wrapper, x0, x, y)
    ev["pgd_steps_used"] = steps
    return x.detach(), ev


def pullback_candidate(
    wrapper,
    x0: torch.Tensor,
    x: torch.Tensor,
    layer: str,
    direction: torch.Tensor,
    eps: float,
    step_size: float,
):
    probe = x.detach().requires_grad_(True)
    h = feature_tensor(wrapper, probe, layer)
    direction = direction.view_as(h)
    direction = direction / direction.flatten(1).norm(dim=1, keepdim=True).view(-1, *([1] * (direction.ndim - 1))).clamp_min(1e-12)
    scalar = (h * direction).sum()
    grad = torch.autograd.grad(scalar, probe)[0]
    mobility_l2 = float(torch.norm(grad.flatten(1), p=2, dim=1).item())
    mobility_l1 = float(torch.norm(grad.flatten(1), p=1, dim=1).item())
    mobility_linf = float(grad.abs().max().item())
    x_next = project_linf(x + step_size * grad.sign(), x0, eps).detach()
    h_next = feature(wrapper, x_next, layer)
    h_cur = h.detach()
    dh = h_next - h_cur
    feature_step = float(torch.norm(dh.flatten(1), dim=1).item())
    road_cost = feature_step / max(mobility_l2, 1e-12)
    road_score = mobility_l2 / max(feature_step, 1e-12)
    return x_next, h_next, {
        "mobility_l2": mobility_l2,
        "mobility_l1": mobility_l1,
        "mobility_linf": mobility_linf,
        "feature_step": feature_step,
        "road_cost": road_cost,
        "road_score": road_score,
    }


def direct_diagnostic(wrapper, x0: torch.Tensor, h0: torch.Tensor, h_goal: torch.Tensor, layer: str) -> dict:
    dh = h_goal - h0
    if float(torch.norm(dh.flatten(1), dim=1).item()) < 1e-12:
        return {"direct_feature_dist": 0.0, "direct_mobility_l2": np.nan, "direct_road_cost": np.nan, "direct_road_score": np.nan}
    probe = x0.detach().requires_grad_(True)
    h = feature_tensor(wrapper, probe, layer)
    direction = dh / dh.flatten(1).norm(dim=1, keepdim=True).view(-1, *([1] * (dh.ndim - 1))).clamp_min(1e-12)
    scalar = (h * direction.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, probe)[0]
    mobility_l2 = float(torch.norm(grad.flatten(1), p=2, dim=1).item())
    feature_dist = float(torch.norm(dh.flatten(1), dim=1).item())
    return {
        "direct_feature_dist": feature_dist,
        "direct_mobility_l2": mobility_l2,
        "direct_road_cost": feature_dist / max(mobility_l2, 1e-12),
        "direct_road_score": mobility_l2 / max(feature_dist, 1e-12),
    }


def run_target_geodesic(wrapper, x0, y, layer, h_goal, eps, step_size, steps):
    x = x0.detach()
    path_rows = []
    h = feature(wrapper, x, layer)
    start_dist = float(torch.norm((h_goal - h).flatten(1), dim=1).item())
    total_cost = 0.0
    total_feature_step = 0.0
    for step in range(steps):
        h = feature(wrapper, x, layer)
        direction = h_goal - h
        before_dist = float(torch.norm(direction.flatten(1), dim=1).item())
        x_next, h_next, extra = pullback_candidate(wrapper, x0, x, layer, direction, eps, step_size)
        after_dist = float(torch.norm((h_goal - h_next).flatten(1), dim=1).item())
        progress = before_dist - after_dist
        total_cost += extra["road_cost"]
        total_feature_step += extra["feature_step"]
        ev = eval_state(wrapper, x0, x_next, y)
        path_rows.append(
            {
                "step": step + 1,
                "chosen": "target_direction",
                "goal_dist_before": before_dist,
                "goal_dist_after": after_dist,
                "goal_progress": progress,
                "progress_per_cost": progress / max(extra["road_cost"], 1e-12),
                "pred": int(ev["pred"]),
                "success": int(ev["success"]),
                **extra,
            }
        )
        x = x_next
    final_h = feature(wrapper, x, layer)
    final_dist = float(torch.norm((h_goal - final_h).flatten(1), dim=1).item())
    final_ev = eval_state(wrapper, x0, x, y)
    return x, {
        "endpoint_feature_dist": final_dist,
        "feature_dist_reduction": start_dist - final_dist,
        "fraction_goal_distance_closed": (start_dist - final_dist) / max(start_dist, 1e-12),
        "path_cost": total_cost,
        "path_feature_step": total_feature_step,
        "success": int(final_ev["success"]),
        "pred": int(final_ev["pred"]),
        "linf": float(final_ev["linf"]),
    }, path_rows


def run_metric_mpc(wrapper, x0, y, layer, h_goal, eps, step_size, steps, candidates, noise_scale, gen):
    x = x0.detach()
    h = feature(wrapper, x, layer)
    start_dist = float(torch.norm((h_goal - h).flatten(1), dim=1).item())
    total_cost = 0.0
    total_feature_step = 0.0
    path_rows = []
    for step in range(steps):
        h = feature(wrapper, x, layer)
        target = h_goal - h
        before_dist = float(torch.norm(target.flatten(1), dim=1).item())
        target_flat = target.flatten(1)
        target_dir = target_flat / target_flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        dirs = [target_dir]
        d = target_dir.shape[1]
        for _ in range(max(0, candidates - 1)):
            noise = torch.randn((1, d), generator=gen, device=x.device)
            cand = target_dir + noise_scale * noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cand = cand / cand.norm(dim=1, keepdim=True).clamp_min(1e-12)
            dirs.append(cand)
        best = None
        for j, direction_flat in enumerate(dirs):
            x_cand, h_cand, extra = pullback_candidate(wrapper, x0, x, layer, direction_flat.view_as(h), eps, step_size)
            after_dist = float(torch.norm((h_goal - h_cand).flatten(1), dim=1).item())
            progress = before_dist - after_dist
            score = progress / max(extra["road_cost"], 1e-12)
            if best is None or score > best["score"]:
                best = {
                    "idx": j,
                    "x": x_cand,
                    "h": h_cand,
                    "after_dist": after_dist,
                    "progress": progress,
                    "score": score,
                    **extra,
                }
        x = best["x"]
        total_cost += best["road_cost"]
        total_feature_step += best["feature_step"]
        ev = eval_state(wrapper, x0, x, y)
        path_rows.append(
            {
                "step": step + 1,
                "chosen": "target_direction" if best["idx"] == 0 else "sampled_direction",
                "chosen_idx": int(best["idx"]),
                "goal_dist_before": before_dist,
                "goal_dist_after": float(best["after_dist"]),
                "goal_progress": float(best["progress"]),
                "progress_per_cost": float(best["score"]),
                "pred": int(ev["pred"]),
                "success": int(ev["success"]),
                **{k: best[k] for k in ["mobility_l2", "mobility_l1", "mobility_linf", "feature_step", "road_cost", "road_score"]},
            }
        )
    final_h = feature(wrapper, x, layer)
    final_dist = float(torch.norm((h_goal - final_h).flatten(1), dim=1).item())
    final_ev = eval_state(wrapper, x0, x, y)
    return x, {
        "endpoint_feature_dist": final_dist,
        "feature_dist_reduction": start_dist - final_dist,
        "fraction_goal_distance_closed": (start_dist - final_dist) / max(start_dist, 1e-12),
        "path_cost": total_cost,
        "path_feature_step": total_feature_step,
        "success": int(final_ev["success"]),
        "pred": int(final_ev["pred"]),
        "linf": float(final_ev["linf"]),
    }, path_rows


def run_pixel_interpolation(wrapper, x0, y, layer, x_goal, h_goal, eps, steps):
    x = x0.detach()
    h0 = feature(wrapper, x, layer)
    start_dist = float(torch.norm((h_goal - h0).flatten(1), dim=1).item())
    total_feature_step = 0.0
    path_rows = []
    for step in range(steps):
        alpha = (step + 1) / max(steps, 1)
        prev_h = feature(wrapper, x, layer)
        x = project_linf(x0 + alpha * (x_goal - x0), x0, eps).detach()
        h = feature(wrapper, x, layer)
        before_dist = float(torch.norm((h_goal - prev_h).flatten(1), dim=1).item())
        after_dist = float(torch.norm((h_goal - h).flatten(1), dim=1).item())
        feature_step = float(torch.norm((h - prev_h).flatten(1), dim=1).item())
        total_feature_step += feature_step
        ev = eval_state(wrapper, x0, x, y)
        path_rows.append(
            {
                "step": step + 1,
                "chosen": "pixel_interpolation",
                "goal_dist_before": before_dist,
                "goal_dist_after": after_dist,
                "goal_progress": before_dist - after_dist,
                "feature_step": feature_step,
                "pred": int(ev["pred"]),
                "success": int(ev["success"]),
            }
        )
    final_h = feature(wrapper, x, layer)
    final_dist = float(torch.norm((h_goal - final_h).flatten(1), dim=1).item())
    final_ev = eval_state(wrapper, x0, x, y)
    return x, {
        "endpoint_feature_dist": final_dist,
        "feature_dist_reduction": start_dist - final_dist,
        "fraction_goal_distance_closed": (start_dist - final_dist) / max(start_dist, 1e-12),
        "path_cost": np.nan,
        "path_feature_step": total_feature_step,
        "success": int(final_ev["success"]),
        "pred": int(final_ev["pred"]),
        "linf": float(final_ev["linf"]),
    }, path_rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("method", dropna=False)
        .agg(
            n=("image_ord", "size"),
            endpoint_asr=("endpoint_success", "mean"),
            final_asr=("success", "mean"),
            mean_fraction_closed=("fraction_goal_distance_closed", "mean"),
            median_fraction_closed=("fraction_goal_distance_closed", "median"),
            mean_endpoint_feature_dist=("endpoint_feature_dist", "mean"),
            median_endpoint_feature_dist=("endpoint_feature_dist", "median"),
            mean_path_cost=("path_cost", "mean"),
            median_path_cost=("path_cost", "median"),
            mean_path_feature_step=("path_feature_step", "mean"),
            mean_direct_road_cost=("direct_road_cost", "mean"),
            mean_linf=("linf", "mean"),
        )
        .reset_index()
        .sort_values(["mean_fraction_closed", "final_asr"], ascending=[False, False])
    )


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=180)
    order = summary.sort_values("mean_fraction_closed", ascending=False)
    axes[0].bar(order["method"], order["mean_fraction_closed"], color="#4c78a8")
    axes[0].set_ylabel("mean fraction of target feature distance closed")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    order2 = summary.sort_values("final_asr", ascending=False)
    axes[1].bar(order2["method"], order2["final_asr"], color="#59a14f")
    axes[1].set_ylabel("final ASR")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "continuous_road_metric_paths_summary.png")
    fig.savefig(out_dir / "continuous_road_metric_paths_summary.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/attack_independent_road_graph_bbb_resnet50_c100"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/continuous_road_metric_paths_bbb_resnet50_c100"))
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--pgd-steps", type=int, default=5)
    p.add_argument("--path-steps", default="3,5,10")
    p.add_argument("--mpc-candidates", type=int, default=8)
    p.add_argument("--mpc-noise-scale", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    wrapper = load_model(args.model, device).eval()
    nodes = pd.read_csv(args.graph_dir / "road_graph_nodes.csv")
    arrays = np.load(args.graph_dir / "road_graph_arrays.npz")
    pixels = arrays["pixels"].astype(np.float32)
    clean_nodes = nodes[nodes.state_kind == "clean"].copy().sort_values("image_ord")
    if args.images > 0:
        clean_nodes = clean_nodes.head(args.images)
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    gen = torch.Generator(device=device).manual_seed(args.seed)

    rows = []
    path_rows = []
    endpoint_rows = []
    for i, row in enumerate(clean_nodes.itertuples(index=False), start=1):
        x0 = torch.as_tensor(pixels[int(row.node_id):int(row.node_id) + 1], dtype=torch.float32, device=device)
        y = torch.tensor([int(row.label)], device=device)
        x_goal, endpoint_ev = pgd_endpoint(wrapper, x0, y, eps, step_size, args.pgd_steps)
        h0 = feature(wrapper, x0, args.layer)
        h_goal = feature(wrapper, x_goal, args.layer)
        direct = direct_diagnostic(wrapper, x0, h0, h_goal, args.layer)
        endpoint_rows.append(
            {
                "image_ord": int(row.image_ord),
                "node_id": int(row.node_id),
                "label": int(row.label),
                "endpoint_success": int(endpoint_ev["success"]),
                "endpoint_pred": int(endpoint_ev["pred"]),
                "endpoint_pgd_steps_used": int(endpoint_ev["pgd_steps_used"]),
                **direct,
            }
        )
        for steps in parse_int_csv(args.path_steps):
            for method_name, runner in [
                ("target_geodesic", run_target_geodesic),
                ("metric_mpc", run_metric_mpc),
                ("pixel_interpolation", run_pixel_interpolation),
            ]:
                if method_name == "target_geodesic":
                    _x, metrics, pr = runner(wrapper, x0, y, args.layer, h_goal, eps, step_size, steps)
                elif method_name == "metric_mpc":
                    _x, metrics, pr = runner(
                        wrapper,
                        x0,
                        y,
                        args.layer,
                        h_goal,
                        eps,
                        step_size,
                        steps,
                        args.mpc_candidates,
                        args.mpc_noise_scale,
                        gen,
                    )
                else:
                    _x, metrics, pr = runner(wrapper, x0, y, args.layer, x_goal, h_goal, eps, steps)
                method = f"{method_name}_{steps}"
                rows.append(
                    {
                        "image_ord": int(row.image_ord),
                        "node_id": int(row.node_id),
                        "label": int(row.label),
                        "method": method,
                        "path_steps": int(steps),
                        "endpoint_success": int(endpoint_ev["success"]),
                        "endpoint_pred": int(endpoint_ev["pred"]),
                        "endpoint_pgd_steps_used": int(endpoint_ev["pgd_steps_used"]),
                        **direct,
                        **metrics,
                    }
                )
                for r in pr:
                    r.update({"image_ord": int(row.image_ord), "method": method, "path_steps": int(steps)})
                path_rows.extend(pr)
        if i % 20 == 0:
            print(f"[{i}/{len(clean_nodes)}] rows={len(rows)}", flush=True)

    per = pd.DataFrame(rows)
    path = pd.DataFrame(path_rows)
    endpoints = pd.DataFrame(endpoint_rows)
    summary = summarize(per)
    per.to_csv(args.output_dir / "continuous_road_metric_paths_per_image.csv", index=False)
    path.to_csv(args.output_dir / "continuous_road_metric_path_steps.csv", index=False)
    endpoints.to_csv(args.output_dir / "continuous_road_metric_endpoints.csv", index=False)
    summary.to_csv(args.output_dir / "continuous_road_metric_paths_summary.csv", index=False)
    plot_summary(summary, args.output_dir)
    metadata = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    metadata.update(
        {
            "planning_uses": ["target representation", "local Jacobian mobility"],
            "planning_excludes": ["margin", "attack loss", "success/failure signal"],
            "endpoint_source": "PGD endpoint for first diagnostic only",
        }
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Continuous Road-Metric Paths",
        "",
        "Planner steps use only target representation and local Jacobian mobility. Margins and attack success are not used for step selection.",
        "",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"- `{r.method}`: closed={r.mean_fraction_closed:.3f}, ASR={r.final_asr:.3f}, "
            f"path_cost={r.mean_path_cost:.4f}, endpoint_dist={r.mean_endpoint_feature_dist:.4f}"
        )
    (args.output_dir / "continuous_road_metric_paths_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
