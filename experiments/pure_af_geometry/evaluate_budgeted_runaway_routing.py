#!/usr/bin/env python3
"""Budget-aware run-away routing through representation highways.

Unlike class-to-class routing, this experiment has no destination city.  The
objective is to spend a bounded L_inf perturbation budget to reduce the true
class confidence/margin as efficiently as possible.

Planning uses the model output objective because this is now an adversarial
routing problem.  The comparison of interest is whether budget-aware route
selection over Jacobian highway moves behaves differently from pure mobility,
random highway routing, and ordinary CE-PGD.
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
from experiments.pure_af_geometry.evaluate_astar_highway_from_start import feature_tensor, load_images  # noqa: E402
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


def parse_float_csv(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def source_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    yy = int(y.item())
    true_logit = logits[:, yy]
    masked = logits.clone()
    masked[:, yy] = -1e9
    return true_logit - masked.max(dim=1).values


def eval_source(wrapper, x0: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> dict:
    with torch.no_grad():
        logits = wrapper(x)
        probs = torch.softmax(logits, dim=1)
        pred = int(logits.argmax(1).item())
        py = float(probs[0, int(y.item())].item())
        m = float(source_margin(logits, y).item())
    return {
        "pred": pred,
        "success": int(pred != int(y.item())),
        "margin": m,
        "p_y": py,
        "linf": float((x - x0).abs().max().item()),
    }


def make_routes(basis: np.ndarray, candidate_k: int) -> pd.DataFrame:
    rows = []
    for pc in range(1, min(candidate_k, basis.shape[0]) + 1):
        rows.append({"route_id": len(rows), "route": f"pc{pc}+", "pc": pc, "sign": 1})
        rows.append({"route_id": len(rows), "route": f"pc{pc}-", "pc": pc, "sign": -1})
    return pd.DataFrame(rows)


def ce_step(wrapper, x0: torch.Tensor, x: torch.Tensor, y: torch.Tensor, eps: float, step_size: float) -> torch.Tensor:
    probe = x.detach().requires_grad_(True)
    logits = wrapper(probe)
    loss = torch.nn.functional.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, probe)[0]
    return project_linf(x + step_size * grad.sign(), x0, eps).detach()


def pgd_attack(wrapper, x0: torch.Tensor, y: torch.Tensor, eps: float, step_size: float, steps: int) -> dict:
    clean = eval_source(wrapper, x0, x0, y)
    x = x0.detach()
    depth = 0
    for _ in range(steps):
        x = ce_step(wrapper, x0, x, y, eps, step_size)
        depth += 1
        if eval_source(wrapper, x0, x, y)["success"]:
            break
    final = eval_source(wrapper, x0, x, y)
    return {
        **final,
        "margin_drop": float(clean["margin"] - final["margin"]),
        "p_y_drop": float(clean["p_y"] - final["p_y"]),
        "depth": depth,
        "evals": depth,
        "path": "ce",
    }


def random_pixel(wrapper, x0: torch.Tensor, y: torch.Tensor, eps: float, step_size: float, steps: int, gen) -> dict:
    clean = eval_source(wrapper, x0, x0, y)
    x = x0.detach()
    depth = 0
    for _ in range(steps):
        direction = torch.randn(x.shape, generator=gen, device=x.device).sign()
        x = project_linf(x + step_size * direction, x0, eps).detach()
        depth += 1
        if eval_source(wrapper, x0, x, y)["success"]:
            break
    final = eval_source(wrapper, x0, x, y)
    return {
        **final,
        "margin_drop": float(clean["margin"] - final["margin"]),
        "p_y_drop": float(clean["p_y"] - final["p_y"]),
        "depth": depth,
        "evals": depth,
        "path": "random_pixel",
    }


def highway_candidate(
    wrapper,
    x0: torch.Tensor,
    x_cur: torch.Tensor,
    y: torch.Tensor,
    layer: str,
    direction: torch.Tensor,
    eps: float,
    step_size: float,
) -> tuple[torch.Tensor, dict]:
    before = eval_source(wrapper, x0, x_cur, y)
    h0 = feature_tensor(wrapper, x_cur, layer).detach()
    probe = x_cur.detach().requires_grad_(True)
    h = feature_tensor(wrapper, probe, layer)
    scalar = (h * direction.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, probe)[0]
    x_next = project_linf(x_cur + step_size * grad.sign(), x0, eps).detach()
    after = eval_source(wrapper, x0, x_next, y)
    h1 = feature_tensor(wrapper, x_next, layer).detach()
    dh = h1 - h0
    feature_speed = float(torch.norm(dh, dim=1).item())
    step_linf = float((x_next - x_cur).abs().max().item())
    budget_before = float(before["linf"])
    budget_after = float(after["linf"])
    budget_increment = max(0.0, budget_after - budget_before)
    margin_drop = float(before["margin"] - after["margin"])
    p_y_drop = float(before["p_y"] - after["p_y"])
    return x_next, {
        **{f"before_{k}": v for k, v in before.items()},
        **{f"after_{k}": v for k, v in after.items()},
        "feature_speed": feature_speed,
        "step_linf": step_linf,
        "budget_before": budget_before,
        "budget_after": budget_after,
        "budget_increment": budget_increment,
        "remaining_budget": float(max(eps - budget_after, 0.0)),
        "margin_drop": margin_drop,
        "p_y_drop": p_y_drop,
    }


def score_candidate(stats: dict, method: str, rng: np.random.Generator) -> float:
    margin_drop = float(stats["margin_drop"])
    py_drop = float(stats["p_y_drop"])
    speed = max(float(stats["feature_speed"]), 0.0)
    step_cost = max(float(stats["step_linf"]), 1e-9)
    inc_cost = max(float(stats["budget_increment"]), 1e-9)
    total_cost = max(float(stats["budget_after"]), 1e-9)
    if method == "margin_drop":
        return margin_drop
    if method == "confidence_drop":
        return py_drop
    if method == "efficiency_increment":
        return max(margin_drop, 0.0) / inc_cost
    if method == "efficiency_step":
        return max(margin_drop, 0.0) / step_cost
    if method == "efficiency_total":
        return max(float(stats["before_margin"]) - float(stats["after_margin"]), 0.0) / total_cost
    if method == "mobility":
        return speed / step_cost
    if method == "mobility_margin":
        return speed * max(margin_drop, 0.0) / step_cost
    if method == "random_highway":
        return float(rng.random())
    raise ValueError(f"Unknown method {method}")


def route_runaway(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    layer: str,
    basis_t: torch.Tensor,
    routes: pd.DataFrame,
    method: str,
    eps: float,
    step_sizes: list[float],
    max_depth: int,
    rng: np.random.Generator,
) -> tuple[dict, list[dict]]:
    clean = eval_source(wrapper, x0, x0, y)
    x = x0.detach()
    selected = []
    path = []
    n_candidates = len(routes) * len(step_sizes)
    for depth in range(1, max_depth + 1):
        candidates = []
        for route in routes.itertuples(index=False):
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            for ss in step_sizes:
                x_next, stats = highway_candidate(wrapper, x0, x, y, layer, direction, eps, ss)
                stats.update(
                    {
                        "depth": depth,
                        "route": str(route.route),
                        "route_id": int(route.route_id),
                        "pc": int(route.pc),
                        "sign": int(route.sign),
                        "candidate_step_size": float(ss),
                        "x_next": x_next,
                    }
                )
                stats["score"] = score_candidate(stats, method, rng)
                candidates.append(stats)
        best = max(candidates, key=lambda r: r["score"])
        x = best.pop("x_next").detach()
        path.append(f"{best['route']}@{best['candidate_step_size'] * 255:.2f}")
        selected.append(best)
        if int(best["after_success"]):
            break
    final = eval_source(wrapper, x0, x, y)
    return {
        **final,
        "margin_drop": float(clean["margin"] - final["margin"]),
        "p_y_drop": float(clean["p_y"] - final["p_y"]),
        "depth": len(selected),
        "evals": int(len(selected) * n_candidates),
        "path": "|".join(path),
    }, selected


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("method", dropna=False)
        .agg(
            n=("success", "size"),
            asr=("success", "mean"),
            mean_margin=("margin", "mean"),
            median_margin=("margin", "median"),
            mean_p_y=("p_y", "mean"),
            median_p_y=("p_y", "median"),
            mean_margin_drop=("margin_drop", "mean"),
            median_margin_drop=("margin_drop", "median"),
            mean_p_y_drop=("p_y_drop", "mean"),
            mean_linf=("linf", "mean"),
            mean_evals=("evals", "mean"),
            mean_depth=("depth", "mean"),
        )
        .reset_index()
        .sort_values(["asr", "mean_margin_drop"], ascending=[False, False])
    )


def paired_deltas(df: pd.DataFrame, baselines: list[str]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for value_col in ["success", "margin_drop", "p_y_drop", "linf"]:
        wide = df.pivot_table(index="image_ord", columns="method", values=value_col, aggfunc="first")
        for baseline in baselines:
            if baseline not in wide:
                continue
            for method in [c for c in wide.columns if c != baseline]:
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
                        "metric": value_col,
                        "n": int(len(d)),
                        "mean_delta": float(d.mean()),
                        "ci_low": float(np.quantile(boots, 0.025)),
                        "ci_high": float(np.quantile(boots, 0.975)),
                        "fraction_better": float((d > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    order = summary.sort_values(["asr", "mean_margin_drop"], ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), dpi=200)
    axes[0].bar(order["method"], order["asr"], color="#4c78a8")
    axes[0].set_ylabel("ASR")
    axes[0].set_title("Run-away routing")
    axes[1].bar(order["method"], order["mean_margin_drop"], color="#59a14f")
    axes[1].set_ylabel("Mean margin drop")
    axes[2].bar(order["method"], order["mean_linf"] * 255.0, color="#f28e2b")
    axes[2].set_ylabel("Mean L_inf used / 255")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "budgeted_runaway_routing_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "budgeted_runaway_routing_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/budgeted_runaway_routing_bbb_resnet50_c200")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--candidate-k", type=int, default=10)
    p.add_argument("--split", default="all")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-sizes", default="0.25,0.5,1.0")
    p.add_argument("--pgd-steps", default="2,3,5")
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--methods", default="margin_drop,confidence_drop,efficiency_step,efficiency_increment,mobility,mobility_margin,random_highway")
    p.add_argument("--delta-baselines", default="pgd2,pgd3,pgd5,random_pixel5,runaway_random_highway_d5")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    store = ArtifactStore(Path(args.input_dir))
    _mean, basis, n_highway_train = fit_highway_basis(
        store, args.model, args.highway_source, args.layer, args.highway_k, args.seed
    )
    routes = make_routes(basis, args.candidate_k)
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    images = load_images(store, args.model, args.split, args.images)
    wrapper = load_model(args.model, device).eval()
    eps = args.eps / 255.0
    step_sizes = [x / 255.0 for x in parse_float_csv(args.step_sizes)]
    rng = np.random.default_rng(args.seed)

    rows = []
    step_rows = []
    methods = parse_csv(args.methods)
    for i, row in enumerate(images.itertuples(index=False), start=1):
        x_cpu, _ = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)

        for steps in parse_int_csv(args.pgd_steps):
            final = pgd_attack(wrapper, x0, y, eps, max(step_sizes), steps)
            rows.append(
                {
                    "image_ord": int(row.image_ord),
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "method": f"pgd{steps}",
                    **final,
                }
            )

        gen = torch.Generator(device=device).manual_seed(args.seed + int(row.image_ord) * 7919)
        final = random_pixel(wrapper, x0, y, eps, max(step_sizes), args.max_depth, gen)
        rows.append(
            {
                "image_ord": int(row.image_ord),
                "dataset_idx": int(row.dataset_idx),
                "label": int(row.label),
                "method": f"random_pixel{args.max_depth}",
                **final,
            }
        )

        for method in methods:
            final, selected = route_runaway(
                wrapper, x0, y, args.layer, basis_t, routes, method, eps, step_sizes, args.max_depth, rng
            )
            method_name = f"runaway_{method}_d{args.max_depth}"
            rows.append(
                {
                    "image_ord": int(row.image_ord),
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "method": method_name,
                    **final,
                }
            )
            for s in selected:
                s.update(
                    {
                        "image_ord": int(row.image_ord),
                        "dataset_idx": int(row.dataset_idx),
                        "label": int(row.label),
                        "method": method_name,
                    }
                )
                step_rows.append(s)
        if i % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(out_dir / "partial_budgeted_runaway_routing_per_image.csv", index=False)
            print(f"[{i}/{len(images)}] rows={len(rows)}", flush=True)

    per = pd.DataFrame(rows)
    steps_df = pd.DataFrame(step_rows)
    summary = summarize(per)
    deltas = paired_deltas(per, parse_csv(args.delta_baselines))
    per.to_csv(out_dir / "budgeted_runaway_routing_per_image.csv", index=False)
    steps_df.to_csv(out_dir / "budgeted_runaway_routing_selected_steps.csv", index=False)
    routes.to_csv(out_dir / "budgeted_runaway_candidate_routes.csv", index=False)
    summary.to_csv(out_dir / "budgeted_runaway_routing_summary.csv", index=False)
    deltas.to_csv(out_dir / "budgeted_runaway_routing_paired_deltas.csv", index=False)
    plot_summary(summary, out_dir)
    metadata = vars(args).copy()
    metadata.update(
        {
            "device": str(device),
            "highway_train_vectors": int(n_highway_train),
            "objective": "minimize true-class confidence/margin under L_inf budget",
            "difference_from_class_routing": "no target city; route away from source class as budget-efficiently as possible",
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Budgeted Run-Away Routing",
        "",
        "The planner has no target class region. It chooses signed highway moves and step sizes to reduce true-class confidence/margin under an L_inf budget.",
        "",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"- `{r.method}`: ASR={r.asr:.3f}, margin_drop={r.mean_margin_drop:.3f}, "
            f"p_y_drop={r.mean_p_y_drop:.3f}, linf={r.mean_linf * 255.0:.3f}/255, evals={r.mean_evals:.1f}"
        )
    (out_dir / "budgeted_runaway_routing_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
