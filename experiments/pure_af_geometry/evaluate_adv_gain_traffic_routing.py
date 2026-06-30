#!/usr/bin/env python3
"""Route through Jacobian highways using adversarial-gain traffic costs.

This diagnostic separates two notions of "traffic":

1. local road quality: how much the hidden representation moves when we pull
   back a signed highway direction through the local Jacobian;
2. semantic/adversarial usefulness: how much that signed feature direction is
   predicted to reduce a class-pair margin by the classifier head.

The route selector is not allowed to use the post-move adversarial margin,
except for the optional oracle ceiling.  Margins are evaluated only after the
selected move is applied.
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


def build_route_table(pair_gain_path: Path, absolute_gain_path: Path, k: int) -> pd.DataFrame:
    pair = pd.read_csv(pair_gain_path)
    absolute = pd.read_csv(absolute_gain_path)
    pair = pair[pair["source_class"].astype(int) >= 0].copy()
    pair = pair[pair["pc"].astype(int) <= k].copy()
    pair["positive_pair_gain"] = np.maximum(pair["class_pair_margin_drop"].astype(float), 0.0)
    pair["absolute_pair_gain"] = np.abs(pair["class_pair_margin_drop"].astype(float))

    untargeted = (
        pair.groupby(["route", "pc", "sign", "source_class"], as_index=False)
        .agg(
            untargeted_gain=("positive_pair_gain", "max"),
            mean_positive_pair_gain=("positive_pair_gain", "mean"),
            mean_absolute_pair_gain=("absolute_pair_gain", "mean"),
            max_absolute_pair_gain=("absolute_pair_gain", "max"),
            best_target_class=("target_class", lambda x: int(x.iloc[0])),
        )
    )
    # Restore best target as the target with maximum positive/signed gain.
    best_rows = (
        pair.sort_values(["route", "source_class", "class_pair_margin_drop"], ascending=[True, True, False])
        .drop_duplicates(["route", "source_class"])
        [["route", "source_class", "target_class", "class_pair_margin_drop"]]
        .rename(columns={"target_class": "best_target_class", "class_pair_margin_drop": "best_signed_pair_gain"})
    )
    untargeted = untargeted.drop(columns=["best_target_class"]).merge(best_rows, on=["route", "source_class"], how="left")

    abs_cols = [
        "route",
        "absolute_adv_gain_rank",
        "mean_best_margin_drop",
        "median_best_margin_drop",
        "logit_effect_l2",
        "logit_effect_linf",
    ]
    abs_keep = absolute[[c for c in abs_cols if c in absolute.columns]].copy()
    routes = untargeted.merge(abs_keep, on="route", how="left")
    routes["pc"] = routes["pc"].astype(int)
    routes["sign"] = routes["sign"].astype(int)
    routes["source_class"] = routes["source_class"].astype(int)
    return routes


def candidate_step(
    wrapper,
    x0: torch.Tensor,
    x_cur: torch.Tensor,
    layer: str,
    direction: torch.Tensor,
    eps: float,
    step_size: float,
) -> tuple[torch.Tensor, float, float]:
    h0 = feature_tensor(wrapper, x_cur, layer).detach()
    probe = x_cur.detach().requires_grad_(True)
    h = feature_tensor(wrapper, probe, layer)
    scalar = (h * direction.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, probe)[0]
    x_next = project_linf(x_cur + step_size * grad.sign(), x0, eps).detach()
    h1 = feature_tensor(wrapper, x_next, layer).detach()
    dh = h1 - h0
    feature_speed = float(torch.norm(dh, dim=1).item())
    realized_alignment = float((dh * direction.view_as(dh)).sum().item() / max(feature_speed, 1e-12))
    return x_next, feature_speed, realized_alignment


def route_score(c: dict, method: str, rng: np.random.Generator, gain_power: float) -> float:
    mobility = max(float(c["feature_speed"]), 0.0)
    gain = max(float(c["untargeted_gain"]), 0.0)
    abs_gain = max(float(c["mean_absolute_pair_gain"]), 0.0)
    if method == "combined":
        return float((mobility + 1e-9) * ((gain + 1e-9) ** gain_power))
    if method == "combined_abs":
        return float((mobility + 1e-9) * ((abs_gain + 1e-9) ** gain_power))
    if method == "mobility":
        return mobility
    if method == "gain":
        return gain
    if method == "absolute_gain":
        return abs_gain
    if method == "random_highway":
        return float(rng.random())
    if method == "margin_oracle":
        return float(c["observed_margin_drop"])
    raise ValueError(f"Unknown traffic method: {method}")


def traffic_route(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    layer: str,
    basis_t: torch.Tensor,
    routes_for_y: pd.DataFrame,
    method: str,
    eps: float,
    step_size: float,
    max_depth: int,
    top_routes: int,
    rng: np.random.Generator,
    gain_power: float,
) -> tuple[dict, list[dict]]:
    clean = eval_state(wrapper, x0, x0, y)
    x = x0.detach()
    rows: list[dict] = []
    path: list[str] = []

    # Limit the action set by absolute adversarial-gain rank if available; this
    # keeps the search comparable across methods while still allowing random and
    # mobility-only selectors to choose among the same signed highways.
    action_pool = routes_for_y.copy()
    if "absolute_adv_gain_rank" in action_pool.columns and top_routes > 0:
        action_pool = action_pool.sort_values("absolute_adv_gain_rank").head(top_routes)
    elif top_routes > 0:
        action_pool = action_pool.head(top_routes)

    for depth in range(1, max_depth + 1):
        current = eval_state(wrapper, x0, x, y)
        candidates = []
        for route in action_pool.itertuples(index=False):
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            x_next, speed, align = candidate_step(wrapper, x0, x, layer, direction, eps, step_size)
            ev = eval_state(wrapper, x0, x_next, y)
            observed_drop = float(current["margin"] - ev["margin"])
            row = {
                "depth": depth,
                "route": str(route.route),
                "pc": int(route.pc),
                "sign": int(route.sign),
                "traffic_method": method,
                "feature_speed": speed,
                "realized_alignment": align,
                "untargeted_gain": float(route.untargeted_gain),
                "mean_absolute_pair_gain": float(route.mean_absolute_pair_gain),
                "best_target_class": int(route.best_target_class),
                "best_signed_pair_gain": float(route.best_signed_pair_gain),
                "absolute_adv_gain_rank": float(getattr(route, "absolute_adv_gain_rank", np.nan)),
                "observed_margin": float(ev["margin"]),
                "observed_margin_drop": observed_drop,
                "observed_success": int(ev["success"]),
                "x_next": x_next,
            }
            row["traffic_score"] = route_score(row, method, rng, gain_power)
            row["traffic_cost"] = float(1.0 / max(row["traffic_score"], 1e-12))
            candidates.append(row)
        if not candidates:
            break
        best = max(candidates, key=lambda r: r["traffic_score"])
        x = best.pop("x_next").detach()
        path.append(best["route"])
        rows.append(best)
        if int(best["observed_success"]):
            break

    final = eval_state(wrapper, x0, x, y)
    return {
        **final,
        "margin_drop": float(clean["margin"] - final["margin"]),
        "evals": int(len(rows) * len(action_pool)),
        "depth": int(len(rows)),
        "path": "|".join(path),
        "clean_margin": float(clean["margin"]),
    }, rows


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
    order = summary.sort_values("asr", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.0), dpi=200)
    axes[0].bar(order["method"], order["asr"], color="#4c78a8")
    axes[0].set_ylabel("ASR")
    axes[0].set_title("Traffic-routed highway moves")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(order["method"], order["mean_margin_drop"], color="#59a14f")
    axes[1].set_ylabel("Mean margin drop")
    axes[1].set_title("Margin progress")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "adv_gain_traffic_routing_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "adv_gain_traffic_routing_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    store = ArtifactStore(Path(args.input_dir))
    _mean, basis, n_highway_train = fit_highway_basis(
        store, args.model, args.highway_source, args.layer, args.highway_k, args.seed
    )
    route_table = build_route_table(Path(args.class_pair_gains), Path(args.absolute_gains), args.highway_k)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    images = load_images(store, args.model, args.split, args.images, args.image_ord_csv, args.image_ords)
    wrapper = load_model(args.model, device).eval()
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=device)
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    selected_rows: list[dict] = []
    traffic_methods = parse_csv(args.traffic_methods)

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

        routes_for_y = route_table[route_table["source_class"] == int(row.label)].copy()
        for method in traffic_methods:
            final, selected = traffic_route(
                wrapper,
                x0,
                y,
                args.layer,
                basis_t,
                routes_for_y,
                method,
                eps,
                step_size,
                args.max_depth,
                args.top_routes,
                rng,
                args.gain_power,
            )
            method_name = f"traffic_{method}_top{args.top_routes}_d{args.max_depth}"
            rows.append(
                {
                    "image_ord": int(row.image_ord),
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "method": method_name,
                    "success": int(final["success"]),
                    "margin": float(final["margin"]),
                    "margin_drop": float(final["margin_drop"]),
                    "evals": int(final["evals"]),
                    "depth": int(final["depth"]),
                    "linf": float(final["linf"]),
                    "path": str(final["path"]),
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
                selected_rows.append(s)

        if i % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(out_dir / "partial_adv_gain_traffic_routing_per_image.csv", index=False)
            print(f"[{i}/{len(images)}] rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    selected_df = pd.DataFrame(selected_rows)
    summary = summarize(df)
    deltas = paired_deltas(df, parse_csv(args.delta_baselines))
    route_table.to_csv(out_dir / "adv_gain_traffic_route_table.csv", index=False)
    df.to_csv(out_dir / "adv_gain_traffic_routing_per_image.csv", index=False)
    selected_df.to_csv(out_dir / "adv_gain_traffic_selected_steps.csv", index=False)
    summary.to_csv(out_dir / "adv_gain_traffic_routing_summary.csv", index=False)
    deltas.to_csv(out_dir / "adv_gain_traffic_routing_paired_deltas.csv", index=False)
    plot_summary(summary, out_dir)

    meta = vars(args).copy()
    meta.update(
        {
            "device": str(device),
            "n_images": int(len(images)),
            "highway_train_vectors": int(n_highway_train),
            "selection_uses": [
                "local feature mobility from highway pullback",
                "classifier-head class-pair margin gain",
            ],
            "selection_excludes": [
                "post-move margin except margin_oracle",
                "PGD/Square/GA trajectory usage",
                "success labels",
            ],
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Adversarial-Gain Traffic Routing",
        "",
        "Traffic selectors choose signed Jacobian-highway moves from local mobility and/or classifier-head margin gain. "
        "Only the `margin_oracle` selector uses post-move margin as a ceiling.",
        "",
        "## Summary",
        "",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"- `{r.method}`: ASR={r.asr:.3f}, mean_margin_drop={r.mean_margin_drop:.3f}, "
            f"mean_evals={r.mean_evals:.1f}, mean_depth={r.mean_depth:.2f}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If `traffic_combined` exceeds `traffic_mobility` and `traffic_gain`, the product of road quality and semantic gain is more informative than either component alone. "
            "If `traffic_gain` alone wins, image-independent semantic direction dominates. "
            "If `traffic_mobility` wins, local Jacobian road quality dominates.",
            "",
        ]
    )
    (out_dir / "adv_gain_traffic_routing_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--absolute-gains", default="analysis_outputs/pure_af_geometry/jacobian_null_response/absolute_highway_adv_gain_bbb_resnet50_layer4/absolute_highway_adv_gain.csv")
    p.add_argument("--class-pair-gains", default="analysis_outputs/pure_af_geometry/jacobian_null_response/absolute_highway_adv_gain_bbb_resnet50_layer4/absolute_highway_class_pair_gains.csv")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/adv_gain_traffic_routing_bbb_resnet50_c200")
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
    p.add_argument("--top-routes", type=int, default=20)
    p.add_argument("--traffic-methods", default="combined,mobility,gain,absolute_gain,combined_abs,random_highway,margin_oracle")
    p.add_argument("--gain-power", type=float, default=1.0)
    p.add_argument("--delta-baselines", default="pgd2,pgd3,pgd5,random_pixel3,traffic_random_highway_top20_d3,traffic_mobility_top20_d3,traffic_gain_top20_d3")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
