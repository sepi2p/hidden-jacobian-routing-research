#!/usr/bin/env python3
"""Identify and test directed cross-class representation highways.

For a signed highway route r and class pair y -> t, the classifier-head
prediction for the pair margin drop is

    gain(y,t,r) = delta z_t(r) - delta z_y(r).

This script identifies routes for each directed pair, measures target
specificity, and optionally tests whether pulling back those routes on
clean source-class images moves predictions toward the target class.
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


def target_margin(logits: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return logits[:, y.item()] - logits[:, t.item()]


def eval_target_state(wrapper, x0: torch.Tensor, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> dict:
    with torch.no_grad():
        logits = wrapper(x)
        pred = int(logits.argmax(1).item())
        tm = float(target_margin(logits, y, t).item())
        probs = torch.softmax(logits, dim=1)
        return {
            "pred": pred,
            "target_success": int(pred == int(t.item())),
            "class_changed": int(pred != int(y.item())),
            "target_margin": tm,
            "target_prob": float(probs[0, int(t.item())].item()),
            "true_prob": float(probs[0, int(y.item())].item()),
            "linf": float((x - x0).abs().max().item()),
        }


def route_pullback_step(
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
    speed = float(torch.norm(dh, dim=1).item())
    align = float((dh * direction.view_as(dh)).sum().item() / max(speed, 1e-12))
    return x_next, speed, align


def targeted_pgd(wrapper, x0: torch.Tensor, y: torch.Tensor, t: torch.Tensor, eps: float, step_size: float, steps: int):
    x = x0.detach()
    clean = eval_target_state(wrapper, x0, x0, y, t)
    path = []
    for _ in range(steps):
        probe = x.detach().requires_grad_(True)
        logits = wrapper(probe)
        loss = torch.nn.functional.cross_entropy(logits, t)
        grad = torch.autograd.grad(loss, probe)[0]
        x = project_linf(x - step_size * grad.sign(), x0, eps).detach()
        ev = eval_target_state(wrapper, x0, x, y, t)
        path.append(ev)
        if ev["target_success"]:
            break
    final = eval_target_state(wrapper, x0, x, y, t)
    return {
        **final,
        "target_margin_drop": float(clean["target_margin"] - final["target_margin"]),
        "evals": int(len(path)),
        "depth": int(len(path)),
        "path": "targeted_ce",
    }


def random_pixel(wrapper, x0: torch.Tensor, y: torch.Tensor, t: torch.Tensor, eps: float, step_size: float, steps: int, gen):
    x = x0.detach()
    clean = eval_target_state(wrapper, x0, x0, y, t)
    path = []
    for _ in range(steps):
        direction = torch.randn(x.shape, generator=gen, device=x.device).sign()
        x = project_linf(x + step_size * direction, x0, eps).detach()
        ev = eval_target_state(wrapper, x0, x, y, t)
        path.append(ev)
        if ev["target_success"]:
            break
    final = eval_target_state(wrapper, x0, x, y, t)
    return {
        **final,
        "target_margin_drop": float(clean["target_margin"] - final["target_margin"]),
        "evals": int(len(path)),
        "depth": int(len(path)),
        "path": "random_pixel",
    }


def build_cross_class_tables(pair_gain_path: Path, k: int, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair = pd.read_csv(pair_gain_path)
    pair = pair[(pair["source_class"].astype(int) >= 0) & (pair["source_class"] != pair["target_class"])].copy()
    pair = pair[pair["pc"].astype(int) <= k].copy()
    pair["pair_gain"] = pair["class_pair_margin_drop"].astype(float)
    pair["positive_pair_gain"] = np.maximum(pair["pair_gain"], 0.0)

    # Target specificity: for the same source class and route, how much more
    # does this route favor target t than the next-best alternative target?
    group_cols = ["route", "pc", "sign", "source_class"]
    max_other = []
    target_rank = []
    target_share = []
    for _, sub in pair.groupby(group_cols, sort=False):
        gains = sub["pair_gain"].to_numpy(float)
        pos = np.maximum(gains, 0.0)
        order = np.argsort(-gains)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        total_pos = float(pos.sum())
        for local_i, idx in enumerate(sub.index):
            others = np.delete(gains, local_i)
            max_other.append((idx, float(others.max()) if len(others) else float("nan")))
            target_rank.append((idx, int(ranks[local_i])))
            target_share.append((idx, float(pos[local_i] / total_pos) if total_pos > 1e-12 else 0.0))
    pair["max_other_target_gain"] = pd.Series(dict(max_other))
    pair["target_rank_for_route"] = pd.Series(dict(target_rank))
    pair["target_positive_share_for_route"] = pd.Series(dict(target_share))
    pair["specificity_gap"] = pair["pair_gain"] - pair["max_other_target_gain"]
    pair["specific_positive_score"] = pair["positive_pair_gain"] * np.maximum(pair["specificity_gap"], 0.0)
    pair["pair_rank_by_gain"] = (
        pair.groupby(["source_class", "target_class"])["pair_gain"].rank(method="first", ascending=False).astype(int)
    )
    pair["pair_rank_by_specificity"] = (
        pair.groupby(["source_class", "target_class"])["specific_positive_score"].rank(method="first", ascending=False).astype(int)
    )

    top_gain = (
        pair.sort_values(["source_class", "target_class", "pair_gain"], ascending=[True, True, False])
        .drop_duplicates(["source_class", "target_class"])
        .copy()
    )
    top_spec = (
        pair.sort_values(["source_class", "target_class", "specific_positive_score"], ascending=[True, True, False])
        .drop_duplicates(["source_class", "target_class"])
        .copy()
    )
    top = top_gain.merge(
        top_spec[
            [
                "source_class",
                "target_class",
                "route",
                "pc",
                "sign",
                "pair_gain",
                "specificity_gap",
                "specific_positive_score",
                "target_rank_for_route",
            ]
        ],
        on=["source_class", "target_class"],
        suffixes=("_top_gain", "_top_specific"),
    )
    top.to_csv(out_dir / "cross_class_highway_top_routes.csv", index=False)
    pair.to_csv(out_dir / "cross_class_highway_all_route_scores.csv", index=False)

    matrix = top_gain.pivot(index="source_class", columns="target_class", values="pair_gain")
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=220)
    im = ax.imshow(matrix.to_numpy(float), cmap="viridis")
    ax.set_xticks(range(matrix.shape[1]), labels=[int(c) for c in matrix.columns])
    ax.set_yticks(range(matrix.shape[0]), labels=[int(i) for i in matrix.index])
    ax.set_xlabel("Target class")
    ax.set_ylabel("Source class")
    ax.set_title("Best directed highway gain")
    fig.colorbar(im, ax=ax, label="max route gain")
    fig.tight_layout()
    fig.savefig(out_dir / "cross_class_highway_gain_matrix.png")
    fig.savefig(out_dir / "cross_class_highway_gain_matrix.pdf")
    plt.close(fig)
    return pair, top


def select_clean_correct_images(dataset, wrapper, device, max_per_class: int, max_total: int) -> pd.DataFrame:
    rows = []
    counts = {c: 0 for c in range(10)}
    for idx in range(len(dataset)):
        x_cpu, label = dataset[idx]
        if counts[int(label)] >= max_per_class:
            continue
        x = x_cpu.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(wrapper(x).argmax(1).item())
        if pred == int(label):
            rows.append({"dataset_idx": int(idx), "label": int(label), "class_ord": counts[int(label)]})
            counts[int(label)] += 1
        if sum(counts.values()) >= max_total:
            break
        if all(v >= max_per_class for v in counts.values()):
            break
    return pd.DataFrame(rows)


def route_to_target(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    layer: str,
    basis_t: torch.Tensor,
    pair_routes: pd.DataFrame,
    method: str,
    eps: float,
    step_size: float,
    max_depth: int,
    candidate_k: int,
    rng: np.random.Generator,
) -> tuple[dict, list[dict]]:
    clean = eval_target_state(wrapper, x0, x0, y, t)
    x = x0.detach()
    selected = []
    path = []
    pool = pair_routes.copy()
    if method == "top_gain":
        pool = pool.sort_values("pair_gain", ascending=False).head(1)
    elif method == "top_specific":
        pool = pool.sort_values("specific_positive_score", ascending=False).head(1)
    else:
        pool = pool.sort_values("pair_gain", ascending=False).head(candidate_k)

    for depth in range(1, max_depth + 1):
        current = eval_target_state(wrapper, x0, x, y, t)
        candidates = []
        for route in pool.itertuples(index=False):
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            x_next, speed, align = route_pullback_step(wrapper, x0, x, layer, direction, eps, step_size)
            ev = eval_target_state(wrapper, x0, x_next, y, t)
            row = {
                "depth": depth,
                "route": str(route.route),
                "pc": int(route.pc),
                "sign": int(route.sign),
                "pair_gain": float(route.pair_gain),
                "specificity_gap": float(route.specificity_gap),
                "specific_positive_score": float(route.specific_positive_score),
                "target_rank_for_route": int(route.target_rank_for_route),
                "feature_speed": speed,
                "realized_alignment": align,
                "observed_target_margin_drop": float(current["target_margin"] - ev["target_margin"]),
                "observed_target_success": int(ev["target_success"]),
                "observed_class_changed": int(ev["class_changed"]),
                "x_next": x_next,
            }
            if method == "traffic_target":
                row["score"] = float(max(row["pair_gain"], 0.0) * max(speed, 1e-9))
            elif method == "traffic_specific":
                row["score"] = float(max(row["specific_positive_score"], 0.0) * max(speed, 1e-9))
            elif method == "random_pair":
                row["score"] = float(rng.random())
            elif method == "target_oracle":
                row["score"] = row["observed_target_margin_drop"]
            else:
                row["score"] = float(max(row["pair_gain"], 0.0))
            candidates.append(row)
        if not candidates:
            break
        best = max(candidates, key=lambda r: r["score"])
        x = best.pop("x_next").detach()
        path.append(best["route"])
        selected.append(best)
        if int(best["observed_target_success"]):
            break
    final = eval_target_state(wrapper, x0, x, y, t)
    return {
        **final,
        "target_margin_drop": float(clean["target_margin"] - final["target_margin"]),
        "evals": int(len(selected) * len(pool)),
        "depth": int(len(selected)),
        "path": "|".join(path),
        "clean_target_margin": float(clean["target_margin"]),
    }, selected


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("method", dropna=False)
        .agg(
            n=("target_success", "size"),
            target_hit_rate=("target_success", "mean"),
            class_changed_rate=("class_changed", "mean"),
            mean_target_margin=("target_margin", "mean"),
            median_target_margin=("target_margin", "median"),
            mean_target_margin_drop=("target_margin_drop", "mean"),
            median_target_margin_drop=("target_margin_drop", "median"),
            mean_target_prob=("target_prob", "mean"),
            mean_evals=("evals", "mean"),
            mean_depth=("depth", "mean"),
            max_linf=("linf", "max"),
        )
        .reset_index()
        .sort_values(["target_hit_rate", "mean_target_margin_drop"], ascending=[False, False])
    )


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), dpi=200)
    order = summary.sort_values("target_hit_rate", ascending=False)
    axes[0].bar(order["method"], order["target_hit_rate"], color="#4c78a8")
    axes[0].set_ylabel("Target hit rate")
    axes[0].set_title("Cross-class highway routing")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(order["method"], order["mean_target_margin_drop"], color="#59a14f")
    axes[1].set_ylabel("Mean y-to-target margin drop")
    axes[1].set_title("Target margin progress")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_class_highway_targeted_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "cross_class_highway_targeted_summary.pdf", bbox_inches="tight")
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
    all_routes, top_routes = build_cross_class_tables(Path(args.class_pair_gains), args.highway_k, out_dir)
    wrapper = load_model(args.model, device).eval()
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    images = select_clean_correct_images(dataset, wrapper, device, args.images_per_class, args.images)
    images.to_csv(out_dir / "cross_class_eval_images.csv", index=False)

    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    rng = np.random.default_rng(args.seed)
    rows = []
    selected_rows = []
    methods = parse_csv(args.methods)
    target_mode = args.targets

    for image_i, image in enumerate(images.itertuples(index=False), start=1):
        x_cpu, _label = dataset[int(image.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(image.label)], device=device)
        if target_mode == "all":
            targets = [c for c in range(10) if c != int(image.label)]
        elif target_mode == "next":
            targets = [(int(image.label) + 1) % 10]
        else:
            targets = [int(x) for x in parse_csv(target_mode) if int(x) != int(image.label)]
        for target_class in targets:
            t = torch.tensor([target_class], device=device)
            pair_routes = all_routes[
                (all_routes.source_class.astype(int) == int(image.label))
                & (all_routes.target_class.astype(int) == int(target_class))
            ].copy()
            for steps in [int(x) for x in parse_csv(args.targeted_pgd_steps)]:
                final = targeted_pgd(wrapper, x0, y, t, eps, step_size, steps)
                rows.append(
                    {
                        "image_ord": image_i - 1,
                        "dataset_idx": int(image.dataset_idx),
                        "source_class": int(image.label),
                        "target_class": target_class,
                        "method": f"targeted_pgd{steps}",
                        **final,
                    }
                )
            gen = torch.Generator(device=device).manual_seed(args.seed + int(image.dataset_idx) * 1009 + target_class * 9173)
            final = random_pixel(wrapper, x0, y, t, eps, step_size, args.max_depth, gen)
            rows.append(
                {
                    "image_ord": image_i - 1,
                    "dataset_idx": int(image.dataset_idx),
                    "source_class": int(image.label),
                    "target_class": target_class,
                    "method": f"random_pixel{args.max_depth}",
                    **final,
                }
            )
            for method in methods:
                final, selected = route_to_target(
                    wrapper,
                    x0,
                    y,
                    t,
                    args.layer,
                    basis_t,
                    pair_routes,
                    method,
                    eps,
                    step_size,
                    args.max_depth,
                    args.candidate_k,
                    rng,
                )
                method_name = f"cross_{method}_k{args.candidate_k}_d{args.max_depth}"
                rows.append(
                    {
                        "image_ord": image_i - 1,
                        "dataset_idx": int(image.dataset_idx),
                        "source_class": int(image.label),
                        "target_class": target_class,
                        "method": method_name,
                        **final,
                    }
                )
                for s in selected:
                    s.update(
                        {
                            "image_ord": image_i - 1,
                            "dataset_idx": int(image.dataset_idx),
                            "source_class": int(image.label),
                            "target_class": target_class,
                            "method": method_name,
                        }
                    )
                    selected_rows.append(s)
        if image_i % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(out_dir / "partial_cross_class_highway_eval.csv", index=False)
            print(f"[{image_i}/{len(images)}] rows={len(rows)}", flush=True)

    eval_df = pd.DataFrame(rows)
    selected_df = pd.DataFrame(selected_rows)
    summary = summarize(eval_df)
    by_pair = (
        eval_df.groupby(["method", "source_class", "target_class"], as_index=False)
        .agg(
            n=("target_success", "size"),
            target_hit_rate=("target_success", "mean"),
            mean_target_margin_drop=("target_margin_drop", "mean"),
            class_changed_rate=("class_changed", "mean"),
        )
    )
    eval_df.to_csv(out_dir / "cross_class_highway_targeted_eval.csv", index=False)
    selected_df.to_csv(out_dir / "cross_class_highway_selected_steps.csv", index=False)
    summary.to_csv(out_dir / "cross_class_highway_targeted_summary.csv", index=False)
    by_pair.to_csv(out_dir / "cross_class_highway_targeted_by_pair.csv", index=False)
    plot_summary(summary, out_dir)

    meta = vars(args).copy()
    meta.update(
        {
            "device": str(device),
            "n_images": int(len(images)),
            "highway_train_vectors": int(n_highway_train),
            "cross_class_definition": "directed source-target class-pair classifier-head margin gain per signed highway",
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Cross-Class Highway Identification",
        "",
        "A cross-class highway is a signed representation route with high directed class-pair gain `gain(y,t,r)=delta z_t(r)-delta z_y(r)` for source class `y` and target class `t`.",
        "",
        "## Targeted Evaluation Summary",
        "",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"- `{r.method}`: target_hit={r.target_hit_rate:.3f}, class_changed={r.class_changed_rate:.3f}, "
            f"target_margin_drop={r.mean_target_margin_drop:.3f}, evals={r.mean_evals:.1f}"
        )
    (out_dir / "cross_class_highway_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--class-pair-gains", default="analysis_outputs/pure_af_geometry/jacobian_null_response/absolute_highway_adv_gain_bbb_resnet50_layer4/absolute_highway_class_pair_gains.csv")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/cross_class_highways_bbb_resnet50_c100")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--images-per-class", type=int, default=10)
    p.add_argument("--targets", default="all")
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--candidate-k", type=int, default=5)
    p.add_argument("--methods", default="top_gain,top_specific,traffic_target,traffic_specific,random_pair,target_oracle")
    p.add_argument("--targeted-pgd-steps", default="3")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
