#!/usr/bin/env python3
"""Rescue weak-PGD failures with white-box highway rerouting.

This experiment tests the "traffic routing" interpretation in the setting where
it should matter most: images where a short PGD route fails.  We first run a
balanced weak PGD attack, isolate the failed final states, then apply several
white-box rescue policies inside the same L_inf ball:

* CE continuation;
* random pixel continuation;
* fixed global highway route;
* exact state-conditioned highway route selection among top-k/random/all routes;
* pixel-gradient selected highway route.

Route construction uses white-box feature pullbacks.  Exact route selection
evaluates candidate margins directly, so it is a diagnostic of whether a better
route exists, not a target-only black-box attack claim.
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


def ce_step(wrapper, x0: torch.Tensor, x_cur: torch.Tensor, y: torch.Tensor, eps: float, step_size: float) -> torch.Tensor:
    probe = x_cur.detach().requires_grad_(True)
    logits = wrapper(probe)
    loss = torch.nn.functional.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, probe)[0]
    return project_linf(x_cur + step_size * grad.sign(), x0, eps).detach()


def random_step(x0: torch.Tensor, x_cur: torch.Tensor, eps: float, step_size: float, gen: torch.Generator) -> torch.Tensor:
    direction = torch.randn(x_cur.shape, generator=gen, device=x_cur.device).sign()
    return project_linf(x_cur + step_size * direction, x0, eps).detach()


def route_step(
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


def pixelgrad_score(
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


def run_pgd(wrapper, x0: torch.Tensor, y: torch.Tensor, eps: float, steps: int, step_size: float):
    states = [x0.detach().clone()]
    x = x0.detach().clone()
    for _ in range(steps):
        x = ce_step(wrapper, x0, x, y, eps, step_size)
        states.append(x.detach().clone())
    return states


def eval_state(wrapper, x0: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> dict:
    ev = eval_clean(wrapper, x, y)
    return {
        "pred": int(ev["pred"]),
        "margin": float(ev["margin"]),
        "success": int(ev["pred"] != int(y.item())),
        "linf": float((x - x0).abs().max().item()),
    }


def route_subset(routes: pd.DataFrame, mode: str, rng: np.random.Generator) -> tuple[pd.DataFrame, str]:
    if mode == "global_rank1":
        return routes[routes.global_rank <= 1].copy(), "observed"
    if mode.startswith("top"):
        n = int(mode.replace("top", ""))
        return routes[routes.global_rank <= n].copy(), "observed"
    if mode.startswith("random"):
        n = int(mode.replace("random", ""))
        return routes.sample(n=min(n, len(routes)), random_state=int(rng.integers(0, 2**31 - 1))).copy(), "observed"
    if mode == "all":
        return routes.copy(), "observed"
    if mode == "pixelgrad":
        return routes.copy(), "pixelgrad"
    raise ValueError(f"Unknown highway rescue mode {mode}")


def highway_rescue_step(
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
    candidates, selector = route_subset(routes, mode, rng)
    candidate_rows: list[dict] = []
    if selector == "pixelgrad":
        grad_x = margin_pixel_grad(wrapper, x_cur, y)
        scored = []
        for route in candidates.itertuples():
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            score = pixelgrad_score(wrapper, x_cur, y, layer, direction, grad_x)
            scored.append((score, route))
        _score, chosen_route = max(scored, key=lambda item: item[0])
        direction = int(chosen_route.sign) * basis_t[int(chosen_route.pc) - 1]
        x_next = route_step(wrapper, x0, x_cur, layer, direction, eps, step_size)
        ev = eval_state(wrapper, x0, x_next, y)
        return x_next, {
            "candidate_evals": 1,
            "chosen_route": str(chosen_route.route),
            "chosen_rank": int(chosen_route.global_rank),
            "chosen_margin_drop": float(current_margin - ev["margin"]),
            "accepted": 1,
            "success": int(ev["success"]),
            "margin": float(ev["margin"]),
            "pred": int(ev["pred"]),
        }, candidate_rows

    chosen = None
    chosen_x = None
    for route in candidates.itertuples():
        direction = int(route.sign) * basis_t[int(route.pc) - 1]
        x_cand = route_step(wrapper, x0, x_cur, layer, direction, eps, step_size)
        ev = eval_state(wrapper, x0, x_cand, y)
        row = {
            "route": str(route.route),
            "pc": int(route.pc),
            "sign": int(route.sign),
            "global_rank": int(route.global_rank),
            "global_score": float(route.train_weighted_margin_drop_score),
            "candidate_margin": float(ev["margin"]),
            "margin_drop": float(current_margin - ev["margin"]),
            "success": int(ev["success"]),
            "pred": int(ev["pred"]),
        }
        candidate_rows.append(row)
        if chosen is None or row["candidate_margin"] < chosen["candidate_margin"]:
            chosen = row
            chosen_x = x_cand
    accepted = bool(chosen["candidate_margin"] < current_margin)
    if accepted:
        x_next = chosen_x.detach()
        next_margin = float(chosen["candidate_margin"])
        success = int(chosen["success"])
        pred = int(chosen["pred"])
    else:
        x_next = x_cur.detach()
        next_margin = float(current_margin)
        success = 0
        pred = label
    return x_next, {
        "candidate_evals": int(len(candidates)),
        "chosen_route": str(chosen["route"]),
        "chosen_rank": int(chosen["global_rank"]),
        "chosen_margin_drop": float(chosen["margin_drop"]),
        "accepted": int(accepted),
        "success": success,
        "margin": next_margin,
        "pred": pred,
    }, candidate_rows


def rescue_from_failed_state(
    wrapper,
    x0: torch.Tensor,
    x_start: torch.Tensor,
    y: torch.Tensor,
    label: int,
    layer: str,
    basis_t: torch.Tensor,
    routes: pd.DataFrame,
    mode: str,
    eps: float,
    step_size: float,
    rescue_steps: int,
    rng: np.random.Generator,
    seed: int,
) -> tuple[torch.Tensor, dict, list[dict]]:
    x_cur = x_start.detach()
    start = eval_state(wrapper, x0, x_cur, y)
    current_margin = float(start["margin"])
    candidate_evals = 0
    accepted_steps = 0
    chosen_ranks: list[float] = []
    candidate_rows: list[dict] = []
    gen = torch.Generator(device=x0.device).manual_seed(seed)
    for step in range(rescue_steps):
        if mode == "ce_continue":
            x_next = ce_step(wrapper, x0, x_cur, y, eps, step_size)
            ev = eval_state(wrapper, x0, x_next, y)
            info = {
                "candidate_evals": 1,
                "accepted": 1,
                "success": int(ev["success"]),
                "margin": float(ev["margin"]),
                "pred": int(ev["pred"]),
                "chosen_rank": np.nan,
                "chosen_margin_drop": float(current_margin - ev["margin"]),
            }
        elif mode == "random_pixel":
            x_next = random_step(x0, x_cur, eps, step_size, gen)
            ev = eval_state(wrapper, x0, x_next, y)
            info = {
                "candidate_evals": 1,
                "accepted": 1,
                "success": int(ev["success"]),
                "margin": float(ev["margin"]),
                "pred": int(ev["pred"]),
                "chosen_rank": np.nan,
                "chosen_margin_drop": float(current_margin - ev["margin"]),
            }
        else:
            x_next, info, rows = highway_rescue_step(
                wrapper, x0, x_cur, y, label, layer, basis_t, routes, mode, eps, step_size, rng, current_margin
            )
            for row in rows:
                row.update({"rescue_mode": mode, "rescue_step": step})
            candidate_rows.extend(rows)
        x_cur = x_next.detach()
        current_margin = float(info["margin"])
        candidate_evals += int(info["candidate_evals"])
        accepted_steps += int(info["accepted"])
        if np.isfinite(info["chosen_rank"]):
            chosen_ranks.append(float(info["chosen_rank"]))
        if int(info["success"]):
            break
    final = eval_state(wrapper, x0, x_cur, y)
    return x_cur, {
        "rescue_success": int(final["success"]),
        "final_pred": int(final["pred"]),
        "final_margin": float(final["margin"]),
        "final_linf": float(final["linf"]),
        "candidate_evals": candidate_evals,
        "accepted_steps": accepted_steps,
        "margin_drop_from_pgd_final": float(start["margin"] - final["margin"]),
        "mean_chosen_rank": float(np.mean(chosen_ranks)) if chosen_ranks else np.nan,
    }, candidate_rows


def load_images(store: ArtifactStore, model: str, split: str, max_images: int) -> pd.DataFrame:
    base = store.outcomes[(store.outcomes.model == model) & (store.outcomes.source == "pgd")][
        ["image_ord", "dataset_idx", "label"]
    ].drop_duplicates()
    if split != "all":
        base = base.merge(store.splits, on="image_ord", how="left")
        base = base[base.split == split]
    base = base.sort_values("image_ord")
    if max_images > 0:
        base = base.head(max_images)
    return base.reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["start", "rescue_steps", "mode"], dropna=False)
        .agg(
            n=("rescue_success", "size"),
            rescue_asr=("rescue_success", "mean"),
            mean_candidate_evals=("candidate_evals", "mean"),
            mean_accepted_steps=("accepted_steps", "mean"),
            mean_margin_drop_from_pgd_final=("margin_drop_from_pgd_final", "mean"),
            median_margin_drop_from_pgd_final=("margin_drop_from_pgd_final", "median"),
            mean_final_margin=("final_margin", "mean"),
            mean_chosen_rank=("mean_chosen_rank", "mean"),
            max_linf=("final_linf", "max"),
        )
        .reset_index()
    )


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for (start, steps), g in df.groupby(["start", "rescue_steps"]):
        if "ce_continue" not in set(g["mode"]):
            continue
        wide_s = g.pivot(index="image_ord", columns="mode", values="rescue_success")
        wide_m = g.pivot(index="image_ord", columns="mode", values="margin_drop_from_pgd_final")
        for mode in [c for c in wide_s.columns if c != "ce_continue"]:
            for metric, wide in [("rescue_success", wide_s), ("margin_drop", wide_m)]:
                d = (wide[mode] - wide["ce_continue"]).dropna().to_numpy(dtype=float)
                if len(d) == 0:
                    continue
                boots = []
                for _ in range(5000):
                    idx = rng.integers(0, len(d), len(d))
                    boots.append(float(d[idx].mean()))
                rows.append(
                    {
                        "start": start,
                        "rescue_steps": int(steps),
                        "mode": mode,
                        "baseline": "ce_continue",
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
    sub = summary[(summary.start == "pgd_final") & (summary.rescue_steps == summary.rescue_steps.max())].copy()
    order = ["ce_continue", "random_pixel", "global_rank1", "top5", "random5", "top10", "random10", "all", "pixelgrad"]
    sub["mode"] = pd.Categorical(sub["mode"], categories=order, ordered=True)
    sub = sub.sort_values("mode")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=180)
    axes[0].bar(sub["mode"].astype(str), sub["rescue_asr"], color="#59a14f")
    axes[0].set_ylabel("Rescue ASR among PGD failures")
    axes[0].set_title("Weak-PGD failure rescue")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(sub["mode"].astype(str), sub["mean_margin_drop_from_pgd_final"], color="#4c78a8")
    axes[1].set_ylabel("Mean margin drop from PGD final")
    axes[1].set_title("Progress after failed PGD state")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "pgd_failure_highway_rescue_summary.png", bbox_inches="tight")
    fig.savefig(out_dir / "pgd_failure_highway_rescue_summary.pdf", bbox_inches="tight")
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
    images = load_images(store, args.model, args.split, args.images)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    wrapper = load_model(args.model, device).eval()
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=device)
    eps = args.eps / 255.0
    pgd_step_size = args.pgd_step_size / 255.0
    rescue_step_size = args.rescue_step_size / 255.0
    rescue_steps_list = parse_int_csv(args.rescue_steps)
    modes = parse_csv(args.modes)
    starts = parse_csv(args.starts)
    rng = np.random.default_rng(args.seed)
    baseline_rows = []
    rescue_rows = []
    candidate_rows = []
    for i, row in enumerate(images.itertuples(index=False), start=1):
        x_cpu, _ = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        clean = eval_state(wrapper, x0, x0, y)
        states = run_pgd(wrapper, x0, y, eps, args.pgd_steps, pgd_step_size)
        pgd_final = eval_state(wrapper, x0, states[-1], y)
        baseline_rows.append(
            {
                "image_ord": int(row.image_ord),
                "dataset_idx": int(row.dataset_idx),
                "label": int(row.label),
                "clean_margin": clean["margin"],
                "pgd_final_margin": pgd_final["margin"],
                "pgd_success": int(pgd_final["success"]),
                "pgd_margin_drop": float(clean["margin"] - pgd_final["margin"]),
                "pgd_final_linf": pgd_final["linf"],
            }
        )
        if int(pgd_final["success"]):
            continue
        start_states = {"pgd_final": states[-1], "clean": x0}
        for start_name in starts:
            x_start = start_states[start_name]
            for steps in rescue_steps_list:
                for mode in modes:
                    seed = args.seed + int(row.image_ord) * 1009 + steps * 37 + sum(ord(c) for c in mode + start_name)
                    _x_rescue, info, cand = rescue_from_failed_state(
                        wrapper,
                        x0,
                        x_start,
                        y,
                        int(row.label),
                        args.layer,
                        basis_t,
                        routes,
                        mode,
                        eps,
                        rescue_step_size,
                        steps,
                        rng,
                        seed,
                    )
                    rescue_rows.append(
                        {
                            "image_ord": int(row.image_ord),
                            "dataset_idx": int(row.dataset_idx),
                            "label": int(row.label),
                            "start": start_name,
                            "rescue_steps": int(steps),
                            "mode": mode,
                            **info,
                        }
                    )
                    for c in cand:
                        c.update({"image_ord": int(row.image_ord), "dataset_idx": int(row.dataset_idx), "label": int(row.label), "start": start_name, "mode": mode, "rescue_steps": steps})
                    candidate_rows.extend(cand)
        if i % args.checkpoint_every == 0:
            pd.DataFrame(baseline_rows).to_csv(out_dir / "partial_pgd_failure_baseline.csv", index=False)
            pd.DataFrame(rescue_rows).to_csv(out_dir / "partial_pgd_failure_rescue_per_image.csv", index=False)
            print(f"[{i}/{len(images)}] pgd_failures={sum(1-r['pgd_success'] for r in baseline_rows)} rescue_rows={len(rescue_rows)}", flush=True)

    baseline = pd.DataFrame(baseline_rows)
    rescue = pd.DataFrame(rescue_rows)
    candidates = pd.DataFrame(candidate_rows)
    summary = summarize(rescue) if not rescue.empty else pd.DataFrame()
    deltas = paired_deltas(rescue) if not rescue.empty else pd.DataFrame()
    routes.to_csv(out_dir / "global_route_ranking.csv", index=False)
    baseline.to_csv(out_dir / "pgd_failure_baseline.csv", index=False)
    rescue.to_csv(out_dir / "pgd_failure_rescue_per_image.csv", index=False)
    candidates.to_csv(out_dir / "pgd_failure_rescue_candidates.csv", index=False)
    summary.to_csv(out_dir / "pgd_failure_rescue_summary.csv", index=False)
    deltas.to_csv(out_dir / "pgd_failure_rescue_paired_deltas.csv", index=False)
    if not summary.empty:
        plot_summary(summary, out_dir)
    meta = vars(args).copy()
    meta.update(
        {
            "device": str(device),
            "highway_train_vectors": int(n_highway_train),
            "n_images": int(len(images)),
            "pgd_asr": float(baseline["pgd_success"].mean()) if not baseline.empty else np.nan,
            "pgd_failures": int((baseline["pgd_success"] == 0).sum()) if not baseline.empty else 0,
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = [
        "# PGD Failure Highway Rescue",
        "",
        f"Weak PGD ASR: {meta['pgd_asr']:.3f}; failures: {meta['pgd_failures']} / {meta['n_images']}.",
        "",
        "## Summary",
        "",
    ]
    if not summary.empty:
        for r in summary.itertuples():
            lines.append(
                f"- start={r.start}, steps={r.rescue_steps}, `{r.mode}`: rescue_ASR={r.rescue_asr:.3f}, "
                f"candidate_evals={r.mean_candidate_evals:.1f}, margin_drop={r.mean_margin_drop_from_pgd_final:.3f}, "
                f"mean_rank={r.mean_chosen_rank:.2f}"
            )
    (out_dir / "pgd_failure_highway_rescue_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[PGD] ASR={meta['pgd_asr']:.3f} failures={meta['pgd_failures']}/{meta['n_images']}", flush=True)
    if not summary.empty:
        print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/pgd_failure_highway_rescue_bbb_resnet50_c200")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--rank-sources", default="pgd,square")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--split", default="all")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--pgd-steps", type=int, default=2)
    p.add_argument("--pgd-step-size", type=float, default=1.0)
    p.add_argument("--rescue-steps", default="1,2,3")
    p.add_argument("--rescue-step-size", type=float, default=1.0)
    p.add_argument("--starts", default="pgd_final")
    p.add_argument("--modes", default="ce_continue,random_pixel,global_rank1,top5,random5,top10,random10,all,pixelgrad")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
