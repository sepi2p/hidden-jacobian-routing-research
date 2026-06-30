#!/usr/bin/env python3
"""Test whether magically entering an adversarial highway improves PGD.

For each image, evaluate all signed high-mobility hidden-layer routes from the
clean state.  Among routes that reduce the true-class margin, choose a
"nearest" adversarial highway by local route energy or feature speed, then
continue PGD from that pre-state inside the same L_inf ball.

This is a white-box diagnostic/upper-bound experiment.  The pre-stage can use
oracle one-step margin-drop information, so it should not be described as a
practical attack unless explicitly compared under matched costs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.evaluate_astar_highway_from_start import eval_state, feature_tensor  # noqa: E402
from experiments.pure_af_geometry.evaluate_image_conditioned_highway_selector import (  # noqa: E402
    ArtifactStore,
    fit_highway_basis,
    project_attack_rows,
    rank_signed_routes,
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


def ce_pgd(wrapper, x0: torch.Tensor, x_start: torch.Tensor, y: torch.Tensor, eps: float, step_size: float, steps: int):
    x = x_start.detach()
    states = []
    for _ in range(max(steps, 0)):
        probe = x.detach().requires_grad_(True)
        logits = wrapper(probe)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, probe)[0]
        x = project_linf(x + step_size * grad.sign(), x0, eps).detach()
        ev = eval_state(wrapper, x0, x, y)
        states.append(ev)
        if int(ev["success"]):
            break
    return x, {**eval_state(wrapper, x0, x, y), "pgd_steps_used": len(states)}


def route_step(wrapper, x0, x_cur, y, layer, direction, eps, step_size):
    h0 = feature_tensor(wrapper, x_cur, layer).detach()
    probe = x_cur.detach().requires_grad_(True)
    h = feature_tensor(wrapper, probe, layer)
    scalar = (h * direction.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, probe)[0]
    x_next = project_linf(x_cur + step_size * grad.sign(), x0, eps).detach()
    h1 = feature_tensor(wrapper, x_next, layer).detach()
    dh = h1 - h0
    ev0 = eval_state(wrapper, x0, x_cur, y)
    ev1 = eval_state(wrapper, x0, x_next, y)
    coeff = (dh * direction.view_as(dh)).sum(dim=1)
    route_energy = float((coeff.pow(2) / dh.pow(2).sum(dim=1).clamp_min(1e-12)).item())
    return x_next, {
        "route_energy": route_energy,
        "feature_speed": float(torch.norm(dh, dim=1).item()),
        "margin_drop": float(ev0["margin"] - ev1["margin"]),
        "after_margin": float(ev1["margin"]),
        "after_pred": int(ev1["pred"]),
        "success": int(ev1["success"]),
        "linf": float(ev1["linf"]),
    }


def evaluate_routes(wrapper, x0, y, layer, basis_t, routes, eps, step_size):
    rows = []
    xs = {}
    for route in routes.itertuples(index=False):
        direction = int(route.sign) * basis_t[int(route.pc) - 1]
        x_next, ev = route_step(wrapper, x0, x0, y, layer, direction, eps, step_size)
        row = {
            "pc": int(route.pc),
            "sign": int(route.sign),
            "route": str(route.route),
            "global_rank": int(route.global_rank),
            "global_score": float(route.train_weighted_margin_drop_score),
            **ev,
        }
        rows.append(row)
        xs[str(route.route)] = x_next
    return pd.DataFrame(rows), xs


def choose_route(candidates: pd.DataFrame, variant: str, rng: np.random.Generator):
    adv = candidates[candidates.margin_drop > 0].copy()
    pool = adv if len(adv) else candidates.copy()
    if variant == "closest_energy":
        return pool.sort_values("route_energy", ascending=False).iloc[0]
    if variant == "closest_speed":
        return pool.sort_values("feature_speed", ascending=False).iloc[0]
    if variant == "highest_reward":
        return pool.sort_values("margin_drop", ascending=False).iloc[0]
    if variant == "global_rank1":
        return candidates.sort_values("global_rank", ascending=True).iloc[0]
    if variant == "random_adv":
        return pool.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
    raise ValueError(f"Unknown variant {variant}")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["method", "budget_mode", "pgd_budget"], dropna=False)
        .agg(
            n=("success", "size"),
            asr=("success", "mean"),
            mean_margin=("margin", "mean"),
            median_margin=("margin", "median"),
            mean_margin_drop=("total_margin_drop", "mean"),
            mean_pre_margin_drop=("pre_margin_drop", "mean"),
            mean_pgd_steps=("pgd_steps_used", "mean"),
            mean_linf=("linf", "mean"),
        )
        .reset_index()
        .sort_values(["pgd_budget", "budget_mode", "asr"], ascending=[True, True, False])
    )


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for budget, sub in df.groupby("pgd_budget"):
        base = sub[(sub.method == "pgd") & (sub.budget_mode == "baseline")]
        if base.empty:
            continue
        wide_s = sub.pivot_table(index="image_ord", columns=["method", "budget_mode"], values="success", aggfunc="first")
        wide_m = sub.pivot_table(index="image_ord", columns=["method", "budget_mode"], values="total_margin_drop", aggfunc="first")
        bkey = ("pgd", "baseline")
        for key in wide_s.columns:
            if key == bkey:
                continue
            for metric, wide in [("success", wide_s), ("margin_drop", wide_m)]:
                d = (wide[key] - wide[bkey]).dropna().to_numpy(float)
                if len(d) == 0:
                    continue
                boots = []
                for _ in range(3000):
                    idx = rng.integers(0, len(d), len(d))
                    boots.append(float(d[idx].mean()))
                rows.append(
                    {
                        "pgd_budget": int(budget),
                        "method": key[0],
                        "budget_mode": key[1],
                        "metric": metric,
                        "n": int(len(d)),
                        "mean_delta": float(d.mean()),
                        "ci_low": float(np.quantile(boots, 0.025)),
                        "ci_high": float(np.quantile(boots, 0.975)),
                        "fraction_better": float((d > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/magical_highway_preattack_pgd_bbb_resnet50_c200")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--rank-sources", default="pgd,square")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--split", default="all")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--pre-step-size", type=float, default=1.0)
    p.add_argument("--pgd-step-size", type=float, default=1.0)
    p.add_argument("--pgd-budgets", default="2,3,5")
    p.add_argument("--variants", default="closest_energy,closest_speed,highest_reward,global_rank1,random_adv")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    store = ArtifactStore(Path(args.input_dir))
    _mean, basis, n_train = fit_highway_basis(store, args.model, args.highway_source, args.layer, args.highway_k, args.seed)
    projected = project_attack_rows(store, args.model, parse_csv(args.rank_sources), args.layer, basis)
    routes = rank_signed_routes(projected, parse_csv(args.rank_sources), args.highway_k)
    wrapper = load_model(args.model, device).eval()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    images = load_images(store, args.model, args.split, args.images)
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=device)
    eps = args.eps / 255.0
    pre_step = args.pre_step_size / 255.0
    pgd_step = args.pgd_step_size / 255.0
    budgets = parse_int_csv(args.pgd_budgets)
    variants = parse_csv(args.variants)
    rng = np.random.default_rng(args.seed)
    rows = []
    candidate_rows = []
    for i, row in enumerate(images.itertuples(index=False), start=1):
        x_cpu, _ = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        clean = eval_state(wrapper, x0, x0, y)
        candidates, x_by_route = evaluate_routes(wrapper, x0, y, args.layer, basis_t, routes, eps, pre_step)
        candidates["image_ord"] = int(row.image_ord)
        candidates["dataset_idx"] = int(row.dataset_idx)
        candidates["label"] = int(row.label)
        candidate_rows.extend(candidates.to_dict("records"))
        chosen = {variant: choose_route(candidates, variant, rng) for variant in variants}
        for budget in budgets:
            _x, ev = ce_pgd(wrapper, x0, x0, y, eps, pgd_step, budget)
            rows.append(
                {
                    "image_ord": int(row.image_ord),
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "method": "pgd",
                    "budget_mode": "baseline",
                    "pgd_budget": int(budget),
                    "chosen_route": "",
                    "chosen_global_rank": np.nan,
                    "pre_success": 0,
                    "pre_margin_drop": 0.0,
                    "success": int(ev["success"]),
                    "pred": int(ev["pred"]),
                    "margin": float(ev["margin"]),
                    "total_margin_drop": float(clean["margin"] - ev["margin"]),
                    "pgd_steps_used": int(ev["pgd_steps_used"]),
                    "linf": float(ev["linf"]),
                }
            )
            for variant, route in chosen.items():
                x_pre = x_by_route[str(route.route)]
                for mode, follow_steps in [("same_total_steps", max(budget - 1, 0)), ("extra_prestep", budget)]:
                    _x, ev = ce_pgd(wrapper, x0, x_pre, y, eps, pgd_step, follow_steps)
                    rows.append(
                        {
                            "image_ord": int(row.image_ord),
                            "dataset_idx": int(row.dataset_idx),
                            "label": int(row.label),
                            "method": variant,
                            "budget_mode": mode,
                            "pgd_budget": int(budget),
                            "chosen_route": str(route.route),
                            "chosen_global_rank": int(route.global_rank),
                            "pre_success": int(route.success),
                            "pre_margin_drop": float(route.margin_drop),
                            "pre_route_energy": float(route.route_energy),
                            "pre_feature_speed": float(route.feature_speed),
                            "success": int(ev["success"]),
                            "pred": int(ev["pred"]),
                            "margin": float(ev["margin"]),
                            "total_margin_drop": float(clean["margin"] - ev["margin"]),
                            "pgd_steps_used": int(ev["pgd_steps_used"]),
                            "linf": float(ev["linf"]),
                        }
                    )
        if i % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(out_dir / "partial_magical_highway_preattack_pgd_per_image.csv", index=False)
            print(f"[{i}/{len(images)}] rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    candidates = pd.DataFrame(candidate_rows)
    summary = summarize(df)
    deltas = paired_deltas(df)
    df.to_csv(out_dir / "magical_highway_preattack_pgd_per_image.csv", index=False)
    candidates.to_csv(out_dir / "magical_highway_preattack_candidates.csv", index=False)
    summary.to_csv(out_dir / "magical_highway_preattack_pgd_summary.csv", index=False)
    deltas.to_csv(out_dir / "magical_highway_preattack_pgd_paired_deltas.csv", index=False)
    meta = vars(args).copy()
    meta.update({"device": str(device), "n_images": int(len(images)), "highway_train_vectors": int(n_train)})
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = ["# Magical Highway Pre-Attack + PGD", "", "## Summary", ""]
    for r in summary.itertuples(index=False):
        lines.append(
            f"- `{r.method}` / `{r.budget_mode}` / PGD budget {r.pgd_budget}: "
            f"ASR={r.asr:.3f}, margin_drop={r.mean_margin_drop:.3f}, pre_drop={r.mean_pre_margin_drop:.3f}"
        )
    (out_dir / "magical_highway_preattack_pgd_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
