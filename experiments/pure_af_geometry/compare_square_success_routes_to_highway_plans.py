#!/usr/bin/env python3
"""Compare white-box traffic routes to observed Square success-flow paths.

For held-out Square-success images, this diagnostic constructs route-planning
paths over signed high-mobility hidden-layer directions and compares those
paths to the actual Square trajectory segments from the same image.  It also
scores the same paths against PGD success-flow and Jacobian-probe bases to
separate attack-flow similarity from generic hidden-Jacobian mobility.

This is not an attack-performance script.  It asks whether the routes found by
the highway/traffic view resemble the adversarial flow that Square actually
used.
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.utils.extmath import randomized_svd
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.evaluate_astar_highway_from_start import (  # noqa: E402
    eval_state,
    feature_tensor,
    route_candidate,
    select_routes,
)
from experiments.pure_af_geometry.evaluate_image_conditioned_highway_selector import (  # noqa: E402
    ArtifactStore,
    fit_highway_basis,
    margin_pixel_grad,
    project_attack_rows,
    rank_signed_routes,
)
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import project_linf  # noqa: E402


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def fit_pca_basis(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x = normalize_rows(x.astype(np.float32))
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    kk = min(k, x.shape[0] - 1, x.shape[1])
    if kk < 1:
        raise ValueError("Not enough vectors to fit PCA basis.")
    _u, _s, vt = randomized_svd(x - mean, n_components=kk, random_state=seed)
    return mean, vt.astype(np.float32)


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    x = normalize_rows(x.astype(np.float32))
    xc = x - mean
    kk = min(k, basis.shape[0])
    num = np.sum((xc @ basis[:kk].T) ** 2, axis=1)
    den = np.sum(xc * xc, axis=1)
    return num / np.clip(den, 1e-12, None)


def cosine_metrics(route_vecs: np.ndarray, square_vecs: np.ndarray) -> dict:
    if len(route_vecs) == 0 or len(square_vecs) == 0:
        return {
            "mean_max_signed_cos": np.nan,
            "mean_max_abs_cos": np.nan,
            "cumulative_cos": np.nan,
            "cumulative_abs_cos": np.nan,
        }
    r = normalize_rows(route_vecs)
    s = normalize_rows(square_vecs)
    sim = r @ s.T
    r_sum = route_vecs.sum(axis=0, keepdims=True)
    s_sum = square_vecs.sum(axis=0, keepdims=True)
    c = float((normalize_rows(r_sum) @ normalize_rows(s_sum).T)[0, 0])
    return {
        "mean_max_signed_cos": float(sim.max(axis=1).mean()),
        "mean_max_abs_cos": float(np.abs(sim).max(axis=1).mean()),
        "cumulative_cos": c,
        "cumulative_abs_cos": abs(c),
    }


def square_success_images(store: ArtifactStore, model: str, split: str, max_images: int) -> pd.DataFrame:
    base = store.outcomes[
        (store.outcomes.model == model) & (store.outcomes.source == "square") & (store.outcomes.final_success == 1)
    ][["image_ord", "dataset_idx", "label", "final_pred"]].drop_duplicates()
    base = base.merge(store.splits, on="image_ord", how="left")
    if split != "all":
        base = base[base.split == split]
    base = base.sort_values("image_ord")
    if max_images > 0:
        base = base.head(max_images)
    return base.reset_index(drop=True)


def observed_square_vectors(store: ArtifactStore, model: str, layer: str, image_ord: int) -> np.ndarray:
    rows, x = store.rows_for(model, "square", layer)
    keep = (rows.image_ord.to_numpy(dtype=int) == int(image_ord)) & (rows.final_success.to_numpy(dtype=int) == 1)
    sub = rows[keep].sort_values("step")
    if sub.empty:
        return np.zeros((0, x.shape[1]), dtype=np.float32)
    return x[sub.index.to_numpy()].astype(np.float32)


def observed_vectors(store: ArtifactStore, model: str, source: str, layer: str, image_ord: int) -> np.ndarray:
    rows, x = store.rows_for(model, source, layer)
    keep = rows.image_ord.to_numpy(dtype=int) == int(image_ord)
    sub = rows[keep].sort_values("step")
    if sub.empty:
        return np.zeros((0, x.shape[1]), dtype=np.float32)
    return x[sub.index.to_numpy()].astype(np.float32)


def fit_source_basis(
    store: ArtifactStore,
    model: str,
    source: str,
    layer: str,
    k: int,
    seed: int,
    successful_only: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    rows, x = store.rows_for(model, source, layer)
    if rows.empty:
        raise RuntimeError(f"No vectors for {source}/{layer}.")
    keep = rows["split"].to_numpy() == "train"
    if successful_only and "final_success" in rows.columns:
        keep &= rows["final_success"].to_numpy(dtype=int) == 1
    if keep.sum() < max(8, k + 2):
        raise RuntimeError(f"Too few vectors for basis {source}/{layer}: {keep.sum()}.")
    mean, basis = fit_pca_basis(x[keep], k, seed)
    return mean, basis, int(keep.sum())


def route_step_with_delta(wrapper, x0, x_cur, y, layer, direction, eps, step_size):
    h_before = feature_tensor(wrapper, x_cur, layer).detach()
    x_next = route_candidate(wrapper, x0, x_cur, layer, direction, eps, step_size)
    h_after = feature_tensor(wrapper, x_next, layer).detach()
    return x_next.detach(), (h_after - h_before).detach().cpu().numpy()[0].astype(np.float32)


def ce_step_with_delta(wrapper, x0, x_cur, y, layer, eps, step_size):
    h_before = feature_tensor(wrapper, x_cur, layer).detach()
    probe = x_cur.detach().requires_grad_(True)
    logits = wrapper(probe)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, probe)[0]
    x_next = project_linf(x_cur + step_size * grad.sign(), x0, eps).detach()
    h_after = feature_tensor(wrapper, x_next, layer).detach()
    return x_next, (h_after - h_before).detach().cpu().numpy()[0].astype(np.float32)


def random_step_with_delta(wrapper, x0, x_cur, y, layer, eps, step_size, gen):
    h_before = feature_tensor(wrapper, x_cur, layer).detach()
    direction = torch.randn(x_cur.shape, generator=gen, device=x_cur.device).sign()
    x_next = project_linf(x_cur + step_size * direction, x0, eps).detach()
    h_after = feature_tensor(wrapper, x_next, layer).detach()
    return x_next, (h_after - h_before).detach().cpu().numpy()[0].astype(np.float32)


def run_greedy_route(wrapper, x0, y, layer, basis_t, routes, action_set, eps, step_size, max_depth, rng):
    x = x0.detach()
    vecs = []
    path = []
    evals = 0
    for _depth in range(max_depth):
        candidates = select_routes(wrapper, x, y, layer, basis_t, routes, action_set, rng)
        best = None
        best_x = None
        best_vec = None
        for route in candidates.itertuples():
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            x_cand, delta = route_step_with_delta(wrapper, x0, x, y, layer, direction, eps, step_size)
            ev = eval_state(wrapper, x0, x_cand, y)
            evals += 1
            if best is None or ev["margin"] < best["margin"]:
                best = {**ev, "route": str(route.route), "global_rank": int(route.global_rank)}
                best_x = x_cand
                best_vec = delta
        x = best_x.detach()
        vecs.append(best_vec)
        path.append(str(best["route"]))
        if int(best["success"]):
            break
    final = eval_state(wrapper, x0, x, y)
    return np.stack(vecs, axis=0), {**final, "evals": evals, "depth": len(vecs), "path": "|".join(path)}


def priority(depth: int, margin_value: float, clean_margin: float, heuristic_weight: float) -> float:
    return float(depth + heuristic_weight * max(margin_value, 0.0) / max(abs(clean_margin), 1e-6))


def run_astar_route(
    wrapper,
    x0,
    y,
    layer,
    basis_t,
    routes,
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
    counter = 0
    frontier = [(priority(0, clean_margin, clean_margin, heuristic_weight), counter, x0.detach(), clean_margin, 0, tuple(), tuple(), 0)]
    best = frontier[0]
    expansions = 0
    total_evals = 0
    while frontier and expansions < max_expansions:
        node = heapq.heappop(frontier)
        _pr, _tie, x, node_margin, depth, path, vecs, node_evals = node
        if node_margin < best[3]:
            best = node
        if node_margin < 0 or depth >= max_depth:
            continue
        children = []
        candidates = select_routes(wrapper, x, y, layer, basis_t, routes, action_set, rng)
        for route in candidates.itertuples():
            direction = int(route.sign) * basis_t[int(route.pc) - 1]
            x_cand, delta = route_step_with_delta(wrapper, x0, x, y, layer, direction, eps, step_size)
            ev = eval_state(wrapper, x0, x_cand, y)
            total_evals += 1
            counter += 1
            child_depth = depth + 1
            child_path = path + (str(route.route),)
            child_vecs = vecs + (delta,)
            child_pr = priority(child_depth, float(ev["margin"]), clean_margin, heuristic_weight)
            child = (child_pr, counter, x_cand.detach(), float(ev["margin"]), child_depth, child_path, child_vecs, node_evals + 1)
            children.append(child)
            if float(ev["margin"]) < best[3]:
                best = child
            if int(ev["success"]):
                frontier = [child]
                break
        expansions += 1
        if best[3] < 0:
            break
        frontier.extend(children)
        frontier = heapq.nsmallest(beam_size, frontier)
        heapq.heapify(frontier)
    _pr, _tie, best_x, _m, depth, path, vecs, _node_evals = best
    final = eval_state(wrapper, x0, best_x, y)
    if len(vecs) == 0:
        vec_arr = np.zeros((0, int(basis_t.shape[1])), dtype=np.float32)
    else:
        vec_arr = np.stack(vecs, axis=0)
    return vec_arr, {
        **final,
        "evals": total_evals,
        "depth": int(depth),
        "path": "|".join(path),
        "expansions": int(expansions),
    }


def run_ce_route(wrapper, x0, y, layer, eps, step_size, max_depth):
    x = x0.detach()
    vecs = []
    for _ in range(max_depth):
        x, delta = ce_step_with_delta(wrapper, x0, x, y, layer, eps, step_size)
        vecs.append(delta)
        if eval_state(wrapper, x0, x, y)["success"]:
            break
    return np.stack(vecs, axis=0), {**eval_state(wrapper, x0, x, y), "evals": len(vecs), "depth": len(vecs), "path": "ce"}


def run_random_route(wrapper, x0, y, layer, eps, step_size, max_depth, gen):
    x = x0.detach()
    vecs = []
    for _ in range(max_depth):
        x, delta = random_step_with_delta(wrapper, x0, x, y, layer, eps, step_size, gen)
        vecs.append(delta)
        if eval_state(wrapper, x0, x, y)["success"]:
            break
    return np.stack(vecs, axis=0), {**eval_state(wrapper, x0, x, y), "evals": len(vecs), "depth": len(vecs), "path": "random"}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("method")
        .agg(
            n=("image_ord", "size"),
            route_asr=("route_success", "mean"),
            mean_observed_square_squareflow_energy=("observed_square_squareflow_energy", "mean"),
            mean_route_squareflow_energy=("route_squareflow_energy", "mean"),
            mean_route_pgdflow_energy=("route_pgdflow_energy", "mean"),
            mean_route_jacobian_energy=("route_jacobian_energy", "mean"),
            mean_route_mobility_energy=("route_mobility_energy", "mean"),
            mean_max_signed_cos=("mean_max_signed_cos", "mean"),
            mean_max_abs_cos=("mean_max_abs_cos", "mean"),
            mean_max_abs_cos_to_pgd=("mean_max_abs_cos_to_pgd", "mean"),
            mean_cumulative_cos=("cumulative_cos", "mean"),
            mean_cumulative_abs_cos=("cumulative_abs_cos", "mean"),
            mean_margin_drop=("route_margin_drop", "mean"),
            mean_evals=("route_evals", "mean"),
            mean_depth=("route_depth", "mean"),
        )
        .reset_index()
        .sort_values(["mean_route_squareflow_energy", "mean_max_abs_cos"], ascending=False)
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/square_success_traffic_route_similarity_bbb_resnet50_test")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--split", default="test")
    p.add_argument("--images", type=int, default=0)
    p.add_argument("--success-k", type=int, default=20)
    p.add_argument("--energy-k", type=int, default=20)
    p.add_argument("--jacobian-source", default="jacobian_probe_all")
    p.add_argument("--mobility-source", default="mobility_top_walk_square_budget")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--rank-sources", default="pgd,square")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--eps", type=float, default=6.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--route-methods", default="greedy_top20,greedy_pixelgrad20,greedy_all,ce,random")
    p.add_argument("--max-expansions", type=int, default=40)
    p.add_argument("--beam-size", type=int, default=10)
    p.add_argument("--heuristic-weight", type=float, default=1.0)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    store = ArtifactStore(Path(args.input_dir))
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    wrapper = load_model(args.model, device).eval()

    square_flow_mean, square_flow_basis, n_square_success_train = fit_source_basis(
        store, args.model, "square", args.layer, args.success_k, args.seed, successful_only=True
    )
    pgd_flow_mean, pgd_flow_basis, n_pgd_success_train = fit_source_basis(
        store, args.model, "pgd", args.layer, args.success_k, args.seed + 1, successful_only=True
    )
    jacobian_mean, jacobian_basis, n_jacobian_train = fit_source_basis(
        store, args.model, args.jacobian_source, args.layer, args.success_k, args.seed + 2, successful_only=False
    )
    mobility_mean, mobility_basis, n_mobility_train = fit_source_basis(
        store, args.model, args.mobility_source, args.layer, args.success_k, args.seed + 3, successful_only=False
    )

    _hw_mean, hw_basis, n_highway_train = fit_highway_basis(
        store, args.model, args.highway_source, args.layer, args.highway_k, args.seed
    )
    projected = project_attack_rows(store, args.model, parse_csv(args.rank_sources), args.layer, hw_basis)
    routes = rank_signed_routes(projected, parse_csv(args.rank_sources), args.highway_k)
    basis_t = torch.as_tensor(hw_basis, dtype=torch.float32, device=device)
    images = square_success_images(store, args.model, args.split, args.images)
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    rng = np.random.default_rng(args.seed)
    rows = []

    for i, row in enumerate(images.itertuples(index=False), start=1):
        square_vecs = observed_square_vectors(store, args.model, args.layer, int(row.image_ord))
        if len(square_vecs) == 0:
            continue
        pgd_vecs = observed_vectors(store, args.model, "pgd", args.layer, int(row.image_ord))
        x_cpu, _ = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        clean = eval_state(wrapper, x0, x0, y)
        observed_square_squareflow_energy = float(
            np.mean(projection_energy(square_vecs, square_flow_mean, square_flow_basis, args.energy_k))
        )
        observed_square_pgdflow_energy = float(
            np.mean(projection_energy(square_vecs, pgd_flow_mean, pgd_flow_basis, args.energy_k))
        )
        observed_square_jacobian_energy = float(
            np.mean(projection_energy(square_vecs, jacobian_mean, jacobian_basis, args.energy_k))
        )
        observed_square_mobility_energy = float(
            np.mean(projection_energy(square_vecs, mobility_mean, mobility_basis, args.energy_k))
        )
        if len(pgd_vecs):
            observed_pgd_squareflow_energy = float(
                np.mean(projection_energy(pgd_vecs, square_flow_mean, square_flow_basis, args.energy_k))
            )
            observed_pgd_pgdflow_energy = float(
                np.mean(projection_energy(pgd_vecs, pgd_flow_mean, pgd_flow_basis, args.energy_k))
            )
        else:
            observed_pgd_squareflow_energy = np.nan
            observed_pgd_pgdflow_energy = np.nan
        for method in parse_csv(args.route_methods):
            gen = torch.Generator(device=device).manual_seed(args.seed + 1009 * int(row.image_ord))
            if method == "ce":
                route_vecs, final = run_ce_route(wrapper, x0, y, args.layer, eps, step_size, args.max_depth)
            elif method == "random":
                route_vecs, final = run_random_route(wrapper, x0, y, args.layer, eps, step_size, args.max_depth, gen)
            elif method.startswith("greedy_"):
                action_set = method.replace("greedy_", "")
                route_vecs, final = run_greedy_route(
                    wrapper, x0, y, args.layer, basis_t, routes, action_set, eps, step_size, args.max_depth, rng
                )
            elif method.startswith("astar_"):
                action_set = method.replace("astar_", "")
                route_vecs, final = run_astar_route(
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
            else:
                raise ValueError(f"Unsupported method {method}")
            if len(route_vecs) == 0:
                continue
            route_squareflow_energy = float(
                np.mean(projection_energy(route_vecs, square_flow_mean, square_flow_basis, args.energy_k))
            )
            route_pgdflow_energy = float(
                np.mean(projection_energy(route_vecs, pgd_flow_mean, pgd_flow_basis, args.energy_k))
            )
            route_jacobian_energy = float(
                np.mean(projection_energy(route_vecs, jacobian_mean, jacobian_basis, args.energy_k))
            )
            route_mobility_energy = float(
                np.mean(projection_energy(route_vecs, mobility_mean, mobility_basis, args.energy_k))
            )
            sims = cosine_metrics(route_vecs, square_vecs)
            pgd_sims = cosine_metrics(route_vecs, pgd_vecs) if len(pgd_vecs) else {
                "mean_max_signed_cos": np.nan,
                "mean_max_abs_cos": np.nan,
                "cumulative_cos": np.nan,
                "cumulative_abs_cos": np.nan,
            }
            rows.append(
                {
                    "image_ord": int(row.image_ord),
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "square_final_pred": int(row.final_pred),
                    "method": method,
                    "square_n_steps": int(len(square_vecs)),
                    "pgd_n_steps": int(len(pgd_vecs)),
                    "observed_square_squareflow_energy": observed_square_squareflow_energy,
                    "observed_square_pgdflow_energy": observed_square_pgdflow_energy,
                    "observed_square_jacobian_energy": observed_square_jacobian_energy,
                    "observed_square_mobility_energy": observed_square_mobility_energy,
                    "observed_pgd_squareflow_energy": observed_pgd_squareflow_energy,
                    "observed_pgd_pgdflow_energy": observed_pgd_pgdflow_energy,
                    "route_squareflow_energy": route_squareflow_energy,
                    "route_pgdflow_energy": route_pgdflow_energy,
                    "route_jacobian_energy": route_jacobian_energy,
                    "route_mobility_energy": route_mobility_energy,
                    "squareflow_energy_gap": route_squareflow_energy - observed_square_squareflow_energy,
                    "pgdflow_energy_gap": route_pgdflow_energy - observed_square_pgdflow_energy,
                    "route_success": int(final["success"]),
                    "route_final_pred": int(final["pred"]),
                    "route_margin": float(final["margin"]),
                    "route_margin_drop": float(clean["margin"] - final["margin"]),
                    "route_evals": int(final["evals"]),
                    "route_depth": int(final["depth"]),
                    "route_path": str(final["path"]),
                    **sims,
                    "mean_max_signed_cos_to_pgd": float(pgd_sims["mean_max_signed_cos"]),
                    "mean_max_abs_cos_to_pgd": float(pgd_sims["mean_max_abs_cos"]),
                    "cumulative_cos_to_pgd": float(pgd_sims["cumulative_cos"]),
                    "cumulative_abs_cos_to_pgd": float(pgd_sims["cumulative_abs_cos"]),
                }
            )
        if i % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(out_dir / "partial_square_route_similarity_per_image.csv", index=False)
            print(f"[{i}/{len(images)}] rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    summary = summarize(df)
    df.to_csv(out_dir / "square_route_similarity_per_image.csv", index=False)
    summary.to_csv(out_dir / "square_route_similarity_summary.csv", index=False)
    routes.to_csv(out_dir / "signed_highway_route_ranking.csv", index=False)
    metadata = vars(args).copy()
    metadata.update(
        {
            "device": str(device),
            "n_images": int(len(images)),
            "n_square_success_train_vectors": int(n_square_success_train),
            "n_pgd_success_train_vectors": int(n_pgd_success_train),
            "n_jacobian_train_vectors": int(n_jacobian_train),
            "n_mobility_train_vectors": int(n_mobility_train),
            "n_highway_train_vectors": int(n_highway_train),
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lines = [
        "# Square Success Route Similarity",
        "",
        "White-box traffic routes are compared to observed held-out Square-success trajectories from the same images.",
        "",
        "## Summary",
        "",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"- `{r.method}`: square_flow_energy={r.mean_route_squareflow_energy:.3f}, "
            f"pgd_flow_energy={r.mean_route_pgdflow_energy:.3f}, jacobian_energy={r.mean_route_jacobian_energy:.3f}, "
            f"max_abs_cos_to_square={r.mean_max_abs_cos:.3f}, max_abs_cos_to_pgd={r.mean_max_abs_cos_to_pgd:.3f}, "
            f"route_ASR={r.route_asr:.3f}"
        )
    (out_dir / "square_route_similarity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
