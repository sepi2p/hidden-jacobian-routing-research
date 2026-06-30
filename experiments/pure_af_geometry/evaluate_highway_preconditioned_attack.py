#!/usr/bin/env python3
"""Fair-budget highway-entry preconditioning attack test.

The pre-stage is treated as part of the attack budget:

* every candidate evaluated during preconditioning counts against the same
  query budget used by Square;
* every state is projected into the same L_inf ball around the clean image;
* the follow-up Square search receives only the remaining budget.

The highway basis is fit from non-adversarial mobility vectors saved by the
balanced reviewer-null run.  No adversarial vectors are used to define the
preconditioning objective.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.extmath import randomized_svd
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.analyze_cifar_benchmark_optimizer_transport import p_selection  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import margin, project_linf  # noqa: E402


def parse_csv(s: str, typ=str) -> list:
    return [typ(x.strip()) for x in s.split(",") if x.strip()]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vector_key(model: str, source: str, layer: str) -> str:
    return f"{model}__{source}__{layer}"


def pca_basis(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < 2:
        raise ValueError("Need at least two rows for PCA.")
    kk = min(k, x.shape[0] - 1, x.shape[1])
    if kk < 1:
        raise ValueError("PCA rank is zero.")
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    xc = x - mean
    _u, _s, vt = randomized_svd(xc, n_components=kk, random_state=seed)
    return mean, vt.astype(np.float32)


def fit_highway_basis(input_dir: Path, model: str, source: str, layer: str, k: int, seed: int):
    rows = pd.read_csv(input_dir / "segment_metadata.csv")
    splits = pd.read_csv(input_dir / "image_splits.csv")
    split_by_image = dict(zip(splits["image_ord"].astype(int), splits["split"].astype(str)))
    arrays = np.load(input_dir / "segment_vectors.npz")
    key = vector_key(model, source, layer)
    sub = rows[(rows.model == model) & (rows.source == source) & (rows.layer == layer)].copy()
    if sub.empty or key not in arrays.files:
        raise RuntimeError(f"Missing highway data for {key}")
    sub["split"] = sub["image_ord"].map(split_by_image).fillna("")
    x = arrays[key][sub["vector_idx"].to_numpy(dtype=int)]
    train = sub["split"].to_numpy() == "train"
    if train.sum() < max(8, k + 2):
        raise RuntimeError(f"Too few train highway vectors for {key}: {train.sum()}")
    mean, basis = pca_basis(x[train], k, seed)
    return mean, basis, int(train.sum())


def load_eval_images(input_dir: Path, split_name: str, max_images: int | None):
    splits = pd.read_csv(input_dir / "image_splits.csv")
    outcomes = pd.read_csv(input_dir / "image_outcomes.csv")
    base = outcomes[outcomes.source == "pgd"][["image_ord", "dataset_idx", "label"]].drop_duplicates()
    sub = base.merge(splits, on="image_ord", how="left")
    sub = sub[sub.split == split_name].sort_values("image_ord")
    if max_images is not None and max_images > 0:
        sub = sub.head(max_images)
    return sub


def feature(wrapper, x: torch.Tensor, layer: str) -> torch.Tensor:
    _logits, feats, _raw = wrapper.forward_with_features(x)
    if layer not in feats:
        raise RuntimeError(f"Layer {layer} not captured by wrapper.")
    return feats[layer]


def highway_energy(delta_h: torch.Tensor, basis_t: torch.Tensor) -> torch.Tensor:
    coeff = delta_h @ basis_t.T
    num = (coeff * coeff).sum(dim=1)
    den = (delta_h * delta_h).sum(dim=1).clamp_min(1e-12)
    return num / den


def candidate_precondition(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    layer: str,
    basis_t: torch.Tensor,
    eps: float,
    pre_queries: int,
    candidates_per_round: int,
    step_size: float,
    mode: str,
    seed: int,
):
    gen = torch.Generator(device=x0.device).manual_seed(seed)
    x_cur = x0.detach().clone()
    used = 0
    pre_rows = []
    with torch.no_grad():
        h_cur = feature(wrapper, x_cur, layer).detach()
        clean_logits = wrapper(x0)
        clean_margin = float(margin(clean_logits, y).item())

    while used < pre_queries:
        bs = min(candidates_per_round, pre_queries - used)
        best_score = None
        best_x = None
        best_metrics = None
        for _ in range(bs):
            noise = torch.randn(x0.shape, generator=gen, device=x0.device).sign()
            cand = project_linf(x_cur + step_size * noise, x0, eps)
            with torch.no_grad():
                logits = wrapper(cand)
                h_cand = feature(wrapper, cand, layer).detach()
                dh = h_cand - h_cur
                e = float(highway_energy(dh, basis_t).item())
                spd = float(torch.norm(dh, dim=1).item())
                m = float(margin(logits, y).item())
            if mode == "random_pre":
                score = float(torch.rand((), generator=gen, device=x0.device).item())
            elif mode == "speed_pre":
                score = spd
            elif mode == "highway_pre":
                score = e
            elif mode == "margin_pre":
                score = clean_margin - m
            elif mode == "boundary_highway_pre":
                score = e * max(clean_margin - m, 0.0)
            elif mode == "joint_highway_margin_pre":
                score = e + max(clean_margin - m, 0.0)
            else:
                raise ValueError(f"unknown precondition mode {mode}")
            if best_score is None or score > best_score:
                best_score = score
                best_x = cand.detach()
                best_metrics = (e, spd, m)
        used += bs
        x_cur = best_x.detach()
        with torch.no_grad():
            h_cur = feature(wrapper, x_cur, layer).detach()
            pred = int(wrapper(x_cur).argmax(1).item())
        pre_rows.append(
            {
                "pre_round": len(pre_rows),
                "pre_queries_used": used,
                "chosen_score": float(best_score),
                "chosen_highway_energy": float(best_metrics[0]),
                "chosen_feature_speed": float(best_metrics[1]),
                "chosen_margin": float(best_metrics[2]),
                "chosen_success": int(pred != int(y.item())),
            }
        )
    return x_cur.detach(), used, pre_rows


def square_from_state(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    x_start: torch.Tensor | None,
    eps: float,
    total_queries: int,
    seed: int,
    p_init: float,
    init_epochs: int,
    pre_queries_used: int,
):
    gen = torch.Generator(device=x0.device).manual_seed(seed)
    c, h, w = x0.shape[1:]
    if x_start is None:
        stripe = torch.where(
            torch.rand((1, c, 1, w), generator=gen, device=x0.device) < 0.5,
            -torch.ones((1, c, 1, w), device=x0.device),
            torch.ones((1, c, 1, w), device=x0.device),
        ) * eps
        x_adv = (x0 + stripe).clamp(0, 1)
    else:
        x_adv = project_linf(x_start, x0, eps)

    queries = int(pre_queries_used)
    with torch.no_grad():
        logits = wrapper(x_adv)
        best_margin = margin(logits, y).detach()
        pred = int(logits.argmax(1).item())
    queries += 1
    if pred != int(y.item()):
        return x_adv.detach(), queries, queries, float(best_margin.item()), pred

    remaining = max(total_queries - queries, 0)
    success_query = math.nan
    final_pred = pred
    for step in range(1, remaining + 1):
        perturbation = (x_adv - x0).detach().clone()
        p = p_selection(p_init, step + init_epochs, max(remaining, 1))
        side = int(round(np.sqrt(p * c * h * w / c)))
        side = min(max(side, 1), h - 1)
        top = int(torch.randint(0, h - side + 1, (1,), generator=gen, device=x0.device).item())
        left = int(torch.randint(0, w - side + 1, (1,), generator=gen, device=x0.device).item())
        patch = torch.where(
            torch.rand((1, c, 1, 1), generator=gen, device=x0.device) < 0.5,
            -torch.ones((1, c, 1, 1), device=x0.device),
            torch.ones((1, c, 1, 1), device=x0.device),
        ) * eps
        perturbation[:, :, top : top + side, left : left + side] = patch
        candidate = (x0 + perturbation).clamp(0, 1)
        with torch.no_grad():
            cand_logits = wrapper(candidate)
            cand_margin = margin(cand_logits, y)
            cand_pred = int(cand_logits.argmax(1).item())
        queries += 1
        if float(cand_margin.item()) < float(best_margin.item()):
            x_adv = candidate.detach()
            best_margin = cand_margin.detach()
            final_pred = cand_pred
        if cand_pred != int(y.item()):
            success_query = queries
            final_pred = cand_pred
            x_adv = candidate.detach()
            best_margin = cand_margin.detach()
            break
    return x_adv.detach(), queries, success_query, float(best_margin.item()), final_pred


def eval_highway_state(wrapper, x0: torch.Tensor, x: torch.Tensor, y: torch.Tensor, layer: str, basis_t: torch.Tensor):
    with torch.no_grad():
        logits = wrapper(x)
        pred = int(logits.argmax(1).item())
        m = float(margin(logits, y).item())
        h0 = feature(wrapper, x0, layer).detach()
        hx = feature(wrapper, x, layer).detach()
        dh = hx - h0
        e = float(highway_energy(dh, basis_t).item())
        spd = float(torch.norm(dh, dim=1).item())
        linf = float((x - x0).abs().max().item())
    return {"pred": pred, "margin": m, "success": int(pred != int(y.item())), "highway_energy": e, "feature_speed": spd, "linf": linf}


def run(args):
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    input_dir = Path(args.input_dir)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    wrapper = load_model(args.model, device).eval()
    _mean, basis, n_train = fit_highway_basis(input_dir, args.model, args.highway_source, args.layer, args.highway_k, args.seed)
    basis_t = torch.tensor(basis, device=device, dtype=torch.float32)
    images = load_eval_images(input_dir, args.split, args.images)
    eps = args.eps / 255.0
    pre_step_size = args.pre_step_size / 255.0
    pre_queries_list = parse_csv(args.pre_queries, int)
    modes = parse_csv(args.pre_modes)
    rows = []
    pre_rows_all = []

    for idx, r in images.iterrows():
        image_ord = int(r.image_ord)
        dataset_idx = int(r.dataset_idx)
        label = int(r.label)
        x_cpu, _ = dataset[dataset_idx]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        clean = eval_highway_state(wrapper, x0, x0, y, args.layer, basis_t)
        for pre_queries in pre_queries_list:
            for mode in modes:
                if mode == "none" and pre_queries != 0:
                    continue
                if mode != "none" and pre_queries == 0:
                    continue
                seed = args.seed + image_ord * 1009 + pre_queries * 37 + sum(ord(c) for c in mode)
                if mode == "none":
                    x_start = None
                    used = 0
                    pre_state = clean.copy()
                    pre_stage_rows = []
                else:
                    x_pre, used, pre_stage_rows = candidate_precondition(
                        wrapper,
                        x0,
                        y,
                        args.layer,
                        basis_t,
                        eps,
                        pre_queries,
                        args.pre_candidates_per_round,
                        pre_step_size,
                        mode,
                        seed,
                    )
                    x_start = x_pre
                    pre_state = eval_highway_state(wrapper, x0, x_pre, y, args.layer, basis_t)
                x_adv, queries_used, q_success, final_margin, final_pred = square_from_state(
                    wrapper,
                    x0,
                    y,
                    x_start,
                    eps,
                    args.total_queries,
                    seed + 17,
                    args.square_p_init,
                    args.square_init_epochs,
                    used,
                )
                final = eval_highway_state(wrapper, x0, x_adv, y, args.layer, basis_t)
                rows.append(
                    {
                        "model": args.model,
                        "layer": args.layer,
                        "image_ord": image_ord,
                        "dataset_idx": dataset_idx,
                        "label": label,
                        "mode": mode,
                        "pre_queries_budget": pre_queries,
                        "pre_queries_used": used,
                        "total_query_budget": args.total_queries,
                        "queries_used": queries_used,
                        "query_to_success": q_success,
                        "clean_margin": clean["margin"],
                        "pre_margin": pre_state["margin"],
                        "final_margin": final_margin,
                        "clean_highway_energy": clean["highway_energy"],
                        "pre_highway_energy": pre_state["highway_energy"],
                        "final_highway_energy": final["highway_energy"],
                        "clean_feature_speed": clean["feature_speed"],
                        "pre_feature_speed": pre_state["feature_speed"],
                        "final_feature_speed": final["feature_speed"],
                        "pre_success": pre_state["success"],
                        "final_success": int(final_pred != label),
                        "final_pred": final_pred,
                        "final_linf": final["linf"],
                    }
                )
                for pr in pre_stage_rows:
                    pr.update(
                        {
                            "model": args.model,
                            "layer": args.layer,
                            "image_ord": image_ord,
                            "dataset_idx": dataset_idx,
                            "label": label,
                            "mode": mode,
                            "pre_queries_budget": pre_queries,
                        }
                    )
                    pre_rows_all.append(pr)
        if (len(rows) // max(1, len(modes))) % max(args.report_every, 1) == 0:
            print(f"[RUN] processed image_ord={image_ord}", flush=True)

    df = pd.DataFrame(rows)
    pre_df = pd.DataFrame(pre_rows_all)
    df.to_csv(out_dir / "highway_precondition_square_per_image.csv", index=False)
    pre_df.to_csv(out_dir / "highway_precondition_pre_stage.csv", index=False)
    summary_rows = []
    for (mode, preq), g in df.groupby(["mode", "pre_queries_budget"]):
        success = g["final_success"].astype(int).to_numpy()
        q = pd.to_numeric(g["query_to_success"], errors="coerce")
        q_filled = q.fillna(args.total_queries)
        summary_rows.append(
            {
                "mode": mode,
                "pre_queries_budget": int(preq),
                "n_images": int(len(g)),
                "asr": float(success.mean()),
                "pre_success_rate": float(g["pre_success"].astype(int).mean()),
                "mean_queries_success_only": float(q.dropna().mean()) if q.notna().any() else np.nan,
                "median_queries_success_only": float(q.dropna().median()) if q.notna().any() else np.nan,
                "mean_queries_all_failures_as_budget": float(q_filled.mean()),
                "median_queries_all_failures_as_budget": float(q_filled.median()),
                "mean_pre_highway_energy": float(g["pre_highway_energy"].mean()),
                "mean_final_highway_energy": float(g["final_highway_energy"].mean()),
                "mean_pre_margin_drop": float((g["clean_margin"] - g["pre_margin"]).mean()),
                "mean_total_margin_drop": float((g["clean_margin"] - g["final_margin"]).mean()),
                "max_linf": float(g["final_linf"].max()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["pre_queries_budget", "mode"])
    summary.to_csv(out_dir / "highway_precondition_square_summary.csv", index=False)
    meta = {
        "script": "experiments/pure_af_geometry/evaluate_highway_preconditioned_attack.py",
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "model": args.model,
        "layer": args.layer,
        "highway_source": args.highway_source,
        "highway_k": args.highway_k,
        "highway_train_vectors": n_train,
        "split": args.split,
        "images": int(len(images)),
        "eps": args.eps,
        "total_queries": args.total_queries,
        "pre_queries": pre_queries_list,
        "pre_modes": modes,
        "pre_candidates_per_round": args.pre_candidates_per_round,
        "pre_step_size": args.pre_step_size,
        "square_p_init": args.square_p_init,
        "square_init_epochs": args.square_init_epochs,
        "seed": args.seed,
        "device": str(device),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = ["# Highway-Preconditioned Square Pilot", "", "Every pre-stage candidate counts against the same total query budget.", ""]
    if not summary.empty:
        lines += ["## Summary", ""]
        for row in summary.itertuples():
            lines.append(
                f"- {row.mode}, pre_queries={row.pre_queries_budget}: ASR={row.asr:.3f}, "
                f"meanQ(all)={row.mean_queries_all_failures_as_budget:.1f}, "
                f"mean_pre_highway={row.mean_pre_highway_energy:.3f}, "
                f"pre_margin_drop={row.mean_pre_margin_drop:.3f}, total_margin_drop={row.mean_total_margin_drop:.3f}"
            )
    (out_dir / "highway_precondition_square_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[DONE] wrote {out_dir}", flush=True)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/highway_precondition_square_bbb_resnet50_test50")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--split", default="test")
    p.add_argument("--images", type=int, default=0, help="0 means all images in split.")
    p.add_argument("--eps", type=float, default=6.0, help="Linf epsilon in /255 units.")
    p.add_argument("--total-queries", type=int, default=250)
    p.add_argument("--pre-queries", default="0,25,50")
    p.add_argument("--pre-modes", default="none,random_pre,speed_pre,highway_pre,margin_pre,boundary_highway_pre,joint_highway_margin_pre")
    p.add_argument("--pre-candidates-per-round", type=int, default=5)
    p.add_argument("--pre-step-size", type=float, default=1.0, help="pre-stage step size in /255 units")
    p.add_argument("--square-p-init", type=float, default=0.8)
    p.add_argument("--square-init-epochs", type=int, default=0)
    p.add_argument("--report-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
