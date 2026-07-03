#!/usr/bin/env python3
"""Multi-model CIFAR-10 benchmark for hidden-Jacobian road routing.

This is the reviewer-facing follow-up to the single-model ResNet50 road-routing
result.  It runs the same constructive road attack on the four BlackboxBench
CIFAR-10 architectures using their existing balanced clean-correct cohorts, and
compares against official AutoAttack APGD components plus FAB where requested.

The script is resumable at method/model level.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import load_model as load_feature_model  # noqa: E402
from experiments.hidden_jacobian_routing.evaluate_efficient_road_routing_ablation import bootstrap_delta  # noqa: E402
from experiments.hidden_jacobian_routing.evaluate_margin_selected_singular_road_on_balanced import logits_stats  # noqa: E402
from experiments.hidden_jacobian_routing.common import margin, project_linf  # noqa: E402
from experiments.hidden_jacobian_routing.trace_jacobian_singular_roads import set_seed  # noqa: E402


BASE = Path("analysis_outputs/hidden_jacobian_routing/jacobian_null_response")
MODEL_CFG = {
    "bbb_resnet50": {
        "layer": "layer4",
        "input_dir": BASE / "balanced_full_bbb_resnet50_c200_auto",
    },
    "bbb_vgg19_bn": {
        "layer": "block5",
        "input_dir": BASE / "balanced_full_bbb_vgg19_bn_c200_final_step1",
    },
    "bbb_densenet": {
        "layer": "denseblock3",
        "input_dir": BASE / "balanced_full_bbb_densenet_c200_final_step1",
    },
    "bbb_inception_v3": {
        "layer": "mixed6",
        "input_dir": BASE / "balanced_full_bbb_inception_v3_c200_final_step1",
    },
}


def patch_autoattack_fab_import() -> None:
    gradcheck = importlib.import_module("torch.autograd.gradcheck")

    def zero_gradients(x):
        if isinstance(x, torch.Tensor):
            if x.grad is not None:
                x.grad.detach_()
                x.grad.zero_()
        elif isinstance(x, (list, tuple)):
            for item in x:
                zero_gradients(item)

    gradcheck.zero_gradients = zero_gradients


def safe_name(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace(",", "_")


def parse_models(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_layer_overrides(s: str) -> dict[str, str]:
    out = {}
    for item in [x.strip() for x in s.split(",") if x.strip()]:
        if ":" not in item:
            raise ValueError(f"Layer override must be model:layer, got {item!r}")
        model, layer = item.split(":", 1)
        out[model.strip()] = layer.strip()
    return out


def normalize_l2(v: torch.Tensor) -> torch.Tensor:
    return v / v.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)


def orthogonalize(v: torch.Tensor, basis: list[torch.Tensor]) -> torch.Tensor:
    out = v
    for b in basis:
        coeff = (out.flatten(1) * b.flatten(1)).sum(dim=1).view(-1, 1, 1, 1)
        out = out - coeff * b
    return out


def feature_fn(wrapper, layer: str):
    def f(inp):
        _logits, feats, _raw = wrapper.forward_with_features(inp)
        if layer not in feats:
            raise RuntimeError(f"layer {layer} not captured; available={list(feats)}")
        return feats[layer]

    return f


def estimate_topk(wrapper, x: torch.Tensor, layer: str, k: int, power_iters: int, seed: int):
    f = feature_fn(wrapper, layer)
    dirs: list[torch.Tensor] = []
    sigmas: list[float] = []
    gen = torch.Generator(device=x.device).manual_seed(seed)
    for _rank in range(k):
        v = torch.randn(x.shape, generator=gen, device=x.device)
        v = normalize_l2(orthogonalize(v, dirs))
        for _ in range(power_iters):
            x_req = x.detach().requires_grad_(True)
            _h, jv = torch.autograd.functional.jvp(f, x_req, v, create_graph=False, strict=False)
            h = f(x_req)
            dot = (h * jv.detach()).sum()
            w = torch.autograd.grad(dot, x_req)[0]
            v = normalize_l2(orthogonalize(w.detach(), dirs))
        with torch.no_grad():
            _h, jv = torch.autograd.functional.jvp(f, x.detach(), v, create_graph=False, strict=False)
            sigma = float(jv.flatten(1).norm(dim=1).item())
        dirs.append(v.detach())
        sigmas.append(sigma)
    return dirs, sigmas


def eval_state(wrapper, x: torch.Tensor, y: torch.Tensor):
    with torch.no_grad():
        logits = wrapper(x)
        pred = int(logits.argmax(1).item())
        m = float(margin(logits, y).item())
    return pred, m


def road_line_search(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    *,
    layer: str,
    eps: float,
    steps: int,
    k: int,
    power_iters: int,
    recompute_every: int,
    eta_grid: list[float],
    seed: int,
):
    x = x0.detach().clone()
    dirs: list[torch.Tensor] = []
    sigmas: list[float] = []
    recomputes = 0
    step_rows = []
    for t in range(steps):
        pred0, m0 = eval_state(wrapper, x, y)
        if t % recompute_every == 0 or not dirs:
            dirs, sigmas = estimate_topk(wrapper, x, layer, k, power_iters, seed + 1009 * t)
            recomputes += 1
        best = None
        for rank, (v, sigma) in enumerate(zip(dirs, sigmas), start=1):
            signed = v.sign()
            for sign in (1, -1):
                for eta in eta_grid:
                    cand = project_linf(x + sign * eta * signed, x0, eps).detach()
                    pred, m = eval_state(wrapper, cand, y)
                    item = {"x": cand, "rank": rank, "sign": sign, "eta": eta, "sigma": sigma, "pred": pred, "margin": m}
                    if best is None or item["margin"] < best["margin"]:
                        best = item
        assert best is not None
        x = best["x"]
        step_rows.append(
            {
                "step": t,
                "pred_before": pred0,
                "pred_after": int(best["pred"]),
                "margin_before": m0,
                "margin_after": float(best["margin"]),
                "chosen_rank": int(best["rank"]),
                "chosen_sign": int(best["sign"]),
                "chosen_eta_255": float(best["eta"] * 255.0),
                "chosen_sigma": float(best["sigma"]),
                "recomputed": int(t % recompute_every == 0 or t == 0),
            }
        )
    return x, step_rows, recomputes


def road_apgd_eta(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    *,
    layer: str,
    eps: float,
    steps: int,
    k: int,
    power_iters: int,
    recompute_every: int,
    eta_init: float,
    eta_min: float,
    eta_decay: float,
    patience: int,
    seed: int,
):
    x = x0.detach().clone()
    best_x = x.detach().clone()
    _pred, best_margin = eval_state(wrapper, best_x, y)
    eta = eta_init
    no_improve = 0
    dirs: list[torch.Tensor] = []
    sigmas: list[float] = []
    recomputes = 0
    step_rows = []
    for t in range(steps):
        pred0, m0 = eval_state(wrapper, x, y)
        if t % recompute_every == 0 or not dirs:
            dirs, sigmas = estimate_topk(wrapper, x, layer, k, power_iters, seed + 1009 * t)
            recomputes += 1
        best = None
        for rank, (v, sigma) in enumerate(zip(dirs, sigmas), start=1):
            signed = v.sign()
            for sign in (1, -1):
                cand = project_linf(x + sign * eta * signed, x0, eps).detach()
                pred, m = eval_state(wrapper, cand, y)
                item = {"x": cand, "rank": rank, "sign": sign, "sigma": sigma, "pred": pred, "margin": m}
                if best is None or item["margin"] < best["margin"]:
                    best = item
        assert best is not None
        x = best["x"]
        improved = float(best["margin"]) < best_margin - 1e-8
        if improved:
            best_x = x.detach().clone()
            best_margin = float(best["margin"])
            no_improve = 0
        else:
            no_improve += 1
        eta_before = eta
        decayed = 0
        if no_improve >= patience and eta > eta_min + 1e-12:
            eta = max(eta_min, eta * eta_decay)
            x = best_x.detach().clone()
            no_improve = 0
            decayed = 1
        step_rows.append(
            {
                "step": t,
                "pred_before": pred0,
                "pred_after": int(best["pred"]),
                "margin_before": m0,
                "margin_after": float(best["margin"]),
                "best_margin_after": float(best_margin),
                "chosen_rank": int(best["rank"]),
                "chosen_sign": int(best["sign"]),
                "eta_255_before": float(eta_before * 255.0),
                "eta_255_after": float(eta * 255.0),
                "eta_decayed": int(decayed),
                "chosen_sigma": float(best["sigma"]),
                "improved_best": int(improved),
                "recomputed": int(t % recompute_every == 0 or t == 0),
            }
        )
    return best_x, step_rows, recomputes


def pgd_margin(wrapper, x0, y, eps, steps, step_size):
    x = x0.detach().clone()
    for _ in range(steps):
        probe = x.detach().requires_grad_(True)
        logits = wrapper(probe)
        loss = (-margin(logits, y)).sum()
        grad = torch.autograd.grad(loss, probe)[0]
        x = project_linf(x + step_size * grad.sign(), x0, eps).detach()
    return x


def run_apgd(wrapper, x_all, y_all, eps, steps, loss, batch_size, seed, device):
    from autoattack.autopgd_pt import APGDAttack

    attack = APGDAttack(wrapper, n_iter=steps, norm="Linf", n_restarts=1, eps=eps, seed=seed, loss=loss, verbose=False, device=str(device))
    adv = []
    for start in range(0, len(x_all), batch_size):
        out = attack.perturb(x_all[start : start + batch_size], y_all[start : start + batch_size])
        if isinstance(out, tuple):
            out = out[1]
        adv.append(out.detach())
    return torch.cat(adv, dim=0)


def run_fab(wrapper, x_all, y_all, eps, steps, batch_size, seed, device):
    patch_autoattack_fab_import()
    from autoattack.fab_pt import FABAttack

    attack = FABAttack(wrapper, norm="Linf", n_restarts=1, n_iter=steps, eps=eps, seed=seed, targeted=False, verbose=False, device=device)
    adv = []
    for start in range(0, len(x_all), batch_size):
        adv.append(attack.perturb(x_all[start : start + batch_size], y_all[start : start + batch_size]).detach())
    return torch.cat(adv, dim=0)


def load_eval_set(input_dir: Path, model_name: str, dataset_root: str, device, max_images: int):
    outcomes = pd.read_csv(input_dir / "image_outcomes.csv")
    sub = (
        outcomes[(outcomes.model == model_name) & (outcomes.source == "pgd")]
        .drop_duplicates("image_ord")
        .sort_values("image_ord")
        .reset_index(drop=True)
    )
    if max_images > 0:
        sub = sub.head(max_images).copy()
    dataset = datasets.CIFAR10(dataset_root, train=False, download=False, transform=transforms.ToTensor())
    xs, ys, image_ords, dataset_indices = [], [], [], []
    for r in sub.itertuples(index=False):
        x, y0 = dataset[int(r.dataset_idx)]
        if int(y0) != int(r.label):
            raise RuntimeError(f"label mismatch at dataset_idx={r.dataset_idx}")
        xs.append(x)
        ys.append(int(y0))
        image_ords.append(int(r.image_ord))
        dataset_indices.append(int(r.dataset_idx))
    return torch.stack(xs).to(device), torch.tensor(ys, device=device), np.asarray(image_ords), np.asarray(dataset_indices), sub


def rows_from_adv(method, family, model_name, layer, x_all, y_all, adv, image_ords, dataset_indices, wrapper, runtime_s, compute):
    with torch.no_grad():
        clean_logits = wrapper(x_all)
        final_logits = wrapper(adv)
        clean_margin = margin(clean_logits, y_all).detach().cpu().numpy()
        final_margin = margin(final_logits, y_all).detach().cpu().numpy()
        final_pred = final_logits.argmax(1).detach().cpu().numpy()
    labels = y_all.detach().cpu().numpy()
    linf = (adv - x_all).abs().flatten(1).max(1).values.detach().cpu().numpy()
    rows = []
    for i in range(len(labels)):
        rows.append(
            {
                "model": model_name,
                "layer": layer,
                "method": method,
                "family": family,
                "image_ord": int(image_ords[i]),
                "dataset_idx": int(dataset_indices[i]),
                "label": int(labels[i]),
                "final_pred": int(final_pred[i]),
                "success": int(final_pred[i] != labels[i]),
                "clean_margin": float(clean_margin[i]),
                "final_margin": float(final_margin[i]),
                "margin_drop": float(clean_margin[i] - final_margin[i]),
                "linf": float(linf[i]),
                "runtime_s_total": runtime_s,
                **compute,
            }
        )
    return rows


def compute_estimate(family: str, steps: int, recomputes: int = 0, power_iters: int = 0, eta_grid_len: int = 1):
    if family in {"apgd", "pgd"}:
        return {"derivative_equiv_est": float(steps), "candidate_forward_est": 0.0}
    if family == "fab":
        return {"derivative_equiv_est": float(10 * steps), "candidate_forward_est": 0.0}
    if family == "road":
        return {"derivative_equiv_est": float(2 * 2 * power_iters * recomputes), "candidate_forward_est": float(2 * 2 * eta_grid_len * steps)}
    return {"derivative_equiv_est": math.nan, "candidate_forward_est": math.nan}


def summarize(per_image: pd.DataFrame):
    return (
        per_image.groupby(["model", "layer", "method", "family"], dropna=False)
        .agg(
            n=("success", "size"),
            asr=("success", "mean"),
            mean_margin_drop=("margin_drop", "mean"),
            median_margin_drop=("margin_drop", "median"),
            max_linf=("linf", "max"),
            runtime_s_total=("runtime_s_total", "first"),
            derivative_equiv_est=("derivative_equiv_est", "first"),
            candidate_forward_est=("candidate_forward_est", "first"),
        )
        .reset_index()
        .sort_values(["model", "family", "method"])
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="bbb_resnet50,bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/multimodel_road_routing_cifar_c200")
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--eps-linf", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--include-fab", action="store_true")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--layer-overrides", default="", help="Comma-separated model:layer overrides for road-routing layers.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []
    eta_grid = [x / 255.0 for x in [0.25, 0.5, 1.0, 1.5, 2.0]]
    eps = args.eps_linf / 255.0
    layer_overrides = parse_layer_overrides(args.layer_overrides)

    for model_name in parse_models(args.models):
        cfg = dict(MODEL_CFG[model_name])
        if model_name in layer_overrides:
            cfg["layer"] = layer_overrides[model_name]
        layer = cfg["layer"]
        wrapper = load_feature_model(model_name, device).eval()
        x_all, y_all, image_ords, dataset_indices, _sub = load_eval_set(cfg["input_dir"], model_name, args.dataset_root, device, args.max_images)
        print(f"[model] {model_name} n={len(x_all)} layer={layer}", flush=True)

        method_rows = []
        settings = [
            ("road_line_search_s10_p3_r1", "road", 10, 3, 1, "line_search"),
            ("road_apgd_eta_s20_p1_r2", "road", 20, 1, 2, "apgd_eta"),
        ]
        for method, family, steps, power_iters, recompute_every, mode in settings:
            fpath = out / f"per_image_{safe_name(model_name + '_' + method)}.csv"
            spath = out / f"per_step_{safe_name(model_name + '_' + method)}.csv"
            if fpath.exists() and not args.overwrite:
                rows = pd.read_csv(fpath).to_dict("records")
                all_rows.extend(rows)
                print(f"[skip] {model_name} {method}", flush=True)
                continue
            t0 = time.perf_counter()
            rows, step_rows = [], []
            for i in range(len(x_all)):
                x0 = x_all[i : i + 1]
                y = y_all[i : i + 1]
                clean_pred, clean_margin = eval_state(wrapper, x0, y)
                if clean_pred != int(y.item()):
                    raise RuntimeError(f"{model_name} selected image not clean-correct: image_ord={image_ords[i]}")
                if mode == "line_search":
                    adv, sr, recomputes = road_line_search(
                        wrapper,
                        x0,
                        y,
                        layer=layer,
                        eps=eps,
                        steps=steps,
                        k=2,
                        power_iters=power_iters,
                        recompute_every=recompute_every,
                        eta_grid=eta_grid,
                        seed=args.seed + int(image_ords[i]) * 1009,
                    )
                    eta_grid_len = len(eta_grid)
                else:
                    adv, sr, recomputes = road_apgd_eta(
                        wrapper,
                        x0,
                        y,
                        layer=layer,
                        eps=eps,
                        steps=steps,
                        k=2,
                        power_iters=power_iters,
                        recompute_every=recompute_every,
                        eta_init=2.0 / 255.0,
                        eta_min=0.125 / 255.0,
                        eta_decay=0.5,
                        patience=2,
                        seed=args.seed + int(image_ords[i]) * 1009,
                    )
                    eta_grid_len = 1
                pred, final_margin = eval_state(wrapper, adv, y)
                rows.append(
                    {
                        "model": model_name,
                        "layer": layer,
                        "method": method,
                        "family": family,
                        "image_ord": int(image_ords[i]),
                        "dataset_idx": int(dataset_indices[i]),
                        "label": int(y.item()),
                        "final_pred": pred,
                        "success": int(pred != int(y.item())),
                        "clean_margin": clean_margin,
                        "final_margin": final_margin,
                        "margin_drop": clean_margin - final_margin,
                        "linf": float((adv - x0).abs().max().item()),
                        "runtime_s_total": math.nan,
                        **compute_estimate("road", steps, recomputes, power_iters, eta_grid_len),
                    }
                )
                for rr in sr:
                    step_rows.append({"model": model_name, "method": method, "image_ord": int(image_ords[i]), **rr})
                if (i + 1) % 25 == 0:
                    print(f"[{model_name} {method}] {i + 1}/{len(x_all)}", flush=True)
            runtime = time.perf_counter() - t0
            for r in rows:
                r["runtime_s_total"] = runtime
            pd.DataFrame(rows).to_csv(fpath, index=False)
            pd.DataFrame(step_rows).to_csv(spath, index=False)
            all_rows.extend(rows)
            print(f"[done] {model_name} {method} runtime={runtime:.1f}s", flush=True)

        if args.skip_baselines:
            continue
        baseline_specs = [("pgd_margin_s20", "pgd"), ("official_apgd_ce_s20", "apgd"), ("official_apgd_dlr_s20", "apgd")]
        if args.include_fab:
            baseline_specs.append(("official_fab_linf_s100", "fab"))
        for method, family in baseline_specs:
            fpath = out / f"per_image_{safe_name(model_name + '_' + method)}.csv"
            if fpath.exists() and not args.overwrite:
                rows = pd.read_csv(fpath).to_dict("records")
                all_rows.extend(rows)
                print(f"[skip] {model_name} {method}", flush=True)
                continue
            t0 = time.perf_counter()
            if method == "pgd_margin_s20":
                adv_batches = []
                for start in range(0, len(x_all), args.batch_size):
                    adv_batches.append(pgd_margin(wrapper, x_all[start : start + args.batch_size], y_all[start : start + args.batch_size], eps, 20, eps / 4.0))
                adv = torch.cat(adv_batches, dim=0)
                compute = compute_estimate("pgd", 20)
            elif method == "official_apgd_ce_s20":
                adv = run_apgd(wrapper, x_all, y_all, eps, 20, "ce", args.batch_size, args.seed, device)
                compute = compute_estimate("apgd", 20)
            elif method == "official_apgd_dlr_s20":
                adv = run_apgd(wrapper, x_all, y_all, eps, 20, "dlr", args.batch_size, args.seed, device)
                compute = compute_estimate("apgd", 20)
            elif method == "official_fab_linf_s100":
                adv = run_fab(wrapper, x_all, y_all, eps, 100, args.batch_size, args.seed, device)
                compute = compute_estimate("fab", 100)
            else:
                raise ValueError(method)
            runtime = time.perf_counter() - t0
            rows = rows_from_adv(method, family, model_name, layer, x_all, y_all, adv, image_ords, dataset_indices, wrapper, runtime, compute)
            pd.DataFrame(rows).to_csv(fpath, index=False)
            all_rows.extend(rows)
            print(f"[done] {model_name} {method} runtime={runtime:.1f}s", flush=True)

    per = pd.DataFrame(all_rows)
    per.to_csv(out / "multimodel_road_benchmark_per_image.csv", index=False)
    summary = summarize(per)
    summary.to_csv(out / "multimodel_road_benchmark_summary.csv", index=False)

    overlap_rows = []
    for model_name, g in per.groupby("model"):
        piv = g.pivot_table(index="image_ord", columns="method", values="success", aggfunc="first")
        for a in [c for c in piv.columns if c.startswith("road_")]:
            for b in [c for c in piv.columns if c.startswith("official_apgd") or c.startswith("pgd_") or c.startswith("official_fab")]:
                aa = piv[a].to_numpy(dtype=int)
                bb = piv[b].to_numpy(dtype=int)
                overlap_rows.append(
                    {
                        "model": model_name,
                        "A": a,
                        "B": b,
                        "A_only": int(((aa == 1) & (bb == 0)).sum()),
                        "B_only": int(((aa == 0) & (bb == 1)).sum()),
                        "both": int(((aa == 1) & (bb == 1)).sum()),
                        "neither": int(((aa == 0) & (bb == 0)).sum()),
                        "delta_asr_A_minus_B": float((aa - bb).mean()),
                    }
                )
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(out / "multimodel_road_benchmark_overlap.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({"models": parse_models(args.models), "eps_255": args.eps_linf, "include_fab": args.include_fab, "seed": args.seed}, indent=2))
    print(summary.to_string(index=False), flush=True)
    print(overlap.to_string(index=False), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
