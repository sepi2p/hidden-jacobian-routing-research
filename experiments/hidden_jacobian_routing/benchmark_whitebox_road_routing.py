#!/usr/bin/env python3
"""Compute-normalized white-box benchmark for hidden-Jacobian road routing.

The goal is not to declare a new state-of-the-art attack from a single CIFAR-10
setting.  The goal is to compare the k=2 hidden-Jacobian road-routing attack
against strong white-box baselines under the same image set, perturbation
budget, and a transparent estimate of derivative cost.

This script is resumable at the method/setting level: each completed setting is
written to ``per_image_<setting>.csv`` and skipped on subsequent runs unless
``--overwrite`` is passed.
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
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import margin, project_linf  # noqa: E402
from experiments.hidden_jacobian_routing.trace_jacobian_singular_roads import set_seed  # noqa: E402
from utils.load_models import load_cifar_model  # noqa: E402


def patch_autoattack_fab_import() -> None:
    """Restore the zero_gradients symbol expected by older AutoAttack FAB."""

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
    return s.replace("/", "_").replace("=", "").replace(":", "_").replace(",", "_")


def parse_settings(s: str) -> list[dict[str, str | int]]:
    out: list[dict[str, str | int]] = []
    if not s.strip():
        return out
    for raw in s.split(","):
        parts = [p.strip() for p in raw.split(":")]
        if len(parts) == 3:
            loss, steps, restarts = parts
            out.append({"loss": loss, "steps": int(steps), "restarts": int(restarts)})
        elif len(parts) == 2:
            loss, steps = parts
            out.append({"loss": loss, "steps": int(steps), "restarts": 1})
        else:
            raise ValueError(f"Bad setting {raw!r}; expected loss:steps[:restarts]")
    return out


def load_balanced_images(args, device):
    bdir = Path(args.balanced_dir)
    meta = json.loads((bdir / "metadata.json").read_text())
    image_outcomes = pd.read_csv(bdir / "image_outcomes.csv")
    pgd_ref = (
        image_outcomes[image_outcomes["source"] == "pgd"]
        .drop_duplicates("image_ord")
        .sort_values("image_ord")
        .reset_index(drop=True)
    )
    if args.max_images > 0:
        pgd_ref = pgd_ref.iloc[: args.max_images].copy()

    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    xs, ys, image_ords, dataset_indices = [], [], [], []
    for r in pgd_ref.itertuples(index=False):
        x, y = dataset[int(r.dataset_idx)]
        if int(y) != int(r.label):
            raise RuntimeError(f"label mismatch for dataset_idx={r.dataset_idx}")
        xs.append(x)
        ys.append(int(y))
        image_ords.append(int(r.image_ord))
        dataset_indices.append(int(r.dataset_idx))
    return (
        torch.stack(xs).to(device),
        torch.tensor(ys, dtype=torch.long, device=device),
        np.asarray(image_ords, dtype=int),
        np.asarray(dataset_indices, dtype=int),
        meta,
    )


def logits_stats(model, x, y):
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1)
        m = margin(logits, y)
        ce = F.cross_entropy(logits, y, reduction="none")
    return pred.detach().cpu().numpy(), m.detach().cpu().numpy(), ce.detach().cpu().numpy()


def dlr_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_sorted, ind_sorted = logits.sort(dim=1)
    true = logits.gather(1, y[:, None]).squeeze(1)
    top1 = x_sorted[:, -1]
    top2 = x_sorted[:, -2]
    top3 = x_sorted[:, -3]
    top_other = torch.where(ind_sorted[:, -1] == y, top2, top1)
    return -(true - top_other) / (top1 - top3 + 1e-12)


def attack_loss(logits: torch.Tensor, y: torch.Tensor, loss: str) -> torch.Tensor:
    if loss == "ce":
        return F.cross_entropy(logits, y, reduction="none")
    if loss == "margin":
        return -margin(logits, y)
    if loss == "dlr":
        return dlr_loss(logits, y)
    raise ValueError(f"unknown PGD loss: {loss}")


def pgd_attack(
    model,
    x0: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    restarts: int,
    loss_name: str,
    step_rule: str,
    seed: int,
) -> torch.Tensor:
    best_x = x0.detach().clone()
    with torch.no_grad():
        best_margin = margin(model(best_x), y)

    if step_rule == "eps_over_4":
        step_size = eps / 4.0
    elif step_rule == "2eps_over_steps":
        step_size = 2.0 * eps / max(steps, 1)
    elif step_rule == "balanced":
        step_size = eps / 2.0
    else:
        raise ValueError(f"unknown step_rule: {step_rule}")

    gen = torch.Generator(device=x0.device).manual_seed(seed)
    for restart in range(restarts):
        if restart == 0:
            x = x0.detach().clone()
        else:
            x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps, generator=gen), x0, eps).detach()
        for _ in range(steps):
            x_req = x.detach().requires_grad_(True)
            logits = model(x_req)
            loss = attack_loss(logits, y, loss_name).sum()
            grad = torch.autograd.grad(loss, x_req)[0]
            x = project_linf(x + step_size * grad.sign(), x0, eps).detach()
        with torch.no_grad():
            m = margin(model(x), y)
            keep = m < best_margin
            best_margin = torch.where(keep, m, best_margin)
            best_x[keep] = x[keep]
    return best_x.detach()


def run_apgd(model, x_all, y_all, eps, steps, restarts, loss, batch_size, seed, device):
    from autoattack.autopgd_pt import APGDAttack

    attack = APGDAttack(
        model,
        n_iter=steps,
        norm="Linf",
        n_restarts=restarts,
        eps=eps,
        seed=seed,
        loss=loss,
        verbose=False,
        device=str(device),
    )
    adv = []
    for start in range(0, len(x_all), batch_size):
        xb = x_all[start : start + batch_size]
        yb = y_all[start : start + batch_size]
        out = attack.perturb(xb, yb)
        if isinstance(out, tuple):
            out = out[1]
        adv.append(out.detach())
    return torch.cat(adv, dim=0)


def run_fab(model, x_all, y_all, eps, steps, restarts, batch_size, seed, device):
    patch_autoattack_fab_import()
    from autoattack.fab_pt import FABAttack

    attack = FABAttack(
        model,
        norm="Linf",
        n_restarts=restarts,
        n_iter=steps,
        eps=eps,
        seed=seed,
        targeted=False,
        verbose=False,
        device=device,
    )
    adv = []
    for start in range(0, len(x_all), batch_size):
        xb = x_all[start : start + batch_size]
        yb = y_all[start : start + batch_size]
        adv.append(attack.perturb(xb, yb).detach())
    return torch.cat(adv, dim=0)


def evaluate_adv(model, x_clean, y, adv):
    clean_pred, clean_margin, clean_ce = logits_stats(model, x_clean, y)
    pred, final_margin, final_ce = logits_stats(model, adv, y)
    linf = (adv - x_clean).abs().flatten(1).max(1).values.detach().cpu().numpy()
    return clean_pred, clean_margin, clean_ce, pred, final_margin, final_ce, linf


def build_rows(
    method: str,
    family: str,
    loss: str,
    steps: int,
    restarts: int,
    image_ords,
    dataset_indices,
    labels,
    stats,
    runtime_s: float,
    compute: dict[str, float],
):
    clean_pred, clean_margin, clean_ce, pred, final_margin, final_ce, linf = stats
    n = len(labels)
    rows = []
    for i in range(n):
        rows.append(
            {
                "method": method,
                "family": family,
                "loss": loss,
                "steps": steps,
                "restarts": restarts,
                "image_ord": int(image_ords[i]),
                "dataset_idx": int(dataset_indices[i]),
                "label": int(labels[i]),
                "clean_pred": int(clean_pred[i]),
                "clean_margin": float(clean_margin[i]),
                "final_pred": int(pred[i]),
                "success": int(pred[i] != labels[i]),
                "final_margin": float(final_margin[i]),
                "margin_drop": float(clean_margin[i] - final_margin[i]),
                "final_ce": float(final_ce[i]),
                "linf": float(linf[i]),
                "runtime_s_total": runtime_s,
                **compute,
            }
        )
    return rows


def compute_estimate(family: str, steps: int, restarts: int, batch_size: int = 0):
    if family == "pgd":
        return {
            "forward_equiv_est": float(steps * restarts + restarts),
            "backward_equiv_est": float(steps * restarts),
            "derivative_equiv_est": float(steps * restarts),
            "candidate_forward_est": 0.0,
            "compute_note": "PGD: one backward per step; forward count is approximate.",
        }
    if family == "apgd":
        return {
            "forward_equiv_est": float(steps * restarts + restarts),
            "backward_equiv_est": float(steps * restarts),
            "derivative_equiv_est": float(steps * restarts),
            "candidate_forward_est": 0.0,
            "compute_note": "APGD: one gradient-equivalent step per iteration, ignoring line-search overhead.",
        }
    if family == "fab":
        return {
            "forward_equiv_est": float(steps * restarts),
            "backward_equiv_est": float(10 * steps * restarts),
            "derivative_equiv_est": float(10 * steps * restarts),
            "candidate_forward_est": 0.0,
            "compute_note": "FAB estimate uses CIFAR-10 class-wise gradients per iteration.",
        }
    if family == "road":
        # k=2, power_iters=3: for each rank and power step, one JVP plus one VJP/grad.
        return {
            "forward_equiv_est": float(1),
            "backward_equiv_est": float(0),
            "derivative_equiv_est": float(12 * steps),
            "candidate_forward_est": float(4 * steps),
            "compute_note": "Road estimate: k=2, power_iters=3 => 12 derivative ops and 4 candidate forwards per step.",
        }
    return {
        "forward_equiv_est": math.nan,
        "backward_equiv_est": math.nan,
        "derivative_equiv_est": math.nan,
        "candidate_forward_est": math.nan,
        "compute_note": "",
    }


def summarize(per_image: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "family", "loss", "steps", "restarts"]
    return (
        per_image.groupby(group_cols, dropna=False)
        .agg(
            n=("success", "size"),
            asr=("success", "mean"),
            mean_margin_drop=("margin_drop", "mean"),
            median_margin_drop=("margin_drop", "median"),
            max_linf=("linf", "max"),
            runtime_s_total=("runtime_s_total", "first"),
            derivative_equiv_est=("derivative_equiv_est", "first"),
            candidate_forward_est=("candidate_forward_est", "first"),
            compute_note=("compute_note", "first"),
        )
        .reset_index()
        .sort_values(["family", "loss", "steps", "restarts", "method"])
    )


def add_existing_road_rows(args, model, x_all, y_all, image_ords, dataset_indices, labels, out: Path):
    rows = []
    road_sources = [
        (10, "analysis_outputs/hidden_jacobian_routing/k2_layer4_road_apgd_regime_c200_s10/k2_layer_road_per_image.csv"),
        (20, "analysis_outputs/hidden_jacobian_routing/k2_layer4_road_apgd_regime_c200_s20/k2_layer_road_per_image.csv"),
    ]
    _, clean_margin, clean_ce = logits_stats(model, x_all, y_all)
    for steps, path in road_sources:
        p = Path(path)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["layer"] == "layer4"].copy()
        if args.max_images > 0:
            allowed = set(map(int, image_ords))
            df = df[df["image_ord"].isin(allowed)].copy()
        compute = compute_estimate("road", steps, 1)
        method = f"k2_layer4_road_margin_sign_{steps}"
        for _, r in df.iterrows():
            idx = int(np.where(image_ords == int(r.image_ord))[0][0])
            rows.append(
                {
                    "method": method,
                    "family": "road",
                    "loss": "margin_selected",
                    "steps": steps,
                    "restarts": 1,
                    "image_ord": int(r.image_ord),
                    "dataset_idx": int(r.dataset_idx),
                    "label": int(r.label),
                    "clean_pred": int(r.label),
                    "clean_margin": float(clean_margin[idx]),
                    "final_pred": int(r.road_final_pred),
                    "success": int(r.road_success),
                    "final_margin": float(r.road_final_margin),
                    "margin_drop": float(r.road_margin_drop),
                    "final_ce": math.nan,
                    "linf": float(args.eps_from_meta_255) / 255.0,
                    "runtime_s_total": math.nan,
                    **compute,
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(out / "per_image_existing_roads.csv", index=False)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--balanced-dir",
        default="analysis_outputs/hidden_jacobian_routing/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto",
    )
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/whitebox_road_benchmark_resnet50_c200")
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pgd-settings", default="ce:10:1,ce:20:1,ce:100:1,ce:120:1,ce:240:1,margin:10:1,margin:20:1,margin:100:1,margin:120:1,margin:240:1,ce:20:5,margin:20:5")
    p.add_argument("--pgd-step-rules", default="eps_over_4,2eps_over_steps")
    p.add_argument("--apgd-settings", default="ce:10:1,ce:20:1,ce:100:1,ce:120:1,ce:240:1,dlr:10:1,dlr:20:1,dlr:100:1,dlr:120:1,dlr:240:1")
    p.add_argument("--fab-settings", default="fab:10:1,fab:20:1,fab:100:1")
    p.add_argument("--skip-fab", action="store_true")
    p.add_argument("--include-existing-roads", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cifar_model("bbb_resnet50").to(device).eval()
    x_all, y_all, image_ords, dataset_indices, meta = load_balanced_images(args, device)
    eps = float(meta["pgd_eps"]) / 255.0
    args.eps_from_meta_255 = float(meta["pgd_eps"])
    labels = y_all.detach().cpu().numpy()
    clean_pred, _, _ = logits_stats(model, x_all, y_all)
    if not bool((clean_pred == labels).all()):
        raise RuntimeError("selected images are not all clean-correct")

    all_rows = []
    if args.include_existing_roads:
        all_rows.extend(add_existing_road_rows(args, model, x_all, y_all, image_ords, dataset_indices, labels, out))

    # Standard PGD baselines.
    for setting in parse_settings(args.pgd_settings):
        for step_rule in [x.strip() for x in args.pgd_step_rules.split(",") if x.strip()]:
            loss = str(setting["loss"])
            steps = int(setting["steps"])
            restarts = int(setting["restarts"])
            method = f"pgd_{loss}_{step_rule}_s{steps}_r{restarts}"
            fpath = out / f"per_image_{safe_name(method)}.csv"
            if fpath.exists() and not args.overwrite:
                all_rows.extend(pd.read_csv(fpath).to_dict("records"))
                print(f"[skip] {method}", flush=True)
                continue
            t0 = time.perf_counter()
            adv = []
            for start in range(0, len(x_all), args.batch_size):
                xb = x_all[start : start + args.batch_size]
                yb = y_all[start : start + args.batch_size]
                adv.append(pgd_attack(model, xb, yb, eps, steps, restarts, loss, step_rule, args.seed + start).detach())
            adv_all = torch.cat(adv, dim=0)
            runtime = time.perf_counter() - t0
            rows = build_rows(
                method,
                "pgd",
                loss,
                steps,
                restarts,
                image_ords,
                dataset_indices,
                labels,
                evaluate_adv(model, x_all, y_all, adv_all),
                runtime,
                compute_estimate("pgd", steps, restarts),
            )
            pd.DataFrame(rows).to_csv(fpath, index=False)
            all_rows.extend(rows)
            print(f"[done] {method} runtime={runtime:.1f}s", flush=True)

    # Official AutoAttack APGD components.
    for setting in parse_settings(args.apgd_settings):
        loss = str(setting["loss"])
        steps = int(setting["steps"])
        restarts = int(setting["restarts"])
        method = f"official_apgd_{loss}_s{steps}_r{restarts}"
        fpath = out / f"per_image_{safe_name(method)}.csv"
        if fpath.exists() and not args.overwrite:
            all_rows.extend(pd.read_csv(fpath).to_dict("records"))
            print(f"[skip] {method}", flush=True)
            continue
        t0 = time.perf_counter()
        adv_all = run_apgd(model, x_all, y_all, eps, steps, restarts, loss, args.batch_size, args.seed, device)
        runtime = time.perf_counter() - t0
        rows = build_rows(
            method,
            "apgd",
            loss,
            steps,
            restarts,
            image_ords,
            dataset_indices,
            labels,
            evaluate_adv(model, x_all, y_all, adv_all),
            runtime,
            compute_estimate("apgd", steps, restarts),
        )
        pd.DataFrame(rows).to_csv(fpath, index=False)
        all_rows.extend(rows)
        print(f"[done] {method} runtime={runtime:.1f}s", flush=True)

    # Official AutoAttack FAB component, with a local compatibility import patch.
    if not args.skip_fab:
        for setting in parse_settings(args.fab_settings):
            steps = int(setting["steps"])
            restarts = int(setting["restarts"])
            method = f"official_fab_linf_s{steps}_r{restarts}"
            fpath = out / f"per_image_{safe_name(method)}.csv"
            if fpath.exists() and not args.overwrite:
                all_rows.extend(pd.read_csv(fpath).to_dict("records"))
                print(f"[skip] {method}", flush=True)
                continue
            t0 = time.perf_counter()
            adv_all = run_fab(model, x_all, y_all, eps, steps, restarts, args.batch_size, args.seed, device)
            runtime = time.perf_counter() - t0
            rows = build_rows(
                method,
                "fab",
                "fab",
                steps,
                restarts,
                image_ords,
                dataset_indices,
                labels,
                evaluate_adv(model, x_all, y_all, adv_all),
                runtime,
                compute_estimate("fab", steps, restarts),
            )
            pd.DataFrame(rows).to_csv(fpath, index=False)
            all_rows.extend(rows)
            print(f"[done] {method} runtime={runtime:.1f}s", flush=True)

    per_image = pd.DataFrame(all_rows)
    per_image.to_csv(out / "whitebox_road_benchmark_per_image.csv", index=False)
    summary = summarize(per_image)
    summary.to_csv(out / "whitebox_road_benchmark_summary.csv", index=False)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "balanced_dir": args.balanced_dir,
                "dataset_root": args.dataset_root,
                "model": "bbb_resnet50",
                "n_images": int(len(x_all)),
                "eps_255": float(meta["pgd_eps"]),
                "batch_size": args.batch_size,
                "pgd_settings": args.pgd_settings,
                "pgd_step_rules": args.pgd_step_rules,
                "apgd_settings": args.apgd_settings,
                "fab_settings": args.fab_settings,
                "include_existing_roads": bool(args.include_existing_roads),
                "seed": args.seed,
                "device": str(device),
                "note": "Official APGD/FAB are AutoAttack components. FAB uses a local import compatibility patch for zero_gradients.",
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
