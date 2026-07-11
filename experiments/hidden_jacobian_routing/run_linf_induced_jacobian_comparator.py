#!/usr/bin/env python3
"""Compare sign-converted L2 singular vectors with L_inf-native mobility.

The L_inf-native direction approximates the (infinity,2) induced Jacobian norm
by alternating a normalized hidden-space JVP with the exact L_inf linear
maximizer sign(J^T u). Multiple restarts are retained and the best local gain
is evaluated under the same candidate sign/radius grid as the clean-start
comparator. Collection checkpoints after every image.
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
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import load_model, margin, project_linf  # noqa: E402
from experiments.hidden_jacobian_routing.run_exact_ko_cleanstart_comparator import (  # noqa: E402
    estimate_topk_right_singular,
    feature_numpy,
    feature_tensor,
    input_grads,
    parse_csv,
    set_seed,
)
from experiments.hidden_jacobian_routing.run_ko_realized_jvp_gain_pilot import batched_realized_jvp_gain  # noqa: E402


def linf_induced_direction(wrapper, x: torch.Tensor, layer: str, iters: int, restarts: int, seed: int):
    def f(inp: torch.Tensor) -> torch.Tensor:
        return feature_tensor(wrapper, inp, layer)

    gen = torch.Generator(device=x.device).manual_seed(seed)
    best_direction = None
    best_gain = -float("inf")
    histories = []
    for restart in range(restarts):
        d = torch.empty(x.shape, device=x.device).bernoulli_(0.5, generator=gen).mul_(2).sub_(1)
        history = []
        for _ in range(iters):
            _h, jv = torch.autograd.functional.jvp(f, x.detach(), d, create_graph=False, strict=False)
            gain = float(jv.flatten(1).norm(dim=1).item())
            history.append(gain)
            u = jv.detach() / jv.detach().flatten(1).norm(dim=1).view(-1, 1).clamp_min(1e-12)
            x_req = x.detach().requires_grad_(True)
            h = f(x_req)
            w = torch.autograd.grad((h * u).sum(), x_req)[0].detach()
            new_d = w.sign()
            if torch.equal(new_d, d):
                d = new_d
                break
            d = new_d
        _h, jv = torch.autograd.functional.jvp(f, x.detach(), d, create_graph=False, strict=False)
        gain = float(jv.flatten(1).norm(dim=1).item())
        histories.append(history + [gain])
        if gain > best_gain:
            best_gain = gain
            best_direction = d.detach().clone()
    assert best_direction is not None
    return best_direction, best_gain, histories


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    per_image = []
    for (model, split_seed, method, dataset_idx), group in df.groupby(["model", "split_seed", "method", "dataset_idx"]):
        best = group.loc[group.margin_drop.idxmax()]
        per_image.append(
            {
                "model": model,
                "split_seed": split_seed,
                "method": method,
                "dataset_idx": dataset_idx,
                "candidate_coverage": int(group.candidate_success.max()),
                "best_margin_drop": float(best.margin_drop),
                "best_candidate_success": int(best.candidate_success),
                "unit_realized_jvp_gain": float(group.unit_realized_jvp_gain.iloc[0]),
                "max_fd_mobility": float(group.candidate_mobility.max()),
            }
        )
    per = pd.DataFrame(per_image)
    summary = (
        per.groupby(["model", "split_seed", "method"])
        .agg(
            n_images=("dataset_idx", "nunique"),
            proposal_coverage=("candidate_coverage", "mean"),
            mean_best_margin_drop=("best_margin_drop", "mean"),
            median_unit_realized_jvp_gain=("unit_realized_jvp_gain", "median"),
            median_max_fd_mobility=("max_fd_mobility", "median"),
        )
        .reset_index()
    )
    return per, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split-seed", type=int, default=1001)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits-csv", default="artifacts/splits/cifar10_exact_splits.csv")
    parser.add_argument("--nested-root", default="analysis_outputs/hidden_jacobian_routing/exact_protocol/phase1_nested_layer_selection")
    parser.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--attack-eps", type=float, default=8.0)
    parser.add_argument("--alpha-grid", default="1,2,4,8")
    parser.add_argument("--l2-power-iters", type=int, default=12)
    parser.add_argument("--linf-iters", type=int, default=12)
    parser.add_argument("--linf-restarts", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "DONE.json"
    if done.exists():
        print(f"[SKIP] {done}")
        return
    set_seed(args.seed + args.split_seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wrapper = load_model(args.model, device)
    wrapper.eval()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    splits = pd.read_csv(args.splits_csv)
    final = splits[(splits.model == args.model) & (splits.split_seed == args.split_seed) & (splits.split == "final_test")]
    final = final.sort_values(["label", "class_ord"])
    if args.max_images > 0:
        final = final.head(args.max_images)
    nested = Path(args.nested_root) / args.model / f"split_seed_{args.split_seed}"
    selected = pd.read_csv(nested / "nested_layer_selection_summary.csv")
    layer = str(selected[selected.layer_rule == "nested_selected_nonlogit"].reported_layer.iloc[0])
    rows_path = out / "linf_comparator_candidates.csv"
    existing = pd.read_csv(rows_path) if rows_path.exists() else pd.DataFrame()
    completed = set(existing.dataset_idx.astype(int)) if len(existing) else set()
    alphas_255 = parse_csv(args.alpha_grid, float)
    eps = args.attack_eps / 255.0

    for n, row in enumerate(final.itertuples(index=False), start=1):
        idx = int(row.dataset_idx)
        if idx in completed:
            continue
        x_cpu, label0 = dataset[idx]
        label = int(row.label)
        if int(label0) != label:
            raise RuntimeError(f"label mismatch for {idx}")
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        logits0, h0 = feature_numpy(wrapper, x0, layer)
        clean_margin = float(margin(logits0, y).item())
        ce_grad, _dlr_grad = input_grads(wrapper, x0, y)
        l2_dirs, l2_sigmas, _ = estimate_topk_right_singular(
            wrapper, x0, layer, 1, args.l2_power_iters, args.tol, args.seed + idx * 1009
        )
        l2_sign = l2_dirs[0].sign()
        linf_dir, linf_gain, histories = linf_induced_direction(
            wrapper, x0, layer, args.linf_iters, args.linf_restarts, args.seed + idx * 9176 + args.split_seed
        )
        gen = torch.Generator(device=device).manual_seed(args.seed + idx * 1291)
        random_sign = torch.empty_like(x0).bernoulli_(0.5, generator=gen).mul_(2).sub_(1)
        directions = {
            "l2_singular_sign": l2_sign,
            "linf_induced": linf_dir,
            "ce_gradient_sign": ce_grad.sign(),
            "random_sign": random_sign,
        }
        image_rows = []
        for method, base in directions.items():
            candidates = []
            meta = []
            for orientation in [-1, 1]:
                for alpha_255 in alphas_255:
                    cand = project_linf(x0 + orientation * (alpha_255 / 255.0) * base, x0, eps).detach()
                    candidates.append(cand - x0)
                    meta.append((orientation, alpha_255, cand))
            deltas = torch.cat(candidates, dim=0)
            realized = batched_realized_jvp_gain(wrapper, x0, layer, deltas, len(deltas))
            unit_gain = float(realized[-1] / max(alphas_255[-1] / 255.0, 1e-12))
            for (orientation, alpha_255, cand), gain in zip(meta, realized):
                logits, hc = feature_numpy(wrapper, cand, layer)
                candidate_margin = float(margin(logits, y).item())
                disp = hc[0] - h0[0]
                image_rows.append(
                    {
                        "model": args.model,
                        "split_seed": args.split_seed,
                        "layer": layer,
                        "dataset_idx": idx,
                        "label": label,
                        "method": method,
                        "orientation": orientation,
                        "alpha_255": alpha_255,
                        "clean_margin": clean_margin,
                        "candidate_margin": candidate_margin,
                        "margin_drop": clean_margin - candidate_margin,
                        "candidate_success": int(int(logits.argmax(1).item()) != label),
                        "candidate_mobility": float(np.linalg.norm(disp)),
                        "realized_jvp_gain": float(gain),
                        "unit_realized_jvp_gain": unit_gain,
                        "l2_singular_value": float(l2_sigmas[0]),
                        "linf_induced_gain": float(linf_gain),
                        "linf_restart_final_min": float(min(history[-1] for history in histories)),
                        "linf_restart_final_max": float(max(history[-1] for history in histories)),
                    }
                )
        existing = pd.concat([existing, pd.DataFrame(image_rows)], ignore_index=True)
        existing.to_csv(rows_path, index=False)
        completed.add(idx)
        print(f"[{args.model} {layer}] {len(completed)}/{len(final)}", flush=True)

    per, summary = summarize(existing)
    per.to_csv(out / "linf_comparator_per_image.csv", index=False)
    summary.to_csv(out / "linf_comparator_summary.csv", index=False)
    done.write_text(json.dumps({"status": "complete", "model": args.model, "split_seed": args.split_seed, "layer": layer, "images": len(completed)}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
