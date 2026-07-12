#!/usr/bin/env python3
"""Mechanism-adaptive attacks that suppress or avoid measured mobility.

Variants share the same L_inf budget, random starts, step count, and margin
objective. They either penalize clean-state JVP motion or constrain the total
input perturbation to be orthogonal to pullbacks of transport PCs. A matched
random input subspace controls for the generic effect of removing dimensions.
Per-image results are append-only and resumable.
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

from experiments.hidden_jacobian_routing.common import load_model, margin, project_linf, set_seed  # noqa: E402
from experiments.hidden_jacobian_routing.run_exact_ko_cleanstart_comparator import feature_numpy, feature_tensor  # noqa: E402


def orthonormal_rows(vectors: list[torch.Tensor]) -> torch.Tensor:
    rows = []
    for vector in vectors:
        v = vector.flatten().float()
        for basis in rows:
            v = v - torch.dot(v, basis) * basis
        norm = v.norm()
        if float(norm.item()) > 1e-10:
            rows.append(v / norm)
    if not rows:
        raise RuntimeError("empty pullback basis")
    return torch.stack(rows)


def transport_pullback_basis(wrapper, x0: torch.Tensor, layer: str, feature_basis: np.ndarray, k: int) -> torch.Tensor:
    x_req = x0.detach().requires_grad_(True)
    h = feature_tensor(wrapper, x_req, layer)
    vectors = []
    for row in feature_basis[:k]:
        u = torch.from_numpy(row).to(x_req.device, dtype=h.dtype).view_as(h)
        vectors.append(torch.autograd.grad((h * u).sum(), x_req, retain_graph=True)[0].detach())
    return orthonormal_rows(vectors)


def random_matched_basis(x0: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=x0.device).manual_seed(seed)
    return orthonormal_rows([torch.randn(x0.shape, generator=gen, device=x0.device) for _ in range(k)])


def project_away(delta: torch.Tensor, basis: torch.Tensor, x0: torch.Tensor, eps: float, rounds: int = 5) -> torch.Tensor:
    shape = delta.shape
    d = delta.flatten(1)
    for _ in range(rounds):
        d = d - (d @ basis.T) @ basis
        d = d.clamp(-eps, eps)
        d = (x0.flatten(1) + d).clamp(0.0, 1.0) - x0.flatten(1)
    return d.view(shape)


def local_jvp(wrapper, x0: torch.Tensor, delta: torch.Tensor, layer: str, create_graph: bool) -> torch.Tensor:
    def f(inp: torch.Tensor) -> torch.Tensor:
        return feature_tensor(wrapper, inp, layer)

    _h, jv = torch.autograd.functional.jvp(f, x0.detach(), delta, create_graph=create_graph, strict=False)
    return jv


def attack(
    wrapper,
    x0: torch.Tensor,
    y: torch.Tensor,
    layer: str,
    method: str,
    eps: float,
    step_size: float,
    steps: int,
    restart_seed: int,
    subspace: torch.Tensor | None = None,
    penalty_lambda: float = 0.0,
) -> torch.Tensor:
    gen = torch.Generator(device=x0.device).manual_seed(restart_seed)
    delta = torch.empty_like(x0).uniform_(-eps, eps, generator=gen)
    if subspace is not None:
        delta = project_away(delta, subspace, x0, eps)
    x = project_linf(x0 + delta, x0, eps).detach()
    for _ in range(steps):
        x_req = x.detach().requires_grad_(True)
        logits = wrapper(x_req)
        objective = -margin(logits, y)
        if method == "jvp_penalty":
            delta_req = x_req - x0
            jv = local_jvp(wrapper, x0, delta_req, layer, create_graph=True)
            rms_gain = jv.flatten(1).norm(dim=1).mean() / np.sqrt(jv[0].numel())
            objective = objective - penalty_lambda * rms_gain
        grad = torch.autograd.grad(objective, x_req)[0]
        x = project_linf(x + step_size * grad.sign(), x0, eps).detach()
        if subspace is not None:
            delta = project_away(x - x0, subspace, x0, eps)
            x = (x0 + delta).clamp(0.0, 1.0).detach()
    return x


def evaluate(wrapper, x0, adv, y, label, layer, mean, basis, transport_input_basis, constrained_basis):
    logits, h = feature_numpy(wrapper, adv, layer)
    _clean_logits, h0 = feature_numpy(wrapper, x0, layer)
    disp = (h[0] - h0[0]).astype(np.float32)
    centered = disp - mean.reshape(-1)
    coeff = centered @ basis.T
    energy = float(np.sum(coeff * coeff) / max(float(np.sum(centered * centered)), 1e-12))
    delta = (adv - x0).detach()
    jv = local_jvp(wrapper, x0, delta, layer, create_graph=False)
    flat = delta.flatten(1)
    transport_proj_ratio = float(((flat @ transport_input_basis.T).norm() / flat.norm().clamp_min(1e-12)).item())
    constrained_proj_ratio = float(((flat @ constrained_basis.T).norm() / flat.norm().clamp_min(1e-12)).item())
    return {
        "success": int(int(logits.argmax(1).item()) != label),
        "final_margin": float(margin(logits, y).item()),
        "hidden_mobility": float(np.linalg.norm(disp)),
        "transport_projection_energy": energy,
        "clean_state_jvp_gain": float(jv.flatten(1).norm(dim=1).item()),
        "transport_input_projection_ratio": transport_proj_ratio,
        "constrained_input_projection_ratio": constrained_proj_ratio,
        "linf_255": float(delta.abs().max().item() * 255.0),
        "l2": float(delta.flatten(1).norm(dim=1).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bbb_resnet50")
    parser.add_argument("--split-seed", type=int, default=1001)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits-csv", default="artifacts/splits/cifar10_exact_splits.csv")
    parser.add_argument("--nested-root", default="analysis_outputs/hidden_jacobian_routing/exact_protocol/phase1_nested_layer_selection")
    parser.add_argument("--ko-root", default="analysis_outputs/hidden_jacobian_routing/exact_protocol/phase1a_ko_cleanstart_comparator")
    parser.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--eps", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=2.0)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--penalty-lambdas", default="0.1,1,10")
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
        if args.max_images % 10 != 0:
            raise ValueError("--max-images must be divisible by 10 for class-balanced selection")
        final = final.groupby("label", group_keys=False).head(args.max_images // 10)
    class_counts = final.label.value_counts().sort_index()
    if list(class_counts.index.astype(int)) != list(range(10)) or class_counts.nunique() != 1:
        raise RuntimeError(f"class-balanced final-test selection failed: {class_counts.to_dict()}")
    nested = Path(args.nested_root) / args.model / f"split_seed_{args.split_seed}"
    selected = pd.read_csv(nested / "nested_layer_selection_summary.csv")
    layer = str(selected[selected.layer_rule == "nested_selected_nonlogit"].reported_layer.iloc[0])
    basis_path = Path(args.ko_root) / args.model / f"split_seed_{args.split_seed}" / "shared_transport_bases" / f"transport_basis_{layer}.npz"
    pack = np.load(basis_path)
    feature_mean = pack["mean"].astype(np.float32)
    feature_basis = pack["basis"][: args.k].astype(np.float32)
    rows_path = out / "mechanism_breaking_rows.csv"
    existing = pd.read_csv(rows_path) if rows_path.exists() else pd.DataFrame()
    completed = set(existing.dataset_idx.astype(int)) if len(existing) else set()
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    lambdas = [float(value) for value in args.penalty_lambdas.split(",")]

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
        pullback = transport_pullback_basis(wrapper, x0, layer, feature_basis, args.k)
        random_basis = random_matched_basis(x0, pullback.shape[0], args.seed + idx * 7919)
        methods = [("baseline_margin_pgd", None, 0.0), ("transport_avoid", pullback, 0.0), ("random_avoid", random_basis, 0.0)]
        methods += [(f"jvp_penalty_lambda_{value:g}", None, value) for value in lambdas]
        image_rows = []
        for method_name, subspace, penalty_lambda in methods:
            base_method = "jvp_penalty" if method_name.startswith("jvp_penalty") else method_name
            for restart in range(args.restarts):
                adv = attack(
                    wrapper,
                    x0,
                    y,
                    layer,
                    base_method,
                    eps,
                    step_size,
                    args.steps,
                    args.seed + idx * 1009 + restart,
                    subspace=subspace,
                    penalty_lambda=penalty_lambda,
                )
                measured_basis = pullback if subspace is None else subspace
                metrics = evaluate(
                    wrapper, x0, adv, y, label, layer, feature_mean, feature_basis, pullback, measured_basis
                )
                image_rows.append(
                    {
                        "model": args.model,
                        "split_seed": args.split_seed,
                        "layer": layer,
                        "dataset_idx": idx,
                        "label": label,
                        "method": method_name,
                        "restart": restart,
                        "penalty_lambda": penalty_lambda,
                        **metrics,
                    }
                )
        existing = pd.concat([existing, pd.DataFrame(image_rows)], ignore_index=True)
        existing.to_csv(rows_path, index=False)
        completed.add(idx)
        print(f"[mechanism-break {args.model} {layer}] {len(completed)}/{len(final)}", flush=True)

    best = existing.sort_values("final_margin").groupby(["dataset_idx", "method"], as_index=False).first()
    summary = (
        best.groupby("method")
        .agg(
            n_images=("dataset_idx", "nunique"),
            asr=("success", "mean"),
            median_final_margin=("final_margin", "median"),
            median_hidden_mobility=("hidden_mobility", "median"),
            median_transport_energy=("transport_projection_energy", "median"),
            median_clean_jvp_gain=("clean_state_jvp_gain", "median"),
            median_transport_input_ratio=("transport_input_projection_ratio", "median"),
            median_constrained_input_ratio=("constrained_input_projection_ratio", "median"),
            max_linf_255=("linf_255", "max"),
        )
        .reset_index()
    )
    best.to_csv(out / "mechanism_breaking_best_per_image.csv", index=False)
    summary.to_csv(out / "mechanism_breaking_summary.csv", index=False)
    done.write_text(
        json.dumps(
            {
                "status": "complete",
                "model": args.model,
                "layer": layer,
                "images": len(completed),
                "class_counts": {str(key): int(value) for key, value in class_counts.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
