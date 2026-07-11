#!/usr/bin/env python3
"""Pilot realized JVP gain for actual clean-start candidates.

The exact clean-start comparator stored ``sigma * alpha`` as a JVP proxy.
For the actual L_inf candidate, the candidate input step is
``x_candidate - x0`` after sign conversion, projection, and clipping.  This
script recomputes the same candidate pool for a slice of images and records
``||J_h(x0)(x_candidate - x0)||_2``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.run_exact_ko_cleanstart_comparator import (  # noqa: E402
    estimate_topk_right_singular,
    feature_numpy,
    feature_tensor,
    fit_pgd20_transport_basis,
    input_grads,
    parse_csv,
    projection_energy,
    selected_layers,
    set_seed,
)
from experiments.hidden_jacobian_routing.common import load_model, margin, project_linf  # noqa: E402


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    y = np.asarray(y, dtype=int)[ok]
    s = np.asarray(s, dtype=float)[ok]
    if len(y) < 4 or len(np.unique(y)) < 2 or np.nanstd(s) < 1e-12:
        return np.nan
    return float(roc_auc_score(y, s))


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    y = np.asarray(y, dtype=int)[ok]
    s = np.asarray(s, dtype=float)[ok]
    if len(y) < 4 or len(np.unique(y)) < 2 or np.nanstd(s) < 1e-12:
        return np.nan
    return float(average_precision_score(y, s))


def cosine_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.flatten().float()
    bb = b.flatten().float()
    return float((aa * bb).sum().item() / max(float(aa.norm().item() * bb.norm().item()), 1e-12))


def batched_realized_jvp_gain(wrapper, x0: torch.Tensor, layer: str, deltas: torch.Tensor, chunk: int) -> np.ndarray:
    gains = []

    def f(inp: torch.Tensor) -> torch.Tensor:
        return feature_tensor(wrapper, inp, layer)

    for start in range(0, deltas.shape[0], chunk):
        d = deltas[start : start + chunk].detach()
        xb = x0.detach().repeat(d.shape[0], 1, 1, 1)
        _h, jv = torch.autograd.functional.jvp(f, xb, d, create_graph=False, strict=False)
        gains.append(jv.flatten(1).norm(dim=1).detach().cpu().numpy())
    return np.concatenate(gains, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-csv", default="analysis_outputs/pure_af_geometry/q1_reviewer_validation/exact_protocol/cifar_splits/cifar10_exact_splits.csv")
    p.add_argument("--nested-root", default="analysis_outputs/pure_af_geometry/q1_reviewer_validation/exact_protocol/phase1_nested_layer_selection")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--basis-root", default="analysis_outputs/pure_af_geometry/q1_reviewer_validation/exact_protocol/phase1a_ko_cleanstart_comparator")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/q1_reviewer_validation/exact_protocol/ko_realized_jvp_gain_pilot")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--split-seed", type=int, default=1001)
    p.add_argument("--candidate-seed", type=int, default=0)
    p.add_argument("--layer-rule", default="nested_selected_nonlogit")
    p.add_argument("--max-images", type=int, default=25)
    p.add_argument("--attack-eps", type=float, default=8.0)
    p.add_argument("--alpha-grid", default="1,2,4,6,8")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--max-k", type=int, default=20)
    p.add_argument("--power-iters", type=int, default=12)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--jvp-chunk", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress-every", type=int, default=5)
    # Needed by imported fit_pgd20_transport_basis.
    p.add_argument("--basis-cache-dir", default="")
    p.add_argument("--max-basis-images", type=int, default=-1)
    p.add_argument("--overwrite-basis", action="store_true")
    p.add_argument("--pgd20-steps", type=int, default=20)
    p.add_argument("--pgd20-step-size", type=float, default=2.0)
    p.add_argument("--progress-every-basis", type=int, default=50)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    basis_cache_dir = args.basis_cache_dir or str(Path(args.basis_root) / args.model / f"split_seed_{args.split_seed}" / "shared_transport_bases")
    basis_args = argparse.Namespace(
        output_dir=str(Path(args.basis_root) / args.model / f"split_seed_{args.split_seed}" / f"candidate_seed_{args.candidate_seed}"),
        basis_cache_dir=basis_cache_dir,
        max_basis_images=args.max_basis_images,
        overwrite_basis=args.overwrite_basis,
        attack_eps=args.attack_eps,
        pgd20_steps=args.pgd20_steps,
        pgd20_step_size=args.pgd20_step_size,
        max_k=args.max_k,
        progress_every=args.progress_every_basis,
        model=args.model,
        split_seed=args.split_seed,
    )

    set_seed(args.candidate_seed + args.split_seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    wrapper = load_model(args.model, device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    splits = pd.read_csv(args.splits_csv)
    split_df = splits[(splits.model == args.model) & (splits.split_seed == args.split_seed)].copy()
    layers = selected_layers(Path(args.nested_root), args.model, args.split_seed)
    layer_map = dict(layers)
    if args.layer_rule not in layer_map:
        raise RuntimeError(f"Layer rule {args.layer_rule} not available for {args.model} split {args.split_seed}: {layers}")
    layer = layer_map[args.layer_rule]
    mean, basis, _ratio, n_basis_vectors = fit_pgd20_transport_basis(basis_args, wrapper, dataset, split_df, layer, device)

    final_df = split_df[split_df.split == "final_test"].sort_values(["label", "class_ord"]).copy()
    if args.max_images > 0:
        final_df = final_df.head(args.max_images)
    eps = args.attack_eps / 255.0
    alpha_values = parse_csv(args.alpha_grid, float)
    alphas = [a / 255.0 for a in alpha_values]
    rows = []
    for n_done, r in enumerate(final_df.itertuples(index=False), start=1):
        x_cpu, y0 = dataset[int(r.dataset_idx)]
        label = int(r.label)
        if int(y0) != label:
            raise RuntimeError(f"Label mismatch idx={r.dataset_idx}")
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        clean_logits, h0_np = feature_numpy(wrapper, x0, layer)
        clean_margin = float(margin(clean_logits, y).item())
        ce_grad, dlr_grad = input_grads(wrapper, x0, y)
        dirs, sigmas, _hidden_jvps = estimate_topk_right_singular(
            wrapper,
            x0,
            layer,
            args.k,
            args.power_iters,
            args.tol,
            args.candidate_seed + int(r.dataset_idx) * 1009 + args.split_seed,
        )
        candidate_meta = []
        candidate_tensors = []
        for rank, (v, sigma) in enumerate(zip(dirs, sigmas), start=1):
            signed_base = v.sign()
            for sign in [-1, 1]:
                direction = sign * signed_base
                ce_cos = cosine_torch(direction, ce_grad)
                dlr_cos = cosine_torch(direction, dlr_grad)
                for alpha_255, alpha in zip(alpha_values, alphas):
                    cand = project_linf(x0 + alpha * direction, x0, eps).detach()
                    candidate_tensors.append(cand - x0)
                    candidate_meta.append((rank, sign, float(alpha_255), float(sigma), float(sigma * alpha), ce_cos, dlr_cos, cand))
        deltas = torch.cat(candidate_tensors, dim=0)
        realized_gains = batched_realized_jvp_gain(wrapper, x0, layer, deltas, args.jvp_chunk)
        for (rank, sign, alpha_255, sigma, proxy_gain, ce_cos, dlr_cos, cand), realized_gain in zip(candidate_meta, realized_gains):
            logits, hc_np = feature_numpy(wrapper, cand, layer)
            pred = int(logits.argmax(1).item())
            cand_margin = float(margin(logits, y).item())
            disp = (hc_np[0] - h0_np[0]).astype(np.float32)
            pe = float(projection_energy(disp[None, :], mean, basis, args.k)[0])
            delta = (cand - x0).detach()
            rows.append(
                {
                    "model": args.model,
                    "split_seed": args.split_seed,
                    "candidate_seed": args.candidate_seed,
                    "layer_rule": args.layer_rule,
                    "layer": layer,
                    "image_ord": int(r.image_ord),
                    "dataset_idx": int(r.dataset_idx),
                    "label": label,
                    "rank": rank,
                    "sign": sign,
                    "alpha_255": alpha_255,
                    "singular_value": sigma,
                    "jvp_proxy_gain": proxy_gain,
                    "realized_jvp_gain": float(realized_gain),
                    "ce_grad_cos": ce_cos,
                    "dlr_grad_cos": dlr_cos,
                    "clean_margin": clean_margin,
                    "candidate_margin": cand_margin,
                    "margin_drop": clean_margin - cand_margin,
                    "candidate_pred": pred,
                    "candidate_success": int(pred != label),
                    "candidate_mobility": float(np.linalg.norm(disp)),
                    "transport_projection_energy": pe,
                    "realized_linf_255": float(delta.abs().max().item() * 255.0),
                    "realized_l2": float(delta.flatten(1).norm(dim=1).item()),
                    "n_clipped_coords": int(((cand <= 1e-7) | (cand >= 1 - 1e-7)).sum().item()),
                    "transport_basis_vectors": int(n_basis_vectors),
                }
            )
        if n_done % args.progress_every == 0:
            print(f"[realized-jvp {args.model} {layer}] {n_done}/{len(final_df)}", flush=True)

    df = pd.DataFrame(rows)
    out_rows = out / f"realized_jvp_candidates_{args.model}_split{args.split_seed}_cand{args.candidate_seed}_{args.layer_rule}_{layer}_n{len(final_df)}.csv"
    df.to_csv(out_rows, index=False)
    metrics = []
    y = df.candidate_success.to_numpy(int)
    for score in ["singular_value", "jvp_proxy_gain", "realized_jvp_gain", "candidate_mobility", "margin_drop", "transport_projection_energy"]:
        metrics.append(
            {
                "model": args.model,
                "split_seed": args.split_seed,
                "candidate_seed": args.candidate_seed,
                "layer_rule": args.layer_rule,
                "layer": layer,
                "n_images": int(df.dataset_idx.nunique()),
                "n_candidates": int(len(df)),
                "n_positive": int(y.sum()),
                "positive_rate": float(y.mean()),
                "score": score,
                "auroc": safe_auroc(y, df[score].to_numpy(float)),
                "auprc": safe_auprc(y, df[score].to_numpy(float)),
            }
        )
    metrics_df = pd.DataFrame(metrics)
    out_metrics = out / f"realized_jvp_metrics_{args.model}_split{args.split_seed}_cand{args.candidate_seed}_{args.layer_rule}_{layer}_n{len(final_df)}.csv"
    metrics_df.to_csv(out_metrics, index=False)
    metadata = {
        "script": "run_ko_realized_jvp_gain_pilot.py",
        "purpose": "pilot actual candidate JVP gain after sign/projection/clipping",
        "rows": str(out_rows),
        "metrics": str(out_metrics),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
