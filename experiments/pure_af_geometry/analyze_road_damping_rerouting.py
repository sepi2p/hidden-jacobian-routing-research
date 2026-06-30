#!/usr/bin/env python3
"""Check whether adaptive attacks keep using a damped road or reroute.

For gamma=1 and gamma=0.25, run PGD and a Square-style score attack through the
forward-pass damped model.  Each accepted/local hidden step is decomposed into
the original road basis U and its orthogonal complement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.evaluate_road_damping_defense import (  # noqa: E402
    DampedResNet50,
    RoadDamping,
    clean_correct_rows,
    load_base_model,
    margin,
    orthonormal_rows,
    project_linf,
    set_seed,
)


def hidden(model: DampedResNet50, x: torch.Tensor) -> torch.Tensor:
    h = model.pooled_layer4(x)
    if model.damping is not None:
        h = model.damping(h)
    return h


def decompose_step(dh: torch.Tensor, u: torch.Tensor) -> tuple[float, float, float]:
    coeff = dh @ u
    proj = coeff @ u.T
    total = float(dh.norm(dim=1).item())
    road = float(proj.norm(dim=1).item())
    orth = float((dh - proj).norm(dim=1).item())
    frac = road * road / max(total * total, 1e-12)
    return frac, road, orth


def final_success(model: DampedResNet50, x: torch.Tensor, y: torch.Tensor) -> tuple[int, float]:
    with torch.no_grad():
        logits = model(x)
    return int(logits.argmax(1).item() != int(y.item())), float(margin(logits, y).item())


def run_pgd(model: DampedResNet50, dataset, rows: pd.DataFrame, u: torch.Tensor, args, device):
    eps = args.eps / 255.0
    alpha = args.pgd_step / 255.0
    step_rows, final_vecs = [], []
    for row in rows.itertuples(index=False):
        torch.manual_seed(args.seed + int(row.dataset_idx) * 1009 + args.pgd_steps)
        x0, _ = dataset[int(row.dataset_idx)]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
        h_prev = hidden(model, x).detach()
        success_step = -1
        for t in range(args.pgd_steps):
            x.requires_grad_(True)
            loss = F.cross_entropy(model(x), y)
            grad = torch.autograd.grad(loss, x)[0]
            x_next = project_linf(x.detach() + alpha * grad.detach().sign(), x0, eps)
            h_next = hidden(model, x_next).detach()
            dh = h_next - h_prev
            frac, road, orth = decompose_step(dh, u)
            succ, cur_margin = final_success(model, x_next, y)
            if succ and success_step < 0:
                success_step = t + 1
            step_rows.append(
                {
                    "attack": "pgd",
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "step": t + 1,
                    "accepted": 1,
                    "current_success": succ,
                    "current_margin": cur_margin,
                    "road_energy_frac": frac,
                    "road_norm": road,
                    "orth_norm": orth,
                    "step_norm": float(dh.norm(dim=1).item()),
                }
            )
            x = x_next
            h_prev = h_next
        succ, final_margin = final_success(model, x, y)
        final_vecs.append((hidden(model, x).detach() - hidden(model, x0).detach()).cpu().numpy()[0])
        step_rows[-1]["final_success"] = succ
        step_rows[-1]["final_margin"] = final_margin
        step_rows[-1]["first_success_step"] = success_step
    return step_rows, np.asarray(final_vecs, dtype=np.float32)


def run_square(model: DampedResNet50, dataset, rows: pd.DataFrame, u: torch.Tensor, args, device):
    eps = args.eps / 255.0
    rng = np.random.default_rng(args.seed + 91)
    step_rows, final_vecs = [], []
    for row in rows.itertuples(index=False):
        x0, _ = dataset[int(row.dataset_idx)]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
        with torch.no_grad():
            best_loss = float(F.cross_entropy(model(x), y).item())
        h_prev = hidden(model, x).detach()
        success_step = -1
        accepted_count = 0
        for q in range(args.square_queries):
            size = max(1, int(round(32 * (1 - q / max(args.square_queries, 1)) ** 0.5)))
            i = int(rng.integers(0, 33 - size))
            j = int(rng.integers(0, 33 - size))
            cand = x.clone()
            sign = -1.0 if rng.random() < 0.5 else 1.0
            cand[:, :, i : i + size, j : j + size] = x0[:, :, i : i + size, j : j + size] + sign * eps
            cand = project_linf(cand, x0, eps)
            with torch.no_grad():
                logits = model(cand)
                loss = float(F.cross_entropy(logits, y).item())
                succ = int(logits.argmax(1).item() != int(row.label))
                cur_margin = float(margin(logits, y).item())
            accepted = int(loss > best_loss or succ)
            if accepted:
                h_next = hidden(model, cand).detach()
                dh = h_next - h_prev
                frac, road, orth = decompose_step(dh, u)
                accepted_count += 1
                step_rows.append(
                    {
                        "attack": "square",
                        "dataset_idx": int(row.dataset_idx),
                        "label": int(row.label),
                        "step": q + 1,
                        "accepted": 1,
                        "current_success": succ,
                        "current_margin": cur_margin,
                        "road_energy_frac": frac,
                        "road_norm": road,
                        "orth_norm": orth,
                        "step_norm": float(dh.norm(dim=1).item()),
                    }
                )
                x = cand
                h_prev = h_next
                best_loss = max(best_loss, loss)
            if succ:
                success_step = q + 1
                break
        succ, final_margin = final_success(model, x, y)
        final_vecs.append((hidden(model, x).detach() - hidden(model, x0).detach()).cpu().numpy()[0])
        if accepted_count == 0:
            step_rows.append(
                {
                    "attack": "square",
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "step": args.square_queries,
                    "accepted": 0,
                    "current_success": succ,
                    "current_margin": final_margin,
                    "road_energy_frac": np.nan,
                    "road_norm": 0.0,
                    "orth_norm": 0.0,
                    "step_norm": 0.0,
                }
            )
        step_rows[-1]["final_success"] = succ
        step_rows[-1]["final_margin"] = final_margin
        step_rows[-1]["first_success_step"] = success_step
    return step_rows, np.asarray(final_vecs, dtype=np.float32)


def subspace_overlap(x: np.ndarray, basis_rows: np.ndarray, k: int) -> dict:
    x = x[np.linalg.norm(x, axis=1) > 1e-9]
    if len(x) < k + 1:
        return {"post_pca_overlap": np.nan, "post_pca_mean_angle_deg": np.nan, "n_final_vecs": int(len(x))}
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(x)
    a = orthonormal_rows(pca.components_, k)
    b = orthonormal_rows(basis_rows, k)
    s = np.linalg.svd(a @ b.T, compute_uv=False)
    s = np.clip(s, 0, 1)
    return {
        "post_pca_overlap": float(np.mean(s**2)),
        "post_pca_mean_angle_deg": float(np.degrees(np.arccos(s)).mean()),
        "n_final_vecs": int(len(x)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/road_damping_rerouting_gamma1_gamma025")
    p.add_argument("--basis-npz", default="analysis_outputs/pure_af_geometry/road_damping_defense_resnet50_pilot/road_damping_bases.npz")
    p.add_argument("--basis-name", default="adv_road")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--images", type=int, default=80)
    p.add_argument("--skip-clean", type=int, default=80)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--gammas", default="1.0,0.25")
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--pgd-steps", type=int, default=100)
    p.add_argument("--pgd-step", type=float, default=2.0)
    p.add_argument("--square-queries", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    base_seq = load_base_model(device)
    basis_pack = np.load(args.basis_npz)
    center = basis_pack["center"].astype(np.float32)
    basis = orthonormal_rows(basis_pack[args.basis_name], args.k)
    u = torch.as_tensor(basis.T, dtype=torch.float32, device=device)
    original = DampedResNet50(base_seq, None).to(device).eval()
    rows = clean_correct_rows(original, dataset, args.skip_clean + args.images + 10, device)
    eval_rows = rows.iloc[args.skip_clean : args.skip_clean + args.images].reset_index(drop=True)

    all_steps = []
    summaries = []
    for gamma in [float(x) for x in args.gammas.split(",") if x.strip()]:
        damping = None if gamma == 1.0 else RoadDamping(basis, center, gamma).to(device)
        model = DampedResNet50(base_seq, damping).to(device).eval()
        for attack in ["pgd", "square"]:
            if attack == "pgd":
                steps, final_vecs = run_pgd(model, dataset, eval_rows, u, args, device)
            else:
                steps, final_vecs = run_square(model, dataset, eval_rows, u, args, device)
            for r in steps:
                r["gamma"] = gamma
                r["basis"] = args.basis_name
            all_steps.extend(steps)
            sdf = pd.DataFrame(steps)
            acc = sdf.groupby("dataset_idx").tail(1)
            summary = {
                "gamma": gamma,
                "attack": attack,
                "basis": args.basis_name,
                "n_images": int(len(eval_rows)),
                "asr": float(acc.final_success.astype(int).mean()),
                "mean_first_success_step": float(acc.first_success_step.replace(-1, np.nan).mean()),
                "mean_road_energy_frac": float(sdf.road_energy_frac.mean()),
                "median_road_energy_frac": float(sdf.road_energy_frac.median()),
                "mean_road_norm": float(sdf.road_norm.mean()),
                "mean_orth_norm": float(sdf.orth_norm.mean()),
                "accepted_steps": int(sdf.accepted.astype(int).sum()),
            }
            summary.update(subspace_overlap(final_vecs, basis, args.k))
            summaries.append(summary)
            pd.DataFrame(all_steps).to_csv(out / "rerouting_step_decomposition.partial.csv", index=False)
            pd.DataFrame(summaries).to_csv(out / "rerouting_summary.partial.csv", index=False)
            print(pd.DataFrame([summary]).to_string(index=False), flush=True)

    pd.DataFrame(all_steps).to_csv(out / "rerouting_step_decomposition.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "rerouting_summary.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
