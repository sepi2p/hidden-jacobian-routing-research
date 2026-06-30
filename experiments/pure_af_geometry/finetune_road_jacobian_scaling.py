#!/usr/bin/env python3
"""Fine-tune a model to scale hidden-Jacobian road components.

This is different from the forward road-damping diagnostic.  The objective here
changes model parameters so that, for a fixed road basis U at pooled layer4,

    U^T J_new(x) r ~= gamma U^T J_old(x) r

while preserving clean predictions and the orthogonal hidden-Jacobian component.
It is an exploratory causal diagnostic, not a production defense.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.evaluate_road_damping_defense import (  # noqa: E402
    DampedResNet50,
    eval_clean,
    fit_candidate_bases,
    margin,
    pgd_asr,
    project_linf,
    square_asr,
)
from utils.load_models import load_cifar_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wrapped_model(device: torch.device) -> DampedResNet50:
    model = load_cifar_model("bbb_resnet50").to(device).eval()
    wrapped = DampedResNet50(model, damping=None).to(device).eval()
    return wrapped


def select_correct(dataset, model: nn.Module, n: int, device: torch.device, max_scan: int = 50000) -> pd.DataFrame:
    rows = []
    for idx in range(min(len(dataset), max_scan)):
        x, y0 = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([int(y0)], device=device)
        with torch.no_grad():
            logits = model(x)
        if int(logits.argmax(1).item()) == int(y0):
            rows.append({"dataset_idx": idx, "label": int(y0), "clean_margin": float(margin(logits, y).item())})
        if len(rows) >= n:
            break
    if len(rows) < n:
        print(f"[WARN] requested {n} clean-correct images, found {len(rows)}", flush=True)
    return pd.DataFrame(rows)


def make_loader(dataset, rows: pd.DataFrame, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    idxs = rows.dataset_idx.to_numpy(dtype=int).tolist()
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(Subset(dataset, idxs), batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True, generator=gen)


def freeze_for_mode(model: DampedResNet50, mode: str) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    if mode in {"layer4", "layer4_fc"}:
        for p in model.net.layer4.parameters():
            p.requires_grad_(True)
    if mode in {"fc", "layer4_fc"}:
        for p in model.net.linear.parameters():
            p.requires_grad_(True)
    if mode == "all":
        for p in model.parameters():
            p.requires_grad_(True)


def jvp_hidden(model: DampedResNet50, x: torch.Tensor, r: torch.Tensor, create_graph: bool) -> torch.Tensor:
    def feat(inp: torch.Tensor) -> torch.Tensor:
        return model.pooled_layer4(inp)

    _base, jvp = torch.autograd.functional.jvp(feat, x, r, create_graph=create_graph, strict=False)
    return jvp.flatten(1)


def road_and_orth(z: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    coeff = z @ u
    proj = coeff @ u.T
    return coeff, z - proj


def step_decomposition(dh: torch.Tensor, u: torch.Tensor) -> dict:
    coeff, orth = road_and_orth(dh.flatten(1), u)
    road_norm = coeff.norm(dim=1)
    orth_norm = orth.norm(dim=1)
    total = dh.flatten(1).norm(dim=1)
    return {
        "road_energy_frac": (road_norm.square() / total.square().clamp_min(1e-12)).detach().cpu().numpy(),
        "road_norm": road_norm.detach().cpu().numpy(),
        "orth_norm": orth_norm.detach().cpu().numpy(),
        "step_norm": total.detach().cpu().numpy(),
    }


def batch_from_subset(dataset, rows: pd.DataFrame, start: int, batch_size: int, device: torch.device):
    batch = rows.iloc[start : start + batch_size]
    xs = torch.stack([dataset[int(r.dataset_idx)][0] for r in batch.itertuples(index=False)]).to(device)
    ys = torch.tensor([int(r.label) for r in batch.itertuples(index=False)], device=device)
    return xs, ys


def evaluate_jvp_scaling(
    old_model: DampedResNet50,
    new_model: DampedResNet50,
    dataset,
    rows: pd.DataFrame,
    u: torch.Tensor,
    args,
    device: torch.device,
) -> dict:
    old_model.eval()
    new_model.eval()
    road_ratios, orth_rel_errors = [], []
    rng = torch.Generator(device=device).manual_seed(args.seed + 991)
    for start in range(0, len(rows), args.eval_jvp_batch_size):
        x, _y = batch_from_subset(dataset, rows, start, args.eval_jvp_batch_size, device)
        r = torch.where(
            torch.rand(x.shape, generator=rng, device=device) < 0.5,
            -torch.ones_like(x),
            torch.ones_like(x),
        ) * (args.probe_eps / 255.0)
        with torch.no_grad():
            pass
        with torch.enable_grad():
            j_old = jvp_hidden(old_model, x, r, create_graph=False).detach()
            j_new = jvp_hidden(new_model, x, r, create_graph=False).detach()
        road_old, orth_old = road_and_orth(j_old, u)
        road_new, orth_new = road_and_orth(j_new, u)
        road_ratios.extend(
            (road_new.norm(dim=1) / road_old.norm(dim=1).clamp_min(1e-8)).detach().cpu().numpy().tolist()
        )
        orth_rel_errors.extend(
            ((orth_new - orth_old).norm(dim=1) / orth_old.norm(dim=1).clamp_min(1e-8)).detach().cpu().numpy().tolist()
        )
    return {
        "heldout_road_ratio_mean": float(np.mean(road_ratios)),
        "heldout_road_ratio_median": float(np.median(road_ratios)),
        "heldout_orth_relerr_mean": float(np.mean(orth_rel_errors)),
        "heldout_orth_relerr_median": float(np.median(orth_rel_errors)),
    }


def track_pgd_road_usage(
    model: DampedResNet50,
    dataset,
    rows: pd.DataFrame,
    u: torch.Tensor,
    gamma: float,
    args,
    device: torch.device,
) -> tuple[list[dict], dict]:
    eps = args.eps / 255.0
    alpha = args.pgd_step / 255.0
    step_rows = []
    final_success = []
    first_success_steps = []
    for row in rows.itertuples(index=False):
        torch.manual_seed(args.seed + int(row.dataset_idx) * 1009 + args.track_pgd_steps)
        x0, _ = dataset[int(row.dataset_idx)]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
        with torch.no_grad():
            h_prev = model.pooled_layer4(x).detach()
        first_success = -1
        for step in range(args.track_pgd_steps):
            x.requires_grad_(True)
            loss = F.cross_entropy(model(x), y)
            grad = torch.autograd.grad(loss, x)[0]
            x_next = project_linf(x.detach() + alpha * grad.detach().sign(), x0, eps)
            with torch.no_grad():
                h_next = model.pooled_layer4(x_next).detach()
                logits = model(x_next)
                pred = int(logits.argmax(1).item())
                cur_success = int(pred != int(row.label))
                cur_margin = float(margin(logits, y).item())
            if cur_success and first_success < 0:
                first_success = step + 1
            dec = step_decomposition(h_next - h_prev, u)
            step_rows.append(
                {
                    "gamma": gamma,
                    "attack": "pgd",
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "step": step + 1,
                    "accepted": 1,
                    "current_success": cur_success,
                    "current_margin": cur_margin,
                    "road_energy_frac": float(dec["road_energy_frac"][0]),
                    "road_norm": float(dec["road_norm"][0]),
                    "orth_norm": float(dec["orth_norm"][0]),
                    "step_norm": float(dec["step_norm"][0]),
                }
            )
            x = x_next
            h_prev = h_next
        with torch.no_grad():
            logits = model(x)
            succ = int(logits.argmax(1).item() != int(row.label))
        final_success.append(succ)
        first_success_steps.append(first_success)
    df = pd.DataFrame(step_rows)
    summary = {
        "gamma": gamma,
        "attack": "pgd",
        "n_images": int(len(rows)),
        "asr": float(np.mean(final_success)),
        "mean_first_success_step": float(pd.Series(first_success_steps).replace(-1, np.nan).mean()),
        "mean_road_energy_frac": float(df.road_energy_frac.mean()),
        "median_road_energy_frac": float(df.road_energy_frac.median()),
        "mean_road_norm": float(df.road_norm.mean()),
        "mean_orth_norm": float(df.orth_norm.mean()),
        "mean_step_norm": float(df.step_norm.mean()),
    }
    return step_rows, summary


def track_square_road_usage(
    model: DampedResNet50,
    dataset,
    rows: pd.DataFrame,
    u: torch.Tensor,
    gamma: float,
    args,
    device: torch.device,
) -> tuple[list[dict], dict]:
    eps = args.eps / 255.0
    rng = np.random.default_rng(args.seed + 911)
    step_rows = []
    final_success = []
    first_success_steps = []
    for row in rows.itertuples(index=False):
        x0, _ = dataset[int(row.dataset_idx)]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
        with torch.no_grad():
            best_loss = float(F.cross_entropy(model(x), y).item())
            h_prev = model.pooled_layer4(x).detach()
        first_success = -1
        accepted_count = 0
        for q in range(args.track_square_queries):
            size = max(1, int(round(32 * (1 - q / max(args.track_square_queries, 1)) ** 0.5)))
            i = int(rng.integers(0, 33 - size))
            j = int(rng.integers(0, 33 - size))
            cand = x.clone()
            sign = -1.0 if rng.random() < 0.5 else 1.0
            cand[:, :, i : i + size, j : j + size] = x0[:, :, i : i + size, j : j + size] + sign * eps
            cand = project_linf(cand, x0, eps)
            with torch.no_grad():
                logits = model(cand)
                loss = float(F.cross_entropy(logits, y).item())
                pred = int(logits.argmax(1).item())
                cur_success = int(pred != int(row.label))
                cur_margin = float(margin(logits, y).item())
            accepted = bool(loss > best_loss or cur_success)
            if accepted:
                with torch.no_grad():
                    h_next = model.pooled_layer4(cand).detach()
                dec = step_decomposition(h_next - h_prev, u)
                accepted_count += 1
                step_rows.append(
                    {
                        "gamma": gamma,
                        "attack": "square",
                        "dataset_idx": int(row.dataset_idx),
                        "label": int(row.label),
                        "step": q + 1,
                        "accepted": 1,
                        "current_success": cur_success,
                        "current_margin": cur_margin,
                        "road_energy_frac": float(dec["road_energy_frac"][0]),
                        "road_norm": float(dec["road_norm"][0]),
                        "orth_norm": float(dec["orth_norm"][0]),
                        "step_norm": float(dec["step_norm"][0]),
                    }
                )
                x = cand
                h_prev = h_next
                best_loss = max(best_loss, loss)
            if cur_success:
                first_success = q + 1
                break
        with torch.no_grad():
            logits = model(x)
            succ = int(logits.argmax(1).item() != int(row.label))
        final_success.append(succ)
        first_success_steps.append(first_success)
        if accepted_count == 0:
            step_rows.append(
                {
                    "gamma": gamma,
                    "attack": "square",
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "step": args.track_square_queries,
                    "accepted": 0,
                    "current_success": succ,
                    "current_margin": float(margin(logits, y).item()),
                    "road_energy_frac": np.nan,
                    "road_norm": 0.0,
                    "orth_norm": 0.0,
                    "step_norm": 0.0,
                }
            )
    df = pd.DataFrame(step_rows)
    summary = {
        "gamma": gamma,
        "attack": "square",
        "n_images": int(len(rows)),
        "asr": float(np.mean(final_success)),
        "mean_first_success_step": float(pd.Series(first_success_steps).replace(-1, np.nan).mean()),
        "mean_road_energy_frac": float(df.road_energy_frac.mean()),
        "median_road_energy_frac": float(df.road_energy_frac.median()),
        "mean_road_norm": float(df.road_norm.mean()),
        "mean_orth_norm": float(df.orth_norm.mean()),
        "mean_step_norm": float(df.step_norm.mean()),
    }
    return step_rows, summary


def train_one_gamma(
    old_model: DampedResNet50,
    train_model: DampedResNet50,
    train_loader: DataLoader,
    u: torch.Tensor,
    gamma: float,
    args,
    device: torch.device,
) -> pd.DataFrame:
    freeze_for_mode(train_model, args.finetune)
    params = [p for p in train_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    old_model.eval()
    train_model.eval()
    rng = torch.Generator(device=device).manual_seed(args.seed + int(round(gamma * 1000)) + 101)
    rows = []
    step = 0
    for epoch in range(args.epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            r = torch.where(
                torch.rand(x.shape, generator=rng, device=device) < 0.5,
                -torch.ones_like(x),
                torch.ones_like(x),
            ) * (args.probe_eps / 255.0)
            with torch.no_grad():
                logits_old = old_model(x)
            logits_new = train_model(x)
            ce = F.cross_entropy(logits_new, y)
            kl = F.kl_div(
                F.log_softmax(logits_new, dim=1),
                F.softmax(logits_old.detach(), dim=1),
                reduction="batchmean",
            )
            with torch.enable_grad():
                j_old = jvp_hidden(old_model, x, r, create_graph=False).detach()
                j_new = jvp_hidden(train_model, x, r, create_graph=True)
            road_old, orth_old = road_and_orth(j_old, u)
            road_new, orth_new = road_and_orth(j_new, u)
            road_loss = F.mse_loss(road_new, gamma * road_old)
            orth_loss = F.mse_loss(orth_new, orth_old)
            loss = ce + args.alpha_kl * kl + args.lambda_road * road_loss + args.beta_orth * orth_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if step % args.log_every == 0:
                rows.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "gamma": gamma,
                        "loss": float(loss.detach().cpu().item()),
                        "ce": float(ce.detach().cpu().item()),
                        "kl": float(kl.detach().cpu().item()),
                        "road_loss": float(road_loss.detach().cpu().item()),
                        "orth_loss": float(orth_loss.detach().cpu().item()),
                        "train_road_ratio": float(
                            (road_new.norm(dim=1) / road_old.norm(dim=1).clamp_min(1e-8)).detach().mean().cpu().item()
                        ),
                    }
                )
            step += 1
            if args.max_steps and step >= args.max_steps:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/road_jacobian_scaling_resnet50_pilot")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--train-images", type=int, default=256)
    p.add_argument("--calib-images", type=int, default=80)
    p.add_argument("--eval-images", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-jvp-batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--finetune", choices=["layer4", "fc", "layer4_fc", "all"], default="layer4_fc")
    p.add_argument("--gammas", default="1.0,0.5,0.25")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--basis-name", default="adv_road")
    p.add_argument("--calib-eps", type=float, default=8.0)
    p.add_argument("--calib-dirs", type=int, default=256)
    p.add_argument("--basis-select", type=int, default=6000)
    p.add_argument("--probe-eps", type=float, default=8.0)
    p.add_argument("--alpha-kl", type=float, default=1.0)
    p.add_argument("--lambda-road", type=float, default=100.0)
    p.add_argument("--beta-orth", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--pgd-step", type=float, default=2.0)
    p.add_argument("--pgd-steps", default="20,100")
    p.add_argument("--square-queries", type=int, default=300)
    p.add_argument("--track-road-usage", action="store_true")
    p.add_argument("--track-pgd-steps", type=int, default=100)
    p.add_argument("--track-square-queries", type=int, default=300)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = datasets.CIFAR10(args.dataset_root, train=True, download=False, transform=transforms.ToTensor())
    test_set = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    old_model = load_wrapped_model(device)
    old_model.eval()
    for param in old_model.parameters():
        param.requires_grad_(False)

    train_rows = select_correct(train_set, old_model, args.train_images, device)
    calib_rows = select_correct(test_set, old_model, args.calib_images, device)
    eval_rows = select_correct(test_set, old_model, args.eval_images, device, max_scan=10000)
    train_rows.to_csv(out_dir / "train_clean_correct.csv", index=False)
    calib_rows.to_csv(out_dir / "calib_clean_correct.csv", index=False)
    eval_rows.to_csv(out_dir / "eval_clean_correct.csv", index=False)

    bases = fit_candidate_bases(old_model, test_set, calib_rows, np.zeros(2048, dtype=np.float32), args, device)
    rng = np.random.default_rng(args.seed + 311)
    q, _ = np.linalg.qr(rng.normal(size=(2048, args.k)).astype(np.float32))
    bases["random"] = q[:, : args.k].T.astype(np.float32)
    if args.basis_name not in bases:
        raise RuntimeError(f"basis {args.basis_name!r} not available; got {sorted(bases)}")
    basis = bases[args.basis_name]
    np.savez_compressed(out_dir / "road_basis.npz", **{args.basis_name: basis})
    u = torch.as_tensor(basis.T, dtype=torch.float32, device=device)

    train_loader = make_loader(train_set, train_rows, args.batch_size, shuffle=True, seed=args.seed)
    summaries = []
    all_usage_steps = []
    usage_summaries = []
    for gamma in [float(x) for x in args.gammas.split(",") if x.strip()]:
        print(f"[gamma={gamma}] fine-tuning true hidden-Jacobian road scaling", flush=True)
        new_model = copy.deepcopy(old_model).to(device).eval()
        gamma_tag = str(gamma).replace(".", "p")
        if abs(gamma - 1.0) < 1e-8:
            hist = pd.DataFrame(
                [
                    {
                        "epoch": 0,
                        "step": 0,
                        "gamma": gamma,
                        "loss": 0.0,
                        "ce": 0.0,
                        "kl": 0.0,
                        "road_loss": 0.0,
                        "orth_loss": 0.0,
                        "train_road_ratio": 1.0,
                        "note": "no_finetune_baseline",
                    }
                ]
            )
        else:
            hist = train_one_gamma(old_model, new_model, train_loader, u, gamma, args, device)
            torch.save(new_model.state_dict(), out_dir / f"model_gamma{gamma_tag}.pt")
        hist.to_csv(out_dir / f"train_history_gamma{gamma_tag}.csv", index=False)
        clean_old = eval_clean(old_model, test_set, eval_rows, device, args.batch_size)
        clean_new = eval_clean(new_model, test_set, eval_rows, device, args.batch_size)
        jvp_stats = evaluate_jvp_scaling(old_model, new_model, test_set, eval_rows, u, args, device)
        row = {
            "basis": args.basis_name,
            "gamma": gamma,
            "k": args.k,
            "finetune": args.finetune,
            "train_images": len(train_rows),
            "eval_images": len(eval_rows),
            "old_clean_acc": clean_old["clean_acc"],
            "old_clean_margin_mean": clean_old["clean_margin_mean"],
            "new_clean_acc": clean_new["clean_acc"],
            "new_clean_margin_mean": clean_new["clean_margin_mean"],
            **jvp_stats,
        }
        for steps in [int(x) for x in args.pgd_steps.split(",") if x.strip()]:
            pgd = pgd_asr(new_model, test_set, eval_rows, args, device, steps=steps)
            row.update(pgd)
        sq = square_asr(new_model, test_set, eval_rows, args, device)
        row.update({"square_asr": sq["square_asr"], "square_robust_acc": sq["square_robust_acc"]})
        summaries.append(row)
        pd.DataFrame(summaries).to_csv(out_dir / "road_jacobian_scaling_summary.partial.csv", index=False)
        print(pd.DataFrame([row]).to_string(index=False), flush=True)
        if args.track_road_usage:
            pgd_steps, pgd_summary = track_pgd_road_usage(new_model, test_set, eval_rows, u, gamma, args, device)
            sq_steps, sq_summary = track_square_road_usage(new_model, test_set, eval_rows, u, gamma, args, device)
            all_usage_steps.extend(pgd_steps)
            all_usage_steps.extend(sq_steps)
            usage_summaries.extend([pgd_summary, sq_summary])
            pd.DataFrame(all_usage_steps).to_csv(out_dir / "road_usage_steps.partial.csv", index=False)
            pd.DataFrame(usage_summaries).to_csv(out_dir / "road_usage_summary.partial.csv", index=False)
            print(pd.DataFrame([pgd_summary, sq_summary]).to_string(index=False), flush=True)

    pd.DataFrame(summaries).to_csv(out_dir / "road_jacobian_scaling_summary.csv", index=False)
    if args.track_road_usage:
        pd.DataFrame(all_usage_steps).to_csv(out_dir / "road_usage_steps.csv", index=False)
        pd.DataFrame(usage_summaries).to_csv(out_dir / "road_usage_summary.csv", index=False)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(vars(args) | {"device": str(device)}, f, indent=2)


if __name__ == "__main__":
    main()
