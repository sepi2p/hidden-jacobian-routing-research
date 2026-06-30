#!/usr/bin/env python3
"""Benchmark-style CIFAR optimizer transport trajectories using learned transport axes.

This reruns the cross-optimizer transport analysis with stronger attack
implementations and larger budgets than the quick exploratory run. The Square
trajectory follows the standard Linf Square Attack p-schedule used in the local
attack implementation, while the query attacks use higher budgets and save a
fixed number of trajectory checkpoints.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import gc
from types import SimpleNamespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar_global_vs_class_success_flow import (  # noqa: E402
    LAYER_GROUPS,
)
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import (  # noqa: E402
    build_mu,
    eval_all,
    load_model,
    margin,
    project_linf,
    select_clean_correct,
)
from experiments.pure_af_geometry.run_cifar_pc_transport_mode_attack import (  # noqa: E402
    build_pc_directions,
)
from attacks.square import p_selection  # noqa: E402


def true_prob(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1).gather(1, y.view(-1, 1)).squeeze(1)


def forward_features(wrapper, x: torch.Tensor):
    with torch.no_grad():
        logits, feats, _raw = wrapper.forward_with_features(x)
    return logits.detach(), {k: v.detach().cpu().numpy()[0].astype(np.float32) for k, v in feats.items()}


def pgd_trajectory(wrapper, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int, step_size: float):
    x0 = x.detach()
    x_adv = x0.clone()
    states = [x_adv.detach().clone()]
    for _ in range(steps):
        probe = x_adv.detach().requires_grad_(True)
        logits = wrapper(probe)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, probe)[0]
        x_adv = project_linf(x_adv + step_size * grad.sign(), x0, eps)
        states.append(x_adv.detach().clone())
    return states


def _smooth_grad(grad: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    return F.avg_pool2d(grad, kernel_size=kernel, stride=1, padding=kernel // 2)


def _scale_invariant_grad(wrapper, x_adv: torch.Tensor, y: torch.Tensor, scales=(1.0, 0.5, 0.25)) -> torch.Tensor:
    total = torch.zeros_like(x_adv)
    for scale in scales:
        probe = x_adv.detach().requires_grad_(True)
        if scale == 1.0:
            inp = probe
        else:
            h, w = probe.shape[-2:]
            small = F.interpolate(probe, scale_factor=scale, mode="bilinear", align_corners=False)
            inp = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
        logits = wrapper(inp)
        loss = F.cross_entropy(logits, y)
        total = total + torch.autograd.grad(loss, probe)[0]
    return total / float(len(scales))


def gradient_variant_trajectory(
    wrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    attack: str,
    decay: float = 1.0,
):
    x0 = x.detach()
    x_adv = x0.clone()
    momentum = torch.zeros_like(x_adv)
    states = [x_adv.detach().clone()]
    for _ in range(steps):
        if attack == "ni_fgsm":
            probe_in = project_linf(x_adv + decay * step_size * momentum.sign(), x0, eps)
        else:
            probe_in = x_adv

        if attack == "si_fgsm":
            grad = _scale_invariant_grad(wrapper, probe_in, y)
        else:
            probe = probe_in.detach().requires_grad_(True)
            logits = wrapper(probe)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, probe)[0]

        if attack == "ti_fgsm":
            grad = _smooth_grad(grad)
        if attack in {"mi_fgsm", "ni_fgsm"}:
            grad_norm = grad.abs().flatten(1).mean(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
            momentum = decay * momentum + grad / grad_norm
            grad = momentum

        x_adv = project_linf(x_adv + step_size * grad.sign(), x0, eps)
        states.append(x_adv.detach().clone())
    return states


def checkpoint_indices(n_steps: int, n_checkpoints: int) -> set[int]:
    if n_steps <= 0:
        return {0}
    n_checkpoints = max(2, min(n_checkpoints, n_steps + 1))
    return set(int(round(x)) for x in np.linspace(0, n_steps, n_checkpoints))


def square_trajectory(
    wrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    seed: int,
    p_init: float,
    init_epochs: int,
    n_checkpoints: int,
):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    x0 = x.detach()
    c, h, w = x.shape[1:]
    # Standard Square Attack starts from random vertical stripes at +/- eps.
    stripe = torch.where(
        torch.rand((1, c, 1, w), generator=gen, device=x.device) < 0.5,
        -torch.ones((1, c, 1, w), device=x.device),
        torch.ones((1, c, 1, w), device=x.device),
    ) * eps
    x_adv = (x0 + stripe).clamp(0, 1)
    states = [x_adv.detach().clone()]
    with torch.no_grad():
        best_margin = margin(wrapper(x_adv), y)
    save_at = checkpoint_indices(steps, n_checkpoints)
    for step in range(1, steps + 1):
        perturbation = x_adv - x0
        p = p_selection(p_init, step + init_epochs, steps)
        side = int(round(np.sqrt(p * c * h * w / c)))
        side = min(max(side, 1), h - 1)
        top = int(torch.randint(0, h - side + 1, (1,), generator=gen, device=x.device).item())
        left = int(torch.randint(0, w - side + 1, (1,), generator=gen, device=x.device).item())
        patch = torch.where(
            torch.rand((1, c, 1, 1), generator=gen, device=x.device) < 0.5,
            -torch.ones((1, c, 1, 1), device=x.device),
            torch.ones((1, c, 1, 1), device=x.device),
        ) * eps
        perturbation[:, :, top : top + side, left : left + side] = patch
        candidate = (x0 + perturbation).clamp(0, 1)
        with torch.no_grad():
            cand_margin = margin(wrapper(candidate), y)
        if float(cand_margin.item()) < float(best_margin.item()):
            x_adv = candidate.detach()
            best_margin = cand_margin.detach()
        if step in save_at:
            states.append(x_adv.detach().clone())
    return states


def random_search_trajectory(wrapper, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int, seed: int, n_checkpoints: int):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    x0 = x.detach()
    x_adv = x0.clone()
    states = [x_adv.detach().clone()]
    with torch.no_grad():
        best_margin = margin(wrapper(x_adv), y)
    save_at = checkpoint_indices(steps, n_checkpoints)
    for step in range(1, steps + 1):
        noise = torch.empty_like(x_adv).uniform_(-eps, eps, generator=gen)
        candidate = (x0 + noise).clamp(0, 1)
        with torch.no_grad():
            cand_margin = margin(wrapper(candidate), y)
        if float(cand_margin.item()) < float(best_margin.item()):
            x_adv = candidate.detach()
            best_margin = cand_margin.detach()
        if step in save_at:
            states.append(x_adv.detach().clone())
    return states


def signhunter_trajectory(wrapper, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int, seed: int, n_checkpoints: int):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    x0 = x.detach()
    n_features = int(np.prod(x0.shape[1:]))
    perturbation = torch.where(
        torch.rand(n_features, generator=gen, device=x.device) < 0.5,
        -torch.ones(n_features, device=x.device),
        torch.ones(n_features, device=x.device),
    ) * eps
    x_adv = (x0 + perturbation.view_as(x0)).clamp(0, 1)
    states = [x_adv.detach().clone()]
    with torch.no_grad():
        best_margin = margin(wrapper(x_adv), y)
    save_at = checkpoint_indices(steps, n_checkpoints)
    query = 0
    while query < steps:
        max_h = int(np.ceil(np.log2(n_features))) + 1
        for h in range(max_h):
            chunk_len = int(np.ceil(n_features / (2**h)))
            for i in range(2**h):
                if query >= steps:
                    break
                start = i * chunk_len
                end = min(start + chunk_len, n_features)
                if start >= end:
                    continue
                perturbation[start:end] *= -1
                candidate = (x0 + perturbation.view_as(x0)).clamp(0, 1)
                with torch.no_grad():
                    cand_margin = margin(wrapper(candidate), y)
                query += 1
                if float(cand_margin.item()) <= float(best_margin.item()):
                    x_adv = candidate.detach()
                    best_margin = cand_margin.detach()
                else:
                    perturbation[start:end] *= -1
                if query in save_at:
                    states.append(x_adv.detach().clone())
            if query >= steps:
                break
    if len(states) == 1 or not torch.equal(states[-1], x_adv):
        states.append(x_adv.detach().clone())
    return states


def nes_trajectory(
    wrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    seed: int,
    samples: int,
    sigma: float,
    n_checkpoints: int,
):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    x0 = x.detach()
    x_adv = x0.clone()
    states = [x_adv.detach().clone()]
    save_at = checkpoint_indices(steps, n_checkpoints)
    for step in range(1, steps + 1):
        grad_est = torch.zeros_like(x_adv)
        for _j in range(samples):
            u = torch.randn(x_adv.shape, generator=gen, device=x.device)
            xp = project_linf(x_adv + sigma * u, x0, eps)
            xm = project_linf(x_adv - sigma * u, x0, eps)
            with torch.no_grad():
                fp = -margin(wrapper(xp), y)
                fm = -margin(wrapper(xm), y)
            grad_est = grad_est + ((fp - fm).view(-1, 1, 1, 1) / (2.0 * sigma)) * u
        grad_est = grad_est / float(max(samples, 1))
        x_adv = project_linf(x_adv + step_size * grad_est.sign(), x0, eps)
        if step in save_at:
            states.append(x_adv.detach().clone())
    return states


def bandit_trajectory(
    wrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    seed: int,
    sigma: float,
    prior_lr: float,
    n_checkpoints: int,
):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    x0 = x.detach()
    x_adv = x0.clone()
    prior = torch.zeros_like(x_adv)
    states = [x_adv.detach().clone()]
    save_at = checkpoint_indices(steps, n_checkpoints)
    for step in range(1, steps + 1):
        u = torch.randn(x_adv.shape, generator=gen, device=x.device)
        u = 0.85 * prior + 0.15 * u
        xp = project_linf(x_adv + sigma * u.sign(), x0, eps)
        xm = project_linf(x_adv - sigma * u.sign(), x0, eps)
        with torch.no_grad():
            fp = -margin(wrapper(xp), y)
            fm = -margin(wrapper(xm), y)
        est = ((fp - fm).view(-1, 1, 1, 1) / (2.0 * sigma)) * u
        prior = 0.9 * prior + prior_lr * est
        x_adv = project_linf(x_adv + step_size * prior.sign(), x0, eps)
        if step in save_at:
            states.append(x_adv.detach().clone())
    return states


def feature_state_rows(wrapper, states, y: torch.Tensor):
    rows = []
    feat_by_step = {}
    for step, state in enumerate(states):
        logits, feats = forward_features(wrapper, state)
        feat_by_step[step] = feats
        pred = int(logits.argmax(1).item())
        rows.append({
            "step": int(step),
            "pred": pred,
            "success_at_step": int(pred != int(y.item())),
            "margin": float(margin(logits, y).item()),
            "true_prob": float(true_prob(logits, y).item()),
        })
    return rows, feat_by_step


def project_trajectory(
    *,
    pc_dirs: dict,
    wrapper,
    source_model: str,
    attack: str,
    dataset_idx: int,
    image_ord: int,
    label: int,
    final_success: int,
    states,
    y: torch.Tensor,
    layer_groups: list[str],
    top_k: int,
):
    state_rows, feat_by_step = feature_state_rows(wrapper, states, y)
    out = []
    n_steps = max(len(states) - 1, 1)
    for layer_group in layer_groups:
        layer = LAYER_GROUPS[layer_group][source_model]
        h0 = feat_by_step[0].get(layer)
        if h0 is None:
            continue
        axes = []
        for pc in range(1, top_k + 1):
            v = pc_dirs.get((source_model, layer, pc))
            if v is not None:
                axes.append(v)
        if not axes:
            continue
        basis = np.stack(axes).astype(np.float32)
        for meta in state_rows:
            step = meta["step"]
            ht = feat_by_step[step].get(layer)
            if ht is None:
                continue
            delta = ht - h0
            coeff = basis @ delta
            transport_energy = float(np.sum(coeff * coeff))
            total_energy = float(np.sum(delta * delta))
            frac = transport_energy / max(total_energy, 1e-12)
            base = {
                "model": source_model,
                "attack": attack,
                "dataset_idx": int(dataset_idx),
                "image_ord": int(image_ord),
                "label": int(label),
                "layer_group": layer_group,
                "layer": layer,
                "step": int(step),
                "normalized_progress": float(step / n_steps),
                "time_bin": min(4, int(np.floor((step / n_steps) * 5.0))) if step < n_steps else 4,
                "final_success": int(final_success),
                "step_success": int(meta["success_at_step"]),
                "pred": int(meta["pred"]),
                "margin": float(meta["margin"]),
                "true_prob": float(meta["true_prob"]),
                "transport_energy_topk": transport_energy,
                "total_feature_energy": total_energy,
                "frac_energy_topk": float(frac),
            }
            for i, c in enumerate(coeff, start=1):
                base[f"pc{i}_coeff"] = float(c)
                base[f"pc{i}_abs_coeff"] = float(abs(c))
            out.append(base)
    return out


def summarize_timeseries(df: pd.DataFrame, out_dir: Path):
    summary = df.groupby(
        ["model", "attack", "layer_group", "layer", "final_success", "time_bin"], dropna=False
    ).agg(
        n=("frac_energy_topk", "size"),
        mean_frac_energy=("frac_energy_topk", "mean"),
        median_frac_energy=("frac_energy_topk", "median"),
        mean_transport_energy=("transport_energy_topk", "mean"),
        mean_total_feature_energy=("total_feature_energy", "mean"),
        mean_margin=("margin", "mean"),
        mean_true_prob=("true_prob", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "attack_axis_energy_summary.csv", index=False)

    final = df.sort_values("step").groupby(["model", "attack", "dataset_idx", "layer_group"], as_index=False).tail(1)
    comp = final.groupby(["model", "attack", "layer_group", "layer"], dropna=False).agg(
        n=("final_success", "size"),
        asr=("final_success", "mean"),
        mean_final_frac_energy=("frac_energy_topk", "mean"),
        median_final_frac_energy=("frac_energy_topk", "median"),
        mean_final_margin=("margin", "mean"),
    ).reset_index()
    comp.to_csv(out_dir / "attack_axis_by_attack_comparison.csv", index=False)
    return summary, comp


def success_prediction(df: pd.DataFrame, out_dir: Path, seed: int):
    rows = []
    pc_cols = [c for c in df.columns if c.endswith("_coeff") or c.endswith("_abs_coeff")]
    for (model, attack, layer_group, layer, time_bin), g in df.groupby(["model", "attack", "layer_group", "layer", "time_bin"]):
        per_img = g.sort_values("step").groupby("dataset_idx", as_index=False).tail(1).copy()
        y = per_img["final_success"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2 or len(per_img) < 20:
            rows.append({
                "model": model,
                "attack": attack,
                "layer_group": layer_group,
                "layer": layer,
                "time_bin": int(time_bin),
                "n": int(len(per_img)),
                "positive_rate": float(np.mean(y)) if len(y) else np.nan,
                "auroc_frac_energy": np.nan,
                "logreg_accuracy": np.nan,
                "note": "single_class_or_too_few",
            })
            continue
        try:
            auroc = float(roc_auc_score(y, per_img["frac_energy_topk"].to_numpy(dtype=float)))
        except ValueError:
            auroc = np.nan
        x = per_img[["frac_energy_topk", "transport_energy_topk", "total_feature_energy", *pc_cols]].fillna(0).to_numpy(dtype=float)
        try:
            xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.35, stratify=y, random_state=seed)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
            clf.fit(xtr, ytr)
            acc = float(accuracy_score(yte, clf.predict(xte)))
            note = ""
        except ValueError as exc:
            acc = np.nan
            note = str(exc)
        rows.append({
            "model": model,
            "attack": attack,
            "layer_group": layer_group,
            "layer": layer,
            "time_bin": int(time_bin),
            "n": int(len(per_img)),
            "positive_rate": float(np.mean(y)),
            "auroc_frac_energy": auroc,
            "logreg_accuracy": acc,
            "note": note,
        })
    pred = pd.DataFrame(rows)
    pred.to_csv(out_dir / "attack_axis_success_prediction.csv", index=False)
    return pred


def plot_outputs(summary: pd.DataFrame, out_dir: Path):
    for metric, filename, ylabel in [
        ("mean_frac_energy", "attack_axis_timeseries_plots.png", "Mean top-5 transport energy fraction"),
        ("mean_transport_energy", "attack_axis_energy_curves.png", "Mean top-5 transport energy"),
    ]:
        layer_groups = ["hidden", "penultimate", "logits"]
        models = list(dict.fromkeys(summary.model))
        fig, axes = plt.subplots(len(models), len(layer_groups), figsize=(15, 3.5 * len(models)), sharex=True, constrained_layout=True)
        if len(models) == 1:
            axes = np.expand_dims(axes, 0)
        for r, model in enumerate(models):
            for c, layer_group in enumerate(layer_groups):
                ax = axes[r, c]
                sub = summary[(summary.model == model) & (summary.layer_group == layer_group)]
                for (attack, success), g in sub.groupby(["attack", "final_success"]):
                    g = g.sort_values("time_bin")
                    label = f"{attack} {'succ' if success else 'fail'}"
                    ax.plot(g.time_bin, g[metric], marker="o", label=label)
                ax.set_title(f"{model} {layer_group}")
                ax.set_xlabel("time bin")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.25)
                if r == 0 and c == len(layer_groups) - 1:
                    ax.legend(fontsize=7)
        fig.savefig(out_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_partial(rows: list[dict], out_dir: Path):
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "partial_attack_axis_projection_timeseries.csv", index=False)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    layer_groups = [g.strip() for g in args.layer_groups.split(",") if g.strip()]

    mu = build_mu(Path(args.layerwise_dir))
    pc_dirs, pc_meta = build_pc_directions(mu, args.top_k)
    pc_meta.to_csv(out_dir / "attack_axis_transport_axes.csv", index=False)

    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())

    rows = []
    completed = set()
    partial = out_dir / "partial_attack_axis_projection_timeseries.csv"
    final = out_dir / "attack_axis_projection_timeseries.csv"
    resume_path = final if final.exists() else partial
    if args.resume and resume_path.exists():
        old = pd.read_csv(resume_path)
        rows = old.to_dict("records")
        for r in old[["model", "attack", "dataset_idx"]].drop_duplicates().itertuples():
            completed.add((r.model, r.attack, int(r.dataset_idx)))
        print(f"[RESUME] rows={len(rows)} completed trajectories={len(completed)}", flush=True)

    eps = args.eps / 255.0
    pgd_step_size = args.pgd_step_size / 255.0 if args.pgd_step_size > 0 else eps / max(args.pgd_steps, 1)
    selected_counts = {}
    for model_name in args.models:
        wrapper = load_model(model_name, device)
        select_args = SimpleNamespace(**{**vars(args), "models": [model_name]})
        selected = select_clean_correct(dataset, {model_name: wrapper}, select_args, device)
        selected_counts[model_name] = len(selected)
        print(f"[MODEL] {model_name} images={len(selected)} attacks={attacks}", flush=True)
        for image_ord, (dataset_idx, label) in enumerate(selected):
            x_cpu, _ = dataset[dataset_idx]
            x = x_cpu.unsqueeze(0).to(device)
            y = torch.tensor([label], device=device)
            for attack in attacks:
                key = (model_name, attack, int(dataset_idx))
                if key in completed:
                    continue
                if attack == "pgd":
                    states = pgd_trajectory(wrapper, x, y, eps, args.pgd_steps, pgd_step_size)
                elif attack in {"mi_fgsm", "ni_fgsm", "ti_fgsm", "si_fgsm"}:
                    states = gradient_variant_trajectory(
                        wrapper, x, y, eps, args.pgd_steps, pgd_step_size, attack, args.momentum_decay
                    )
                elif attack == "square":
                    states = square_trajectory(
                        wrapper,
                        x,
                        y,
                        eps,
                        args.square_steps,
                        args.seed + image_ord * 1009 + len(model_name) * 17,
                        args.square_p_init,
                        args.square_init_epochs,
                        args.saved_checkpoints,
                    )
                elif attack == "random_search":
                    states = random_search_trajectory(
                        wrapper,
                        x,
                        y,
                        eps,
                        args.query_steps,
                        args.seed + image_ord * 1009 + len(model_name) * 17,
                        args.saved_checkpoints,
                    )
                elif attack == "signhunter":
                    states = signhunter_trajectory(
                        wrapper,
                        x,
                        y,
                        eps,
                        args.query_steps,
                        args.seed + image_ord * 1009 + len(model_name) * 17,
                        args.saved_checkpoints,
                    )
                elif attack == "nes":
                    states = nes_trajectory(
                        wrapper,
                        x,
                        y,
                        eps,
                        args.query_steps,
                        args.query_step_size / 255.0,
                        args.seed + image_ord * 1009 + len(model_name) * 17,
                        args.nes_samples,
                        args.nes_sigma / 255.0,
                        args.saved_checkpoints,
                    )
                elif attack == "bandit":
                    states = bandit_trajectory(
                        wrapper,
                        x,
                        y,
                        eps,
                        args.query_steps,
                        args.query_step_size / 255.0,
                        args.seed + image_ord * 1009 + len(model_name) * 17,
                        args.nes_sigma / 255.0,
                        args.bandit_prior_lr,
                        args.saved_checkpoints,
                    )
                else:
                    raise ValueError(f"Unknown attack: {attack}")
                final_eval = eval_all({model_name: wrapper}, states[-1], y)[model_name]
                rows.extend(project_trajectory(
                    pc_dirs=pc_dirs,
                    wrapper=wrapper,
                    source_model=model_name,
                    attack=attack,
                    dataset_idx=int(dataset_idx),
                    image_ord=int(image_ord),
                    label=int(label),
                    final_success=int(final_eval["success"]),
                    states=states,
                    y=y,
                    layer_groups=layer_groups,
                    top_k=args.top_k,
                ))
                completed.add(key)
            if (image_ord + 1) % args.checkpoint_every == 0:
                write_partial(rows, out_dir)
                print(f"  {model_name}: {image_ord + 1}/{len(selected)} rows={len(rows)}", flush=True)
        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "attack_axis_projection_timeseries.csv", index=False)
    summary, comp = summarize_timeseries(df, out_dir)
    pred = success_prediction(df[df.step > 0].copy(), out_dir, args.seed)
    plot_outputs(summary, out_dir)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "args": vars(args),
            "selected_counts": selected_counts,
            "eps": eps,
            "pgd_step_size": pgd_step_size,
            "n_rows": int(len(df)),
        }, f, indent=2)
    print(f"[SAVED] {out_dir}", flush=True)
    print(comp.to_string(index=False), flush=True)
    best_pred = pred.sort_values("auroc_frac_energy", ascending=False).head(12)
    print(best_pred.to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layerwise-dir", default="analysis_outputs/pure_af_geometry/cifar_layerwise_success_flow_c10_s3_g120")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_benchmark_optimizer_transport")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--models", default="bbb_resnet50,bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--attacks", default="pgd,mi_fgsm,ni_fgsm,ti_fgsm,si_fgsm,square,nes,signhunter,random_search,bandit")
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--pgd-steps", type=int, default=40)
    p.add_argument("--pgd-step-size", type=float, default=0.0, help="Step size in /255 units; <=0 uses eps/steps.")
    p.add_argument("--square-steps", type=int, default=1000)
    p.add_argument("--square-p-init", type=float, default=0.3)
    p.add_argument("--square-init-epochs", type=int, default=0)
    p.add_argument("--query-steps", type=int, default=250)
    p.add_argument("--query-step-size", type=float, default=1.0, help="Query-search update step size in /255 units.")
    p.add_argument("--nes-samples", type=int, default=20)
    p.add_argument("--nes-sigma", type=float, default=1.0, help="NES finite-difference sigma in /255 units.")
    p.add_argument("--bandit-prior-lr", type=float, default=0.5)
    p.add_argument("--momentum-decay", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--layer-groups", default="hidden,penultimate,logits")
    p.add_argument("--saved-checkpoints", type=int, default=41)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
