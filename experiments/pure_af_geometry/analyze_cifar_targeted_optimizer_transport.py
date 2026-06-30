#!/usr/bin/env python3
"""Targeted CIFAR optimizer transport validation.

This script tests whether targeted attacks also accumulate energy in the
previously learned transport coordinates.  The coordinates are not refit on the
targeted trajectories; they are built from the existing CIFAR class-flow
artifacts and then used as a fixed coordinate system.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import gc
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

from attacks.square import p_selection  # noqa: E402
from experiments.pure_af_geometry.analyze_cifar_global_vs_class_success_flow import LAYER_GROUPS  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import (  # noqa: E402
    build_mu,
    load_model,
    project_linf,
    select_clean_correct,
)
from experiments.pure_af_geometry.run_cifar_pc_transport_mode_attack import build_pc_directions  # noqa: E402


def target_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tgt = logits.gather(1, target.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, target.view(-1, 1), -1e9)
    return tgt - masked.max(1).values


def source_margin(logits: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    src = logits.gather(1, source.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, source.view(-1, 1), -1e9)
    return src - masked.max(1).values


def checkpoint_indices(n_steps: int, n_checkpoints: int) -> set[int]:
    if n_steps <= 0:
        return {0}
    n_checkpoints = max(2, min(n_checkpoints, n_steps + 1))
    return set(int(round(x)) for x in np.linspace(0, n_steps, n_checkpoints))


def forward_features(wrapper, x: torch.Tensor):
    with torch.no_grad():
        logits, feats, _raw = wrapper.forward_with_features(x)
    return logits.detach(), {k: v.detach().cpu().numpy()[0].astype(np.float32) for k, v in feats.items()}


def targeted_pgd_trajectory(wrapper, x, target, eps, steps, step_size):
    x0 = x.detach()
    x_adv = x0.clone()
    states = [x_adv.detach().clone()]
    for _ in range(steps):
        probe = x_adv.detach().requires_grad_(True)
        logits = wrapper(probe)
        loss = F.cross_entropy(logits, target)
        grad = torch.autograd.grad(loss, probe)[0]
        x_adv = project_linf(x_adv - step_size * grad.sign(), x0, eps)
        states.append(x_adv.detach().clone())
    return states


def targeted_mi_fgsm_trajectory(wrapper, x, target, eps, steps, step_size, decay):
    x0 = x.detach()
    x_adv = x0.clone()
    momentum = torch.zeros_like(x_adv)
    states = [x_adv.detach().clone()]
    for _ in range(steps):
        probe = x_adv.detach().requires_grad_(True)
        logits = wrapper(probe)
        loss = F.cross_entropy(logits, target)
        grad = torch.autograd.grad(loss, probe)[0]
        grad_norm = grad.abs().flatten(1).mean(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        momentum = decay * momentum + grad / grad_norm
        x_adv = project_linf(x_adv - step_size * momentum.sign(), x0, eps)
        states.append(x_adv.detach().clone())
    return states


def targeted_square_trajectory(wrapper, x, target, eps, steps, seed, p_init, n_checkpoints):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    x0 = x.detach()
    c, h, w = x.shape[1:]
    stripe = torch.where(
        torch.rand((1, c, 1, w), generator=gen, device=x.device) < 0.5,
        -torch.ones((1, c, 1, w), device=x.device),
        torch.ones((1, c, 1, w), device=x.device),
    ) * eps
    x_adv = (x0 + stripe).clamp(0, 1)
    states = [x_adv.detach().clone()]
    with torch.no_grad():
        best = target_margin(wrapper(x_adv), target)
    save_at = checkpoint_indices(steps, n_checkpoints)
    for step in range(1, steps + 1):
        perturbation = x_adv - x0
        p = p_selection(p_init, step, steps)
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
            cand = target_margin(wrapper(candidate), target)
        if float(cand.item()) > float(best.item()):
            x_adv = candidate.detach()
            best = cand.detach()
        if step in save_at:
            states.append(x_adv.detach().clone())
    if not torch.equal(states[-1], x_adv):
        states.append(x_adv.detach().clone())
    return states


def state_features(wrapper, states, source, target):
    rows = []
    feat_by_step = {}
    for step, state in enumerate(states):
        logits, feats = forward_features(wrapper, state)
        probs = F.softmax(logits, dim=1)
        pred = int(logits.argmax(1).item())
        feat_by_step[step] = feats
        rows.append(
            {
                "step": int(step),
                "pred": pred,
                "source_prob": float(probs[0, int(source.item())].item()),
                "target_prob": float(probs[0, int(target.item())].item()),
                "source_margin": float(source_margin(logits, source).item()),
                "target_margin": float(target_margin(logits, target).item()),
                "target_success_at_step": int(pred == int(target.item())),
                "untargeted_success_at_step": int(pred != int(source.item())),
            }
        )
    return rows, feat_by_step


def project_targeted_trajectory(
    *,
    pc_dirs: dict,
    wrapper,
    model: str,
    attack: str,
    dataset_idx: int,
    image_ord: int,
    source_label: int,
    target_label: int,
    final_target_success: int,
    states,
    source: torch.Tensor,
    target: torch.Tensor,
    layer_groups: list[str],
    top_k: int,
):
    state_rows, feat_by_step = state_features(wrapper, states, source, target)
    out = []
    n_steps = max(len(states) - 1, 1)
    for layer_group in layer_groups:
        layer = LAYER_GROUPS[layer_group][model]
        h0 = feat_by_step[0].get(layer)
        if h0 is None:
            continue
        axes = [pc_dirs[(model, layer, pc)] for pc in range(1, top_k + 1) if (model, layer, pc) in pc_dirs]
        if not axes:
            continue
        basis = np.stack(axes).astype(np.float32)
        for meta in state_rows:
            step = int(meta["step"])
            ht = feat_by_step[step].get(layer)
            if ht is None:
                continue
            delta = ht - h0
            coeff = basis @ delta
            transport_energy = float(np.sum(coeff * coeff))
            total_energy = float(np.sum(delta * delta))
            row = {
                "model": model,
                "attack": attack,
                "dataset_idx": int(dataset_idx),
                "image_ord": int(image_ord),
                "source_label": int(source_label),
                "target_label": int(target_label),
                "layer_group": layer_group,
                "layer": layer,
                "step": step,
                "normalized_progress": float(step / n_steps),
                "time_bin": min(4, int(np.floor((step / n_steps) * 5.0))) if step < n_steps else 4,
                "final_target_success": int(final_target_success),
                "target_success_at_step": int(meta["target_success_at_step"]),
                "untargeted_success_at_step": int(meta["untargeted_success_at_step"]),
                "pred": int(meta["pred"]),
                "source_prob": float(meta["source_prob"]),
                "target_prob": float(meta["target_prob"]),
                "source_margin": float(meta["source_margin"]),
                "target_margin": float(meta["target_margin"]),
                "transport_energy_topk": transport_energy,
                "total_feature_energy": total_energy,
                "frac_energy_topk": float(transport_energy / max(total_energy, 1e-12)),
            }
            for i, c in enumerate(coeff, start=1):
                row[f"pc{i}_coeff"] = float(c)
                row[f"pc{i}_abs_coeff"] = float(abs(c))
            out.append(row)
    return out


def summarize(df: pd.DataFrame, out_dir: Path, seed: int):
    summary = df.groupby(
        ["model", "attack", "layer_group", "layer", "final_target_success", "time_bin"], dropna=False
    ).agg(
        n=("frac_energy_topk", "size"),
        mean_frac_energy=("frac_energy_topk", "mean"),
        median_frac_energy=("frac_energy_topk", "median"),
        mean_target_margin=("target_margin", "mean"),
        mean_target_prob=("target_prob", "mean"),
        mean_source_prob=("source_prob", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "targeted_attack_axis_energy_summary.csv", index=False)

    final = df.sort_values("step").groupby(["model", "attack", "dataset_idx", "layer_group"], as_index=False).tail(1)
    comp = final.groupby(["model", "attack", "layer_group", "layer"], dropna=False).agg(
        n=("final_target_success", "size"),
        targeted_asr=("final_target_success", "mean"),
        mean_final_frac_energy=("frac_energy_topk", "mean"),
        median_final_frac_energy=("frac_energy_topk", "median"),
        mean_final_target_margin=("target_margin", "mean"),
        mean_final_target_prob=("target_prob", "mean"),
    ).reset_index()
    comp.to_csv(out_dir / "targeted_attack_axis_by_attack_comparison.csv", index=False)

    pc_cols = [c for c in df.columns if c.endswith("_coeff") or c.endswith("_abs_coeff")]
    rows = []
    for (model, attack, layer_group, layer, time_bin), g in df[df.step > 0].groupby(
        ["model", "attack", "layer_group", "layer", "time_bin"]
    ):
        per_img = g.sort_values("step").groupby("dataset_idx", as_index=False).tail(1)
        y = per_img["final_target_success"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2 or len(per_img) < 20:
            rows.append(
                {
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
                }
            )
            continue
        auroc = float(roc_auc_score(y, per_img["frac_energy_topk"].to_numpy(dtype=float)))
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
        rows.append(
            {
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
            }
        )
    pred = pd.DataFrame(rows)
    pred.to_csv(out_dir / "targeted_attack_axis_success_prediction.csv", index=False)
    plot_summary(summary, out_dir)
    return comp, pred


def plot_summary(summary: pd.DataFrame, out_dir: Path):
    models = list(dict.fromkeys(summary.model))
    layer_groups = ["hidden", "penultimate", "logits"]
    fig, axes = plt.subplots(len(models), len(layer_groups), figsize=(15, 3.4 * len(models)), sharex=True, constrained_layout=True)
    if len(models) == 1:
        axes = np.expand_dims(axes, 0)
    for r, model in enumerate(models):
        for c, layer_group in enumerate(layer_groups):
            ax = axes[r, c]
            sub = summary[(summary.model == model) & (summary.layer_group == layer_group)]
            for (attack, success), g in sub.groupby(["attack", "final_target_success"]):
                g = g.sort_values("time_bin")
                ax.plot(g.time_bin, g.mean_frac_energy, marker="o", label=f"{attack} {'succ' if success else 'fail'}")
            ax.set_title(f"{model} {layer_group}")
            ax.set_xlabel("time bin")
            ax.set_ylabel("mean top-k energy fraction")
            ax.grid(alpha=0.25)
            if r == 0 and c == len(layer_groups) - 1:
                ax.legend(fontsize=7)
    fig.savefig(out_dir / "targeted_attack_axis_timeseries.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_partial(rows, out_dir: Path):
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "partial_targeted_attack_axis_projection_timeseries.csv", index=False)


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
    pc_meta.to_csv(out_dir / "targeted_attack_axis_transport_axes.csv", index=False)

    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    rows = []
    completed = set()
    partial = out_dir / "partial_targeted_attack_axis_projection_timeseries.csv"
    final = out_dir / "targeted_attack_axis_projection_timeseries.csv"
    resume_path = final if final.exists() else partial
    if args.resume and resume_path.exists():
        old = pd.read_csv(resume_path)
        rows = old.to_dict("records")
        for r in old[["model", "attack", "dataset_idx"]].drop_duplicates().itertuples():
            completed.add((r.model, r.attack, int(r.dataset_idx)))
        print(f"[RESUME] rows={len(rows)} completed={len(completed)}", flush=True)

    eps = args.eps / 255.0
    step_size = args.pgd_step_size / 255.0 if args.pgd_step_size > 0 else eps / max(args.pgd_steps, 1)
    selected_counts = {}
    for model_name in args.models:
        wrapper = load_model(model_name, device)
        select_args = argparse.Namespace(**{**vars(args), "models": [model_name], "images": args.images})
        selected = select_clean_correct(dataset, {model_name: wrapper}, select_args, device)
        selected_counts[model_name] = len(selected)
        print(f"[MODEL] {model_name} images={len(selected)} attacks={attacks}", flush=True)
        for image_ord, (dataset_idx, label) in enumerate(selected):
            x_cpu, _ = dataset[dataset_idx]
            x = x_cpu.unsqueeze(0).to(device)
            source = torch.tensor([label], device=device)
            target_label = (int(label) + args.target_offset) % 10
            if target_label == int(label):
                target_label = (target_label + 1) % 10
            target = torch.tensor([target_label], device=device)
            for attack in attacks:
                key = (model_name, attack, int(dataset_idx))
                if key in completed:
                    continue
                if attack == "targeted_pgd":
                    states = targeted_pgd_trajectory(wrapper, x, target, eps, args.pgd_steps, step_size)
                elif attack == "targeted_mi_fgsm":
                    states = targeted_mi_fgsm_trajectory(wrapper, x, target, eps, args.pgd_steps, step_size, args.momentum_decay)
                elif attack == "targeted_square":
                    states = targeted_square_trajectory(
                        wrapper,
                        x,
                        target,
                        eps,
                        args.square_steps,
                        args.seed + image_ord * 1009 + len(model_name) * 17,
                        args.square_p_init,
                        args.saved_checkpoints,
                    )
                else:
                    raise ValueError(f"Unknown attack: {attack}")
                with torch.no_grad():
                    pred = int(wrapper(states[-1]).argmax(1).item())
                final_target_success = int(pred == int(target.item()))
                rows.extend(
                    project_targeted_trajectory(
                        pc_dirs=pc_dirs,
                        wrapper=wrapper,
                        model=model_name,
                        attack=attack,
                        dataset_idx=int(dataset_idx),
                        image_ord=int(image_ord),
                        source_label=int(label),
                        target_label=int(target_label),
                        final_target_success=final_target_success,
                        states=states,
                        source=source,
                        target=target,
                        layer_groups=layer_groups,
                        top_k=args.top_k,
                    )
                )
                completed.add(key)
            if (image_ord + 1) % args.checkpoint_every == 0:
                write_partial(rows, out_dir)
                print(f"  {model_name}: {image_ord + 1}/{len(selected)} rows={len(rows)}", flush=True)
        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "targeted_attack_axis_projection_timeseries.csv", index=False)
    comp, pred = summarize(df, out_dir, args.seed)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "selected_counts": selected_counts,
                "eps": eps,
                "pgd_step_size": step_size,
                "n_rows": int(len(df)),
                "note": "Target labels use deterministic (source+target_offset) mod 10.",
            },
            f,
            indent=2,
        )
    print(f"[SAVED] {out_dir}", flush=True)
    print(comp.to_string(index=False), flush=True)
    print(pred.sort_values("auroc_frac_energy", ascending=False).head(12).to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layerwise-dir", default="analysis_outputs/pure_af_geometry/cifar_layerwise_success_flow_c10_s3_g120")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_targeted_optimizer_transport")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--models", default="bbb_resnet50,bbb_vgg19_bn")
    p.add_argument("--attacks", default="targeted_pgd,targeted_mi_fgsm,targeted_square")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--pgd-steps", type=int, default=40)
    p.add_argument("--pgd-step-size", type=float, default=0.0, help="Step size in /255 units; <=0 uses eps/steps.")
    p.add_argument("--square-steps", type=int, default=1000)
    p.add_argument("--square-p-init", type=float, default=0.3)
    p.add_argument("--momentum-decay", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--target-offset", type=int, default=1)
    p.add_argument("--layer-groups", default="hidden,penultimate,logits")
    p.add_argument("--saved-checkpoints", type=int, default=41)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
