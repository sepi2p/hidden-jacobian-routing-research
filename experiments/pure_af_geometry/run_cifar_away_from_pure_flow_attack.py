#!/usr/bin/env python3
"""CIFAR-10 away-from-pure-flow attack sweep.

The pure-flow directions are class-wise vectors estimated from successful
random-noise-to-pure GA trajectories. Attacks pull the negative feature-space
direction back to pixels and step inside an L_inf ball.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import (  # noqa: E402
    get_npz,
    load_model,
    normalize_rows,
)


MODELS = ["bbb_resnet50", "bbb_vgg19_bn", "bbb_densenet", "bbb_inception_v3"]
DOMINANT = {
    "bbb_resnet50": "layer2",
    "bbb_vgg19_bn": "block2",
    "bbb_densenet": "denseblock3",
    "bbb_inception_v3": "mixed6",
}
PENULTIMATE = {
    "bbb_resnet50": "avgpool",
    "bbb_vgg19_bn": "penultimate",
    "bbb_densenet": "penultimate",
    "bbb_inception_v3": "penultimate",
}
VARIANT_LAYER = {
    "away_hidden": DOMINANT,
    "away_penultimate": PENULTIMATE,
    "away_logits": {m: "logits" for m in MODELS},
}


def segment_key(row) -> tuple:
    seed = int(str(row.run_id).rsplit("_seed", 1)[-1])
    return int(row.target_class), seed, str(row.run_id)


def build_mu(layerwise_dir: Path) -> dict[tuple[str, str, int], np.ndarray]:
    seg = pd.read_csv(layerwise_dir / "segments.csv")
    vec_npz = np.load(layerwise_dir / "segment_vectors.npz")
    out = {}
    needed_layers = set()
    for mapping in VARIANT_LAYER.values():
        needed_layers.update(mapping.values())
    for model in MODELS:
        for layer in needed_layers:
            if layer not in set(seg.loc[seg.model == model, "layer"]):
                continue
            group = seg[(seg.model == model) & (seg.layer == layer) & (seg.success == 1)].reset_index(drop=True)
            if group.empty:
                continue
            arr = get_npz(vec_npz, "vectors", model, layer)
            per_run = defaultdict(list)
            for r in group.itertuples():
                per_run[segment_key(r)].append(arr[int(r.vector_idx)])
            by_class = defaultdict(list)
            for (target, _seed, _run_id), vals in per_run.items():
                v = np.sum(np.stack(vals), axis=0)
                if np.linalg.norm(v) > 1e-12:
                    by_class[target].append(v)
            for target, vals in by_class.items():
                mu = np.mean(normalize_rows(np.stack(vals)), axis=0)
                mu = mu / np.clip(np.linalg.norm(mu), 1e-12, None)
                out[(model, layer, int(target))] = mu.astype(np.float32)
    return out


def project_linf(x_adv: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)


def margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.view(-1, 1), -1e9)
    return true - masked.max(1).values


def select_clean_correct(dataset, wrappers, args, device):
    selected = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        ok = True
        with torch.no_grad():
            for model in args.models:
                if int(wrappers[model](x).argmax(1).item()) != int(y.item()):
                    ok = False
                    break
        if ok:
            selected.append((idx, int(y.item())))
        if len(selected) >= args.images:
            break
    return selected


def feature_pixel_grad(wrapper, x_adv, y, layer, direction_np):
    x_probe = x_adv.detach().requires_grad_(True)
    logits, feats, raw = wrapper.forward_with_features(x_probe)
    h = feats[layer]
    direction = torch.as_tensor(direction_np, dtype=h.dtype, device=h.device).view_as(h)
    scalar = (h * direction).sum()
    grad_x = torch.autograd.grad(scalar, x_probe, retain_graph=True)[0]
    logp = F.log_softmax(logits, dim=1)[0, int(y.item())]
    raw_tensors = [r for label in wrapper.labels for r in raw.get(label, [])]
    raw_grads = torch.autograd.grad(logp, raw_tensors, retain_graph=False, allow_unused=True)
    grad_feats = wrapper.aggregate_grads(raw, list(raw_grads))
    gh = grad_feats[layer]
    gh_np = gh.detach().cpu().numpy()[0]
    local_cos = float(np.dot(normalize_rows(gh_np[None, :])[0], normalize_rows(direction_np[None, :])[0]))
    return grad_x, logits.detach(), local_cos


def ce_pixel_grad(wrapper, x_adv, y):
    x_probe = x_adv.detach().requires_grad_(True)
    logits = wrapper(x_probe)
    loss = F.cross_entropy(logits, y)
    grad_x = torch.autograd.grad(loss, x_probe)[0]
    return grad_x, logits.detach()


def random_direction(shape, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=gen, device=device)


def attack_one(wrapper, x, y, variant, layer, mu_np, eps, steps, step_size, seed, device):
    x0 = x.detach()
    x_adv = x0.clone()
    ce_cosines = []
    local_cosines = []
    if variant == "random_direction":
        rnd = random_direction(x.shape, seed, device)
    for _step in range(steps):
        if variant == "ce_pgd":
            grad, _ = ce_pixel_grad(wrapper, x_adv, y)
            direction = grad
            local_cosines.append(np.nan)
        elif variant == "random_direction":
            direction = rnd
            local_cosines.append(np.nan)
        else:
            direction_np = -mu_np
            direction, _logits, local_cos = feature_pixel_grad(wrapper, x_adv, y, layer, direction_np)
            local_cosines.append(local_cos)
        ce_grad, _ = ce_pixel_grad(wrapper, x_adv, y)
        ce_cos = torch.sum(
            F.normalize(direction.flatten(1), dim=1) * F.normalize(ce_grad.flatten(1), dim=1),
            dim=1,
        )
        ce_cosines.append(float(ce_cos.item()))
        x_adv = project_linf(x_adv + step_size * direction.sign(), x0, eps)
    return x_adv.detach(), float(np.nanmean(ce_cosines)), float(np.nanmean(local_cosines))


def eval_all(wrappers, x_adv, y):
    out = {}
    with torch.no_grad():
        for model, wrapper in wrappers.items():
            logits = wrapper(x_adv)
            probs = F.softmax(logits, dim=1)
            pred = int(logits.argmax(1).item())
            out[model] = {
                "pred": pred,
                "success": int(pred != int(y.item())),
                "true_prob": float(probs[0, int(y.item())].item()),
                "margin": float(margin(logits, y).item()),
            }
    return out


def write_outputs(df: pd.DataFrame, out_dir: Path, final: bool):
    prefix = "" if final else "partial_"
    if df.empty:
        return pd.DataFrame()
    df.to_csv(out_dir / f"{prefix}cifar_away_flow_attack_per_image.csv", index=False)
    source = df[df.source_model == df.target_model]
    summary = source.groupby(["source_model", "variant", "layer", "eps_255", "steps"], dropna=False).agg(
        asr=("target_success", "mean"),
        n=("target_success", "size"),
        mean_margin=("target_margin", "mean"),
        mean_true_prob=("target_true_prob", "mean"),
        mean_ce_grad_cosine=("ce_grad_cosine", "mean"),
        mean_feature_logp_grad_cosine=("feature_logp_grad_cosine", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / f"{prefix}cifar_away_flow_attack_summary.csv", index=False)
    summary.to_csv(out_dir / f"{prefix}cifar_away_flow_step_sweep.csv", index=False)
    df.groupby(["source_model", "variant", "layer", "eps_255", "steps"], dropna=False).agg(
        mean_ce_grad_cosine=("ce_grad_cosine", "mean"),
        mean_feature_logp_grad_cosine=("feature_logp_grad_cosine", "mean"),
        n=("ce_grad_cosine", "size"),
    ).reset_index().to_csv(out_dir / f"{prefix}cifar_away_flow_gradient_alignment.csv", index=False)
    transfer = df.groupby(["source_model", "target_model", "variant", "layer", "eps_255", "steps"], dropna=False).agg(
        asr=("target_success", "mean"),
        n=("target_success", "size"),
    ).reset_index()
    transfer.to_csv(out_dir / f"{prefix}cifar_away_flow_transfer_matrix.csv", index=False)
    if final:
        plot_summary(summary, out_dir)
    return summary


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    eps_values = [float(x) / 255.0 for x in args.eps.split(",")]
    steps_values = [int(x) for x in args.steps.split(",")]
    wrappers = {m: load_model(m, device) for m in args.models}
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    selected = select_clean_correct(dataset, wrappers, args, device)
    mu = build_mu(Path(args.layerwise_dir))
    rows = []
    completed = set()
    partial_path = out_dir / "partial_cifar_away_flow_attack_per_image.csv"
    final_path = out_dir / "cifar_away_flow_attack_per_image.csv"
    resume_path = final_path if final_path.exists() else partial_path
    if args.resume and resume_path.exists():
        old = pd.read_csv(resume_path)
        rows = old.to_dict("records")
        source_done = old[old.source_model == old.target_model]
        for r in source_done[["source_model", "dataset_idx", "eps_255", "steps", "variant"]].drop_duplicates().itertuples():
            completed.add((r.source_model, int(r.dataset_idx), float(r.eps_255), int(r.steps), r.variant))
        print(f"[RESUME] loaded {len(rows)} rows; completed source evaluations={len(completed)}", flush=True)
    for source_model in args.models:
        wrapper = wrappers[source_model]
        print(f"[MODEL] {source_model} n_images={len(selected)}", flush=True)
        for image_ord, (dataset_idx, label) in enumerate(selected):
            x_cpu, _ = dataset[dataset_idx]
            x = x_cpu.unsqueeze(0).to(device)
            y = torch.tensor([label], device=device)
            clean_eval = eval_all(wrappers, x, y)
            for eps in eps_values:
                for steps in steps_values:
                    step_size = eps / max(steps, 1)
                    for variant in ["away_hidden", "away_penultimate", "away_logits", "random_direction", "ce_pgd"]:
                        done_key = (source_model, int(dataset_idx), float(eps * 255), int(steps), variant)
                        if done_key in completed:
                            continue
                        layer = VARIANT_LAYER.get(variant, {}).get(source_model, "")
                        mu_np = mu.get((source_model, layer, label)) if layer else None
                        if variant.startswith("away") and mu_np is None:
                            continue
                        adv, ce_cos, local_cos = attack_one(
                            wrapper, x, y, variant, layer, mu_np, eps, steps, step_size,
                            args.seed + image_ord * 1000 + steps * 10 + int(eps * 255), device,
                        )
                        evals = eval_all(wrappers, adv, y)
                        src = evals[source_model]
                        for target_model, ev in evals.items():
                            rows.append({
                                "source_model": source_model,
                                "target_model": target_model,
                                "dataset_idx": int(dataset_idx),
                                "image_ord": int(image_ord),
                                "label": int(label),
                                "variant": variant,
                                "layer": layer,
                                "eps": float(eps),
                                "eps_255": float(eps * 255),
                                "steps": int(steps),
                                "step_size": float(step_size),
                                "source_success": int(src["success"]),
                                "target_success": int(ev["success"]),
                                "target_pred": int(ev["pred"]),
                                "target_margin": float(ev["margin"]),
                                "target_true_prob": float(ev["true_prob"]),
                                "clean_target_margin": float(clean_eval[target_model]["margin"]),
                                "clean_target_true_prob": float(clean_eval[target_model]["true_prob"]),
                                "ce_grad_cosine": ce_cos,
                                "feature_logp_grad_cosine": local_cos,
                            })
                        completed.add(done_key)
            if (image_ord + 1) % args.checkpoint_every == 0:
                write_outputs(pd.DataFrame(rows), out_dir, final=False)
                print(f"  {source_model}: {image_ord + 1}/{len(selected)} checkpoint rows={len(rows)}", flush=True)
            elif (image_ord + 1) % 50 == 0:
                print(f"  {source_model}: {image_ord + 1}/{len(selected)}", flush=True)
    df = pd.DataFrame(rows)
    summary = write_outputs(df, out_dir, final=True)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"args": {**vars(args), "eps_values": eps_values, "steps_values": steps_values}, "n_images": len(selected)}, f, indent=2)
    print(f"[SAVED] {out_dir}", flush=True)
    print(summary[(summary.eps_255 == max([e * 255 for e in eps_values])) & (summary.steps == max(steps_values))].to_string(index=False))


def plot_summary(summary, out_dir: Path):
    sub = summary[summary.steps == summary.steps.max()].copy()
    eps_max = sub.eps_255.max()
    sub = sub[sub.eps_255 == eps_max]
    models = list(dict.fromkeys(sub.source_model))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, model in zip(axes, models):
        g = sub[sub.source_model == model].sort_values("asr", ascending=False)
        labels = [f"{r.variant}\\n{r.layer}" if isinstance(r.layer, str) and r.layer else r.variant for r in g.itertuples()]
        ax.bar(np.arange(len(g)), g.asr)
        ax.set_title(model)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(np.arange(len(g)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Source ASR")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"CIFAR away-from-pure flow attack, eps={eps_max:.0f}/255, steps={int(summary.steps.max())}")
    fig.savefig(out_dir / "cifar_away_flow_summary.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layerwise-dir", default="analysis_outputs/pure_af_geometry/cifar_layerwise_success_flow_c10_s3_g120")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_away_flow_attack")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--models", default="bbb_resnet50,bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--images", type=int, default=500)
    p.add_argument("--eps", default="2,4,8")
    p.add_argument("--steps", default="1,2,5,10,20")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
