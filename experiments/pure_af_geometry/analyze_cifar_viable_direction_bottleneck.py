#!/usr/bin/env python3
"""Estimate whether successful adversarial directions form a bottleneck.

For each training checkpoint, sample many random L_inf perturbation directions
around clean-correct CIFAR-10 images. Measure:

* viable direction fraction: how often a random direction is already adversarial;
* representation concentration of the successful random directions;
* correlation between viable-direction fraction and previously measured
  success-flow concentration.

This is a no-gradient diagnostic. It consumes saved training-dynamics
checkpoints and does not require rerunning PGD/Square trajectories.
"""

from __future__ import annotations

import argparse
import json
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
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.run_cifar_training_dynamics_transport import (  # noqa: E402
    FeatureWrapper,
    LAYERS,
    dim_stats,
    load_checkpoint_model,
    margin,
    normalize_rows,
    pca_basis,
    parse_csv,
    select_clean_correct,
)


def acc_stage(acc: float) -> str:
    if acc < 0.15:
        return "init_or_random"
    if acc < 0.40:
        return "early"
    if acc < 0.70:
        return "middle"
    if acc < 0.85:
        return "late"
    return "mature"


def build_eval_dataset(root: str):
    return datasets.CIFAR10(root, train=False, download=False, transform=transforms.ToTensor())


def load_manifest(args) -> pd.DataFrame:
    manifest_path = Path(args.training_output_dir) / "checkpoint_manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
    else:
        rows = []
        for p in sorted(Path(args.checkpoint_dir).glob("seed*/resnet18_seed*.pt")):
            if p.name.endswith("latest.pt"):
                continue
            state = torch.load(p, map_location="cpu")
            rows.append(
                {
                    "seed": int(state.get("seed", -1)),
                    "epoch": int(state.get("epoch", -1)),
                    "acc": float(state.get("acc", np.nan)),
                    "tag": str(state.get("tag", p.stem.split("_")[-1])),
                    "path": str(p),
                }
            )
        df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No checkpoint manifest rows found.")
    include = set(parse_csv(args.include_tags)) if args.include_tags.strip() else set()
    if include:
        df = df[df["tag"].isin(include)].copy()
    seeds = set(parse_csv(args.model_seeds, int)) if args.model_seeds.strip() else set()
    if seeds:
        df = df[df["seed"].isin(seeds)].copy()
    if df.empty:
        raise RuntimeError("No checkpoints remain after filtering.")
    return df.sort_values(["seed", "epoch", "tag"]).reset_index(drop=True)


def reservoir_add(store: list[np.ndarray], arr: np.ndarray, max_items: int, rng: np.random.Generator):
    if max_items <= 0:
        return
    for row in arr:
        if len(store) < max_items:
            store.append(row.astype(np.float32, copy=False))
        else:
            j = int(rng.integers(0, len(store) + 1))
            if j < max_items:
                store[j] = row.astype(np.float32, copy=False)


@torch.no_grad()
def clean_features(wrapper: FeatureWrapper, x: torch.Tensor, y: int):
    logits, feats = wrapper.forward_features_nograd(x)
    pred = int(logits.argmax(1).item())
    if pred != int(y):
        return None
    py = float(F.softmax(logits, 1)[0, int(y)].item())
    m = float(margin(logits, torch.tensor([int(y)], device=x.device)).item())
    return pred, py, m, {k: v[0].astype(np.float32) for k, v in feats.items()}


def sample_checkpoint(args, ckpt_row: pd.Series, dataset, device: torch.device):
    out = Path(args.output_dir)
    shard = out / "random_direction_shards" / f"seed{int(ckpt_row.seed)}" / str(ckpt_row.tag)
    shard.mkdir(parents=True, exist_ok=True)
    meta_path = shard / "viable_direction_metadata.csv"
    vec_path = shard / "viable_direction_vectors.npz"
    cfg_path = shard / "config.json"
    if meta_path.exists() and vec_path.exists() and not args.recompute:
        print(f"[SKIP] seed={ckpt_row.seed} tag={ckpt_row.tag}", flush=True)
        return

    wrapper = load_checkpoint_model(str(ckpt_row.path), device)
    selected = select_clean_correct(wrapper, dataset, args, device)
    eps = args.eps / 255.0
    meta_rows = []
    vec_store = {(layer, outcome): [] for layer in LAYERS for outcome in ["success", "failed"]}
    rng = np.random.default_rng(args.seed + int(ckpt_row.seed) * 100003 + int(ckpt_row.epoch) * 1009)

    for image_ord, (idx, label) in enumerate(selected):
        x_cpu, _ = dataset[idx]
        x0 = x_cpu.unsqueeze(0).to(device)
        base = clean_features(wrapper, x0, label)
        if base is None:
            continue
        _pred0, py0, margin0, feats0 = base
        remaining = args.directions_per_image
        dir_start = 0
        while remaining > 0:
            bs = min(args.direction_batch_size, remaining)
            gen = torch.Generator(device=device).manual_seed(
                args.seed + int(ckpt_row.seed) * 1000003 + idx * 9176 + dir_start
            )
            signs = torch.where(
                torch.rand((bs,) + tuple(x0.shape[1:]), generator=gen, device=device) < 0.5,
                -torch.ones((bs,) + tuple(x0.shape[1:]), device=device),
                torch.ones((bs,) + tuple(x0.shape[1:]), device=device),
            )
            x_adv = (x0 + eps * signs).clamp(0, 1)
            logits, feats = wrapper.forward_features_nograd(x_adv)
            pred = logits.argmax(1)
            y = torch.full((bs,), int(label), device=device, dtype=torch.long)
            success = (pred != y).detach().cpu().numpy().astype(bool)
            py = F.softmax(logits, 1)[:, int(label)].detach().cpu().numpy()
            m = margin(logits, y).detach().cpu().numpy()
            for j in range(bs):
                dir_idx = dir_start + j
                meta_rows.append(
                    {
                        "seed": int(ckpt_row.seed),
                        "tag": str(ckpt_row.tag),
                        "epoch": int(ckpt_row.epoch),
                        "checkpoint_acc": float(ckpt_row.acc),
                        "stage": acc_stage(float(ckpt_row.acc)),
                        "image_ord": image_ord,
                        "dataset_idx": idx,
                        "label": int(label),
                        "direction_idx": dir_idx,
                        "direction_seed": int(args.seed + int(ckpt_row.seed) * 1000003 + idx * 9176 + dir_start),
                        "pred": int(pred[j].item()),
                        "success": int(success[j]),
                        "start_margin": margin0,
                        "random_margin": float(m[j]),
                        "margin_drop": float(margin0 - m[j]),
                        "start_p_y": py0,
                        "random_p_y": float(py[j]),
                        "p_y_drop": float(py0 - py[j]),
                    }
                )
            for layer, h in feats.items():
                if layer not in feats0:
                    continue
                disp = h.astype(np.float32) - feats0[layer][None, :]
                reservoir_add(vec_store[(layer, "success")], disp[success], args.max_store_per_outcome, rng)
                reservoir_add(vec_store[(layer, "failed")], disp[~success], args.max_store_per_outcome, rng)
            remaining -= bs
            dir_start += bs

    packed = {}
    for (layer, outcome), vals in vec_store.items():
        if vals:
            packed[f"{outcome}__{layer}"] = np.stack(vals).astype(np.float32)
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
    np.savez_compressed(vec_path, **packed)
    cfg_path.write_text(
        json.dumps(
            {
                "checkpoint": str(ckpt_row.path),
                "seed": int(ckpt_row.seed),
                "tag": str(ckpt_row.tag),
                "epoch": int(ckpt_row.epoch),
                "acc": float(ckpt_row.acc),
                "eps": args.eps,
                "images": args.images,
                "directions_per_image": args.directions_per_image,
                "max_store_per_outcome": args.max_store_per_outcome,
            },
            indent=2,
        )
        + "\n"
    )
    wrapper.close()
    del wrapper
    torch.cuda.empty_cache()
    print(f"[SAVED] {meta_path}", flush=True)


def analyze_outputs(args):
    out = Path(args.output_dir)
    summary_rows = []
    dim_rows = []
    for meta_path in sorted((out / "random_direction_shards").glob("seed*/*/viable_direction_metadata.csv")):
        vec_path = meta_path.with_name("viable_direction_vectors.npz")
        meta = pd.read_csv(meta_path)
        if meta.empty or not vec_path.exists():
            continue
        ck = meta.iloc[0]
        summary_rows.append(
            {
                "seed": int(ck.seed),
                "tag": str(ck.tag),
                "epoch": int(ck.epoch),
                "checkpoint_acc": float(ck.checkpoint_acc),
                "stage": str(ck.stage),
                "n_images": int(meta.dataset_idx.nunique()),
                "n_directions": int(len(meta)),
                "viable_fraction": float(meta.success.mean()),
                "mean_start_margin": float(meta.groupby("dataset_idx").start_margin.first().mean()),
                "mean_random_margin": float(meta.random_margin.mean()),
                "mean_margin_drop": float(meta.margin_drop.mean()),
                "mean_p_y_drop": float(meta.p_y_drop.mean()),
            }
        )
        npz = np.load(vec_path, allow_pickle=False)
        for layer in LAYERS:
            for outcome in ["success", "failed"]:
                key = f"{outcome}__{layer}"
                if key not in npz.files:
                    continue
                x = normalize_rows(npz[key].astype(np.float32))
                if len(x) < args.min_vectors_for_dim:
                    continue
                _mean, _basis, ratio = pca_basis(x, min(args.max_k, len(x), x.shape[1]))
                stats = dim_stats(ratio)
                dim_rows.append(
                    {
                        "seed": int(ck.seed),
                        "tag": str(ck.tag),
                        "epoch": int(ck.epoch),
                        "checkpoint_acc": float(ck.checkpoint_acc),
                        "stage": str(ck.stage),
                        "layer": layer,
                        "outcome": outcome,
                        "n_vectors": int(len(x)),
                        **stats,
                    }
                )
    summary = pd.DataFrame(summary_rows).sort_values(["seed", "epoch", "tag"])
    dims = pd.DataFrame(dim_rows).sort_values(["seed", "epoch", "tag", "layer", "outcome"])
    summary.to_csv(out / "viable_direction_summary.csv", index=False)
    dims.to_csv(out / "viable_direction_dimensionality.csv", index=False)
    corr = correlate_with_flow(args, summary, dims)
    corr.to_csv(out / "viable_direction_flow_correlations.csv", index=False)
    make_plots(out, summary, dims)
    print(f"[ANALYSIS] wrote {out}", flush=True)


def correlate_with_flow(args, summary: pd.DataFrame, dims: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    formation_dir = Path(args.formation_dir)
    rows = []
    # Random successful direction concentration vs viable fraction.
    if not dims.empty:
        for layer_group, dd in [
            ("all_hidden_pooled", dims[(dims.layer != "logits") & (dims.outcome == "success")]),
            ("all_layers", dims[dims.outcome == "success"]),
        ]:
            agg = dd.groupby(["seed", "tag", "epoch", "checkpoint_acc", "stage"], as_index=False).agg(
                success_random_dim80=("dim80", "mean"),
                success_random_pc5=("pc5_cum_var", "mean"),
                success_random_erank=("effective_rank", "mean"),
            )
            merged = agg.merge(summary, on=["seed", "tag", "epoch", "checkpoint_acc", "stage"], how="inner")
            for metric, sign in [("success_random_dim80", -1), ("success_random_pc5", 1), ("success_random_erank", -1)]:
                rows.extend(corr_rows(merged, layer_group, f"random_viable_{metric}", sign * merged[metric], merged))
    # Existing PGD/Square success-flow concentration vs viable fraction.
    fdim_path = formation_dir / "formation_dimensionality.csv"
    if fdim_path.exists():
        fdim = pd.read_csv(fdim_path)
        for attack in sorted(fdim.attack.unique()):
            dd = fdim[(fdim.attack == attack) & (fdim.layer != "logits")]
            agg = dd.groupby(["seed", "tag", "epoch", "checkpoint_acc", "stage"], as_index=False).agg(
                flow_dim80=("dim80", "mean"),
                flow_pc5=("pc5_cum_var", "mean"),
                flow_erank=("effective_rank", "mean"),
            )
            merged = agg.merge(summary, on=["seed", "tag", "epoch", "checkpoint_acc", "stage"], how="inner")
            for metric, sign in [("flow_dim80", -1), ("flow_pc5", 1), ("flow_erank", -1)]:
                rows.extend(corr_rows(merged, f"{attack}_success_flow_hidden_pooled", metric, sign * merged[metric], merged))
    return pd.DataFrame(rows)


def corr_rows(df: pd.DataFrame, group: str, metric: str, signed_strength: pd.Series, merged: pd.DataFrame):
    rows = []
    tmp = merged.copy()
    tmp["signed_strength"] = signed_strength
    for target in ["viable_fraction", "mean_start_margin", "mean_margin_drop", "mean_p_y_drop"]:
        gg = tmp[["signed_strength", target, "checkpoint_acc"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(gg) < 5 or gg.signed_strength.nunique() < 2 or gg[target].nunique() < 2:
            continue
        resid = np.nan
        if gg.checkpoint_acc.nunique() > 2:
            sx = gg.signed_strength - np.polyval(np.polyfit(gg.checkpoint_acc, gg.signed_strength, 1), gg.checkpoint_acc)
            sy = gg[target] - np.polyval(np.polyfit(gg.checkpoint_acc, gg[target], 1), gg.checkpoint_acc)
            resid = sx.corr(sy, method="pearson")
        rows.append(
            {
                "flow_group": group,
                "flow_metric": metric,
                "target": target,
                "n": int(len(gg)),
                "pearson": float(gg.signed_strength.corr(gg[target], method="pearson")),
                "spearman": float(gg.signed_strength.corr(gg[target], method="spearman")),
                "pearson_resid_acc": float(resid) if not pd.isna(resid) else np.nan,
            }
        )
    return rows


def make_plots(out: Path, summary: pd.DataFrame, dims: pd.DataFrame):
    if summary.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    for seed, g in summary.groupby("seed"):
        g = g.sort_values("checkpoint_acc")
        axes[0].plot(g.checkpoint_acc, g.viable_fraction, marker="o", label=f"seed {seed}")
        axes[1].plot(g.checkpoint_acc, g.mean_start_margin, marker="o", label=f"seed {seed}")
    axes[0].set_title("Viable random-direction fraction")
    axes[0].set_xlabel("checkpoint clean accuracy")
    axes[0].set_ylabel("fraction adversarial")
    axes[1].set_title("Mean clean margin")
    axes[1].set_xlabel("checkpoint clean accuracy")
    axes[1].set_ylabel("margin")
    if not dims.empty:
        agg = (
            dims[(dims.outcome == "success") & (dims.layer != "logits")]
            .groupby(["seed", "tag", "checkpoint_acc"], as_index=False)
            .agg(pc5=("pc5_cum_var", "mean"))
        )
        merged = agg.merge(summary[["seed", "tag", "checkpoint_acc", "viable_fraction"]], on=["seed", "tag", "checkpoint_acc"])
        sc = axes[2].scatter(merged.viable_fraction, merged.pc5, c=merged.checkpoint_acc, cmap="viridis", s=45)
        axes[2].set_title("Viability vs concentration")
        axes[2].set_xlabel("viable fraction")
        axes[2].set_ylabel("successful-random PC5 variance")
        fig.colorbar(sc, ax=axes[2], label="clean accuracy")
    for ax in axes:
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out / "viable_direction_bottleneck_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-output-dir", default="analysis_outputs/pure_af_geometry/cifar_training_dynamics_transport_v1")
    p.add_argument("--formation-dir", default="analysis_outputs/pure_af_geometry/cifar_training_dynamics_transport_v1/formation_analysis")
    p.add_argument("--checkpoint-dir", default="checkpoints/cifar10_resnet18_training_dynamics_v1")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_viable_direction_bottleneck_v1")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model-seeds", default="0,1,2")
    p.add_argument("--include-tags", default="init,acc15,acc25,acc40,acc55,acc70,acc82,acc90,final")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--directions-per-image", type=int, default=128)
    p.add_argument("--direction-batch-size", type=int, default=128)
    p.add_argument("--max-store-per-outcome", type=int, default=4000)
    p.add_argument("--min-vectors-for-dim", type=int, default=50)
    p.add_argument("--max-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=404)
    p.add_argument("--stage", choices=["sample", "analyze", "all"], default="all")
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.stage in {"sample", "all"}:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = build_eval_dataset(args.dataset_root)
        manifest = load_manifest(args)
        manifest.to_csv(Path(args.output_dir) / "checkpoint_manifest_used.csv", index=False)
        for row in manifest.itertuples(index=False):
            sample_checkpoint(args, row, dataset, device)
    if args.stage in {"analyze", "all"}:
        analyze_outputs(args)
    (Path(args.output_dir) / "metadata.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()

