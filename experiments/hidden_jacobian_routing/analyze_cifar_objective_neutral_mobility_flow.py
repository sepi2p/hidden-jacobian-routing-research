#!/usr/bin/env python3
"""Objective-neutral mobility versus adversarial success-flow overlap.

This experiment asks whether adversarial success-flow is an adversarially useful
subset of high-mobility representation directions.

The explorers ("ants") do not optimize labels, margins, CE, or target classes.
For each clean image they sample random L_inf sign directions. We then measure:

* representation mobility: ||h(x+delta)-h(x)||;
* overlap with adversarial success-flow bases learned from PGD/Square;
* post-hoc adversarial outcome, used only for analysis.

If high-mobility directions have higher success-flow energy, and high-flow
random directions are more often adversarial, this supports the interpretation
that adversarial flows are one useful subset of easy transport channels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.run_cifar_training_dynamics_transport import (  # noqa: E402
    LAYERS,
    load_checkpoint_model,
    normalize_rows,
    pca_basis,
    projection_energy,
    select_clean_correct,
    transport_vectors,
)


def parse_csv(s: str, typ=str):
    return [typ(x.strip()) for x in s.split(",") if x.strip()]


def stage(acc: float) -> str:
    if acc < 0.15:
        return "init_or_random"
    if acc < 0.40:
        return "early"
    if acc < 0.70:
        return "middle"
    if acc < 0.85:
        return "late"
    return "mature"


def load_manifest(args) -> pd.DataFrame:
    df = pd.read_csv(Path(args.training_output_dir) / "checkpoint_manifest.csv")
    include = set(parse_csv(args.include_tags)) if args.include_tags.strip() else set()
    if include:
        df = df[df["tag"].isin(include)].copy()
    seeds = set(parse_csv(args.model_seeds, int)) if args.model_seeds.strip() else set()
    if seeds:
        df = df[df["seed"].isin(seeds)].copy()
    if df.empty:
        raise RuntimeError("No checkpoints match filters.")
    return df.sort_values(["seed", "epoch", "tag"]).reset_index(drop=True)


def build_dataset(root: str):
    return datasets.CIFAR10(root, train=False, download=False, transform=transforms.ToTensor())


def load_success_flow_basis(base: Path, seed: int, tag: str, attack: str, layer: str, max_k: int):
    shard_dir = base / "trajectory_shards" / f"seed{seed}" / tag / attack
    metas = sorted(shard_dir.glob("meta_*.csv"))
    if not metas:
        return None
    meta_path = metas[0]
    npz_path = meta_path.with_name(meta_path.name.replace("meta_", "states_").replace(".csv", ".npz"))
    if not npz_path.exists():
        return None
    meta = pd.read_csv(meta_path)
    if meta.empty:
        return None
    npz = np.load(npz_path, allow_pickle=False)
    rows, x = transport_vectors(meta, npz, layer)
    if rows.empty or len(x) == 0:
        return None
    success = rows.final_success.to_numpy(int) == 1
    if success.sum() < 20:
        return None
    mean, basis, ratio = pca_basis(x[success].astype(np.float32), max_k)
    return {"mean": mean, "basis": basis, "ratio": ratio, "n_basis": int(success.sum())}


def safe_auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    if len(pos_scores) < 2 or len(neg_scores) < 2:
        return np.nan
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    s = np.r_[pos_scores, neg_scores]
    if np.nanstd(s) <= 1e-12:
        return np.nan
    return float(roc_auc_score(y, s))


def corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    sx = pd.Series(x[ok])
    sy = pd.Series(y[ok])
    if sx.nunique() < 2 or sy.nunique() < 2:
        return np.nan
    return float(sx.corr(sy, method=kind))


def bin_index(values: np.ndarray, n_bins: int):
    ranks = pd.Series(values).rank(method="first").to_numpy()
    bins = np.ceil(ranks / len(values) * n_bins).astype(int) - 1
    return np.clip(bins, 0, n_bins - 1)


def analyze_checkpoint(args, row, dataset, device):
    base = Path(args.training_output_dir)
    seed = int(row.seed)
    tag = str(row.tag)
    wrapper = load_checkpoint_model(str(row.path), device)
    selected = select_clean_correct(wrapper, dataset, args, device)
    eps = args.eps / 255.0
    max_k = max(parse_csv(args.ks, int))
    basis_cache = {}
    for attack in parse_csv(args.basis_attacks):
        for layer in LAYERS:
            basis = load_success_flow_basis(base, seed, tag, attack, layer, max_k)
            if basis is not None:
                basis_cache[(attack, layer)] = basis
    if not basis_cache:
        wrapper.close()
        return [], []

    values = {
        (basis_attack, layer): {"mobility": [], "energy": {k: [] for k in parse_csv(args.ks, int)}, "success": [], "image_ord": []}
        for basis_attack, layer in basis_cache
    }

    for image_ord, (idx, label) in enumerate(selected):
        x_cpu, _ = dataset[idx]
        x0 = x_cpu.unsqueeze(0).to(device)
        with torch.no_grad():
            logits0, feats0 = wrapper.forward_features_nograd(x0)
            if int(logits0.argmax(1).item()) != int(label):
                continue
        gen = torch.Generator(device=device).manual_seed(args.seed + seed * 1000003 + int(idx) * 7919)
        remaining = args.directions_per_image
        offset = 0
        while remaining > 0:
            bs = min(args.direction_batch_size, remaining)
            signs = torch.where(
                torch.rand((bs,) + tuple(x0.shape[1:]), generator=gen, device=device) < 0.5,
                -torch.ones((bs,) + tuple(x0.shape[1:]), device=device),
                torch.ones((bs,) + tuple(x0.shape[1:]), device=device),
            )
            xr = (x0 + eps * signs).clamp(0, 1)
            with torch.no_grad():
                logits, feats = wrapper.forward_features_nograd(xr)
            success = (logits.argmax(1).detach().cpu().numpy() != int(label)).astype(np.int8)
            for layer in LAYERS:
                if layer not in feats0 or layer not in feats:
                    continue
                disp = feats[layer].astype(np.float32) - feats0[layer][0][None, :].astype(np.float32)
                mob = np.linalg.norm(disp, axis=1).astype(np.float32)
                for basis_attack in parse_csv(args.basis_attacks):
                    key = (basis_attack, layer)
                    if key not in basis_cache:
                        continue
                    b = basis_cache[key]
                    rec = values[key]
                    rec["mobility"].append(mob)
                    rec["success"].append(success)
                    rec["image_ord"].append(np.full(len(mob), int(image_ord), dtype=np.int32))
                    for k in parse_csv(args.ks, int):
                        rec["energy"][k].append(projection_energy(disp, b["mean"], b["basis"], k).astype(np.float32))
            remaining -= bs
            offset += bs

    metric_rows = []
    bin_rows = []
    per_image_rows = []
    for (basis_attack, layer), rec in values.items():
        if not rec["mobility"]:
            continue
        mobility = np.concatenate(rec["mobility"])
        success = np.concatenate(rec["success"]).astype(bool)
        image_ids = np.concatenate(rec["image_ord"]).astype(np.int32)
        for k, chunks in rec["energy"].items():
            energy = np.concatenate(chunks)
            q20, q80 = np.quantile(mobility, [0.2, 0.8])
            low = mobility <= q20
            high = mobility >= q80
            eq20, eq80 = np.quantile(energy, [0.2, 0.8])
            low_flow = energy <= eq20
            high_flow = energy >= eq80
            metric_rows.append(
                {
                    "seed": seed,
                    "tag": tag,
                    "epoch": int(row.epoch),
                    "checkpoint_acc": float(row.acc),
                    "stage": stage(float(row.acc)),
                    "basis_attack": basis_attack,
                    "layer": layer,
                    "k": int(k),
                    "n_directions": int(len(mobility)),
                    "n_adv_random": int(success.sum()),
                    "random_asr": float(success.mean()),
                    "mobility_mean": float(np.mean(mobility)),
                    "flow_energy_mean": float(np.mean(energy)),
                    "mobility_flow_pearson": corr(mobility, energy, "pearson"),
                    "mobility_flow_spearman": corr(mobility, energy, "spearman"),
                    "low_mobility_flow_energy": float(np.mean(energy[low])),
                    "high_mobility_flow_energy": float(np.mean(energy[high])),
                    "high_vs_low_mobility_flow_auroc": safe_auroc(energy[high], energy[low]),
                    "low_flow_random_asr": float(success[low_flow].mean()) if low_flow.any() else np.nan,
                    "high_flow_random_asr": float(success[high_flow].mean()) if high_flow.any() else np.nan,
                    "adv_random_flow_energy": float(np.mean(energy[success])) if success.any() else np.nan,
                    "nonadv_random_flow_energy": float(np.mean(energy[~success])) if (~success).any() else np.nan,
                    "adv_vs_nonadv_flow_auroc": safe_auroc(energy[success], energy[~success]) if success.any() else np.nan,
                    "n_basis_segments": int(basis_cache[(basis_attack, layer)]["n_basis"]),
                }
            )
            for img in np.unique(image_ids):
                imask = image_ids == img
                imob = mobility[imask]
                ieng = energy[imask]
                isucc = success[imask]
                if len(imob) < 4:
                    continue
                iq20, iq80 = np.quantile(imob, [0.2, 0.8])
                ilow_m = imob <= iq20
                ihigh_m = imob >= iq80
                ieq20, ieq80 = np.quantile(ieng, [0.2, 0.8])
                ilow_e = ieng <= ieq20
                ihigh_e = ieng >= ieq80
                per_image_rows.append(
                    {
                        "seed": seed,
                        "tag": tag,
                        "epoch": int(row.epoch),
                        "checkpoint_acc": float(row.acc),
                        "stage": stage(float(row.acc)),
                        "basis_attack": basis_attack,
                        "layer": layer,
                        "k": int(k),
                        "image_ord": int(img),
                        "n_directions": int(imask.sum()),
                        "n_adv_random": int(isucc.sum()),
                        "random_asr": float(isucc.mean()),
                        "mobility_flow_spearman": corr(imob, ieng, "spearman"),
                        "high_vs_low_mobility_flow_auroc": safe_auroc(ieng[ihigh_m], ieng[ilow_m]),
                        "low_flow_random_asr": float(isucc[ilow_e].mean()) if ilow_e.any() else np.nan,
                        "high_flow_random_asr": float(isucc[ihigh_e].mean()) if ihigh_e.any() else np.nan,
                        "adv_vs_nonadv_flow_auroc": safe_auroc(ieng[isucc], ieng[~isucc]) if isucc.any() and (~isucc).any() else np.nan,
                    }
                )
            mb = bin_index(mobility, args.n_bins)
            eb = bin_index(energy, args.n_bins)
            for i in range(args.n_bins):
                for j in range(args.n_bins):
                    mask = (mb == i) & (eb == j)
                    if mask.sum() == 0:
                        continue
                    bin_rows.append(
                        {
                            "seed": seed,
                            "tag": tag,
                            "epoch": int(row.epoch),
                            "checkpoint_acc": float(row.acc),
                            "stage": stage(float(row.acc)),
                            "basis_attack": basis_attack,
                            "layer": layer,
                            "k": int(k),
                            "mobility_bin": int(i),
                            "flow_bin": int(j),
                            "n": int(mask.sum()),
                            "random_asr": float(success[mask].mean()),
                            "mobility_mean": float(mobility[mask].mean()),
                            "flow_energy_mean": float(energy[mask].mean()),
                        }
                    )
    wrapper.close()
    del wrapper
    torch.cuda.empty_cache()
    return metric_rows, bin_rows, per_image_rows


def make_plots(metrics: pd.DataFrame, out: Path):
    if metrics.empty:
        return
    sub = metrics[(metrics["k"] == 20) & (metrics["layer"].isin(["layer2", "layer3", "layer4", "avgpool"]))].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    for basis_attack, marker in [("pgd", "o"), ("square", "s")]:
        g = sub[sub.basis_attack == basis_attack].groupby("checkpoint_acc", as_index=False).agg(
            corr=("mobility_flow_spearman", "mean"),
            auroc=("high_vs_low_mobility_flow_auroc", "mean"),
            asr_gap=("high_flow_random_asr", "mean"),
            low_asr=("low_flow_random_asr", "mean"),
        )
        axes[0].plot(g.checkpoint_acc, g["corr"], marker=marker, label=basis_attack.upper())
        axes[1].plot(g.checkpoint_acc, g["auroc"], marker=marker, label=basis_attack.upper())
        axes[2].plot(g.checkpoint_acc, g["asr_gap"] - g["low_asr"], marker=marker, label=basis_attack.upper())
    axes[0].set_title("Mobility-flow correlation")
    axes[0].set_ylabel("Spearman")
    axes[1].set_title("High-mobility vs low-mobility")
    axes[1].set_ylabel("flow-energy AUROC")
    axes[2].set_title("High-flow minus low-flow random ASR")
    axes[2].set_ylabel("ASR difference")
    for ax in axes:
        ax.set_xlabel("checkpoint clean accuracy")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
    fig.savefig(out / "objective_neutral_mobility_flow_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-output-dir", default="analysis_outputs/hidden_jacobian_routing/cifar_training_dynamics_transport_v1")
    p.add_argument("--checkpoint-dir", default="checkpoints/cifar10_resnet18_training_dynamics_v1")
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/cifar_objective_neutral_mobility_flow_v1")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--model-seeds", default="0,1,2")
    p.add_argument("--include-tags", default="init,acc15,acc25,acc40,acc55,acc70,acc82,acc90,final")
    p.add_argument("--basis-attacks", default="pgd,square")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--directions-per-image", type=int, default=128)
    p.add_argument("--direction-batch-size", type=int, default=128)
    p.add_argument("--ks", default="5,10,20")
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument("--seed", type=int, default=515)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args.dataset_root)
    manifest = load_manifest(args)
    manifest.to_csv(out / "checkpoint_manifest_used.csv", index=False)
    all_metrics = []
    all_bins = []
    all_per_image = []
    for row in manifest.itertuples(index=False):
        print(f"[CHECKPOINT] seed={row.seed} tag={row.tag} acc={row.acc:.4f}", flush=True)
        metric_rows, bin_rows, per_image_rows = analyze_checkpoint(args, row, dataset, device)
        all_metrics.extend(metric_rows)
        all_bins.extend(bin_rows)
        all_per_image.extend(per_image_rows)
        pd.DataFrame(all_metrics).to_csv(out / "objective_neutral_mobility_flow_metrics.partial.csv", index=False)
        pd.DataFrame(all_per_image).to_csv(out / "objective_neutral_mobility_flow_per_image.partial.csv", index=False)
    metrics = pd.DataFrame(all_metrics)
    bins = pd.DataFrame(all_bins)
    per_image = pd.DataFrame(all_per_image)
    metrics.to_csv(out / "objective_neutral_mobility_flow_metrics.csv", index=False)
    bins.to_csv(out / "objective_neutral_mobility_flow_bins.csv", index=False)
    per_image.to_csv(out / "objective_neutral_mobility_flow_per_image.csv", index=False)
    make_plots(metrics, out)
    (out / "metadata.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"[DONE] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
