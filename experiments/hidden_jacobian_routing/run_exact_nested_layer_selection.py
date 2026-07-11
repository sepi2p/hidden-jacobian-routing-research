#!/usr/bin/env python3
"""Exact nested layer selection for the Q1 reviewer protocol.

This implements reviewer item 2 exactly for CIFAR-10 natural checkpoints:

* CIFAR-10 clean-correct n=1000 per model from the exact split manifest.
* 400/200/400 basis-fit/layer-validation/final-test split.
* split seeds {1001,1002,1003}, one run per invocation.
* weak PGD-CE: eps=2/255, steps=5, step size=0.5/255, one random start.
* layer selection by validation AUROC of success-vs-failure projection energy
  E_20, with the selected layer frozen before final-test reporting.

The script is resumable: if the expected output files already exist and
``--overwrite`` is not set, it exits without recomputing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import load_model, margin, project_linf  # noqa: E402


LAYER_CANDIDATES = {
    "bbb_resnet50": ["layer1", "layer2", "layer3", "layer4", "avgpool", "logits"],
    "bbb_vgg19_bn": ["block1", "block2", "block3", "block4", "block5", "penultimate", "logits"],
    "bbb_densenet": ["denseblock1", "denseblock2", "denseblock3", "penultimate", "logits"],
    "bbb_inception_v3": ["mixed5", "mixed6", "mixed7", "penultimate", "logits"],
}

PRESPECIFIED_PRELOGIT = {
    "bbb_resnet50": "avgpool",
    "bbb_vgg19_bn": "penultimate",
    "bbb_densenet": "penultimate",
    "bbb_inception_v3": "penultimate",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pca_basis(x: np.ndarray, max_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    xc = x.astype(np.float32) - mean
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s.astype(np.float64) ** 2
    ratio = var / np.clip(var.sum(), 1e-12, None)
    return mean, vt[: min(max_k, vt.shape[0])].astype(np.float32), ratio


def pca_dims(x: np.ndarray) -> dict:
    if len(x) < 2:
        return {"dim80": np.nan, "dim90": np.nan, "effective_rank": np.nan, "pc1_var": np.nan}
    _mean, _basis, ratio = pca_basis(x, min(x.shape[0], x.shape[1]))
    csum = np.cumsum(ratio)
    entropy = -float(np.sum(ratio[ratio > 0] * np.log(ratio[ratio > 0])))
    return {
        "dim80": int(np.searchsorted(csum, 0.80) + 1),
        "dim90": int(np.searchsorted(csum, 0.90) + 1),
        "effective_rank": float(np.exp(entropy)),
        "pc1_var": float(ratio[0]) if len(ratio) else np.nan,
    }


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, basis.shape[0])
    xc = x.astype(np.float32) - mean.astype(np.float32)
    coeff = xc @ basis[:kk].T
    denom = np.sum(xc * xc, axis=1)
    return np.sum(coeff * coeff, axis=1) / np.clip(denom, 1e-12, None)


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    if len(y) < 4 or len(np.unique(y)) < 2 or np.nanstd(score) < 1e-12:
        return np.nan
    return float(roc_auc_score(y, score))


def bootstrap_auc_by_image(score_df: pd.DataFrame, reps: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    img = score_df[["image_ord", "success", "energy"]].drop_duplicates("image_ord")
    ids = img.image_ord.to_numpy()
    if len(ids) < 4 or img.success.nunique() < 2:
        return np.nan, np.nan, np.nan
    vals = []
    for _ in range(reps):
        sample = rng.choice(ids, size=len(ids), replace=True)
        sub = img[img.image_ord.isin(sample)]
        vals.append(safe_auc(sub.success.to_numpy(), sub.energy.to_numpy()))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(vals)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def feature_numpy(wrapper, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    with torch.no_grad():
        logits, feats, _raw = wrapper.forward_with_features(x)
    return logits.detach(), {k: v.detach().cpu().numpy().astype(np.float32) for k, v in feats.items()}


def pgd_trajectory(wrapper, x0: torch.Tensor, y: torch.Tensor, eps: float, step_size: float, steps: int, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator(device=x0.device).manual_seed(seed)
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps, generator=gen), x0, eps).detach()
    states = [x.detach().clone()]
    for _ in range(steps):
        x_req = x.detach().requires_grad_(True)
        logits = wrapper(x_req)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_req)[0]
        x = project_linf(x + step_size * grad.sign(), x0, eps).detach()
        states.append(x.detach().clone())
    return states


def collect_vectors(args, wrapper, dataset, split_df: pd.DataFrame, layers: list[str], device: torch.device):
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    rows = []
    outcomes = []
    for n_done, r in enumerate(split_df.sort_values(["label", "class_ord"]).itertuples(index=False), start=1):
        x_cpu, y0 = dataset[int(r.dataset_idx)]
        label = int(r.label)
        if int(y0) != label:
            raise RuntimeError(f"Label mismatch at dataset_idx={r.dataset_idx}: split={label}, dataset={int(y0)}")
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        clean_logits, _clean_feats = feature_numpy(wrapper, x0)
        clean_pred = int(clean_logits.argmax(1).item())
        if clean_pred != label:
            raise RuntimeError(f"Split image is not clean-correct: idx={r.dataset_idx}, pred={clean_pred}, label={label}")
        states = pgd_trajectory(
            wrapper,
            x0,
            y,
            eps,
            step_size,
            args.steps,
            args.attack_seed + int(r.dataset_idx) * 1009 + int(r.split_seed),
        )
        final_logits, _final_feats = feature_numpy(wrapper, states[-1])
        final_pred = int(final_logits.argmax(1).item())
        success = int(final_pred != label)
        outcomes.append(
            {
                "model": args.model,
                "split_seed": int(r.split_seed),
                "image_ord": int(r.image_ord),
                "dataset_idx": int(r.dataset_idx),
                "label": label,
                "split": str(r.split),
                "clean_pred": clean_pred,
                "final_pred": final_pred,
                "success": success,
                "clean_margin": float(margin(clean_logits, y).item()),
                "final_margin": float(margin(final_logits, y).item()),
            }
        )
        feats_by_step = []
        for st in states:
            _logits, feats = feature_numpy(wrapper, st)
            feats_by_step.append(feats)
        for step in range(len(feats_by_step) - 1):
            for layer in layers:
                if layer not in feats_by_step[step] or layer not in feats_by_step[step + 1]:
                    continue
                v = feats_by_step[step + 1][layer][0] - feats_by_step[step][layer][0]
                idx = len(arrays[layer])
                arrays[layer].append(v.astype(np.float32))
                rows.append(
                    {
                        "model": args.model,
                        "split_seed": int(r.split_seed),
                        "image_ord": int(r.image_ord),
                        "dataset_idx": int(r.dataset_idx),
                        "label": label,
                        "split": str(r.split),
                        "layer": layer,
                        "step": int(step),
                        "vector_idx": int(idx),
                        "success": success,
                        "final_pred": final_pred,
                    }
                )
        if n_done % args.progress_every == 0:
            print(f"[{args.model} seed={args.split_seed}] processed {n_done}/{len(split_df)}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(outcomes), arrays


def analyze_layers(args, meta: pd.DataFrame, arrays: dict[str, list[np.ndarray]], layers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    score_rows = []
    metric_rows = []
    basis_pack = {}
    for layer in layers:
        if layer not in arrays or len(arrays[layer]) < 2:
            continue
        mat = np.stack(arrays[layer]).astype(np.float32)
        m = meta[meta.layer == layer].copy().reset_index(drop=True)
        train_idx = m[(m.split == "basis_fit") & (m.success.astype(int) == 1)].vector_idx.to_numpy(dtype=int)
        if len(train_idx) < 2:
            continue
        mean, basis, ratio = pca_basis(mat[train_idx], args.max_k)
        basis_pack[layer] = {"mean": mean, "basis": basis, "explained_variance": ratio[: args.max_k]}
        dims = pca_dims(mat[train_idx])
        for split in ["layer_validation", "final_test"]:
            sub = m[m.split == split].copy()
            if sub.empty:
                continue
            energies = projection_energy(mat[sub.vector_idx.to_numpy(dtype=int)], mean, basis, args.k)
            sub["energy"] = energies
            score_rows.append(sub[["model", "split_seed", "image_ord", "dataset_idx", "label", "split", "layer", "step", "success", "energy"]])
            y = sub.success.to_numpy(dtype=int)
            auc = safe_auc(y, energies)
            img = sub.groupby("image_ord", as_index=False).agg(success=("success", "first"), energy=("energy", "mean"))
            img_auc = safe_auc(img.success.to_numpy(dtype=int), img.energy.to_numpy(dtype=float))
            bmean, blo, bhi = bootstrap_auc_by_image(
                sub[["image_ord", "success", "energy"]], args.bootstrap_reps, args.bootstrap_seed + stable_int(f"{layer}:{split}") % 100000
            )
            pos = sub[sub.success.astype(int) == 1].energy
            neg = sub[sub.success.astype(int) == 0].energy
            metric_rows.append(
                {
                    "model": args.model,
                    "split_seed": args.split_seed,
                    "layer": layer,
                    "split": split,
                    "k": args.k,
                    "n_train_success_segments": int(len(train_idx)),
                    "n_segments": int(len(sub)),
                    "n_images": int(sub.image_ord.nunique()),
                    "n_success_images": int(img.success.sum()),
                    "n_failed_images": int((1 - img.success).sum()),
                    "segment_auroc": auc,
                    "image_auroc": img_auc,
                    "image_bootstrap_auroc_mean": bmean,
                    "image_bootstrap_auroc_lo": blo,
                    "image_bootstrap_auroc_hi": bhi,
                    "projection_energy_success_mean": float(pos.mean()) if len(pos) else np.nan,
                    "projection_energy_failed_mean": float(neg.mean()) if len(neg) else np.nan,
                    **dims,
                }
            )
    score_df = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    return score_df, metrics, basis_pack


def save_basis_npz(path: Path, basis_pack: dict) -> None:
    payload = {}
    for layer, pack in basis_pack.items():
        safe = layer.replace("/", "_")
        payload[f"{safe}__mean"] = pack["mean"].astype(np.float32)
        payload[f"{safe}__basis"] = pack["basis"].astype(np.float32)
        payload[f"{safe}__explained_variance"] = np.asarray(pack["explained_variance"], dtype=np.float64)
    np.savez_compressed(path, **payload)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-csv", default="analysis_outputs/hidden_jacobian_routing/exact_protocol/cifar_splits/cifar10_exact_splits.csv")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split-seed", type=int, required=True)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--step-size", type=float, default=0.5)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--max-k", type=int, default=40)
    p.add_argument("--attack-seed", type=int, default=0)
    p.add_argument("--bootstrap-seed", type=int, default=12345)
    p.add_argument("--bootstrap-reps", type=int, default=10000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--max-images", type=int, default=-1, help="Debug/smoke mode only. Do not use for promoted evidence.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    sentinel = out / "nested_layer_selection_summary.csv"
    if sentinel.exists() and not args.overwrite:
        print(f"[SKIP] existing {sentinel}")
        return
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.attack_seed + args.split_seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    wrapper = load_model(args.model, device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    splits = pd.read_csv(args.splits_csv)
    split_df = splits[(splits.model == args.model) & (splits.split_seed == args.split_seed)].copy()
    if split_df.empty:
        raise RuntimeError(f"No split rows for model={args.model} split_seed={args.split_seed}")
    if args.max_images > 0:
        split_df = split_df.sort_values(["split", "label", "class_ord"]).groupby("split", group_keys=False).head(args.max_images)
    layers = [x for x in LAYER_CANDIDATES[args.model] if x in wrapper.labels]
    rows, outcomes, arrays = collect_vectors(args, wrapper, dataset, split_df, layers, device)
    rows.to_csv(out / "nested_layer_segment_metadata.csv", index=False)
    outcomes.to_csv(out / "nested_layer_image_outcomes.csv", index=False)
    np.savez_compressed(out / "nested_layer_segment_vectors.npz", **{k: np.stack(v).astype(np.float32) for k, v in arrays.items()})
    scores, metrics, basis_pack = analyze_layers(args, rows, arrays, layers)
    scores.to_csv(out / "nested_layer_projection_scores.csv", index=False)
    metrics.to_csv(out / "nested_layer_metrics_all_layers.csv", index=False)
    save_basis_npz(out / "nested_layer_bases.npz", basis_pack)

    val = metrics[metrics.split == "layer_validation"].copy()
    val_nonlogit = val[val.layer != "logits"].copy()
    if val_nonlogit.empty:
        raise RuntimeError("No non-logit validation metrics available.")
    val_nonlogit["selection_score"] = val_nonlogit["image_auroc"].fillna(val_nonlogit["segment_auroc"])
    selected = val_nonlogit.sort_values(["selection_score", "layer"], ascending=[False, True]).iloc[0]
    selected_layer = str(selected.layer)
    prespecified_layer = PRESPECIFIED_PRELOGIT[args.model]
    summary = []
    for rule, layer in [("nested_selected_nonlogit", selected_layer), ("prespecified_prelogit", prespecified_layer), ("logits", "logits")]:
        for split in ["layer_validation", "final_test"]:
            sub = metrics[(metrics.layer == layer) & (metrics.split == split)].copy()
            if sub.empty:
                continue
            row = sub.iloc[0].to_dict()
            row.update({"layer_rule": rule, "selected_layer": selected_layer, "reported_layer": layer})
            summary.append(row)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(sentinel, index=False)
    metadata = {
        "experiment": "q1_exact_nested_layer_selection",
        "model": args.model,
        "split_seed": args.split_seed,
        "layers": layers,
        "selected_layer": selected_layer,
        "prespecified_prelogit": prespecified_layer,
        "attack": "PGD-CE",
        "eps_255": args.eps,
        "steps": args.steps,
        "step_size_255": args.step_size,
        "random_starts": 1,
        "split_rule": "basis_fit trains PCA on successful local steps; layer_validation selects layer by E20 AUROC; final_test reports frozen layer once",
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "max_images_debug": args.max_images,
        "promotable": bool(args.max_images < 0),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[DONE] selected_layer={selected_layer} output={out}")


if __name__ == "__main__":
    main()
