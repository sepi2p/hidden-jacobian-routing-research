#!/usr/bin/env python3
"""Non-adversarial optimization controls for attack-transport geometry.

This experiment asks whether the observed adversarial transport structure is
specific to successful adversarial optimization or a generic consequence of
optimizing neural representations. It collects local representation transport
vectors for PGD and for several non-adversarial objectives, then compares their
dimensionality, projection energy, and subspace overlap.
"""

from __future__ import annotations

import argparse
import gc
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
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import load_model  # noqa: E402
from experiments.hidden_jacobian_routing.common import LAYER_GROUPS  # noqa: E402
from experiments.hidden_jacobian_routing.common import margin, project_linf  # noqa: E402


KS = [5, 10, 20, 50]
OBJECTIVES = [
    "same_class_feature_match",
    "different_image_feature_match",
    "activation_max",
    "feature_norm_max",
    "random_feature_direction",
]


def parse_csv(x: str) -> list[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def pca_stats(x: np.ndarray) -> dict:
    if len(x) < 2:
        return {
            "n": int(len(x)),
            "d": int(x.shape[1]) if x.ndim == 2 else 0,
            "pc1_var": np.nan,
            "pc10_cum_var": np.nan,
            "dim80": np.nan,
            "dim90": np.nan,
            "effective_rank": np.nan,
        }
    xc = x - x.mean(axis=0, keepdims=True)
    _u, s, _vt = np.linalg.svd(xc, full_matrices=False)
    var = s * s
    ratios = var / np.clip(var.sum(), 1e-12, None)
    csum = np.cumsum(ratios)
    entropy = -float(np.sum(ratios[ratios > 0] * np.log(ratios[ratios > 0])))
    return {
        "n": int(len(x)),
        "d": int(x.shape[1]),
        "pc1_var": float(ratios[0]),
        "pc10_cum_var": float(csum[min(9, len(csum) - 1)]),
        "dim80": int(np.searchsorted(csum, 0.8) + 1),
        "dim90": int(np.searchsorted(csum, 0.9) + 1),
        "effective_rank": float(np.exp(entropy)),
    }


def fit_basis(x: np.ndarray, max_k: int) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    return mean.astype(np.float32), vt[: min(max_k, vt.shape[0])].astype(np.float32)


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, basis.shape[0])
    xc = x - mean
    coeff = xc @ basis[:kk].T
    return np.sum(coeff * coeff, axis=1) / np.clip(np.sum(xc * xc, axis=1), 1e-12, None)


def subspace_overlap(a: np.ndarray, b: np.ndarray, k: int) -> dict:
    kk = min(k, a.shape[0], b.shape[0])
    if kk == 0:
        return {
            "k": int(k),
            "mean_principal_angle_deg": np.nan,
            "max_principal_angle_deg": np.nan,
            "projection_overlap": np.nan,
            "grassmann_distance": np.nan,
            "subspace_affinity": np.nan,
        }
    s = np.linalg.svd(a[:kk] @ b[:kk].T, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return {
        "k": int(k),
        "mean_principal_angle_deg": float(np.mean(angles)),
        "max_principal_angle_deg": float(np.max(angles)),
        "projection_overlap": float(np.sum(s * s) / kk),
        "grassmann_distance": float(np.linalg.norm(np.sin(np.radians(angles)))),
        "subspace_affinity": float(np.linalg.norm(s) / np.sqrt(kk)),
    }


def safe_key(*parts: str) -> str:
    return "__".join(str(p).replace(":", "_").replace("/", "_").replace("|", "_") for p in parts)


def select_clean_correct(dataset, wrapper, model_name: str, n_images: int, batch_size: int, device):
    selected = []
    for start in range(0, len(dataset), batch_size):
        xs, ys, idxs = [], [], []
        for idx in range(start, min(start + batch_size, len(dataset))):
            x, y = dataset[idx]
            xs.append(x)
            ys.append(int(y))
            idxs.append(idx)
        x = torch.stack(xs).to(device)
        y = torch.tensor(ys, device=device)
        with torch.no_grad():
            pred = wrapper(x).argmax(1)
        for local_i, ok in enumerate(pred.eq(y).detach().cpu().tolist()):
            if ok:
                selected.append({"dataset_idx": int(idxs[local_i]), "label": int(ys[local_i])})
                if len(selected) >= n_images:
                    counts = pd.Series([r["label"] for r in selected]).value_counts().sort_index().to_dict()
                    print(f"[SELECT] {model_name} clean_correct={len(selected)} class_counts={counts}", flush=True)
                    return selected
    counts = pd.Series([r["label"] for r in selected]).value_counts().sort_index().to_dict()
    print(f"[SELECT] {model_name} clean_correct={len(selected)} class_counts={counts}", flush=True)
    return selected


def load_batch(dataset, selected_rows, device):
    xs, ys, idxs = [], [], []
    for r in selected_rows:
        x, y = dataset[int(r["dataset_idx"])]
        xs.append(x)
        ys.append(int(y))
        idxs.append(int(r["dataset_idx"]))
    return torch.stack(xs).to(device), torch.tensor(ys, device=device), idxs


def feature_tensor(wrapper, x: torch.Tensor, layer: str):
    logits, feats, _raw = wrapper.forward_with_features(x)
    if layer not in feats:
        raise KeyError(f"Layer {layer} not found. Available: {sorted(feats)}")
    return logits, feats[layer]


def build_ref_lookup(selected: list[dict], seed: int) -> dict[tuple[int, str], int | None]:
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i, r in enumerate(selected):
        by_class[int(r["label"])].append(i)
    all_indices = list(range(len(selected)))
    lookup = {}
    for i, r in enumerate(selected):
        label = int(r["label"])
        same = [j for j in by_class[label] if j != i]
        diff = [j for j in all_indices if int(selected[j]["label"]) != label]
        lookup[(i, "same_class_feature_match")] = rng.choice(same) if same else None
        lookup[(i, "different_image_feature_match")] = rng.choice(diff) if diff else None
    return lookup


def append_vectors(
    rows: list[dict],
    arrays: dict[str, list[np.ndarray]],
    *,
    model: str,
    source: str,
    objective: str,
    layer_group: str,
    layer: str,
    image_ords: list[int],
    dataset_indices: list[int],
    labels: torch.Tensor,
    step: int,
    vec: np.ndarray,
    pred_before: np.ndarray,
    pred_after: np.ndarray,
    margin_before: np.ndarray,
    final_pred: np.ndarray | None = None,
):
    key = safe_key(model, source, objective, layer_group, layer)
    start = len(arrays[key])
    arrays[key].extend(vec.astype(np.float32))
    labels_cpu = labels.detach().cpu().numpy().astype(int)
    for j, image_ord in enumerate(image_ords):
        rows.append(
            {
                "model": model,
                "source": source,
                "objective": objective,
                "layer_group": layer_group,
                "layer": layer,
                "image_ord": int(image_ord),
                "dataset_idx": int(dataset_indices[j]),
                "label": int(labels_cpu[j]),
                "step": int(step),
                "pred_before": int(pred_before[j]),
                "pred_after": int(pred_after[j]),
                "margin_before": float(margin_before[j]),
                "final_pred": int(final_pred[j]) if final_pred is not None else -1,
                "final_adversarial_success": int(final_pred[j] != labels_cpu[j]) if final_pred is not None else -1,
                "vector_key": key,
                "vector_idx": int(start + j),
            }
        )


def update_final_fields(rows: list[dict], row_start: int, labels: torch.Tensor, final_pred: torch.Tensor):
    y = labels.detach().cpu().numpy().astype(int)
    p = final_pred.detach().cpu().numpy().astype(int)
    for row in rows[row_start:]:
        local = row["image_ord_in_batch"]
        row["final_pred"] = int(p[local])
        row["final_adversarial_success"] = int(p[local] != y[local])
        del row["image_ord_in_batch"]


def collect_pgd_batch(wrapper, dataset, selected, model: str, layer_groups: list[str], args, device):
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    rows: list[dict] = []
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    selected_ord = list(range(len(selected)))
    for batch_start in range(0, len(selected), args.batch_size):
        batch_rows = selected[batch_start : batch_start + args.batch_size]
        image_ords = selected_ord[batch_start : batch_start + len(batch_rows)]
        x, y, dataset_indices = load_batch(dataset, batch_rows, device)
        x0 = x.detach()
        x_adv = x0.clone()
        row_start = len(rows)
        tmp_rows: list[dict] = []
        tmp_arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        for step in range(args.steps):
            probe = x_adv.detach().requires_grad_(True)
            logits, feats, _raw = wrapper.forward_with_features(probe)
            pred_before = logits.detach().argmax(1).cpu().numpy()
            margin_before = margin(logits.detach(), y).cpu().numpy()
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, probe)[0]
            x_adv = project_linf(x_adv + step_size * grad.sign(), x0, eps)
            with torch.no_grad():
                logits_next, feats_next, _raw_next = wrapper.forward_with_features(x_adv)
            pred_after = logits_next.detach().argmax(1).cpu().numpy()
            for layer_group in layer_groups:
                layer = LAYER_GROUPS[layer_group][model]
                vec = (feats_next[layer].detach() - feats[layer].detach()).cpu().numpy()
                before_len = len(tmp_rows)
                append_vectors(
                    tmp_rows,
                    tmp_arrays,
                    model=model,
                    source="adversarial",
                    objective="pgd",
                    layer_group=layer_group,
                    layer=layer,
                    image_ords=image_ords,
                    dataset_indices=dataset_indices,
                    labels=y,
                    step=step,
                    vec=vec,
                    pred_before=pred_before,
                    pred_after=pred_after,
                    margin_before=margin_before,
                )
                for k in range(before_len, len(tmp_rows)):
                    tmp_rows[k]["image_ord_in_batch"] = k - before_len
        with torch.no_grad():
            final_pred = wrapper(x_adv).argmax(1)
        update_final_fields(tmp_rows, 0, y, final_pred)
        rows.extend(tmp_rows)
        for key, vals in tmp_arrays.items():
            arrays[key].extend(vals)
    return rows, arrays


def control_loss(objective: str, h: torch.Tensor, target_h: torch.Tensor | None, random_dir: torch.Tensor | None):
    if objective in {"same_class_feature_match", "different_image_feature_match"}:
        return F.mse_loss(h, target_h), "min"
    if objective == "activation_max":
        return h.mean(), "max"
    if objective == "feature_norm_max":
        return h.pow(2).sum(dim=1).mean(), "max"
    if objective == "random_feature_direction":
        return (h * random_dir).sum(dim=1).mean(), "max"
    raise ValueError(f"Unknown objective: {objective}")


def collect_control_batch(
    wrapper,
    dataset,
    selected,
    ref_lookup,
    model: str,
    objective: str,
    layer_group: str,
    args,
    device,
):
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    layer = LAYER_GROUPS[layer_group][model]
    rows: list[dict] = []
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    selected_ord = list(range(len(selected)))
    gen = torch.Generator(device=device).manual_seed(args.seed + abs(hash((model, objective, layer_group))) % 100000)
    for batch_start in range(0, len(selected), args.batch_size):
        batch_indices = list(range(batch_start, min(batch_start + args.batch_size, len(selected))))
        valid = []
        refs = []
        for i in batch_indices:
            ref_i = ref_lookup.get((i, objective))
            if objective in {"same_class_feature_match", "different_image_feature_match"} and ref_i is None:
                continue
            valid.append(i)
            refs.append(ref_i)
        if not valid:
            continue
        batch_rows = [selected[i] for i in valid]
        image_ords = [selected_ord[i] for i in valid]
        x, y, dataset_indices = load_batch(dataset, batch_rows, device)
        x0 = x.detach()
        x_adv = x0.clone()
        target_h = None
        random_dir = None
        if objective in {"same_class_feature_match", "different_image_feature_match"}:
            ref_rows = [selected[int(i)] for i in refs]
            x_ref, _y_ref, _idx_ref = load_batch(dataset, ref_rows, device)
            with torch.no_grad():
                _logits_ref, target_h = feature_tensor(wrapper, x_ref, layer)
                target_h = target_h.detach()
        elif objective == "random_feature_direction":
            with torch.no_grad():
                _logits0, h0 = feature_tensor(wrapper, x, layer)
            random_dir = torch.randn(h0.shape, generator=gen, device=device, dtype=h0.dtype)
            random_dir = F.normalize(random_dir.flatten(1), dim=1).view_as(h0)

        tmp_rows: list[dict] = []
        tmp_arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        for step in range(args.steps):
            probe = x_adv.detach().requires_grad_(True)
            logits, h = feature_tensor(wrapper, probe, layer)
            pred_before = logits.detach().argmax(1).cpu().numpy()
            margin_before = margin(logits.detach(), y).cpu().numpy()
            loss, direction = control_loss(objective, h, target_h, random_dir)
            grad = torch.autograd.grad(loss, probe)[0]
            sign = -1.0 if direction == "min" else 1.0
            x_adv = project_linf(x_adv + sign * step_size * grad.sign(), x0, eps)
            with torch.no_grad():
                logits_next, h_next = feature_tensor(wrapper, x_adv, layer)
            pred_after = logits_next.detach().argmax(1).cpu().numpy()
            vec = (h_next.detach() - h.detach()).cpu().numpy()
            before_len = len(tmp_rows)
            append_vectors(
                tmp_rows,
                tmp_arrays,
                model=model,
                source="control",
                objective=objective,
                layer_group=layer_group,
                layer=layer,
                image_ords=image_ords,
                dataset_indices=dataset_indices,
                labels=y,
                step=step,
                vec=vec,
                pred_before=pred_before,
                pred_after=pred_after,
                margin_before=margin_before,
            )
            for k in range(before_len, len(tmp_rows)):
                tmp_rows[k]["image_ord_in_batch"] = k - before_len
        with torch.no_grad():
            final_pred = wrapper(x_adv).argmax(1)
        update_final_fields(tmp_rows, 0, y, final_pred)
        rows.extend(tmp_rows)
        for key, vals in tmp_arrays.items():
            arrays[key].extend(vals)
    return rows, arrays


def arrays_to_npz(arrays: dict[str, list[np.ndarray]], path: Path):
    packed = {}
    for key, vals in arrays.items():
        packed[f"vectors__{key}"] = np.stack(vals).astype(np.float32) if vals else np.empty((0, 1), dtype=np.float32)
    np.savez_compressed(path, **packed)


def get_vectors(npz, key: str) -> np.ndarray:
    name = f"vectors__{key}"
    if name not in npz:
        return np.empty((0, 1), dtype=np.float32)
    return npz[name]


def merge_arrays(dst: dict[str, list[np.ndarray]], src: dict[str, list[np.ndarray]]):
    offsets = {}
    for key, vals in src.items():
        offsets[key] = len(dst[key])
        dst[key].extend(vals)
    return offsets


def offset_rows(rows: list[dict], offsets: dict[str, int]):
    for row in rows:
        row["vector_idx"] += offsets[row["vector_key"]]


def train_test_masks(meta: pd.DataFrame, seed: int, train_frac: float):
    images = np.array(sorted(meta["image_ord"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(images)
    n_train = max(1, int(round(len(images) * train_frac)))
    train_images = set(images[:n_train].tolist())
    return meta["image_ord"].isin(train_images).to_numpy(), ~meta["image_ord"].isin(train_images).to_numpy()


def metric_rows_for_pair(
    adv_train: np.ndarray,
    adv_test: np.ndarray,
    control_train: np.ndarray,
    control_test: np.ndarray,
    base: dict,
):
    rows = []
    if len(adv_train) < 8 or len(adv_test) < 5 or len(control_test) < 5:
        return rows
    adv_mean, adv_basis = fit_basis(adv_train, max(KS))
    control_mean, control_basis = fit_basis(control_train, max(KS)) if len(control_train) >= 8 else (None, None)
    for k in KS:
        pos = projection_energy(adv_test, adv_mean, adv_basis, k)
        neg = projection_energy(control_test, adv_mean, adv_basis, k)
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        score = np.r_[pos, neg]
        rows.append(
            {
                **base,
                "basis_source": "adversarial_pgd",
                "k": int(k),
                "auroc_adv_vs_control": float(roc_auc_score(y, score)),
                "adv_mean_energy": float(np.mean(pos)),
                "control_mean_energy": float(np.mean(neg)),
                "n_adv_test": int(len(pos)),
                "n_control_test": int(len(neg)),
            }
        )
        if control_basis is not None:
            pos_c = projection_energy(adv_test, control_mean, control_basis, k)
            neg_c = projection_energy(control_test, control_mean, control_basis, k)
            rows.append(
                {
                    **base,
                    "basis_source": "control",
                    "k": int(k),
                    "auroc_adv_vs_control": float(roc_auc_score(y, np.r_[pos_c, neg_c])),
                    "adv_mean_energy": float(np.mean(pos_c)),
                    "control_mean_energy": float(np.mean(neg_c)),
                    "n_adv_test": int(len(pos_c)),
                    "n_control_test": int(len(neg_c)),
                }
            )
    return rows


def analyze(meta: pd.DataFrame, vectors_npz, args, out_dir: Path):
    dim_rows = []
    metric_rows = []
    overlap_rows = []
    transfer_rows = []

    groups = meta.groupby(["model", "source", "objective", "layer_group", "layer"], dropna=False)
    cached = {}
    for key, group in groups:
        model, source, objective, layer_group, layer = key
        arr = get_vectors(vectors_npz, group["vector_key"].iloc[0])
        idx = group["vector_idx"].to_numpy(int)
        x = normalize_rows(arr[idx])
        cached[key] = (group.reset_index(drop=True), x)
        stats = pca_stats(x)
        dim_rows.append(
            {
                "model": model,
                "source": source,
                "objective": objective,
                "layer_group": layer_group,
                "layer": layer,
                **stats,
            }
        )

    for (model, layer_group, layer), sub in meta.groupby(["model", "layer_group", "layer"], dropna=False):
        adv_key = (model, "adversarial", "pgd", layer_group, layer)
        if adv_key not in cached:
            continue
        adv_meta, adv_x = cached[adv_key]
        adv_success = adv_meta["final_adversarial_success"].to_numpy(int) == 1
        train_mask, test_mask = train_test_masks(adv_meta, args.seed, args.train_frac)
        adv_train = adv_x[train_mask & adv_success]
        adv_test = adv_x[test_mask & adv_success]
        if len(adv_train) < 8 or len(adv_test) < 5:
            continue
        adv_mean, adv_basis = fit_basis(adv_train, max(KS))
        for objective in args.objectives:
            control_key = (model, "control", objective, layer_group, layer)
            if control_key not in cached:
                continue
            control_meta, control_x = cached[control_key]
            c_train_mask, c_test_mask = train_test_masks(control_meta, args.seed, args.train_frac)
            control_train = control_x[c_train_mask]
            control_test = control_x[c_test_mask]
            base = {
                "model": model,
                "control_objective": objective,
                "layer_group": layer_group,
                "layer": layer,
            }
            metric_rows.extend(metric_rows_for_pair(adv_train, adv_test, control_train, control_test, base))
            if len(control_train) >= 8:
                _cmean, control_basis = fit_basis(control_train, max(KS))
                for k in KS:
                    overlap_rows.append(
                        {
                            **base,
                            **subspace_overlap(adv_basis, control_basis, k),
                        }
                    )
                for k in KS:
                    adv_score = projection_energy(adv_test, adv_mean, adv_basis, k)
                    control_on_adv_basis = projection_energy(control_test, adv_mean, adv_basis, k)
                    transfer_rows.append(
                        {
                            **base,
                            "basis_train": "adversarial_pgd",
                            "eval_positive": "adversarial_success",
                            "eval_negative": "control",
                            "k": int(k),
                            "positive_mean_energy": float(np.mean(adv_score)),
                            "negative_mean_energy": float(np.mean(control_on_adv_basis)),
                            "auroc": float(roc_auc_score(np.r_[np.ones(len(adv_score)), np.zeros(len(control_on_adv_basis))], np.r_[adv_score, control_on_adv_basis])),
                            "n_positive": int(len(adv_score)),
                            "n_negative": int(len(control_on_adv_basis)),
                        }
                    )

    dim_df = pd.DataFrame(dim_rows)
    metric_df = pd.DataFrame(metric_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    transfer_df = pd.DataFrame(transfer_rows)
    dim_df.to_csv(out_dir / "adversarial_vs_control_dimensionality.csv", index=False)
    metric_df.to_csv(out_dir / "adversarial_vs_control_projection_metrics.csv", index=False)
    overlap_df.to_csv(out_dir / "adversarial_vs_control_subspace_overlap.csv", index=False)
    transfer_df.to_csv(out_dir / "adversarial_control_basis_transfer.csv", index=False)
    write_summary_plot(metric_df, overlap_df, out_dir)
    return dim_df, metric_df, overlap_df, transfer_df


def write_summary_plot(metric_df: pd.DataFrame, overlap_df: pd.DataFrame, out_dir: Path):
    if metric_df.empty:
        return
    view = metric_df[(metric_df["basis_source"] == "adversarial_pgd") & (metric_df["k"] == 20)].copy()
    if view.empty:
        return
    order = list(dict.fromkeys(view["control_objective"].tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    ax = axes[0]
    data = [view.loc[view.control_objective == obj, "auroc_adv_vs_control"].dropna().to_numpy() for obj in order]
    ax.boxplot(data, labels=[o.replace("_", "\n") for o in order], showfliers=False)
    ax.axhline(0.5, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_ylabel("AUROC")
    ax.set_title("Adversarial transport basis separates tested controls")
    ax.tick_params(axis="x", labelsize=7)
    if not overlap_df.empty:
        ov = overlap_df[overlap_df["k"] == 20]
        data2 = [ov.loc[ov.control_objective == obj, "projection_overlap"].dropna().to_numpy() for obj in order]
        axes[1].boxplot(data2, labels=[o.replace("_", "\n") for o in order], showfliers=False)
        axes[1].set_ylabel("Projection overlap")
        axes[1].set_title("Adversarial vs control subspace overlap")
        axes[1].tick_params(axis="x", labelsize=7)
    fig.savefig(out_dir / "nonadversarial_control_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "nonadversarial_control_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = parse_csv(args.models)
    layer_groups = parse_csv(args.layer_groups)
    objectives = parse_csv(args.objectives)
    args.objectives = objectives
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    all_rows = []
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    selected_by_model = {}

    for model_name in models:
        wrapper = load_model(model_name, device).eval()
        selected = select_clean_correct(dataset, wrapper, model_name, args.images, args.batch_size, device)
        selected_by_model[model_name] = selected
        ref_lookup = build_ref_lookup(selected, args.seed)
        pgd_rows, pgd_arrays = collect_pgd_batch(wrapper, dataset, selected, model_name, layer_groups, args, device)
        offsets = merge_arrays(arrays, pgd_arrays)
        offset_rows(pgd_rows, offsets)
        all_rows.extend(pgd_rows)
        n_success = pd.DataFrame(pgd_rows)[["image_ord", "final_adversarial_success"]].drop_duplicates()["final_adversarial_success"].sum()
        print(f"[PGD] {model_name} success_images={int(n_success)}/{len(selected)} rows={len(pgd_rows)}", flush=True)
        del pgd_rows, pgd_arrays
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for layer_group in layer_groups:
            for objective in objectives:
                print(f"[CONTROL] {model_name} {layer_group} {objective}", flush=True)
                rows, arrs = collect_control_batch(wrapper, dataset, selected, ref_lookup, model_name, objective, layer_group, args, device)
                offsets = merge_arrays(arrays, arrs)
                offset_rows(rows, offsets)
                all_rows.extend(rows)
                print(f"  rows={len(rows)}", flush=True)
                del rows, arrs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    meta = pd.DataFrame(all_rows)
    meta.to_csv(out_dir / "control_segment_metadata.csv", index=False)
    arrays_to_npz(arrays, out_dir / "control_segment_vectors.npz")
    vectors_npz = np.load(out_dir / "control_segment_vectors.npz")
    dim_df, metric_df, overlap_df, transfer_df = analyze(meta, vectors_npz, args, out_dir)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "args": {k: (",".join(v) if isinstance(v, list) else v) for k, v in vars(args).items()},
                "selected_by_model": selected_by_model,
                "outputs": {
                    "metadata": str(out_dir / "control_segment_metadata.csv"),
                    "vectors": str(out_dir / "control_segment_vectors.npz"),
                    "projection_metrics": str(out_dir / "adversarial_vs_control_projection_metrics.csv"),
                    "dimensionality": str(out_dir / "adversarial_vs_control_dimensionality.csv"),
                    "overlap": str(out_dir / "adversarial_vs_control_subspace_overlap.csv"),
                    "basis_transfer": str(out_dir / "adversarial_control_basis_transfer.csv"),
                },
            },
            f,
            indent=2,
        )
    print("\n[SUMMARY] projection metrics")
    if not metric_df.empty:
        compact = metric_df[(metric_df.basis_source == "adversarial_pgd") & (metric_df.k == 20)]
        print(compact.groupby("control_objective")["auroc_adv_vs_control"].agg(["mean", "median", "min", "max"]).to_string())
    print(f"[SAVED] {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/nonadversarial_optimization_controls_c200_s40")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--models", default="bbb_resnet50,bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eps", type=float, default=8.0, help="Linf epsilon in pixel/255 units.")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--step-size", type=float, default=2.0, help="Step size in pixel/255 units.")
    p.add_argument("--layer-groups", default="hidden,penultimate,logits")
    p.add_argument("--objectives", default=",".join(OBJECTIVES))
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
