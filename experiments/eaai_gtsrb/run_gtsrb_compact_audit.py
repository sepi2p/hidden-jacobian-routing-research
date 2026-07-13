#!/usr/bin/env python3
"""Run the compact hidden-Jacobian audit on frozen GTSRB checkpoints.

The script is resumable at model/stage boundaries and saves feature trajectories
so analysis and figures can be regenerated without rerunning attacks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from gtsrb_common import (
    FeatureCapture,
    gtsrb_dataset,
    load_checkpoint,
    set_seed,
    sha256,
    write_json,
)


K_VALUES = (5, 10, 20, 40)
WEAK_ATTACK_GRID = (
    (0.25, 1, 0.25),
    (0.50, 2, 0.25),
    (1.00, 3, 0.50),
    (2.00, 5, 0.50),
    (4.00, 5, 1.00),
    (4.00, 10, 1.00),
    (8.00, 10, 2.00),
)


class IndexedDataset(Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        return x, y, index


def margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, y[:, None], -torch.inf)
    return true - other.max(1).values


def project_linf(x: torch.Tensor, clean: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x, clean + eps), clean - eps).clamp(0.0, 1.0)


def deterministic_noise(
    x: torch.Tensor, indices: torch.Tensor, eps: float, base_seed: int
) -> torch.Tensor:
    rows = []
    for idx in indices.detach().cpu().tolist():
        generator = torch.Generator(device=x.device).manual_seed(base_seed + int(idx) * 1009)
        rows.append(torch.empty_like(x[:1]).uniform_(-eps, eps, generator=generator))
    return torch.cat(rows, dim=0)


def pgd_batch(
    wrapper: FeatureCapture,
    clean: torch.Tensor,
    y: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
    steps: int,
    step_size: float,
    seed: int,
    capture: bool,
):
    x = project_linf(clean + deterministic_noise(clean, indices, eps, seed), clean, eps)
    feature_states: dict[str, list[np.ndarray]] = {}
    pixel_states: list[torch.Tensor] = []
    first_grad_l1 = None
    first_grad_l2 = None
    for step in range(steps + 1):
        with torch.no_grad():
            logits, feats = wrapper.forward_with_features(x)
        if capture:
            pixel_states.append(x.detach().clone())
            for layer, value in feats.items():
                feature_states.setdefault(layer, []).append(
                    value.detach().cpu().numpy().astype(np.float32)
                )
        if step == steps:
            break
        x_req = x.detach().requires_grad_(True)
        logits = wrapper(x_req)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_req)[0]
        if step == 0:
            flat = grad.detach().flatten(1)
            first_grad_l1 = flat.abs().mean(1).cpu().numpy()
            first_grad_l2 = flat.norm(dim=1).cpu().numpy()
        x = project_linf(x + step_size * grad.sign(), clean, eps).detach()
    with torch.no_grad():
        clean_logits = wrapper(clean)
        initial_logits = wrapper(pixel_states[0] if capture else project_linf(
            clean + deterministic_noise(clean, indices, eps, seed), clean, eps
        ))
        final_logits = wrapper(x)
    result = {
        "success": (final_logits.argmax(1) != y).cpu().numpy().astype(int),
        "clean_margin": margin(clean_logits, y).cpu().numpy(),
        "clean_loss": F.cross_entropy(clean_logits, y, reduction="none").cpu().numpy(),
        "initial_margin": margin(initial_logits, y).cpu().numpy(),
        "final_margin": margin(final_logits, y).cpu().numpy(),
        "first_grad_l1": first_grad_l1,
        "first_grad_l2": first_grad_l2,
    }
    if capture:
        with torch.no_grad():
            first_logits = wrapper(pixel_states[1])
        result["first_step_margin_drop"] = (
            margin(initial_logits, y) - margin(first_logits, y)
        ).cpu().numpy()
        result["feature_states"] = {
            layer: np.stack(values, axis=1)
            for layer, values in feature_states.items()
        }
        result["pixel_states"] = pixel_states
    return result


def select_clean_correct(
    model, dataset, images: int, batch_size: int, device: torch.device
) -> pd.DataFrame:
    indexed = IndexedDataset(dataset)
    loader = DataLoader(indexed, batch_size=batch_size, shuffle=False, num_workers=4)
    by_class: dict[int, list[int]] = {label: [] for label in range(43)}
    target_per_class = math.ceil(images / 43)
    with torch.no_grad():
        for x, y, idx in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            for label, prediction, dataset_idx in zip(y.tolist(), pred.tolist(), idx.tolist()):
                if prediction == label and len(by_class[label]) < target_per_class:
                    by_class[label].append(dataset_idx)
            if all(len(v) >= target_per_class for v in by_class.values()):
                break
    rows = []
    for round_idx in range(target_per_class):
        for label in range(43):
            if round_idx < len(by_class[label]) and len(rows) < images:
                rows.append({"dataset_idx": by_class[label][round_idx], "label": label})
    if len(rows) < images:
        raise RuntimeError(f"Only found {len(rows)} class-balanced clean-correct images")
    out = pd.DataFrame(rows)
    out.insert(0, "image_ord", np.arange(len(out)))
    return out


def assign_splits(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces = []
    for _label, group in rows.groupby("label"):
        local = group.copy()
        order = rng.permutation(len(local))
        n_fit = int(round(0.40 * len(local)))
        n_val = int(round(0.20 * len(local)))
        split = np.full(len(local), "final_test", dtype=object)
        split[order[:n_fit]] = "basis_fit"
        split[order[n_fit : n_fit + n_val]] = "layer_validation"
        local["split"] = split
        pieces.append(local)
    return pd.concat(pieces, ignore_index=True).sort_values("image_ord")


def load_images(dataset, rows: pd.DataFrame):
    xs, ys, ids = [], [], []
    for row in rows.itertuples(index=False):
        x, y = dataset[int(row.dataset_idx)]
        if int(y) != int(row.label):
            raise RuntimeError("GTSRB label mismatch")
        xs.append(x)
        ys.append(int(y))
        ids.append(int(row.dataset_idx))
    return torch.stack(xs), torch.tensor(ys), torch.tensor(ids)


def tune_weak_attack(
    wrapper, dataset, rows, batch_size, device, target_asr, seed
) -> pd.DataFrame:
    x_all, y_all, ids_all = load_images(dataset, rows)
    records = []
    for eps_255, steps, step_255 in WEAK_ATTACK_GRID:
        successes = []
        for start in range(0, len(rows), batch_size):
            result = pgd_batch(
                wrapper,
                x_all[start : start + batch_size].to(device),
                y_all[start : start + batch_size].to(device),
                ids_all[start : start + batch_size].to(device),
                eps_255 / 255.0,
                steps,
                step_255 / 255.0,
                seed,
                capture=False,
            )
            successes.extend(result["success"].tolist())
        asr = float(np.mean(successes))
        records.append(
            {
                "eps_255": eps_255,
                "steps": steps,
                "step_size_255": step_255,
                "asr": asr,
                "successes": int(np.sum(successes)),
                "failures": int(len(successes) - np.sum(successes)),
                "distance_to_target_asr": abs(asr - target_asr),
            }
        )
        print(f"[tune] eps={eps_255}/255 steps={steps} ASR={asr:.3f}", flush=True)
    return pd.DataFrame(records).sort_values("distance_to_target_asr")


def collect_trajectories(
    wrapper, dataset, rows, setting, batch_size, device, seed, output
) -> None:
    metadata = []
    arrays: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(rows), batch_size):
        chunk = rows.iloc[start : start + batch_size]
        x, y, ids = load_images(dataset, chunk)
        result = pgd_batch(
            wrapper,
            x.to(device),
            y.to(device),
            ids.to(device),
            float(setting.eps_255) / 255.0,
            int(setting.steps),
            float(setting.step_size_255) / 255.0,
            seed,
            capture=True,
        )
        for local_i, row in enumerate(chunk.itertuples(index=False)):
            metadata.append(
                {
                    "image_ord": int(row.image_ord),
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "split": row.split,
                    "success": int(result["success"][local_i]),
                    "clean_margin": float(result["clean_margin"][local_i]),
                    "clean_loss": float(result["clean_loss"][local_i]),
                    "initial_margin": float(result["initial_margin"][local_i]),
                    "final_margin": float(result["final_margin"][local_i]),
                    "first_step_margin_drop": float(result["first_step_margin_drop"][local_i]),
                    "first_grad_l1": float(result["first_grad_l1"][local_i]),
                    "first_grad_l2": float(result["first_grad_l2"][local_i]),
                }
            )
        for layer, values in result["feature_states"].items():
            arrays.setdefault(layer, []).append(values.astype(np.float16))
        print(f"[collect] {min(start + len(chunk), len(rows))}/{len(rows)}", flush=True)
    pd.DataFrame(metadata).to_csv(output / "trajectory_metadata.csv", index=False)
    np.savez_compressed(
        output / "feature_trajectories.npz",
        **{layer: np.concatenate(parts, axis=0) for layer, parts in arrays.items()},
    )


def fit_pca(x: np.ndarray, max_k: int):
    mean = x.mean(0, keepdims=True)
    centered = x - mean
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular.astype(np.float64) ** 2
    ratio = variance / np.clip(variance.sum(), 1e-12, None)
    return mean.astype(np.float32), vt[:max_k].astype(np.float32), ratio


def projection_energy(x, mean, basis, k):
    centered = x - mean
    coeff = centered @ basis[: min(k, len(basis))].T
    return (coeff * coeff).sum(1) / np.clip((centered * centered).sum(1), 1e-12, None)


def pca_dimensions(ratio):
    csum = np.cumsum(ratio)
    positive = ratio[ratio > 0]
    return {
        "dim80": int(np.searchsorted(csum, 0.80) + 1),
        "dim90": int(np.searchsorted(csum, 0.90) + 1),
        "effective_rank": float(np.exp(-(positive * np.log(positive)).sum())),
    }


def safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))


def bootstrap_auc(labels, scores, reps, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) == 2:
            values.append(roc_auc_score(labels[idx], scores[idx]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def analyze_trajectories(output: Path, bootstrap_reps: int, seed: int):
    meta = pd.read_csv(output / "trajectory_metadata.csv")
    trajectories = np.load(output / "feature_trajectories.npz")
    selection_rows = []
    cache = {}
    for layer in trajectories.files:
        values = trajectories[layer].astype(np.float32)
        local_steps = np.diff(values, axis=1)
        image_vectors = local_steps.mean(axis=1)
        cache[layer] = image_vectors
        fit_mask = (meta.split == "basis_fit") & (meta.success == 1)
        mean, basis, ratio = fit_pca(image_vectors[fit_mask], max(K_VALUES))
        dims = pca_dimensions(ratio)
        for k in K_VALUES:
            val_mask = meta.split == "layer_validation"
            score = projection_energy(image_vectors[val_mask], mean, basis, k)
            selection_rows.append(
                {
                    "layer": layer,
                    "k": k,
                    "validation_auroc": safe_auc(meta.success[val_mask].to_numpy(), score),
                    **dims,
                }
            )
    selection = pd.DataFrame(selection_rows).sort_values("validation_auroc", ascending=False)
    selection.to_csv(output / "layer_k_validation.csv", index=False)
    chosen = selection.iloc[0]
    layer = str(chosen.layer)
    k = int(chosen.k)
    image_vectors = cache[layer]
    fit_mask = (meta.split == "basis_fit") & (meta.success == 1)
    mean, basis, ratio = fit_pca(image_vectors[fit_mask], max(K_VALUES))
    test_mask = meta.split == "final_test"
    energy = projection_energy(image_vectors[test_mask], mean, basis, k)
    labels = meta.success[test_mask].to_numpy(dtype=int)
    auc = safe_auc(labels, energy)
    lo, hi = bootstrap_auc(labels, energy, bootstrap_reps, seed)
    test_rows = meta[test_mask].copy()
    test_rows["transport_energy"] = energy
    test_rows.to_csv(output / "final_test_scores.csv", index=False)
    dims = pca_dimensions(ratio)
    final = {
        "selected_layer": layer,
        "selected_k": k,
        "validation_auroc": float(chosen.validation_auroc),
        "final_test_auroc": auc,
        "final_test_auroc_ci_low": lo,
        "final_test_auroc_ci_high": hi,
        "final_test_n": int(test_mask.sum()),
        "final_test_successes": int(labels.sum()),
        "final_test_failures": int(len(labels) - labels.sum()),
        **dims,
    }

    train_mask = meta.split.isin(["basis_fit", "layer_validation"])
    base_columns = [
        "clean_margin",
        "clean_loss",
        "first_grad_l1",
        "first_grad_l2",
        "first_step_margin_drop",
    ]
    all_energy = projection_energy(image_vectors, mean, basis, k)
    x_base = pd.get_dummies(meta[base_columns + ["label"]], columns=["label"]).to_numpy()
    x_full = np.column_stack([x_base, all_energy])
    y = meta.success.to_numpy(dtype=int)
    base_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    full_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    base_model.fit(x_base[train_mask], y[train_mask])
    full_model.fit(x_full[train_mask], y[train_mask])
    base_score = base_model.predict_proba(x_base[test_mask])[:, 1]
    full_score = full_model.predict_proba(x_full[test_mask])[:, 1]
    final.update(
        {
            "difficulty_base_auprc": float(average_precision_score(labels, base_score)),
            "difficulty_plus_energy_auprc": float(average_precision_score(labels, full_score)),
            "incremental_auprc": float(
                average_precision_score(labels, full_score)
                - average_precision_score(labels, base_score)
            ),
            "test_prevalence": float(labels.mean()),
        }
    )
    write_json(output / "core_diagnostic_summary.json", final)
    np.savez_compressed(
        output / "selected_basis.npz", mean=mean, basis=basis, variance_ratio=ratio
    )
    return final


def first_attack_step(wrapper, x0, y, idx, setting, seed):
    eps = float(setting["eps_255"]) / 255.0
    alpha = float(setting["step_size_255"]) / 255.0
    x = project_linf(x0 + deterministic_noise(x0, idx, eps, seed), x0, eps).detach()
    x_req = x.requires_grad_(True)
    logits = wrapper(x_req)
    grad = torch.autograd.grad(F.cross_entropy(logits, y), x_req)[0]
    x_next = project_linf(x + alpha * grad.sign(), x0, eps).detach()
    return x.detach(), x_next, grad.detach()


def exact_jvp_diagnostics(
    wrapper, dataset, rows, setting, layer, device, seed, max_images, output
):
    records = []
    subset = rows[rows.split == "final_test"].head(max_images)
    for done, row in enumerate(subset.itertuples(index=False), start=1):
        x_cpu, label = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(label)], device=device)
        idx = torch.tensor([int(row.dataset_idx)], device=device)
        x, x_next, grad = first_attack_step(wrapper, x0, y, idx, setting, seed)
        tangent = x_next - x

        def feature_fn(value):
            return wrapper.feature(value, layer)

        feature, jvp = torch.autograd.functional.jvp(
            feature_fn, x, tangent, create_graph=False, strict=False
        )
        with torch.no_grad():
            finite = feature_fn(x_next) - feature
        flat_fd = finite.flatten()
        flat_jvp = jvp.flatten()
        cosine = F.cosine_similarity(flat_fd[None], flat_jvp[None]).item()
        denominator = flat_fd.norm().item() + flat_jvp.norm().item() + 1e-8
        records.append(
            {
                "dataset_idx": int(row.dataset_idx),
                "label": int(label),
                "fd_norm": float(flat_fd.norm()),
                "jvp_norm": float(flat_jvp.norm()),
                "fd_jvp_cosine": cosine,
                "stabilized_residual_ratio": float((flat_fd - flat_jvp).norm()) / denominator,
                "input_step_linf": float(tangent.abs().max()),
                "gradient_l2": float(grad.flatten().norm()),
            }
        )
        if done % 25 == 0:
            print(f"[jvp] {done}/{len(subset)}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "exact_jvp_per_image.csv", index=False)
    rho = spearmanr(frame.fd_norm, frame.jvp_norm).statistic
    summary = {
        "n": len(frame),
        "median_fd_jvp_cosine": float(frame.fd_jvp_cosine.median()),
        "median_stabilized_residual_ratio": float(frame.stabilized_residual_ratio.median()),
        "fd_jvp_norm_spearman": float(rho),
    }
    write_json(output / "exact_jvp_summary.json", summary)
    return summary


def coordinate_stress(output: Path, model, seed: int):
    meta = pd.read_csv(output / "trajectory_metadata.csv")
    values = np.load(output / "feature_trajectories.npz")["penultimate"].astype(np.float32)
    vectors = np.diff(values, axis=1).mean(axis=1)
    if model.architecture == "resnet18":
        classifier = model.backbone.fc
    elif model.architecture == "convnext_tiny":
        classifier = model.backbone.classifier[2]
    else:
        raise ValueError(model.architecture)
    weight = classifier.weight.detach().cpu().numpy().astype(np.float32)
    bias = classifier.bias.detach().cpu().numpy().astype(np.float32)
    reference_features = values[:, 0, :]
    reference_logits = reference_features @ weight.T + bias[None]
    rng = np.random.default_rng(seed)
    rows = []
    for sigma in [0.0, 0.5, 1.0, 2.0]:
        scale = np.ones(vectors.shape[1], dtype=np.float32)
        if sigma:
            scale = np.exp(rng.normal(0.0, sigma, vectors.shape[1])).astype(np.float32)
        transformed = vectors * scale[None]
        compensated_weight = weight / scale[None]
        transformed_logits = (reference_features * scale[None]) @ compensated_weight.T + bias[None]
        max_logit_error = float(np.max(np.abs(reference_logits - transformed_logits)))
        fit = (meta.split == "basis_fit") & (meta.success == 1)
        mean, basis, ratio = fit_pca(transformed[fit], max(K_VALUES))
        test = meta.split == "final_test"
        scores = projection_energy(transformed[test], mean, basis, 20)
        dims = pca_dimensions(ratio)
        rows.append(
            {
                "transform": "identity" if sigma == 0 else f"function_preserving_diag_sigma_{sigma}",
                "sigma": sigma,
                "auroc_k20": safe_auc(meta.success[test].to_numpy(), scores),
                **dims,
                "max_compensated_logit_error": max_logit_error,
                "logit_equivalence_verified": max_logit_error < 1e-4,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "function_preserving_coordinate_stress.csv", index=False)


def strong_scope_attacks(model, dataset, rows, args, device, output):
    scope = rows[rows.split == "final_test"].head(args.scope_images)
    x_cpu, y_cpu, _ids = load_images(dataset, scope)
    x_all = x_cpu.to(device)
    y_all = y_cpu.to(device)
    records = []
    from autoattack.autopgd_pt import APGDAttack
    from autoattack.square import SquareAttack

    attacks = [
        (
            "official_apgd_ce50",
            APGDAttack(
                model,
                n_iter=50,
                norm="Linf",
                n_restarts=1,
                eps=args.strong_eps / 255.0,
                seed=args.seed,
                loss="ce",
                verbose=False,
                device=str(device),
            ),
        ),
        (
            f"official_square_q{args.square_queries}",
            SquareAttack(
                model,
                norm="Linf",
                n_queries=args.square_queries,
                eps=args.strong_eps / 255.0,
                p_init=0.8,
                n_restarts=1,
                seed=args.seed,
                verbose=False,
                targeted=False,
                loss="margin",
                device=str(device),
            ),
        ),
    ]
    for name, attack in attacks:
        parts = []
        started = time.perf_counter()
        batch_size = args.scope_batch_size if "apgd" in name else min(32, args.scope_batch_size)
        for start in range(0, len(x_all), batch_size):
            result = attack.perturb(
                x_all[start : start + batch_size], y_all[start : start + batch_size]
            )
            if isinstance(result, tuple):
                result = result[1]
            parts.append(result.detach())
        adv = torch.cat(parts)
        with torch.no_grad():
            pred = model(adv).argmax(1)
        success = (pred != y_all).cpu().numpy().astype(int)
        linf = (adv - x_all).abs().flatten(1).max(1).values.cpu().numpy()
        for row, ok, norm in zip(scope.itertuples(index=False), success, linf):
            records.append(
                {
                    "method": name,
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(row.label),
                    "success": int(ok),
                    "linf": float(norm),
                }
            )
        print(f"[scope] {name} ASR={success.mean():.3f} time={time.perf_counter()-started:.1f}s", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "strong_attack_per_image.csv", index=False)
    frame.groupby("method").agg(asr=("success", "mean"), n=("success", "size"), median_linf=("linf", "median")).reset_index().to_csv(
        output / "strong_attack_summary.csv", index=False
    )


def run_model(args, architecture, device):
    output = Path(args.output_dir) / architecture
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    required = [
        output / "feature_trajectories.npz",
        output / "core_diagnostic_summary.json",
        output / "exact_jvp_summary.json",
        output / "function_preserving_coordinate_stress.csv",
    ]
    if args.strong_scope:
        required.append(output / "strong_attack_summary.csv")
    if manifest_path.exists() and all(path.exists() for path in required) and not args.overwrite:
        print(f"[{architecture}] complete; skipping", flush=True)
        return
    checkpoint = Path(args.checkpoint_dir) / architecture / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model, payload = load_checkpoint(checkpoint, device)
    wrapper = FeatureCapture(model)
    dataset = gtsrb_dataset(
        args.data_dir,
        "test",
        int(payload["image_size"]),
        training=False,
        download=False,
    )
    try:
        selection_path = output / "image_split.csv"
        if selection_path.exists():
            rows = pd.read_csv(selection_path)
        else:
            rows = select_clean_correct(model, dataset, args.images, args.batch_size, device)
            rows = assign_splits(rows, args.split_seed)
            rows.to_csv(selection_path, index=False)

        tune_path = output / "weak_attack_grid.csv"
        if tune_path.exists():
            tune = pd.read_csv(tune_path)
        else:
            tune_rows = rows[rows.split == "layer_validation"].head(args.tune_images)
            tune = tune_weak_attack(
                wrapper,
                dataset,
                tune_rows,
                args.batch_size,
                device,
                args.target_asr,
                args.seed,
            )
            tune.to_csv(tune_path, index=False)
        setting = tune.iloc[0]
        write_json(output / "selected_weak_attack.json", setting.to_dict())

        trajectory_path = output / "feature_trajectories.npz"
        if not trajectory_path.exists():
            collect_trajectories(
                wrapper,
                dataset,
                rows,
                setting,
                args.batch_size,
                device,
                args.seed,
                output,
            )
        core = analyze_trajectories(output, args.bootstrap_reps, args.seed)
        exact = exact_jvp_diagnostics(
            wrapper,
            dataset,
            rows,
            setting.to_dict(),
            core["selected_layer"],
            device,
            args.seed,
            args.jvp_images,
            output,
        )
        coordinate_stress(output, model, args.seed + 77)
        if args.strong_scope and not (output / "strong_attack_summary.csv").exists():
            strong_scope_attacks(model, dataset, rows, args, device, output)
        write_json(
            manifest_path,
            {
                "architecture": architecture,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
                "image_size": int(payload["image_size"]),
                "layers": wrapper.layers,
                "selected_weak_attack": setting.to_dict(),
                "core_summary": core,
                "jvp_summary": exact,
                "args": vars(args),
            },
        )
    finally:
        wrapper.close()


def combine(args):
    rows = []
    for architecture in args.models.split(","):
        architecture = architecture.strip()
        path = Path(args.output_dir) / architecture / "manifest.json"
        if path.exists():
            value = json.loads(path.read_text())
            rows.append(
                {
                    "architecture": architecture,
                    **value["selected_weak_attack"],
                    **value["core_summary"],
                    **value["jvp_summary"],
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(Path(args.output_dir) / "gtsrb_compact_audit_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/gtsrb")
    parser.add_argument("--checkpoint-dir", default="analysis_outputs/eaai_gtsrb/checkpoints")
    parser.add_argument("--output-dir", default="analysis_outputs/eaai_gtsrb/diagnostics")
    parser.add_argument("--models", default="resnet18,convnext_tiny")
    parser.add_argument("--images", type=int, default=1000)
    parser.add_argument("--tune-images", type=int, default=300)
    parser.add_argument("--jvp-images", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-asr", type=float, default=0.65)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--strong-scope", action="store_true")
    parser.add_argument("--strong-eps", type=float, default=8.0)
    parser.add_argument("--scope-images", type=int, default=200)
    parser.add_argument("--scope-batch-size", type=int, default=25)
    parser.add_argument("--square-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--split-seed", type=int, default=1307)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for architecture in [x.strip() for x in args.models.split(",") if x.strip()]:
        run_model(args, architecture, device)
        combine(args)
    combine(args)


if __name__ == "__main__":
    main()
