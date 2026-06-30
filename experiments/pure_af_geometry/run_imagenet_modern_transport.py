#!/usr/bin/env python3
"""Resumable ImageNet transport pipeline for CNN/ConvNeXt/ViT models.

The script is intentionally artifact-first:

* an image manifest is independent of any model;
* clean predictions are cached per model;
* trajectory feature-state shards are cached per model/attack/config;
* analysis consumes all available shards and can be rerun without attacks.

This lets us add models, layers, attacks, or more images later without rerunning
completed shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODEL_REGISTRY = {
    "resnet50": {
        "ctor": models.resnet50,
        "weights": "ResNet50_Weights",
        "layers": {
            "stage1": "layer1",
            "stage2": "layer2",
            "stage3": "layer3",
            "stage4": "layer4",
            "penultimate": "avgpool",
        },
    },
    "densenet121": {
        "ctor": models.densenet121,
        "weights": "DenseNet121_Weights",
        "layers": {
            "stage1": "features.denseblock1",
            "stage2": "features.denseblock2",
            "stage3": "features.denseblock3",
            "stage4": "features.denseblock4",
            "penultimate": "features.norm5",
        },
    },
    "vgg16_bn": {
        "ctor": models.vgg16_bn,
        "weights": "VGG16_BN_Weights",
        "layers": {
            "stage1": "features.6",
            "stage2": "features.13",
            "stage3": "features.23",
            "stage4": "features.33",
            "penultimate": "classifier.4",
        },
    },
    "convnext_tiny": {
        "ctor": models.convnext_tiny,
        "weights": "ConvNeXt_Tiny_Weights",
        "layers": {
            "stage1": "features.1",
            "stage2": "features.3",
            "stage3": "features.5",
            "stage4": "features.7",
            "penultimate": "avgpool",
        },
    },
    "vit_b_16": {
        "ctor": models.vit_b_16,
        "weights": "ViT_B_16_Weights",
        "layers": {
            "stage1": "encoder.layers.encoder_layer_2",
            "stage2": "encoder.layers.encoder_layer_5",
            "stage3": "encoder.layers.encoder_layer_8",
            "stage4": "encoder.layers.encoder_layer_11",
            "block3": "encoder.layers.encoder_layer_2",
            "block6": "encoder.layers.encoder_layer_5",
            "block9": "encoder.layers.encoder_layer_8",
            "block12": "encoder.layers.encoder_layer_11",
            "penultimate": "encoder.ln",
        },
    },
}


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def stable_hash(obj: dict) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def normalize_input(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(x.device, dtype=x.dtype)
    return (x - mean) / std


def margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.view(-1, 1), -1e9)
    other = masked.max(1).values
    return true - other


def project_linf(x_adv: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(torch.max(torch.min(x_adv, x0 + eps), x0 - eps), 0.0, 1.0)


def vectorize_feature(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, (tuple, list)):
        x = x[0]
    if x.ndim == 4:
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    elif x.ndim == 3:
        x = x[:, 0, :]  # CLS token for ViTs; harmless for sequence-like outputs.
    elif x.ndim > 2:
        x = x.flatten(1)
    return x.detach().float().cpu()


class FeatureModel(nn.Module):
    def __init__(self, model_name: str, layer_groups: list[str], device: torch.device):
        super().__init__()
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model {model_name}. Available: {sorted(MODEL_REGISTRY)}")
        spec = MODEL_REGISTRY[model_name]
        weights_enum = getattr(models, spec["weights"])
        weights = weights_enum.DEFAULT
        self.base = spec["ctor"](weights=weights).to(device).eval()
        self.model_name = model_name
        self.device = device
        self.layer_map = {g: spec["layers"][g] for g in layer_groups if g in spec["layers"]}
        self.layer_map["logits"] = "logits"
        missing = [g for g in layer_groups if g not in self.layer_map]
        if missing:
            print(f"[WARN] Unsupported layer groups for {model_name}; skipping: {missing}")
        self._features: dict[str, torch.Tensor] = {}
        modules = dict(self.base.named_modules())
        self._handles = []
        for group, module_name in self.layer_map.items():
            if group == "logits":
                continue
            if module_name not in modules:
                raise ValueError(f"Missing module {module_name} for {model_name}.")
            self._handles.append(modules[module_name].register_forward_hook(self._make_hook(group)))

    def _make_hook(self, group: str):
        def hook(_module, _inputs, output):
            self._features[group] = vectorize_feature(output)

        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(normalize_input(x))

    @torch.no_grad()
    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        self._features = {}
        logits = self.forward(x)
        feats = {k: v.numpy()[0].astype(np.float32) for k, v in self._features.items()}
        feats["logits"] = logits.detach().float().cpu().numpy()[0].astype(np.float32)
        return logits, feats

    def close(self) -> None:
        for h in self._handles:
            h.remove()


def build_dataset(root: str):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    return datasets.ImageFolder(root, transform=transform)


def manifest_path(out_dir: Path, images: int, seed: int) -> Path:
    return out_dir / "manifests" / f"imagenet_manifest_seed{seed}_n{images}.csv"


def ensure_manifest(dataset, out_dir: Path, images: int, seed: int) -> Path:
    out = manifest_path(out_dir, images, seed)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(dataset))
    rng.shuffle(all_indices)
    chosen = all_indices[:images]
    rows = []
    for order, idx in enumerate(chosen):
        path, label = dataset.samples[int(idx)]
        rows.append(
            {
                "image_ord": order,
                "dataset_idx": int(idx),
                "label": int(label),
                "path": path,
                "class_name": dataset.classes[int(label)],
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def clean_cache_path(out_dir: Path, model_name: str, manifest: Path) -> Path:
    return out_dir / "clean_predictions" / f"{model_name}__{manifest.stem}.csv"


def ensure_clean_predictions(dataset, manifest_df: pd.DataFrame, out_dir: Path, model_name: str, batch_size: int, device: torch.device) -> Path:
    out = clean_cache_path(out_dir, model_name, Path(manifest_df.attrs["path"]))
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    model = FeatureModel(model_name, ["logits"], device)
    rows = []
    with torch.no_grad():
        for start in range(0, len(manifest_df), batch_size):
            sub = manifest_df.iloc[start : start + batch_size]
            xs = []
            ys = []
            for r in sub.itertuples(index=False):
                x, y = dataset[int(r.dataset_idx)]
                xs.append(x)
                ys.append(y)
            x = torch.stack(xs).to(device)
            y = torch.tensor(ys, device=device)
            logits = model(x)
            pred = logits.argmax(1)
            m = margin(logits, y)
            prob = torch.softmax(logits, 1).gather(1, y.view(-1, 1)).squeeze(1)
            for row, p, mm, pp in zip(sub.itertuples(index=False), pred.cpu(), m.cpu(), prob.cpu()):
                rows.append(
                    {
                        "image_ord": int(row.image_ord),
                        "dataset_idx": int(row.dataset_idx),
                        "label": int(row.label),
                        "pred": int(p),
                        "correct": int(p == row.label),
                        "margin": float(mm),
                        "p_y": float(pp),
                    }
                )
    model.close()
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def pgd_step(model: FeatureModel, x_adv: torch.Tensor, x0: torch.Tensor, y: torch.Tensor, eps: float, step_size: float):
    z = x_adv.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(z), y)
    grad = torch.autograd.grad(loss, z)[0]
    return project_linf(z + step_size * grad.sign(), x0, eps).detach()


def square_size(step: int, max_steps: int, h: int, min_square: int) -> int:
    progress = step / max(max_steps, 1)
    frac = 0.7 * (1.0 - progress) + 0.08 * progress
    return int(max(min_square, min(h, round(h * frac))))


def square_p_selection_benchmark(p_init: float, it: int, n_queries: int) -> float:
    """Piecewise-constant p schedule from Square Attack."""
    it = int(it / max(n_queries, 1) * 10000)
    if 10 < it <= 50:
        return p_init / 2
    if 50 < it <= 200:
        return p_init / 4
    if 200 < it <= 500:
        return p_init / 8
    if 500 < it <= 1000:
        return p_init / 16
    if 1000 < it <= 2000:
        return p_init / 32
    if 2000 < it <= 4000:
        return p_init / 64
    if 4000 < it <= 6000:
        return p_init / 128
    if 6000 < it <= 8000:
        return p_init / 256
    if 8000 < it:
        return p_init / 512
    return p_init


def square_step(
    model: FeatureModel,
    x_adv: torch.Tensor,
    x0: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    step: int,
    max_steps: int,
    min_square: int,
    gen: torch.Generator,
    best_margin: torch.Tensor,
):
    _b, c, h, w = x0.shape
    side = square_size(step, max_steps, h, min_square)
    top = int(torch.randint(0, h - side + 1, (1,), generator=gen, device=x0.device).item())
    left = int(torch.randint(0, w - side + 1, (1,), generator=gen, device=x0.device).item())
    candidate = x_adv.clone()
    patch = (torch.rand((1, c, side, side), generator=gen, device=x0.device) * 2.0 - 1.0) * eps
    candidate[:, :, top : top + side, left : left + side] = x0[:, :, top : top + side, left : left + side] + patch
    candidate = project_linf(candidate, x0, eps)
    with torch.no_grad():
        cand_margin = margin(model(candidate), y)
    if float(cand_margin.item()) < float(best_margin.item()):
        return candidate.detach(), cand_margin.detach()
    return x_adv.detach(), best_margin.detach()


def square_benchmark_init(
    model: FeatureModel,
    x0: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    gen: torch.Generator,
):
    _b, c, _h, w = x0.shape
    signs = torch.where(
        torch.rand((1, c, 1, w), generator=gen, device=x0.device) < 0.5,
        -torch.ones((1, c, 1, w), device=x0.device),
        torch.ones((1, c, 1, w), device=x0.device),
    )
    x_best = torch.clamp(x0 + eps * signs, 0.0, 1.0)
    with torch.no_grad():
        best_margin = margin(model(x_best), y)
    return x_best.detach(), best_margin.detach()


def square_benchmark_step(
    model: FeatureModel,
    x_best: torch.Tensor,
    x0: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    i_iter: int,
    n_queries: int,
    p_init: float,
    gen: torch.Generator,
    best_margin: torch.Tensor,
):
    """Linf Square Attack update, matching the AutoAttack/Square schedule."""
    _b, c, h, w = x0.shape
    n_features = c * h * w
    p = square_p_selection_benchmark(p_init, i_iter, n_queries)
    side = max(int(round(math.sqrt(p * n_features / c))), 1)
    side = min(side, h, w)
    top = int(torch.randint(0, h - side + 1, (1,), generator=gen, device=x0.device).item())
    left = int(torch.randint(0, w - side + 1, (1,), generator=gen, device=x0.device).item())
    new_deltas = torch.zeros_like(x_best)
    signs = torch.where(
        torch.rand((1, c, 1, 1), generator=gen, device=x0.device) < 0.5,
        -torch.ones((1, c, 1, 1), device=x0.device),
        torch.ones((1, c, 1, 1), device=x0.device),
    )
    new_deltas[:, :, top : top + side, left : left + side] = 2.0 * eps * signs
    x_new = x_best + new_deltas
    x_new = torch.min(torch.max(x_new, x0 - eps), x0 + eps)
    x_new = torch.clamp(x_new, 0.0, 1.0)
    with torch.no_grad():
        cand_margin = margin(model(x_new), y)
    improved = (cand_margin < best_margin) | (cand_margin <= 0.0)
    if bool(improved.item()):
        return x_new.detach(), cand_margin.detach()
    return x_best.detach(), best_margin.detach()


def shard_id(config: dict) -> str:
    return stable_hash(config)


def collect_shard(
    dataset,
    manifest_df: pd.DataFrame,
    out_dir: Path,
    model_name: str,
    attack: str,
    layer_groups: list[str],
    eps: float,
    steps: int,
    step_size: float,
    image_limit: int,
    seed: int,
    batch_start: int,
    batch_count: int,
    device: torch.device,
    square_min_size: int,
    record_every: int,
    square_p_init: float,
):
    cfg = {
        "model": model_name,
        "attack": attack,
        "layers": layer_groups,
        "eps": eps,
        "steps": steps,
        "step_size": step_size,
        "image_limit": image_limit,
        "seed": seed,
        "batch_start": batch_start,
        "batch_count": batch_count,
        "record_every": record_every,
        "square_p_init": square_p_init,
        "manifest": manifest_df.attrs["path"],
        "version": 1,
    }
    sid = shard_id(cfg)
    shard_dir = out_dir / "trajectory_shards" / model_name / attack
    shard_dir.mkdir(parents=True, exist_ok=True)
    out_npz = shard_dir / f"states_{sid}.npz"
    out_meta = shard_dir / f"meta_{sid}.csv"
    out_json = shard_dir / f"config_{sid}.json"
    if out_npz.exists() and out_meta.exists() and out_json.exists():
        print(f"[SKIP] existing shard {model_name}/{attack} {sid}")
        return

    clean = pd.read_csv(clean_cache_path(out_dir, model_name, Path(manifest_df.attrs["path"])))
    selected = clean[clean.correct == 1].merge(manifest_df, on=["image_ord", "dataset_idx", "label"], how="left")
    selected = selected.sort_values("image_ord").head(image_limit)
    selected = selected.iloc[batch_start : batch_start + batch_count]
    if selected.empty:
        raise RuntimeError(f"No selected clean-correct images for {model_name}")

    model = FeatureModel(model_name, layer_groups, device)
    arrays: dict[str, list[np.ndarray]] = {g: [] for g in model.layer_map}
    meta_rows = []
    eps_f = eps / 255.0
    step_f = step_size / 255.0 if step_size > 0 else eps_f / max(steps, 1)

    for local_ord, row in enumerate(selected.itertuples(index=False)):
        print(
            f"[RUN] {model_name}/{attack} image {local_ord + 1}/{len(selected)} "
            f"(dataset_idx={int(row.dataset_idx)}, label={int(row.label)})",
            flush=True,
        )
        x_cpu, y_int = dataset[int(row.dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        x_adv = x0.clone()
        y = torch.tensor([int(y_int)], device=device)
        gen = torch.Generator(device=device).manual_seed(seed + int(row.dataset_idx) * 9973 + local_ord)
        with torch.no_grad():
            best_margin = margin(model(x_adv), y)
        image_meta_start = len(meta_rows)
        final_success = 0
        for step in range(steps + 1):
            should_record = step == 0 or step >= steps or (step % max(record_every, 1) == 0)
            if should_record:
                logits, feats = model.forward_with_features(x_adv)
                pred = int(logits.argmax(1).item())
                m = float(margin(logits, y).item())
                py = float(torch.softmax(logits, 1).gather(1, y.view(1, 1)).item())
                now_success = int(pred != int(y_int))
                vector_idx = len(arrays["logits"])
                for g, v in feats.items():
                    arrays[g].append(v.astype(np.float32))
                meta_rows.append(
                    {
                        "model": model_name,
                        "attack": attack,
                        "image_ord": int(row.image_ord),
                        "dataset_idx": int(row.dataset_idx),
                        "label": int(y_int),
                        "step": int(step),
                        "pred": pred,
                        "success_at_step": now_success,
                        "margin": m,
                        "p_y": py,
                        "vector_idx": vector_idx,
                    }
                )
            else:
                with torch.no_grad():
                    pred = int(model(x_adv).argmax(1).item())
                now_success = int(pred != int(y_int))
            final_success = max(final_success, now_success)
            if attack == "square_benchmark" and now_success and step > 0:
                break
            if step >= steps:
                break
            if attack == "pgd":
                x_adv = pgd_step(model, x_adv, x0, y, eps_f, step_f)
            elif attack == "square":
                x_adv, best_margin = square_step(
                    model, x_adv, x0, y, eps_f, step + 1, steps, square_min_size, gen, best_margin
                )
            elif attack == "square_benchmark":
                if step == 0:
                    x_adv, best_margin = square_benchmark_init(model, x0, y, eps_f, gen)
                else:
                    x_adv, best_margin = square_benchmark_step(
                        model, x_adv, x0, y, eps_f, step - 1, steps, square_p_init, gen, best_margin
                    )
            else:
                raise ValueError(attack)

        if meta_rows and meta_rows[-1]["dataset_idx"] == int(row.dataset_idx) and meta_rows[-1]["step"] != int(step):
            logits, feats = model.forward_with_features(x_adv)
            pred = int(logits.argmax(1).item())
            m = float(margin(logits, y).item())
            py = float(torch.softmax(logits, 1).gather(1, y.view(1, 1)).item())
            now_success = int(pred != int(y_int))
            final_success = max(final_success, now_success)
            vector_idx = len(arrays["logits"])
            for g, v in feats.items():
                arrays[g].append(v.astype(np.float32))
            meta_rows.append(
                {
                    "model": model_name,
                    "attack": attack,
                    "image_ord": int(row.image_ord),
                    "dataset_idx": int(row.dataset_idx),
                    "label": int(y_int),
                    "step": int(step),
                    "pred": pred,
                    "success_at_step": now_success,
                    "margin": m,
                    "p_y": py,
                    "vector_idx": vector_idx,
                }
            )
        for r in meta_rows[image_meta_start:]:
            r["final_success"] = int(final_success)

    packed = {f"states__{g}": np.stack(vals).astype(np.float32) for g, vals in arrays.items()}
    np.savez_compressed(out_npz, **packed)
    pd.DataFrame(meta_rows).to_csv(out_meta, index=False)
    out_json.write_text(json.dumps(cfg, indent=2) + "\n")
    model.close()
    print(f"[SAVED] {out_npz}")


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def pca_basis(x: np.ndarray, max_k: int):
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s * s
    ratios = var / np.clip(var.sum(), 1e-12, None)
    return mean.astype(np.float32), vt[:max_k].astype(np.float32), ratios.astype(np.float32)


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, len(basis))
    xc = x - mean
    coeff = xc @ basis[:kk].T
    return np.sum(coeff * coeff, axis=1) / np.clip(np.sum(xc * xc, axis=1), 1e-12, None)


def dim_stats(x: np.ndarray) -> dict:
    if len(x) < 3:
        return {"pc1_var": np.nan, "pc10_cum_var": np.nan, "dim80": np.nan, "dim90": np.nan, "effective_rank": np.nan}
    _mean, _basis, ratios = pca_basis(x, min(x.shape[0], x.shape[1]))
    csum = np.cumsum(ratios)
    entropy = -float(np.sum(ratios[ratios > 0] * np.log(ratios[ratios > 0])))
    return {
        "pc1_var": float(ratios[0]),
        "pc10_cum_var": float(csum[min(9, len(csum) - 1)]),
        "dim80": int(np.searchsorted(csum, 0.8) + 1),
        "dim90": int(np.searchsorted(csum, 0.9) + 1),
        "effective_rank": float(np.exp(entropy)),
    }


def load_all_shards(out_dir: Path):
    rows = []
    shard_infos = []
    for meta_path in sorted((out_dir / "trajectory_shards").glob("*/*/meta_*.csv")):
        npz_path = meta_path.with_name(meta_path.name.replace("meta_", "states_").replace(".csv", ".npz"))
        if not npz_path.exists():
            continue
        meta = pd.read_csv(meta_path)
        meta["shard_meta_path"] = str(meta_path)
        meta["shard_npz_path"] = str(npz_path)
        rows.append(meta)
        shard_infos.append((meta_path, npz_path))
    if not rows:
        raise RuntimeError(f"No trajectory shards found under {out_dir}")
    return pd.concat(rows, ignore_index=True), shard_infos


def transport_vectors_for(meta: pd.DataFrame, npz_cache: dict[str, np.lib.npyio.NpzFile], layer: str) -> tuple[pd.DataFrame, np.ndarray]:
    vector_rows = []
    vectors = []
    group_cols = ["model", "attack", "dataset_idx", "shard_npz_path"]
    for (_model, _attack, _idx, npz_path), g in meta.sort_values("step").groupby(group_cols, dropna=False):
        key = f"states__{layer}"
        if key not in npz_cache[npz_path].files:
            continue
        arr = npz_cache[npz_path][key]
        idx = g["vector_idx"].to_numpy(int)
        states = arr[idx]
        if len(states) < 2:
            continue
        local = states[1:] - states[:-1]
        local = normalize_rows(local.astype(np.float32))
        base = g.iloc[:-1].copy()
        base["segment_step"] = g["step"].to_numpy()[1:]
        vector_rows.append(base)
        vectors.append(local)
    if not vectors:
        return pd.DataFrame(), np.empty((0, 1), dtype=np.float32)
    return pd.concat(vector_rows, ignore_index=True), np.concatenate(vectors, axis=0)


def analyze(out_dir: Path, ks: list[int], seed: int):
    out_analysis = out_dir / "analysis"
    out_analysis.mkdir(parents=True, exist_ok=True)
    meta, shard_infos = load_all_shards(out_dir)
    npz_cache = {str(npz): np.load(npz, allow_pickle=False) for _m, npz in shard_infos}
    layers = sorted({k.replace("states__", "") for npz in npz_cache.values() for k in npz.files})
    rng = np.random.default_rng(seed)
    dim_rows = []
    metric_rows = []
    signature_rows = []

    for (model, attack), m0 in meta.groupby(["model", "attack"], dropna=False):
        for layer in layers:
            group_shards = set(m0.shard_npz_path)
            if not any(f"states__{layer}" in npz_cache[path].files for path in group_shards):
                continue
            rows, x = transport_vectors_for(m0, npz_cache, layer)
            if len(rows) < 20:
                continue
            success = rows["final_success"].to_numpy(int) == 1
            if success.sum() < 10:
                continue
            xs = x[success]
            xf = x[~success]
            dim_rows.append({"model": model, "attack": attack, "layer": layer, "n_success": int(len(xs)), **dim_stats(xs)})
            # Stable image split.
            image_ids = np.array(sorted(rows["dataset_idx"].unique()))
            rng.shuffle(image_ids)
            train_ids = set(image_ids[: max(1, int(0.6 * len(image_ids)))])
            train = rows["dataset_idx"].isin(train_ids).to_numpy() & success
            test_success = (~rows["dataset_idx"].isin(train_ids).to_numpy()) & success
            test_failed = (~rows["dataset_idx"].isin(train_ids).to_numpy()) & (~success)
            if train.sum() < 10 or test_success.sum() < 5:
                continue
            mean, basis, _ratios = pca_basis(x[train], max(ks))
            for k in ks:
                rand = rng.normal(size=(max(test_success.sum(), 1000), x.shape[1])).astype(np.float32)
                rand = normalize_rows(rand)
                es = projection_energy(x[test_success], mean, basis, k)
                er = projection_energy(rand, mean, basis, k)
                y = np.concatenate([np.ones_like(es), np.zeros_like(er)])
                score = np.concatenate([es, er])
                metric_rows.append(
                    {
                        "model": model,
                        "attack": attack,
                        "layer": layer,
                        "comparison": "success_vs_random",
                        "k": k,
                        "auroc": float(roc_auc_score(y, score)),
                        "success_mean_energy": float(np.mean(es)),
                        "negative_mean_energy": float(np.mean(er)),
                        "n_success": int(len(es)),
                        "n_negative": int(len(er)),
                    }
                )
                if test_failed.sum() >= 5:
                    ef = projection_energy(x[test_failed], mean, basis, k)
                    y = np.concatenate([np.ones_like(es), np.zeros_like(ef)])
                    score = np.concatenate([es, ef])
                    metric_rows.append(
                        {
                            "model": model,
                            "attack": attack,
                            "layer": layer,
                            "comparison": "success_vs_failed",
                            "k": k,
                            "auroc": float(roc_auc_score(y, score)),
                            "success_mean_energy": float(np.mean(es)),
                            "negative_mean_energy": float(np.mean(ef)),
                            "n_success": int(len(es)),
                            "n_negative": int(len(ef)),
                        }
                    )
            coeff = (x[success] - mean) @ basis[:5].T
            energy = np.mean(coeff * coeff, axis=0)
            frac = energy / np.clip(np.sum(energy), 1e-12, None)
            signature_rows.append(
                {"model": model, "attack": attack, "layer": layer, **{f"pc{i+1}_frac": float(frac[i]) for i in range(len(frac))}}
            )

    pd.DataFrame(dim_rows).to_csv(out_analysis / "modern_imagenet_dimensionality.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_analysis / "modern_imagenet_projection_metrics.csv", index=False)
    sig = pd.DataFrame(signature_rows)
    sig.to_csv(out_analysis / "modern_imagenet_transport_signatures.csv", index=False)
    sim_rows = []
    if not sig.empty:
        for (model, layer), g in sig.groupby(["model", "layer"]):
            attacks = sorted(g.attack.unique())
            for i, a in enumerate(attacks):
                for b in attacks[i + 1 :]:
                    va = g[g.attack == a][[f"pc{j}_frac" for j in range(1, 6)]].iloc[0].to_numpy(float)
                    vb = g[g.attack == b][[f"pc{j}_frac" for j in range(1, 6)]].iloc[0].to_numpy(float)
                    cos = float(np.dot(va, vb) / np.clip(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12, None))
                    sim_rows.append({"model": model, "layer": layer, "attack_a": a, "attack_b": b, "signature_cosine": cos})
    pd.DataFrame(sim_rows).to_csv(out_analysis / "modern_imagenet_optimizer_similarity.csv", index=False)
    print(f"[ANALYSIS] wrote {out_analysis}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["collect", "analyze", "all"], default="all")
    ap.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    ap.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/imagenet_modern_transport_v1")
    ap.add_argument("--models", default="resnet50,convnext_tiny,vit_b_16")
    ap.add_argument("--attacks", default="pgd,square")
    ap.add_argument("--layer-groups", default="stage1,stage2,stage3,stage4,penultimate,logits")
    ap.add_argument("--images", type=int, default=200)
    ap.add_argument("--image-limit", type=int, default=120)
    ap.add_argument("--batch-start", type=int, default=0)
    ap.add_argument("--batch-count", type=int, default=120)
    ap.add_argument("--manifest-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=4.0)
    ap.add_argument("--pgd-steps", type=int, default=10)
    ap.add_argument("--square-steps", type=int, default=120)
    ap.add_argument("--step-size", type=float, default=1.0)
    ap.add_argument("--square-min-size", type=int, default=8)
    ap.add_argument("--square-p-init", type=float, default=0.8)
    ap.add_argument("--record-every", type=int, default=1)
    ap.add_argument("--clean-batch-size", type=int, default=32)
    ap.add_argument("--ks", default="5,10,20,50")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args.imagenet_root)
    manifest = ensure_manifest(dataset, out_dir, args.images, args.manifest_seed)
    manifest_df = pd.read_csv(manifest)
    manifest_df.attrs["path"] = str(manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = parse_csv(args.models)
    attacks = parse_csv(args.attacks)
    layer_groups = parse_csv(args.layer_groups)

    run_meta = {
        "args": vars(args),
        "manifest": str(manifest),
        "available_models": sorted(MODEL_REGISTRY),
        "note": "CLIP/foundation encoders are not collected in this run unless added as model registry entries; existing shards remain reusable.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    if args.stage in {"collect", "all"}:
        for model_name in model_names:
            ensure_clean_predictions(dataset, manifest_df, out_dir, model_name, args.clean_batch_size, device)
            for attack in attacks:
                steps = args.pgd_steps if attack == "pgd" else args.square_steps
                collect_shard(
                    dataset,
                    manifest_df,
                    out_dir,
                    model_name,
                    attack,
                    layer_groups,
                    args.eps,
                    steps,
                    args.step_size,
                    args.image_limit,
                    args.seed,
                    args.batch_start,
                    args.batch_count,
                    device,
                    args.square_min_size,
                    args.record_every if attack in {"square_benchmark"} else 1,
                    args.square_p_init,
                )

    if args.stage in {"analyze", "all"}:
        analyze(out_dir, [int(k) for k in parse_csv(args.ks)], args.seed)


if __name__ == "__main__":
    main()
