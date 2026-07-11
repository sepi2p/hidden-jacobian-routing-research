#!/usr/bin/env python3
"""Create exact CIFAR-10 splits for the Q1 reviewer validation protocol.

For each requested checkpoint, this script selects the first 100 clean-correct
test images per class under sorted CIFAR-10 index order.  For each split seed,
it then creates a class-balanced 40/20/40 basis-fit/layer-validation/final-test
split.  The output is intended to be the common image table for all promoted
Q1 validation experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import CIFARFeatureWrapper, margin  # noqa: E402
from utils.load_models import load_cifar_model  # noqa: E402


DEFAULT_MODELS = ["bbb_resnet50", "bbb_vgg19_bn", "bbb_densenet", "bbb_inception_v3"]
DEFAULT_SPLIT_SEEDS = [1001, 1002, 1003]


LAYER_CANDIDATES = {
    "bbb_resnet50": ["layer1", "layer2", "layer3", "layer4", "avgpool", "logits"],
    "bbb_vgg19_bn": ["block1", "block2", "block3", "block4", "block5", "penultimate", "logits"],
    "bbb_densenet": ["denseblock1", "denseblock2", "denseblock3", "penultimate", "logits"],
    "bbb_inception_v3": ["mixed5", "mixed6", "mixed7", "penultimate", "logits"],
}


ATTACK_REGISTRY = [
    {"attack": "pgd_ce_mixed", "loss": "ce", "eps_255": 2, "steps_or_queries": 5, "step_size_255": 0.5, "random_starts": 1, "purpose": "nested_layer_selection"},
    {"attack": "pgd_ce20", "loss": "ce", "eps_255": 8, "steps_or_queries": 20, "step_size_255": None, "random_starts": 1, "purpose": "full_budget_trajectory"},
    {"attack": "apgd_ce50", "loss": "ce", "eps_255": 8, "steps_or_queries": 50, "step_size_255": None, "random_starts": 1, "purpose": "full_budget_trajectory"},
    {"attack": "apgd_dlr50", "loss": "dlr", "eps_255": 8, "steps_or_queries": 50, "step_size_255": None, "random_starts": 1, "purpose": "full_budget_trajectory"},
    {"attack": "square5000", "loss": "ce_margin", "eps_255": 8, "steps_or_queries": 5000, "step_size_255": None, "random_starts": 0, "purpose": "full_budget_trajectory", "p_init": 0.8},
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def model_weight_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for key, tensor in sorted(model.state_dict().items()):
            h.update(key.encode("utf-8"))
            arr = tensor.detach().cpu().contiguous().numpy()
            h.update(arr.tobytes())
    return h.hexdigest()


def clean_correct_table(model_name: str, dataset, device: torch.device, per_class: int, batch_size: int) -> tuple[pd.DataFrame, dict]:
    base = load_cifar_model(model_name).to(device).eval()
    wrapper = CIFARFeatureWrapper(model_name, base).to(device).eval()
    rows: list[dict] = []
    counts = {c: 0 for c in range(10)}

    xs, ys, idxs = [], [], []
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        xs.append(x)
        ys.append(int(y))
        idxs.append(idx)
        if len(xs) == batch_size or idx == len(dataset) - 1:
            xb = torch.stack(xs).to(device)
            yb = torch.tensor(ys, dtype=torch.long, device=device)
            with torch.no_grad():
                logits = wrapper(xb)
                preds = logits.argmax(1)
                margins = margin(logits, yb)
                probs = F.softmax(logits, dim=1).gather(1, yb.view(-1, 1)).squeeze(1)
            for j, dataset_idx in enumerate(idxs):
                label = int(ys[j])
                pred = int(preds[j].item())
                if pred == label and counts[label] < per_class:
                    counts[label] += 1
                    rows.append(
                        {
                            "model": model_name,
                            "dataset_idx": int(dataset_idx),
                            "label": label,
                            "clean_pred": pred,
                            "clean_margin": float(margins[j].item()),
                            "clean_true_prob": float(probs[j].item()),
                            "class_ord": counts[label] - 1,
                        }
                    )
            xs, ys, idxs = [], [], []
            if all(v >= per_class for v in counts.values()):
                break

    missing = {c: per_class - counts[c] for c in range(10) if counts[c] < per_class}
    if missing:
        raise RuntimeError(f"{model_name} lacks {per_class} clean-correct images per class: {missing}")

    meta = {
        "model": model_name,
        "loaded_model_state_sha256": model_weight_hash(base),
        "available_layers": wrapper.labels,
        "layer_candidates": LAYER_CANDIDATES.get(model_name, wrapper.labels),
        "clean_correct_per_class": per_class,
        "selection_rule": "first clean-correct test images per class under sorted CIFAR-10 index order",
    }
    return pd.DataFrame(rows).sort_values(["label", "class_ord", "dataset_idx"]).reset_index(drop=True), meta


def make_splits(base: pd.DataFrame, split_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(split_seed)
    chunks = []
    for label, g in base.groupby("label", sort=True):
        g = g.sort_values("class_ord").copy()
        order = np.arange(len(g))
        rng.shuffle(order)
        split = np.empty(len(g), dtype=object)
        split[order[:40]] = "basis_fit"
        split[order[40:60]] = "layer_validation"
        split[order[60:]] = "final_test"
        gg = g.copy()
        gg["split_seed"] = split_seed
        gg["split"] = split
        gg["split_class_ord"] = [
            int(np.where(order == i)[0][0]) for i in range(len(g))
        ]
        chunks.append(gg)
    out = pd.concat(chunks, ignore_index=True)
    out["image_ord"] = out.groupby(["model", "split_seed"]).cumcount()
    return out.sort_values(["model", "split_seed", "label", "class_ord"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/exact_protocol/cifar_splits")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--split-seeds", default=",".join(str(x) for x in DEFAULT_SPLIT_SEEDS))
    p.add_argument("--per-class", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    models = parse_csv(args.models)
    split_seeds = [int(x) for x in parse_csv(args.split_seeds)]

    all_split_rows = []
    model_meta = []
    for model_name in models:
        base, meta = clean_correct_table(model_name, dataset, device, args.per_class, args.batch_size)
        base.to_csv(out_dir / f"{model_name}_clean_correct_pool.csv", index=False)
        model_meta.append(meta)
        for split_seed in split_seeds:
            all_split_rows.append(make_splits(base, split_seed))
        print(f"[OK] {model_name}: {len(base)} clean-correct images, split seeds={split_seeds}", flush=True)

    splits = pd.concat(all_split_rows, ignore_index=True)
    splits.to_csv(out_dir / "cifar10_exact_splits.csv", index=False)
    pd.DataFrame(model_meta).to_csv(out_dir / "model_registry.csv", index=False)
    pd.DataFrame(ATTACK_REGISTRY).to_csv(out_dir / "attack_registry.csv", index=False)
    layer_rows = []
    for model_name in models:
        for layer in LAYER_CANDIDATES.get(model_name, []):
            layer_rows.append({"model": model_name, "layer": layer, "candidate": 1})
    pd.DataFrame(layer_rows).to_csv(out_dir / "layer_registry.csv", index=False)

    metadata = {
        "dataset": "CIFAR-10 test",
        "dataset_root": args.dataset_root,
        "per_class": args.per_class,
        "n_per_model": args.per_class * 10,
        "split_rule": "within each class, shuffle selected clean-correct pool with split_seed; assign 40 basis_fit, 20 layer_validation, 40 final_test",
        "split_seeds": split_seeds,
        "candidate_seeds": [0, 1, 2, 3, 4],
        "bootstrap_seed": 12345,
        "bootstrap_resamples": 10000,
        "primary_k": 20,
        "k_sensitivity": [5, 10, 20, 40],
        "trajectory_schema": {
            "local_step": "h_l(x_{t+1}) - h_l(x_t)",
            "cumulative_displacement": "h_l(x_t) - h_l(x_0)",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(out_dir / "cifar10_exact_splits.csv")


if __name__ == "__main__":
    main()
