#!/usr/bin/env python3
"""Audit whether AF/VF geometry follows generator family more than confidence."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.load_models import load_imagenet_model

LAYERS = ["sem0", "sem1", "sem4", "vf0", "vf1", "vf4"]
LAYER_NAMES = ["sem0", "sem1", "sem2", "sem3", "sem4"]
VIS_NAMES = ["vf0", "vf1", "vf2", "vf3", "vf4"]


def load_autoencoder(checkpoint_path: Path, device: torch.device, key: str):
    module_path = REPO_ROOT / "external_repos" / "DifAttack" / "autoencoder.py"
    spec = importlib.util.spec_from_file_location("difattack_autoencoder", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = module.Autoencoder().to(device).eval()
    model.load_state_dict(ckpt[key])
    return model


def load_image(path: str, transform, device: torch.device) -> torch.Tensor:
    return transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def load_tensor(path: str, device: torch.device) -> torch.Tensor:
    return torch.load(path, map_location=device).float().to(device).clamp(0.0, 1.0)


def encode(model, image: torch.Tensor):
    with torch.no_grad():
        parts = model(image.mul(2.0).sub(1.0))
    return {**dict(zip(VIS_NAMES, parts[1:6])), **dict(zip(LAYER_NAMES, parts[6:11]))}


def distance(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.detach().float().flatten(1)
    bf = b.detach().float().flatten(1)
    d = af - bf
    return {
        "l2": float(torch.norm(d, dim=1).item()),
        "mean_abs": float(d.abs().mean().item()),
        "cosine": float(F.cosine_similarity(af, bf, dim=1).item()),
    }


def logits_info(model, image: torch.Tensor, label: int) -> dict[str, float | int]:
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
    masked = logits.clone()
    masked[:, label] = -torch.inf
    next_logit, next_idx = masked.max(dim=1)
    return {
        "pred": int(logits.argmax(1).item()),
        "prob": float(probs[0, label].item()),
        "margin": float((logits[0, label] - next_logit[0]).item()),
        "next_best_class": int(next_idx.item()),
    }


def find_strong_clean(dataset, classifier, label: int, device: torch.device) -> tuple[int, str, torch.Tensor, dict[str, float | int]]:
    best = None
    for idx, (path, y) in enumerate(dataset.samples):
        if int(y) != label:
            continue
        image, _ = dataset[idx]
        image = image.unsqueeze(0).to(device)
        info = logits_info(classifier, image, label)
        if info["pred"] != label:
            continue
        if best is None or info["margin"] > best[3]["margin"]:
            best = (idx, path, image, info)
    if best is None:
        raise RuntimeError(f"No clean-correct image found for class {label}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-manifest", required=True)
    parser.add_argument("--dirty-manifest", required=True)
    parser.add_argument("--difattackpp-checkpoint", default="external_repos/DifAttack_assets/difattack_plus/ResNet18.pth.tar")
    parser.add_argument("--checkpoint-key", default="state_dict_adv")
    parser.add_argument("--target-model", default="resnet18")
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/family_confidence_audit")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    classifier = load_imagenet_model(args.target_model).to(device).eval()
    ae = load_autoencoder(Path(args.difattackpp_checkpoint), device, args.checkpoint_key)

    pure = pd.read_csv(args.pure_manifest)
    dirty = pd.read_csv(args.dirty_manifest)
    pure = pure[(pure["final_prob"].astype(float) >= 0.9999) & (pure["final_pred"].astype(int) == pure["target_class"].astype(int))]
    dirty = dirty[(dirty["success"].astype(int) == 1) & (dirty["final_pred"].astype(int) == dirty["target_class"].astype(int))]

    labels = sorted(set(pure["target_class"].astype(int)) & set(dirty["target_class"].astype(int)))
    rows = []
    sample_rows = []
    encoded = {}

    for label in labels:
        for init in ["real", "random"]:
            p_rows = pure[(pure["target_class"].astype(int) == label) & (pure["init_mode"] == init)]
            d_rows = dirty[(dirty["target_class"].astype(int) == label) & (dirty["init_mode"] == init)]
            if p_rows.empty or d_rows.empty:
                continue
            p_row = p_rows.iloc[0]
            d_row = d_rows.iloc[0]
            p_img = load_image(p_row["final_image"], transform, device)
            d_img = load_tensor(d_row["final_tensor"], device) if "final_tensor" in d_row and isinstance(d_row["final_tensor"], str) and d_row["final_tensor"] else load_image(d_row["final_image"], transform, device)
            for kind, row, image in [("pure", p_row, p_img), ("dirty", d_row, d_img)]:
                key = (label, kind, init)
                encoded[key] = encode(ae, image)
                info = logits_info(classifier, image, label)
                sample_rows.append({
                    "label": label,
                    "kind": kind,
                    "init_mode": init,
                    "pred": info["pred"],
                    "prob": info["prob"],
                    "margin": info["margin"],
                    "next_best_class": info["next_best_class"],
                    "path": row.get("final_tensor", "") if kind == "dirty" else row.get("final_image", ""),
                })
        idx, path, strong_img, strong_info = find_strong_clean(dataset, classifier, label, device)
        encoded[(label, "strong_clean", "natural")] = encode(ae, strong_img)
        sample_rows.append({
            "label": label,
            "kind": "strong_clean",
            "init_mode": "natural",
            "pred": strong_info["pred"],
            "prob": strong_info["prob"],
            "margin": strong_info["margin"],
            "next_best_class": strong_info["next_best_class"],
            "dataset_idx": idx,
            "path": path,
        })

        comparisons = [
            ("same_init_real_pure_dirty", (label, "pure", "real"), (label, "dirty", "real")),
            ("same_init_random_pure_dirty", (label, "pure", "random"), (label, "dirty", "random")),
            ("pure_real_to_strong_clean", (label, "pure", "real"), (label, "strong_clean", "natural")),
            ("dirty_real_to_strong_clean", (label, "dirty", "real"), (label, "strong_clean", "natural")),
            ("pure_random_to_strong_clean", (label, "pure", "random"), (label, "strong_clean", "natural")),
            ("dirty_random_to_strong_clean", (label, "dirty", "random"), (label, "strong_clean", "natural")),
            ("cross_init_pure_real_random", (label, "pure", "real"), (label, "pure", "random")),
            ("cross_init_dirty_real_random", (label, "dirty", "real"), (label, "dirty", "random")),
        ]
        for comp, a_key, b_key in comparisons:
            if a_key not in encoded or b_key not in encoded:
                continue
            for layer in LAYERS:
                vals = distance(encoded[a_key][layer], encoded[b_key][layer])
                rows.append({
                    "label": label,
                    "comparison": comp,
                    "layer": layer,
                    **vals,
                })

    dist_df = pd.DataFrame(rows)
    sample_df = pd.DataFrame(sample_rows)
    dist_path = out_dir / "family_confidence_distances.csv"
    sample_path = out_dir / "sample_summary.csv"
    dist_df.to_csv(dist_path, index=False)
    sample_df.to_csv(sample_path, index=False)

    agg = dist_df.groupby(["comparison", "layer"]).agg(
        n=("l2", "size"), mean_l2=("l2", "mean"), median_l2=("l2", "median"), mean_cos=("cosine", "mean")
    ).reset_index()
    agg_path = out_dir / "family_confidence_aggregate.csv"
    agg.to_csv(agg_path, index=False)

    # Ratios that directly test family/path dominance.
    ratio_rows = []
    for label in labels:
        for layer in LAYERS:
            sub = dist_df[(dist_df["label"] == label) & (dist_df["layer"] == layer)].set_index("comparison")
            def get(name):
                return float(sub.loc[name, "l2"]) if name in sub.index else np.nan
            same_real = get("same_init_real_pure_dirty")
            same_random = get("same_init_random_pure_dirty")
            pure_cross = get("cross_init_pure_real_random")
            dirty_cross = get("cross_init_dirty_real_random")
            real_strong = get("pure_real_to_strong_clean")
            dirty_real_strong = get("dirty_real_to_strong_clean")
            ratio_rows.append({
                "label": label,
                "layer": layer,
                "same_real": same_real,
                "same_random": same_random,
                "cross_pure": pure_cross,
                "cross_dirty": dirty_cross,
                "pure_real_to_strong": real_strong,
                "dirty_real_to_strong": dirty_real_strong,
                "cross_pure_over_same_real": pure_cross / same_real if same_real else np.nan,
                "cross_dirty_over_same_random": dirty_cross / same_random if same_random else np.nan,
                "pure_real_strong_over_same_real": real_strong / same_real if same_real else np.nan,
                "dirty_real_strong_over_same_real": dirty_real_strong / same_real if same_real else np.nan,
            })
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_path = out_dir / "family_dominance_ratios.csv"
    ratio_df.to_csv(ratio_path, index=False)

    ratio_agg = ratio_df.groupby("layer").agg(
        n=("label", "size"),
        mean_cross_pure_over_same_real=("cross_pure_over_same_real", "mean"),
        mean_cross_dirty_over_same_random=("cross_dirty_over_same_random", "mean"),
        mean_pure_real_strong_over_same_real=("pure_real_strong_over_same_real", "mean"),
        mean_dirty_real_strong_over_same_real=("dirty_real_strong_over_same_real", "mean"),
    ).reset_index()
    ratio_agg_path = out_dir / "family_dominance_ratio_aggregate.csv"
    ratio_agg.to_csv(ratio_agg_path, index=False)

    metadata = {
        "pure_manifest": args.pure_manifest,
        "dirty_manifest": args.dirty_manifest,
        "outputs": {
            "sample_summary": str(sample_path),
            "family_confidence_distances": str(dist_path),
            "family_confidence_aggregate": str(agg_path),
            "family_dominance_ratios": str(ratio_path),
            "family_dominance_ratio_aggregate": str(ratio_agg_path),
        },
        "labels": labels,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
