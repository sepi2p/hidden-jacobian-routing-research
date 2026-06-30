#!/usr/bin/env python3
"""Classify image/generation family from DifAttack++ AF/VF features.

This is a representation probe for the pure-AF geometry track. It builds a
balanced per-class dataset with one sample from each available family and
runs leave-one-ImageNet-class-out evaluation, so family prediction must
generalize across target classes rather than memorize class identity.
"""

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
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.load_models import load_imagenet_model

LAYER_NAMES = ["sem0", "sem1", "sem2", "sem3", "sem4"]
VIS_NAMES = ["vf0", "vf1", "vf2", "vf3", "vf4"]
DEFAULT_LAYERS = ["sem0", "sem1", "sem4", "vf0", "vf1", "vf4"]
FAMILIES = ["clean", "pure_real", "pure_random", "dirty_real", "dirty_random"]


def parse_layers(text: str) -> list[str]:
    layers = [x.strip() for x in text.split(",") if x.strip()]
    known = set(LAYER_NAMES + VIS_NAMES)
    bad = [x for x in layers if x not in known]
    if bad:
        raise ValueError(f"Unknown layers {bad}; known={sorted(known)}")
    return layers


def load_autoencoder(checkpoint_path: Path, device: torch.device, key: str):
    module_path = REPO_ROOT / "external_repos" / "DifAttack" / "autoencoder.py"
    spec = importlib.util.spec_from_file_location("difattack_autoencoder", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if key not in ckpt:
        raise KeyError(f"Checkpoint {checkpoint_path} lacks key {key!r}; keys={sorted(ckpt.keys())}")
    model = module.Autoencoder().to(device).eval()
    model.load_state_dict(ckpt[key])
    return model


def image_from_path(path: str | Path, transform, device: torch.device) -> torch.Tensor:
    return transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def image_from_tensor(path: str | Path, device: torch.device) -> torch.Tensor:
    return torch.load(path, map_location=device).float().clamp(0.0, 1.0).to(device)


def encode(model, image: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        parts = model(image.mul(2.0).sub(1.0))
    return {**dict(zip(VIS_NAMES, parts[1:6])), **dict(zip(LAYER_NAMES, parts[6:11]))}


def pooled(x: torch.Tensor, pool_size: int) -> np.ndarray:
    y = F.adaptive_avg_pool2d(x.detach().float(), (pool_size, pool_size))
    return y.flatten(start_dim=1).cpu().numpy()[0].astype(np.float32)


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


def choose_row(df: pd.DataFrame, label: int, init_mode: str) -> pd.Series | None:
    sub = df[(df["target_class"].astype(int) == label) & (df["init_mode"] == init_mode)].copy()
    if sub.empty:
        return None
    if "final_margin" in sub.columns:
        sub["abs_margin"] = sub["final_margin"].astype(float).abs()
        sub = sub.sort_values(["abs_margin", "final_prob"], ascending=[True, False])
    return sub.iloc[0]


def build_samples(args, layers: list[str]) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    classifier = load_imagenet_model(args.target_model).to(device).eval()
    ae = load_autoencoder(Path(args.difattackpp_checkpoint), device, args.checkpoint_key)

    pure = pd.read_csv(args.pure_manifest)
    dirty = pd.read_csv(args.dirty_manifest)
    pure = pure[(pure["final_pred"].astype(int) == pure["target_class"].astype(int)) & (pure["final_prob"].astype(float) >= args.pure_prob_threshold)]
    dirty = dirty[(dirty["success"].astype(int) == 1) & (dirty["final_pred"].astype(int) == dirty["target_class"].astype(int))]
    labels = sorted(set(pure["target_class"].astype(int)) & set(dirty["target_class"].astype(int)))
    if args.classes:
        wanted = set(int(x) for x in args.classes.split(",") if x.strip())
        labels = [x for x in labels if x in wanted]

    rows: list[dict[str, object]] = []
    vectors: dict[str, list[np.ndarray]] = {layer: [] for layer in layers}

    def add_sample(sample_id: str, family: str, label: int, image: torch.Tensor, path: str, extra: dict[str, object]) -> None:
        parts = encode(ae, image)
        info = logits_info(classifier, image, label)
        rows.append({
            "sample_id": sample_id,
            "family": family,
            "label": int(label),
            "path": path,
            "pred": info["pred"],
            "prob": info["prob"],
            "margin": info["margin"],
            "next_best_class": info["next_best_class"],
            **extra,
        })
        for layer in layers:
            vectors[layer].append(pooled(parts[layer], args.pool_size))

    for label in labels:
        idx, path, image, _info = find_strong_clean(dataset, classifier, label, device)
        add_sample(f"clean_c{label:04d}", "clean", label, image, path, {"init_mode": "natural", "dataset_idx": idx})

        for init in ["real", "random"]:
            row = choose_row(pure, label, init)
            if row is not None:
                image = image_from_path(row["final_image"], transform, device)
                add_sample(
                    f"pure_{init}_c{label:04d}",
                    f"pure_{init}",
                    label,
                    image,
                    str(row["final_image"]),
                    {"init_mode": init, "dataset_idx": "", "source_run": row.get("run_name", "")},
                )
            row = choose_row(dirty, label, init)
            if row is not None:
                tensor_path = row.get("final_tensor", "")
                if isinstance(tensor_path, str) and tensor_path:
                    image = image_from_tensor(tensor_path, device)
                    path = tensor_path
                else:
                    image = image_from_path(row["final_image"], transform, device)
                    path = row["final_image"]
                add_sample(
                    f"dirty_{init}_c{label:04d}",
                    f"dirty_{init}",
                    label,
                    image,
                    str(path),
                    {"init_mode": init, "dataset_idx": "", "source_run": row.get("run_name", "")},
                )
        print(f"[features] class={label} samples={len([r for r in rows if r['label'] == label])}", flush=True)

    return rows, {layer: np.stack(vals, axis=0) for layer, vals in vectors.items()}


def evaluate_leave_one_class_out(rows: list[dict[str, object]], vectors: dict[str, np.ndarray], out_dir: Path) -> list[dict[str, object]]:
    y = np.array([r["family"] for r in rows])
    labels = np.array([int(r["label"]) for r in rows])
    sample_ids = np.array([r["sample_id"] for r in rows])
    families = [f for f in FAMILIES if f in set(y)]
    summary_rows = []
    prediction_rows = []

    for layer, X in vectors.items():
        pred_all = np.empty_like(y, dtype=object)
        fold_rows = []
        for heldout in sorted(set(labels)):
            train = labels != heldout
            test = labels == heldout
            clf = make_pipeline(
                StandardScaler(),
                RidgeClassifier(alpha=1.0, class_weight="balanced"),
            )
            clf.fit(X[train], y[train])
            pred = clf.predict(X[test])
            pred_all[test] = pred
            fold_acc = accuracy_score(y[test], pred)
            fold_rows.append({"layer": layer, "heldout_label": int(heldout), "n_test": int(test.sum()), "accuracy": float(fold_acc)})
            for sid, true_family, pred_family in zip(sample_ids[test], y[test], pred):
                prediction_rows.append({
                    "layer": layer,
                    "heldout_label": int(heldout),
                    "sample_id": sid,
                    "true_family": true_family,
                    "pred_family": pred_family,
                    "correct": int(true_family == pred_family),
                })
        acc = accuracy_score(y, pred_all)
        cm = confusion_matrix(y, pred_all, labels=families)
        pd.DataFrame(cm, index=families, columns=families).to_csv(out_dir / f"confusion_{layer}.csv")
        pd.DataFrame(fold_rows).to_csv(out_dir / f"fold_accuracy_{layer}.csv", index=False)
        report = classification_report(y, pred_all, labels=families, output_dict=True, zero_division=0)
        (out_dir / f"classification_report_{layer}.json").write_text(json.dumps(report, indent=2))
        summary_rows.append({
            "layer": layer,
            "n_samples": int(len(y)),
            "n_classes_heldout": int(len(set(labels))),
            "n_families": int(len(families)),
            "leave_one_class_out_accuracy": float(acc),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
        })

    pd.DataFrame(prediction_rows).to_csv(out_dir / "leave_one_class_predictions.csv", index=False)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-manifest", required=True)
    parser.add_argument("--dirty-manifest", required=True)
    parser.add_argument("--difattackpp-checkpoint", default="external_repos/DifAttack_assets/difattack_plus/ResNet18.pth.tar")
    parser.add_argument("--checkpoint-key", default="state_dict_adv")
    parser.add_argument("--target-model", default="resnet18")
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--pure-prob-threshold", type=float, default=0.9999)
    parser.add_argument("--layers", default=",".join(DEFAULT_LAYERS))
    parser.add_argument("--classes", default="", help="Optional comma-separated class ids")
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/family_classifier_multiclass10_resnet18_adv")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, vectors = build_samples(args, layers)
    sample_path = out_dir / "sample_manifest.csv"
    with sample_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out_dir / "pooled_family_features.npz", sample_id=np.array([r["sample_id"] for r in rows]), **vectors)

    summary_rows = evaluate_leave_one_class_out(rows, vectors, out_dir)
    summary_path = out_dir / "layer_family_classifier_summary.csv"
    pd.DataFrame(summary_rows).sort_values("leave_one_class_out_accuracy", ascending=False).to_csv(summary_path, index=False)

    metadata = {
        "pure_manifest": args.pure_manifest,
        "dirty_manifest": args.dirty_manifest,
        "difattackpp_checkpoint": args.difattackpp_checkpoint,
        "checkpoint_key": args.checkpoint_key,
        "target_model": args.target_model,
        "layers": layers,
        "families": sorted(set(r["family"] for r in rows)),
        "labels": sorted(set(int(r["label"]) for r in rows)),
        "n_samples": len(rows),
        "outputs": {
            "sample_manifest": str(sample_path),
            "summary": str(summary_path),
            "predictions": str(out_dir / "leave_one_class_predictions.csv"),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
