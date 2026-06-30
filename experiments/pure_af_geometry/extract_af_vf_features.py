#!/usr/bin/env python3
"""Extract DifAttack++ AF/VF features for pure, clean, and adversarial images.

Outputs are isolated under analysis_outputs/pure_af_geometry by default. The
extractor stores full-layer norms plus pooled vectors for PCA/clustering and
shift-alignment analysis. It keeps pure init modes and regularization variants
as explicit metadata columns.
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
from torchvision import datasets, transforms, utils as tv_utils

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LAYER_NAMES = ["sem0", "sem1", "sem2", "sem3", "sem4"]
VIS_NAMES = ["vf0", "vf1", "vf2", "vf3", "vf4"]
SELECTED = ["sem0", "sem1", "sem4", "vf0", "vf1", "vf4"]


def load_autoencoder(checkpoint_path: Path, device: torch.device, key: str):
    module_path = REPO_ROOT / "external_repos" / "DifAttack" / "autoencoder.py"
    spec = importlib.util.spec_from_file_location("difattack_autoencoder", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if key not in ckpt:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain key {key!r}; keys={sorted(ckpt.keys())}")
    model = module.Autoencoder().to(device).eval()
    model.load_state_dict(ckpt[key])
    return model


def read_ints(path: Path) -> set[int]:
    vals: set[int] = set()
    if not path:
        return vals
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals.add(int(line.split()[0].split(",")[0]))
    return vals


def parse_int_set(text: str) -> set[int]:
    vals: set[int] = set()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = [int(x) for x in chunk.split("-", 1)]
            vals.update(range(a, b + 1))
        else:
            vals.add(int(chunk))
    return vals


def tensor_from_image(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def tensor_to_unit(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    if x.max() > 2.0:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


def encode_parts(model, image: torch.Tensor):
    with torch.no_grad():
        parts = model(image.mul(2.0).sub(1.0))
        recon = torch.clamp(parts[0] * 0.5 + 0.5, 0.0, 1.0)
    return recon, list(parts[1:6]), list(parts[6:11])


def pooled(x: torch.Tensor, pool_size: int) -> np.ndarray:
    y = F.adaptive_avg_pool2d(x.detach().float(), (pool_size, pool_size))
    return y.flatten(start_dim=1).cpu().numpy()[0].astype(np.float32)


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    flat = x.detach().float().flatten(start_dim=1)
    return {
        "norm_l2": float(torch.norm(flat, dim=1).item()),
        "norm_l1_mean": float(flat.abs().mean(dim=1).item()),
        "mean": float(flat.mean(dim=1).item()),
        "std": float(flat.std(dim=1, unbiased=False).item()),
    }


def class_clean_indices(dataset, labels: set[int], per_class: int, exclude: set[int]) -> list[int]:
    selected: list[int] = []
    counts = {label: 0 for label in labels}
    for idx, (_path, label) in enumerate(dataset.samples):
        label = int(label)
        if label not in labels or idx in exclude or counts[label] >= per_class:
            continue
        selected.append(idx)
        counts[label] += 1
        if all(count >= per_class for count in counts.values()):
            break
    missing = {label: per_class - count for label, count in counts.items() if count < per_class}
    if missing:
        raise RuntimeError(f"Not enough clean ImageNet examples for classes: {missing}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-manifest", required=True)
    parser.add_argument("--difattackpp-checkpoint", required=True)
    parser.add_argument("--checkpoint-key", default="state_dict_adv")
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--clean-per-class", type=int, default=20)
    parser.add_argument("--exclude-indices-file", default="")
    parser.add_argument("--adv-successes-csv", default="", help="Optional successes.csv with tensor_path containing clean_uint8/adv_uint8")
    parser.add_argument("--adv-labels", default="manifest", help="manifest, all, or explicit labels/ranges like 0,5-9")
    parser.add_argument("--adv-per-class", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/features")
    parser.add_argument("--save-reconstructions", action="store_true")
    parser.add_argument("--max-pure", type=int, default=0, help="debug cap; 0 means all")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = out_dir / "reconstructions"
    if args.save_reconstructions:
        recon_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.pure_manifest)
    if args.max_pure:
        manifest = manifest.head(args.max_pure)
    labels = {int(x) for x in manifest["target_class"].unique()}

    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    exclude = read_ints(Path(args.exclude_indices_file)) if args.exclude_indices_file else set()
    clean_indices = class_clean_indices(dataset, labels, args.clean_per_class, exclude)

    model = load_autoencoder(Path(args.difattackpp_checkpoint), device, args.checkpoint_key)

    rows: list[dict[str, object]] = []
    vectors: dict[str, list[np.ndarray]] = {name: [] for name in SELECTED}
    sample_ids: list[str] = []

    def record(sample_id: str, source: str, label: int, image: torch.Tensor, extra: dict[str, object]) -> None:
        recon, vis, sem = encode_parts(model, image)
        if args.save_reconstructions:
            tv_utils.save_image(recon.cpu(), recon_dir / f"{sample_id}.png")
        layer_map = {**dict(zip(VIS_NAMES, vis)), **dict(zip(LAYER_NAMES, sem))}
        for layer_name in SELECTED:
            vectors[layer_name].append(pooled(layer_map[layer_name], args.pool_size))
        sample_ids.append(sample_id)
        base = {"sample_id": sample_id, "source": source, "label": int(label), **extra}
        for layer_name in SELECTED:
            stats = tensor_stats(layer_map[layer_name])
            for stat_name, value in stats.items():
                base[f"{layer_name}_{stat_name}"] = value
        rows.append(base)

    for row_idx, row in manifest.iterrows():
        image_path = Path(row["final_image"])
        image = tensor_from_image(image_path, args.image_size, device)
        sample_id = f"pure_{row_idx:05d}_c{int(row['target_class']):04d}"
        record(
            sample_id,
            "pure",
            int(row["target_class"]),
            image,
            {
                "init_mode": row.get("init_mode", ""),
                "regularization": row.get("regularization", ""),
                "pure_run_name": row.get("run_name", ""),
                "dataset_idx": "",
                "image_path": str(image_path),
                "pair_id": "",
                "adv_final_pred": "",
            },
        )
        print(f"[pure] {row_idx + 1}/{len(manifest)} {sample_id}", flush=True)

    for count, dataset_idx in enumerate(clean_indices, start=1):
        image, label = dataset[dataset_idx]
        sample_id = f"clean_idx{dataset_idx:05d}_c{int(label):04d}"
        record(
            sample_id,
            "clean",
            int(label),
            image.unsqueeze(0).to(device),
            {
                "init_mode": "",
                "regularization": "",
                "pure_run_name": "",
                "dataset_idx": int(dataset_idx),
                "image_path": dataset.samples[dataset_idx][0],
                "pair_id": "",
                "adv_final_pred": "",
            },
        )
        if count % 20 == 0:
            print(f"[clean] {count}/{len(clean_indices)}", flush=True)

    adv_rows_used = 0
    if args.adv_successes_csv:
        adv = pd.read_csv(args.adv_successes_csv)
        adv = adv[adv["success"].astype(int) == 1].copy()
        if args.adv_labels == "manifest":
            wanted_labels = labels
        elif args.adv_labels == "all":
            wanted_labels = set(int(x) for x in adv["label"].unique())
        else:
            wanted_labels = parse_int_set(args.adv_labels)
        adv = adv[adv["label"].astype(int).isin(wanted_labels)]
        adv = adv.sort_values(["label", "query_cnt", "dataset_idx"])
        counts: dict[int, int] = {label: 0 for label in wanted_labels}
        for _idx, row in adv.iterrows():
            label = int(row["label"])
            if counts.get(label, 0) >= args.adv_per_class:
                continue
            tensor_path = Path(row["tensor_path"])
            if not tensor_path.exists():
                continue
            data = torch.load(tensor_path, map_location="cpu")
            pair_id = f"advpair_idx{int(row['dataset_idx']):05d}_c{label:04d}"
            clean = tensor_to_unit(data["clean_uint8"]).unsqueeze(0).to(device)
            adv_img = tensor_to_unit(data["adv_uint8"]).unsqueeze(0).to(device)
            common = {
                "init_mode": "",
                "regularization": "",
                "pure_run_name": "",
                "dataset_idx": int(row["dataset_idx"]),
                "image_path": row.get("image_path", ""),
                "pair_id": pair_id,
                "adv_final_pred": int(row.get("final_pred", -1)),
            }
            record(f"{pair_id}_clean", "adv_clean", label, clean, common)
            record(f"{pair_id}_adv", "adv", label, adv_img, common)
            counts[label] = counts.get(label, 0) + 1
            adv_rows_used += 1
            if sum(counts.values()) % 20 == 0:
                print(f"[adv] pairs={sum(counts.values())}", flush=True)

    stats_path = out_dir / "feature_stats.csv"
    with stats_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    npz_path = out_dir / "pooled_features.npz"
    np.savez_compressed(
        npz_path,
        sample_id=np.array(sample_ids),
        **{name: np.stack(vals, axis=0) for name, vals in vectors.items()},
    )

    metadata = {
        "experiment": "pure_af_geometry",
        "pure_manifest": args.pure_manifest,
        "adv_successes_csv": args.adv_successes_csv,
        "difattackpp_checkpoint": args.difattackpp_checkpoint,
        "checkpoint_key": args.checkpoint_key,
        "selected_layers": SELECTED,
        "pool_size": args.pool_size,
        "stats_csv": str(stats_path),
        "pooled_features_npz": str(npz_path),
        "samples": len(sample_ids),
        "pure_samples": int(len(manifest)),
        "clean_samples": int(len(clean_indices)),
        "adv_pairs": int(adv_rows_used),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[DONE] stats={stats_path} pooled={npz_path}")


if __name__ == "__main__":
    main()
