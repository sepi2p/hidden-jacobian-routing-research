#!/usr/bin/env python3
"""Swap DifAttack++ AF/VF parts between real-init and random-init pure images.

For each class, this decodes combinations such as VF_real + AF_random and
VF_random + AF_real, then measures visual deltas and classifier confidence.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, utils as tv_utils

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.load_models import load_imagenet_model

SEM_IDX = {"sem0": 0, "sem1": 1, "sem2": 2, "sem3": 3, "sem4": 4}


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


def load_image(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def encode(model, image: torch.Tensor):
    with torch.no_grad():
        parts = model(image.mul(2.0).sub(1.0))
    return list(parts[1:6]), list(parts[6:11])


def decode_unit(model, vis, sem) -> torch.Tensor:
    with torch.no_grad():
        decoded = model.decode(*vis, *sem)
    return torch.clamp(decoded * 0.5 + 0.5, 0.0, 1.0)


def logits_metrics(model, image: torch.Tensor, target: int) -> dict[str, float | int]:
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
    pred = int(logits.argmax(dim=1).item())
    target_prob = float(probs[0, target].item())
    target_logit = logits[0, target]
    masked = logits.clone()
    masked[:, target] = -torch.inf
    margin = float((target_logit - masked.max(dim=1).values[0]).item())
    return {"pred": pred, "target_prob": target_prob, "target_margin": margin}


def delta_metrics(a: torch.Tensor, b: torch.Tensor, prefix: str) -> dict[str, float]:
    d = (a - b).detach().float()
    flat = d.flatten(start_dim=1)
    return {
        f"{prefix}_linf": float(d.abs().max().item()),
        f"{prefix}_l2": float(torch.norm(flat, dim=1).item()),
        f"{prefix}_mean_abs": float(d.abs().mean().item()),
    }


def make_sem(base_sem, donor_sem, swap_mode: str):
    sem = [x.clone() for x in base_sem]
    if swap_mode == "all":
        return [x.clone() for x in donor_sem]
    sem[SEM_IDX[swap_mode]] = donor_sem[SEM_IDX[swap_mode]].clone()
    return sem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-manifest", required=True)
    parser.add_argument("--difattackpp-checkpoint", required=True)
    parser.add_argument("--checkpoint-key", default="state_dict_adv")
    parser.add_argument("--target-model", default="resnet18")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--swap-modes", default="all,sem0,sem1,sem4")
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/reconstruction_swaps")
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "images"
    if args.save_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.pure_manifest)
    manifest = manifest[manifest["final_prob"].astype(float) >= 0.9999].copy()
    ae = load_autoencoder(Path(args.difattackpp_checkpoint), device, args.checkpoint_key)
    clf = load_imagenet_model(args.target_model).to(device).eval()

    rows = []
    swap_modes = [x.strip() for x in args.swap_modes.split(",") if x.strip()]
    for label, group in manifest.groupby("target_class"):
        label = int(label)
        real_rows = group[group["init_mode"] == "real"]
        random_rows = group[group["init_mode"] == "random"]
        if real_rows.empty or random_rows.empty:
            continue
        real_row = real_rows.iloc[0]
        random_row = random_rows.iloc[0]
        real_img = load_image(Path(real_row["final_image"]), args.image_size, device)
        random_img = load_image(Path(random_row["final_image"]), args.image_size, device)
        real_vis, real_sem = encode(ae, real_img)
        random_vis, random_sem = encode(ae, random_img)

        originals = {"real_original": real_img, "random_original": random_img}
        for name, img in originals.items():
            row = {"label": label, "swap_name": name, "swap_mode": "original", "vf_source": name.split("_")[0], "af_source": name.split("_")[0]}
            row.update(logits_metrics(clf, img, label))
            row.update(delta_metrics(img, real_img, "vs_real"))
            row.update(delta_metrics(img, random_img, "vs_random"))
            rows.append(row)

        specs = [
            ("vf_real_af_random", real_vis, real_sem, random_sem, real_img, random_img),
            ("vf_random_af_real", random_vis, random_sem, real_sem, random_img, real_img),
        ]
        for base_name, vis, base_sem, donor_sem, base_img, donor_img in specs:
            for swap_mode in swap_modes:
                sem = make_sem(base_sem, donor_sem, swap_mode)
                decoded = decode_unit(ae, vis, sem)
                if args.save_images:
                    tv_utils.save_image(decoded.cpu(), image_dir / f"class{label:04d}_{base_name}_{swap_mode}.png")
                row = {
                    "label": label,
                    "swap_name": base_name,
                    "swap_mode": swap_mode,
                    "vf_source": "real" if "vf_real" in base_name else "random",
                    "af_source": "random" if "af_random" in base_name else "real",
                }
                row.update(logits_metrics(clf, decoded, label))
                row.update(delta_metrics(decoded, real_img, "vs_real"))
                row.update(delta_metrics(decoded, random_img, "vs_random"))
                row.update(delta_metrics(decoded, base_img, "vs_vf_source"))
                row.update(delta_metrics(decoded, donor_img, "vs_af_source"))
                rows.append(row)

    csv_path = out_dir / "swap_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "pure_manifest": args.pure_manifest,
        "difattackpp_checkpoint": args.difattackpp_checkpoint,
        "checkpoint_key": args.checkpoint_key,
        "target_model": args.target_model,
        "swap_modes": swap_modes,
        "rows": len(rows),
        "metrics_csv": str(csv_path),
        "images_dir": str(image_dir) if args.save_images else "",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
