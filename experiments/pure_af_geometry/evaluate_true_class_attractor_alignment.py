#!/usr/bin/env python3
"""Test whether class-attractor directions strengthen true-class evidence near real images."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ResNet18WithFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.base = models.resnet18(pretrained=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_feature(x, "logits")[0]

    def forward_with_feature(self, x: torch.Tensor, layer: str) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.normalize(x)
        z = self.base.conv1(z)
        z = self.base.bn1(z)
        z = self.base.relu(z)
        z = self.base.maxpool(z)
        z = self.base.layer1(z)
        z = self.base.layer2(z)
        z = self.base.layer3(z)
        z = self.base.layer4(z)
        layer4 = F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
        avgpool = self.base.avgpool(z).flatten(1)
        logits = self.base.fc(avgpool)
        if layer == "layer4":
            return logits, layer4
        if layer == "avgpool":
            return logits, avgpool
        if layer == "logits":
            return logits, logits
        raise ValueError(f"Unsupported layer: {layer}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_classes(value: str) -> list[int]:
    if "-" in value:
        lo, hi = value.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in value.split(",") if x.strip()]


def load_attractors(path: str, device: torch.device):
    data = np.load(path)
    classes = data["classes"].astype(int).tolist()
    class_to_pos = {cls: i for i, cls in enumerate(classes)}
    out = {"classes": classes, "class_to_pos": class_to_pos}
    for layer in ["avgpool", "layer4", "logits"]:
        values = torch.from_numpy(data[f"clf_{layer}"]).float().to(device)
        out[layer] = F.normalize(values, dim=1)
    return out


def project_linf(x: torch.Tensor, clean: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x, clean + eps), clean - eps).clamp(0.0, 1.0)


def select_indices(dataset, classes: list[int], per_class: int, model, device, seed: int, batch_size: int) -> list[int]:
    rng = random.Random(seed)
    selected = {cls: [] for cls in classes}
    candidates = []
    for cls in classes:
        values = [idx for idx, (_path, label) in enumerate(dataset.samples) if int(label) == cls]
        rng.shuffle(values)
        candidates.extend(values)
    loader = DataLoader(Subset(dataset, candidates), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    seen = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            ok = (pred == y).detach().cpu().tolist()
            labels = y.detach().cpu().tolist()
            for offset, good in enumerate(ok):
                cls = int(labels[offset])
                if good and len(selected[cls]) < per_class:
                    selected[cls].append(candidates[seen + offset])
            seen += len(ok)
            if all(len(selected[cls]) >= per_class for cls in classes):
                break
    missing = {cls: per_class - len(v) for cls, v in selected.items() if len(v) < per_class}
    if missing:
        raise RuntimeError(f"Missing clean-correct images: {missing}")
    out = []
    for cls in classes:
        out.extend(selected[cls])
    return out


def variant_parts(variant: str) -> tuple[str, str]:
    if variant == "ce_true":
        return "ce", "logits"
    if variant.startswith("attractor_only_"):
        return "attractor", variant.replace("attractor_only_", "")
    if variant.startswith("ce_plus_attractor_"):
        return "both", variant.replace("ce_plus_attractor_", "")
    raise ValueError(f"Unknown variant: {variant}")


def optimize(
    clean: torch.Tensor,
    label: torch.Tensor,
    model: ResNet18WithFeatures,
    variant: str,
    attractors,
    eps: float,
    step_size: float,
    steps: int,
    beta: float,
) -> torch.Tensor:
    mode, layer = variant_parts(variant)
    adv = clean.detach().clone()
    with torch.no_grad():
        _logits_clean, h_clean = model.forward_with_feature(clean, layer)
    class_to_pos = attractors["class_to_pos"]
    mu = attractors[layer][torch.tensor([class_to_pos[int(label.item())]], device=clean.device)]
    for _ in range(steps):
        adv.requires_grad_(True)
        logits, h_adv = model.forward_with_feature(adv, layer)
        delta_h = F.normalize(h_adv - h_clean, dim=1)
        align = (delta_h * mu).sum(1).mean()
        ce = F.cross_entropy(logits, label)
        if mode == "ce":
            loss = ce
        elif mode == "attractor":
            loss = -align
        else:
            loss = ce - beta * align
        grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
        adv = project_linf(adv.detach() - step_size * grad.sign(), clean, eps)
    return adv.detach()


def metrics(model: ResNet18WithFeatures, clean: torch.Tensor, adv: torch.Tensor, label: torch.Tensor, attractors) -> dict[str, float]:
    out = {}
    with torch.no_grad():
        clean_logits = model(clean)
        adv_logits = model(adv)
        y = int(label.item())
        out["clean_pred"] = int(clean_logits.argmax(1).item())
        out["adv_pred"] = int(adv_logits.argmax(1).item())
        out["clean_true_logit"] = float(clean_logits[0, y].item())
        out["adv_true_logit"] = float(adv_logits[0, y].item())
        out["delta_true_logit"] = out["adv_true_logit"] - out["clean_true_logit"]
        out["clean_true_prob"] = float(clean_logits.softmax(1)[0, y].item())
        out["adv_true_prob"] = float(adv_logits.softmax(1)[0, y].item())
        for layer in ["avgpool", "layer4", "logits"]:
            _lc, h_clean = model.forward_with_feature(clean, layer)
            _la, h_adv = model.forward_with_feature(adv, layer)
            delta = h_adv - h_clean
            pos = attractors["class_to_pos"][y]
            mu = attractors[layer][pos : pos + 1]
            out[f"{layer}_cos_mu_y"] = float((F.normalize(delta, dim=1) * mu).sum(1).item())
            out[f"{layer}_delta_l2"] = float(delta.norm(p=2, dim=1).item())
    d = adv - clean
    out["linf"] = float(d.abs().max().item())
    out["l2"] = float(d.flatten(1).norm(p=2, dim=1).item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--attractor-npz", default="analysis_outputs/pure_af_geometry/class_attractor_validation_resnet18_c10_s5/class_mean_directions.npz")
    parser.add_argument("--output-csv", default="analysis_outputs/pure_af_geometry/true_class_attractor_alignment_resnet18_100.csv")
    parser.add_argument("--classes", default="0-9")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variants",
        default="ce_true,attractor_only_avgpool,attractor_only_logits,attractor_only_layer4,ce_plus_attractor_avgpool,ce_plus_attractor_logits,ce_plus_attractor_layer4",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    model = ResNet18WithFeatures().to(device).eval()
    attractors = load_attractors(args.attractor_npz, device)
    classes = parse_classes(args.classes)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    indices = select_indices(dataset, classes, args.per_class, model, device, args.seed, args.batch_size)[: args.max_images]

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_idx", "source_class", "variant", "clean_pred", "adv_pred",
        "clean_true_logit", "adv_true_logit", "delta_true_logit", "clean_true_prob", "adv_true_prob",
        "avgpool_cos_mu_y", "layer4_cos_mu_y", "logits_cos_mu_y",
        "avgpool_delta_l2", "layer4_delta_l2", "logits_delta_l2", "linf", "l2", "elapsed_sec",
    ]
    started = time.time()
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for image_i, idx in enumerate(indices, start=1):
            clean_cpu, label_int = dataset[idx]
            clean = clean_cpu.unsqueeze(0).to(device)
            label = torch.tensor([int(label_int)], device=device)
            for variant in variants:
                t0 = time.time()
                adv = optimize(clean, label, model, variant, attractors, args.eps, args.step_size, args.steps, args.beta)
                row = {
                    "dataset_idx": idx,
                    "source_class": int(label.item()),
                    "variant": variant,
                    **metrics(model, clean, adv, label, attractors),
                    "elapsed_sec": time.time() - t0,
                }
                writer.writerow(row)
                f.flush()
            if image_i % 10 == 0:
                print(f"[progress] images={image_i}/{len(indices)} elapsed={time.time() - started:.1f}s", flush=True)

    metadata = {
        "args": vars(args),
        "indices": indices,
        "variants": variants,
        "classes": classes,
        "output_csv": str(output),
        "elapsed_sec": time.time() - started,
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
