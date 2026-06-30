#!/usr/bin/env python3
"""Test whether moving away from true-class attractors creates adversarial examples."""

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


def load_torchvision_model(name: str, device: torch.device) -> nn.Module:
    if name == "densenet121":
        base = models.densenet121(pretrained=True)
    elif name == "vgg16":
        base = models.vgg16_bn(pretrained=True)
    elif name == "resnet18":
        base = models.resnet18(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return nn.Sequential(
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        base,
    ).to(device).eval()


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
    out = {"classes": classes, "class_to_pos": {cls: i for i, cls in enumerate(classes)}}
    for layer in ["avgpool", "layer4", "logits"]:
        values = torch.from_numpy(data[f"clf_{layer}"]).float().to(device)
        out[layer] = F.normalize(values, dim=1)
    return out


def project_linf(x: torch.Tensor, clean: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x, clean + eps), clean - eps).clamp(0.0, 1.0)


def true_margin(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    true_logit = logits.gather(1, label.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, label.view(-1, 1), -1e9)
    return true_logit - masked.max(1).values


def select_indices(dataset, classes: list[int], per_class: int, model, device, seed: int, batch_size: int, allow_partial: bool) -> list[int]:
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
    if missing and not allow_partial:
        raise RuntimeError(f"Missing clean-correct images: {missing}")
    if missing:
        print(f"[WARN] using partial clean-correct class counts; missing={missing}", flush=True)
    out = []
    for cls in classes:
        out.extend(selected[cls])
    return out


def variant_parts(variant: str) -> tuple[str, str]:
    if variant == "ce_untargeted":
        return "ce", "logits"
    if variant == "random_direction":
        return "random", "avgpool"
    if variant.startswith("away_"):
        return "away", variant.replace("away_", "")
    if variant.startswith("ce_plus_away_"):
        return "both", variant.replace("ce_plus_away_", "")
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
    rng: torch.Generator,
) -> torch.Tensor:
    mode, layer = variant_parts(variant)
    adv = clean.detach().clone()
    with torch.no_grad():
        _logits_clean, h_clean = model.forward_with_feature(clean, layer)
    y = int(label.item())
    mu = attractors[layer][attractors["class_to_pos"][y] : attractors["class_to_pos"][y] + 1]
    if mode == "random":
        mu = F.normalize(torch.randn(mu.shape, generator=rng, device=clean.device), dim=1)
    for _ in range(steps):
        adv.requires_grad_(True)
        logits, h_adv = model.forward_with_feature(adv, layer)
        ce = F.cross_entropy(logits, label)
        cos_plus = (F.normalize(h_adv - h_clean, dim=1) * mu).sum(1).mean()
        cos_away = -cos_plus
        if mode == "ce":
            objective = ce
        elif mode in {"away", "random"}:
            objective = cos_away if mode == "away" else cos_plus
        else:
            objective = ce + beta * cos_away
        grad = torch.autograd.grad(objective, adv, retain_graph=False, create_graph=False)[0]
        adv = project_linf(adv.detach() + step_size * grad.sign(), clean, eps)
    return adv.detach()


def model_metrics(logits_clean: torch.Tensor, logits_adv: torch.Tensor, label: torch.Tensor, prefix: str) -> dict[str, float]:
    y = int(label.item())
    clean_prob = logits_clean.softmax(1)[0, y]
    adv_prob = logits_adv.softmax(1)[0, y]
    clean_margin = true_margin(logits_clean, label)
    adv_margin = true_margin(logits_adv, label)
    return {
        f"{prefix}_pred": int(logits_adv.argmax(1).item()),
        f"{prefix}_untargeted_success": int(logits_adv.argmax(1).item() != y),
        f"{prefix}_clean_true_logit": float(logits_clean[0, y].item()),
        f"{prefix}_adv_true_logit": float(logits_adv[0, y].item()),
        f"{prefix}_delta_true_logit": float((logits_adv[0, y] - logits_clean[0, y]).item()),
        f"{prefix}_clean_true_prob": float(clean_prob.item()),
        f"{prefix}_adv_true_prob": float(adv_prob.item()),
        f"{prefix}_delta_true_prob": float((adv_prob - clean_prob).item()),
        f"{prefix}_clean_true_margin": float(clean_margin.item()),
        f"{prefix}_adv_true_margin": float(adv_margin.item()),
        f"{prefix}_delta_true_margin": float((adv_margin - clean_margin).item()),
    }


def metrics(model: ResNet18WithFeatures, targets, clean: torch.Tensor, adv: torch.Tensor, label: torch.Tensor, attractors) -> dict[str, float]:
    out = {}
    y = int(label.item())
    with torch.no_grad():
        clean_logits = model(clean)
        adv_logits = model(adv)
        out.update(model_metrics(clean_logits, adv_logits, label, "surrogate"))
        for layer in ["avgpool", "layer4", "logits"]:
            _lc, h_clean = model.forward_with_feature(clean, layer)
            _la, h_adv = model.forward_with_feature(adv, layer)
            delta = h_adv - h_clean
            mu = attractors[layer][attractors["class_to_pos"][y] : attractors["class_to_pos"][y] + 1]
            cos_plus = float((F.normalize(delta, dim=1) * mu).sum(1).item())
            out[f"{layer}_cos_plus_mu_y"] = cos_plus
            out[f"{layer}_cos_minus_mu_y"] = -cos_plus
            out[f"{layer}_delta_l2"] = float(delta.norm(p=2, dim=1).item())
        for name, target_model in targets.items():
            out.update(model_metrics(target_model(clean), target_model(adv), label, name))
    d = adv - clean
    out["linf"] = float(d.abs().max().item())
    out["l2"] = float(d.flatten(1).norm(p=2, dim=1).item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--attractor-npz", default="analysis_outputs/pure_af_geometry/class_attractor_validation_resnet18_c10_s5/class_mean_directions.npz")
    parser.add_argument("--output-csv", default="analysis_outputs/pure_af_geometry/away_from_true_attractor_resnet18_100.csv")
    parser.add_argument("--classes", default="0-9")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-models", default="densenet121,vgg16")
    parser.add_argument("--allow-partial", action="store_true", help="Use all available clean-correct images if per-class target cannot be met.")
    parser.add_argument(
        "--variants",
        default="ce_untargeted,random_direction,away_avgpool,away_layer4,away_logits,ce_plus_away_avgpool,ce_plus_away_layer4,ce_plus_away_logits",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    model = ResNet18WithFeatures().to(device).eval()
    target_names = [x.strip() for x in args.target_models.split(",") if x.strip()]
    targets = {name: load_torchvision_model(name, device) for name in target_names}
    attractors = load_attractors(args.attractor_npz, device)
    classes = parse_classes(args.classes)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    indices = select_indices(dataset, classes, args.per_class, model, device, args.seed, args.batch_size, args.allow_partial)[: args.max_images]
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed + 12345)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_idx", "source_class", "variant",
        "surrogate_pred", "surrogate_untargeted_success",
        "surrogate_clean_true_logit", "surrogate_adv_true_logit", "surrogate_delta_true_logit",
        "surrogate_clean_true_prob", "surrogate_adv_true_prob", "surrogate_delta_true_prob",
        "surrogate_clean_true_margin", "surrogate_adv_true_margin", "surrogate_delta_true_margin",
        "avgpool_cos_plus_mu_y", "avgpool_cos_minus_mu_y", "layer4_cos_plus_mu_y", "layer4_cos_minus_mu_y",
        "logits_cos_plus_mu_y", "logits_cos_minus_mu_y",
        "avgpool_delta_l2", "layer4_delta_l2", "logits_delta_l2", "linf", "l2", "elapsed_sec",
    ]
    for name in target_names:
        fieldnames += [
            f"{name}_pred", f"{name}_untargeted_success",
            f"{name}_clean_true_logit", f"{name}_adv_true_logit", f"{name}_delta_true_logit",
            f"{name}_clean_true_prob", f"{name}_adv_true_prob", f"{name}_delta_true_prob",
            f"{name}_clean_true_margin", f"{name}_adv_true_margin", f"{name}_delta_true_margin",
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
                adv = optimize(clean, label, model, variant, attractors, args.eps, args.step_size, args.steps, args.beta, rng)
                row = {
                    "dataset_idx": idx,
                    "source_class": int(label.item()),
                    "variant": variant,
                    **metrics(model, targets, clean, adv, label, attractors),
                    "elapsed_sec": time.time() - t0,
                }
                writer.writerow(row)
                f.flush()
            if image_i % 10 == 0:
                print(f"[progress] images={image_i}/{len(indices)} elapsed={time.time() - started:.1f}s", flush=True)

    summary = {
        "args": vars(args),
        "indices": indices,
        "classes": classes,
        "variants": variants,
        "target_models": target_names,
        "output_csv": str(output),
        "elapsed_sec": time.time() - started,
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
