#!/usr/bin/env python3
"""Fast targeted transfer test for classifier class-attractor directions."""

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
        raise ValueError(f"Unsupported attractor layer: {layer}")


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


def load_attractors(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    data = np.load(path)
    classes = data["classes"].astype(int).tolist()
    class_to_pos = {cls: i for i, cls in enumerate(classes)}
    out = {}
    for layer in ["avgpool", "layer4", "logits"]:
        key = f"clf_{layer}"
        values = torch.from_numpy(data[key]).float().to(device)
        out[layer] = F.normalize(values, dim=1)
    out["classes"] = torch.tensor(classes, dtype=torch.long, device=device)
    out["class_to_pos"] = class_to_pos
    return out


def project_linf(x: torch.Tensor, clean: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x, clean + eps), clean - eps).clamp(0.0, 1.0)


def target_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_logit = logits.gather(1, target.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, target.view(-1, 1), -1e9)
    return target_logit - masked.max(1).values


def select_indices(dataset, classes: list[int], per_class: int, surrogate, device, seed: int, batch_size: int) -> list[int]:
    rng = random.Random(seed)
    selected: dict[int, list[int]] = {cls: [] for cls in classes}
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
            pred = surrogate(x).argmax(1)
            ok = (pred == y).detach().cpu().tolist()
            labels = y.detach().cpu().tolist()
            for offset, is_ok in enumerate(ok):
                cls = int(labels[offset])
                if is_ok and len(selected[cls]) < per_class:
                    selected[cls].append(candidates[seen + offset])
            seen += len(ok)
            if all(len(selected[cls]) >= per_class for cls in classes):
                break
    missing = {cls: per_class - len(values) for cls, values in selected.items() if len(values) < per_class}
    if missing:
        raise RuntimeError(f"Could not select enough clean ResNet18-correct images: {missing}")
    ordered = []
    for cls in classes:
        ordered.extend(selected[cls])
    return ordered


def attack(
    *,
    clean: torch.Tensor,
    label: torch.Tensor,
    target: torch.Tensor,
    surrogate: ResNet18WithFeatures,
    variant: str,
    attractors: dict[str, torch.Tensor],
    beta: float,
    eps: float,
    step_size: float,
    steps: int,
    random_start: bool,
    decay: float,
) -> torch.Tensor:
    if random_start:
        adv = project_linf(clean + torch.empty_like(clean).uniform_(-eps, eps), clean, eps)
    else:
        adv = clean.detach().clone()
    momentum = torch.zeros_like(clean)
    layer = ""
    if variant.startswith("pgd_attractor_"):
        layer = variant.replace("pgd_attractor_", "")
        with torch.no_grad():
            _logits_clean, h_clean = surrogate.forward_with_feature(clean, layer)
        class_to_pos = attractors["class_to_pos"]
        mu_pos = [class_to_pos[int(t.item())] for t in target]
        mu = attractors[layer][torch.tensor(mu_pos, device=clean.device)]
    for _step in range(steps):
        adv.requires_grad_(True)
        if variant == "sini_fgsm":
            loss = 0.0
            look = project_linf(adv - step_size * decay * momentum.sign(), clean, eps)
            for scale in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                logits = surrogate(look * scale)
                loss = loss + F.cross_entropy(logits, target)
            loss = loss / 5.0
        else:
            logits, h_adv = surrogate.forward_with_feature(adv, layer or "logits")
            loss = F.cross_entropy(logits, target)
            if layer:
                delta_h = F.normalize(h_adv - h_clean, dim=1)
                align = (delta_h * mu).sum(1).mean()
                loss = loss - beta * align
        grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
        if variant in {"mi_fgsm", "sini_fgsm"}:
            grad = grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
            momentum = decay * momentum + grad
            direction = momentum.sign()
        else:
            direction = grad.sign()
        adv = project_linf(adv.detach() - step_size * direction, clean, eps)
    return adv.detach()


def evaluate_logits(logits: torch.Tensor, label: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(1)
    return {
        "pred": int(pred.item()),
        "targeted_success": int(pred.item() == int(target.item())),
        "untargeted_success": int(pred.item() != int(label.item())),
        "target_prob": float(logits.softmax(1)[0, int(target.item())].item()),
        "target_margin": float(target_margin(logits, target).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--attractor-npz", default="analysis_outputs/pure_af_geometry/class_attractor_validation_resnet18_c10_s5/class_mean_directions.npz")
    parser.add_argument("--output-csv", default="analysis_outputs/pure_af_geometry/attractor_guided_targeted_transfer_resnet18_100.csv")
    parser.add_argument("--classes", default="0-9")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--step-size", type=float, default=0.005)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--variants", default="pgd,mi_fgsm,sini_fgsm,pgd_attractor_avgpool,pgd_attractor_logits,pgd_attractor_layer4")
    parser.add_argument("--target-models", default="densenet121,vgg16")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    classes = parse_classes(args.classes)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    target_model_names = [x.strip() for x in args.target_models.split(",") if x.strip()]

    surrogate = ResNet18WithFeatures().to(device).eval()
    targets = {name: load_torchvision_model(name, device) for name in target_model_names}
    attractors = load_attractors(args.attractor_npz, device)
    indices = select_indices(dataset, classes, args.per_class, surrogate, device, args.seed, args.batch_size)
    indices = indices[: args.max_images]
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset_idx", "source_class", "target_class", "variant", "surrogate_pred_clean",
        "surrogate_targeted_success", "surrogate_untargeted_success", "surrogate_target_prob", "surrogate_target_margin",
        "linf", "l2", "elapsed_sec",
    ]
    for name in target_model_names:
        fieldnames += [
            f"{name}_pred", f"{name}_targeted_success", f"{name}_untargeted_success",
            f"{name}_target_prob", f"{name}_target_margin",
        ]

    done = set()
    if output.exists():
        old = list(csv.DictReader(output.open()))
        done = {(int(r["dataset_idx"]), r["variant"]) for r in old}

    write_header = not output.exists()
    started = time.time()
    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for count, idx in enumerate(indices, start=1):
            clean_cpu, label_int = dataset[idx]
            label = torch.tensor([int(label_int)], device=device)
            target_int = classes[(classes.index(int(label_int)) + 1) % len(classes)]
            target = torch.tensor([target_int], device=device)
            clean = clean_cpu.unsqueeze(0).to(device)
            with torch.no_grad():
                clean_pred = int(surrogate(clean).argmax(1).item())
            for variant in variants:
                if (idx, variant) in done:
                    continue
                t0 = time.time()
                adv = attack(
                    clean=clean,
                    label=label,
                    target=target,
                    surrogate=surrogate,
                    variant=variant,
                    attractors=attractors,
                    beta=args.beta,
                    eps=args.eps,
                    step_size=args.step_size,
                    steps=args.steps,
                    random_start=args.random_start,
                    decay=1.0,
                )
                delta = adv - clean
                with torch.no_grad():
                    sur_eval = evaluate_logits(surrogate(adv), label, target)
                    row = {
                        "dataset_idx": idx,
                        "source_class": int(label.item()),
                        "target_class": int(target.item()),
                        "variant": variant,
                        "surrogate_pred_clean": clean_pred,
                        "surrogate_targeted_success": sur_eval["targeted_success"],
                        "surrogate_untargeted_success": sur_eval["untargeted_success"],
                        "surrogate_target_prob": sur_eval["target_prob"],
                        "surrogate_target_margin": sur_eval["target_margin"],
                        "linf": float(delta.abs().max().item()),
                        "l2": float(delta.flatten(1).norm(p=2, dim=1).item()),
                        "elapsed_sec": time.time() - t0,
                    }
                    for name, model in targets.items():
                        metrics = evaluate_logits(model(adv), label, target)
                        row[f"{name}_pred"] = metrics["pred"]
                        row[f"{name}_targeted_success"] = metrics["targeted_success"]
                        row[f"{name}_untargeted_success"] = metrics["untargeted_success"]
                        row[f"{name}_target_prob"] = metrics["target_prob"]
                        row[f"{name}_target_margin"] = metrics["target_margin"]
                writer.writerow(row)
                f.flush()
            if count % 10 == 0:
                print(f"[progress] images={count}/{len(indices)} elapsed={time.time() - started:.1f}s out={output}", flush=True)

    metadata = {
        "args": vars(args),
        "indices": indices,
        "classes": classes,
        "variants": variants,
        "target_models": target_model_names,
        "output_csv": str(output),
        "elapsed_sec": time.time() - started,
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
