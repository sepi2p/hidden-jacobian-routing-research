#!/usr/bin/env python3
"""Compare true-class attractor removal directions with local CE feature gradients."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
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
        layer4 = self.base.avgpool(z).flatten(1)
        avgpool = layer4
        logits = self.base.fc(avgpool)
        if layer == "layer4":
            return logits, layer4
        if layer == "avgpool":
            return logits, avgpool
        if layer == "logits":
            return logits, logits
        raise ValueError(f"Unsupported layer: {layer}")


def load_attractors(path: str, device: torch.device):
    data = np.load(path)
    classes = data["classes"].astype(int).tolist()
    class_to_pos = {cls: i for i, cls in enumerate(classes)}
    out = {"classes": classes, "class_to_pos": class_to_pos}
    for layer in ["avgpool", "layer4", "logits"]:
        values = torch.from_numpy(data[f"clf_{layer}"]).float().to(device)
        out[layer] = F.normalize(values, dim=1)
    return out


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((F.normalize(a, dim=1) * F.normalize(b, dim=1)).sum(1).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--attractor-npz", default="analysis_outputs/pure_af_geometry/class_attractor_validation_resnet18_c10_s5/class_mean_directions.npz")
    parser.add_argument("--indices-metadata", default="analysis_outputs/pure_af_geometry/away_from_true_attractor_resnet18_100.csv.metadata.json")
    parser.add_argument("--output-csv", default="analysis_outputs/pure_af_geometry/attractor_gradient_alignment_resnet18_100.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    indices = json.loads(Path(args.indices_metadata).read_text())["indices"]
    model = ResNet18WithFeatures().to(device).eval()
    attractors = load_attractors(args.attractor_npz, device)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in indices:
        clean_cpu, label_int = dataset[int(idx)]
        clean = clean_cpu.unsqueeze(0).to(device)
        label = torch.tensor([int(label_int)], device=device)
        y = int(label.item())
        row = {"dataset_idx": int(idx), "source_class": y}
        for layer in ["avgpool", "layer4", "logits"]:
            clean_req = clean.detach().clone().requires_grad_(True)
            logits, h = model.forward_with_feature(clean_req, layer)
            h.retain_grad()
            ce = F.cross_entropy(logits, label)
            grad_h = torch.autograd.grad(ce, h, retain_graph=False, create_graph=False)[0].detach()
            mu = attractors[layer][attractors["class_to_pos"][y] : attractors["class_to_pos"][y] + 1]
            neg_mu = -mu
            row[f"{layer}_cos_grad_ce_with_minus_mu"] = cos(grad_h, neg_mu)
            row[f"{layer}_cos_grad_ce_with_plus_mu"] = cos(grad_h, mu)
            row[f"{layer}_grad_ce_l2"] = float(grad_h.norm(p=2, dim=1).item())
            row[f"{layer}_minus_mu_l2"] = float(neg_mu.norm(p=2, dim=1).item())

            if layer == "logits":
                with torch.no_grad():
                    analytic = logits.softmax(1)
                    analytic[0, y] -= 1.0
                    row["logits_cos_analytic_ce_grad_with_minus_mu"] = cos(analytic, neg_mu)
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "imagenet_root": args.imagenet_root,
        "attractor_npz": args.attractor_npz,
        "indices_metadata": args.indices_metadata,
        "output_csv": str(output),
        "rows": len(rows),
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
