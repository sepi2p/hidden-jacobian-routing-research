#!/usr/bin/env python3
"""Compare class-wise universal perturbations with away-from-attractor directions."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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

    def forward(self, x):
        return (x - self.mean) / self.std


class ResNet18WithFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.base = models.resnet18(pretrained=True)

    def forward(self, x):
        return self.forward_with_feature(x, "logits")[0]

    def forward_with_feature(self, x, layer: str):
        z = self.normalize(x)
        z = self.base.conv1(z)
        z = self.base.bn1(z)
        z = self.base.relu(z)
        z = self.base.maxpool(z)
        z = self.base.layer1(z)
        z = self.base.layer2(z)
        z = self.base.layer3(z)
        z = self.base.layer4(z)
        pooled = self.base.avgpool(z).flatten(1)
        logits = self.base.fc(pooled)
        if layer in {"avgpool", "layer4"}:
            return logits, pooled
        if layer == "logits":
            return logits, logits
        raise ValueError(layer)


def load_target(name: str, device):
    if name == "densenet121":
        base = models.densenet121(pretrained=True)
    elif name == "vgg16":
        base = models.vgg16_bn(pretrained=True)
    else:
        raise ValueError(name)
    return nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), base).to(device).eval()


def load_attractors(path: str, device):
    data = np.load(path)
    classes = data["classes"].astype(int).tolist()
    out = {"classes": classes, "class_to_pos": {c: i for i, c in enumerate(classes)}}
    for layer in ["avgpool", "layer4", "logits"]:
        out[layer] = F.normalize(torch.from_numpy(data[f"clf_{layer}"]).float().to(device), dim=1)
    return out


def project(x, clean, eps):
    return torch.max(torch.min(x, clean + eps), clean - eps).clamp(0, 1)


def train_uap(dataset, indices, label, model, device, args):
    delta = torch.zeros((1, 3, 224, 224), device=device)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    batches = list(loader)
    rng = random.Random(args.seed + int(label))
    for _ in range(args.uap_steps):
        x, y = rng.choice(batches)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        delta = delta.detach().requires_grad_(True)
        adv = (x + delta).clamp(0, 1)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, delta)[0]
        delta = (delta + args.step_size * grad.sign()).detach().clamp(-args.eps, args.eps)
    return delta.detach()


def eval_rows(dataset, indices, label, delta, model, targets, attractors, device):
    rows = []
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=2)
    shifts = {"avgpool": [], "logits": []}
    for offset, (x, y) in enumerate(loader):
        idx = int(indices[offset])
        x = x.to(device)
        y = y.to(device)
        adv = project(x + delta, x, float(delta.abs().max().item()))
        with torch.no_grad():
            clean_logits = model(x)
            adv_logits = model(adv)
            pred = int(adv_logits.argmax(1).item())
            row = {
                "dataset_idx": idx,
                "source_class": int(label),
                "method": "class_uap",
                "surrogate_pred": pred,
                "surrogate_untargeted_success": int(pred != int(y.item())),
                "surrogate_delta_true_logit": float((adv_logits[0, int(y.item())] - clean_logits[0, int(y.item())]).item()),
            }
            for name, target in targets.items():
                c = target(x)
                a = target(adv)
                p = int(a.argmax(1).item())
                row[f"{name}_untargeted_success"] = int(p != int(y.item()))
                row[f"{name}_delta_true_logit"] = float((a[0, int(y.item())] - c[0, int(y.item())]).item())
            for layer in ["avgpool", "logits"]:
                _lc, hc = model.forward_with_feature(x, layer)
                _la, ha = model.forward_with_feature(adv, layer)
                shifts[layer].append((ha - hc).detach())
            row["linf"] = float((adv - x).abs().max().item())
            rows.append(row)
    cos_rows = {}
    for layer, values in shifts.items():
        mean_shift = torch.cat(values, 0).mean(0, keepdim=True)
        mu = attractors[layer][attractors["class_to_pos"][int(label)] : attractors["class_to_pos"][int(label)] + 1]
        cos_rows[f"{layer}_mean_shift_cos_minus_mu"] = float((F.normalize(mean_shift, dim=1) * -mu).sum(1).item())
        cos_rows[f"{layer}_mean_shift_cos_plus_mu"] = -cos_rows[f"{layer}_mean_shift_cos_minus_mu"]
    for row in rows:
        row.update(cos_rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    p.add_argument("--cohort-metadata", default="analysis_outputs/pure_af_geometry/away_from_true_attractor_resnet18_all_available_c10.csv.metadata.json")
    p.add_argument("--away-csv", default="analysis_outputs/pure_af_geometry/away_from_true_attractor_resnet18_all_available_c10.csv")
    p.add_argument("--attractor-npz", default="analysis_outputs/pure_af_geometry/class_attractor_validation_resnet18_c10_s5/class_mean_directions.npz")
    p.add_argument("--output-csv", default="analysis_outputs/pure_af_geometry/class_uap_vs_attractor_resnet18_c10_holdout.csv")
    p.add_argument("--summary-csv", default="analysis_outputs/pure_af_geometry/class_uap_vs_attractor_resnet18_c10_holdout_summary.csv")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--step-size", type=float, default=0.005)
    p.add_argument("--uap-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    indices = json.loads(Path(args.cohort_metadata).read_text())["indices"]
    by_class = defaultdict(list)
    for idx in indices:
        by_class[int(dataset.samples[int(idx)][1])].append(int(idx))
    rng = random.Random(args.seed)
    splits = {}
    for cls, vals in sorted(by_class.items()):
        vals = vals[:]
        rng.shuffle(vals)
        n_train = max(1, int(round(len(vals) * args.train_frac)))
        splits[cls] = {"train": vals[:n_train], "eval": vals[n_train:]}

    model = ResNet18WithFeatures().to(device).eval()
    targets = {name: load_target(name, device) for name in ["densenet121", "vgg16"]}
    attractors = load_attractors(args.attractor_npz, device)
    rows = []
    for cls, split in splits.items():
        print(f"[class {cls}] train={len(split['train'])} eval={len(split['eval'])}", flush=True)
        delta = train_uap(dataset, split["train"], cls, model, device, args)
        rows.extend(eval_rows(dataset, split["eval"], cls, delta, model, targets, attractors, device))

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    uap = pd.DataFrame(rows)
    away = pd.read_csv(args.away_csv)
    eval_indices = sorted(uap["dataset_idx"].unique().tolist())
    away = away[away["dataset_idx"].isin(eval_indices) & away["variant"].isin(["away_logits", "away_avgpool", "away_layer4"])].copy()
    away["method"] = away["variant"]
    rename = {
        "densenet121_untargeted_success": "densenet121_untargeted_success",
        "vgg16_untargeted_success": "vgg16_untargeted_success",
    }
    combined = pd.concat([uap, away.rename(columns=rename)], ignore_index=True, sort=False)
    summary = combined.groupby("method").agg(
        n=("dataset_idx", "size"),
        surrogate_asr=("surrogate_untargeted_success", "mean"),
        densenet121_asr=("densenet121_untargeted_success", "mean"),
        vgg16_asr=("vgg16_untargeted_success", "mean"),
        surrogate_delta_true_logit=("surrogate_delta_true_logit", "mean"),
        avgpool_mean_shift_cos_minus_mu=("avgpool_mean_shift_cos_minus_mu", "mean"),
        logits_mean_shift_cos_minus_mu=("logits_mean_shift_cos_minus_mu", "mean"),
        linf=("linf", "mean"),
    ).reset_index()
    summary.to_csv(args.summary_csv, index=False)
    out.with_suffix(out.suffix + ".metadata.json").write_text(json.dumps({"args": vars(args), "splits": splits, "rows": len(rows)}, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
