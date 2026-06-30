#!/usr/bin/env python3
"""Test dirty-image boundary flag directions against class-attractor removal directions."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
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
        layer4 = F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
        avgpool = self.base.avgpool(z).flatten(1)
        logits = self.base.fc(avgpool)
        if layer == "layer4":
            return logits, layer4
        if layer == "avgpool":
            return logits, avgpool
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


def true_margin(logits, label):
    y = label.view(-1, 1)
    true = logits.gather(1, y).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y, -1e9)
    return true - masked.max(1).values


def project_linf(x, clean, eps):
    return torch.max(torch.min(x, clean + eps), clean - eps).clamp(0, 1)


def load_attractors(path: str, device):
    data = np.load(path)
    classes = data["classes"].astype(int).tolist()
    out = {"classes": classes, "class_to_pos": {c: i for i, c in enumerate(classes)}}
    for layer in ["avgpool", "layer4", "logits"]:
        out[layer] = F.normalize(torch.from_numpy(data[f"clf_{layer}"]).float().to(device), dim=1)
    return out


def image_tensor_from_path(path: str, transform, device):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def load_pt_tensor(path: str, transform, device):
    x = torch.load(path, map_location="cpu")
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x.float().to(device).clamp(0, 1)


def build_boundary_dirs(args, model, attractors, transform, device):
    pure = pd.read_csv(args.pure_manifest)
    dirty = pd.read_csv(args.dirty_manifest)
    pure = pure[(pure["init_mode"] == args.init_mode) & (pure["target_class"].between(0, 9))]
    dirty = dirty[(dirty["init_mode"] == args.init_mode) & (dirty["target_class"].between(0, 9))]
    dirs = {}
    meta = {}
    with torch.no_grad():
        for cls in sorted(set(pure["target_class"].astype(int)) & set(dirty["target_class"].astype(int))):
            prow = pure[pure["target_class"].astype(int) == cls].iloc[0]
            drow = dirty[dirty["target_class"].astype(int) == cls].iloc[0]
            pure_x = image_tensor_from_path(prow["final_image"], transform, device)
            dirty_x = load_pt_tensor(drow["final_tensor"], transform, device)
            dirs[cls] = {}
            meta[cls] = {"pure_run": prow["run_name"], "dirty_run": drow["run_name"]}
            for layer in ["avgpool", "layer4", "logits"]:
                _lp, hp = model.forward_with_feature(pure_x, layer)
                _ld, hd = model.forward_with_feature(dirty_x, layer)
                b = F.normalize(hd - hp, dim=1)
                mu = attractors[layer][attractors["class_to_pos"][cls] : attractors["class_to_pos"][cls] + 1]
                away = -mu
                par = (away * b).sum(1, keepdim=True) * b
                orth = away - par
                orth = F.normalize(orth, dim=1)
                dirs[cls][layer] = {"b": b.detach(), "away": away.detach(), "orth": orth.detach()}
                meta[cls][f"{layer}_cos_b_away_mu"] = float((b * away).sum(1).item())
    return dirs, meta


def select_indices(dataset, indices_metadata, max_images):
    values = json.loads(Path(indices_metadata).read_text())["indices"]
    return [int(x) for x in values[:max_images]]


def variant_spec(variant):
    if variant == "ce_untargeted":
        return "ce", "logits"
    parts = variant.split("__", 1)
    if len(parts) != 2:
        raise ValueError(variant)
    base, layer = parts
    return base, layer


def optimize(clean, label, model, dirs, variant, eps, step_size, steps, beta):
    mode, layer = variant_spec(variant)
    adv = clean.detach().clone()
    first_mis = -1
    if mode != "ce":
        with torch.no_grad():
            _lc, h_clean = model.forward_with_feature(clean, layer)
        target_dir = {"away_mu": "away", "dirty_parallel": "b", "dirty_orthogonal": "orth", "ce_plus_dirty_orthogonal": "orth"}[mode]
        direction = dirs[int(label.item())][layer][target_dir]
    else:
        h_clean = None
        direction = None
    for step in range(1, steps + 1):
        adv.requires_grad_(True)
        logits, h_adv = model.forward_with_feature(adv, layer)
        ce = F.cross_entropy(logits, label)
        if mode == "ce":
            objective = ce
        else:
            align = (F.normalize(h_adv - h_clean, dim=1) * direction).sum(1).mean()
            objective = ce + beta * align if mode == "ce_plus_dirty_orthogonal" else align
        grad = torch.autograd.grad(objective, adv)[0]
        adv = project_linf(adv.detach() + step_size * grad.sign(), clean, eps)
        if first_mis < 0:
            with torch.no_grad():
                pred = int(model(adv).argmax(1).item())
            if pred != int(label.item()):
                first_mis = step
    return adv.detach(), first_mis


def eval_model_metrics(clean_logits, adv_logits, label, prefix):
    pred = int(adv_logits.argmax(1).item())
    y = int(label.item())
    cm = true_margin(clean_logits, label)
    am = true_margin(adv_logits, label)
    return {
        f"{prefix}_pred": pred,
        f"{prefix}_untargeted_success": int(pred != y),
        f"{prefix}_delta_true_logit": float((adv_logits[0, y] - clean_logits[0, y]).item()),
        f"{prefix}_delta_true_margin": float((am - cm).item()),
    }


def measure(clean, adv, label, model, target_models, dirs, layer):
    y = int(label.item())
    out = {}
    with torch.no_grad():
        cl = model(clean)
        al = model(adv)
        out.update(eval_model_metrics(cl, al, label, "surrogate"))
        _lc, hc = model.forward_with_feature(clean, layer)
        _la, ha = model.forward_with_feature(adv, layer)
        delta = F.normalize(ha - hc, dim=1)
        for name, key in [("b", "b"), ("minus_mu", "away"), ("orth", "orth")]:
            out[f"cos_with_{name}"] = float((delta * dirs[y][layer][key]).sum(1).item())
        for name, tm in target_models.items():
            out.update(eval_model_metrics(tm(clean), tm(adv), label, name))
    d = adv - clean
    out["linf"] = float(d.abs().max().item())
    out["l2"] = float(d.flatten(1).norm(p=2, dim=1).item())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    p.add_argument("--indices-metadata", default="analysis_outputs/pure_af_geometry/away_from_true_attractor_resnet18_100.csv.metadata.json")
    p.add_argument("--attractor-npz", default="analysis_outputs/pure_af_geometry/class_attractor_validation_resnet18_c10_s5/class_mean_directions.npz")
    p.add_argument("--pure-manifest", default="analysis_outputs/pure_af_geometry/multiclass10_pure_resnet18/manifest.csv")
    p.add_argument("--dirty-manifest", default="analysis_outputs/pure_af_geometry/dirty_multiclass10_ga_resnet18/manifest.csv")
    p.add_argument("--init-mode", default="real", choices=["real", "random"])
    p.add_argument("--output-csv", default="analysis_outputs/pure_af_geometry/dirty_boundary_flag_orthogonal_resnet18_100.csv")
    p.add_argument("--summary-csv", default="analysis_outputs/pure_af_geometry/dirty_boundary_flag_orthogonal_resnet18_100_summary.csv")
    p.add_argument("--max-images", type=int, default=100)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--step-size", type=float, default=0.005)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    model = ResNet18WithFeatures().to(device).eval()
    targets = {name: load_target(name, device) for name in ["densenet121", "vgg16"]}
    attractors = load_attractors(args.attractor_npz, device)
    dirs, dir_meta = build_boundary_dirs(args, model, attractors, transform, device)
    indices = select_indices(dataset, args.indices_metadata, args.max_images)
    variants = ["ce_untargeted"]
    for layer in ["logits", "avgpool", "layer4"]:
        variants += [
            f"away_mu__{layer}",
            f"dirty_parallel__{layer}",
            f"dirty_orthogonal__{layer}",
            f"ce_plus_dirty_orthogonal__{layer}",
        ]
    rows = []
    for image_i, idx in enumerate(indices, start=1):
        clean_cpu, label_int = dataset[idx]
        clean = clean_cpu.unsqueeze(0).to(device)
        label = torch.tensor([int(label_int)], device=device)
        for variant in variants:
            mode, layer = variant_spec(variant)
            adv, first_mis = optimize(clean, label, model, dirs, variant, args.eps, args.step_size, args.steps, args.beta)
            rows.append({
                "dataset_idx": int(idx),
                "source_class": int(label.item()),
                "variant": variant,
                "variant_base": mode,
                "layer": layer,
                "first_misclassification_step": int(first_mis),
                **measure(clean, adv, label, model, targets, dirs, layer),
            })
        if image_i % 10 == 0:
            print(f"[progress] {image_i}/{len(indices)}", flush=True)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    df = pd.DataFrame(rows)
    summary = df.groupby(["variant", "variant_base", "layer"]).agg(
        n=("dataset_idx", "size"),
        surrogate_asr=("surrogate_untargeted_success", "mean"),
        densenet121_asr=("densenet121_untargeted_success", "mean"),
        vgg16_asr=("vgg16_untargeted_success", "mean"),
        surrogate_delta_true_logit=("surrogate_delta_true_logit", "mean"),
        surrogate_delta_true_margin=("surrogate_delta_true_margin", "mean"),
        cos_with_b=("cos_with_b", "mean"),
        cos_with_minus_mu=("cos_with_minus_mu", "mean"),
        cos_with_orth=("cos_with_orth", "mean"),
        mean_first_mis_step=("first_misclassification_step", lambda s: float(np.mean([x for x in s if x > 0])) if any(x > 0 for x in s) else np.nan),
        linf=("linf", "mean"),
    ).reset_index()
    summary.to_csv(args.summary_csv, index=False)
    out.with_suffix(out.suffix + ".metadata.json").write_text(json.dumps({
        "args": vars(args),
        "indices": indices,
        "direction_metadata": dir_meta,
        "variants": variants,
        "rows": len(rows),
    }, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
