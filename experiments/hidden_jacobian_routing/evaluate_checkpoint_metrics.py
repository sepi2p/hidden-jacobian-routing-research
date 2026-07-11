#!/usr/bin/env python3
"""Evaluate clean accuracy, NLL, confidence, and ECE for registered CIFAR models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.hidden_jacobian_routing.common import MODELS, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--dataset-root", default="data/cifar10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs/hidden_jacobian_routing/checkpoint_metrics.csv"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    rows = []
    for name in args.models.split(","):
        model = load_model(name.strip(), device).eval()
        confidences, predictions, labels = [], [], []
        nll = 0.0
        with torch.inference_mode():
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                probs = logits.softmax(1)
                confidence, prediction = probs.max(1)
                nll += F.cross_entropy(logits, y, reduction="sum").item()
                confidences.append(confidence.cpu())
                predictions.append(prediction.cpu())
                labels.append(y.cpu())
        confidence = torch.cat(confidences)
        prediction = torch.cat(predictions)
        label = torch.cat(labels)
        correct = prediction.eq(label)
        edges = torch.linspace(0, 1, args.ece_bins + 1)
        ece = 0.0
        for index in range(args.ece_bins):
            mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
            if mask.any():
                ece += mask.float().mean().item() * abs(
                    confidence[mask].mean().item() - correct[mask].float().mean().item()
                )
        rows.append(
            {
                "model": name.strip(),
                "n_test": len(label),
                "clean_accuracy": correct.float().mean().item(),
                "nll": nll / len(label),
                "ece15": ece,
                "mean_confidence": confidence.mean().item(),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
