#!/usr/bin/env python3
"""Shared GTSRB data and model utilities for the EAAI case study."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, models, transforms
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet18_Weights


NUM_CLASSES = 43
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_MODELS = ("resnet18", "convnext_tiny")
FEATURE_MODULES = {
    "resnet18": {
        "stage2": "backbone.layer2",
        "stage3": "backbone.layer3",
        "stage4": "backbone.layer4",
        "penultimate": "backbone.avgpool",
    },
    "convnext_tiny": {
        "stage2": "backbone.features.3",
        "stage3": "backbone.features.5",
        "stage4": "backbone.features.7",
        "penultimate": "backbone.classifier.1",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Normalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class GTSRBClassifier(nn.Module):
    """Pixel-space classifier with normalization included in the forward pass."""

    def __init__(self, architecture: str, pretrained: bool = True) -> None:
        super().__init__()
        self.architecture = architecture
        self.normalize = Normalize()
        if architecture == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, NUM_CLASSES)
        elif architecture == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
            self.backbone.classifier[2] = nn.Linear(
                self.backbone.classifier[2].in_features, NUM_CLASSES
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.normalize(x))


class FeatureCapture(nn.Module):
    """Capture pooled hidden tensors at explicitly registered module outputs."""

    def __init__(self, model: GTSRBClassifier) -> None:
        super().__init__()
        self.model = model
        self.features: dict[str, torch.Tensor] = {}
        modules = dict(model.named_modules())
        self.handles = []
        for label, module_name in FEATURE_MODULES[model.architecture].items():
            if module_name not in modules:
                raise KeyError(f"Missing feature module {module_name}")
            self.handles.append(
                modules[module_name].register_forward_hook(self._hook(label))
            )

    @staticmethod
    def _pool(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 4:
            return F.adaptive_avg_pool2d(value, 1).flatten(1)
        return value.flatten(1)

    def _hook(self, label: str):
        def capture(_module, _inputs, output):
            self.features[label] = self._pool(output)

        return capture

    @property
    def layers(self) -> list[str]:
        return list(FEATURE_MODULES[self.model.architecture]) + ["logits"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.features = {}
        logits = self.model(x)
        self.features["logits"] = logits
        return logits

    def forward_with_features(self, x: torch.Tensor):
        logits = self.forward(x)
        return logits, dict(self.features)

    def feature(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        _logits = self.forward(x)
        return self.features[layer]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def build_model(architecture: str, pretrained: bool = True) -> GTSRBClassifier:
    if architecture not in SUPPORTED_MODELS:
        raise ValueError(f"Expected one of {SUPPORTED_MODELS}, got {architecture}")
    return GTSRBClassifier(architecture, pretrained=pretrained)


def train_transform(image_size: int) -> transforms.Compose:
    # Horizontal flips are invalid for directional traffic signs.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(
                degrees=8,
                translate=(0.08, 0.08),
                scale=(0.90, 1.10),
            ),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15),
            transforms.ToTensor(),
        ]
    )


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def gtsrb_dataset(
    root: str | Path,
    split: str,
    image_size: int,
    training: bool = False,
    download: bool = False,
):
    transform = train_transform(image_size) if training else eval_transform(image_size)
    return datasets.GTSRB(
        str(root), split=split, transform=transform, download=download
    )


def stratified_train_val_indices(dataset, val_fraction: float, seed: int):
    labels = np.asarray([int(label) for _path, label in dataset._samples])
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return ece


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: str | Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(payload["architecture"], pretrained=False)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload


def write_json(path: str | Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
