"""Shared utilities for the hidden-Jacobian routing paper experiments."""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from utils.load_models import load_cifar_model
from attacks.square import p_selection


MODELS = ["bbb_resnet50", "bbb_vgg19_bn", "bbb_densenet", "bbb_inception_v3"]

DOMINANT = {
    "bbb_resnet50": "layer2",
    "bbb_vgg19_bn": "block2",
    "bbb_densenet": "denseblock3",
    "bbb_inception_v3": "mixed6",
}

PENULTIMATE = {
    "bbb_resnet50": "avgpool",
    "bbb_vgg19_bn": "penultimate",
    "bbb_densenet": "penultimate",
    "bbb_inception_v3": "penultimate",
}

LAYER_GROUPS = {
    "hidden": DOMINANT,
    "penultimate": PENULTIMATE,
    "logits": {m: "logits" for m in MODELS},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def project_linf(x_adv: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)


def margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.view(-1, 1), -1e9)
    return true - masked.max(1).values


class CIFARFeatureWrapper(nn.Module):
    """CIFAR classifier wrapper that exposes the paper's hidden-layer hooks."""

    def __init__(self, spec: str, model: nn.Module):
        super().__init__()
        self.spec = spec
        self.name = spec.replace(":", "_")
        self.model = model
        self.enabled = False
        self.captures: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.handles = []
        self.labels = self._register_hooks()

    def _make_hook(self, label: str):
        def hook(_module, _inp, out):
            if self.enabled and torch.is_tensor(out):
                self.captures[label].append(out)
        return hook

    def _register(self, module_name: str, label: str) -> bool:
        modules = dict(self.model.named_modules())
        if module_name not in modules:
            return False
        self.handles.append(modules[module_name].register_forward_hook(self._make_hook(label)))
        return True

    def _register_hooks(self) -> list[str]:
        labels = []
        if self.spec == "bbb_resnet50":
            for label, module_name in [
                ("layer1", "1.layer1"),
                ("layer2", "1.layer2"),
                ("layer3", "1.layer3"),
                ("layer4", "1.layer4"),
                ("avgpool", "1.layer4"),
            ]:
                if self._register(module_name, label):
                    labels.append(label)
        elif self.spec == "bbb_vgg19_bn":
            for label, module_name in [
                ("block1", "1.features.5"),
                ("block2", "1.features.12"),
                ("block3", "1.features.25"),
                ("block4", "1.features.38"),
                ("block5", "1.features.51"),
                ("penultimate", "1.features.52"),
            ]:
                if self._register(module_name, label):
                    labels.append(label)
        elif self.spec == "bbb_densenet":
            for label, module_name in [
                ("denseblock1", "1.dense1"),
                ("denseblock2", "1.dense2"),
                ("denseblock3", "1.dense3"),
                ("penultimate", "1.avgpool"),
            ]:
                if self._register(module_name, label):
                    labels.append(label)
        elif self.spec == "bbb_inception_v3":
            for label, module_name in [
                ("mixed5", "1.Mixed_5d"),
                ("mixed6", "1.Mixed_6e"),
                ("mixed7", "1.Mixed_7c"),
                ("penultimate", "1.avgpool"),
            ]:
                if self._register(module_name, label):
                    labels.append(label)
        if not labels:
            last_conv = None
            for module_name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module_name
            if last_conv is None:
                raise RuntimeError(f"No Conv2d feature found for {self.spec}")
            self._register(last_conv, "final_conv")
            labels.append("final_conv")
        labels.append("logits")
        return labels

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x.flatten(1)

    def _aggregate(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        pooled = [self._pool(x) for x in tensors]
        if len(pooled) == 1:
            return pooled[0]
        if all(p.shape == pooled[0].shape for p in pooled):
            return torch.stack(pooled, dim=0).mean(dim=0)
        return torch.cat(pooled, dim=1)

    def forward(self, x):
        return self.model(x)

    def forward_with_features(self, x):
        self.captures = defaultdict(list)
        self.enabled = True
        try:
            logits = self.model(x)
        finally:
            self.enabled = False
        feats = {}
        raw = {}
        for label in self.labels:
            if label == "logits":
                feats[label] = logits
                raw[label] = [logits]
            else:
                outs = self.captures.get(label, [])
                if not outs:
                    continue
                feats[label] = self._aggregate(outs)
                raw[label] = outs
        return logits, feats, raw

    def aggregate_grads(self, raw_by_label: dict[str, list[torch.Tensor]], raw_grads: list[torch.Tensor | None]):
        out = {}
        cursor = 0
        for label in self.labels:
            raws = raw_by_label.get(label, [])
            grads = []
            for raw in raws:
                g = raw_grads[cursor]
                cursor += 1
                grads.append(torch.zeros_like(raw) if g is None else g)
            if grads:
                out[label] = self._aggregate(grads)
        return out

    def pooled_layer4(self, x: torch.Tensor) -> torch.Tensor:
        _logits, feats, _raw = self.forward_with_features(x)
        if "layer4" in feats:
            return feats["layer4"]
        if "avgpool" in feats:
            return feats["avgpool"]
        return next(iter(feats.values()))


def load_model(spec: str, device: torch.device):
    model = load_cifar_model(spec).to(device).eval()
    return CIFARFeatureWrapper(spec, model).to(device).eval()


def select_clean_correct(dataset, wrappers, args, device):
    selected = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        ok = True
        with torch.no_grad():
            for model in args.models:
                if int(wrappers[model](x).argmax(1).item()) != int(y.item()):
                    ok = False
                    break
        if ok:
            selected.append((idx, int(y.item())))
        if len(selected) >= args.images:
            break
    return selected


def eval_all(wrappers, x_adv, y):
    out = {}
    with torch.no_grad():
        for model, wrapper in wrappers.items():
            logits = wrapper(x_adv)
            probs = F.softmax(logits, dim=1)
            out[model] = {
                "pred": int(logits.argmax(1).item()),
                "margin": float(margin(logits, y).item()),
                "true_prob": float(probs[0, int(y.item())].item()),
            }
    return out


def checkpoint_indices(n_steps: int, n_checkpoints: int) -> set[int]:
    if n_steps <= 0:
        return {0}
    n_checkpoints = max(2, min(n_checkpoints, n_steps + 1))
    return set(int(round(x)) for x in np.linspace(0, n_steps, n_checkpoints))


def square_trajectory(
    wrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int,
    seed: int,
    p_init: float,
    init_epochs: int,
    n_checkpoints: int,
):
    gen = torch.Generator(device=x.device).manual_seed(seed)
    x0 = x.detach()
    c, h, w = x.shape[1:]
    stripe = torch.where(
        torch.rand((1, c, 1, w), generator=gen, device=x.device) < 0.5,
        -torch.ones((1, c, 1, w), device=x.device),
        torch.ones((1, c, 1, w), device=x.device),
    ) * eps
    x_adv = (x0 + stripe).clamp(0, 1)
    states = [x_adv.detach().clone()]
    with torch.no_grad():
        best_margin = margin(wrapper(x_adv), y)
    save_at = checkpoint_indices(steps, n_checkpoints)
    for step in range(1, steps + 1):
        perturbation = x_adv - x0
        p = p_selection(p_init, step + init_epochs, steps)
        side = int(round(np.sqrt(p * c * h * w / c)))
        side = min(max(side, 1), h - 1)
        top = int(torch.randint(0, h - side + 1, (1,), generator=gen, device=x.device).item())
        left = int(torch.randint(0, w - side + 1, (1,), generator=gen, device=x.device).item())
        patch = torch.where(
            torch.rand((1, c, 1, 1), generator=gen, device=x.device) < 0.5,
            -torch.ones((1, c, 1, 1), device=x.device),
            torch.ones((1, c, 1, 1), device=x.device),
        ) * eps
        perturbation[:, :, top : top + side, left : left + side] = patch
        candidate = (x0 + perturbation).clamp(0, 1)
        with torch.no_grad():
            cand_margin = margin(wrapper(candidate), y)
        if float(cand_margin.item()) < float(best_margin.item()):
            x_adv = candidate.detach()
            best_margin = cand_margin.detach()
        if step in save_at:
            states.append(x_adv.detach().clone())
    return states
