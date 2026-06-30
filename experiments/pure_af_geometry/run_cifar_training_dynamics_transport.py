#!/usr/bin/env python3
"""Track adversarial transport geometry as CIFAR-10 classifiers learn.

This script is artifact-first:

* training checkpoints are saved when validation accuracy crosses target values;
* trajectory feature states are saved per seed/checkpoint/attack shard;
* analysis can be rerun without retraining or recollecting trajectories.

The goal is to test whether adversarial transport structure emerges during
supervised learning rather than appearing automatically in untrained networks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surro_models.cifar10_models.resnet import ResNet18  # noqa: E402


LAYERS = ["layer1", "layer2", "layer3", "layer4", "avgpool", "logits"]


def parse_csv(s: str, typ=str):
    return [typ(x.strip()) for x in s.split(",") if x.strip()]


def stable_hash(obj: dict) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class FeatureWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.enabled = False
        self.captures: dict[str, list[torch.Tensor]] = defaultdict(list)
        modules = dict(model.named_modules())
        self.handles = []
        for layer in ["layer1", "layer2", "layer3", "layer4"]:
            self.handles.append(modules[layer].register_forward_hook(self._hook(layer)))

    def _hook(self, label: str):
        def hook(_module, _inp, out):
            if self.enabled:
                self.captures[label].append(out)
                if label == "layer4":
                    self.captures["avgpool"].append(F.avg_pool2d(out, 4).flatten(1))

        return hook

    @staticmethod
    def pool(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def forward_features_nograd(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        self.captures = defaultdict(list)
        self.enabled = True
        try:
            logits = self.model(x)
        finally:
            self.enabled = False
        feats = {}
        for layer in ["layer1", "layer2", "layer3", "layer4"]:
            vals = self.captures.get(layer, [])
            if vals:
                feats[layer] = self.pool(vals[-1]).detach().cpu().numpy().astype(np.float32)
        vals = self.captures.get("avgpool", [])
        if vals:
            feats["avgpool"] = vals[-1].detach().cpu().numpy().astype(np.float32)
        feats["logits"] = logits.detach().cpu().numpy().astype(np.float32)
        return logits, feats

    def close(self):
        for h in self.handles:
            h.remove()


def build_datasets(root: str):
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_tf = transforms.ToTensor()
    train = datasets.CIFAR10(root, train=True, download=False, transform=train_tf)
    test = datasets.CIFAR10(root, train=False, download=False, transform=test_tf)
    return train, test


def build_eval_dataset(root: str):
    return datasets.CIFAR10(root, train=False, download=False, transform=transforms.ToTensor())


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 0) -> float:
    model.eval()
    ok = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            ok += int((model(x).argmax(1) == y).sum().item())
            total += int(y.numel())
    return ok / max(total, 1)


def checkpoint_name(seed: int, tag: str) -> str:
    clean = tag.replace(".", "p")
    return f"resnet18_seed{seed}_{clean}.pt"


def save_checkpoint(path: Path, model: nn.Module, opt, sched, seed: int, epoch: int, acc: float, tag: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "net": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": opt.state_dict() if opt is not None else None,
            "scheduler": sched.state_dict() if sched is not None else None,
            "seed": seed,
            "epoch": epoch,
            "acc": float(acc),
            "tag": tag,
        },
        path,
    )


def train_checkpoints(args, device: torch.device):
    train_set, test_set = build_datasets(args.dataset_root)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    ckpt_dir = Path(args.checkpoint_dir)
    targets = parse_csv(args.accuracy_targets, float)
    seeds = parse_csv(args.model_seeds, int)
    rows = []

    for seed in seeds:
        seed_dir = ckpt_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        latest = seed_dir / "latest.pt"
        set_seed(seed)
        model = ResNet18().to(device)
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        start_epoch = 0
        reached = set()
        if latest.exists() and args.resume:
            state = torch.load(latest, map_location=device)
            model.load_state_dict(state["net"])
            if state.get("optimizer"):
                opt.load_state_dict(state["optimizer"])
            if state.get("scheduler"):
                sched.load_state_dict(state["scheduler"])
            start_epoch = int(state.get("epoch", 0))
            for p in seed_dir.glob("resnet18_seed*_acc*.pt"):
                tag = p.stem.split("_")[-1]
                reached.add(tag)
            print(f"[RESUME] seed={seed} epoch={start_epoch} reached={sorted(reached)}", flush=True)

        if start_epoch == 0:
            init_acc = evaluate_accuracy(model, test_loader, device, args.eval_batches)
            init_path = seed_dir / checkpoint_name(seed, "init")
            if not init_path.exists():
                save_checkpoint(init_path, model, opt, sched, seed, 0, init_acc, "init")
            rows.append({"seed": seed, "epoch": 0, "acc": init_acc, "tag": "init", "path": str(init_path)})
            print(f"[SAVE INIT] seed={seed} acc={init_acc:.4f}", flush=True)

        for epoch in range(start_epoch + 1, args.epochs + 1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
            sched.step()
            acc = evaluate_accuracy(model, test_loader, device, args.eval_batches)
            save_checkpoint(latest, model, opt, sched, seed, epoch, acc, "latest")
            rows.append({"seed": seed, "epoch": epoch, "acc": acc, "tag": "latest", "path": str(latest)})
            for target in targets:
                tag = f"acc{int(round(target * 100)):02d}"
                if tag not in reached and acc >= target:
                    path = seed_dir / checkpoint_name(seed, tag)
                    save_checkpoint(path, model, opt, sched, seed, epoch, acc, tag)
                    reached.add(tag)
                    rows.append({"seed": seed, "epoch": epoch, "acc": acc, "tag": tag, "path": str(path)})
                    print(f"[SAVE TARGET] seed={seed} tag={tag} epoch={epoch} acc={acc:.4f}", flush=True)
            if epoch % args.save_epoch_every == 0 or epoch == args.epochs:
                tag = f"epoch{epoch:03d}"
                path = seed_dir / checkpoint_name(seed, tag)
                if not path.exists():
                    save_checkpoint(path, model, opt, sched, seed, epoch, acc, tag)
                rows.append({"seed": seed, "epoch": epoch, "acc": acc, "tag": tag, "path": str(path)})
            print(f"[TRAIN] seed={seed} epoch={epoch}/{args.epochs} acc={acc:.4f}", flush=True)

        final = seed_dir / checkpoint_name(seed, "final")
        if not final.exists():
            acc = evaluate_accuracy(model, test_loader, device, args.eval_batches)
            save_checkpoint(final, model, opt, sched, seed, args.epochs, acc, "final")
            rows.append({"seed": seed, "epoch": args.epochs, "acc": acc, "tag": "final", "path": str(final)})
        del model
        torch.cuda.empty_cache()

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest.to_csv(Path(args.output_dir) / "training_checkpoint_events.csv", index=False)


def checkpoint_manifest(args) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(args.checkpoint_dir).glob("seed*/resnet18_seed*.pt")):
        if p.name.endswith("latest.pt"):
            continue
        state = torch.load(p, map_location="cpu")
        rows.append(
            {
                "seed": int(state.get("seed", -1)),
                "epoch": int(state.get("epoch", -1)),
                "acc": float(state.get("acc", np.nan)),
                "tag": str(state.get("tag", p.stem.split("_")[-1])),
                "path": str(p),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No checkpoints found under {args.checkpoint_dir}")
    df = df.sort_values(["seed", "epoch", "tag"]).drop_duplicates(["seed", "tag"], keep="last")
    include = set(parse_csv(args.include_tags)) if args.include_tags.strip() else set()
    if include:
        df = df[df["tag"].isin(include)].copy()
        if df.empty:
            raise RuntimeError(f"No checkpoints match --include-tags {sorted(include)}")
    if args.max_checkpoints:
        df = df.sort_values(["seed", "epoch", "tag"]).groupby("seed", group_keys=False).head(args.max_checkpoints)
    df.to_csv(Path(args.output_dir) / "checkpoint_manifest.csv", index=False)
    return df


def load_checkpoint_model(path: str, device: torch.device) -> FeatureWrapper:
    state = torch.load(path, map_location=device)
    model = ResNet18().to(device)
    model.load_state_dict(state["net"] if "net" in state else state)
    model.eval()
    return FeatureWrapper(model).to(device).eval()


def margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.view(-1, 1), -1e9)
    return true - masked.max(1).values


def project_linf(x_adv: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)


def pgd_next(wrapper: FeatureWrapper, x: torch.Tensor, x0: torch.Tensor, y: torch.Tensor, eps: float, alpha: float):
    z = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(wrapper(z), y)
    grad = torch.autograd.grad(loss, z)[0]
    return project_linf(z + alpha * grad.sign(), x0, eps).detach()


def square_p_selection(p_init: float, it: int, n_queries: int) -> float:
    it = int(it / max(n_queries, 1) * 10000)
    if 10 < it <= 50:
        return p_init / 2
    if 50 < it <= 200:
        return p_init / 4
    if 200 < it <= 500:
        return p_init / 8
    if 500 < it <= 1000:
        return p_init / 16
    if 1000 < it <= 2000:
        return p_init / 32
    if 2000 < it <= 4000:
        return p_init / 64
    if 4000 < it <= 6000:
        return p_init / 128
    if 6000 < it <= 8000:
        return p_init / 256
    if 8000 < it:
        return p_init / 512
    return p_init


def square_init(wrapper: FeatureWrapper, x0: torch.Tensor, y: torch.Tensor, eps: float, gen: torch.Generator):
    _b, c, _h, w = x0.shape
    signs = torch.where(
        torch.rand((1, c, 1, w), generator=gen, device=x0.device) < 0.5,
        -torch.ones((1, c, 1, w), device=x0.device),
        torch.ones((1, c, 1, w), device=x0.device),
    )
    x_best = (x0 + eps * signs).clamp(0, 1)
    with torch.no_grad():
        best_margin = margin(wrapper(x_best), y)
    return x_best.detach(), best_margin.detach()


def square_next(
    wrapper: FeatureWrapper,
    x_best: torch.Tensor,
    x0: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    step: int,
    n_queries: int,
    p_init: float,
    gen: torch.Generator,
    best_margin: torch.Tensor,
):
    _b, c, h, w = x0.shape
    p = square_p_selection(p_init, step, n_queries)
    side = max(int(round(np.sqrt(p * h * w))), 1)
    side = min(side, h, w)
    top = int(torch.randint(0, h - side + 1, (1,), generator=gen, device=x0.device).item())
    left = int(torch.randint(0, w - side + 1, (1,), generator=gen, device=x0.device).item())
    delta = torch.zeros_like(x_best)
    signs = torch.where(
        torch.rand((1, c, 1, 1), generator=gen, device=x0.device) < 0.5,
        -torch.ones((1, c, 1, 1), device=x0.device),
        torch.ones((1, c, 1, 1), device=x0.device),
    )
    delta[:, :, top : top + side, left : left + side] = 2.0 * eps * signs
    cand = torch.min(torch.max(x_best + delta, x0 - eps), x0 + eps).clamp(0, 1)
    with torch.no_grad():
        cand_margin = margin(wrapper(cand), y)
    if bool(((cand_margin < best_margin) | (cand_margin <= 0)).item()):
        return cand.detach(), cand_margin.detach()
    return x_best.detach(), best_margin.detach()


def select_clean_correct(wrapper: FeatureWrapper, dataset, args, device: torch.device) -> list[tuple[int, int]]:
    selected = []
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        xb = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(wrapper(xb).argmax(1).item())
        if pred == int(y):
            selected.append((idx, int(y)))
        if len(selected) >= args.images:
            break
    return selected


def collect_clean_motion(wrapper: FeatureWrapper, dataset, selected, args, device: torch.device):
    rows = []
    arrays = {layer: [] for layer in LAYERS}
    gen = torch.Generator().manual_seed(args.seed + 717)
    for image_ord, (idx, label) in enumerate(selected[: args.clean_motion_images]):
        x_cpu, _ = dataset[idx]
        x = x_cpu.unsqueeze(0).to(device)
        with torch.no_grad():
            logits0, feats0 = wrapper.forward_features_nograd(x)
            if int(logits0.argmax(1).item()) != label:
                continue
        base = {k: v[0] for k, v in feats0.items()}
        variants = [
            ("crop", TF.resized_crop(x_cpu, 2, 2, 28, 28, [32, 32], antialias=True)),
            ("color", TF.adjust_contrast(TF.adjust_brightness(x_cpu, 1.2), 0.85).clamp(0, 1)),
            ("blur", TF.gaussian_blur(x_cpu, [5, 5], [0.8, 0.8])),
            ("noise", (x_cpu + torch.randn(x_cpu.shape, generator=gen) * 0.03).clamp(0, 1)),
        ]
        for motion, xv_cpu in variants:
            xv = xv_cpu.unsqueeze(0).to(device)
            with torch.no_grad():
                logits, feats = wrapper.forward_features_nograd(xv)
                if int(logits.argmax(1).item()) != label:
                    continue
            for layer, h in feats.items():
                v = h[0] - base[layer]
                if np.linalg.norm(v) <= 1e-12:
                    continue
                vector_idx = len(arrays[layer])
                arrays[layer].append(v.astype(np.float32))
                rows.append({"image_ord": image_ord, "dataset_idx": idx, "label": label, "motion": motion, "layer": layer, "vector_idx": vector_idx})
    return pd.DataFrame(rows), arrays


def shard_config(args, row, attack: str) -> dict:
    return {
        "seed": int(row.seed),
        "tag": str(row.tag),
        "epoch": int(row.epoch),
        "acc": round(float(row.acc), 6),
        "checkpoint": str(row.path),
        "attack": attack,
        "images": args.images,
        "eps": args.eps,
        "pgd_steps": args.pgd_steps,
        "square_steps": args.square_steps,
        "square_p_init": args.square_p_init,
        "square_record_every": args.square_record_every,
        "version": 1,
    }


def collect_attack_shard(args, ckpt_row, attack: str, dataset, device: torch.device):
    cfg = shard_config(args, ckpt_row, attack)
    sid = stable_hash(cfg)
    shard_dir = Path(args.output_dir) / "trajectory_shards" / f"seed{int(ckpt_row.seed)}" / str(ckpt_row.tag) / attack
    shard_dir.mkdir(parents=True, exist_ok=True)
    out_npz = shard_dir / f"states_{sid}.npz"
    out_meta = shard_dir / f"meta_{sid}.csv"
    out_cfg = shard_dir / f"config_{sid}.json"
    if out_npz.exists() and out_meta.exists() and out_cfg.exists() and not args.recollect:
        print(f"[SKIP SHARD] seed={ckpt_row.seed} tag={ckpt_row.tag} attack={attack}", flush=True)
        return
    wrapper = load_checkpoint_model(str(ckpt_row.path), device)
    selected = select_clean_correct(wrapper, dataset, args, device)
    if len(selected) < max(10, args.images // 5):
        print(f"[WARN] few clean-correct images seed={ckpt_row.seed} tag={ckpt_row.tag}: {len(selected)}", flush=True)
    arrays: dict[str, list[np.ndarray]] = {layer: [] for layer in LAYERS}
    meta_rows = []
    eps = args.eps / 255.0
    alpha = args.step_size / 255.0 if args.step_size > 0 else eps / max(args.pgd_steps, 1)
    max_steps = args.pgd_steps if attack == "pgd" else args.square_steps

    for image_ord, (idx, label) in enumerate(selected):
        x_cpu, _ = dataset[idx]
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        x = x0.clone()
        best_margin = None
        gen = torch.Generator(device=device).manual_seed(args.seed + int(ckpt_row.seed) * 100000 + idx)
        final_success = 0
        image_meta_start = len(meta_rows)
        for step in range(max_steps + 1):
            record = step == 0 or step == max_steps or attack == "pgd" or (step % max(args.square_record_every, 1) == 0)
            with torch.no_grad():
                logits = wrapper(x)
                pred = int(logits.argmax(1).item())
            now_success = int(pred != label)
            final_success = max(final_success, now_success)
            if record:
                with torch.no_grad():
                    logits, feats = wrapper.forward_features_nograd(x)
                    pred = int(logits.argmax(1).item())
                    m = float(margin(logits, y).item())
                    py = float(F.softmax(logits, 1)[0, label].item())
                vector_idx = len(arrays["logits"])
                for layer, h in feats.items():
                    arrays[layer].append(h[0].astype(np.float32))
                meta_rows.append(
                    {
                        "seed": int(ckpt_row.seed),
                        "tag": str(ckpt_row.tag),
                        "epoch": int(ckpt_row.epoch),
                        "checkpoint_acc": float(ckpt_row.acc),
                        "attack": attack,
                        "image_ord": image_ord,
                        "dataset_idx": idx,
                        "label": label,
                        "step": step,
                        "pred": pred,
                        "success_at_step": int(pred != label),
                        "margin": m,
                        "p_y": py,
                        "vector_idx": vector_idx,
                    }
                )
            if attack == "square" and now_success and step > 0 and args.square_stop_on_success:
                break
            if step >= max_steps:
                break
            if attack == "pgd":
                x = pgd_next(wrapper, x, x0, y, eps, alpha)
            elif attack == "square":
                if step == 0:
                    x, best_margin = square_init(wrapper, x0, y, eps, gen)
                else:
                    x, best_margin = square_next(wrapper, x, x0, y, eps, step - 1, max_steps, args.square_p_init, gen, best_margin)
            else:
                raise ValueError(attack)
        for r in meta_rows[image_meta_start:]:
            r["final_success"] = int(final_success)
    packed = {f"states__{layer}": np.stack(vals).astype(np.float32) for layer, vals in arrays.items() if vals}
    np.savez_compressed(out_npz, **packed)
    pd.DataFrame(meta_rows).to_csv(out_meta, index=False)
    out_cfg.write_text(json.dumps(cfg, indent=2) + "\n")
    wrapper.close()
    del wrapper
    torch.cuda.empty_cache()
    print(f"[SAVED SHARD] {out_npz}", flush=True)


def collect_trajectories(args, device: torch.device):
    out = Path(args.output_dir)
    ckpts = checkpoint_manifest(args)
    dataset = build_eval_dataset(args.dataset_root)
    attacks = parse_csv(args.attacks)
    for row in ckpts.itertuples(index=False):
        for attack in attacks:
            collect_attack_shard(args, row, attack, dataset, device)
        if args.collect_clean_motion:
            wrapper = load_checkpoint_model(str(row.path), device)
            selected = select_clean_correct(wrapper, dataset, args, device)
            clean_rows, clean_arrays = collect_clean_motion(wrapper, dataset, selected, args, device)
            clean_dir = out / "clean_motion" / f"seed{int(row.seed)}" / str(row.tag)
            clean_dir.mkdir(parents=True, exist_ok=True)
            clean_rows.to_csv(clean_dir / "clean_motion_meta.csv", index=False)
            np.savez_compressed(clean_dir / "clean_motion_vectors.npz", **{f"vectors__{k}": np.stack(v).astype(np.float32) for k, v in clean_arrays.items() if v})
            wrapper.close()
            del wrapper
            torch.cuda.empty_cache()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def pca_basis(x: np.ndarray, max_k: int):
    x = normalize_rows(x.astype(np.float32))
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s * s
    ratio = var / np.clip(var.sum(), 1e-12, None)
    return mean.astype(np.float32), vt[: min(max_k, len(vt))].astype(np.float32), ratio.astype(np.float32)


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    x = normalize_rows(x.astype(np.float32))
    xc = x - mean
    kk = min(k, basis.shape[0])
    coeff = xc @ basis[:kk].T
    return np.sum(coeff * coeff, axis=1) / np.clip(np.sum(xc * xc, axis=1), 1e-12, None)


def dim_stats(ratio: np.ndarray):
    csum = np.cumsum(ratio)
    ent = -float(np.sum(ratio[ratio > 0] * np.log(ratio[ratio > 0])))
    return {
        "pc1_var": float(ratio[0]) if len(ratio) else np.nan,
        "pc5_cum_var": float(csum[min(4, len(csum) - 1)]) if len(csum) else np.nan,
        "pc10_cum_var": float(csum[min(9, len(csum) - 1)]) if len(csum) else np.nan,
        "dim80": int(np.searchsorted(csum, 0.8) + 1) if len(csum) else np.nan,
        "dim90": int(np.searchsorted(csum, 0.9) + 1) if len(csum) else np.nan,
        "effective_rank": float(np.exp(ent)) if len(csum) else np.nan,
    }


def transport_vectors(meta: pd.DataFrame, npz, layer: str):
    rows = []
    vecs = []
    key = f"states__{layer}"
    if key not in npz.files:
        return pd.DataFrame(), np.empty((0, 1), dtype=np.float32)
    arr = npz[key]
    for (_idx, _attack), g in meta.sort_values("step").groupby(["dataset_idx", "attack"]):
        states = arr[g.vector_idx.to_numpy(int)]
        if len(states) < 2:
            continue
        v = normalize_rows((states[1:] - states[:-1]).astype(np.float32))
        r = g.iloc[:-1].copy()
        r["segment_end_step"] = g.step.to_numpy()[1:]
        rows.append(r)
        vecs.append(v)
    if not vecs:
        return pd.DataFrame(), np.empty((0, 1), dtype=np.float32)
    return pd.concat(rows, ignore_index=True), np.concatenate(vecs, axis=0)


def analyze(args):
    out = Path(args.output_dir)
    analysis = out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    ks = parse_csv(args.ks, int)
    rng = np.random.default_rng(args.seed)
    dim_rows = []
    metric_rows = []
    signature_rows = []
    basis_cache = {}

    for meta_path in sorted((out / "trajectory_shards").glob("seed*/*/*/meta_*.csv")):
        npz_path = meta_path.with_name(meta_path.name.replace("meta_", "states_").replace(".csv", ".npz"))
        if not npz_path.exists():
            continue
        meta = pd.read_csv(meta_path)
        if meta.empty:
            continue
        npz = np.load(npz_path, allow_pickle=False)
        ck = meta.iloc[0]
        for layer in LAYERS:
            rows, x = transport_vectors(meta, npz, layer)
            if len(rows) < 20:
                continue
            success = rows.final_success.to_numpy(int) == 1
            if success.sum() < 8:
                continue
            train_ids = sorted(rows.dataset_idx.unique())
            rng.shuffle(train_ids)
            train_ids = set(train_ids[: max(1, int(0.6 * len(train_ids)))])
            train = rows.dataset_idx.isin(train_ids).to_numpy() & success
            test_success = (~rows.dataset_idx.isin(train_ids).to_numpy()) & success
            test_failed = (~rows.dataset_idx.isin(train_ids).to_numpy()) & (~success)
            if train.sum() < 8 or test_success.sum() < 4:
                continue
            mean, basis, ratio = pca_basis(x[train], max(ks))
            basis_cache[(int(ck.seed), str(ck.tag), str(ck.attack), layer)] = basis
            stats = dim_stats(ratio)
            dim_rows.append(
                {
                    "seed": int(ck.seed),
                    "tag": str(ck.tag),
                    "epoch": int(ck.epoch),
                    "checkpoint_acc": float(ck.checkpoint_acc),
                    "attack": str(ck.attack),
                    "layer": layer,
                    "n_success_segments": int(success.sum()),
                    "n_total_segments": int(len(rows)),
                    **stats,
                }
            )
            for k in ks:
                es = projection_energy(x[test_success], mean, basis, k)
                rand = normalize_rows(rng.normal(size=(max(len(es), 1000), x.shape[1])).astype(np.float32))
                er = projection_energy(rand, mean, basis, k)
                metric_rows.append(
                    {
                        "seed": int(ck.seed),
                        "tag": str(ck.tag),
                        "epoch": int(ck.epoch),
                        "checkpoint_acc": float(ck.checkpoint_acc),
                        "attack": str(ck.attack),
                        "layer": layer,
                        "comparison": "success_vs_random",
                        "k": k,
                        "auroc": float(roc_auc_score(np.r_[np.ones(len(es)), np.zeros(len(er))], np.r_[es, er])),
                        "success_energy": float(np.mean(es)),
                        "negative_energy": float(np.mean(er)),
                        "n_success": int(len(es)),
                        "n_negative": int(len(er)),
                    }
                )
                if test_failed.sum() >= 4:
                    ef = projection_energy(x[test_failed], mean, basis, k)
                    metric_rows.append(
                        {
                            "seed": int(ck.seed),
                            "tag": str(ck.tag),
                            "epoch": int(ck.epoch),
                            "checkpoint_acc": float(ck.checkpoint_acc),
                            "attack": str(ck.attack),
                            "layer": layer,
                            "comparison": "success_vs_failed",
                            "k": k,
                            "auroc": float(roc_auc_score(np.r_[np.ones(len(es)), np.zeros(len(ef))], np.r_[es, ef])),
                            "success_energy": float(np.mean(es)),
                            "negative_energy": float(np.mean(ef)),
                            "n_success": int(len(es)),
                            "n_negative": int(len(ef)),
                        }
                    )
            coeff = (normalize_rows(x[success]) - mean) @ basis[:5].T
            en = np.mean(coeff * coeff, axis=0)
            frac = en / np.clip(en.sum(), 1e-12, None)
            signature_rows.append(
                {
                    "seed": int(ck.seed),
                    "tag": str(ck.tag),
                    "epoch": int(ck.epoch),
                    "checkpoint_acc": float(ck.checkpoint_acc),
                    "attack": str(ck.attack),
                    "layer": layer,
                    **{f"pc{i+1}_frac": float(frac[i]) for i in range(len(frac))},
                }
            )

    dim = pd.DataFrame(dim_rows)
    metrics = pd.DataFrame(metric_rows)
    sig = pd.DataFrame(signature_rows)
    dim.to_csv(analysis / "training_dynamics_dimensionality.csv", index=False)
    metrics.to_csv(analysis / "training_dynamics_projection_metrics.csv", index=False)
    sig.to_csv(analysis / "training_dynamics_transport_signatures.csv", index=False)
    sim_rows = []
    if not sig.empty:
        pc_cols = [f"pc{i}_frac" for i in range(1, 6)]
        for (seed, tag, layer), g in sig.groupby(["seed", "tag", "layer"]):
            attacks = sorted(g.attack.unique())
            for i, a in enumerate(attacks):
                for b in attacks[i + 1 :]:
                    va = g[g.attack == a][pc_cols].iloc[0].to_numpy(float)
                    vb = g[g.attack == b][pc_cols].iloc[0].to_numpy(float)
                    cos = float(np.dot(va, vb) / np.clip(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12, None))
                    sim_rows.append({"seed": seed, "tag": tag, "layer": layer, "attack_a": a, "attack_b": b, "signature_cosine": cos})
    pd.DataFrame(sim_rows).to_csv(analysis / "training_dynamics_optimizer_similarity.csv", index=False)
    make_plots(dim, metrics, pd.DataFrame(sim_rows), analysis)
    print(f"[ANALYSIS] wrote {analysis}", flush=True)


def make_plots(dim: pd.DataFrame, metrics: pd.DataFrame, sim: pd.DataFrame, analysis: Path):
    if dim.empty and metrics.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    if not dim.empty:
        sub = dim[dim.attack == "pgd"]
        for layer, g in sub.groupby("layer"):
            y = g.groupby("checkpoint_acc").dim80.mean().sort_index()
            axes[0].plot(y.index, y.values, marker="o", label=layer)
        axes[0].set_title("PGD dim80 over training")
        axes[0].set_xlabel("checkpoint accuracy")
        axes[0].set_ylabel("dim80")
    if not metrics.empty:
        sub = metrics[(metrics.attack == "pgd") & (metrics.comparison == "success_vs_random") & (metrics.k == 20)]
        for layer, g in sub.groupby("layer"):
            y = g.groupby("checkpoint_acc").auroc.mean().sort_index()
            axes[1].plot(y.index, y.values, marker="o", label=layer)
        axes[1].set_title("PGD success-vs-random AUROC")
        axes[1].set_xlabel("checkpoint accuracy")
        axes[1].set_ylim(0.45, 1.02)
    if not sim.empty:
        sub = sim[((sim.attack_a == "pgd") & (sim.attack_b == "square")) | ((sim.attack_a == "square") & (sim.attack_b == "pgd"))]
        for layer, g in sub.groupby("layer"):
            # join checkpoint accuracy through tag from metrics/dim if available.
            acc_map = dim.drop_duplicates(["seed", "tag"])[["seed", "tag", "checkpoint_acc"]]
            gg = g.merge(acc_map, on=["seed", "tag"], how="left")
            y = gg.groupby("checkpoint_acc").signature_cosine.mean().sort_index()
            axes[2].plot(y.index, y.values, marker="o", label=layer)
        axes[2].set_title("PGD/Square signature similarity")
        axes[2].set_xlabel("checkpoint accuracy")
        axes[2].set_ylim(0.0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=6)
    fig.savefig(analysis / "training_dynamics_transport_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["train", "collect", "analyze", "all"], default="all")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_training_dynamics_transport")
    p.add_argument("--checkpoint-dir", default="checkpoints/cifar10_resnet18_training_dynamics")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model-seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--accuracy-targets", default="0.15,0.25,0.40,0.55,0.70,0.82,0.90")
    p.add_argument("--save-epoch-every", type=int, default=10)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--eval-batches", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--recollect", action="store_true")
    p.add_argument("--attacks", default="pgd,square")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--step-size", type=float, default=2.0)
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--square-steps", type=int, default=500)
    p.add_argument("--square-p-init", type=float, default=0.8)
    p.add_argument("--square-record-every", type=int, default=25)
    p.add_argument("--square-stop-on-success", action="store_true")
    p.add_argument("--collect-clean-motion", action="store_true")
    p.add_argument("--clean-motion-images", type=int, default=100)
    p.add_argument("--ks", default="5,10,20,50")
    p.add_argument("--max-checkpoints", type=int, default=0)
    p.add_argument("--include-tags", default="")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage in {"train", "all"}:
        train_checkpoints(args, device)
    if args.stage in {"collect", "all"}:
        collect_trajectories(args, device)
    if args.stage in {"analyze", "all"}:
        analyze(args)
    (Path(args.output_dir) / "metadata.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
