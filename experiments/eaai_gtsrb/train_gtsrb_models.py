#!/usr/bin/env python3
"""Train reproducible GTSRB checkpoints for the EAAI application case study."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from gtsrb_common import (
    SUPPORTED_MODELS,
    build_model,
    expected_calibration_error,
    gtsrb_dataset,
    set_seed,
    sha256,
    stratified_train_val_indices,
    write_json,
)


def evaluate(model, loader, device):
    model.eval()
    losses: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            losses.append(F.cross_entropy(logits, y, reduction="none").cpu().numpy())
            probabilities.append(logits.softmax(1).cpu().numpy())
            labels.append(y.cpu().numpy())
    probs = np.concatenate(probabilities)
    target = np.concatenate(labels)
    return {
        "loss": float(np.concatenate(losses).mean()),
        "accuracy": float((probs.argmax(1) == target).mean()),
        "nll": float(-np.log(np.clip(probs[np.arange(len(target)), target], 1e-12, 1)).mean()),
        "ece15": expected_calibration_error(probs, target, bins=15),
        "n": int(len(target)),
    }


def append_history(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_one(args, architecture: str, device: torch.device) -> None:
    model_dir = Path(args.output_dir) / architecture
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best.pt"
    last_path = model_dir / "last.pt"
    done_path = model_dir / "complete.json"
    if done_path.exists() and best_path.exists() and not args.overwrite:
        print(f"[{architecture}] complete; skipping", flush=True)
        return

    set_seed(args.seed)
    raw_train = gtsrb_dataset(
        args.data_dir, "train", args.image_size, training=False, download=args.download
    )
    train_idx, val_idx = stratified_train_val_indices(
        raw_train, args.val_fraction, args.split_seed
    )
    train_data = gtsrb_dataset(
        args.data_dir, "train", args.image_size, training=True, download=False
    )
    eval_train = gtsrb_dataset(
        args.data_dir, "train", args.image_size, training=False, download=False
    )
    test_data = gtsrb_dataset(
        args.data_dir, "test", args.image_size, training=False, download=args.download
    )
    batch_size = args.batch_size if architecture == "resnet18" else args.convnext_batch_size
    loader_args = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(train_data, train_idx),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **loader_args,
    )
    val_loader = DataLoader(
        Subset(eval_train, val_idx), batch_size=batch_size * 2, shuffle=False, **loader_args
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size * 2, shuffle=False, **loader_args
    )

    model = build_model(architecture, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    best_accuracy = -1.0
    if last_path.exists() and not args.overwrite:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["epoch"]) + 1
        best_accuracy = float(state["best_accuracy"])
        print(f"[{architecture}] resuming at epoch {start_epoch}", flush=True)

    history_path = model_dir / "history.csv"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        start = time.perf_counter()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(y)
            seen += len(y)
        scheduler.step()
        val = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_nll": val["nll"],
            "val_ece15": val["ece15"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - start,
        }
        append_history(history_path, row)
        payload = {
            "architecture": architecture,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "best_accuracy": max(best_accuracy, val["accuracy"]),
            "image_size": args.image_size,
            "num_classes": 43,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "val_fraction": args.val_fraction,
            "train_indices": train_idx,
            "val_indices": val_idx,
        }
        torch.save(payload, last_path)
        if val["accuracy"] > best_accuracy:
            best_accuracy = val["accuracy"]
            torch.save(payload, best_path)
        print(
            f"[{architecture}] epoch={epoch}/{args.epochs} "
            f"train_loss={row['train_loss']:.4f} val_acc={val['accuracy']:.4f} "
            f"time={row['seconds']:.1f}s",
            flush=True,
        )

    best_model_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_model_state["model_state"])
    test = evaluate(model, test_loader, device)
    write_json(
        done_path,
        {
            "architecture": architecture,
            "best_epoch": int(best_model_state["epoch"]),
            "validation_best_accuracy": float(best_accuracy),
            "test": test,
            "checkpoint": str(best_path.resolve()),
            "checkpoint_sha256": sha256(best_path),
            "image_size": args.image_size,
            "train_images": len(train_idx),
            "validation_images": len(val_idx),
            "test_images": len(test_data),
        },
    )
    print(f"[{architecture}] test={test}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/gtsrb")
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/eaai_gtsrb/checkpoints",
    )
    parser.add_argument("--models", default=",".join(SUPPORTED_MODELS))
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--convnext-batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--split-seed", type=int, default=1307)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_json(
        Path(args.output_dir) / "training_protocol.json",
        {
            "args": vars(args),
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "models": [x.strip() for x in args.models.split(",") if x.strip()],
            "estimated_batches_per_epoch": {
                "resnet18": math.ceil(35288 / args.batch_size),
                "convnext_tiny": math.ceil(35288 / args.convnext_batch_size),
            },
        },
    )
    for architecture in [x.strip() for x in args.models.split(",") if x.strip()]:
        train_one(args, architecture, device)


if __name__ == "__main__":
    main()
