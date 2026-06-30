#!/usr/bin/env python3
"""Generate classifier-pure ImageNet images with a simple genetic algorithm.

This script is part of the isolated pure_af_geometry experiment. It writes only
under the requested output directory and records a manifest for downstream
AF/VF analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, utils as tv_utils

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.load_models import load_imagenet_model


def parse_ints(text: str) -> list[int]:
    vals: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = [int(x) for x in chunk.split("-", 1)]
            vals.extend(range(start, end + 1))
        else:
            vals.append(int(chunk))
    return vals


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_name_offset(*parts: str) -> int:
    total = 0
    for part in parts:
        for ch in part:
            total = (total * 131 + ord(ch)) % 1000003
    return total


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    return dh + dw


def margin_and_prob(logits: torch.Tensor, target: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_logits = logits[:, target]
    masked = logits.clone()
    masked[:, target] = -torch.inf
    other_logits = masked.max(dim=1).values
    margin = target_logits - other_logits
    prob = torch.softmax(logits, dim=1)[:, target]
    pred = logits.argmax(dim=1)
    return margin, prob, pred


def topk_dict(logits: torch.Tensor, k: int = 5) -> list[dict[str, float | int]]:
    probs = torch.softmax(logits, dim=1)
    values, indices = torch.topk(probs, k=min(k, probs.shape[1]), dim=1)
    return [
        {"class": int(cls), "prob": float(prob)}
        for cls, prob in zip(indices[0].detach().cpu(), values[0].detach().cpu())
    ]


def load_real_start(dataset, target: int, sample_offset: int, device: torch.device) -> torch.Tensor:
    matches = [idx for idx, (_path, label) in enumerate(dataset.samples) if int(label) == target]
    if not matches:
        raise ValueError(f"No ImageNet samples found for class {target}")
    image, _label = dataset[matches[sample_offset % len(matches)]]
    return image.unsqueeze(0).to(device)


def mutate(pop: torch.Tensor, *, pixel_sigma: float, pixel_rate: float, block_rate: float, block_size: int) -> torch.Tensor:
    out = pop.clone()
    if pixel_rate > 0:
        mask = torch.rand_like(out[:, :1]) < pixel_rate
        noise = torch.randn_like(out) * pixel_sigma
        out = torch.where(mask, out + noise, out)

    if block_rate > 0 and block_size > 0:
        n, _c, h, w = out.shape
        for i in range(n):
            if random.random() >= block_rate:
                continue
            y = random.randint(0, max(0, h - block_size))
            x = random.randint(0, max(0, w - block_size))
            patch = torch.rand((3, block_size, block_size), device=out.device)
            out[i, :, y : y + block_size, x : x + block_size] = patch
    return out.clamp(0.0, 1.0)


def make_children(parents: torch.Tensor, count: int, crossover: str) -> torch.Tensor:
    if count <= 0:
        return parents[:0]
    n = parents.shape[0]
    a = parents[torch.randint(0, n, (count,), device=parents.device)]
    b = parents[torch.randint(0, n, (count,), device=parents.device)]
    if crossover == "uniform":
        mask = torch.rand((count, 1, *parents.shape[2:]), device=parents.device) < 0.5
        return torch.where(mask, a, b)
    if crossover == "average":
        lam = torch.rand((count, 1, 1, 1), device=parents.device)
        return lam * a + (1.0 - lam) * b
    raise ValueError(f"Unsupported crossover: {crossover}")


def evaluate_population(
    model,
    pop: torch.Tensor,
    target: int,
    start: torch.Tensor | None,
    lambda_tv: float,
    lambda_l2: float,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    margins = []
    probs = []
    preds = []
    fitnesses = []
    with torch.no_grad():
        for batch in pop.split(batch_size):
            logits = model(batch)
            margin, prob, pred = margin_and_prob(logits, target)
            fitness = margin.clone()
            if lambda_tv:
                fitness = fitness - lambda_tv * total_variation(batch)
            if lambda_l2 and start is not None:
                diff = (batch - start).flatten(start_dim=1)
                fitness = fitness - lambda_l2 * torch.norm(diff, dim=1)
            margins.append(margin)
            probs.append(prob)
            preds.append(pred)
            fitnesses.append(fitness)
    return {
        "margin": torch.cat(margins),
        "prob": torch.cat(probs),
        "pred": torch.cat(preds),
        "fitness": torch.cat(fitnesses),
    }


def run_one(args, model, dataset, target: int, init_mode: str, reg_name: str, sample_id: int, device: torch.device) -> dict[str, object]:
    run_name = f"class{target:04d}_{init_mode}_{reg_name}_sample{sample_id:02d}"
    run_dir = Path(args.output_dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    start = None
    if init_mode == "random":
        pop = torch.rand((args.population, 3, args.image_size, args.image_size), device=device)
    elif init_mode == "real":
        start = load_real_start(dataset, target, sample_id, device)
        pop = (start + torch.randn((args.population, 3, args.image_size, args.image_size), device=device) * args.real_init_sigma).clamp(0.0, 1.0)
        pop[0:1] = start
    else:
        raise ValueError(f"Unsupported init mode: {init_mode}")

    lambda_tv = args.lambda_tv if reg_name == "weak_tv" else 0.0
    lambda_l2 = args.lambda_l2 if reg_name == "weak_tv" else 0.0

    history_path = run_dir / "history.csv"
    best = None
    generations_to_success = None
    fieldnames = ["generation", "best_fitness", "best_margin", "best_prob", "best_pred", "mean_fitness"]
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for gen in range(args.generations + 1):
            stats = evaluate_population(model, pop, target, start, lambda_tv, lambda_l2, args.eval_batch_size)
            order = torch.argsort(stats["fitness"], descending=True)
            pop = pop[order]
            for key in stats:
                stats[key] = stats[key][order]

            cur = {
                "image": pop[0:1].detach().clone(),
                "fitness": float(stats["fitness"][0].item()),
                "margin": float(stats["margin"][0].item()),
                "prob": float(stats["prob"][0].item()),
                "pred": int(stats["pred"][0].item()),
                "generation": gen,
            }
            if best is None or cur["fitness"] > best["fitness"]:
                best = cur
            if args.success_mode == "prob":
                success_now = cur["prob"] >= args.prob_threshold
            elif args.success_mode == "margin":
                success_now = cur["margin"] >= args.margin_threshold
            else:
                success_now = cur["margin"] >= args.margin_threshold or cur["prob"] >= args.prob_threshold
            if generations_to_success is None and success_now:
                generations_to_success = gen

            writer.writerow(
                {
                    "generation": gen,
                    "best_fitness": cur["fitness"],
                    "best_margin": cur["margin"],
                    "best_prob": cur["prob"],
                    "best_pred": cur["pred"],
                    "mean_fitness": float(stats["fitness"].mean().item()),
                }
            )

            if gen % args.save_every == 0 or gen == args.generations:
                tv_utils.save_image(cur["image"].cpu(), run_dir / f"best_gen{gen:04d}.png")

            if args.stop_on_success and generations_to_success is not None:
                break
            if gen == args.generations:
                break

            parents = pop[: args.parents]
            elite = pop[: args.elite]
            children = make_children(parents, args.population - args.elite, args.crossover)
            children = mutate(
                children,
                pixel_sigma=args.pixel_sigma,
                pixel_rate=args.pixel_rate,
                block_rate=args.block_rate,
                block_size=args.block_size,
            )
            pop = torch.cat([elite, children], dim=0)

    assert best is not None
    final_path = run_dir / "final_best.png"
    tv_utils.save_image(best["image"].cpu(), final_path)
    with torch.no_grad():
        logits = model(best["image"])
    metadata = {
        "run_name": run_name,
        "target_class": target,
        "init_mode": init_mode,
        "regularization": reg_name,
        "sample_id": sample_id,
        "final_image": str(final_path),
        "history_csv": str(history_path),
        "final_margin": best["margin"],
        "final_prob": best["prob"],
        "final_pred": best["pred"],
        "final_fitness": best["fitness"],
        "generations_to_success": generations_to_success,
        "completed_generations": best["generation"],
        "top5": topk_dict(logits),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="resnet18")
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--classes", default="0-9", help="comma list and/or ranges, e.g. 0,5,10-19")
    parser.add_argument("--images-per-class", type=int, default=5)
    parser.add_argument("--init-modes", default="random,real")
    parser.add_argument("--regularizations", default="none,weak_tv")
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/pure_images_resnet18")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--parents", type=int, default=16)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--margin-threshold", type=float, default=20.0)
    parser.add_argument("--prob-threshold", type=float, default=0.9999)
    parser.add_argument("--success-mode", choices=["prob", "margin", "margin_or_prob"], default="margin_or_prob")
    parser.add_argument("--pixel-sigma", type=float, default=0.08)
    parser.add_argument("--pixel-rate", type=float, default=0.03)
    parser.add_argument("--block-rate", type=float, default=0.4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--real-init-sigma", type=float, default=0.08)
    parser.add_argument("--lambda-tv", type=float, default=0.1)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--crossover", choices=["uniform", "average"], default="uniform")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.parents < args.elite:
        raise ValueError("--parents must be >= --elite")
    if args.population <= args.elite:
        raise ValueError("--population must be > --elite")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    model = load_imagenet_model(args.target_model).to(device).eval()

    manifest_path = output_dir / "manifest.csv"
    rows = []
    started = time.time()
    for target in parse_ints(args.classes):
        for init_mode in [x.strip() for x in args.init_modes.split(",") if x.strip()]:
            for reg_name in [x.strip() for x in args.regularizations.split(",") if x.strip()]:
                for sample_id in range(args.images_per_class):
                    run_seed = args.seed + target * 10000 + sample_id * 100 + stable_name_offset(init_mode, reg_name)
                    set_seed(run_seed)
                    print(f"[START] class={target} init={init_mode} reg={reg_name} sample={sample_id}", flush=True)
                    row = run_one(args, model, dataset, target, init_mode, reg_name, sample_id, device)
                    rows.append(row)
                    print(
                        f"[DONE] {row['run_name']} margin={row['final_margin']:.4f} "
                        f"prob={row['final_prob']:.6f} pred={row['final_pred']}",
                        flush=True,
                    )
                    with manifest_path.open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(rows)

    metadata = {
        "experiment": "pure_af_geometry",
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "args": vars(args),
        "elapsed_sec": time.time() - started,
        "manifest_csv": str(manifest_path),
        "rows": len(rows),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[DONE] manifest={manifest_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
