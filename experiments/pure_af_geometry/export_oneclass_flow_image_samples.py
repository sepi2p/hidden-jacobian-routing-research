#!/usr/bin/env python3
"""Export visual samples for one-class pure and adversarial flow diagnostics."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, utils as tv_utils

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import (  # noqa: E402
    eval_population,
    make_children,
    mutate,
)
from experiments.pure_af_geometry.analyze_cifar_attack_axis_projection import (  # noqa: E402
    pgd_trajectory,
    square_trajectory,
)
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import (  # noqa: E402
    eval_all,
    load_model,
)


CIFAR10_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def image_stats(wrapper, x: torch.Tensor, y: torch.Tensor) -> dict:
    with torch.no_grad():
        logits = wrapper(x)
        probs = F.softmax(logits, dim=1)
    pred = int(logits.argmax(1).item())
    target = int(y.item())
    masked = logits.clone()
    masked[:, target] = -1e9
    return {
        "pred": pred,
        "pred_name": CIFAR10_NAMES[pred],
        "target_prob": float(probs[0, target].item()),
        "target_logit": float(logits[0, target].item()),
        "margin": float((logits[0, target] - masked.max(1).values[0]).item()),
    }


def run_ga_sample(wrapper, target: int, args: argparse.Namespace, device):
    gen = torch.Generator(device=device).manual_seed(args.seed + target * 1000)
    pop = torch.rand((args.population, 3, 32, 32), generator=gen, device=device)
    y = torch.tensor([target], device=device)
    noise_start = None
    pure_end = None
    pure_generation = -1
    pure_stats = None
    for generation in range(args.generations + 1):
        stats = eval_population(wrapper, pop, target, args.eval_batch_size)
        order = torch.argsort(stats["fitness"], descending=True)
        pop = pop[order]
        for k in stats:
            stats[k] = stats[k][order]
        cur = pop[:1].detach().clone()
        if generation == 0:
            noise_start = cur
        if float(stats["prob"][0].item()) >= args.prob_threshold and int(stats["pred"][0].item()) == target:
            pure_end = cur
            pure_generation = generation
            pure_stats = {
                "prob": float(stats["prob"][0].item()),
                "pred": int(stats["pred"][0].item()),
                "target_logit": float(stats["fitness"][0].item()),
            }
            break
        if generation == args.generations:
            pure_end = cur
            pure_generation = generation
            pure_stats = {
                "prob": float(stats["prob"][0].item()),
                "pred": int(stats["pred"][0].item()),
                "target_logit": float(stats["fitness"][0].item()),
            }
            break
        parents = pop[: args.parents]
        elite = pop[: args.elite]
        children = make_children(parents, args.population - args.elite, args.crossover, gen)
        children = mutate(children, args, gen)
        pop = torch.cat([elite, children], dim=0)
    return noise_start, pure_end, pure_generation, pure_stats, y


def select_clean_correct(dataset, wrapper, target: int, device, max_scan: int):
    selected = []
    y = torch.tensor([target], device=device)
    for idx in range(len(dataset)):
        x_cpu, label = dataset[idx]
        if int(label) != target:
            continue
        x = x_cpu.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(wrapper(x).argmax(1).item())
        if pred == target:
            selected.append((idx, x, y))
            if len(selected) >= max_scan:
                break
    return selected


def save_single(path: Path, x: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tv_utils.save_image(x.detach().cpu().clamp(0, 1), path)


def save_gallery(path: Path, items: list[tuple[str, torch.Tensor]]) -> None:
    fig, axes = plt.subplots(1, len(items), figsize=(2.15 * len(items), 2.45), squeeze=False)
    for ax, (title, img) in zip(axes.ravel(), items):
        arr = img.detach().cpu()[0].permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(arr, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--generations", type=int, default=160)
    p.add_argument("--population", type=int, default=80)
    p.add_argument("--parents", type=int, default=20)
    p.add_argument("--elite", type=int, default=5)
    p.add_argument("--crossover", type=float, default=0.5)
    p.add_argument("--pixel-sigma", type=float, default=0.08)
    p.add_argument("--pixel-rate", type=float, default=0.08)
    p.add_argument("--block-rate", type=float, default=0.35)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--prob-threshold", type=float, default=0.999)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--pgd-steps", type=int, default=20)
    p.add_argument("--square-steps", type=int, default=120)
    p.add_argument("--square-min-size", type=int, default=2)
    p.add_argument("--square-search", type=int, default=50)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = load_model(args.model, device)
    target = int(args.target_class)
    target_name = CIFAR10_NAMES[target]
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())

    rows = []
    noise_start, pure_end, pure_generation, pure_raw, y = run_ga_sample(wrapper, target, args, device)
    save_single(out_dir / "ga_noise_start.png", noise_start)
    save_single(out_dir / "ga_pure_end.png", pure_end)
    rows.append({"sample": "ga_noise_start", "dataset_idx": -1, "generation": 0, **image_stats(wrapper, noise_start, y)})
    rows.append({"sample": "ga_pure_end", "dataset_idx": -1, "generation": pure_generation, **image_stats(wrapper, pure_end, y)})
    print(f"[GA] target={target_name} pure_generation={pure_generation} prob={pure_raw['prob']:.6f} pred={pure_raw['pred']}", flush=True)

    selected = select_clean_correct(dataset, wrapper, target, device, args.square_search)
    if not selected:
        raise RuntimeError(f"No clean-correct images found for class {target}")

    eps = args.eps / 255.0
    pgd_step = eps / max(args.pgd_steps, 1)
    pgd_idx, pgd_start, y = selected[0]
    pgd_states = pgd_trajectory(wrapper, pgd_start, y, eps, args.pgd_steps, pgd_step)
    pgd_end = pgd_states[-1]
    save_single(out_dir / "pgd_start_clean.png", pgd_start)
    save_single(out_dir / "pgd_end_adv.png", pgd_end)
    rows.append({"sample": "pgd_start_clean", "dataset_idx": int(pgd_idx), "generation": 0, **image_stats(wrapper, pgd_start, y)})
    rows.append({"sample": "pgd_end_adv", "dataset_idx": int(pgd_idx), "generation": args.pgd_steps, **image_stats(wrapper, pgd_end, y)})

    square_pack = None
    for ord_i, (idx, x, y) in enumerate(selected):
        states = square_trajectory(wrapper, x, y, eps, args.square_steps, args.seed + ord_i * 997, args.square_min_size)
        final = states[-1]
        final_eval = eval_all({args.model: wrapper}, final, y)[args.model]
        if int(final_eval["success"]):
            square_pack = (idx, x, final, y, final_eval)
            break
    if square_pack is None:
        idx, x, y = selected[0]
        states = square_trajectory(wrapper, x, y, eps, args.square_steps, args.seed, args.square_min_size)
        final = states[-1]
        final_eval = eval_all({args.model: wrapper}, final, y)[args.model]
        square_pack = (idx, x, final, y, final_eval)
    square_idx, square_start, square_end, y, square_eval = square_pack
    save_single(out_dir / "square_start_clean.png", square_start)
    save_single(out_dir / "square_end_adv.png", square_end)
    rows.append({"sample": "square_start_clean", "dataset_idx": int(square_idx), "generation": 0, **image_stats(wrapper, square_start, y)})
    rows.append({"sample": "square_end_adv", "dataset_idx": int(square_idx), "generation": args.square_steps, **image_stats(wrapper, square_end, y)})

    save_gallery(
        out_dir / "flow_image_samples_gallery.png",
        [
            ("GA noise", noise_start),
            ("GA pure", pure_end),
            ("PGD clean", pgd_start),
            ("PGD adv", pgd_end),
            ("Square clean", square_start),
            ("Square adv", square_end),
        ],
    )

    with open(out_dir / "flow_image_samples_metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
