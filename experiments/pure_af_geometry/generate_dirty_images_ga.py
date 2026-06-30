#!/usr/bin/env python3
"""Generate classifier-dirty images near a target decision boundary.

A dirty image for class c is still predicted as c, but its target logit margin
logit_c - max_{j != c} logit_j is as small positive as possible.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms, utils as tv_utils

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.load_models import load_imagenet_model


def parse_ints(text: str) -> list[int]:
    vals = []
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            a, b = [int(x) for x in chunk.split('-', 1)]
            vals.extend(range(a, b + 1))
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


def logits_info(logits: torch.Tensor, target: int):
    target_logits = logits[:, target]
    masked = logits.clone()
    masked[:, target] = -torch.inf
    other_logits, other_idx = masked.max(dim=1)
    margin = target_logits - other_logits
    prob = torch.softmax(logits, dim=1)[:, target]
    pred = logits.argmax(dim=1)
    return margin, prob, pred, other_idx


def load_real_start(dataset, target: int, sample_offset: int, device: torch.device) -> torch.Tensor:
    matches = [idx for idx, (_path, label) in enumerate(dataset.samples) if int(label) == target]
    if not matches:
        raise ValueError(f'No ImageNet samples found for class {target}')
    image, _label = dataset[matches[sample_offset % len(matches)]]
    return image.unsqueeze(0).to(device)


def mutate(pop: torch.Tensor, pixel_sigma: float, pixel_rate: float, block_rate: float, block_size: int) -> torch.Tensor:
    out = pop.clone()
    if pixel_rate > 0:
        mask = torch.rand_like(out[:, :1]) < pixel_rate
        out = torch.where(mask, out + torch.randn_like(out) * pixel_sigma, out)
    if block_rate > 0 and block_size > 0:
        n, _c, h, w = out.shape
        for i in range(n):
            if random.random() >= block_rate:
                continue
            y = random.randint(0, max(0, h - block_size))
            x = random.randint(0, max(0, w - block_size))
            patch = torch.rand((3, block_size, block_size), device=out.device)
            out[i, :, y:y + block_size, x:x + block_size] = patch
    return out.clamp(0.0, 1.0)


def make_children(parents: torch.Tensor, count: int) -> torch.Tensor:
    n = parents.shape[0]
    a = parents[torch.randint(0, n, (count,), device=parents.device)]
    b = parents[torch.randint(0, n, (count,), device=parents.device)]
    mask = torch.rand((count, 1, *parents.shape[2:]), device=parents.device) < 0.5
    return torch.where(mask, a, b)


def evaluate(model, pop: torch.Tensor, target: int, target_margin: float, wrong_penalty: float, batch_size: int):
    margins, probs, preds, nexts, fitnesses = [], [], [], [], []
    with torch.no_grad():
        for batch in pop.split(batch_size):
            logits = model(batch)
            margin, prob, pred, next_idx = logits_info(logits, target)
            positive = margin > 0
            # Maximize closeness to target_margin while strongly penalizing crossing the boundary.
            fitness = -torch.abs(margin - target_margin)
            fitness = torch.where(positive, fitness, fitness - wrong_penalty - margin.abs())
            margins.append(margin)
            probs.append(prob)
            preds.append(pred)
            nexts.append(next_idx)
            fitnesses.append(fitness)
    return {
        'margin': torch.cat(margins),
        'prob': torch.cat(probs),
        'pred': torch.cat(preds),
        'next_best_class': torch.cat(nexts),
        'fitness': torch.cat(fitnesses),
    }


def top5(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=1)
    vals, idxs = torch.topk(probs, 5, dim=1)
    return [{'class': int(i), 'prob': float(v)} for v, i in zip(vals[0].cpu(), idxs[0].cpu())]


def run_one(args, model, dataset, target: int, init_mode: str, sample_id: int, device: torch.device):
    run_name = f'class{target:04d}_{init_mode}_dirty_sample{sample_id:02d}'
    run_dir = Path(args.output_dir) / 'runs' / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if init_mode == 'random':
        pop = torch.rand((args.population, 3, args.image_size, args.image_size), device=device)
    elif init_mode == 'real':
        start = load_real_start(dataset, target, sample_id, device)
        pop = (start + torch.randn((args.population, 3, args.image_size, args.image_size), device=device) * args.real_init_sigma).clamp(0.0, 1.0)
        pop[0:1] = start
    else:
        raise ValueError(init_mode)

    best = None
    generations_to_success = None
    history_path = run_dir / 'history.csv'
    with history_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['generation','best_fitness','best_margin','best_prob','best_pred','best_next_best_class','mean_fitness'])
        writer.writeheader()
        for gen in range(args.generations + 1):
            stats = evaluate(model, pop, target, args.target_margin, args.wrong_penalty, args.eval_batch_size)
            order = torch.argsort(stats['fitness'], descending=True)
            pop = pop[order]
            for key in stats:
                stats[key] = stats[key][order]
            cur = {
                'image': pop[0:1].detach().clone(),
                'fitness': float(stats['fitness'][0].item()),
                'margin': float(stats['margin'][0].item()),
                'prob': float(stats['prob'][0].item()),
                'pred': int(stats['pred'][0].item()),
                'next_best_class': int(stats['next_best_class'][0].item()),
                'generation': gen,
            }
            valid_dirty = cur['pred'] == target and 0.0 < cur['margin'] <= args.max_success_margin
            if best is None:
                best = cur
            else:
                best_valid = best['pred'] == target and best['margin'] > 0
                cur_valid = cur['pred'] == target and cur['margin'] > 0
                if (cur_valid and not best_valid) or (cur_valid and abs(cur['margin'] - args.target_margin) < abs(best['margin'] - args.target_margin)) or (not best_valid and cur['fitness'] > best['fitness']):
                    best = cur
            if generations_to_success is None and valid_dirty:
                generations_to_success = gen
            writer.writerow({
                'generation': gen,
                'best_fitness': cur['fitness'],
                'best_margin': cur['margin'],
                'best_prob': cur['prob'],
                'best_pred': cur['pred'],
                'best_next_best_class': cur['next_best_class'],
                'mean_fitness': float(stats['fitness'].mean().item()),
            })
            if gen % args.save_every == 0 or gen == args.generations:
                tv_utils.save_image(cur['image'].cpu(), run_dir / f'best_gen{gen:05d}.png')
            if args.stop_on_success and generations_to_success is not None:
                break
            if gen == args.generations:
                break
            elite = pop[:args.elite]
            children = make_children(pop[:args.parents], args.population - args.elite)
            children = mutate(children, args.pixel_sigma, args.pixel_rate, args.block_rate, args.block_size)
            pop = torch.cat([elite, children], dim=0)

    final_path = run_dir / 'final_best.png'
    tensor_path = run_dir / 'final_best.pt'
    tv_utils.save_image(best['image'].cpu(), final_path)
    torch.save(best['image'].cpu(), tensor_path)
    with torch.no_grad():
        logits = model(best['image'])
    metadata = {
        'run_name': run_name,
        'target_class': target,
        'init_mode': init_mode,
        'sample_id': sample_id,
        'final_image': str(final_path),
        'final_tensor': str(tensor_path),
        'history_csv': str(history_path),
        'final_margin': best['margin'],
        'final_prob': best['prob'],
        'final_pred': best['pred'],
        'final_next_best_class': best['next_best_class'],
        'final_fitness': best['fitness'],
        'generations_to_success': generations_to_success,
        'completed_generations': best['generation'],
        'target_margin': args.target_margin,
        'max_success_margin': args.max_success_margin,
        'success': int(best['pred'] == target and 0.0 < best['margin'] <= args.max_success_margin),
        'top5': top5(logits),
    }
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-model', default='resnet18')
    p.add_argument('--imagenet-root', default='/home/sepi/Study/coding/data/imagenet/val')
    p.add_argument('--classes', default='1')
    p.add_argument('--images-per-class', type=int, default=1)
    p.add_argument('--init-modes', default='real,random')
    p.add_argument('--output-dir', default='analysis_outputs/pure_af_geometry/dirty_images_resnet18')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--population', type=int, default=64)
    p.add_argument('--parents', type=int, default=16)
    p.add_argument('--elite', type=int, default=4)
    p.add_argument('--generations', type=int, default=10000)
    p.add_argument('--eval-batch-size', type=int, default=32)
    p.add_argument('--target-margin', type=float, default=0.01)
    p.add_argument('--max-success-margin', type=float, default=0.05)
    p.add_argument('--wrong-penalty', type=float, default=20.0)
    p.add_argument('--pixel-sigma', type=float, default=0.06)
    p.add_argument('--pixel-rate', type=float, default=0.03)
    p.add_argument('--block-rate', type=float, default=0.3)
    p.add_argument('--block-size', type=int, default=24)
    p.add_argument('--real-init-sigma', type=float, default=0.08)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--stop-on-success', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    model = load_imagenet_model(args.target_model).to(device).eval()
    rows = []
    started = time.time()
    for target in parse_ints(args.classes):
        for init_mode in [x.strip() for x in args.init_modes.split(',') if x.strip()]:
            for sample_id in range(args.images_per_class):
                set_seed(args.seed + target * 10000 + sample_id * 100 + stable_name_offset(init_mode, 'dirty'))
                print(f'[START] class={target} init={init_mode} dirty sample={sample_id}', flush=True)
                row = run_one(args, model, dataset, target, init_mode, sample_id, device)
                rows.append(row)
                print(f"[DONE] {row['run_name']} margin={row['final_margin']:.6f} prob={row['final_prob']:.6f} pred={row['final_pred']} success={row['success']}", flush=True)
                pd_rows = rows
                with (out / 'manifest.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(pd_rows[0].keys()))
                    writer.writeheader(); writer.writerows(pd_rows)
    (out / 'metadata.json').write_text(json.dumps({'experiment':'pure_af_geometry_dirty_ga','args':vars(args),'elapsed_sec':time.time()-started,'rows':len(rows)}, indent=2))
    print(f'[DONE] manifest={out / "manifest.csv"} rows={len(rows)}')


if __name__ == '__main__':
    main()
