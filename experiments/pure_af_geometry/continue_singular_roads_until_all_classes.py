#!/usr/bin/env python3
"""Continue selected singular roads until all classes are visited or a cap is reached."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.trace_jacobian_singular_roads import (  # noqa: E402
    estimate_top_direction,
    feat,
    load_model,
    logits_stats,
    project_linf,
    set_seed,
)


def ordered_unique(xs) -> list[int]:
    out: list[int] = []
    for x in xs:
        x = int(x)
        if not out or out[-1] != x:
            out.append(x)
    return out


def load_candidates(path: str, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(["n_unique_classes", "image_id"], ascending=[False, True])
    return df.head(top_n)[["image_id", "direction"]].copy()


def trace_candidate(model, dataset, image_id: int, direction: str, args, device):
    x0, label0 = dataset[int(image_id)]
    x0 = x0.unsqueeze(0).to(device)
    y = torch.tensor([int(label0)], device=device)
    sign = 1.0 if direction == "forward" else -1.0
    h0 = feat(model, x0).detach()
    pred0, m0, conf0 = logits_stats(model, x0, y)

    x = x0.detach().clone()
    prev_v = None
    rows = []
    seen = set()
    first_all_step = -1
    for t in range(args.max_steps + 1):
        v, sigma = estimate_top_direction(model, x, args.power_iters, prev_v, args.seed + int(image_id) * 1009 + t)
        if prev_v is not None:
            cos_prev = float((v.flatten(1) * prev_v.flatten(1)).sum().item())
            if cos_prev < 0:
                v = -v
                cos_prev = -cos_prev
        else:
            cos_prev = np.nan
        h = feat(model, x).detach()
        pred, m, conf = logits_stats(model, x, y)
        seen.add(pred)
        if len(seen) == 10 and first_all_step < 0:
            first_all_step = t
        rows.append(
            {
                "image_id": int(image_id),
                "direction": direction,
                "step": t,
                "pred0": pred0,
                "pred": pred,
                "label": int(label0),
                "changed_class": int(pred != pred0),
                "success": int(pred != int(label0)),
                "margin0": m0,
                "margin": m,
                "confidence": conf,
                "sigma1_est": sigma,
                "sigma_ratio": np.nan,
                "cos_prev": cos_prev,
                "linf_from_start": float((x - x0).abs().max().item()),
                "l2_from_start": float((x - x0).flatten(1).norm(dim=1).item()),
                "hidden_dist_from_start": float((h - h0).norm(dim=1).item()),
                "n_seen_classes": len(seen),
                "seen_classes": " ".join(map(str, sorted(seen))),
                "first_all_classes_step": first_all_step,
            }
        )
        if len(seen) == 10:
            break
        if t == args.max_steps:
            break
        step = sign * args.step_l2 * v
        x = project_linf(x + step, x0, args.eps / 255.0).detach()
        prev_v = v.detach()

    sigma0 = max(float(rows[0]["sigma1_est"]), 1e-12)
    for row in rows:
        row["sigma_ratio"] = float(row["sigma1_est"]) / sigma0
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--step-l2", type=float, default=0.18)
    p.add_argument("--power-iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=31)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    candidates = load_candidates(args.candidate_csv, args.top_n)
    all_rows = []
    for _, row in candidates.iterrows():
        print(f"[trace] image={int(row.image_id)} direction={row.direction}", flush=True)
        all_rows.extend(trace_candidate(model, dataset, int(row.image_id), str(row.direction), args, device))
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "singular_road_until_all_trace_steps.csv", index=False)

    summary_rows = []
    for (image_id, direction), g in df.groupby(["image_id", "direction"]):
        g = g.sort_values("step")
        preds = g.pred.to_numpy(dtype=int)
        seq = ordered_unique(preds)
        seen = sorted(set(preds.tolist()))
        first_all = int(g.first_all_classes_step.max())
        summary_rows.append(
            {
                "image_id": int(image_id),
                "direction": direction,
                "n_steps": int(g.step.max()),
                "n_unique_classes": len(seen),
                "visited_classes": " ".join(map(str, seen)),
                "class_sequence": "->".join(map(str, seq)),
                "reached_all_classes": int(first_all >= 0),
                "first_all_classes_step": first_all,
                "final_pred": int(preds[-1]),
                "final_margin": float(g.margin.iloc[-1]),
                "final_sigma_ratio": float(g.sigma_ratio.iloc[-1]),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["reached_all_classes", "n_unique_classes"], ascending=[False, False])
    summary.to_csv(out / "singular_road_until_all_summary.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(vars(args), indent=2))
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
