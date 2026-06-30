#!/usr/bin/env python3
"""Compare direct targeted PGD with sequence-guided targeted PGD.

Question: for class-0 images, is it easier to target class 5 directly, or to
move through an observed road sequence such as 0 -> 3 -> 2 -> 5?

All variants remain projected inside the original L_inf ball around x0.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.evaluate_road_damping_defense import project_linf  # noqa: E402
from utils.load_models import load_cifar_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(device: torch.device):
    return load_cifar_model("bbb_resnet50").to(device).eval()


def select_clean_correct_class(dataset, model, source_class: int, n: int, device: torch.device):
    rows = []
    for idx in range(len(dataset)):
        x, y0 = dataset[idx]
        if int(y0) != int(source_class):
            continue
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(model(x).argmax(1).item())
        if pred == int(y0):
            rows.append((idx, int(y0)))
        if len(rows) >= n:
            break
    return rows


def eval_state(model, x, source: int, final_target: int, current_target: int) -> dict:
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = int(logits.argmax(1).item())
        target_logit = float(logits[0, final_target].item())
        other = logits.clone()
        other[0, final_target] = -1e9
        final_target_margin = float((logits[0, final_target] - other.max(1).values[0]).item())
        cur = logits.clone()
        cur[0, current_target] = -1e9
        current_target_margin = float((logits[0, current_target] - cur.max(1).values[0]).item())
        true_margin = float((logits[0, source] - torch.cat([logits[:, :source], logits[:, source + 1 :]], dim=1).max(1).values[0]).item())
    return {
        "pred": pred,
        "final_target_prob": float(probs[0, final_target].item()),
        "final_target_logit": target_logit,
        "final_target_margin": final_target_margin,
        "current_target_margin": current_target_margin,
        "true_margin": true_margin,
        "final_target_success": int(pred == final_target),
        "untargeted_success": int(pred != source),
    }


def targeted_step(model, x, x0, target: int, eps: float, step_size: float):
    t = torch.tensor([int(target)], device=x.device)
    probe = x.detach().requires_grad_(True)
    logits = model(probe)
    loss = F.cross_entropy(logits, t)
    grad = torch.autograd.grad(loss, probe)[0]
    return project_linf(x.detach() - step_size * grad.sign(), x0, eps).detach()


def run_direct(model, x0, source: int, target: int, eps: float, step_size: float, total_steps: int):
    x = x0.detach().clone()
    rows = []
    first_final = -1
    for step in range(total_steps + 1):
        st = eval_state(model, x, source, target, target)
        if st["final_target_success"] and first_final < 0:
            first_final = step
        rows.append(
            {
                "step": step,
                "method": "direct",
                "current_stage": 0,
                "current_target": target,
                "stage_hit_count": int(st["final_target_success"]),
                "first_final_target_step": first_final,
                **st,
            }
        )
        if step == total_steps:
            break
        x = targeted_step(model, x, x0, target, eps, step_size)
    return rows, x


def run_strict_sequence(model, x0, source: int, sequence: list[int], eps: float, step_size: float, total_steps: int):
    x = x0.detach().clone()
    rows = []
    stage = 0
    stage_hits = []
    first_final = -1
    final_target = sequence[-1]
    for step in range(total_steps + 1):
        current_target = sequence[min(stage, len(sequence) - 1)]
        st = eval_state(model, x, source, final_target, current_target)
        if st["pred"] == current_target and len(stage_hits) <= stage:
            stage_hits.append({"target": current_target, "step": step})
            if stage < len(sequence) - 1:
                stage += 1
                current_target = sequence[stage]
                st = eval_state(model, x, source, final_target, current_target)
        if st["final_target_success"] and first_final < 0:
            first_final = step
        rows.append(
            {
                "step": step,
                "method": "strict_sequence",
                "current_stage": stage,
                "current_target": current_target,
                "stage_hit_count": len(stage_hits),
                "first_final_target_step": first_final,
                **st,
            }
        )
        if step == total_steps:
            break
        x = targeted_step(model, x, x0, current_target, eps, step_size)
    return rows, x, stage_hits


def run_fixed_schedule(model, x0, source: int, sequence: list[int], eps: float, step_size: float, total_steps: int):
    x = x0.detach().clone()
    rows = []
    final_target = sequence[-1]
    stage_len = max(1, total_steps // len(sequence))
    first_final = -1
    hit_targets = set()
    for step in range(total_steps + 1):
        stage = min(step // stage_len, len(sequence) - 1)
        current_target = sequence[stage]
        st = eval_state(model, x, source, final_target, current_target)
        if st["pred"] in sequence:
            hit_targets.add(st["pred"])
        if st["final_target_success"] and first_final < 0:
            first_final = step
        rows.append(
            {
                "step": step,
                "method": "fixed_schedule_sequence",
                "current_stage": stage,
                "current_target": current_target,
                "stage_hit_count": len(hit_targets),
                "first_final_target_step": first_final,
                **st,
            }
        )
        if step == total_steps:
            break
        x = targeted_step(model, x, x0, current_target, eps, step_size)
    return rows, x


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    final = df.sort_values("step").groupby(["method", "image_id"]).tail(1)
    out = (
        final.groupby("method")
        .agg(
            n=("image_id", "count"),
            final_target_asr=("final_target_success", "mean"),
            untargeted_asr=("untargeted_success", "mean"),
            mean_final_target_margin=("final_target_margin", "mean"),
            median_final_target_margin=("final_target_margin", "median"),
            mean_target_prob=("final_target_prob", "mean"),
            mean_stage_hit_count=("stage_hit_count", "mean"),
            reached_all_sequence=("stage_hit_count", lambda x: float(np.mean(np.asarray(x) >= 3))),
        )
        .reset_index()
    )
    hit = (
        final[final.first_final_target_step >= 0]
        .groupby("method")
        .agg(mean_steps_to_target=("first_final_target_step", "mean"), median_steps_to_target=("first_final_target_step", "median"))
        .reset_index()
    )
    return out.merge(hit, on="method", how="left")


def make_plot(summary: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), constrained_layout=True)
    methods = summary.method.tolist()
    labels = ["direct 0->5", "fixed 0->3->2->5", "strict 0->3->2->5"]
    label_map = {
        "direct": labels[0],
        "fixed_schedule_sequence": labels[1],
        "strict_sequence": labels[2],
    }
    x = np.arange(len(methods))
    axes[0].bar(x, summary.final_target_asr, color="#4C78A8")
    axes[0].set_ylabel("targeted ASR to class 5")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(x, summary.mean_final_target_margin, color="#F58518")
    axes[1].axhline(0, color="black", lw=1, ls="--")
    axes[1].set_ylabel("mean target margin")
    axes[2].bar(x, summary.mean_stage_hit_count, color="#54A24B")
    axes[2].set_ylabel("mean sequence targets reached")
    axes[2].set_ylim(0, 3.2)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([label_map.get(m, m) for m in methods], rotation=20, ha="right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "direct_vs_sequence_targeted_pgd.png", dpi=220)
    fig.savefig(out / "direct_vs_sequence_targeted_pgd.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/direct_vs_sequence_targeted_pgd_class0_to5")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--source-class", type=int, default=0)
    p.add_argument("--direct-target", type=int, default=5)
    p.add_argument("--sequence", default="3,2,5")
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    selected = select_clean_correct_class(dataset, model, args.source_class, args.images, device)
    sequence = [int(x) for x in args.sequence.split(",") if x.strip()]
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0

    all_rows = []
    stage_rows = []
    for image_id, label in selected:
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(device)
        for rows, _x in [
            run_direct(model, x0, label, args.direct_target, eps, step_size, args.steps),
            run_fixed_schedule(model, x0, label, sequence, eps, step_size, args.steps),
        ]:
            for r in rows:
                r.update({"image_id": image_id, "label": label, "source_class": args.source_class, "sequence": args.sequence})
            all_rows.extend(rows)
        rows, _x, hits = run_strict_sequence(model, x0, label, sequence, eps, step_size, args.steps)
        for r in rows:
            r.update({"image_id": image_id, "label": label, "source_class": args.source_class, "sequence": args.sequence})
        all_rows.extend(rows)
        for h in hits:
            h.update({"image_id": image_id, "label": label})
            stage_rows.append(h)

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "direct_vs_sequence_targeted_pgd_timeseries.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(out / "strict_sequence_stage_hits.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "direct_vs_sequence_targeted_pgd_summary.csv", index=False)
    make_plot(summary, out)
    (out / "metadata.json").write_text(json.dumps(vars(args), indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
