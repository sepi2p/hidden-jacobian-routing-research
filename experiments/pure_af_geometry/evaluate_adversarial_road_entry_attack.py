#!/usr/bin/env python3
"""Evaluate targeted attacks guided by an adversarial-road library.

The road library is built from successful targeted attacks on one split of
class-0 images.  Each library segment stores its hidden position, hidden
transport direction, and target-margin gain.  Held-out class-0 images are then
attacked by direct targeted CE or by variants that enter/follow/use these
adversarial road segments.
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

from experiments.pure_af_geometry.compare_direct_vs_sequence_targeted_pgd import (  # noqa: E402
    eval_state,
    run_direct,
    select_clean_correct_class,
    targeted_step,
)
from experiments.pure_af_geometry.evaluate_road_damping_defense import project_linf  # noqa: E402
from experiments.pure_af_geometry.trace_jacobian_singular_roads import feat, load_model, set_seed  # noqa: E402


def hidden(model, x: torch.Tensor) -> torch.Tensor:
    return feat(model, x)


def target_margin_from_logits(logits: torch.Tensor, target: int) -> float:
    other = logits.clone()
    other[:, target] = -1e9
    return float((logits[:, target] - other.max(1).values).item())


def direct_targeted_states(model, x0, source: int, target: int, eps: float, step_size: float, steps: int):
    x = x0.detach().clone()
    states = []
    for step in range(steps + 1):
        with torch.no_grad():
            logits = model(x)
            pred = int(logits.argmax(1).item())
            h = hidden(model, x).detach().cpu().numpy()[0].astype(np.float32)
            tm = target_margin_from_logits(logits, target)
        states.append({"step": step, "x": x.detach(), "h": h, "pred": pred, "target_margin": tm})
        if step == steps:
            break
        x = targeted_step(model, x, x0, target, eps, step_size)
    return states


def build_adv_road_library(model, dataset, library_rows, source: int, target: int, eps: float, step_size: float, steps: int):
    segments = []
    image_rows = []
    for image_id, label in library_rows:
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(next(model.parameters()).device)
        states = direct_targeted_states(model, x0, source, target, eps, step_size, steps)
        final_success = int(states[-1]["pred"] == target)
        image_rows.append(
            {
                "image_id": image_id,
                "final_pred": states[-1]["pred"],
                "final_target_success": final_success,
                "final_target_margin": states[-1]["target_margin"],
            }
        )
        if not final_success:
            continue
        for a, b in zip(states[:-1], states[1:]):
            dh = b["h"] - a["h"]
            norm = float(np.linalg.norm(dh))
            gain = float(b["target_margin"] - a["target_margin"])
            if norm < 1e-8:
                continue
            segments.append(
                {
                    "segment_id": len(segments),
                    "library_image_id": image_id,
                    "step": int(a["step"]),
                    "h_start": a["h"],
                    "h_end": b["h"],
                    "direction": (dh / norm).astype(np.float32),
                    "norm": norm,
                    "target_margin_gain": gain,
                    "pred_start": int(a["pred"]),
                    "pred_end": int(b["pred"]),
                }
            )
    return segments, pd.DataFrame(image_rows)


def choose_segment(model, x, segments, gain_weight: float = 0.0, top_k: int = 256):
    with torch.no_grad():
        h = hidden(model, x).detach().cpu().numpy()[0]
    starts = np.stack([s["h_start"] for s in segments], axis=0)
    dists = np.linalg.norm(starts - h[None, :], axis=1)
    if top_k and len(dists) > top_k:
        idx = np.argpartition(dists, top_k)[:top_k]
    else:
        idx = np.arange(len(dists))
    gains = np.asarray([max(0.0, segments[i]["target_margin_gain"]) for i in idx], dtype=np.float32)
    score = dists[idx] - gain_weight * gains
    j = int(idx[int(np.argmin(score))])
    return segments[j], float(dists[j])


def hidden_target_step(model, x, x0, h_target_np, eps: float, step_size: float):
    target = torch.tensor(h_target_np, device=x.device, dtype=torch.float32).view(1, -1)
    probe = x.detach().requires_grad_(True)
    loss = F.mse_loss(hidden(model, probe), target)
    grad = torch.autograd.grad(loss, probe)[0]
    return project_linf(x.detach() - step_size * grad.sign(), x0, eps).detach()


def road_direction_step(model, x, x0, direction_np, eps: float, step_size: float):
    direction = torch.tensor(direction_np, device=x.device, dtype=torch.float32).view(1, -1)
    probe = x.detach().requires_grad_(True)
    score = (hidden(model, probe) * direction).sum()
    grad = torch.autograd.grad(score, probe)[0]
    return project_linf(x.detach() + step_size * grad.sign(), x0, eps).detach()


def road_guided_ce_step(model, x, x0, target: int, direction_np, eps: float, step_size: float, alpha: float):
    direction = torch.tensor(direction_np, device=x.device, dtype=torch.float32).view(1, -1)
    t = torch.tensor([int(target)], device=x.device)
    probe = x.detach().requires_grad_(True)
    logits = model(probe)
    ce = F.cross_entropy(logits, t)
    road_score = (hidden(model, probe) * direction).sum()
    loss = ce - alpha * road_score
    grad = torch.autograd.grad(loss, probe)[0]
    return project_linf(x.detach() - step_size * grad.sign(), x0, eps).detach()


def run_adv_road(model, x0, source: int, target: int, segments, eps: float, step_size: float, total_steps: int, entry_steps: int, mode: str, gain_weight: float, alpha: float):
    x = x0.detach().clone()
    rows = []
    first_final = -1
    chosen0, entry_dist0 = choose_segment(model, x, segments, gain_weight=gain_weight)
    for step in range(total_steps + 1):
        seg, dist = choose_segment(model, x, segments, gain_weight=gain_weight)
        st = eval_state(model, x, source, target, target)
        if st["final_target_success"] and first_final < 0:
            first_final = step
        st.update(
            {
                "step": step,
                "method": mode,
                "phase": "entry" if step < entry_steps and mode.startswith("entry") else "road",
                "first_final_target_step": first_final,
                "nearest_segment_id": int(seg["segment_id"]),
                "nearest_segment_dist": dist,
                "entry_distance": entry_dist0,
                "chosen_entry_segment_id": int(chosen0["segment_id"]),
                "segment_target_margin_gain": float(seg["target_margin_gain"]),
                "stage_hit_count": int(st["final_target_success"]),
            }
        )
        rows.append(st)
        if step == total_steps:
            break
        if mode == "adv_road_dynamic":
            x = road_direction_step(model, x, x0, seg["direction"], eps, step_size)
        elif mode == "entry_then_adv_road":
            if step < entry_steps:
                x = hidden_target_step(model, x, x0, chosen0["h_start"], eps, step_size)
            else:
                x = road_direction_step(model, x, x0, seg["direction"], eps, step_size)
        elif mode == "entry_then_direct_ce":
            if step < entry_steps:
                x = hidden_target_step(model, x, x0, chosen0["h_start"], eps, step_size)
            else:
                x = targeted_step(model, x, x0, target, eps, step_size)
        elif mode == "adv_road_guided_ce":
            x = road_guided_ce_step(model, x, x0, target, seg["direction"], eps, step_size, alpha)
        else:
            raise ValueError(mode)
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    final = df.sort_values("step").groupby(["method", "image_id"]).tail(1)
    out = (
        final.groupby("method")
        .agg(
            n=("image_id", "count"),
            target_asr=("final_target_success", "mean"),
            untargeted_asr=("untargeted_success", "mean"),
            mean_target_margin=("final_target_margin", "mean"),
            median_target_margin=("final_target_margin", "median"),
            mean_target_prob=("final_target_prob", "mean"),
            mean_entry_distance=("entry_distance", "mean"),
            mean_nearest_segment_dist=("nearest_segment_dist", "mean"),
            mean_segment_gain=("segment_target_margin_gain", "mean"),
            mean_steps_to_target=("first_final_target_step", lambda x: np.mean([v for v in x if v >= 0]) if any(v >= 0 for v in x) else np.nan),
        )
        .reset_index()
    )
    return out


def make_plot(summary: pd.DataFrame, out: Path):
    order = ["direct_ce", "adv_road_dynamic", "entry_then_adv_road", "entry_then_direct_ce", "adv_road_guided_ce"]
    labels = {
        "direct_ce": "direct CE",
        "adv_road_dynamic": "adv-road only",
        "entry_then_adv_road": "entry + adv-road",
        "entry_then_direct_ce": "entry + CE",
        "adv_road_guided_ce": "CE + adv-road prior",
    }
    d = summary.set_index("method").reindex(order).dropna(how="all").reset_index()
    x = np.arange(len(d))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.4), constrained_layout=True)
    axes[0].bar(x, d.target_asr, color="#4C78A8")
    axes[0].set_ylabel("target ASR")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(x, d.mean_target_margin, color="#F58518")
    axes[1].axhline(0, color="black", ls="--", lw=1)
    axes[1].set_ylabel("mean target margin")
    axes[2].bar(x, d.mean_steps_to_target, color="#54A24B")
    axes[2].set_ylabel("mean steps to target")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(m, m) for m in d.method], rotation=25, ha="right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "adversarial_road_entry_attack_summary.png", dpi=220)
    fig.savefig(out / "adversarial_road_entry_attack_summary.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/adversarial_road_entry_attack_class0_to5")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--source-class", type=int, default=0)
    p.add_argument("--target-class", type=int, default=5)
    p.add_argument("--library-images", type=int, default=40)
    p.add_argument("--test-images", type=int, default=30)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--entry-steps", type=int, default=5)
    p.add_argument("--gain-weight", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    selected = select_clean_correct_class(dataset, model, args.source_class, args.library_images + args.test_images, device)
    library_rows = selected[: args.library_images]
    test_rows = selected[args.library_images :]
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    segments, library_summary = build_adv_road_library(
        model, dataset, library_rows, args.source_class, args.target_class, eps, step_size, args.steps
    )
    library_summary.to_csv(out / "adversarial_road_library_images.csv", index=False)
    pd.DataFrame(
        [
            {k: v for k, v in s.items() if k not in {"h_start", "h_end", "direction"}}
            for s in segments
        ]
    ).to_csv(out / "adversarial_road_library_segments.csv", index=False)
    if not segments:
        raise RuntimeError("No successful library segments found.")

    all_rows = []
    for image_id, label in test_rows:
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(device)
        direct_rows, _ = run_direct(model, x0, label, args.target_class, eps, step_size, args.steps)
        for r in direct_rows:
            r.update(
                {
                    "method": "direct_ce",
                    "image_id": image_id,
                    "label": label,
                    "source_class": args.source_class,
                    "target_class": args.target_class,
                    "entry_distance": np.nan,
                    "nearest_segment_dist": np.nan,
                    "segment_target_margin_gain": np.nan,
                }
            )
        all_rows.extend(direct_rows)
        for mode in ["adv_road_dynamic", "entry_then_adv_road", "entry_then_direct_ce", "adv_road_guided_ce"]:
            rows = run_adv_road(
                model,
                x0,
                label,
                args.target_class,
                segments,
                eps,
                step_size,
                args.steps,
                args.entry_steps,
                mode,
                args.gain_weight,
                args.alpha,
            )
            for r in rows:
                r.update({"image_id": image_id, "label": label, "source_class": args.source_class, "target_class": args.target_class})
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "adversarial_road_entry_attack_timeseries.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "adversarial_road_entry_attack_summary.csv", index=False)
    make_plot(summary, out)
    meta = vars(args)
    meta.update(
        {
            "n_library_images": len(library_rows),
            "n_test_images": len(test_rows),
            "n_successful_library_images": int(library_summary.final_target_success.sum()),
            "n_library_segments": len(segments),
        }
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Library successful images: {meta['n_successful_library_images']} / {len(library_rows)}")
    print(f"Library segments: {len(segments)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
