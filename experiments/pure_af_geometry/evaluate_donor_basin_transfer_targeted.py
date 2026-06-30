#!/usr/bin/env python3
"""Test donor-route basin transfer for targeted attacks.

For a hard class-0 image x and target class 5, ask whether moving x toward the
representation neighborhood of an easy donor image x2 makes the later targeted
attack easier.  Donor anchors are extracted from successful targeted PGD
trajectories on a disjoint donor split.

All variants keep the same original L_inf budget around the recipient image.
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


def hidden_np(model, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return hidden(model, x).detach().cpu().numpy()[0].astype(np.float32)


def hidden_target_step(model, x, x0, h_target_np, eps: float, step_size: float):
    h_target = torch.tensor(h_target_np, device=x.device, dtype=torch.float32).view(1, -1)
    probe = x.detach().requires_grad_(True)
    loss = F.mse_loss(hidden(model, probe), h_target)
    grad = torch.autograd.grad(loss, probe)[0]
    return project_linf(x.detach() - step_size * grad.sign(), x0, eps).detach()


def targeted_states(model, x0, source: int, target: int, eps: float, step_size: float, steps: int):
    x = x0.detach().clone()
    states = []
    first_hit = -1
    for step in range(steps + 1):
        st = eval_state(model, x, source, target, target)
        if st["final_target_success"] and first_hit < 0:
            first_hit = step
        states.append({"step": step, "x": x.detach(), "h": hidden_np(model, x), "first_hit": first_hit, **st})
        if step == steps:
            break
        x = targeted_step(model, x, x0, target, eps, step_size)
    return states


def build_donor_anchors(model, dataset, donor_rows, source: int, target: int, eps: float, step_size: float, steps: int, easy_hit_step: int):
    anchors = []
    donor_summary = []
    for image_id, label in donor_rows:
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(next(model.parameters()).device)
        states = targeted_states(model, x0, source, target, eps, step_size, steps)
        first_hit = states[-1]["first_hit"]
        success = int(first_hit >= 0)
        is_easy = int(success and first_hit <= easy_hit_step)
        donor_summary.append(
            {
                "image_id": image_id,
                "success": success,
                "first_hit_step": first_hit,
                "is_easy_donor": is_easy,
                "final_target_margin": states[-1]["final_target_margin"],
            }
        )
        if not is_easy:
            continue
        margins = np.asarray([s["final_target_margin"] for s in states], dtype=float)
        gains = np.diff(margins)
        best_gain_start = int(np.argmax(gains)) if len(gains) else 0
        hit_step = int(first_hit)
        mid_step = max(0, hit_step // 2)
        anchor_specs = {
            "donor_clean": 0,
            "donor_mid": mid_step,
            "donor_hit": hit_step,
            "donor_best_gain_start": best_gain_start,
            "donor_best_gain_end": min(best_gain_start + 1, len(states) - 1),
        }
        for kind, step_idx in anchor_specs.items():
            s = states[int(step_idx)]
            anchors.append(
                {
                    "anchor_id": len(anchors),
                    "donor_image_id": image_id,
                    "anchor_kind": kind,
                    "donor_first_hit_step": hit_step,
                    "anchor_step": int(step_idx),
                    "anchor_pred": int(s["pred"]),
                    "anchor_target_margin": float(s["final_target_margin"]),
                    "h": s["h"],
                }
            )
    return anchors, pd.DataFrame(donor_summary)


def choose_anchor(model, x, anchors, anchor_kind: str | None):
    h = hidden_np(model, x)
    candidates = [a for a in anchors if anchor_kind is None or a["anchor_kind"] == anchor_kind]
    if not candidates:
        raise RuntimeError(f"No anchors available for kind={anchor_kind}")
    dists = np.asarray([np.linalg.norm(h - a["h"]) for a in candidates], dtype=float)
    j = int(np.argmin(dists))
    return candidates[j], float(dists[j])


def run_prestage_then_ce(
    model,
    x0,
    source: int,
    target: int,
    anchor,
    eps: float,
    step_size: float,
    total_steps: int,
    pre_steps: int,
    method: str,
):
    x = x0.detach().clone()
    rows = []
    first_final = -1
    for step in range(total_steps + 1):
        phase = "prestage" if step < pre_steps else "targeted_ce"
        st = eval_state(model, x, source, target, target)
        if st["final_target_success"] and first_final < 0:
            first_final = step
        st.update(
            {
                "step": step,
                "method": method,
                "phase": phase,
                "first_final_target_step": first_final,
                "stage_hit_count": int(st["final_target_success"]),
                "anchor_id": int(anchor["anchor_id"]),
                "anchor_kind": anchor["anchor_kind"],
                "donor_image_id": int(anchor["donor_image_id"]),
                "donor_first_hit_step": int(anchor["donor_first_hit_step"]),
                "anchor_step": int(anchor["anchor_step"]),
                "anchor_pred": int(anchor["anchor_pred"]),
                "anchor_target_margin": float(anchor["anchor_target_margin"]),
            }
        )
        rows.append(st)
        if step == total_steps:
            break
        if step < pre_steps:
            x = hidden_target_step(model, x, x0, anchor["h"], eps, step_size)
        else:
            x = targeted_step(model, x, x0, target, eps, step_size)
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    final = df.sort_values("step").groupby(["method", "image_id"]).tail(1)
    return (
        final.groupby("method")
        .agg(
            n=("image_id", "count"),
            target_asr=("final_target_success", "mean"),
            untargeted_asr=("untargeted_success", "mean"),
            mean_target_margin=("final_target_margin", "mean"),
            median_target_margin=("final_target_margin", "median"),
            mean_target_prob=("final_target_prob", "mean"),
            mean_steps_to_target=("first_final_target_step", lambda x: np.mean([v for v in x if v >= 0]) if any(v >= 0 for v in x) else np.nan),
            mean_anchor_margin=("anchor_target_margin", "mean"),
            mean_donor_hit_step=("donor_first_hit_step", "mean"),
        )
        .reset_index()
    )


def make_plot(summary: pd.DataFrame, out: Path):
    order = [
        "direct_ce",
        "pre5_donor_clean",
        "pre5_donor_mid",
        "pre5_donor_hit",
        "pre5_donor_best_gain_start",
        "pre5_donor_best_gain_end",
        "pre10_donor_hit",
        "pre10_donor_best_gain_end",
    ]
    d = summary.set_index("method").reindex([m for m in order if m in set(summary.method)]).reset_index()
    x = np.arange(len(d))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5), constrained_layout=True)
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
        ax.set_xticklabels(d.method, rotation=30, ha="right", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "donor_basin_transfer_targeted_summary.png", dpi=220)
    fig.savefig(out / "donor_basin_transfer_targeted_summary.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/donor_basin_transfer_class0_to5")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--source-class", type=int, default=0)
    p.add_argument("--target-class", type=int, default=5)
    p.add_argument("--donor-images", type=int, default=80)
    p.add_argument("--recipient-images", type=int, default=80)
    p.add_argument("--max-hard-recipients", type=int, default=30)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--easy-hit-step", type=int, default=5)
    p.add_argument("--hard-hit-step", type=int, default=20)
    p.add_argument("--pre-steps", default="5,10")
    p.add_argument("--anchor-kinds", default="donor_clean,donor_mid,donor_hit,donor_best_gain_start,donor_best_gain_end")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    selected = select_clean_correct_class(
        dataset, model, args.source_class, args.donor_images + args.recipient_images, device
    )
    donor_rows = selected[: args.donor_images]
    recipient_rows = selected[args.donor_images :]
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0

    anchors, donor_summary = build_donor_anchors(
        model,
        dataset,
        donor_rows,
        args.source_class,
        args.target_class,
        eps,
        step_size,
        args.steps,
        args.easy_hit_step,
    )
    donor_summary.to_csv(out / "donor_pool_summary.csv", index=False)
    pd.DataFrame([{k: v for k, v in a.items() if k != "h"} for a in anchors]).to_csv(
        out / "donor_anchor_library.csv", index=False
    )
    if not anchors:
        raise RuntimeError("No easy donor anchors found.")

    recipient_screen = []
    for image_id, label in recipient_rows:
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(device)
        states = targeted_states(model, x0, label, args.target_class, eps, step_size, args.steps)
        first_hit = states[-1]["first_hit"]
        hard = int(first_hit < 0 or first_hit > args.hard_hit_step)
        recipient_screen.append(
            {
                "image_id": image_id,
                "label": label,
                "first_hit_step": first_hit,
                "direct_success": int(first_hit >= 0),
                "hard_recipient": hard,
                "final_target_margin": states[-1]["final_target_margin"],
            }
        )
    rec_df = pd.DataFrame(recipient_screen)
    rec_df.to_csv(out / "recipient_screen.csv", index=False)
    hard_ids = rec_df[rec_df.hard_recipient == 1].image_id.tolist()
    if len(hard_ids) < args.max_hard_recipients:
        # Add slowest/lowest-margin cases if strict failures are too few.
        extra = rec_df[~rec_df.image_id.isin(hard_ids)].sort_values(["direct_success", "first_hit_step", "final_target_margin"], ascending=[True, False, True]).image_id.tolist()
        hard_ids += extra[: args.max_hard_recipients - len(hard_ids)]
    hard_ids = hard_ids[: args.max_hard_recipients]
    test_rows = [(int(r.image_id), int(r.label)) for _, r in rec_df[rec_df.image_id.isin(hard_ids)].iterrows()]

    all_rows = []
    anchor_kinds = [x for x in args.anchor_kinds.split(",") if x]
    pre_steps_list = [int(x) for x in args.pre_steps.split(",") if x]
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
                    "anchor_id": -1,
                    "anchor_kind": "none",
                    "donor_image_id": -1,
                    "donor_first_hit_step": np.nan,
                    "anchor_step": np.nan,
                    "anchor_pred": np.nan,
                    "anchor_target_margin": np.nan,
                }
            )
        all_rows.extend(direct_rows)
        for pre_steps in pre_steps_list:
            for kind in anchor_kinds:
                anchor, dist = choose_anchor(model, x0, anchors, kind)
                method = f"pre{pre_steps}_{kind}"
                rows = run_prestage_then_ce(
                    model, x0, label, args.target_class, anchor, eps, step_size, args.steps, pre_steps, method
                )
                for r in rows:
                    r.update(
                        {
                            "image_id": image_id,
                            "label": label,
                            "source_class": args.source_class,
                            "target_class": args.target_class,
                            "entry_hidden_distance": dist,
                        }
                    )
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "donor_basin_transfer_targeted_timeseries.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "donor_basin_transfer_targeted_summary.csv", index=False)
    make_plot(summary, out)
    meta = vars(args)
    meta.update(
        {
            "n_clean_correct_selected": len(selected),
            "n_easy_donors": int(donor_summary.is_easy_donor.sum()),
            "n_anchors": len(anchors),
            "n_hard_recipients": len(test_rows),
        }
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Easy donors: {meta['n_easy_donors']} / {len(donor_rows)}")
    print(f"Anchors: {len(anchors)}")
    print(f"Hard recipients tested: {len(test_rows)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
