#!/usr/bin/env python3
"""Max-close donor-basin transfer test.

This is a stronger version of donor-basin transfer: first use many projected
hidden-matching steps to move a hard recipient as close as possible to an easy
donor anchor, then run targeted PGD without resetting the original L_inf budget.
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
from experiments.pure_af_geometry.evaluate_donor_basin_transfer_targeted import (  # noqa: E402
    build_donor_anchors,
    hidden_target_step,
    targeted_states,
)
from experiments.pure_af_geometry.trace_jacobian_singular_roads import feat, load_model, set_seed  # noqa: E402


def hidden_np(model, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return feat(model, x).detach().cpu().numpy()[0].astype(np.float32)


def choose_anchor_by_closability(model, x0, anchors, anchor_kinds: set[str], probe_steps: int, eps: float, step_size: float):
    """Choose the anchor that becomes closest after a short hidden-match probe."""
    candidates = [a for a in anchors if a["anchor_kind"] in anchor_kinds]
    if not candidates:
        raise RuntimeError(f"No anchors for {anchor_kinds}")
    scored = []
    for a in candidates:
        x = x0.detach().clone()
        d0 = float(np.linalg.norm(hidden_np(model, x) - a["h"]))
        for _ in range(probe_steps):
            x = hidden_target_step(model, x, x0, a["h"], eps, step_size)
        d1 = float(np.linalg.norm(hidden_np(model, x) - a["h"]))
        scored.append((d1, d0, a))
    scored.sort(key=lambda z: z[0])
    d1, d0, a = scored[0]
    return a, d0, d1


def run_maxclose_then_ce(model, x0, source: int, target: int, anchor, eps: float, step_size: float, pre_steps: int, attack_steps: int):
    x = x0.detach().clone()
    rows = []
    d0 = float(np.linalg.norm(hidden_np(model, x) - anchor["h"]))
    best_d = d0
    for step in range(pre_steps):
        st = eval_state(model, x, source, target, target)
        dist = float(np.linalg.norm(hidden_np(model, x) - anchor["h"]))
        best_d = min(best_d, dist)
        st.update(
            {
                "global_step": step,
                "phase": "maxclose",
                "method": f"maxclose{pre_steps}_then_ce{attack_steps}",
                "anchor_id": int(anchor["anchor_id"]),
                "anchor_kind": anchor["anchor_kind"],
                "donor_image_id": int(anchor["donor_image_id"]),
                "donor_first_hit_step": int(anchor["donor_first_hit_step"]),
                "anchor_distance": dist,
                "best_anchor_distance": best_d,
                "first_final_target_step": -1,
            }
        )
        rows.append(st)
        x = hidden_target_step(model, x, x0, anchor["h"], eps, step_size)
    first_final = -1
    for k in range(attack_steps + 1):
        step = pre_steps + k
        st = eval_state(model, x, source, target, target)
        if st["final_target_success"] and first_final < 0:
            first_final = step
        dist = float(np.linalg.norm(hidden_np(model, x) - anchor["h"]))
        best_d = min(best_d, dist)
        st.update(
            {
                "global_step": step,
                "phase": "targeted_ce",
                "method": f"maxclose{pre_steps}_then_ce{attack_steps}",
                "anchor_id": int(anchor["anchor_id"]),
                "anchor_kind": anchor["anchor_kind"],
                "donor_image_id": int(anchor["donor_image_id"]),
                "donor_first_hit_step": int(anchor["donor_first_hit_step"]),
                "anchor_distance": dist,
                "best_anchor_distance": best_d,
                "first_final_target_step": first_final,
            }
        )
        rows.append(st)
        if k == attack_steps:
            break
        x = targeted_step(model, x, x0, target, eps, step_size)
    return rows, d0, best_d


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    final = df.sort_values("global_step").groupby(["method", "image_id"]).tail(1)
    return (
        final.groupby("method")
        .agg(
            n=("image_id", "count"),
            target_asr=("final_target_success", "mean"),
            untargeted_asr=("untargeted_success", "mean"),
            mean_target_margin=("final_target_margin", "mean"),
            median_target_margin=("final_target_margin", "median"),
            mean_target_prob=("final_target_prob", "mean"),
            mean_anchor_distance=("anchor_distance", "mean"),
            mean_best_anchor_distance=("best_anchor_distance", "mean"),
            mean_steps_to_target=("first_final_target_step", lambda x: np.mean([v for v in x if v >= 0]) if any(v >= 0 for v in x) else np.nan),
        )
        .reset_index()
    )


def make_plot(summary: pd.DataFrame, out: Path):
    d = summary.copy()
    x = np.arange(len(d))
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.4), constrained_layout=True)
    axes[0].bar(x, d.target_asr, color="#4C78A8")
    axes[0].set_ylabel("target ASR")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(x, d.mean_target_margin, color="#F58518")
    axes[1].axhline(0, color="black", ls="--", lw=1)
    axes[1].set_ylabel("mean target margin")
    axes[2].bar(x, d.mean_best_anchor_distance, color="#72B7B2")
    axes[2].set_ylabel("best hidden distance to donor")
    axes[3].bar(x, d.mean_steps_to_target, color="#54A24B")
    axes[3].set_ylabel("mean steps to target")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(d.method, rotation=30, ha="right", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "donor_basin_maxclose_targeted_summary.png", dpi=220)
    fig.savefig(out / "donor_basin_maxclose_targeted_summary.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/donor_basin_maxclose_class0_to5_e2")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--source-class", type=int, default=0)
    p.add_argument("--target-class", type=int, default=5)
    p.add_argument("--donor-images", type=int, default=80)
    p.add_argument("--recipient-images", type=int, default=80)
    p.add_argument("--max-hard-recipients", type=int, default=30)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--screen-steps", type=int, default=20)
    p.add_argument("--attack-steps", type=int, default=20)
    p.add_argument("--pre-steps", default="20,40,80")
    p.add_argument("--probe-steps", type=int, default=5)
    p.add_argument("--easy-hit-step", type=int, default=5)
    p.add_argument("--anchor-kinds", default="donor_hit,donor_best_gain_end")
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
        args.screen_steps,
        args.easy_hit_step,
    )
    donor_summary.to_csv(out / "donor_pool_summary.csv", index=False)
    pd.DataFrame([{k: v for k, v in a.items() if k != "h"} for a in anchors]).to_csv(out / "donor_anchor_library.csv", index=False)
    anchor_kinds = set(args.anchor_kinds.split(","))
    anchors = [a for a in anchors if a["anchor_kind"] in anchor_kinds]
    if not anchors:
        raise RuntimeError("No anchors after filtering.")

    rec_rows = []
    for image_id, label in recipient_rows:
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(device)
        states = targeted_states(model, x0, label, args.target_class, eps, step_size, args.screen_steps)
        first_hit = states[-1]["first_hit"]
        rec_rows.append(
            {
                "image_id": image_id,
                "label": label,
                "first_hit_step": first_hit,
                "direct_success": int(first_hit >= 0),
                "final_target_margin": states[-1]["final_target_margin"],
            }
        )
    rec_df = pd.DataFrame(rec_rows)
    rec_df.to_csv(out / "recipient_screen.csv", index=False)
    hard = rec_df[rec_df.direct_success == 0].copy()
    if len(hard) < args.max_hard_recipients:
        extra = rec_df[rec_df.direct_success == 1].sort_values(["first_hit_step", "final_target_margin"], ascending=[False, True])
        hard = pd.concat([hard, extra.head(args.max_hard_recipients - len(hard))], ignore_index=True)
    hard = hard.head(args.max_hard_recipients)

    pre_steps_list = [int(x) for x in args.pre_steps.split(",") if x]
    all_rows = []
    distance_rows = []
    for _, rec in hard.iterrows():
        image_id = int(rec.image_id)
        label = int(rec.label)
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(device)
        # Baselines: direct with attack budget only and direct with same total steps as maxclose variants.
        for steps, name in [(args.attack_steps, f"direct_ce{args.attack_steps}"), (max(pre_steps_list) + args.attack_steps, f"direct_ce{max(pre_steps_list)+args.attack_steps}")]:
            rows, _ = run_direct(model, x0, label, args.target_class, eps, step_size, steps)
            for r in rows:
                r.update(
                    {
                        "global_step": r["step"],
                        "method": name,
                        "image_id": image_id,
                        "label": label,
                        "anchor_distance": np.nan,
                        "best_anchor_distance": np.nan,
                    }
                )
            all_rows.extend(rows)
        anchor, d0_probe, d1_probe = choose_anchor_by_closability(
            model, x0, anchors, anchor_kinds, args.probe_steps, eps, step_size
        )
        for pre_steps in pre_steps_list:
            rows, d0, best_d = run_maxclose_then_ce(
                model, x0, label, args.target_class, anchor, eps, step_size, pre_steps, args.attack_steps
            )
            for r in rows:
                r.update({"image_id": image_id, "label": label})
            all_rows.extend(rows)
            distance_rows.append(
                {
                    "image_id": image_id,
                    "anchor_id": int(anchor["anchor_id"]),
                    "anchor_kind": anchor["anchor_kind"],
                    "pre_steps": pre_steps,
                    "initial_distance": d0,
                    "probe_final_distance": d1_probe,
                    "best_distance": best_d,
                    "distance_reduction": d0 - best_d,
                    "distance_reduction_frac": (d0 - best_d) / max(d0, 1e-12),
                }
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "donor_basin_maxclose_targeted_timeseries.csv", index=False)
    pd.DataFrame(distance_rows).to_csv(out / "donor_basin_maxclose_distances.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "donor_basin_maxclose_targeted_summary.csv", index=False)
    make_plot(summary, out)
    meta = vars(args)
    meta.update({"n_easy_donors": int(donor_summary.is_easy_donor.sum()), "n_anchors": len(anchors), "n_hard_recipients": int(len(hard))})
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Easy donors: {meta['n_easy_donors']} / {len(donor_rows)}")
    print(f"Anchors after filter: {len(anchors)}")
    print(f"Hard recipients: {len(hard)}")
    print(summary.to_string(index=False))
    print("\nDistance reduction:")
    print(pd.DataFrame(distance_rows).groupby("pre_steps").agg(mean_initial=("initial_distance","mean"), mean_best=("best_distance","mean"), mean_reduction_frac=("distance_reduction_frac","mean")).reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
