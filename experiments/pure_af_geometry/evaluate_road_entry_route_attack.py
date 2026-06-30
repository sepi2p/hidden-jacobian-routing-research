#!/usr/bin/env python3
"""Test whether entering a road-route basin helps targeted attacks.

This tests the hypothesis:

  road continuation gives class-transition routes, but route usefulness is
  image-dependent.  If a clean image can first be moved near the entry of a
  target-reaching route, then following that route may make targeted attack
  easier.

The pilot builds a route library from separate class-0 images, extracts observed
routes matching 0 -> 3 -> 2 -> 5, and attacks held-out class-0 images.
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
    run_strict_sequence,
    targeted_step,
)
from experiments.pure_af_geometry.evaluate_road_damping_defense import project_linf  # noqa: E402
from experiments.pure_af_geometry.trace_jacobian_singular_roads import (  # noqa: E402
    estimate_top_direction,
    feat,
    load_model,
    logits_stats,
    select_images_class,
    set_seed,
)


def hidden(model, x):
    return feat(model, x)


def trace_road_states(model, x0, y, sign: float, steps: int, step_l2: float, power_iters: int, eps: float, seed: int):
    x = x0.detach().clone()
    prev_v = None
    states = []
    for t in range(steps + 1):
        v, sigma = estimate_top_direction(model, x, power_iters, prev_v, seed + t)
        if prev_v is not None and float((v.flatten(1) * prev_v.flatten(1)).sum().item()) < 0:
            v = -v
        pred, margin, conf = logits_stats(model, x, y)
        with torch.no_grad():
            h = hidden(model, x).detach().cpu().numpy()[0].astype(np.float32)
        states.append({"step": t, "x": x.detach(), "h": h, "pred": pred, "margin": margin, "sigma": sigma})
        if t == steps:
            break
        x = project_linf(x + sign * step_l2 * v, x0, eps).detach()
        prev_v = v.detach()
    return states


def extract_route(states: list[dict], route: list[int]):
    idxs = []
    start = 0
    for target in route:
        found = None
        for i in range(start, len(states)):
            if int(states[i]["pred"]) == int(target):
                found = i
                break
        if found is None:
            return None
        idxs.append(found)
        start = found + 1
    return [states[i] for i in idxs]


def build_route_library(model, dataset, library_rows, route: list[int], args, device):
    routes = []
    rows = []
    eps = args.eps / 255.0
    for image_id, label in library_rows:
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        for direction, sign in [("forward", 1.0), ("backward", -1.0)]:
            states = trace_road_states(
                model,
                x0,
                y,
                sign,
                args.library_road_steps,
                args.road_step_l2,
                args.power_iters,
                eps,
                args.seed + image_id * 1009 + (0 if sign > 0 else 777),
            )
            route_states = extract_route(states, route)
            seq = []
            for s in states:
                p = int(s["pred"])
                if not seq or seq[-1] != p:
                    seq.append(p)
            rows.append(
                {
                    "library_image_id": image_id,
                    "direction": direction,
                    "class_sequence": "->".join(map(str, seq)),
                    "contains_route": int(route_states is not None),
                    "n_unique_classes": len(set(seq)),
                }
            )
            if route_states is not None:
                routes.append(
                    {
                        "route_id": len(routes),
                        "library_image_id": image_id,
                        "direction": direction,
                        "route": route,
                        "states": route_states,
                        "entry_h": route_states[0]["h"],
                        "waypoints_h": [s["h"] for s in route_states[1:]],
                        "hit_steps": [int(s["step"]) for s in route_states],
                    }
                )
    return routes, pd.DataFrame(rows)


def choose_route(model, x0, routes):
    with torch.no_grad():
        h = hidden(model, x0).detach().cpu().numpy()[0]
    dists = [float(np.linalg.norm(h - r["entry_h"])) for r in routes]
    j = int(np.argmin(dists))
    return routes[j], dists[j]


def hidden_target_step(model, x, x0, h_target_np, eps: float, step_size: float):
    h_target = torch.tensor(h_target_np, device=x.device, dtype=torch.float32).view(1, -1)
    probe = x.detach().requires_grad_(True)
    loss = F.mse_loss(hidden(model, probe), h_target)
    grad = torch.autograd.grad(loss, probe)[0]
    return project_linf(x.detach() - step_size * grad.sign(), x0, eps).detach()


def run_entry_then_strict_sequence(model, x0, source: int, route: list[int], chosen_route, eps: float, step_size: float, total_steps: int, entry_steps: int):
    x = x0.detach().clone()
    rows = []
    final_target = route[-1]
    for step in range(entry_steps):
        st = eval_state(model, x, source, final_target, route[0])
        st.update({"step": step, "phase": "entry", "method": "entry_then_strict_sequence", "current_target": route[0]})
        rows.append(st)
        x = hidden_target_step(model, x, x0, chosen_route["entry_h"], eps, step_size)
    # strict CE sequence after entry.
    stage = 1 if len(route) > 1 else 0
    first_final = -1
    stage_hits = []
    for k in range(total_steps - entry_steps + 1):
        step = entry_steps + k
        current_target = route[min(stage, len(route) - 1)]
        st = eval_state(model, x, source, final_target, current_target)
        if st["pred"] == current_target and len(stage_hits) <= stage:
            stage_hits.append({"target": current_target, "step": step})
            if stage < len(route) - 1:
                stage += 1
                current_target = route[stage]
                st = eval_state(model, x, source, final_target, current_target)
        if st["final_target_success"] and first_final < 0:
            first_final = step
        st.update(
            {
                "step": step,
                "phase": "sequence_ce",
                "method": "entry_then_strict_sequence",
                "current_target": current_target,
                "current_stage": stage,
                "stage_hit_count": len(stage_hits),
                "first_final_target_step": first_final,
            }
        )
        rows.append(st)
        if k == total_steps - entry_steps:
            break
        x = targeted_step(model, x, x0, current_target, eps, step_size)
    return rows, x


def run_waypoint_route(model, x0, source: int, route: list[int], chosen_route, eps: float, step_size: float, total_steps: int, entry_steps: int):
    x = x0.detach().clone()
    rows = []
    final_target = route[-1]
    h_targets = [chosen_route["entry_h"]] + chosen_route["waypoints_h"]
    first_final = -1
    for step in range(total_steps + 1):
        if step < entry_steps:
            stage = 0
        else:
            remaining = max(1, total_steps - entry_steps)
            stage = 1 + min(len(h_targets) - 2, int((step - entry_steps) / max(1, remaining / max(1, len(h_targets) - 1))))
        current_target = route[min(stage, len(route) - 1)]
        st = eval_state(model, x, source, final_target, current_target)
        if st["final_target_success"] and first_final < 0:
            first_final = step
        st.update(
            {
                "step": step,
                "phase": "hidden_waypoint",
                "method": "entry_then_waypoints",
                "current_target": current_target,
                "current_stage": stage,
                "stage_hit_count": int(st["pred"] in route),
                "first_final_target_step": first_final,
            }
        )
        rows.append(st)
        if step == total_steps:
            break
        x = hidden_target_step(model, x, x0, h_targets[stage], eps, step_size)
    return rows, x


def summarize(df):
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
            mean_entry_distance=("entry_distance", "mean"),
            mean_steps_to_target=("first_final_target_step", lambda x: np.mean([v for v in x if v >= 0]) if any(v >= 0 for v in x) else np.nan),
        )
        .reset_index()
    )


def make_plot(summary, out):
    order = ["direct", "strict_sequence", "entry_then_strict_sequence", "entry_then_waypoints"]
    summary = summary.set_index("method").reindex(order).dropna(how="all").reset_index()
    labels = {
        "direct": "direct CE",
        "strict_sequence": "strict CE route",
        "entry_then_strict_sequence": "entry + CE route",
        "entry_then_waypoints": "entry + waypoints",
    }
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    axes[0].bar(x, summary.target_asr, color="#4C78A8")
    axes[0].set_ylabel("target ASR")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(x, summary.mean_target_margin, color="#F58518")
    axes[1].axhline(0, color="black", lw=1, ls="--")
    axes[1].set_ylabel("mean target margin")
    axes[2].bar(x, summary.mean_steps_to_target, color="#54A24B")
    axes[2].set_ylabel("mean steps to target")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(m, m) for m in summary.method], rotation=20, ha="right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "road_entry_route_attack_summary.png", dpi=220)
    fig.savefig(out / "road_entry_route_attack_summary.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/road_entry_route_attack_class0_to5_pilot")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--source-class", type=int, default=0)
    p.add_argument("--route", default="0,3,2,5")
    p.add_argument("--library-images", type=int, default=16)
    p.add_argument("--test-images", type=int, default=30)
    p.add_argument("--library-road-steps", type=int, default=220)
    p.add_argument("--road-step-l2", type=float, default=0.18)
    p.add_argument("--power-iters", type=int, default=3)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--entry-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    selected = select_images_class(dataset, model, args.library_images + args.test_images, device, args.source_class)
    library_rows = selected[: args.library_images]
    test_rows = selected[args.library_images :]
    route = [int(x) for x in args.route.split(",") if x.strip()]
    routes, route_scan = build_route_library(model, dataset, library_rows, route, args, device)
    route_scan.to_csv(out / "road_route_library_scan.csv", index=False)
    route_meta = [
        {
            "route_id": r["route_id"],
            "library_image_id": r["library_image_id"],
            "direction": r["direction"],
            "hit_steps": " ".join(map(str, r["hit_steps"])),
        }
        for r in routes
    ]
    pd.DataFrame(route_meta).to_csv(out / "road_route_library.csv", index=False)
    if not routes:
        raise RuntimeError(f"No road-library route found for {route}")

    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    all_rows = []
    for image_id, label in test_rows:
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(device)
        chosen, entry_dist = choose_route(model, x0, routes)
        method_runs = []
        rows, _ = run_direct(model, x0, label, route[-1], eps, step_size, args.steps)
        method_runs.append(("direct", rows))
        rows, _x, _hits = run_strict_sequence(model, x0, label, route[1:], eps, step_size, args.steps)
        method_runs.append(("strict_sequence", rows))
        rows, _ = run_entry_then_strict_sequence(model, x0, label, route, chosen, eps, step_size, args.steps, args.entry_steps)
        method_runs.append(("entry_then_strict_sequence", rows))
        rows, _ = run_waypoint_route(model, x0, label, route, chosen, eps, step_size, args.steps, args.entry_steps)
        method_runs.append(("entry_then_waypoints", rows))
        for method, rows in method_runs:
            for r in rows:
                r.update(
                    {
                        "image_id": image_id,
                        "label": label,
                        "method": method,
                        "source_class": args.source_class,
                        "route": args.route,
                        "chosen_route_id": chosen["route_id"],
                        "chosen_library_image_id": chosen["library_image_id"],
                        "chosen_direction": chosen["direction"],
                        "entry_distance": entry_dist,
                    }
                )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "road_entry_route_attack_timeseries.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "road_entry_route_attack_summary.csv", index=False)
    make_plot(summary, out)
    (out / "metadata.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Found {len(routes)} route-library entries for {route}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
