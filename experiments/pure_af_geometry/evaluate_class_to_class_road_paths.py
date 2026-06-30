#!/usr/bin/env python3
"""Continuous road-metric paths between class regions.

This experiment removes the exact adversarial endpoint from the target.  Each
target class is represented by clean class prototypes/centroids in hidden
representation space.  A source image then travels toward a target "city" using
only representation distance and local Jacobian mobility.

No margin, CE loss, attack trajectory, or adversarial endpoint is used for path
planning.  Prediction labels are used only for selecting clean prototypes and
for evaluating where the path ends.
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
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.evaluate_astar_highway_from_start import eval_state, feature_tensor  # noqa: E402
from experiments.pure_af_geometry.evaluate_continuous_road_metric_paths import (  # noqa: E402
    direct_diagnostic,
    run_metric_mpc,
    run_target_geodesic,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def feature(wrapper, x: torch.Tensor, layer: str) -> torch.Tensor:
    with torch.no_grad():
        return feature_tensor(wrapper, x, layer).detach()


def collect_clean_prototypes(
    wrapper,
    dataset,
    layer: str,
    n_per_class: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], pd.DataFrame]:
    feats: dict[int, list[np.ndarray]] = {c: [] for c in range(10)}
    pixels: dict[int, list[np.ndarray]] = {c: [] for c in range(10)}
    rows = []
    done = set()
    for start in range(0, len(dataset), batch_size):
        xs, ys, idxs = [], [], []
        for idx in range(start, min(start + batch_size, len(dataset))):
            x, y = dataset[idx]
            if y in done:
                continue
            xs.append(x)
            ys.append(int(y))
            idxs.append(idx)
        if not xs:
            if len(done) == 10:
                break
            continue
        xb = torch.stack(xs).to(device)
        with torch.no_grad():
            logits, fdict, _raw = wrapper.forward_with_features(xb)
            preds = logits.argmax(1).detach().cpu().numpy()
            h = fdict[layer].detach().cpu().numpy().astype(np.float32)
        for j, (idx, y, pred) in enumerate(zip(idxs, ys, preds)):
            if int(pred) != int(y):
                continue
            feats[y].append(h[j])
            pixels[y].append(xs[j].numpy().astype(np.float32))
            rows.append({"dataset_idx": int(idx), "label": int(y), "pred": int(pred), "prototype_idx": len(feats[y]) - 1})
            if len(feats[y]) >= n_per_class:
                done.add(y)
        if len(done) == 10:
            break
    missing = [c for c in range(10) if len(feats[c]) == 0]
    if missing:
        raise RuntimeError(f"No clean prototypes for classes {missing}")
    centroids = {c: np.stack(feats[c], axis=0).mean(axis=0).astype(np.float32) for c in range(10)}
    arrays = {c: np.stack(feats[c], axis=0).astype(np.float32) for c in range(10)}
    pix_arrays = {c: np.stack(pixels[c], axis=0).astype(np.float32) for c in range(10)}
    meta = pd.DataFrame(rows)
    for c in range(10):
        meta.loc[meta.label == c, "n_for_class"] = len(feats[c])
    return centroids, arrays, pix_arrays, meta


def target_feature_for_source(
    h0: np.ndarray,
    target_class: int,
    centroids: dict[int, np.ndarray],
    prototypes: dict[int, np.ndarray],
    mode: str,
) -> tuple[np.ndarray, int]:
    if mode == "centroid":
        return centroids[target_class], -1
    if mode == "nearest_prototype":
        arr = prototypes[target_class]
        d = np.linalg.norm(arr - h0.reshape(1, -1), axis=1)
        j = int(np.argmin(d))
        return arr[j], j
    raise ValueError(f"Unknown target mode {mode}")


def parse_targets(label: int, spec: str) -> list[int]:
    if spec == "next":
        return [(label + 1) % 10]
    if spec == "all":
        return [c for c in range(10) if c != label]
    if spec.startswith("fixed:"):
        t = int(spec.split(":", 1)[1])
        return [t] if t != label else []
    return [int(x.strip()) for x in spec.split(",") if x.strip() and int(x.strip()) != label]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["method", "target_mode", "target_spec"], dropna=False)
        .agg(
            n=("image_ord", "size"),
            target_hit_rate=("target_hit", "mean"),
            class_changed_rate=("class_changed", "mean"),
            mean_fraction_closed=("fraction_goal_distance_closed", "mean"),
            median_fraction_closed=("fraction_goal_distance_closed", "median"),
            mean_final_target_dist=("endpoint_feature_dist", "mean"),
            median_final_target_dist=("endpoint_feature_dist", "median"),
            mean_path_cost=("path_cost", "mean"),
            median_path_cost=("path_cost", "median"),
            mean_direct_cost=("direct_road_cost", "mean"),
            mean_linf=("linf", "mean"),
        )
        .reset_index()
        .sort_values(["target_hit_rate", "mean_fraction_closed"], ascending=[False, False])
    )


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=180)
    order = summary.sort_values(["target_hit_rate", "mean_fraction_closed"], ascending=False)
    labels = order["method"] + "\n" + order["target_mode"]
    axes[0].bar(labels, order["target_hit_rate"], color="#4c78a8")
    axes[0].set_ylabel("target class hit rate")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, order["mean_fraction_closed"], color="#59a14f")
    axes[1].set_ylabel("mean fraction of target-region distance closed")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "class_to_class_road_paths_summary.png")
    fig.savefig(out_dir / "class_to_class_road_paths_summary.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/attack_independent_road_graph_bbb_resnet50_c100"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/class_to_class_road_paths_bbb_resnet50_c100"))
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--targets", default="next", help="next, all, fixed:C, or comma-separated class IDs.")
    p.add_argument("--target-modes", default="centroid,nearest_prototype")
    p.add_argument("--prototypes-per-class", type=int, default=80)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--path-steps", default="3,5,10")
    p.add_argument("--mpc-candidates", type=int, default=8)
    p.add_argument("--mpc-noise-scale", type=float, default=0.75)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    wrapper = load_model(args.model, device).eval()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    nodes = pd.read_csv(args.graph_dir / "road_graph_nodes.csv")
    arrays = np.load(args.graph_dir / "road_graph_arrays.npz")
    pixels = arrays["pixels"].astype(np.float32)
    clean_nodes = nodes[nodes.state_kind == "clean"].copy().sort_values("image_ord")
    if args.images > 0:
        clean_nodes = clean_nodes.head(args.images)
    centroids, prototypes, prototype_pixels, prototype_meta = collect_clean_prototypes(
        wrapper, dataset, args.layer, args.prototypes_per_class, args.batch_size, device
    )
    prototype_meta.to_csv(args.output_dir / "class_prototype_metadata.csv", index=False)
    np.savez_compressed(
        args.output_dir / "class_prototypes.npz",
        **{f"class_{c}_features": prototypes[c] for c in range(10)},
        **{f"class_{c}_pixels": prototype_pixels[c] for c in range(10)},
        **{f"class_{c}_centroid": centroids[c] for c in range(10)},
    )
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rows = []
    path_rows = []
    endpoint_rows = []
    target_modes = [x.strip() for x in args.target_modes.split(",") if x.strip()]

    for i, row in enumerate(clean_nodes.itertuples(index=False), start=1):
        x0 = torch.as_tensor(pixels[int(row.node_id):int(row.node_id) + 1], dtype=torch.float32, device=device)
        y = torch.tensor([int(row.label)], device=device)
        h0_t = feature(wrapper, x0, args.layer)
        h0 = h0_t.detach().cpu().numpy()[0].astype(np.float32)
        source_pred = int(eval_state(wrapper, x0, x0, y)["pred"])
        for target_class in parse_targets(int(row.label), args.targets):
            for target_mode in target_modes:
                h_goal_np, proto_idx = target_feature_for_source(h0, target_class, centroids, prototypes, target_mode)
                h_goal = torch.as_tensor(h_goal_np.reshape(1, -1), dtype=torch.float32, device=device)
                direct = direct_diagnostic(wrapper, x0, h0_t, h_goal, args.layer)
                endpoint_rows.append(
                    {
                        "image_ord": int(row.image_ord),
                        "node_id": int(row.node_id),
                        "source_label": int(row.label),
                        "source_pred": source_pred,
                        "target_class": int(target_class),
                        "target_mode": target_mode,
                        "prototype_idx": int(proto_idx),
                        **direct,
                    }
                )
                for steps in parse_int_csv(args.path_steps):
                    for method_name, runner in [("target_geodesic", run_target_geodesic), ("metric_mpc", run_metric_mpc)]:
                        if method_name == "target_geodesic":
                            _x, metrics, pr = runner(wrapper, x0, y, args.layer, h_goal, eps, step_size, steps)
                        else:
                            _x, metrics, pr = runner(
                                wrapper,
                                x0,
                                y,
                                args.layer,
                                h_goal,
                                eps,
                                step_size,
                                steps,
                                args.mpc_candidates,
                                args.mpc_noise_scale,
                                gen,
                            )
                        final_pred = int(metrics["pred"])
                        method = f"{method_name}_{steps}"
                        rows.append(
                            {
                                "image_ord": int(row.image_ord),
                                "node_id": int(row.node_id),
                                "source_label": int(row.label),
                                "source_pred": source_pred,
                                "target_class": int(target_class),
                                "target_spec": args.targets,
                                "target_mode": target_mode,
                                "prototype_idx": int(proto_idx),
                                "method": method,
                                "path_steps": int(steps),
                                "target_hit": int(final_pred == int(target_class)),
                                "class_changed": int(final_pred != int(row.label)),
                                **direct,
                                **metrics,
                            }
                        )
                        for r in pr:
                            r.update(
                                {
                                    "image_ord": int(row.image_ord),
                                    "source_label": int(row.label),
                                    "target_class": int(target_class),
                                    "target_mode": target_mode,
                                    "method": method,
                                    "path_steps": int(steps),
                                }
                            )
                        path_rows.extend(pr)
        if i % 20 == 0:
            print(f"[{i}/{len(clean_nodes)}] rows={len(rows)}", flush=True)

    per = pd.DataFrame(rows)
    path = pd.DataFrame(path_rows)
    endpoints = pd.DataFrame(endpoint_rows)
    summary = summarize(per)
    per.to_csv(args.output_dir / "class_to_class_road_paths_per_image.csv", index=False)
    path.to_csv(args.output_dir / "class_to_class_road_path_steps.csv", index=False)
    endpoints.to_csv(args.output_dir / "class_to_class_targets.csv", index=False)
    summary.to_csv(args.output_dir / "class_to_class_road_paths_summary.csv", index=False)
    plot_summary(summary, args.output_dir)
    metadata = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    metadata.update(
        {
            "planning_uses": ["target class clean representation prototype/centroid", "local Jacobian mobility"],
            "planning_excludes": ["margin", "CE loss", "attack trajectories", "adversarial endpoints"],
        }
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Class-to-Class Continuous Road Paths",
        "",
        "Planner steps use clean target-class representation regions and local Jacobian mobility. No margins, losses, attack trajectories, or adversarial endpoints are used during planning.",
        "",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"- `{r.method}`/{r.target_mode}: target_hit={r.target_hit_rate:.3f}, "
            f"class_changed={r.class_changed_rate:.3f}, closed={r.mean_fraction_closed:.3f}, "
            f"path_cost={r.mean_path_cost:.4f}"
        )
    (args.output_dir / "class_to_class_road_paths_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
