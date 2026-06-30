#!/usr/bin/env python3
"""Build an attack-independent representation road graph.

The graph construction uses no adversarial trajectories, no attack success
labels, and no margins.  It samples reachable image states using clean images,
random pixel walks, random L_inf perturbations, and mobility-only walks.  Nodes
are hidden representations.  Edges connect nearby nodes in a low-dimensional
feature chart and are weighted by local Jacobian mobility:

    mobility_x(dh) = || grad_x <h(x), dh / ||dh||> ||

High-mobility, short-distance edges form the empirical "roads/highways" of the
representation map.  Attack overlays should be run in a separate script after
this graph is frozen.
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
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.evaluate_astar_highway_from_start import feature_tensor  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import project_linf  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature(wrapper, x: torch.Tensor, layer: str) -> torch.Tensor:
    with torch.no_grad():
        return feature_tensor(wrapper, x, layer).detach()


def random_sign_step(x0: torch.Tensor, x: torch.Tensor, eps: float, step_size: float, gen: torch.Generator) -> torch.Tensor:
    direction = torch.randn(x.shape, generator=gen, device=x.device).sign()
    return project_linf(x + step_size * direction, x0, eps).detach()


def random_linf_state(x0: torch.Tensor, eps: float, gen: torch.Generator) -> torch.Tensor:
    noise = (torch.rand(x0.shape, generator=gen, device=x0.device) * 2.0 - 1.0) * eps
    return (x0 + noise).clamp(0, 1).detach()


def pullback_step(wrapper, x0: torch.Tensor, x: torch.Tensor, layer: str, direction: torch.Tensor, eps: float, step_size: float) -> torch.Tensor:
    probe = x.detach().requires_grad_(True)
    h = feature_tensor(wrapper, probe, layer)
    scalar = (h * direction.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, probe)[0]
    return project_linf(x + step_size * grad.sign(), x0, eps).detach()


def mobility_step(
    wrapper,
    x0: torch.Tensor,
    x: torch.Tensor,
    layer: str,
    eps: float,
    step_size: float,
    candidates: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor, float]:
    h0 = feature(wrapper, x, layer)
    best_x = None
    best_speed = -1.0
    d = h0.numel()
    for _ in range(candidates):
        v = torch.randn((1, d), generator=gen, device=x.device)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        x_next = pullback_step(wrapper, x0, x, layer, v, eps, step_size)
        h1 = feature(wrapper, x_next, layer)
        speed = float(torch.norm(h1 - h0, dim=1).item())
        if speed > best_speed:
            best_speed = speed
            best_x = x_next
    return best_x.detach(), best_speed


def local_mobility(wrapper, x: torch.Tensor, layer: str, dh: np.ndarray) -> tuple[float, float, float]:
    dh_t = torch.as_tensor(dh, dtype=torch.float32, device=x.device).view(1, -1)
    dh_t = dh_t / dh_t.norm(dim=1, keepdim=True).clamp_min(1e-12)
    probe = x.detach().requires_grad_(True)
    h = feature_tensor(wrapper, probe, layer)
    scalar = (h * dh_t.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, probe)[0]
    return (
        float(torch.norm(grad.flatten(1), p=2, dim=1).item()),
        float(torch.norm(grad.flatten(1), p=1, dim=1).item()),
        float(grad.abs().max().item()),
    )


def clean_indices(dataset, wrapper, device: torch.device, n: int, batch_size: int, clean_correct: bool) -> list[int]:
    if not clean_correct:
        return list(range(n))
    out = []
    for start in range(0, len(dataset), batch_size):
        xs = []
        ys = []
        idxs = []
        for idx in range(start, min(start + batch_size, len(dataset))):
            x, y = dataset[idx]
            xs.append(x)
            ys.append(y)
            idxs.append(idx)
        xb = torch.stack(xs).to(device)
        with torch.no_grad():
            pred = wrapper(xb).argmax(1).detach().cpu().numpy()
        for idx, y, p in zip(idxs, ys, pred):
            if int(y) == int(p):
                out.append(idx)
                if len(out) >= n:
                    return out
    return out


def add_node(rows, xs, feats, node_id: int, image_ord: int, dataset_idx: int, label: int, kind: str, step: int, x: torch.Tensor, h: torch.Tensor) -> int:
    rows.append(
        {
            "node_id": node_id,
            "image_ord": image_ord,
            "dataset_idx": dataset_idx,
            "label": label,
            "state_kind": kind,
            "step": step,
            "pixel_norm_l2": float(torch.norm(x.flatten(1), dim=1).item()),
            "feature_norm_l2": float(torch.norm(h.flatten(1), dim=1).item()),
        }
    )
    xs.append(x.detach().cpu())
    feats.append(h.detach().cpu().numpy()[0].astype(np.float32))
    return node_id + 1


def union_find_components(n: int, edges: pd.DataFrame, score_col: str, q: float) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    threshold = float(edges[score_col].quantile(q))
    parent = list(range(n))
    size = [1] * n

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for r in edges[edges[score_col] >= threshold].itertuples(index=False):
        union(int(r.src), int(r.dst))
    roots = [find(i) for i in range(n)]
    comp = pd.Series(roots).value_counts().reset_index()
    comp.columns = ["component_root", "n_nodes"]
    comp["highway_quantile"] = q
    comp["score_threshold"] = threshold
    return comp.sort_values("n_nodes", ascending=False)


def plot_graph(nodes: pd.DataFrame, edges: pd.DataFrame, out_dir: Path, max_edges: int = 2500) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.3), dpi=220)
    if len(edges):
        draw = edges.sort_values("road_score", ascending=False).head(max_edges)
        norm = plt.Normalize(draw["road_score"].min(), draw["road_score"].max())
        cmap = plt.cm.viridis
        for r in draw.itertuples(index=False):
            x = [nodes.loc[int(r.src), "pc1"], nodes.loc[int(r.dst), "pc1"]]
            y = [nodes.loc[int(r.src), "pc2"], nodes.loc[int(r.dst), "pc2"]]
            ax.plot(x, y, color=cmap(norm(float(r.road_score))), alpha=0.18, linewidth=0.6)
    colors = {
        "clean": "#222222",
        "random_walk": "#4c78a8",
        "random_linf": "#f58518",
        "mobility_walk": "#54a24b",
    }
    for kind, sub in nodes.groupby("state_kind"):
        ax.scatter(sub["pc1"], sub["pc2"], s=10, alpha=0.75, label=kind, color=colors.get(kind, None))
    ax.set_xlabel("road-map PC1")
    ax.set_ylabel("road-map PC2")
    ax.set_title("Attack-independent representation road graph")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "attack_independent_road_graph_pc12.png")
    fig.savefig(out_dir / "attack_independent_road_graph_pc12.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/attack_independent_road_graph_bbb_resnet50_c100"))
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--clean-correct", action="store_true")
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--random-walk-steps", type=int, default=3)
    p.add_argument("--random-linf-samples", type=int, default=3)
    p.add_argument("--mobility-walk-steps", type=int, default=3)
    p.add_argument("--mobility-candidates", type=int, default=6)
    p.add_argument("--chart-dim", type=int, default=5)
    p.add_argument("--knn", type=int, default=6)
    p.add_argument("--edge-batch-limit", type=int, default=0, help="Optional cap on directed kNN edges for quick smoke tests.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    wrapper = load_model(args.model, device).eval()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    idxs = clean_indices(dataset, wrapper, device, args.images, args.batch_size, args.clean_correct)
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    rows, xs, feats = [], [], []
    temporal_edges = []
    node_id = 0
    gen = torch.Generator(device=device).manual_seed(args.seed)

    for image_ord, dataset_idx in enumerate(idxs):
        x_cpu, label = dataset[int(dataset_idx)]
        x0 = x_cpu.unsqueeze(0).to(device)
        h0 = feature(wrapper, x0, args.layer)
        clean_id = node_id
        node_id = add_node(rows, xs, feats, node_id, image_ord, int(dataset_idx), int(label), "clean", 0, x0, h0)

        x = x0.detach()
        prev = clean_id
        for step in range(1, args.random_walk_steps + 1):
            x = random_sign_step(x0, x, eps, step_size, gen)
            h = feature(wrapper, x, args.layer)
            cur = node_id
            node_id = add_node(rows, xs, feats, node_id, image_ord, int(dataset_idx), int(label), "random_walk", step, x, h)
            temporal_edges.append({"src": prev, "dst": cur, "edge_kind": "temporal_random_walk"})
            prev = cur

        for sample in range(1, args.random_linf_samples + 1):
            x = random_linf_state(x0, eps, gen)
            h = feature(wrapper, x, args.layer)
            cur = node_id
            node_id = add_node(rows, xs, feats, node_id, image_ord, int(dataset_idx), int(label), "random_linf", sample, x, h)
            temporal_edges.append({"src": clean_id, "dst": cur, "edge_kind": "radial_random_linf"})

        x = x0.detach()
        prev = clean_id
        for step in range(1, args.mobility_walk_steps + 1):
            x, _speed = mobility_step(wrapper, x0, x, args.layer, eps, step_size, args.mobility_candidates, gen)
            h = feature(wrapper, x, args.layer)
            cur = node_id
            node_id = add_node(rows, xs, feats, node_id, image_ord, int(dataset_idx), int(label), "mobility_walk", step, x, h)
            temporal_edges.append({"src": prev, "dst": cur, "edge_kind": "temporal_mobility_walk"})
            prev = cur

        if (image_ord + 1) % 20 == 0:
            print(f"[states] {image_ord + 1}/{len(idxs)} images, nodes={node_id}", flush=True)

    nodes = pd.DataFrame(rows)
    feat_arr = np.stack(feats, axis=0)
    pca = PCA(n_components=min(args.chart_dim, feat_arr.shape[0] - 1, feat_arr.shape[1]), random_state=args.seed)
    coords = pca.fit_transform(feat_arr)
    for j in range(coords.shape[1]):
        nodes[f"pc{j+1}"] = coords[:, j]

    nn = NearestNeighbors(n_neighbors=min(args.knn + 1, len(nodes)), metric="euclidean")
    nn.fit(coords)
    dists, inds = nn.kneighbors(coords)
    edge_rows = []
    for src in range(len(nodes)):
        x_src = xs[src].to(device)
        for dist, dst in zip(dists[src, 1:], inds[src, 1:]):
            dh = feat_arr[int(dst)] - feat_arr[int(src)]
            if np.linalg.norm(dh) < 1e-12:
                continue
            mob_l2, mob_l1, mob_linf = local_mobility(wrapper, x_src, args.layer, dh)
            feature_dist = float(np.linalg.norm(dh))
            pc_dist = float(dist)
            edge_rows.append(
                {
                    "src": int(src),
                    "dst": int(dst),
                    "edge_kind": "knn_jacobian",
                    "src_kind": str(nodes.loc[src, "state_kind"]),
                    "dst_kind": str(nodes.loc[int(dst), "state_kind"]),
                    "same_image": int(nodes.loc[src, "image_ord"] == nodes.loc[int(dst), "image_ord"]),
                    "feature_dist": feature_dist,
                    "pc_dist": pc_dist,
                    "mobility_l2": mob_l2,
                    "mobility_l1": mob_l1,
                    "mobility_linf": mob_linf,
                    "road_cost": feature_dist / max(mob_l2, 1e-12),
                    "road_score": mob_l2 / max(feature_dist, 1e-12),
                }
            )
            if args.edge_batch_limit > 0 and len(edge_rows) >= args.edge_batch_limit:
                break
        if args.edge_batch_limit > 0 and len(edge_rows) >= args.edge_batch_limit:
            break
        if (src + 1) % 200 == 0:
            print(f"[edges] {src + 1}/{len(nodes)} nodes, edges={len(edge_rows)}", flush=True)

    edges = pd.DataFrame(edge_rows)
    temporal = pd.DataFrame(temporal_edges)
    if not temporal.empty:
        temporal = temporal.merge(nodes[["node_id", "state_kind"]].rename(columns={"node_id": "src", "state_kind": "src_kind"}), on="src", how="left")
        temporal = temporal.merge(nodes[["node_id", "state_kind"]].rename(columns={"node_id": "dst", "state_kind": "dst_kind"}), on="dst", how="left")

    comps = union_find_components(len(nodes), edges, "road_score", 0.9)
    summaries = []
    if not edges.empty:
        summaries.append(
            {
                "n_images": len(idxs),
                "n_nodes": len(nodes),
                "n_knn_edges": len(edges),
                "n_temporal_edges": len(temporal),
                "pc_explained_var_1": float(pca.explained_variance_ratio_[0]),
                "pc_explained_var_top": float(pca.explained_variance_ratio_.sum()),
                "road_score_q50": float(edges["road_score"].quantile(0.5)),
                "road_score_q90": float(edges["road_score"].quantile(0.9)),
                "road_score_q95": float(edges["road_score"].quantile(0.95)),
                "road_cost_q50": float(edges["road_cost"].quantile(0.5)),
                "road_cost_q10": float(edges["road_cost"].quantile(0.1)),
                "largest_q90_component": int(comps["n_nodes"].max()) if not comps.empty else 0,
            }
        )
    summary = pd.DataFrame(summaries)

    nodes.to_csv(args.output_dir / "road_graph_nodes.csv", index=False)
    edges.to_csv(args.output_dir / "road_graph_edges.csv", index=False)
    temporal.to_csv(args.output_dir / "road_graph_temporal_edges.csv", index=False)
    comps.to_csv(args.output_dir / "road_graph_highway_components_q90.csv", index=False)
    summary.to_csv(args.output_dir / "road_graph_summary.csv", index=False)
    np.savez_compressed(
        args.output_dir / "road_graph_arrays.npz",
        features=feat_arr.astype(np.float32),
        pixels=torch.cat(xs, dim=0).numpy().astype(np.float32),
        coords=coords.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )
    plot_graph(nodes, edges, args.output_dir)

    metadata = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    metadata.update(
        {
            "device": str(device),
            "construction_excludes": ["PGD", "Square", "attack trajectories", "margins", "success labels"],
            "edge_weight": "Jacobian mobility only",
            "n_nodes": int(len(nodes)),
            "n_edges": int(len(edges)),
        }
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Attack-Independent Road Graph",
        "",
        "This graph is constructed without PGD, Square, margins, target classes, or adversarial success labels.",
        "",
    ]
    if not summary.empty:
        r = summary.iloc[0]
        lines += [
            f"- nodes: {int(r.n_nodes)}",
            f"- directed kNN Jacobian edges: {int(r.n_knn_edges)}",
            f"- temporal non-adversarial edges: {int(r.n_temporal_edges)}",
            f"- road-score median/q90/q95: {r.road_score_q50:.4f} / {r.road_score_q90:.4f} / {r.road_score_q95:.4f}",
            f"- largest q90 highway component: {int(r.largest_q90_component)} nodes",
            "",
            "Interpretation: high road-score edges are low-effort/high-mobility representation moves under the local hidden-layer Jacobian. Attack trajectories can be overlaid only after this graph is frozen.",
        ]
    (args.output_dir / "road_graph_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
