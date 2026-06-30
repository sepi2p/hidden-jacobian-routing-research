#!/usr/bin/env python3
"""Add adversarial endpoints to a frozen attack-independent road graph.

The road graph itself is built without attacks, margins, or success labels.
This script uses the frozen graph, generates adversarial endpoint images, adds
those endpoint representations as target nodes, and asks whether low-cost
Jacobian road paths connect clean nodes to their adversarial endpoints.

Important: adversarial endpoints are used only as targets after the graph is
frozen.  Edge weights still use representation distance and Jacobian mobility,
not margins or attack success.
"""

from __future__ import annotations

import argparse
import heapq
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
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.build_attack_independent_road_graph import local_mobility  # noqa: E402
from experiments.pure_af_geometry.evaluate_astar_highway_from_start import eval_state, feature_tensor  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import project_linf  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ce_step(wrapper, x0: torch.Tensor, x: torch.Tensor, y: torch.Tensor, eps: float, step_size: float) -> torch.Tensor:
    probe = x.detach().requires_grad_(True)
    logits = wrapper(probe)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, probe)[0]
    return project_linf(x + step_size * grad.sign(), x0, eps).detach()


def pgd_endpoint(wrapper, x0: torch.Tensor, y: torch.Tensor, eps: float, step_size: float, steps: int) -> tuple[torch.Tensor, dict]:
    x = x0.detach()
    states = []
    for _ in range(steps):
        x = ce_step(wrapper, x0, x, y, eps, step_size)
        ev = eval_state(wrapper, x0, x, y)
        states.append(ev)
        if int(ev["success"]):
            break
    final = eval_state(wrapper, x0, x, y)
    final["pgd_steps_used"] = len(states)
    return x.detach(), final


def feature(wrapper, x: torch.Tensor, layer: str) -> torch.Tensor:
    with torch.no_grad():
        return feature_tensor(wrapper, x, layer).detach()


def project_coords(h: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (h.reshape(1, -1) - mean.reshape(1, -1)) @ components.T


def build_adjacency(edges: pd.DataFrame, n_nodes: int, undirected: bool) -> list[list[tuple[int, float, int]]]:
    adj: list[list[tuple[int, float, int]]] = [[] for _ in range(n_nodes)]
    for edge_idx, r in enumerate(edges.itertuples(index=False)):
        src = int(r.src)
        dst = int(r.dst)
        cost = float(r.road_cost)
        if not np.isfinite(cost) or cost <= 0:
            continue
        adj[src].append((dst, cost, edge_idx))
        if undirected:
            adj[dst].append((src, cost, edge_idx))
    return adj


def dijkstra(adj: list[list[tuple[int, float, int]]], src: int, dst: int) -> tuple[float, list[int], list[int]]:
    n = len(adj)
    dist = [float("inf")] * n
    prev = [-1] * n
    prev_edge = [-1] * n
    dist[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        if u == dst:
            break
        for v, w, eidx in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                prev_edge[v] = eidx
                heapq.heappush(heap, (nd, v))
    if not np.isfinite(dist[dst]):
        return float("inf"), [], []
    path = []
    edge_path = []
    cur = dst
    while cur != -1:
        path.append(cur)
        if prev_edge[cur] != -1:
            edge_path.append(prev_edge[cur])
        cur = prev[cur]
    path.reverse()
    edge_path.reverse()
    return dist[dst], path, edge_path


def summarize_path_edges(edge_path: list[int], edges_aug: pd.DataFrame) -> dict:
    if not edge_path:
        return {
            "path_road_score_mean": np.nan,
            "path_road_score_min": np.nan,
            "path_feature_dist_sum": np.nan,
            "path_n_endpoint_edges": 0,
            "path_n_original_edges": 0,
        }
    sub = edges_aug.iloc[edge_path]
    return {
        "path_road_score_mean": float(sub["road_score"].mean()),
        "path_road_score_min": float(sub["road_score"].min()),
        "path_feature_dist_sum": float(sub["feature_dist"].sum()),
        "path_n_endpoint_edges": int((sub["edge_kind"] == "endpoint_connection").sum()),
        "path_n_original_edges": int((sub["edge_kind"] != "endpoint_connection").sum()),
    }


def plot_summary(per: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=180)
    ok = per[per["adv_success"] == 1].copy()
    axes[0].hist(ok["path_cost"].replace([np.inf, -np.inf], np.nan).dropna(), bins=24, color="#4c78a8", alpha=0.85)
    axes[0].set_title("Shortest road cost to adversarial endpoint")
    axes[0].set_xlabel("path cost")
    axes[0].set_ylabel("images")
    ratio = ok["direct_cost"] / ok["path_cost"].replace(0, np.nan)
    axes[1].hist(ratio.replace([np.inf, -np.inf], np.nan).dropna(), bins=24, color="#59a14f", alpha=0.85)
    axes[1].axvline(1.0, color="black", linewidth=1)
    axes[1].set_title("Direct clean->adv cost / graph path cost")
    axes[1].set_xlabel("ratio")
    axes[1].set_ylabel("images")
    fig.tight_layout()
    fig.savefig(out_dir / "road_graph_paths_to_adversarial_nodes.png")
    fig.savefig(out_dir / "road_graph_paths_to_adversarial_nodes.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/attack_independent_road_graph_bbb_resnet50_c100"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/road_graph_paths_to_adversarial_nodes_bbb_resnet50_c100"))
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--pgd-steps", type=int, default=5)
    p.add_argument("--connect-k", type=int, default=12)
    p.add_argument("--exclude-clean-endpoint-edge", action="store_true")
    p.add_argument("--undirected", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    wrapper = load_model(args.model, device).eval()
    nodes = pd.read_csv(args.graph_dir / "road_graph_nodes.csv")
    edges = pd.read_csv(args.graph_dir / "road_graph_edges.csv")
    arrays = np.load(args.graph_dir / "road_graph_arrays.npz")
    if "pixels" not in arrays.files:
        raise RuntimeError(f"{args.graph_dir}/road_graph_arrays.npz does not contain pixels. Rebuild the road graph with the patched builder.")
    features = arrays["features"].astype(np.float32)
    pixels = arrays["pixels"].astype(np.float32)
    coords = arrays["coords"].astype(np.float32)
    pca_mean = arrays["pca_mean"].astype(np.float32)
    pca_components = arrays["pca_components"].astype(np.float32)

    clean_nodes = nodes[nodes.state_kind == "clean"].copy().sort_values("image_ord")
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    nn = NearestNeighbors(n_neighbors=min(args.connect_k, len(nodes)), metric="euclidean").fit(coords)

    per_rows = []
    adv_node_rows = []
    endpoint_edge_rows = []
    adv_features = []
    adv_coords = []
    direct_lookup = {}
    base_n = len(nodes)
    base_edges = edges.copy()
    base_edges["edge_kind"] = "knn_jacobian"

    for j, row in enumerate(clean_nodes.itertuples(index=False)):
        clean_node = int(row.node_id)
        x0 = torch.as_tensor(pixels[clean_node:clean_node + 1], dtype=torch.float32, device=device)
        y = torch.tensor([int(row.label)], device=device)
        x_adv, ev = pgd_endpoint(wrapper, x0, y, eps, step_size, args.pgd_steps)
        h_adv = feature(wrapper, x_adv, args.layer).detach().cpu().numpy()[0].astype(np.float32)
        z_adv = project_coords(h_adv, pca_mean, pca_components)[0].astype(np.float32)
        adv_id = base_n + j
        adv_features.append(h_adv)
        adv_coords.append(z_adv)
        adv_node_rows.append(
            {
                "node_id": adv_id,
                "image_ord": int(row.image_ord),
                "dataset_idx": int(row.dataset_idx),
                "label": int(row.label),
                "state_kind": "pgd_adversarial_endpoint" if int(ev["success"]) else "pgd_nonadversarial_endpoint",
                "step": int(ev["pgd_steps_used"]),
                "adv_success": int(ev["success"]),
                "adv_pred": int(ev["pred"]),
                "linf": float(ev["linf"]),
                "pc1": float(z_adv[0]) if len(z_adv) > 0 else np.nan,
                "pc2": float(z_adv[1]) if len(z_adv) > 1 else np.nan,
                "pc3": float(z_adv[2]) if len(z_adv) > 2 else np.nan,
            }
        )

        dist, neigh = nn.kneighbors(z_adv.reshape(1, -1))
        direct_dh = h_adv - features[clean_node]
        if np.linalg.norm(direct_dh) > 1e-12:
            x_clean = torch.as_tensor(pixels[clean_node:clean_node + 1], dtype=torch.float32, device=device)
            direct_mob_l2, _direct_mob_l1, _direct_mob_linf = local_mobility(wrapper, x_clean, args.layer, direct_dh)
            direct_feature_dist = float(np.linalg.norm(direct_dh))
            direct_cost = direct_feature_dist / max(direct_mob_l2, 1e-12)
            direct_score = direct_mob_l2 / max(direct_feature_dist, 1e-12)
        else:
            direct_cost = np.nan
            direct_score = np.nan
        direct_lookup[int(adv_id)] = (direct_cost, direct_score)

        candidates = [int(n) for n in neigh[0] if int(n) != clean_node or not args.exclude_clean_endpoint_edge]
        if not args.exclude_clean_endpoint_edge:
            candidates = [clean_node] + candidates
        candidates = list(dict.fromkeys(candidates))
        for src in candidates:
            dh = h_adv - features[src]
            if np.linalg.norm(dh) < 1e-12:
                continue
            x_src = torch.as_tensor(pixels[src:src + 1], dtype=torch.float32, device=device)
            mob_l2, mob_l1, mob_linf = local_mobility(wrapper, x_src, args.layer, dh)
            feature_dist = float(np.linalg.norm(dh))
            pc_dist = float(np.linalg.norm(z_adv - coords[src]))
            road_cost = feature_dist / max(mob_l2, 1e-12)
            road_score = mob_l2 / max(feature_dist, 1e-12)
            endpoint_edge_rows.append(
                {
                    "src": int(src),
                    "dst": int(adv_id),
                    "edge_kind": "endpoint_connection",
                    "src_kind": str(nodes.loc[src, "state_kind"]),
                    "dst_kind": "pgd_adversarial_endpoint" if int(ev["success"]) else "pgd_nonadversarial_endpoint",
                    "same_image": int(nodes.loc[src, "image_ord"] == int(row.image_ord)),
                    "feature_dist": feature_dist,
                    "pc_dist": pc_dist,
                    "mobility_l2": mob_l2,
                    "mobility_l1": mob_l1,
                    "mobility_linf": mob_linf,
                    "road_cost": road_cost,
                    "road_score": road_score,
                    "target_image_ord": int(row.image_ord),
                    "target_adv_success": int(ev["success"]),
                }
            )

    adv_nodes = pd.DataFrame(adv_node_rows)
    endpoint_edges = pd.DataFrame(endpoint_edge_rows)
    edges_aug = pd.concat([base_edges, endpoint_edges], ignore_index=True)
    total_nodes = base_n + len(adv_nodes)
    adj = build_adjacency(edges_aug, total_nodes, args.undirected)

    for adv in adv_nodes.itertuples(index=False):
        clean_node = int(clean_nodes[clean_nodes.image_ord == int(adv.image_ord)]["node_id"].iloc[0])
        cost, path, edge_path = dijkstra(adj, clean_node, int(adv.node_id))
        extra = summarize_path_edges(edge_path, edges_aug)
        direct_cost, direct_score = direct_lookup.get(int(adv.node_id), (np.nan, np.nan))
        per_rows.append(
            {
                "image_ord": int(adv.image_ord),
                "clean_node": clean_node,
                "adv_node": int(adv.node_id),
                "label": int(adv.label),
                "adv_success": int(adv.adv_success),
                "adv_pred": int(adv.adv_pred),
                "adv_pgd_steps": int(adv.step),
                "path_found": int(len(path) > 0),
                "path_cost": float(cost),
                "path_hops": int(max(len(path) - 1, 0)),
                "path_nodes": "|".join(map(str, path)),
                "path_edges": "|".join(map(str, edge_path)),
                "direct_cost": direct_cost,
                "direct_road_score": direct_score,
                "direct_over_path_cost": float(direct_cost / cost) if np.isfinite(cost) and cost > 0 and np.isfinite(direct_cost) else np.nan,
                **extra,
            }
        )

    per = pd.DataFrame(per_rows)
    success = per[per.adv_success == 1]
    success_found = success[(success.path_found == 1) & np.isfinite(success.path_cost)]
    summary = pd.DataFrame(
        [
            {
                "n_images": int(len(per)),
                "n_adv_success": int(per.adv_success.sum()),
                "adv_asr": float(per.adv_success.mean()),
                "path_found_frac_success_adv": float(success.path_found.mean()) if len(success) else np.nan,
                "n_success_paths_found": int(len(success_found)),
                "mean_path_cost_success_adv": float(success_found["path_cost"].mean()) if len(success_found) else np.nan,
                "median_path_cost_success_adv": float(success_found["path_cost"].median()) if len(success_found) else np.nan,
                "mean_direct_cost_success_adv": float(success["direct_cost"].mean()) if len(success) else np.nan,
                "median_direct_cost_success_adv": float(success["direct_cost"].median()) if len(success) else np.nan,
                "mean_direct_over_path_cost_success_adv": float(success_found["direct_over_path_cost"].mean()) if len(success_found) else np.nan,
                "median_direct_over_path_cost_success_adv": float(success_found["direct_over_path_cost"].median()) if len(success_found) else np.nan,
                "mean_path_hops_success_adv": float(success_found["path_hops"].mean()) if len(success_found) else np.nan,
                "median_path_hops_success_adv": float(success_found["path_hops"].median()) if len(success_found) else np.nan,
            }
        ]
    )

    adv_nodes.to_csv(args.output_dir / "adversarial_endpoint_nodes.csv", index=False)
    endpoint_edges.to_csv(args.output_dir / "adversarial_endpoint_edges.csv", index=False)
    per.to_csv(args.output_dir / "road_graph_paths_to_adversarial_nodes.csv", index=False)
    summary.to_csv(args.output_dir / "road_graph_paths_to_adversarial_summary.csv", index=False)
    if len(adv_features):
        np.savez_compressed(
            args.output_dir / "adversarial_endpoint_arrays.npz",
            features=np.stack(adv_features).astype(np.float32),
            coords=np.stack(adv_coords).astype(np.float32),
        )
    plot_summary(per, args.output_dir)
    metadata = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    metadata.update(
        {
            "graph_dir": str(args.graph_dir),
            "edge_weights_use": "representation distance and local Jacobian mobility only",
            "adversarial_use": "endpoints only after graph freeze",
            "n_base_nodes": int(base_n),
            "n_endpoint_nodes": int(len(adv_nodes)),
            "n_endpoint_edges": int(len(endpoint_edges)),
        }
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Road-Graph Paths to Adversarial Endpoints",
        "",
        "The base road graph was constructed without adversarial information. PGD endpoints are added only as target nodes after the graph is frozen.",
        "",
    ]
    r = summary.iloc[0]
    lines += [
        f"- images: {int(r.n_images)}",
        f"- PGD endpoint ASR: {r.adv_asr:.3f}",
        f"- path found among successful endpoints: {r.path_found_frac_success_adv:.3f}",
        f"- median shortest path cost: {r.median_path_cost_success_adv:.4f}",
        f"- median direct clean-to-adv cost: {r.median_direct_cost_success_adv:.4f}",
        f"- median direct/path cost ratio: {r.median_direct_over_path_cost_success_adv:.3f}",
        f"- median path hops: {r.median_path_hops_success_adv:.1f}",
        "",
        "If direct/path ratio is greater than one, the graph road network offers a lower-cost multi-edge route than the direct local move from clean to adversarial endpoint.",
    ]
    (args.output_dir / "road_graph_paths_to_adversarial_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
