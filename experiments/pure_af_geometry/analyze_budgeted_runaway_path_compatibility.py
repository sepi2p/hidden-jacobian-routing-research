#!/usr/bin/env python3
"""Compare budgeted run-away routes with PGD/Square trajectory routes.

All paths are mapped into the same signed highway vocabulary.  For each
adversarial local step, we project the hidden transport vector onto the
highway basis and assign the signed route with largest absolute coefficient.
We then compare route usage and per-image route-set overlap against the
budgeted run-away planner's selected signed routes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.evaluate_image_conditioned_highway_selector import (  # noqa: E402
    ArtifactStore,
    fit_highway_basis,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def signed_route(pc_index: int, coeff: float) -> str:
    return f"pc{pc_index + 1}{'+' if coeff >= 0 else '-'}"


def route_vocab(k: int) -> list[str]:
    out = []
    for pc in range(1, k + 1):
        out.extend([f"pc{pc}+", f"pc{pc}-"])
    return out


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(float)
    q = q.astype(float)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cosine_sim(p: np.ndarray, q: np.ndarray) -> float:
    den = float(np.linalg.norm(p) * np.linalg.norm(q))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(p, q) / den)


def distribution_for_routes(routes: pd.Series, vocab: list[str]) -> np.ndarray:
    counts = routes.value_counts().reindex(vocab).fillna(0).to_numpy(float)
    return counts / max(counts.sum(), 1.0)


def top_overlap(p: np.ndarray, q: np.ndarray, vocab: list[str], k: int) -> int:
    a = set(np.array(vocab)[np.argsort(-p)[:k]])
    b = set(np.array(vocab)[np.argsort(-q)[:k]])
    return len(a & b)


def project_attack_steps(
    store: ArtifactStore,
    model: str,
    source: str,
    layer: str,
    basis: np.ndarray,
    k: int,
) -> pd.DataFrame:
    rows, x = store.rows_for(model, source, layer)
    if rows.empty:
        return pd.DataFrame()
    coeff = x @ basis[:k].T
    norms = np.linalg.norm(x, axis=1)
    top_idx = np.argmax(np.abs(coeff), axis=1)
    top_coeff = coeff[np.arange(len(coeff)), top_idx]
    out = rows.copy()
    out["route"] = [signed_route(int(i), float(c)) for i, c in zip(top_idx, top_coeff)]
    out["pc"] = top_idx + 1
    out["sign"] = np.where(top_coeff >= 0, 1, -1)
    out["top_abs_coeff"] = np.abs(top_coeff)
    out["top_signed_coeff"] = top_coeff
    out["route_energy"] = (top_coeff ** 2) / np.maximum(norms ** 2, 1e-12)
    out["topk_energy"] = (coeff ** 2).sum(axis=1) / np.maximum(norms ** 2, 1e-12)
    out["margin_drop"] = out["margin_before"].astype(float) - out["margin_after"].astype(float)
    out["prob_drop"] = out["true_prob_before"].astype(float) - out["true_prob_after"].astype(float)
    return out


def per_image_overlap(
    budget: pd.DataFrame,
    attack: pd.DataFrame,
    vocab: list[str],
    rng: np.random.Generator,
    n_boot_random: int = 200,
) -> pd.DataFrame:
    rows = []
    attack_groups = {int(k): g for k, g in attack.groupby("image_ord")}
    for (method, image_ord), b in budget.groupby(["method", "image_ord"]):
        image_ord = int(image_ord)
        if image_ord not in attack_groups:
            continue
        a = attack_groups[image_ord]
        b_routes = list(b["route"].astype(str))
        a_routes = list(a["route"].astype(str))
        b_set = set(b_routes)
        a_set = set(a_routes)
        if not b_set or not a_set:
            continue
        exact_inter = len(b_set & a_set)
        exact_union = len(b_set | a_set)
        pc_inter = len(set(b["pc"].astype(int)) & set(a["pc"].astype(int)))
        pc_union = len(set(b["pc"].astype(int)) | set(a["pc"].astype(int)))
        coverage = float(np.mean([r in a_set for r in b_routes]))
        pc_coverage = float(np.mean([int(pc) in set(a["pc"].astype(int)) for pc in b["pc"]]))
        attack_route_energy_on_budget = float(a[a["route"].isin(b_set)]["route_energy"].sum() / max(a["route_energy"].sum(), 1e-12))
        random_coverages = []
        random_jaccards = []
        for _ in range(n_boot_random):
            sampled = set(rng.choice(vocab, size=len(b_set), replace=False))
            random_coverages.append(float(np.mean([r in a_set for r in sampled])))
            random_jaccards.append(len(sampled & a_set) / max(len(sampled | a_set), 1))
        rows.append(
            {
                "method": method,
                "image_ord": image_ord,
                "attack_source": str(a["source"].iloc[0]),
                "attack_final_success": int(a["final_success"].max()),
                "budget_success": int(b["after_success"].max()) if "after_success" in b else np.nan,
                "n_budget_steps": int(len(b_routes)),
                "n_attack_steps": int(len(a_routes)),
                "route_jaccard": exact_inter / max(exact_union, 1),
                "route_any_match": int(exact_inter > 0),
                "route_coverage": coverage,
                "pc_jaccard": pc_inter / max(pc_union, 1),
                "pc_any_match": int(pc_inter > 0),
                "pc_coverage": pc_coverage,
                "attack_energy_fraction_on_budget_routes": attack_route_energy_on_budget,
                "random_route_jaccard_mean": float(np.mean(random_jaccards)),
                "random_route_coverage_mean": float(np.mean(random_coverages)),
                "route_jaccard_minus_random": exact_inter / max(exact_union, 1) - float(np.mean(random_jaccards)),
                "route_coverage_minus_random": coverage - float(np.mean(random_coverages)),
            }
        )
    return pd.DataFrame(rows)


def summarize_overlap(overlap: pd.DataFrame) -> pd.DataFrame:
    return (
        overlap.groupby(["method", "attack_source", "attack_final_success"], dropna=False)
        .agg(
            n=("image_ord", "size"),
            mean_route_jaccard=("route_jaccard", "mean"),
            mean_route_coverage=("route_coverage", "mean"),
            mean_route_any=("route_any_match", "mean"),
            mean_pc_jaccard=("pc_jaccard", "mean"),
            mean_pc_coverage=("pc_coverage", "mean"),
            mean_pc_any=("pc_any_match", "mean"),
            mean_energy_fraction=("attack_energy_fraction_on_budget_routes", "mean"),
            mean_jaccard_minus_random=("route_jaccard_minus_random", "mean"),
            mean_coverage_minus_random=("route_coverage_minus_random", "mean"),
        )
        .reset_index()
        .sort_values(["attack_source", "attack_final_success", "mean_route_coverage"], ascending=[True, False, False])
    )


def summarize_distribution(budget: pd.DataFrame, attacks: dict[str, pd.DataFrame], vocab: list[str]) -> pd.DataFrame:
    rows = []
    budget_dists = {}
    for method, sub in budget.groupby("method"):
        budget_dists[method] = distribution_for_routes(sub["route"], vocab)
    attack_dists = {src: distribution_for_routes(df["route"], vocab) for src, df in attacks.items()}
    for method, bd in budget_dists.items():
        for src, ad in attack_dists.items():
            rows.append(
                {
                    "method": method,
                    "attack_source": src,
                    "cosine": cosine_sim(bd, ad),
                    "js_divergence": js_divergence(bd, ad),
                    "top3_overlap": top_overlap(bd, ad, vocab, 3),
                    "top5_overlap": top_overlap(bd, ad, vocab, 5),
                    "budget_top_routes": ",".join(np.array(vocab)[np.argsort(-bd)[:5]]),
                    "attack_top_routes": ",".join(np.array(vocab)[np.argsort(-ad)[:5]]),
                }
            )
    return pd.DataFrame(rows)


def plot_distribution(dist_summary: pd.DataFrame, out_dir: Path) -> None:
    if dist_summary.empty:
        return
    methods = list(dist_summary["method"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.34 * len(methods))), dpi=200)
    for ax, metric, title in [
        (axes[0], "cosine", "Route distribution cosine"),
        (axes[1], "js_divergence", "Route distribution JS divergence"),
    ]:
        pivot = dist_summary.pivot(index="method", columns="attack_source", values=metric).reindex(methods)
        im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis" if metric == "cosine" else "magma_r")
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "budgeted_runaway_attack_path_distribution_compatibility.png", bbox_inches="tight")
    fig.savefig(out_dir / "budgeted_runaway_attack_path_distribution_compatibility.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--budget-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/budgeted_runaway_routing_bbb_resnet50_c200")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/budgeted_runaway_path_compatibility_bbb_resnet50_c200")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--candidate-k", type=int, default=8)
    p.add_argument("--attack-sources", default="pgd,square")
    p.add_argument("--budget-methods", default="runaway_mobility_margin_d5,runaway_margin_drop_d5,runaway_efficiency_increment_d5,runaway_mobility_d5,runaway_random_highway_d5")
    p.add_argument("--successful-only-distribution", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(Path(args.input_dir))
    _mean, basis, n_train = fit_highway_basis(store, args.model, args.highway_source, args.layer, args.highway_k, args.seed)
    vocab = route_vocab(args.candidate_k)
    budget = pd.read_csv(Path(args.budget_dir) / "budgeted_runaway_routing_selected_steps.csv")
    keep_methods = parse_csv(args.budget_methods)
    budget = budget[budget["method"].isin(keep_methods)].copy()
    budget = budget[budget["pc"].astype(int) <= args.candidate_k].copy()
    budget["route"] = budget["route"].astype(str)

    attacks = {}
    attack_all = []
    for src in parse_csv(args.attack_sources):
        projected = project_attack_steps(store, args.model, src, args.layer, basis, args.candidate_k)
        if projected.empty:
            continue
        projected["source"] = src
        attacks[src] = projected
        attack_all.append(projected)
        projected.to_csv(out_dir / f"{src}_projected_signed_routes.csv", index=False)

    overlap_frames = []
    rng = np.random.default_rng(args.seed)
    for src, projected in attacks.items():
        overlap_frames.append(per_image_overlap(budget, projected, vocab, rng))
    overlap = pd.concat(overlap_frames, ignore_index=True) if overlap_frames else pd.DataFrame()
    overlap_summary = summarize_overlap(overlap) if not overlap.empty else pd.DataFrame()

    dist_attacks = {}
    for src, projected in attacks.items():
        if args.successful_only_distribution:
            projected = projected[projected["final_success"].astype(int) == 1].copy()
        dist_attacks[src] = projected
    dist_summary = summarize_distribution(budget, dist_attacks, vocab)

    overlap.to_csv(out_dir / "budgeted_runaway_attack_path_overlap_per_image.csv", index=False)
    overlap_summary.to_csv(out_dir / "budgeted_runaway_attack_path_overlap_summary.csv", index=False)
    dist_summary.to_csv(out_dir / "budgeted_runaway_attack_path_distribution_summary.csv", index=False)
    budget.to_csv(out_dir / "budgeted_runaway_routes_used_for_compatibility.csv", index=False)
    plot_distribution(dist_summary, out_dir)

    metadata = {
        **vars(args),
        "highway_train_vectors": int(n_train),
        "route_vocabulary": vocab,
        "compatibility_definition": "PGD/Square hidden local steps are assigned to signed highway routes by largest absolute PC coefficient; budgeted routes are compared as signed route sets and distributions.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Budgeted Run-Away / Attack Path Compatibility",
        "",
        "Attack local steps and budgeted run-away routes were mapped into the same signed highway vocabulary.",
        "",
        "## Distribution similarity",
        "",
    ]
    for r in dist_summary.itertuples(index=False):
        lines.append(
            f"- `{r.method}` vs `{r.attack_source}`: cosine={r.cosine:.3f}, JS={r.js_divergence:.3f}, "
            f"top5_overlap={r.top5_overlap}, budget_top={r.budget_top_routes}, attack_top={r.attack_top_routes}"
        )
    lines += ["", "## Per-image overlap summary", ""]
    for r in overlap_summary.itertuples(index=False):
        lines.append(
            f"- `{r.method}` vs `{r.attack_source}` success={r.attack_final_success}: "
            f"route_coverage={r.mean_route_coverage:.3f}, route_jaccard={r.mean_route_jaccard:.3f}, "
            f"pc_coverage={r.mean_pc_coverage:.3f}, coverage_minus_random={r.mean_coverage_minus_random:.3f}, n={r.n}"
        )
    (out_dir / "budgeted_runaway_attack_path_compatibility_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(dist_summary.to_string(index=False), flush=True)
    print(overlap_summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
