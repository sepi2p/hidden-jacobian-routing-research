#!/usr/bin/env python3
"""Rank signed mobility-highway routes by margin drop and test usage.

This asks whether successful adversarial trajectories use the most
boundary-leading highway routes, or whether their highway use is uniform.

Protocol:
1. Fit a high-mobility highway basis from non-adversarial mobility controls.
2. On train-split attack steps only, score each signed PC route by its
   coefficient-weighted mean true-class margin drop.
3. On held-out steps/images, assign each step to its dominant signed route.
4. Compare route-rank usage for successful attacks, failed attacks, and controls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import randomized_svd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_csv(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def vector_key(model: str, source: str, layer: str) -> str:
    return f"{model}__{source}__{layer}"


def pca_basis(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    kk = min(k, x.shape[0] - 1, x.shape[1])
    if kk < 1:
        raise ValueError("PCA rank is zero.")
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    xc = x - mean
    _u, _s, vt = randomized_svd(xc, n_components=kk, random_state=seed)
    return mean, vt.astype(np.float32)


def safe_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    scores = np.r_[pos, neg]
    if np.std(scores) < 1e-12:
        return np.nan
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(labels, scores))


class Store:
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.rows = pd.read_csv(input_dir / "segment_metadata.csv")
        self.splits = pd.read_csv(input_dir / "image_splits.csv")
        self.arrays = np.load(input_dir / "segment_vectors.npz")
        self.split_by_image = dict(zip(self.splits["image_ord"].astype(int), self.splits["split"].astype(str)))

    def rows_for(self, model: str, source: str, layer: str) -> tuple[pd.DataFrame, np.ndarray]:
        key = vector_key(model, source, layer)
        sub = self.rows[(self.rows.model == model) & (self.rows.source == source) & (self.rows.layer == layer)].copy()
        if sub.empty or key not in self.arrays.files:
            return sub, np.zeros((0, 0), dtype=np.float32)
        sub["split"] = sub["image_ord"].map(self.split_by_image).fillna("")
        x = self.arrays[key][sub["vector_idx"].to_numpy(dtype=int)]
        return sub.reset_index(drop=True), x


def fit_highway(store: Store, model: str, source: str, layer: str, k: int, seed: int):
    rows, x = store.rows_for(model, source, layer)
    if rows.empty:
        raise RuntimeError(f"No highway source rows for {source}/{layer}.")
    train = rows["split"].to_numpy() == "train"
    if train.sum() < max(8, k + 2):
        raise RuntimeError(f"Too few highway train vectors: {train.sum()}.")
    mean, basis = pca_basis(x[train], k, seed)
    return mean, basis, int(train.sum())


def project_rows(rows: pd.DataFrame, x: np.ndarray, basis: np.ndarray) -> pd.DataFrame:
    coeff = x @ basis.T
    out = rows.copy()
    out["margin_drop"] = out["margin_before"].astype(float) - out["margin_after"].astype(float)
    out["feature_speed"] = np.linalg.norm(x, axis=1)
    out["highway_energy"] = np.sum(coeff * coeff, axis=1) / np.clip(np.sum(x * x, axis=1), 1e-12, None)
    abs_coeff = np.abs(coeff)
    dominant_idx = np.argmax(abs_coeff, axis=1)
    dominant_sign = np.where(coeff[np.arange(len(coeff)), dominant_idx] >= 0, 1, -1)
    out["dominant_pc"] = dominant_idx + 1
    out["dominant_sign"] = dominant_sign
    out["dominant_abs_coeff"] = abs_coeff[np.arange(len(coeff)), dominant_idx]
    for j in range(coeff.shape[1]):
        out[f"pc{j+1}_coeff"] = coeff[:, j]
    return out


def collect_projected(store: Store, model: str, sources: list[str], layer: str, basis: np.ndarray) -> pd.DataFrame:
    frames = []
    for source in sources:
        rows, x = store.rows_for(model, source, layer)
        if rows.empty:
            continue
        frames.append(project_rows(rows, x, basis))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def score_routes(train_df: pd.DataFrame, rank_sources: list[str], k: int, min_weight: float) -> pd.DataFrame:
    train = train_df[(train_df["split"] == "train") & (train_df["source"].isin(rank_sources))].copy()
    routes = []
    for pc in range(1, k + 1):
        coeff = train[f"pc{pc}_coeff"].to_numpy(dtype=float)
        margin_drop = train["margin_drop"].to_numpy(dtype=float)
        for sign, label in [(1, "+"), (-1, "-")]:
            w = np.maximum(sign * coeff, 0.0)
            total = float(w.sum())
            if total <= min_weight:
                score = np.nan
                mean_drop_unweighted = np.nan
                n_active = 0
            else:
                score = float(np.sum(w * margin_drop) / total)
                active = w > 0
                mean_drop_unweighted = float(np.mean(margin_drop[active])) if active.any() else np.nan
                n_active = int(active.sum())
            routes.append(
                {
                    "pc": pc,
                    "sign": sign,
                    "sign_label": label,
                    "route": f"pc{pc}{label}",
                    "weighted_margin_drop_score": score,
                    "unweighted_active_margin_drop": mean_drop_unweighted,
                    "coefficient_weight": total,
                    "n_active_train_steps": n_active,
                }
            )
    route_df = pd.DataFrame(routes)
    route_df = route_df.sort_values("weighted_margin_drop_score", ascending=False, na_position="last").reset_index(drop=True)
    route_df["route_rank"] = np.arange(1, len(route_df) + 1)
    route_df["rank_percentile"] = (route_df["route_rank"] - 1) / max(len(route_df) - 1, 1)
    return route_df


def add_route_ranks(df: pd.DataFrame, route_df: pd.DataFrame, top_sets: list[int]) -> pd.DataFrame:
    route_map = {(int(r.pc), int(r.sign)): int(r.route_rank) for r in route_df.itertuples()}
    score_map = {(int(r.pc), int(r.sign)): float(r.weighted_margin_drop_score) for r in route_df.itertuples()}
    name_map = {(int(r.pc), int(r.sign)): str(r.route) for r in route_df.itertuples()}
    out = df.copy()
    out["dominant_route_rank"] = [route_map.get((int(pc), int(sign)), np.nan) for pc, sign in zip(out.dominant_pc, out.dominant_sign)]
    out["dominant_route_score"] = [score_map.get((int(pc), int(sign)), np.nan) for pc, sign in zip(out.dominant_pc, out.dominant_sign)]
    out["dominant_route"] = [name_map.get((int(pc), int(sign)), "") for pc, sign in zip(out.dominant_pc, out.dominant_sign)]
    for n in top_sets:
        out[f"top{n}_route"] = (out["dominant_route_rank"] <= n).astype(int)
    return out


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    if counts.sum() <= 0:
        return np.nan
    p = counts / counts.sum()
    p = p[p > 0]
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(counts))) if len(counts) > 1 else 0.0


def summarize_usage(df: pd.DataFrame, route_df: pd.DataFrame, top_sets: list[int]) -> pd.DataFrame:
    rows = []
    eval_df = df[df["split"] == "test"].copy()
    n_routes = len(route_df)
    expected_uniform_top = {n: n / n_routes for n in top_sets}
    for keys, g in eval_df.groupby(["source", "final_success"], dropna=False):
        source, success = keys
        row = {
            "source": source,
            "final_success": int(success) if pd.notna(success) else -1,
            "n_steps": int(len(g)),
            "n_images": int(g["image_ord"].nunique()),
            "mean_route_rank": float(g["dominant_route_rank"].mean()),
            "median_route_rank": float(g["dominant_route_rank"].median()),
            "mean_route_score": float(g["dominant_route_score"].mean()),
            "mean_margin_drop": float(g["margin_drop"].mean()),
            "mean_highway_energy": float(g["highway_energy"].mean()),
            "mean_feature_speed": float(g["feature_speed"].mean()),
        }
        counts = g["dominant_route_rank"].value_counts().reindex(range(1, n_routes + 1), fill_value=0).to_numpy()
        row["route_entropy_normalized"] = entropy_from_counts(counts)
        if counts.sum() > 0:
            try:
                _stat, p = chisquare(counts)
                row["chisquare_uniform_p"] = float(p)
            except Exception:
                row["chisquare_uniform_p"] = np.nan
        for n in top_sets:
            frac = float(g[f"top{n}_route"].mean())
            row[f"top{n}_route_frac"] = frac
            row[f"top{n}_uniform_expected"] = expected_uniform_top[n]
            high = g[g[f"top{n}_route"] == 1]
            row[f"top{n}_mean_margin_drop"] = float(high["margin_drop"].mean()) if len(high) else np.nan
        rows.append(row)

    # Image-level attack success prediction from top route usage.
    pred_rows = []
    attacks = eval_df[eval_df["source"].isin(["pgd", "square"])].copy()
    for source, g in attacks.groupby("source"):
        agg_spec = {
            "dominant_route_rank": "mean",
            "dominant_route_score": "mean",
            "margin_drop": "mean",
            "highway_energy": "mean",
            "feature_speed": "mean",
        }
        for n in top_sets:
            agg_spec[f"top{n}_route"] = "mean"
        img = g.groupby(["image_ord", "final_success"], as_index=False).agg(agg_spec)
        pos = img[img.final_success == 1]
        neg = img[img.final_success == 0]
        for feature in ["dominant_route_rank", "dominant_route_score", "margin_drop", "highway_energy", "feature_speed"] + [
            f"top{n}_route" for n in top_sets
        ]:
            # Lower route rank is better, so invert for AUROC.
            pos_scores = -pos[feature].to_numpy(dtype=float) if feature == "dominant_route_rank" else pos[feature].to_numpy(dtype=float)
            neg_scores = -neg[feature].to_numpy(dtype=float) if feature == "dominant_route_rank" else neg[feature].to_numpy(dtype=float)
            pred_rows.append(
                {
                    "source": source,
                    "feature": feature,
                    "image_level_auroc_success_vs_failed": safe_auc(pos_scores, neg_scores),
                    "success_mean": float(pos[feature].mean()) if len(pos) else np.nan,
                    "failed_mean": float(neg[feature].mean()) if len(neg) else np.nan,
                    "success_images": int(len(pos)),
                    "failed_images": int(len(neg)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def write_summary(out_dir: Path, route_df: pd.DataFrame, usage: pd.DataFrame, pred: pd.DataFrame, top_sets: list[int]):
    lines = [
        "# Ranked Boundary-Leading Highway Analysis",
        "",
        "Signed highway routes are ranked by coefficient-weighted train-split margin drop. Held-out trajectories are then tested for top-rank route usage.",
        "",
        "## Top Routes",
        "",
    ]
    for r in route_df.head(10).itertuples():
        lines.append(
            f"- rank {r.route_rank}: {r.route}, score={r.weighted_margin_drop_score:.4f}, "
            f"active_train_steps={r.n_active_train_steps}"
        )
    lines += ["", "## Held-Out Usage", ""]
    for r in usage.sort_values(["source", "final_success"]).itertuples():
        bits = [
            f"{r.source}, final_success={r.final_success}: n={r.n_steps}, mean_rank={r.mean_route_rank:.2f}, "
            f"entropy={r.route_entropy_normalized:.3f}, mean_score={r.mean_route_score:.3f}, margin_drop={r.mean_margin_drop:.3f}"
        ]
        for n in top_sets:
            bits.append(f"top{n}={getattr(r, f'top{n}_route_frac'):.3f} (uniform {getattr(r, f'top{n}_uniform_expected'):.3f})")
        lines.append("- " + "; ".join(bits))
    lines += ["", "## Image-Level Success Prediction", ""]
    key = pred[pred["feature"].isin(["dominant_route_rank", "dominant_route_score"] + [f"top{n}_route" for n in top_sets])]
    for r in key.sort_values(["source", "feature"]).itertuples():
        lines.append(
            f"- {r.source}, {r.feature}: AUROC={r.image_level_auroc_success_vs_failed:.3f}, "
            f"success_mean={r.success_mean:.3f}, failed_mean={r.failed_mean:.3f}"
        )
    lines += [
        "",
        "## Gate",
        "",
        "If successful attacks overuse top-ranked routes relative to uniform, failures, and mobility/random controls, then adversarial success is routed through boundary-leading highway modes rather than uniformly across high-mobility highways.",
    ]
    (out_dir / "ranked_boundary_highway_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/ranked_boundary_highways_bbb_resnet50_c200_auto"))
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--rank-sources", default="pgd,square")
    p.add_argument("--eval-sources", default="pgd,square,mobility_top_walk_square_budget,random_sign_walk_square_budget,correlated_random_walk_square_budget")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--top-sets", default="1,3,5,10")
    p.add_argument("--min-weight", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = Store(args.input_dir)
    _mean, basis, n_train = fit_highway(store, args.model, args.highway_source, args.layer, args.highway_k, args.seed)
    projected = collect_projected(store, args.model, parse_csv(args.eval_sources), args.layer, basis)
    routes = score_routes(projected, parse_csv(args.rank_sources), args.highway_k, args.min_weight)
    top_sets = parse_int_csv(args.top_sets)
    ranked = add_route_ranks(projected, routes, top_sets)
    usage, pred = summarize_usage(ranked, routes, top_sets)

    routes.to_csv(args.output_dir / "ranked_highway_routes.csv", index=False)
    ranked.to_csv(args.output_dir / "ranked_highway_step_usage.csv", index=False)
    usage.to_csv(args.output_dir / "ranked_highway_usage_summary.csv", index=False)
    pred.to_csv(args.output_dir / "ranked_highway_success_prediction.csv", index=False)
    meta = {
        "script": "experiments/pure_af_geometry/analyze_ranked_boundary_highways.py",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "layer": args.layer,
        "highway_source": args.highway_source,
        "rank_sources": parse_csv(args.rank_sources),
        "eval_sources": parse_csv(args.eval_sources),
        "highway_k": args.highway_k,
        "highway_train_vectors": n_train,
        "top_sets": top_sets,
        "seed": args.seed,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_summary(args.output_dir, routes, usage, pred, top_sets)
    print(f"[DONE] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
