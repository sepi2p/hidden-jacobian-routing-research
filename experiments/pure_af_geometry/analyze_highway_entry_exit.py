#!/usr/bin/env python3
"""Analyze entry/exit behavior for high-mobility representation highways.

This tests the revised theory:

    high-mobility hidden-layer directions are the substrate, and successful
    adversarial attacks are routed uses of that substrate.

The highway basis is fit only from non-adversarial mobility/Jacobian controls.
No adversarial vectors are used to define the highway or its thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import randomized_svd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def vector_key(model: str, source: str, layer: str) -> str:
    return f"{model}__{source}__{layer}"


def safe_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    scores = np.r_[pos, neg]
    if np.std(scores) < 1e-12:
        return np.nan
    return float(roc_auc_score(labels, scores))


def bootstrap_auc(pos: np.ndarray, neg: np.ndarray, seed: int, reps: int) -> tuple[float, float, float]:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        p = rng.choice(pos, len(pos), replace=True)
        n = rng.choice(neg, len(neg), replace=True)
        vals.append(safe_auc(p, n))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    return float(vals.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def pca_basis(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < 2:
        raise ValueError("Need at least two rows for PCA.")
    kk = min(k, x.shape[0] - 1, x.shape[1])
    if kk < 1:
        raise ValueError("PCA rank is zero.")
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    xc = x - mean
    _u, _s, vt = randomized_svd(xc, n_components=kk, random_state=seed)
    return mean, vt.astype(np.float32)


def projection_scores(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kk = min(k, basis.shape[0], x.shape[1])
    if kk < 1 or len(x) == 0:
        zeros = np.zeros(len(x), dtype=np.float64)
        return zeros, zeros, zeros
    b = basis[:kk]
    coeff_uncentered = x @ b.T
    energy_uncentered = np.sum(coeff_uncentered * coeff_uncentered, axis=1) / np.clip(
        np.sum(x * x, axis=1), 1e-12, None
    )
    xc = x - mean
    coeff_centered = xc @ b.T
    energy_centered = np.sum(coeff_centered * coeff_centered, axis=1) / np.clip(
        np.sum(xc * xc, axis=1), 1e-12, None
    )
    speed = np.linalg.norm(x, axis=1)
    return energy_uncentered, energy_centered, speed


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


def fit_highway(store: Store, model: str, source: str, layer: str, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    rows, x = store.rows_for(model, source, layer)
    if rows.empty:
        raise RuntimeError(f"No rows for highway source {source}/{layer}.")
    train = rows["split"].to_numpy() == "train"
    if train.sum() < max(8, min(k, x.shape[1]) + 2):
        raise RuntimeError(f"Too few train vectors for highway source {source}/{layer}.")
    mean, basis = pca_basis(x[train], k, seed)
    return mean, basis, int(train.sum())


def compute_thresholds(
    store: Store,
    model: str,
    layer: str,
    mean: np.ndarray,
    basis: np.ndarray,
    k: int,
    highway_source: str,
    threshold_sources: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    thresholds = {}
    all_sources = [highway_source] + [s for s in threshold_sources if s != highway_source]
    for source in all_sources:
        sub, x = store.rows_for(model, source, layer)
        if sub.empty:
            continue
        train = sub["split"].to_numpy() == "train"
        if not train.any():
            continue
        e, ec, speed = projection_scores(x[train], mean, basis, k)
        for q in [0.25, 0.50, 0.75, 0.90]:
            rows.append(
                {
                    "source": source,
                    "quantile": q,
                    "energy_uncentered": float(np.quantile(e, q)),
                    "energy_centered": float(np.quantile(ec, q)),
                    "speed": float(np.quantile(speed, q)),
                    "n_train_steps": int(train.sum()),
                }
            )
        if source == highway_source:
            thresholds["highway_q25"] = float(np.quantile(e, 0.25))
            thresholds["highway_q50"] = float(np.quantile(e, 0.50))
        if source.startswith("random_sign"):
            thresholds[f"{source}_q90"] = float(np.quantile(e, 0.90))
        if source.startswith("correlated_random"):
            thresholds[f"{source}_q90"] = float(np.quantile(e, 0.90))
    if "highway_q25" not in thresholds:
        raise RuntimeError("Could not define highway threshold.")
    return thresholds, pd.DataFrame(rows)


def trajectory_metrics_for_source(
    store: Store,
    model: str,
    source: str,
    layer: str,
    mean: np.ndarray,
    basis: np.ndarray,
    k: int,
    thresholds: dict[str, float],
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub, x = store.rows_for(model, source, layer)
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    mask = sub["split"].to_numpy() == split
    sub = sub[mask].copy().reset_index(drop=True)
    x = x[mask]
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    e, ec, speed = projection_scores(x, mean, basis, k)
    sub["highway_energy"] = e
    sub["highway_energy_centered"] = ec
    sub["feature_speed"] = speed
    sub["margin_drop_step"] = sub["margin_before"].astype(float) - sub["margin_after"].astype(float)
    sub["p_y_drop_step"] = sub["true_prob_before"].astype(float) - sub["true_prob_after"].astype(float)
    step_rows = []
    traj_rows = []
    for image_ord, g in sub.groupby("image_ord", sort=True):
        g = g.sort_values("step").copy()
        n = len(g)
        if n == 0:
            continue
        boundary_candidates = g[g["step_success_after"].astype(int) == 1]
        boundary_step = int(boundary_candidates["step"].iloc[0]) if len(boundary_candidates) else np.nan
        first_margin = float(g["margin_before"].iloc[0])
        final_margin = float(g["margin_after"].iloc[-1])
        margin_drop_total = first_margin - final_margin
        first_py = float(g["true_prob_before"].iloc[0])
        final_py = float(g["true_prob_after"].iloc[-1])
        py_drop_total = first_py - final_py
        early_n = max(1, int(math.ceil(0.2 * n)))
        base = {
            "model": model,
            "source": source,
            "layer": layer,
            "image_ord": int(image_ord),
            "dataset_idx": int(g["dataset_idx"].iloc[0]),
            "label": int(g["label"].iloc[0]),
            "final_success": int(g["final_success"].iloc[0]),
            "final_pred": int(g["final_pred"].iloc[0]),
            "n_steps": int(n),
            "boundary_step": boundary_step,
            "boundary_frac": float(boundary_step / max(n - 1, 1)) if np.isfinite(boundary_step) else np.nan,
            "mean_highway_energy": float(g["highway_energy"].mean()),
            "max_highway_energy": float(g["highway_energy"].max()),
            "early20_mean_highway_energy": float(g["highway_energy"].iloc[:early_n].mean()),
            "early20_max_highway_energy": float(g["highway_energy"].iloc[:early_n].max()),
            "mean_feature_speed": float(g["feature_speed"].mean()),
            "max_feature_speed": float(g["feature_speed"].max()),
            "margin_drop_total": float(margin_drop_total),
            "p_y_drop_total": float(py_drop_total),
        }
        for tname, tval in thresholds.items():
            above = g["highway_energy"].to_numpy(dtype=float) >= float(tval)
            if above.any():
                entry_pos = int(np.argmax(above))
                entry_step = int(g["step"].iloc[entry_pos])
                entry_frac = float(entry_pos / max(n - 1, 1))
            else:
                entry_pos = -1
                entry_step = np.nan
                entry_frac = np.nan
            dwell = float(np.mean(above))
            early_dwell = float(np.mean(above[:early_n]))
            if np.isfinite(boundary_step) and np.isfinite(entry_step):
                lead_steps = int(boundary_step - entry_step)
                pre_boundary = int(entry_step < boundary_step)
                at_or_before_boundary = int(entry_step <= boundary_step)
            else:
                lead_steps = np.nan
                pre_boundary = 0
                at_or_before_boundary = 0
            base[f"{tname}_entered"] = int(entry_pos >= 0)
            base[f"{tname}_entry_step"] = entry_step
            base[f"{tname}_entry_frac"] = entry_frac
            base[f"{tname}_dwell_frac"] = dwell
            base[f"{tname}_early20_dwell_frac"] = early_dwell
            base[f"{tname}_lead_steps_to_boundary"] = lead_steps
            base[f"{tname}_pre_boundary_entry"] = pre_boundary
            base[f"{tname}_at_or_before_boundary_entry"] = at_or_before_boundary
            sub[f"{tname}_above"] = sub["highway_energy"] >= float(tval)
        traj_rows.append(base)
        step_rows.append(g)
    return pd.concat(step_rows, ignore_index=True), pd.DataFrame(traj_rows)


def summarize_sources(traj: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    if traj.empty:
        return pd.DataFrame()
    for (source, layer), g in traj.groupby(["source", "layer"]):
        row = {
            "source": source,
            "layer": layer,
            "n_images": int(len(g)),
            "success_rate": float(g["final_success"].mean()),
            "mean_highway_energy": float(g["mean_highway_energy"].mean()),
            "early20_mean_highway_energy": float(g["early20_mean_highway_energy"].mean()),
            "mean_margin_drop_total": float(g["margin_drop_total"].mean()),
            "median_margin_drop_total": float(g["margin_drop_total"].median()),
            "mean_feature_speed": float(g["mean_feature_speed"].mean()),
        }
        for tname in thresholds:
            row[f"{tname}_entry_rate"] = float(g[f"{tname}_entered"].mean())
            row[f"{tname}_early20_entry_rate"] = float((g[f"{tname}_early20_dwell_frac"] > 0).mean())
            row[f"{tname}_mean_dwell_frac"] = float(g[f"{tname}_dwell_frac"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_metrics(traj: pd.DataFrame, thresholds: dict[str, float], seed: int, reps: int) -> pd.DataFrame:
    rows = []
    if traj.empty:
        return pd.DataFrame()
    features = [
        "mean_highway_energy",
        "max_highway_energy",
        "early20_mean_highway_energy",
        "early20_max_highway_energy",
        "mean_feature_speed",
        "max_feature_speed",
    ]
    for tname in thresholds:
        features += [f"{tname}_entered", f"{tname}_early20_dwell_frac", f"{tname}_dwell_frac"]
    for (source, layer), g in traj[traj["source"].isin(["pgd", "square"])].groupby(["source", "layer"]):
        pos = g[g.final_success.astype(int) == 1]
        neg = g[g.final_success.astype(int) == 0]
        for feat in features:
            p = pos[feat].to_numpy(dtype=float)
            n = neg[feat].to_numpy(dtype=float)
            bmean, blo, bhi = bootstrap_auc(p, n, seed, reps)
            rows.append(
                {
                    "source": source,
                    "layer": layer,
                    "feature": feat,
                    "auroc": safe_auc(p, n),
                    "bootstrap_auroc_mean": bmean,
                    "bootstrap_auroc_lo": blo,
                    "bootstrap_auroc_hi": bhi,
                    "pos_mean": float(np.nanmean(p)) if len(p) else np.nan,
                    "neg_mean": float(np.nanmean(n)) if len(n) else np.nan,
                    "pos_images": int(len(pos)),
                    "neg_images": int(len(neg)),
                }
            )
    return pd.DataFrame(rows)


def temporal_curves(step_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if step_df.empty:
        return pd.DataFrame()
    for (source, layer, final_success), g in step_df.groupby(["source", "layer", "final_success"]):
        for image_ord, gi in g.groupby("image_ord"):
            gi = gi.sort_values("step")
            n = len(gi)
            for _, r in gi.iterrows():
                rows.append(
                    {
                        "source": source,
                        "layer": layer,
                        "final_success": int(final_success),
                        "image_ord": int(image_ord),
                        "time_frac": float(r["step"] / max(n - 1, 1)),
                        "highway_energy": float(r["highway_energy"]),
                        "feature_speed": float(r["feature_speed"]),
                        "margin_after": float(r["margin_after"]),
                        "margin_drop_step": float(r["margin_drop_step"]),
                    }
                )
    df = pd.DataFrame(rows)
    bins = np.linspace(0, 1, 6)
    df["time_bin"] = pd.cut(df["time_frac"], bins=bins, include_lowest=True, labels=["0-20", "20-40", "40-60", "60-80", "80-100"])
    out = (
        df.groupby(["source", "layer", "final_success", "time_bin"], observed=True)
        .agg(
            n_steps=("highway_energy", "size"),
            mean_highway_energy=("highway_energy", "mean"),
            mean_feature_speed=("feature_speed", "mean"),
            mean_margin_after=("margin_after", "mean"),
            mean_margin_drop_step=("margin_drop_step", "mean"),
        )
        .reset_index()
    )
    return out


def maybe_plot(out_dir: Path, curves: pd.DataFrame, pred: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not curves.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
        for ax, source in zip(axes, ["pgd", "square"]):
            sub = curves[(curves.source == source) & (curves.layer == "layer4")]
            for success, label, color in [(1, "success", "#1f77b4"), (0, "failed", "#d62728")]:
                ss = sub[sub.final_success == success]
                if ss.empty:
                    continue
                x = np.arange(len(ss))
                ax.plot(x, ss["mean_highway_energy"], marker="o", label=label, color=color)
            ax.set_title(source.upper())
            ax.set_xticks(range(5))
            ax.set_xticklabels(["0-20", "20-40", "40-60", "60-80", "80-100"], rotation=30)
            ax.set_ylabel("highway energy")
            ax.set_xlabel("trajectory progress")
            ax.legend(frameon=False)
        fig.savefig(out_dir / "highway_energy_over_time.png", dpi=220)
        plt.close(fig)
    if not pred.empty:
        fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
        sub = pred[(pred.layer == "layer4") & (pred.feature.isin(["early20_mean_highway_energy", "mean_highway_energy", "mean_feature_speed"]))]
        labels = []
        vals = []
        colors = []
        for source in ["pgd", "square"]:
            for feat in ["early20_mean_highway_energy", "mean_highway_energy", "mean_feature_speed"]:
                row = sub[(sub.source == source) & (sub.feature == feat)]
                if row.empty:
                    continue
                labels.append(f"{source}\n{feat.replace('_', ' ')}")
                vals.append(float(row["auroc"].iloc[0]))
                colors.append("#4c78a8" if source == "pgd" else "#f58518")
        ax.bar(np.arange(len(vals)), vals, color=colors)
        ax.axhline(0.5, color="black", lw=1, ls="--")
        ax.set_ylim(0, 1)
        ax.set_ylabel("success-prediction AUROC")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        fig.savefig(out_dir / "highway_success_prediction.png", dpi=220)
        plt.close(fig)


def write_summary(out_dir: Path, source_summary: pd.DataFrame, pred: pd.DataFrame, thresholds: dict[str, float], threshold_df: pd.DataFrame):
    lines = [
        "# Highway Entry/Exit Analysis",
        "",
        "The highway basis and thresholds are fit from non-adversarial mobility/Jacobian controls only. Adversarial trajectories are then evaluated on held-out images.",
        "",
        "## Thresholds",
        "",
    ]
    for k, v in thresholds.items():
        lines.append(f"- `{k}`: {v:.4f}")
    lines += ["", "## Source Summary", ""]
    if source_summary.empty:
        lines.append("No source summary produced.")
    else:
        for r in source_summary.sort_values(["layer", "source"]).itertuples():
            lines.append(
                f"- {r.source}, {r.layer}: success_rate={r.success_rate:.3f}, "
                f"mean_highway_energy={r.mean_highway_energy:.3f}, early20={r.early20_mean_highway_energy:.3f}, "
                f"margin_drop={r.mean_margin_drop_total:.3f}, speed={r.mean_feature_speed:.3f}"
            )
    lines += ["", "## Success Prediction From Highway Features", ""]
    if pred.empty:
        lines.append("No prediction metrics produced.")
    else:
        key = pred[(pred.layer == "layer4") & (pred.feature.isin(["early20_mean_highway_energy", "mean_highway_energy", "max_highway_energy", "mean_feature_speed"]))]
        for r in key.sort_values(["source", "feature"]).itertuples():
            lines.append(
                f"- {r.source}, {r.feature}: AUROC={r.auroc:.3f} "
                f"[{r.bootstrap_auroc_lo:.3f}, {r.bootstrap_auroc_hi:.3f}], "
                f"success_mean={r.pos_mean:.3f}, failed_mean={r.neg_mean:.3f}"
            )
    lines += [
        "",
        "## Interpretation Gate",
        "",
        "If successful attacks enter the non-adversarial highway earlier or with higher dwell than failures, while mobility/random controls enter highways without comparable margin progress, the highway theory is supported: the highway is a model-induced mobility substrate and adversarial success is a routed use of it.",
    ]
    out_dir.joinpath("highway_entry_exit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/highway_entry_exit_bbb_resnet50_c200_auto"))
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layers", default="layer4")
    p.add_argument("--highway-sources", default="mobility_top_walk_square_budget,jacobian_probe_all")
    p.add_argument("--eval-sources", default="pgd,square,mobility_top_walk_square_budget,random_sign_walk_square_budget,correlated_random_walk_square_budget")
    p.add_argument("--threshold-sources", default="random_sign_walk_square_budget,correlated_random_walk_square_budget,mobility_top_walk_square_budget")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--split", default="test")
    p.add_argument("--bootstrap-reps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = Store(args.input_dir)
    layers = parse_csv(args.layers)
    highway_sources = parse_csv(args.highway_sources)
    eval_sources = parse_csv(args.eval_sources)
    threshold_sources = parse_csv(args.threshold_sources)

    all_step = []
    all_traj = []
    all_threshold_rows = []
    all_source_summary = []
    all_pred = []
    all_curves = []
    basis_rows = []

    for layer in layers:
        for highway_source in highway_sources:
            mean, basis, n_train = fit_highway(store, args.model, highway_source, layer, args.highway_k, args.seed)
            thresholds, threshold_df = compute_thresholds(
                store, args.model, layer, mean, basis, args.highway_k, highway_source, threshold_sources
            )
            threshold_df["model"] = args.model
            threshold_df["layer"] = layer
            threshold_df["highway_source"] = highway_source
            all_threshold_rows.append(threshold_df)
            basis_rows.append(
                {
                    "model": args.model,
                    "layer": layer,
                    "highway_source": highway_source,
                    "highway_k": args.highway_k,
                    "train_vectors": n_train,
                    **{f"threshold_{k}": v for k, v in thresholds.items()},
                }
            )
            step_frames = []
            traj_frames = []
            for source in eval_sources:
                step, traj = trajectory_metrics_for_source(
                    store,
                    args.model,
                    source,
                    layer,
                    mean,
                    basis,
                    args.highway_k,
                    thresholds,
                    args.split,
                )
                if not step.empty:
                    step["highway_source"] = highway_source
                    step_frames.append(step)
                if not traj.empty:
                    traj["highway_source"] = highway_source
                    traj_frames.append(traj)
            if not traj_frames:
                continue
            step_df = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame()
            traj_df = pd.concat(traj_frames, ignore_index=True)
            all_step.append(step_df)
            all_traj.append(traj_df)
            source_summary = summarize_sources(traj_df, thresholds)
            source_summary["model"] = args.model
            source_summary["highway_source"] = highway_source
            all_source_summary.append(source_summary)
            pred = prediction_metrics(traj_df, thresholds, args.seed, args.bootstrap_reps)
            pred["model"] = args.model
            pred["highway_source"] = highway_source
            all_pred.append(pred)
            curves = temporal_curves(step_df)
            curves["model"] = args.model
            curves["highway_source"] = highway_source
            all_curves.append(curves)

    step_all = pd.concat(all_step, ignore_index=True) if all_step else pd.DataFrame()
    traj_all = pd.concat(all_traj, ignore_index=True) if all_traj else pd.DataFrame()
    threshold_all = pd.concat(all_threshold_rows, ignore_index=True) if all_threshold_rows else pd.DataFrame()
    source_all = pd.concat(all_source_summary, ignore_index=True) if all_source_summary else pd.DataFrame()
    pred_all = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    curves_all = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    basis_all = pd.DataFrame(basis_rows)

    step_all.to_csv(args.output_dir / "highway_step_metrics.csv", index=False)
    traj_all.to_csv(args.output_dir / "highway_trajectory_metrics.csv", index=False)
    threshold_all.to_csv(args.output_dir / "highway_thresholds.csv", index=False)
    source_all.to_csv(args.output_dir / "highway_source_summary.csv", index=False)
    pred_all.to_csv(args.output_dir / "highway_success_prediction_metrics.csv", index=False)
    curves_all.to_csv(args.output_dir / "highway_temporal_curves.csv", index=False)
    basis_all.to_csv(args.output_dir / "highway_basis_metadata.csv", index=False)
    metadata = {
        "script": "experiments/pure_af_geometry/analyze_highway_entry_exit.py",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "layers": layers,
        "highway_sources": highway_sources,
        "eval_sources": eval_sources,
        "threshold_sources": threshold_sources,
        "highway_k": args.highway_k,
        "split": args.split,
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Summarize the first highway source for readability; CSVs contain all.
    first_hs = highway_sources[0] if highway_sources else ""
    write_summary(
        args.output_dir,
        source_all[source_all.highway_source == first_hs] if not source_all.empty else source_all,
        pred_all[pred_all.highway_source == first_hs] if not pred_all.empty else pred_all,
        {k.replace("threshold_", ""): v for k, v in basis_rows[0].items() if k.startswith("threshold_")} if basis_rows else {},
        threshold_all,
    )
    if first_hs:
        maybe_plot(
            args.output_dir,
            curves_all[curves_all.highway_source == first_hs] if not curves_all.empty else curves_all,
            pred_all[pred_all.highway_source == first_hs] if not pred_all.empty else pred_all,
        )
    print(f"[DONE] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
