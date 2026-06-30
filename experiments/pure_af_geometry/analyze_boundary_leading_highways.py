#!/usr/bin/env python3
"""Classify mobility highways as boundary-leading or adversarially neutral.

This script uses logged local steps and a non-adversarial highway basis.  It
asks whether high-mobility/highway-aligned steps differ in their effect on the
true-class margin, and whether successful attacks preferentially use the
margin-descending subset.
"""

from __future__ import annotations

import argparse
import json
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


def pca_basis(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    kk = min(k, x.shape[0] - 1, x.shape[1])
    if kk < 1:
        raise ValueError("PCA rank is zero.")
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    xc = x - mean
    _u, _s, vt = randomized_svd(xc, n_components=kk, random_state=seed)
    return mean, vt.astype(np.float32)


def projection_scores(x: np.ndarray, basis: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kk = min(k, basis.shape[0], x.shape[1])
    b = basis[:kk]
    coeff = x @ b.T
    highway_energy = np.sum(coeff * coeff, axis=1) / np.clip(np.sum(x * x, axis=1), 1e-12, None)
    speed = np.linalg.norm(x, axis=1)
    signed_pc1 = coeff[:, 0] if coeff.shape[1] else np.zeros(len(x))
    return highway_energy, speed, signed_pc1


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
    train = rows["split"].to_numpy() == "train"
    if train.sum() < max(8, k + 2):
        raise RuntimeError(f"Too few highway train vectors: {train.sum()}")
    mean, basis = pca_basis(x[train], k, seed)
    e, speed, _pc1 = projection_scores(x[train], basis, k)
    thresholds = {
        "highway_q50": float(np.quantile(e, 0.50)),
        "highway_q75": float(np.quantile(e, 0.75)),
        "highway_q90": float(np.quantile(e, 0.90)),
        "speed_q50": float(np.quantile(speed, 0.50)),
        "speed_q75": float(np.quantile(speed, 0.75)),
        "speed_q90": float(np.quantile(speed, 0.90)),
    }
    return mean, basis, thresholds, int(train.sum())


def classify_margin(delta: pd.Series, tol: float) -> pd.Series:
    out = np.full(len(delta), "neutral", dtype=object)
    vals = delta.to_numpy(dtype=float)
    out[vals > tol] = "boundary_leading"
    out[vals < -tol] = "class_supporting"
    return pd.Series(out, index=delta.index)


def analyze_steps(store: Store, model: str, sources: list[str], layer: str, basis: np.ndarray, thresholds: dict, k: int, margin_tol: float):
    frames = []
    for source in sources:
        rows, x = store.rows_for(model, source, layer)
        if rows.empty:
            continue
        e, speed, pc1 = projection_scores(x, basis, k)
        df = rows.copy()
        df["highway_energy"] = e
        df["feature_speed"] = speed
        df["signed_pc1"] = pc1
        df["margin_drop"] = df["margin_before"].astype(float) - df["margin_after"].astype(float)
        df["p_y_drop"] = df["true_prob_before"].astype(float) - df["true_prob_after"].astype(float)
        df["margin_class"] = classify_margin(df["margin_drop"], margin_tol)
        for name, value in thresholds.items():
            if name.startswith("highway"):
                df[f"{name}_highway_step"] = (df["highway_energy"] >= value).astype(int)
            if name.startswith("speed"):
                df[f"{name}_fast_step"] = (df["feature_speed"] >= value).astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(step_df: pd.DataFrame, thresholds: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_rows = []
    class_rows = []
    pred_rows = []
    if step_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for (source, split, final_success), g in step_df.groupby(["source", "split", "final_success"], dropna=False):
        row = {
            "source": source,
            "split": split,
            "final_success": int(final_success) if pd.notna(final_success) else -1,
            "n_steps": int(len(g)),
            "n_images": int(g["image_ord"].nunique()),
            "mean_margin_drop": float(g["margin_drop"].mean()),
            "median_margin_drop": float(g["margin_drop"].median()),
            "frac_boundary_leading": float((g["margin_class"] == "boundary_leading").mean()),
            "frac_neutral": float((g["margin_class"] == "neutral").mean()),
            "frac_class_supporting": float((g["margin_class"] == "class_supporting").mean()),
            "mean_highway_energy": float(g["highway_energy"].mean()),
            "mean_speed": float(g["feature_speed"].mean()),
        }
        for name in thresholds:
            if name.startswith("highway"):
                col = f"{name}_highway_step"
                high = g[g[col] == 1]
                row[f"{name}_step_frac"] = float((g[col] == 1).mean())
                row[f"{name}_boundary_leading_frac_given_highway"] = float((high["margin_class"] == "boundary_leading").mean()) if len(high) else np.nan
                row[f"{name}_mean_margin_drop_given_highway"] = float(high["margin_drop"].mean()) if len(high) else np.nan
            if name.startswith("speed"):
                col = f"{name}_fast_step"
                high = g[g[col] == 1]
                row[f"{name}_step_frac"] = float((g[col] == 1).mean())
                row[f"{name}_boundary_leading_frac_given_fast"] = float((high["margin_class"] == "boundary_leading").mean()) if len(high) else np.nan
                row[f"{name}_mean_margin_drop_given_fast"] = float(high["margin_drop"].mean()) if len(high) else np.nan
        source_rows.append(row)

    for (source, margin_class), g in step_df.groupby(["source", "margin_class"]):
        class_rows.append(
            {
                "source": source,
                "margin_class": margin_class,
                "n_steps": int(len(g)),
                "n_images": int(g["image_ord"].nunique()),
                "mean_highway_energy": float(g["highway_energy"].mean()),
                "mean_speed": float(g["feature_speed"].mean()),
                "mean_margin_drop": float(g["margin_drop"].mean()),
                "mean_abs_pc1": float(np.abs(g["signed_pc1"]).mean()),
                "pc1_positive_frac": float((g["signed_pc1"] > 0).mean()),
            }
        )

    test = step_df[(step_df["split"] == "test") & (step_df["source"].isin(["pgd", "square"]))]
    for source, g in test.groupby("source"):
        pos = g[(g.final_success == 1)]
        neg = g[(g.final_success == 0)]
        for feature in ["highway_energy", "feature_speed", "margin_drop", "signed_pc1"]:
            pred_rows.append(
                {
                    "source": source,
                    "unit": "step",
                    "feature": feature,
                    "auroc_success_vs_failed": safe_auc(pos[feature].to_numpy(), neg[feature].to_numpy()),
                    "pos_mean": float(pos[feature].mean()) if len(pos) else np.nan,
                    "neg_mean": float(neg[feature].mean()) if len(neg) else np.nan,
                    "pos_steps": int(len(pos)),
                    "neg_steps": int(len(neg)),
                }
            )
        # Image-level summaries.
        img = g.groupby(["image_ord", "final_success"], as_index=False).agg(
            highway_energy=("highway_energy", "mean"),
            feature_speed=("feature_speed", "mean"),
            margin_drop=("margin_drop", "mean"),
            boundary_leading_frac=("margin_class", lambda s: float((s == "boundary_leading").mean())),
            q75_highway_frac=("highway_q75_highway_step", "mean"),
        )
        posi = img[img.final_success == 1]
        negi = img[img.final_success == 0]
        for feature in ["highway_energy", "feature_speed", "margin_drop", "boundary_leading_frac", "q75_highway_frac"]:
            pred_rows.append(
                {
                    "source": source,
                    "unit": "image_mean",
                    "feature": feature,
                    "auroc_success_vs_failed": safe_auc(posi[feature].to_numpy(), negi[feature].to_numpy()),
                    "pos_mean": float(posi[feature].mean()) if len(posi) else np.nan,
                    "neg_mean": float(negi[feature].mean()) if len(negi) else np.nan,
                    "pos_steps": int(len(posi)),
                    "neg_steps": int(len(negi)),
                }
            )
    return pd.DataFrame(source_rows), pd.DataFrame(class_rows), pd.DataFrame(pred_rows)


def write_summary(out_dir: Path, source_summary: pd.DataFrame, class_summary: pd.DataFrame, prediction: pd.DataFrame):
    lines = [
        "# Boundary-Leading Highway Analysis",
        "",
        "This analysis classifies high-mobility/highway-aligned local steps by whether they decrease the true-class margin.",
        "",
        "## Source Summary",
        "",
    ]
    if source_summary.empty:
        lines.append("No source summary produced.")
    else:
        key = source_summary[(source_summary.split == "test") & (source_summary.source.isin(["pgd", "square", "mobility_top_walk_square_budget", "random_sign_walk_square_budget", "correlated_random_walk_square_budget"]))]
        for r in key.sort_values(["source", "final_success"]).itertuples():
            lines.append(
                f"- {r.source}, final_success={r.final_success}: n_steps={r.n_steps}, "
                f"mean_margin_drop={r.mean_margin_drop:.3f}, boundary_leading_frac={r.frac_boundary_leading:.3f}, "
                f"mean_highway={r.mean_highway_energy:.3f}, q75_highway_boundary_frac={getattr(r, 'highway_q75_boundary_leading_frac_given_highway'):.3f}"
            )
    lines += ["", "## Margin-Class Summary", ""]
    if not class_summary.empty:
        for r in class_summary.sort_values(["source", "margin_class"]).itertuples():
            lines.append(
                f"- {r.source}, {r.margin_class}: n={r.n_steps}, "
                f"mean_highway={r.mean_highway_energy:.3f}, mean_speed={r.mean_speed:.3f}, "
                f"pc1_positive_frac={r.pc1_positive_frac:.3f}"
            )
    lines += ["", "## Success Prediction", ""]
    if not prediction.empty:
        key = prediction[(prediction.unit == "image_mean") & (prediction.feature.isin(["boundary_leading_frac", "q75_highway_frac", "highway_energy", "feature_speed"]))]
        for r in key.sort_values(["source", "feature"]).itertuples():
            lines.append(
                f"- {r.source}, {r.feature}: AUROC={r.auroc_success_vs_failed:.3f}, "
                f"success_mean={r.pos_mean:.3f}, failed_mean={r.neg_mean:.3f}"
            )
    lines += [
        "",
        "## Gate",
        "",
        "Support for boundary-leading highways requires high highway alignment to be common in both useful and useless motion, while successful attacks show a larger fraction of margin-descending highway steps than controls/failures.",
    ]
    (out_dir / "boundary_leading_highway_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/boundary_leading_highways_bbb_resnet50_c200_auto"))
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--eval-sources", default="pgd,square,mobility_top_walk_square_budget,random_sign_walk_square_budget,correlated_random_walk_square_budget")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--margin-tol", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = Store(args.input_dir)
    _mean, basis, thresholds, n_train = fit_highway(store, args.model, args.highway_source, args.layer, args.highway_k, args.seed)
    step_df = analyze_steps(store, args.model, parse_csv(args.eval_sources), args.layer, basis, thresholds, args.highway_k, args.margin_tol)
    source_summary, class_summary, prediction = summarize(step_df, thresholds)
    step_df.to_csv(args.output_dir / "boundary_leading_step_metrics.csv", index=False)
    source_summary.to_csv(args.output_dir / "boundary_leading_source_summary.csv", index=False)
    class_summary.to_csv(args.output_dir / "boundary_leading_margin_class_summary.csv", index=False)
    prediction.to_csv(args.output_dir / "boundary_leading_success_prediction.csv", index=False)
    pd.DataFrame([{**thresholds, "model": args.model, "layer": args.layer, "highway_source": args.highway_source, "highway_k": args.highway_k, "train_vectors": n_train}]).to_csv(
        args.output_dir / "boundary_leading_thresholds.csv", index=False
    )
    meta = {
        "script": "experiments/pure_af_geometry/analyze_boundary_leading_highways.py",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "layer": args.layer,
        "highway_source": args.highway_source,
        "eval_sources": parse_csv(args.eval_sources),
        "highway_k": args.highway_k,
        "margin_tol": args.margin_tol,
        "seed": args.seed,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_summary(args.output_dir, source_summary, class_summary, prediction)
    print(f"[DONE] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
