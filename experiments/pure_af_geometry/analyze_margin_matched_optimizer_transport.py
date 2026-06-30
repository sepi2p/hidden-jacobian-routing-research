#!/usr/bin/env python3
"""Margin-matched optimizer transport-coordinate similarity.

This analysis addresses a reviewer-facing alternative explanation: optimizer
signatures may look similar only because successful attacks approach similar
decision-boundary regions. We therefore compare optimizer signatures within
matched margin and trajectory-progress bins, using only saved trajectory rows.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_IN = Path("analysis_outputs/pure_af_geometry/cifar_expanded_optimizer_transport/attack_axis_projection_timeseries.csv")
DEFAULT_OUT = Path("analysis_outputs/pure_af_geometry/cifar_expanded_optimizer_transport/margin_matched")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def pc_energy_signature(df: pd.DataFrame, k: int) -> np.ndarray:
    vals = []
    for i in range(1, k + 1):
        c = df[f"pc{i}_coeff"].to_numpy(dtype=np.float64)
        vals.append(float(np.mean(c * c)))
    vec = np.asarray(vals, dtype=np.float64)
    s = float(vec.sum())
    if s > 1e-12:
        vec = vec / s
    return vec


def add_margin_bins(df: pd.DataFrame, n_bins: int, group_cols: list[str]) -> pd.DataFrame:
    out = []
    for _, g in df.groupby(group_cols, dropna=False):
        g = g.copy()
        if g["margin"].nunique() < 2 or len(g) < n_bins:
            g["margin_bin"] = 0
        else:
            try:
                g["margin_bin"] = pd.qcut(g["margin"], q=n_bins, labels=False, duplicates="drop")
            except ValueError:
                g["margin_bin"] = 0
        out.append(g)
    return pd.concat(out, ignore_index=True)


def summarize_pairwise(sig: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, g in sig.groupby(["model", "layer_group", "time_bin", "margin_bin"], dropna=False):
        attacks = sorted(g["attack"].unique())
        if len(attacks) < 2:
            continue
        by_attack = {r["attack"]: np.asarray(r["signature"], dtype=np.float64) for _, r in g.iterrows()}
        for a, b in combinations(attacks, 2):
            rows.append({
                "model": keys[0],
                "layer_group": keys[1],
                "time_bin": int(keys[2]),
                "margin_bin": int(keys[3]),
                "attack_a": a,
                "attack_b": b,
                "cosine": cosine(by_attack[a], by_attack[b]),
                "n_a": int(g.loc[g["attack"] == a, "n_rows"].iloc[0]),
                "n_b": int(g.loc[g["attack"] == b, "n_rows"].iloc[0]),
            })
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        return pairwise, pairwise
    summary = pairwise.groupby(["attack_a", "attack_b"], dropna=False).agg(
        settings=("cosine", "count"),
        mean=("cosine", "mean"),
        median=("cosine", "median"),
        min=("cosine", "min"),
        max=("cosine", "max"),
    ).reset_index().sort_values(["mean", "settings"], ascending=[False, False])
    return pairwise, summary


def write_latex_selected(summary: pd.DataFrame, out_path: Path) -> None:
    selected = [
        ("pgd", "square"),
        ("pgd", "nes"),
        ("square", "signhunter"),
        ("pgd", "bandit"),
        ("pgd", "random_search"),
        ("mi_fgsm", "ni_fgsm"),
        ("pgd", "mi_fgsm"),
    ]
    rows = []
    for a, b in selected:
        mask = ((summary["attack_a"] == a) & (summary["attack_b"] == b)) | (
            (summary["attack_a"] == b) & (summary["attack_b"] == a)
        )
        hit = summary.loc[mask]
        if hit.empty:
            continue
        r = hit.iloc[0]
        rows.append((a.replace("_", "-"), b.replace("_", "-"), int(r["settings"]), r["mean"], r["median"], r["min"], r["max"]))
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Attack A & Attack B & Settings & Mean & Median & Min & Max \\",
        r"\midrule",
    ]
    for a, b, settings, mean, median, mn, mx in rows:
        lines.append(f"{a} & {b} & {settings} & {mean:.3f} & {median:.3f} & {mn:.3f} & {mx:.3f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--margin-bins", type=int, default=5)
    ap.add_argument("--min-rows", type=int, default=3)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    df = df[(df["step"] > 0) & (df["final_success"] == 1)].copy()
    df = df[np.isfinite(df["margin"])].copy()
    df = add_margin_bins(df, args.margin_bins, ["model", "layer_group", "time_bin"])

    sig_rows = []
    group_cols = ["model", "layer_group", "time_bin", "margin_bin", "attack"]
    for keys, g in df.groupby(group_cols, dropna=False):
        if len(g) < args.min_rows:
            continue
        sig_rows.append({
            "model": keys[0],
            "layer_group": keys[1],
            "time_bin": int(keys[2]),
            "margin_bin": int(keys[3]),
            "attack": keys[4],
            "n_rows": int(len(g)),
            "mean_margin": float(g["margin"].mean()),
            "signature": pc_energy_signature(g, args.top_k),
        })
    sig = pd.DataFrame(sig_rows)
    sig_out = sig.copy()
    for i in range(args.top_k):
        sig_out[f"pc{i+1}_energy_frac"] = sig_out["signature"].map(lambda x, i=i: float(x[i]))
    sig_out = sig_out.drop(columns=["signature"])
    sig_out.to_csv(args.out_dir / "margin_matched_optimizer_signatures.csv", index=False)

    pairwise, summary = summarize_pairwise(sig)
    pairwise.to_csv(args.out_dir / "margin_matched_optimizer_pairwise.csv", index=False)
    summary.to_csv(args.out_dir / "margin_matched_optimizer_similarity_summary.csv", index=False)
    write_latex_selected(summary, args.out_dir / "table_margin_matched_optimizer_similarity.tex")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
