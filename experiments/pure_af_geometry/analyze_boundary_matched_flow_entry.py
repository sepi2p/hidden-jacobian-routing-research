#!/usr/bin/env python3
"""Boundary-matched success-flow entry analysis.

This diagnostic tests whether the on-flow/off-flow effect in the flow-entry
experiments can be explained by a simple boundary-proximity proxy. It reuses
saved cross-attack entry summaries and tracks, then compares on-flow and
off-flow images after matching on starting margin and compares trajectory rows
within current-margin bins before success.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("analysis_outputs/pure_af_geometry/offflow_entry_diagnostic")
DEFAULT_OUTPUT = Path("analysis_outputs/pure_af_geometry/boundary_matched_flow_entry")


def parse_stem(path: Path) -> dict[str, str | int]:
    m = re.match(
        r"cross_entry_(?P<model>.+?)_(?P<layer_group>[^_]+)_class(?P<class_id>\d+)_"
        r"basis-(?P<basis_attack>[^_]+)_test-(?P<test_attack>[^_]+)_k(?P<k>\d+)_n(?P<n>\d+)"
        r"(?:_(?P<selection>margin_matched))?_summary",
        path.stem,
    )
    if m is None:
        raise ValueError(f"Unexpected summary name: {path.name}")
    out: dict[str, str | int] = m.groupdict()
    out["class_id"] = int(out["class_id"])
    out["k"] = int(out["k"])
    out["n"] = int(out["n"])
    if out.get("selection") is None:
        out["selection"] = "extreme"
    return out


def paired_nearest_match(df: pd.DataFrame, covariate: str = "start_margin") -> pd.DataFrame:
    off = df[df["group"] == "offflow"].copy().reset_index(drop=True)
    on = df[df["group"] == "onflow"].copy().reset_index(drop=True)
    if off.empty or on.empty:
        return pd.DataFrame()

    pairs = []
    used_on: set[int] = set()
    for off_i, r in off.sort_values(covariate).iterrows():
        available = on.loc[[i for i in on.index if i not in used_on]]
        if available.empty:
            break
        distances = (available[covariate] - r[covariate]).abs()
        on_i = int(distances.idxmin())
        used_on.add(on_i)
        s = on.loc[on_i]
        pairs.append(
            {
                "pair_id": len(pairs),
                "off_dataset_idx": int(r["dataset_idx"]),
                "on_dataset_idx": int(s["dataset_idx"]),
                "off_start_margin": float(r[covariate]),
                "on_start_margin": float(s[covariate]),
                "abs_margin_gap": float(abs(s[covariate] - r[covariate])),
                "off_initial_pe": float(r["initial_projection_energy"]),
                "on_initial_pe": float(s["initial_projection_energy"]),
                "off_success": int(r["success"]),
                "on_success": int(s["success"]),
                "off_success_step": float(r["success_step"]) if pd.notna(r["success_step"]) else np.nan,
                "on_success_step": float(s["success_step"]) if pd.notna(s["success_step"]) else np.nan,
                "off_final_margin": float(r["final_margin"]),
                "on_final_margin": float(s["final_margin"]),
            }
        )
    return pd.DataFrame(pairs)


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        stats.append(float(np.mean(sample)))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def sign_flip_pvalue(values: np.ndarray, seed: int, n_perm: int = 20000) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    obs = abs(float(np.mean(values)))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(values), replace=True)
        if abs(float(np.mean(values * signs))) >= obs:
            count += 1
    return float((count + 1) / (n_perm + 1))


def summarize_pairs(pairs: pd.DataFrame, context: dict[str, str | int], steps: int, seed: int, n_boot: int) -> dict:
    if pairs.empty:
        return {}
    censored = float(steps + 1)
    off_steps = pairs["off_success_step"].fillna(censored).to_numpy(dtype=np.float64)
    on_steps = pairs["on_success_step"].fillna(censored).to_numpy(dtype=np.float64)
    success_diff = pairs["on_success"].to_numpy(dtype=np.float64) - pairs["off_success"].to_numpy(dtype=np.float64)
    step_diff = on_steps - off_steps
    margin_gap = pairs["on_start_margin"].to_numpy(dtype=np.float64) - pairs["off_start_margin"].to_numpy(dtype=np.float64)
    pe_diff = pairs["on_initial_pe"].to_numpy(dtype=np.float64) - pairs["off_initial_pe"].to_numpy(dtype=np.float64)
    success_lo, success_hi = bootstrap_ci(success_diff, n_boot, seed)
    step_lo, step_hi = bootstrap_ci(step_diff, n_boot, seed + 17)
    return {
        **context,
        "n_pairs": int(len(pairs)),
        "mean_abs_start_margin_gap": float(pairs["abs_margin_gap"].mean()),
        "median_abs_start_margin_gap": float(pairs["abs_margin_gap"].median()),
        "off_success_rate": float(pairs["off_success"].mean()),
        "on_success_rate": float(pairs["on_success"].mean()),
        "paired_success_diff": float(success_diff.mean()),
        "paired_success_diff_ci_low": success_lo,
        "paired_success_diff_ci_high": success_hi,
        "paired_success_p_signflip": sign_flip_pvalue(success_diff, seed + 31),
        "off_median_success_or_censored_step": float(np.median(off_steps)),
        "on_median_success_or_censored_step": float(np.median(on_steps)),
        "paired_censored_step_diff": float(step_diff.mean()),
        "paired_censored_step_diff_ci_low": step_lo,
        "paired_censored_step_diff_ci_high": step_hi,
        "paired_step_p_signflip": sign_flip_pvalue(step_diff, seed + 47),
        "mean_start_margin_diff_on_minus_off": float(margin_gap.mean()),
        "mean_initial_pe_diff_on_minus_off": float(pe_diff.mean()),
    }


def load_tracks_for_summary(summary_path: Path) -> pd.DataFrame | None:
    track_path = summary_path.with_name(summary_path.name.replace("_summary.csv", "_tracks.csv"))
    if not track_path.exists():
        return None
    return pd.read_csv(track_path)


def step_level_margin_matched(tracks: pd.DataFrame, context: dict[str, str | int], bins: int) -> pd.DataFrame:
    rows = []
    t = tracks.copy()
    t = t[(t["step"] > 0) & np.isfinite(t["margin"]) & np.isfinite(t["learned_projection_energy"])].copy()
    if t.empty:
        return pd.DataFrame()

    # Only pre-success rows: once the first successful row is reached, later rows
    # are not present in the saved tracks. Keep non-crossing rows for the boundary
    # proxy and analyze the crossing row separately through image-level summaries.
    t = t[t["success"] == 0].copy()
    if t.empty:
        return pd.DataFrame()
    try:
        t["margin_bin"] = pd.qcut(t["margin"], q=bins, labels=False, duplicates="drop")
    except ValueError:
        t["margin_bin"] = 0

    for margin_bin, g in t.groupby("margin_bin", dropna=False):
        off = g[g["group"] == "offflow"]["learned_projection_energy"].to_numpy(dtype=np.float64)
        on = g[g["group"] == "onflow"]["learned_projection_energy"].to_numpy(dtype=np.float64)
        if len(off) == 0 or len(on) == 0:
            continue
        rows.append(
            {
                **context,
                "margin_bin": int(margin_bin),
                "n_off_rows": int(len(off)),
                "n_on_rows": int(len(on)),
                "mean_margin": float(g["margin"].mean()),
                "off_mean_learned_pe": float(np.mean(off)),
                "on_mean_learned_pe": float(np.mean(on)),
                "on_minus_off_mean_learned_pe": float(np.mean(on) - np.mean(off)),
                "off_median_learned_pe": float(np.median(off)),
                "on_median_learned_pe": float(np.median(on)),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return
    labels = [
        f"c{r.class_id} {r.basis_attack}->{r.test_attack}"
        for r in summary.itertuples(index=False)
    ]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)
    axes[0].bar(x, summary["paired_success_diff"], color="#2563eb")
    axes[0].axhline(0, color="#111827", lw=0.8)
    axes[0].set_ylabel("matched ASR difference\n(on-flow - off-flow)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].bar(x, summary["paired_censored_step_diff"], color="#7c3aed")
    axes[1].axhline(0, color="#111827", lw=0.8)
    axes[1].set_ylabel("matched censored step difference\n(on-flow - off-flow)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    fig.suptitle("Boundary-matched flow-entry effect", fontsize=12)
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--include-wrong-class", action="store_true")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--margin-bins", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_paths = sorted((args.input_dir / "cross_attack").glob("*_summary.csv"))
    summary_paths += sorted(args.input_dir.glob("*_summary.csv"))
    if args.include_wrong_class:
        summary_paths += sorted((args.input_dir / "wrong_class").glob("*_summary.csv"))

    pair_tables = []
    summary_rows = []
    step_tables = []
    for summary_path in summary_paths:
        context = parse_stem(summary_path)
        context["source"] = "wrong_class" if "wrong_class" in summary_path.parts else "cross_attack"
        df = pd.read_csv(summary_path)
        pairs = paired_nearest_match(df)
        if pairs.empty:
            continue
        for key, value in context.items():
            pairs[key] = value
        pair_tables.append(pairs)
        summary_rows.append(summarize_pairs(pairs, context, args.steps, args.seed + len(summary_rows) * 101, args.bootstrap))

        tracks = load_tracks_for_summary(summary_path)
        if tracks is not None:
            step = step_level_margin_matched(tracks, context, args.margin_bins)
            if not step.empty:
                step_tables.append(step)

    pairs_out = pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame()
    summary_out = pd.DataFrame(summary_rows)
    step_out = pd.concat(step_tables, ignore_index=True) if step_tables else pd.DataFrame()

    pairs_out.to_csv(args.output_dir / "boundary_matched_flow_entry_pairs.csv", index=False)
    summary_out.to_csv(args.output_dir / "boundary_matched_flow_entry_summary.csv", index=False)
    step_out.to_csv(args.output_dir / "current_margin_bin_projection_energy.csv", index=False)
    plot_summary(summary_out[summary_out["source"] == "cross_attack"].copy(), args.output_dir / "boundary_matched_flow_entry_effects")

    metadata = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "n_summary_files": len(summary_paths),
        "n_pairs": int(len(pairs_out)),
        "n_step_bins": int(len(step_out)),
        "steps_censor_value": int(args.steps + 1),
        "matching": "nearest-neighbor without replacement on start_margin within each summary file",
        "step_level_check": "pre-success rows binned by current margin; compares learned projection energy within bins",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if not summary_out.empty:
        print(summary_out.to_string(index=False))
    if not step_out.empty:
        print("\nStep-level margin-bin mean effect:")
        print(
            step_out.groupby(["source", "basis_attack", "test_attack"], as_index=False)
            .agg(
                bins=("margin_bin", "count"),
                mean_on_minus_off_pe=("on_minus_off_mean_learned_pe", "mean"),
                min_on_minus_off_pe=("on_minus_off_mean_learned_pe", "min"),
                max_on_minus_off_pe=("on_minus_off_mean_learned_pe", "max"),
            )
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
