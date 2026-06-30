#!/usr/bin/env python3
"""Post-process transport-hybrid black-box benchmark runs.

The script expects output directories produced by
`evaluate_square_learned_correction_transfer.py`, e.g.

    resnet50_to_bbb_vgg19_bn_e2_q100_n200/

It builds the application tables requested for Benchmark A:

* ASR curve table by target.
* Q=100 delta table against CE/transport alternating.
* Success-overlap tables.
* Paired bootstrap CIs for ASR differences.
* Proposal acceptance/margin-improvement diagnostics when curve logs exist.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = [
    "vanilla_square",
    "nes",
    "random_coeff_square",
    "jvp_coeff_square",
    "one_shot_surrogate_pgd",
    "coeff_only",
    "target_accepted_ce",
    "target_accepted_transport",
    "square_init_coeff20",
    "square_init_coeff30",
    "ce_transport_alt",
]

METHOD_LABELS = {
    "vanilla_square": "Square",
    "nes": "NES",
    "random_coeff_square": "random coeff Square",
    "jvp_coeff_square": "JVP coeff Square",
    "one_shot_surrogate_pgd": "one-shot surrogate PGD",
    "coeff_only": "coeff only",
    "target_accepted_ce": "target-accepted CE",
    "target_accepted_transport": "target-accepted transport",
    "square_init_coeff10": "square init coeff10",
    "square_init_coeff20": "square init coeff20",
    "square_init_coeff30": "square init coeff30",
    "ce_transport_alt": "CE/transport alternating",
}

DELTA_BASE = "ce_transport_alt"
DELTA_COMPARISONS = [
    "vanilla_square",
    "nes",
    "one_shot_surrogate_pgd",
    "target_accepted_ce",
    "target_accepted_transport",
    "random_coeff_square",
    "jvp_coeff_square",
]

OVERLAP_PAIRS = [
    ("target_accepted_ce", "target_accepted_transport"),
    ("ce_transport_alt", "target_accepted_ce"),
    ("ce_transport_alt", "target_accepted_transport"),
    ("ce_transport_alt", "one_shot_surrogate_pgd"),
    ("square_init_coeff30", "coeff_only"),
]


def parse_run_dir(path: Path) -> dict[str, object] | None:
    m = re.search(r"resnet50_to_(?P<target>.+)_e(?P<eps>\d+(?:\.\d+)?)_q(?P<budget>\d+)_n(?P<n>\d+)$", path.name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "target": d["target"],
        "eps": float(d["eps"]),
        "query_budget": int(d["budget"]),
        "n_config": int(d["n"]),
    }


def load_results(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    curves = []
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        info = parse_run_dir(run_dir)
        if info is None:
            continue
        result_path = run_dir / "square_learned_correction_transfer_results.csv"
        if result_path.exists():
            df = pd.read_csv(result_path)
            for k, v in info.items():
                df[k] = v
            results.append(df)
        curve_path = run_dir / "square_learned_correction_transfer_curves.csv"
        if curve_path.exists():
            c = pd.read_csv(curve_path)
            for k, v in info.items():
                c[k] = v
            curves.append(c)
    if not results:
        raise RuntimeError(f"No result CSVs found under {base}")
    result_df = pd.concat(results, ignore_index=True)
    curve_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return result_df, curve_df


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def asr_curve_table(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["target", "method", "query_budget"], as_index=False)
        .agg(
            asr=("success", "mean"),
            n=("success", "size"),
            mean_query_failure_counted=("success_query", lambda x: np.where(np.asarray(x) >= 0, np.asarray(x), np.nan).mean()),
        )
    )
    pivot = g.pivot_table(index=["target", "method"], columns="query_budget", values="asr")
    for b in [25, 50, 100, 250]:
        if b not in pivot.columns:
            pivot[b] = np.nan
    pivot = pivot[[25, 50, 100, 250]].reset_index()
    pivot["method_label"] = pivot["method"].map(method_label)
    pivot["method_order"] = pivot["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(999).astype(int)
    pivot = pivot.sort_values(["target", "method_order", "method"]).drop(columns=["method_order"])
    pivot = pivot.rename(columns={25: "ASR@25", 50: "ASR@50", 100: "ASR@100", 250: "ASR@250"})
    return pivot[["target", "method", "method_label", "ASR@25", "ASR@50", "ASR@100", "ASR@250"]]


def delta_table(df: pd.DataFrame, budget: int = 100) -> pd.DataFrame:
    rows = []
    sub = df[df.query_budget == budget]
    for target, g in sub.groupby("target"):
        asr = g.groupby("method")["success"].mean().to_dict()
        base = asr.get(DELTA_BASE, np.nan)
        for other in DELTA_COMPARISONS:
            rows.append(
                {
                    "target": target,
                    "query_budget": budget,
                    "comparison": f"{method_label(DELTA_BASE)} - {method_label(other)}",
                    "delta_asr": base - asr.get(other, np.nan),
                    "delta_asr_percent_points": 100.0 * (base - asr.get(other, np.nan)),
                }
            )
    return pd.DataFrame(rows)


def success_overlap_table(df: pd.DataFrame, budget: int = 100) -> pd.DataFrame:
    rows = []
    sub = df[df.query_budget == budget]
    for target, g in sub.groupby("target"):
        pivot = g.pivot_table(index="image_id", columns="method", values="success", aggfunc="max").fillna(0).astype(int)
        for a, b in OVERLAP_PAIRS:
            if a not in pivot.columns or b not in pivot.columns:
                rows.append({"target": target, "query_budget": budget, "pair": f"{method_label(a)} vs {method_label(b)}", "A_only": np.nan, "B_only": np.nan, "both": np.nan, "neither": np.nan})
                continue
            av = pivot[a].to_numpy().astype(bool)
            bv = pivot[b].to_numpy().astype(bool)
            rows.append(
                {
                    "target": target,
                    "query_budget": budget,
                    "pair": f"{method_label(a)} vs {method_label(b)}",
                    "A": method_label(a),
                    "B": method_label(b),
                    "A_only": int(np.logical_and(av, ~bv).sum()),
                    "B_only": int(np.logical_and(~av, bv).sum()),
                    "both": int(np.logical_and(av, bv).sum()),
                    "neither": int(np.logical_and(~av, ~bv).sum()),
                    "n": int(len(pivot)),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_ci(diff: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0:
        return np.nan, np.nan
    boots = []
    for _ in range(reps):
        idx = rng.integers(0, len(diff), len(diff))
        boots.append(float(diff[idx].mean()))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_ci_table(df: pd.DataFrame, budgets: list[int], reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for (target, budget), g in df[df.query_budget.isin(budgets)].groupby(["target", "query_budget"]):
        pivot = g.pivot_table(index="image_id", columns="method", values="success", aggfunc="max").fillna(0)
        methods = [
            "vanilla_square",
            "nes",
            "one_shot_surrogate_pgd",
            "target_accepted_ce",
            "target_accepted_transport",
            "random_coeff_square",
            "jvp_coeff_square",
            "square_init_coeff20",
            "square_init_coeff30",
            "coeff_only",
        ]
        if DELTA_BASE not in pivot.columns:
            continue
        base = pivot[DELTA_BASE].to_numpy(dtype=float)
        for other in methods:
            if other not in pivot.columns:
                continue
            diff = base - pivot[other].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(diff, reps, seed + int(budget) + len(target) + len(other))
            rows.append(
                {
                    "target": target,
                    "query_budget": budget,
                    "comparison": f"{method_label(DELTA_BASE)} - {method_label(other)}",
                    "paired_delta_asr": float(diff.mean()),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n": int(len(diff)),
                }
            )
    return pd.DataFrame(rows)


def proposal_diagnostics(curves: pd.DataFrame) -> pd.DataFrame:
    if curves.empty or "proposal" not in curves.columns:
        return pd.DataFrame()
    c = curves.copy()
    c = c.sort_values(["target", "query_budget", "method", "image_id", "query"]).reset_index(drop=True)
    prev = c.groupby(["target", "query_budget", "method", "image_id"])["target_margin"].shift(1)
    c["observed_margin_drop"] = (prev - c["target_margin"]).clip(lower=0)
    if "improvement" in c.columns:
        c["improvement"] = pd.to_numeric(c["improvement"], errors="coerce")
        c["proposal_margin_drop"] = c["improvement"].where(c["improvement"].notna(), c["observed_margin_drop"])
    else:
        c["proposal_margin_drop"] = c["observed_margin_drop"]
    return (
        c.groupby(["target", "query_budget", "method", "proposal"], dropna=False)
        .agg(
            n_proposals=("accepted", "size"),
            accept_rate=("accepted", "mean"),
            mean_margin_drop=("proposal_margin_drop", "mean"),
            mean_margin_drop_accepted=("proposal_margin_drop", lambda x: np.nanmean(np.asarray(x)[np.asarray(c.loc[x.index, "accepted"]) == 1]) if (np.asarray(c.loc[x.index, "accepted"]) == 1).any() else np.nan),
        )
        .reset_index()
    )


def df_to_md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_No rows._"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
        else:
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(work.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_markdown(asr: pd.DataFrame, delta: pd.DataFrame, overlap: pd.DataFrame, ci: pd.DataFrame, out: Path) -> None:
    lines = ["# Transport Hybrid Benchmark Summary", ""]
    lines += ["## ASR Curves", ""]
    for target, g in asr.groupby("target"):
        lines += [f"### {target}", "", df_to_md(g.drop(columns=["target", "method"])), ""]
    lines += ["## Q=100 Delta Table", "", df_to_md(delta), ""]
    lines += ["## Q=100 Success Overlap", "", df_to_md(overlap, floatfmt=".0f"), ""]
    lines += ["## Paired Bootstrap CIs", "", df_to_md(ci), ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="analysis_outputs/hidden_jacobian_routing/transport_hybrid_benchmark_a")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    base = Path(args.base_dir)
    out = Path(args.output_dir) if args.output_dir else base / "posthoc_summary"
    out.mkdir(parents=True, exist_ok=True)

    results, curves = load_results(base)
    results.to_csv(out / "combined_per_image_results.csv", index=False)
    if not curves.empty:
        curves.to_csv(out / "combined_query_curves.csv", index=False)

    asr = asr_curve_table(results)
    delta = delta_table(results, 100)
    overlap = success_overlap_table(results, 100)
    ci = paired_ci_table(results, [100, 250], args.bootstrap_reps, args.seed)
    diag = proposal_diagnostics(curves)

    asr.to_csv(out / "asr_curve_table_by_target.csv", index=False)
    delta.to_csv(out / "delta_table_q100.csv", index=False)
    overlap.to_csv(out / "success_overlap_table_q100.csv", index=False)
    ci.to_csv(out / "paired_bootstrap_ci_q100_q250.csv", index=False)
    if not diag.empty:
        diag.to_csv(out / "proposal_acceptance_margin_diagnostics.csv", index=False)
    write_markdown(asr, delta, overlap, ci, out / "transport_hybrid_benchmark_summary.md")

    print(f"[OK] Wrote summary tables to {out}")
    print(f"[INFO] Runs summarized: {results[['target', 'query_budget']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
