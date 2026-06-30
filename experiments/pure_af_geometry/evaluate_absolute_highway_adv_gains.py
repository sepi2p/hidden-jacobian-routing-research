#!/usr/bin/env python3
"""Image-free adversarial gain of signed hidden-representation highways.

This diagnostic ranks signed high-mobility highway directions without using
attack trajectories or clean images in the scoring step.  For a penultimate
feature direction v and a linear classifier head W, moving along v changes the
class-pair margin z_y - z_t by (W_y - W_t) v.  The adversarial gain for class
y against competitor t is therefore W_t v - W_y v.

For earlier hidden layers this exact state-free score is not available because
the downstream mapping is nonlinear and state-dependent.  The script therefore
supports the pooled ResNet layer4/penultimate case used by the current highway
experiments, and the logits case as an exact identity-head control.
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
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.evaluate_image_conditioned_highway_selector import (  # noqa: E402
    ArtifactStore,
    fit_highway_basis,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_corr(x: np.ndarray, y: np.ndarray, fn) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return float(fn(x[mask], y[mask]).statistic)


def topk_overlap(a: pd.DataFrame, a_col: str, b_col: str, k: int) -> int:
    aa = set(a.nsmallest(k, a_col)["route"].astype(str))
    bb = set(a.nsmallest(k, b_col)["route"].astype(str))
    return len(aa & bb)


def classifier_head_matrix(wrapper, layer: str, dim: int) -> tuple[np.ndarray, str]:
    if layer == "logits":
        return np.eye(dim, dtype=np.float32), "identity_logits_head"

    matches = []
    for name, module in wrapper.model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[1] == dim:
            matches.append((name, module))
    if not matches:
        raise RuntimeError(
            f"Could not find a linear classifier head with input dimension {dim}. "
            "For non-penultimate hidden layers, no exact image-free margin gain exists."
        )
    # Prefer the final matching linear layer.
    name, module = matches[-1]
    return module.weight.detach().cpu().numpy().astype(np.float32), name


def signed_routes_from_basis(basis: np.ndarray) -> pd.DataFrame:
    rows = []
    for pc in range(1, basis.shape[0] + 1):
        for sign, label in [(1, "+"), (-1, "-")]:
            rows.append({"pc": pc, "sign": sign, "route": f"pc{pc}{label}"})
    return pd.DataFrame(rows)


def route_gain_rows(routes: pd.DataFrame, basis: np.ndarray, head_w: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_classes = int(head_w.shape[0])
    route_rows = []
    pair_rows = []
    for route in routes.itertuples(index=False):
        v = int(route.sign) * basis[int(route.pc) - 1]
        v = v / max(float(np.linalg.norm(v)), 1e-12)
        dlogits = head_w @ v
        gains = []
        for y in range(n_classes):
            competitors = [t for t in range(n_classes) if t != y]
            pair_gains = np.array([float(dlogits[t] - dlogits[y]) for t in competitors], dtype=float)
            best_idx = int(np.argmax(pair_gains))
            best_t = int(competitors[best_idx])
            best_gain = float(pair_gains[best_idx])
            gains.append(best_gain)
            for t, gain in zip(competitors, pair_gains):
                pair_rows.append(
                    {
                        "route": str(route.route),
                        "pc": int(route.pc),
                        "sign": int(route.sign),
                        "source_class": int(y),
                        "target_class": int(t),
                        "class_pair_margin_drop": float(gain),
                    }
                )
            route_rows.append(
                {
                    "route": str(route.route),
                    "pc": int(route.pc),
                    "sign": int(route.sign),
                    "source_class": int(y),
                    "best_target_class": best_t,
                    "best_class_margin_drop": best_gain,
                    "true_logit_delta": float(dlogits[y]),
                    "best_target_logit_delta": float(dlogits[best_t]),
                }
            )

        pair_gain_matrix = np.array(
            [
                dlogits[t] - dlogits[y]
                for y in range(n_classes)
                for t in range(n_classes)
                if t != y
            ],
            dtype=float,
        )
        gains_arr = np.array(gains, dtype=float)
        route_rows.append(
            {
                "route": str(route.route),
                "pc": int(route.pc),
                "sign": int(route.sign),
                "source_class": -1,
                "best_target_class": -1,
                "best_class_margin_drop": float(gains_arr.mean()),
                "mean_best_margin_drop": float(gains_arr.mean()),
                "median_best_margin_drop": float(np.median(gains_arr)),
                "min_best_margin_drop": float(gains_arr.min()),
                "max_best_margin_drop": float(gains_arr.max()),
                "frac_classes_positive_best_drop": float((gains_arr > 0).mean()),
                "mean_all_pair_margin_drop": float(pair_gain_matrix.mean()),
                "mean_positive_pair_margin_drop": float(pair_gain_matrix[pair_gain_matrix > 0].mean())
                if np.any(pair_gain_matrix > 0)
                else 0.0,
                "max_pair_margin_drop": float(pair_gain_matrix.max()),
                "frac_pairs_positive_drop": float((pair_gain_matrix > 0).mean()),
                "logit_effect_l2": float(np.linalg.norm(dlogits)),
                "logit_effect_linf": float(np.max(np.abs(dlogits))),
                **{f"logit_delta_class_{c}": float(dlogits[c]) for c in range(n_classes)},
            }
        )

    per_class = pd.DataFrame([r for r in route_rows if r["source_class"] >= 0])
    summary = pd.DataFrame([r for r in route_rows if r["source_class"] == -1])
    summary = summary.sort_values("mean_best_margin_drop", ascending=False).reset_index(drop=True)
    summary["absolute_adv_gain_rank"] = np.arange(1, len(summary) + 1)
    summary["absolute_adv_gain_percentile"] = (summary["absolute_adv_gain_rank"] - 1) / max(len(summary) - 1, 1)
    return summary, per_class.merge(pd.DataFrame(pair_rows), on=["route", "pc", "sign", "source_class"], how="outer")


def compare_to_clean_rank(abs_df: pd.DataFrame, clean_rank_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not clean_rank_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    clean = pd.read_csv(clean_rank_path)
    merged = abs_df.merge(clean, on=["route", "pc", "sign"], how="inner", suffixes=("_absolute", "_clean"))
    rows = []
    pairs = [
        ("absolute_adv_gain_rank", "clean_intervention_rank"),
        ("mean_best_margin_drop", "mean_margin_drop"),
        ("mean_best_margin_drop", "mean_feature_speed"),
        ("mean_best_margin_drop", "mean_route_energy"),
    ]
    if "global_rank" in merged.columns:
        pairs.append(("absolute_adv_gain_rank", "global_rank"))
        pairs.append(("mean_best_margin_drop", "global_score"))
    for a, b in pairs:
        x = merged[a].to_numpy(float)
        y = merged[b].to_numpy(float)
        rows.append(
            {
                "x": a,
                "y": b,
                "n": int((np.isfinite(x) & np.isfinite(y)).sum()),
                "pearson": safe_corr(x, y, pearsonr),
                "spearman": safe_corr(x, y, spearmanr),
                "kendall": safe_corr(x, y, kendalltau),
            }
        )
    for k in [1, 3, 5, 10, 20]:
        if k <= len(merged):
            rows.append(
                {
                    "x": f"top{k}_absolute",
                    "y": f"top{k}_clean_intervention",
                    "n": len(merged),
                    "pearson": np.nan,
                    "spearman": float(topk_overlap(merged, "absolute_adv_gain_rank", "clean_intervention_rank", k)),
                    "kendall": np.nan,
                }
            )
    return merged, pd.DataFrame(rows)


def plot_summary(abs_df: pd.DataFrame, out_dir: Path) -> None:
    top = abs_df.nsmallest(min(20, len(abs_df)), "absolute_adv_gain_rank").copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(top["route"], top["mean_best_margin_drop"], color="#4c78a8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean best class-pair margin drop")
    ax.set_xlabel("Signed highway route")
    ax.set_title("Image-free highway adversarial gain")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "absolute_highway_adv_gain_top_routes.png", dpi=220)
    fig.savefig(out_dir / "absolute_highway_adv_gain_top_routes.pdf")
    plt.close(fig)


def simple_markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(format(value, floatfmt))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_note(
    out_dir: Path,
    args: argparse.Namespace,
    abs_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    head_name: str,
    n_train: int,
) -> None:
    lines = [
        "# Absolute Highway Adversarial Gain",
        "",
        "This diagnostic ranks signed highway directions without using PGD, Square, or clean images in the scoring step.",
        "",
        f"- model: `{args.model}`",
        f"- layer: `{args.layer}`",
        f"- highway source used to define routes: `{args.highway_source}`",
        f"- highway training vectors: `{n_train}`",
        f"- classifier head: `{head_name}`",
        "",
        "For a signed highway direction `v`, the classifier-head logit change is `Wv`. "
        "For source class `y` and competitor `t`, the one-step margin-drop derivative is `W_t v - W_y v`. "
        "The absolute score used here averages the best competitor gain over all source classes.",
        "",
        "## Top Routes",
        "",
        simple_markdown_table(abs_df[
            [
                "absolute_adv_gain_rank",
                "route",
                "mean_best_margin_drop",
                "median_best_margin_drop",
                "max_best_margin_drop",
                "frac_classes_positive_best_drop",
                "logit_effect_l2",
            ]
        ].head(12)),
        "",
    ]
    if not corr_df.empty:
        lines += [
            "## Relation to Clean-Start Intervention Ranking",
            "",
            simple_markdown_table(corr_df),
            "",
        ]
    lines += [
        "## Interpretation",
        "",
        "This is the closest available image-free highway gain for the pooled penultimate representation: it measures whether a direction is aligned with class-pair margin reduction under the final classifier head. "
        "It is not an attack trajectory statistic and should not be described as PGD- or Square-derived.",
        "",
        "However, it is still a classifier-head proxy. For earlier nonlinear layers there is no state-free adversarial gain, because the downstream Jacobian depends on the current activation state.",
        "",
    ]
    (out_dir / "absolute_highway_adv_gain_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/absolute_highway_adv_gain_bbb_resnet50_layer4"))
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--clean-ranking", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/interventional_highway_ranking_clean_c200/clean_interventional_highway_ranking.csv"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    store = ArtifactStore(args.input_dir)
    _mean, basis, n_train = fit_highway_basis(store, args.model, args.highway_source, args.layer, args.highway_k, args.seed)
    wrapper = load_model(args.model, torch.device("cpu")).eval()
    head_w, head_name = classifier_head_matrix(wrapper, args.layer, basis.shape[1])
    routes = signed_routes_from_basis(basis)
    abs_df, pair_df = route_gain_rows(routes, basis, head_w)
    merged, corr_df = compare_to_clean_rank(abs_df, args.clean_ranking)

    abs_df.to_csv(args.output_dir / "absolute_highway_adv_gain.csv", index=False)
    pair_df.to_csv(args.output_dir / "absolute_highway_class_pair_gains.csv", index=False)
    if not merged.empty:
        merged.to_csv(args.output_dir / "absolute_vs_clean_intervention_rank_comparison.csv", index=False)
    if not corr_df.empty:
        corr_df.to_csv(args.output_dir / "absolute_vs_clean_intervention_correlations.csv", index=False)
    plot_summary(abs_df, args.output_dir)
    metadata = {
        "script": "experiments/pure_af_geometry/evaluate_absolute_highway_adv_gains.py",
        "input_dir": str(args.input_dir),
        "model": args.model,
        "layer": args.layer,
        "highway_source": args.highway_source,
        "highway_k": args.highway_k,
        "highway_train_vectors": n_train,
        "head_name": head_name,
        "scoring": "image_free_linear_head_class_pair_margin_drop",
        "clean_ranking": str(args.clean_ranking),
        "seed": args.seed,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_note(args.output_dir, args, abs_df, corr_df, head_name, n_train)
    print(f"Wrote {args.output_dir}")
    print(abs_df.head(12).to_string(index=False))
    if not corr_df.empty:
        print(corr_df.to_string(index=False))


if __name__ == "__main__":
    main()
