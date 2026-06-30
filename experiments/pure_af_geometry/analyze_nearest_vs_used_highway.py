#!/usr/bin/env python3
"""Compare nearest clean-start adversarial highways with routes attacks use.

"Nearest" is approximated from clean-start one-step route candidates:
among routes that drop the true-class margin, choose the one with the largest
local route energy or feature speed.  The used route is the dominant signed
highway assignment of each observed attack step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def nearest_routes(candidates: pd.DataFrame) -> pd.DataFrame:
    adv = candidates[candidates.margin_drop > 0].copy()
    records = []
    for image_ord, g in adv.groupby("image_ord"):
        rec = {"image_ord": int(image_ord)}
        selectors = [
            ("closest_energy", "route_energy", False),
            ("closest_speed", "feature_speed", False),
            ("highest_reward", "margin_drop", False),
            ("global_best", "global_rank", True),
        ]
        for name, col, ascending in selectors:
            row = g.sort_values(col, ascending=ascending).iloc[0]
            rec[f"{name}_route"] = row.route
            rec[f"{name}_rank"] = int(row.global_rank)
            rec[f"{name}_margin_drop"] = float(row.margin_drop)
            rec[f"{name}_route_energy"] = float(row.route_energy)
            rec[f"{name}_feature_speed"] = float(row.feature_speed)
        records.append(rec)
    return pd.DataFrame(records)


def analyze(candidates: pd.DataFrame, usage: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    adv = candidates[candidates.margin_drop > 0].copy()
    nearest = nearest_routes(candidates)
    rows = []
    for source in sources:
        used = usage[
            (usage.source == source)
            & (usage.final_success == 1)
            & (usage.split == "test")
            & (usage.image_ord.isin(nearest.image_ord))
        ].copy()
        first = used.sort_values(["image_ord", "step"]).groupby("image_ord").first().reset_index()
        all_routes = used.groupby("image_ord")["dominant_route"].apply(lambda s: set(s)).reset_index(name="used_routes")
        merged = first.merge(all_routes, on="image_ord").merge(nearest, on="image_ord")
        for row in merged.itertuples(index=False):
            image_ord = int(row.image_ord)
            first_used = str(row.dominant_route)
            used_set = set(row.used_routes)
            cand = adv[adv.image_ord == image_ord].copy()
            out = {
                "source": source,
                "image_ord": image_ord,
                "first_used_route": first_used,
                "all_used_routes": "|".join(sorted(used_set)),
                "n_used_routes": len(used_set),
            }
            for name, col, ascending in [
                ("energy", "route_energy", False),
                ("speed", "feature_speed", False),
                ("reward", "margin_drop", False),
                ("global", "global_rank", True),
            ]:
                ranked = cand.sort_values(col, ascending=ascending).reset_index(drop=True)
                route_to_pos = {route: i + 1 for i, route in enumerate(ranked.route)}
                rank = route_to_pos.get(first_used, np.nan)
                out[f"first_rank_by_{name}"] = rank
                out[f"first_percentile_by_{name}"] = (rank - 1) / max(len(ranked) - 1, 1) if pd.notna(rank) else np.nan
            for selector in ["closest_energy", "closest_speed", "highest_reward", "global_best"]:
                route = getattr(row, f"{selector}_route")
                out[f"{selector}_route"] = route
                out[f"first_equals_{selector}"] = int(first_used == route)
                out[f"any_uses_{selector}"] = int(route in used_set)
            rows.append(out)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, g in df.groupby("source"):
        row = {"source": source, "n_images": len(g), "mean_n_used_routes": float(g.n_used_routes.mean())}
        for metric in ["energy", "speed", "reward", "global"]:
            row[f"mean_first_rank_by_{metric}"] = float(g[f"first_rank_by_{metric}"].mean())
            row[f"median_first_rank_by_{metric}"] = float(g[f"first_rank_by_{metric}"].median())
            row[f"mean_first_percentile_by_{metric}"] = float(g[f"first_percentile_by_{metric}"].mean())
        for selector in ["closest_energy", "closest_speed", "highest_reward", "global_best"]:
            row[f"first_equals_{selector}"] = float(g[f"first_equals_{selector}"].mean())
            row[f"any_uses_{selector}"] = float(g[f"any_uses_{selector}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidate-csv",
        default="analysis_outputs/pure_af_geometry/jacobian_null_response/image_conditioned_highway_selector_bbb_resnet50_c50/image_conditioned_selector_route_candidates.csv",
    )
    p.add_argument(
        "--usage-csv",
        default="analysis_outputs/pure_af_geometry/jacobian_null_response/ranked_boundary_highways_bbb_resnet50_c200_auto/ranked_highway_step_usage.csv",
    )
    p.add_argument("--sources", default="pgd,square")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/nearest_vs_used_highway")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidate_csv)
    usage = pd.read_csv(args.usage_csv)
    sources = [x.strip() for x in args.sources.split(",") if x.strip()]
    per_image = analyze(candidates, usage, sources)
    summary = summarize(per_image)
    per_image.to_csv(out_dir / "nearest_vs_used_highway_per_image.csv", index=False)
    summary.to_csv(out_dir / "nearest_vs_used_highway_summary.csv", index=False)
    lines = ["# Nearest Versus Used Highway", "", "## Summary", ""]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.source}`: first=closest_energy {row.first_equals_closest_energy:.3f}, "
            f"any=closest_energy {row.any_uses_closest_energy:.3f}, "
            f"first=highest_reward {row.first_equals_highest_reward:.3f}, "
            f"any=highest_reward {row.any_uses_highest_reward:.3f}"
        )
    (out_dir / "nearest_vs_used_highway_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"[SAVED] {out_dir}")


if __name__ == "__main__":
    main()
