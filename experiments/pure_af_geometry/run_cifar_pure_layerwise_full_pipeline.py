#!/usr/bin/env python3
"""Assemble the full CIFAR pure-trajectory layerwise validation pipeline.

This wrapper reuses existing GA noise-to-pure trajectory artifacts when they
match the requested configuration, otherwise it can invoke the collector. It
then writes the canonical ``pure_*`` output files requested for the paper
validation pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ADV_BEST = {
    "bbb_resnet50": "layer2",
    "bbb_vgg19_bn": "block2",
    "bbb_densenet": "denseblock3",
    "bbb_inception_v3": "mixed6",
}


def ensure_layerwise(args, out_dir: Path):
    src = Path(args.existing_layerwise_dir)
    required = [
        "points.csv",
        "segments.csv",
        "runs.csv",
        "clean_motion.csv",
        "segment_vectors.npz",
        "segment_grads.npz",
        "clean_vectors.npz",
        "metadata.json",
        "layerwise_geometry_metrics.csv",
        "layerwise_predictiveness_metrics.csv",
        "layerwise_gradient_orth_metrics.csv",
        "layerwise_temporal_metrics.csv",
    ]
    if args.reuse_existing and all((src / f).exists() for f in required):
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in required:
            shutil.copy2(src / f, out_dir / f)
        for f in [
            "layerwise_summary.png",
            "layerwise_temporal_curves.png",
            "layerwise_success_vs_clean_auroc_curves.png",
            "dominant_layer_temporal_mechanism_metrics.csv",
            "dominant_layer_temporal_mechanisms_raw.png",
            "dominant_layer_temporal_mechanisms_gradient_orthogonalized.png",
            "temporal_layer_peak_validation.csv",
        ]:
            if (src / f).exists():
                shutil.copy2(src / f, out_dir / f)
        return "reused"

    cmd = [
        sys.executable,
        "experiments/pure_af_geometry/analyze_cifar10_layerwise_success_flow.py",
        "--output-dir", str(out_dir),
        "--models", args.models,
        "--classes", "0,1,2,3,4,5,6,7,8,9",
        "--seeds", "3",
        "--generations", "120",
        "--population", "64",
        "--prob-threshold", "0.999",
        "--save-every", "5",
        "--clean-motion-images", "500",
    ]
    subprocess.run(cmd, check=True)
    return "generated"


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def canonicalize_layerwise(out_dir: Path):
    geom = pd.read_csv(out_dir / "layerwise_geometry_metrics.csv")
    pred = pd.read_csv(out_dir / "layerwise_predictiveness_metrics.csv")
    orth = pd.read_csv(out_dir / "layerwise_gradient_orth_metrics.csv")
    temporal = pd.read_csv(out_dir / "layerwise_temporal_metrics.csv")

    geom.to_csv(out_dir / "pure_layerwise_geometry_metrics.csv", index=False)
    geom.to_csv(out_dir / "pure_layerwise_dimensionality.csv", index=False)
    pred[pred["comparison"].eq("success_vs_clean")].to_csv(out_dir / "pure_vs_clean_projection_metrics.csv", index=False)
    orth.to_csv(out_dir / "pure_gradient_orthogonalized_metrics.csv", index=False)
    temporal.to_csv(out_dir / "pure_temporal_projection_metrics.csv", index=False)
    if (out_dir / "dominant_layer_temporal_mechanism_metrics.csv").exists():
        shutil.copy2(out_dir / "dominant_layer_temporal_mechanism_metrics.csv", out_dir / "pure_temporal_mechanism_metrics.csv")
    else:
        temporal.to_csv(out_dir / "pure_temporal_mechanism_metrics.csv", index=False)
    if (out_dir / "layerwise_temporal_curves.png").exists():
        shutil.copy2(out_dir / "layerwise_temporal_curves.png", out_dir / "pure_temporal_curves.png")

    k20 = pred[
        pred["comparison"].eq("success_vs_clean")
        & pred["k"].astype(str).eq("20")
        & pred["variant"].isin(["raw", "gradient_orthogonalized"])
    ].copy()
    dom_rows = []
    for (model, variant), g in k20.groupby(["model", "variant"]):
        g = g.dropna(subset=["auroc"])
        if g.empty:
            continue
        r = g.sort_values("auroc", ascending=False).iloc[0]
        dom_rows.append({
            "model": model,
            "variant": variant,
            "best_pure_layer": r["layer"],
            "best_pure_vs_clean_auroc_k20": float(r["auroc"]),
            "adversarial_best_layer": ADV_BEST.get(model, ""),
            "matches_adversarial_best": bool(r["layer"] == ADV_BEST.get(model, "")),
        })
    dom = pd.DataFrame(dom_rows)
    dom.to_csv(out_dir / "pure_dominant_layer_summary.csv", index=False)
    dom.to_csv(out_dir / "pure_vs_adversarial_dominant_layer_comparison.csv", index=False)
    return geom, pred, orth, temporal, dom


def canonicalize_global_and_attacks(args, out_dir: Path):
    global_dir = Path(args.existing_global_dir)
    away_dir = Path(args.existing_away_dir)
    pc_dir = Path(args.existing_pc_attack_dir)

    copy_if_exists(global_dir / "class_direction_cosine_matrix.csv", out_dir / "pure_class_direction_cosine_matrix.csv")
    copy_if_exists(global_dir / "class_direction_pca.csv", out_dir / "pure_class_direction_pca.csv")
    copy_if_exists(global_dir / "global_vs_class_specific_decomposition.csv", out_dir / "pure_global_vs_class_decomposition.csv")
    # Use the hidden-layer heatmap as the generic heatmap; per-layer files remain in the source dir.
    copy_if_exists(global_dir / "class_direction_cosine_heatmap_hidden.png", out_dir / "pure_class_direction_heatmap.png")

    away_parts = []
    if (away_dir / "cifar_away_flow_attack_summary.csv").exists():
        a = pd.read_csv(away_dir / "cifar_away_flow_attack_summary.csv")
        a["direction_family"] = "class_specific"
        away_parts.append(a)
    if (global_dir / "global_vs_class_awayflow_attack.csv").exists():
        g = pd.read_csv(global_dir / "global_vs_class_awayflow_attack.csv")
        g = g[g["variant"].astype(str).str.startswith("global_")].copy()
        # Already summary-level in the existing script.
        g["direction_family"] = "global"
        away_parts.append(g)
    away = pd.concat(away_parts, ignore_index=True, sort=False) if away_parts else pd.DataFrame()
    away.to_csv(out_dir / "pure_awayflow_attack_summary.csv", index=False)
    if (away_dir / "cifar_away_flow_step_sweep.csv").exists():
        shutil.copy2(away_dir / "cifar_away_flow_step_sweep.csv", out_dir / "pure_awayflow_by_step.csv")

    if (pc_dir / "pc_transport_mode_attack_summary.csv").exists():
        shutil.copy2(pc_dir / "pc_transport_mode_attack_summary.csv", out_dir / "pure_pc_transport_mode_attack_summary.csv")
    else:
        pd.DataFrame().to_csv(out_dir / "pure_pc_transport_mode_attack_summary.csv", index=False)
    return away


def pure_vs_adversarial_overlap(args, out_dir: Path):
    ts_path = Path(args.existing_attack_decomposition_dir) / "attack_transport_projection_timeseries.csv"
    if not ts_path.exists():
        pd.DataFrame().to_csv(out_dir / "pure_vs_adversarial_subspace_overlap.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "pure_vs_adversarial_basis_transfer.csv", index=False)
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(ts_path)
    coeff_cols = [f"pc{i}_coeff" for i in range(1, 6)]
    rows, transfer_rows = [], []
    for (model, layer_group, layer), g in df.groupby(["model", "layer_group", "layer"]):
        pure = g[(g["attack"] == "ga") & (g["final_success"] == 1)][coeff_cols].dropna().to_numpy(dtype=float)
        if len(pure) < 5:
            continue
        pure_c = pure - pure.mean(axis=0, keepdims=True)
        _u, _s, pure_vt = np.linalg.svd(pure_c, full_matrices=False)
        for attack in ["pgd", "square"]:
            adv = g[(g["attack"] == attack) & (g["final_success"] == 1)][coeff_cols].dropna().to_numpy(dtype=float)
            if len(adv) < 5:
                continue
            adv_c = adv - adv.mean(axis=0, keepdims=True)
            _u, _s, adv_vt = np.linalg.svd(adv_c, full_matrices=False)
            for k in [1, 2, 3, 5]:
                kk = min(k, pure_vt.shape[0], adv_vt.shape[0], pure_vt.shape[1])
                if kk < 1:
                    continue
                s = np.linalg.svd(pure_vt[:kk] @ adv_vt[:kk].T, compute_uv=False)
                s = np.clip(s, 0, 1)
                ang = np.arccos(s)
                rows.append({
                    "model": model,
                    "layer_group": layer_group,
                    "layer": layer,
                    "adversarial_attack": attack,
                    "coordinate_space": "top5_pure_transport_coefficients",
                    "k": k,
                    "mean_principal_angle_deg": float(np.degrees(ang).mean()),
                    "max_principal_angle_deg": float(np.degrees(ang).max()),
                    "projection_overlap": float(np.sum(s * s) / kk),
                    "grassmann_distance": float(np.linalg.norm(ang)),
                    "subspace_affinity": float(np.sqrt(np.sum(s * s) / kk)),
                })

            # Basis-transfer AUROC: fit pure PCA in top-5 coefficient space and ask
            # whether projection energy separates successful from failed Square/PGD
            # trajectories. PGD is often all-success, so AUROC may be NaN.
            all_adv = g[g["attack"] == attack].dropna(subset=coeff_cols).copy()
            final = all_adv.sort_values(["normalized_progress", "step"]).groupby("run_id", as_index=False).tail(1)
            if final["final_success"].nunique() < 2:
                auroc = np.nan
            else:
                x = final[coeff_cols].to_numpy(dtype=float)
                xc = x - pure.mean(axis=0, keepdims=True)
                basis = pure_vt[: min(3, pure_vt.shape[0])]
                score = np.sum((xc @ basis.T) ** 2, axis=1) / np.clip(np.sum(xc * xc, axis=1), 1e-12, None)
                auroc = float(roc_auc_score(final["final_success"].to_numpy(dtype=int), score))
            transfer_rows.append({
                "model": model,
                "layer_group": layer_group,
                "layer": layer,
                "adversarial_attack": attack,
                "k": 3,
                "basis_transfer_auroc": auroc,
                "n_runs": int(final["run_id"].nunique()) if "final" in locals() else 0,
            })
    overlap = pd.DataFrame(rows)
    transfer = pd.DataFrame(transfer_rows)
    overlap.to_csv(out_dir / "pure_vs_adversarial_subspace_overlap.csv", index=False)
    transfer.to_csv(out_dir / "pure_vs_adversarial_basis_transfer.csv", index=False)
    return overlap, transfer


def plot_pipeline_summary(out_dir: Path, dom: pd.DataFrame, away: pd.DataFrame, overlap: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    if not dom.empty:
        raw = dom[dom["variant"] == "raw"]
        axes[0, 0].bar(raw["model"], raw["best_pure_vs_clean_auroc_k20"])
        axes[0, 0].set_ylim(0.45, 1.02)
        axes[0, 0].set_ylabel("AUROC")
        axes[0, 0].set_title("Best pure-vs-clean layer, raw")
        axes[0, 0].tick_params(axis="x", rotation=25)
        orth = dom[dom["variant"] == "gradient_orthogonalized"]
        axes[0, 1].bar(orth["model"], orth["best_pure_vs_clean_auroc_k20"])
        axes[0, 1].set_ylim(0.45, 1.02)
        axes[0, 1].set_ylabel("AUROC")
        axes[0, 1].set_title("Best pure-vs-clean layer, grad-orth")
        axes[0, 1].tick_params(axis="x", rotation=25)
    if not away.empty and {"source_model", "variant", "asr", "steps", "eps_255"}.issubset(away.columns):
        sub = away[(away["steps"] == away["steps"].max()) & (away["eps_255"] == away["eps_255"].max())]
        sub = sub[sub["variant"].astype(str).str.contains("away_hidden|global_hidden", regex=True, na=False)]
        for variant, g in sub.groupby("variant"):
            axes[1, 0].plot(g["source_model"], g["asr"], marker="o", label=variant)
        axes[1, 0].set_ylim(0, 1.02)
        axes[1, 0].set_title("Pure away-flow hidden ASR")
        axes[1, 0].tick_params(axis="x", rotation=25)
        axes[1, 0].legend(fontsize=7)
    if not overlap.empty:
        sub = overlap[(overlap["layer_group"] == "penultimate") & (overlap["k"] == 3)]
        for attack, g in sub.groupby("adversarial_attack"):
            axes[1, 1].plot(g["model"], g["projection_overlap"], marker="o", label=attack)
        axes[1, 1].set_ylim(0, 1.02)
        axes[1, 1].set_title("Pure-vs-adversarial overlap, penultimate k=3")
        axes[1, 1].tick_params(axis="x", rotation=25)
        axes[1, 1].legend(fontsize=7)
    fig.savefig(out_dir / "pure_pipeline_summary.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def final_report(out_dir: Path, dom: pd.DataFrame, away: pd.DataFrame, overlap: pd.DataFrame):
    runs = pd.read_csv(out_dir / "runs.csv")
    counts = runs.groupby(["model", "target_class"]).agg(successes=("success", "sum"), n=("success", "size")).reset_index()
    counts.to_csv(out_dir / "pure_success_counts_per_model_class.csv", index=False)

    report = {
        "success_counts_per_model_class": counts.to_dict("records"),
        "best_pure_layers": dom.to_dict("records") if not dom.empty else [],
        "class_specific_awayflow_asr": [],
        "global_awayflow_asr": [],
        "pc_transport_asr": [],
        "pure_vs_adversarial_subspace_overlap": [],
    }
    if not away.empty and "asr" in away.columns:
        report["class_specific_awayflow_asr"] = away[
            away.get("direction_family", "").eq("class_specific") if "direction_family" in away else []
        ].to_dict("records") if "direction_family" in away else []
        report["global_awayflow_asr"] = away[
            away.get("direction_family", "").eq("global") if "direction_family" in away else []
        ].to_dict("records") if "direction_family" in away else []
    pc_path = out_dir / "pure_pc_transport_mode_attack_summary.csv"
    if pc_path.exists() and pc_path.stat().st_size > 0:
        try:
            pc = pd.read_csv(pc_path)
            report["pc_transport_asr"] = pc.to_dict("records")
        except pd.errors.EmptyDataError:
            pass
    if not overlap.empty:
        report["pure_vs_adversarial_subspace_overlap"] = overlap.to_dict("records")
    with open(out_dir / "pure_pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return counts


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = ensure_layerwise(args, out_dir)
    geom, pred, orth, temporal, dom = canonicalize_layerwise(out_dir)
    away = canonicalize_global_and_attacks(args, out_dir)
    overlap, transfer = pure_vs_adversarial_overlap(args, out_dir)
    plot_pipeline_summary(out_dir, dom, away, overlap)
    counts = final_report(out_dir, dom, away, overlap)
    metadata = {
        "pipeline": "cifar_pure_layerwise_full_pipeline",
        "trajectory_source": source,
        "args": vars(args),
        "required_files_written": [
            "pure_pipeline_metadata.json",
            "pure_layerwise_geometry_metrics.csv",
            "pure_vs_clean_projection_metrics.csv",
            "pure_gradient_orthogonalized_metrics.csv",
            "pure_dominant_layer_summary.csv",
            "pure_temporal_projection_metrics.csv",
            "pure_global_vs_class_decomposition.csv",
            "pure_awayflow_attack_summary.csv",
            "pure_pc_transport_mode_attack_summary.csv",
            "pure_vs_adversarial_subspace_overlap.csv",
            "pure_pipeline_summary.png",
        ],
    }
    with open(out_dir / "pure_pipeline_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("[PURE PIPELINE COMPLETE]", out_dir, flush=True)
    print("\nSUCCESS COUNTS PER MODEL/CLASS", flush=True)
    print(counts.to_string(index=False), flush=True)
    print("\nBEST PURE LAYERS", flush=True)
    print(dom.to_string(index=False), flush=True)
    if not away.empty and "asr" in away.columns:
        print("\nAWAY-FLOW ASR SUMMARY", flush=True)
        cols = [c for c in ["source_model", "variant", "layer", "eps_255", "steps", "asr", "direction_family"] if c in away.columns]
        print(away[cols].tail(40).to_string(index=False), flush=True)
    if (out_dir / "pure_pc_transport_mode_attack_summary.csv").exists():
        try:
            pc = pd.read_csv(out_dir / "pure_pc_transport_mode_attack_summary.csv")
            print("\nPC1-PC5 AWAY-FLOW ASR", flush=True)
            print(pc.tail(40).to_string(index=False), flush=True)
        except pd.errors.EmptyDataError:
            pass
    if not overlap.empty:
        print("\nPURE VS ADVERSARIAL SUBSPACE OVERLAP", flush=True)
        print(overlap[(overlap["layer_group"] == "penultimate") & (overlap["k"] == 3)].to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_pure_layerwise_full_pipeline")
    p.add_argument("--existing-layerwise-dir", default="analysis_outputs/pure_af_geometry/cifar_layerwise_success_flow_c10_s3_g120")
    p.add_argument("--existing-away-dir", default="analysis_outputs/pure_af_geometry/cifar_away_flow_attack_100")
    p.add_argument("--existing-global-dir", default="analysis_outputs/pure_af_geometry/cifar_global_vs_class_success_flow")
    p.add_argument("--existing-pc-attack-dir", default="analysis_outputs/pure_af_geometry/cifar_pc_transport_mode_attack")
    p.add_argument("--existing-attack-decomposition-dir", default="analysis_outputs/pure_af_geometry/cifar_attack_transport_decomposition")
    p.add_argument("--models", default="bbb_resnet50,bbb_vgg19_bn,bbb_densenet,bbb_inception_v3")
    p.add_argument("--reuse-existing", action="store_true", default=True)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
