#!/usr/bin/env python3
"""Decompose PGD, Square, and GA trajectories in learned transport PCs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar_global_vs_class_success_flow import LAYER_GROUPS  # noqa: E402
from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import get_npz  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import build_mu  # noqa: E402
from experiments.pure_af_geometry.run_cifar_pc_transport_mode_attack import build_pc_directions  # noqa: E402


PC_COLS = [f"pc{i}_coeff" for i in range(1, 6)]


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / max(len(a) + len(b) - 2, 1))
    return float((np.mean(a) - np.mean(b)) / max(pooled, 1e-12))


def add_energy_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i in range(1, 6):
        c = f"pc{i}_coeff"
        df[f"pc{i}_abs_coeff"] = df[c].abs()
        df[f"pc{i}_energy"] = df[c] ** 2
    df["transport_energy_top5"] = df[[f"pc{i}_energy" for i in range(1, 6)]].sum(axis=1)
    return df


def load_pgd_square(projection_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(projection_csv)
    df = df[df["attack"].isin(["pgd", "square"])].copy()
    df["run_id"] = df["attack"] + "_idx" + df["dataset_idx"].astype(str)
    df["target_class"] = df["label"]
    keep = [
        "model", "attack", "run_id", "dataset_idx", "image_ord", "label", "target_class",
        "layer_group", "layer", "step", "normalized_progress", "time_bin", "final_success",
        "step_success", "pred", "margin", "true_prob", *PC_COLS,
    ]
    return add_energy_cols(df[keep])


def load_ga(layerwise_dir: Path, pc_dirs: dict, top_k: int) -> pd.DataFrame:
    seg = pd.read_csv(layerwise_dir / "segments.csv")
    points = pd.read_csv(layerwise_dir / "points.csv")
    runs = pd.read_csv(layerwise_dir / "runs.csv")
    vec_npz = np.load(layerwise_dir / "segment_vectors.npz")
    point_lookup = points.set_index(["model", "run_id", "layer", "generation"])
    run_success = runs.set_index(["model", "run_id"])["success"].to_dict()
    rows = []
    for group, mapping in LAYER_GROUPS.items():
        for model, layer in mapping.items():
            key = f"vectors__{model}__{layer}"
            if key not in vec_npz.files:
                continue
            axes = []
            for pc in range(1, top_k + 1):
                v = pc_dirs.get((model, layer, pc))
                if v is not None:
                    axes.append(v)
            if len(axes) < top_k:
                continue
            basis = np.stack(axes).astype(np.float32)
            arr = get_npz(vec_npz, "vectors", model, layer)
            sub = seg[(seg.model == model) & (seg.layer == layer)].sort_values(["run_id", "start_generation"])
            for run_id, g in sub.groupby("run_id", sort=False):
                cumulative = np.zeros(basis.shape[1], dtype=np.float32)
                n = len(g)
                final_success = int(run_success.get((model, run_id), int(g["success"].max())))
                for step_i, r in enumerate(g.itertuples(), start=1):
                    cumulative += arr[int(r.vector_idx)]
                    coeff = basis @ cumulative
                    meta = {}
                    try:
                        p = point_lookup.loc[(model, run_id, layer, int(r.end_generation))]
                        if isinstance(p, pd.DataFrame):
                            p = p.iloc[0]
                        meta = {
                            "pred": int(p["pred"]),
                            "margin": float(p["margin"]),
                            "true_prob": float(p["prob"]),
                            "step_success": int(int(p["pred"]) == int(r.target_class)),
                        }
                    except KeyError:
                        meta = {"pred": np.nan, "margin": np.nan, "true_prob": np.nan, "step_success": np.nan}
                    row = {
                        "model": model,
                        "attack": "ga",
                        "run_id": str(run_id),
                        "dataset_idx": np.nan,
                        "image_ord": np.nan,
                        "label": int(r.target_class),
                        "target_class": int(r.target_class),
                        "layer_group": group,
                        "layer": layer,
                        "step": int(r.end_generation),
                        "normalized_progress": float(step_i / max(n, 1)),
                        "time_bin": min(4, int(np.floor((step_i / max(n, 1)) * 5.0))) if step_i < n else 4,
                        "final_success": final_success,
                        **meta,
                    }
                    for i, c in enumerate(coeff, start=1):
                        row[f"pc{i}_coeff"] = float(c)
                    rows.append(row)
    return add_energy_cols(pd.DataFrame(rows))


def energy_breakdown(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    long = df.melt(
        id_vars=["model", "attack", "layer_group", "layer", "run_id", "final_success", "time_bin"],
        value_vars=[f"pc{i}_energy" for i in range(1, 6)],
        var_name="pc",
        value_name="energy",
    )
    long["pc"] = long["pc"].str.extract(r"pc(\d+)").astype(int)
    summary = long.groupby(["model", "attack", "layer_group", "layer", "final_success", "pc"], dropna=False).agg(
        n=("energy", "size"),
        mean_energy=("energy", "mean"),
        median_energy=("energy", "median"),
        total_energy=("energy", "sum"),
    ).reset_index()
    totals = summary.groupby(["model", "attack", "layer_group", "layer", "final_success"])["mean_energy"].transform("sum")
    summary["fraction_of_pc_energy"] = summary["mean_energy"] / np.clip(totals, 1e-12, None)
    summary.to_csv(out_dir / "attack_pc_energy_breakdown.csv", index=False)
    plot_energy_breakdown(summary, out_dir)
    return summary


def plot_energy_breakdown(summary: pd.DataFrame, out_dir: Path):
    for group in ["hidden", "penultimate", "logits"]:
        sub = summary[(summary.layer_group == group) & (summary.final_success == 1)]
        if sub.empty:
            continue
        models = list(dict.fromkeys(sub.model))
        attacks = ["pgd", "square", "ga"]
        fig, axes = plt.subplots(len(models), len(attacks), figsize=(13, 3.2 * len(models)), sharey=False, constrained_layout=True)
        if len(models) == 1:
            axes = np.expand_dims(axes, 0)
        for r, model in enumerate(models):
            for c, attack in enumerate(attacks):
                ax = axes[r, c]
                g = sub[(sub.model == model) & (sub.attack == attack)].sort_values("pc")
                ax.bar(g.pc.astype(str), g.fraction_of_pc_energy)
                ax.set_ylim(0, 1)
                ax.set_title(f"{model} {attack}")
                ax.set_xlabel("PC")
                ax.set_ylabel("energy share")
                ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Successful trajectory PC energy share: {group}")
        fig.savefig(out_dir / f"attack_pc_energy_breakdown_{group}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    # Requested generic filename: use hidden panel as a compact default if present.
    src = out_dir / "attack_pc_energy_breakdown_penultimate.png"
    if src.exists():
        import shutil
        shutil.copy2(src, out_dir / "attack_pc_energy_breakdown.png")


def temporal_curves(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    long = df.melt(
        id_vars=["model", "attack", "layer_group", "layer", "run_id", "final_success", "time_bin", "normalized_progress"],
        value_vars=PC_COLS,
        var_name="pc",
        value_name="coeff",
    )
    long["pc"] = long["pc"].str.extract(r"pc(\d+)").astype(int)
    summary = long.groupby(["model", "attack", "layer_group", "layer", "final_success", "time_bin", "pc"], dropna=False).agg(
        n=("coeff", "size"),
        mean_coeff=("coeff", "mean"),
        mean_abs_coeff=("coeff", lambda x: float(np.mean(np.abs(x)))),
        mean_energy=("coeff", lambda x: float(np.mean(np.asarray(x) ** 2))),
    ).reset_index()
    summary.to_csv(out_dir / "attack_pc_temporal_curves.csv", index=False)
    plot_temporal(summary, out_dir)
    return summary


def plot_temporal(summary: pd.DataFrame, out_dir: Path):
    for group in ["hidden", "penultimate", "logits"]:
        sub = summary[summary.layer_group == group]
        if sub.empty:
            continue
        models = list(dict.fromkeys(sub.model))
        attacks = ["pgd", "square", "ga"]
        fig, axes = plt.subplots(len(models), len(attacks), figsize=(14, 3.2 * len(models)), constrained_layout=True)
        if len(models) == 1:
            axes = np.expand_dims(axes, 0)
        for r, model in enumerate(models):
            for c, attack in enumerate(attacks):
                ax = axes[r, c]
                g = sub[(sub.model == model) & (sub.attack == attack) & (sub.final_success == 1)]
                for pc, pcg in g.groupby("pc"):
                    pcg = pcg.sort_values("time_bin")
                    ax.plot(pcg.time_bin, pcg.mean_abs_coeff, marker="o", label=f"PC{pc}")
                ax.set_title(f"{model} {attack}")
                ax.set_xlabel("time bin")
                ax.set_ylabel("mean |coeff|")
                ax.grid(alpha=0.25)
                if r == 0 and c == len(attacks) - 1:
                    ax.legend(fontsize=7)
        fig.suptitle(f"Temporal PC activation, successful trajectories: {group}")
        fig.savefig(out_dir / f"attack_pc_temporal_curves_{group}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    src = out_dir / "attack_pc_temporal_curves_penultimate.png"
    if src.exists():
        import shutil
        shutil.copy2(src, out_dir / "attack_pc_temporal_curves.png")


def run_signatures(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    final = df.sort_values(["normalized_progress", "step"]).groupby(["model", "attack", "layer_group", "layer", "run_id"], as_index=False).tail(1)
    rows = []
    for r in final.itertuples():
        vals = np.array([getattr(r, f"pc{i}_energy") for i in range(1, 6)], dtype=float)
        total = float(np.sum(vals))
        row = {
            "model": r.model,
            "attack": r.attack,
            "run_id": r.run_id,
            "layer_group": r.layer_group,
            "layer": r.layer,
            "final_success": int(r.final_success),
            "top5_energy": total,
        }
        for i, v in enumerate(vals, start=1):
            row[f"pc{i}_energy"] = float(v)
            row[f"pc{i}_energy_fraction"] = float(v / max(total, 1e-12))
        rows.append(row)
    sig = pd.DataFrame(rows)
    sig.to_csv(out_dir / "attack_run_signatures.csv", index=False)

    sim_rows = []
    for (model, group, layer), g in sig[sig.final_success == 1].groupby(["model", "layer_group", "layer"]):
        centroids = {}
        for attack, ag in g.groupby("attack"):
            v = ag[[f"pc{i}_energy_fraction" for i in range(1, 6)]].mean().to_numpy(dtype=float)
            centroids[attack] = v / max(np.linalg.norm(v), 1e-12)
        attacks = sorted(centroids)
        for a in attacks:
            for b in attacks:
                sim_rows.append({
                    "model": model,
                    "layer_group": group,
                    "layer": layer,
                    "attack_a": a,
                    "attack_b": b,
                    "cosine_similarity": float(np.dot(centroids[a], centroids[b])),
                })
    sim = pd.DataFrame(sim_rows)
    sim.to_csv(out_dir / "attack_signature_similarity.csv", index=False)
    plot_embedding(sig, out_dir)
    return sig


def plot_embedding(sig: pd.DataFrame, out_dir: Path):
    use = sig[(sig.final_success == 1) & (sig.layer_group.isin(["penultimate", "hidden"]))].copy()
    if len(use) < 5:
        return
    x = use[[f"pc{i}_energy_fraction" for i in range(1, 6)]].to_numpy(dtype=float)
    x = StandardScaler().fit_transform(x)
    emb = PCA(n_components=2, random_state=0).fit_transform(x)
    use["emb1"] = emb[:, 0]
    use["emb2"] = emb[:, 1]
    use.to_csv(out_dir / "attack_signature_embedding.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, group in zip(axes, ["hidden", "penultimate"]):
        sub = use[use.layer_group == group]
        for attack, g in sub.groupby("attack"):
            ax.scatter(g.emb1, g.emb2, s=18, alpha=0.75, label=attack)
        ax.set_title(group)
        ax.set_xlabel("PC signature PCA1")
        ax.set_ylabel("PC signature PCA2")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(out_dir / "attack_signature_embedding.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def success_mechanisms(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    final = df.sort_values(["normalized_progress", "step"]).groupby(["model", "attack", "layer_group", "layer", "run_id"], as_index=False).tail(1)
    rows = []
    for keys, g in final.groupby(["model", "attack", "layer_group", "layer"]):
        model, attack, group, layer = keys
        y = g["final_success"].to_numpy(dtype=int)
        for i in range(1, 6):
            score = g[f"pc{i}_energy"].to_numpy(dtype=float)
            if len(np.unique(y)) < 2:
                auroc = np.nan
            else:
                auroc = float(roc_auc_score(y, score))
            succ = score[y == 1]
            fail = score[y == 0]
            rows.append({
                "model": model,
                "attack": attack,
                "layer_group": group,
                "layer": layer,
                "pc": i,
                "n": int(len(g)),
                "success_rate": float(np.mean(y)) if len(y) else np.nan,
                "success_mean_energy": float(np.mean(succ)) if len(succ) else np.nan,
                "failed_mean_energy": float(np.mean(fail)) if len(fail) else np.nan,
                "cohen_d_success_minus_failed": cohen_d(succ, fail),
                "auroc_individual_pc_energy": auroc,
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "attack_pc_success_mechanisms.csv", index=False)
    return out


def shared_vs_specific(energy: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    use = energy[energy.final_success == 1].copy()
    for (model, group, layer, attack), g in use.groupby(["model", "layer_group", "layer", "attack"]):
        g = g.sort_values("mean_energy", ascending=False)
        for rank, r in enumerate(g.itertuples(), start=1):
            rows.append({
                "model": model,
                "layer_group": group,
                "layer": layer,
                "attack": attack,
                "pc": int(r.pc),
                "rank": int(rank),
                "mean_energy": float(r.mean_energy),
                "fraction_of_pc_energy": float(r.fraction_of_pc_energy),
            })
    ranks = pd.DataFrame(rows)
    overlap_rows = []
    for (model, group, layer), g in ranks.groupby(["model", "layer_group", "layer"]):
        attacks = sorted(g.attack.unique())
        for k in [1, 2, 3]:
            top = {a: set(g[(g.attack == a) & (g["rank"] <= k)]["pc"].astype(int)) for a in attacks}
            for i, a in enumerate(attacks):
                for b in attacks[i + 1 :]:
                    inter = len(top[a] & top[b])
                    union = len(top[a] | top[b])
                    overlap_rows.append({
                        "model": model,
                        "layer_group": group,
                        "layer": layer,
                        "attack_a": a,
                        "attack_b": b,
                        "top_k": k,
                        "intersection": inter,
                        "jaccard": inter / max(union, 1),
                    })
    out = pd.DataFrame(overlap_rows)
    ranks.to_csv(out_dir / "attack_pc_rankings.csv", index=False)
    out.to_csv(out_dir / "shared_vs_specific_transport_modes.csv", index=False)
    return out


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mu = build_mu(Path(args.layerwise_dir))
    pc_dirs, pc_meta = build_pc_directions(mu, args.top_k)
    pc_meta.to_csv(out_dir / "transport_pc_metadata.csv", index=False)
    print("[LOAD] PGD/Square projection rows", flush=True)
    pgd_square = load_pgd_square(Path(args.axis_projection_csv))
    print("[LOAD] GA cumulative projections", flush=True)
    ga = load_ga(Path(args.layerwise_dir), pc_dirs, args.top_k)
    df = pd.concat([pgd_square, ga], ignore_index=True, sort=False)
    df.to_csv(out_dir / "attack_transport_projection_timeseries.csv", index=False)
    energy = energy_breakdown(df, out_dir)
    temporal_curves(df, out_dir)
    run_signatures(df, out_dir)
    success_mechanisms(df, out_dir)
    shared_vs_specific(energy, out_dir)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"args": vars(args), "n_rows": int(len(df)), "attacks": sorted(df.attack.unique().tolist())}, f, indent=2)
    print(f"[SAVED] {out_dir}", flush=True)
    print(energy[(energy.final_success == 1) & (energy.layer_group == "penultimate")].sort_values(
        ["model", "attack", "fraction_of_pc_energy"], ascending=[True, True, False]
    ).groupby(["model", "attack"]).head(2).to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--axis-projection-csv", default="analysis_outputs/pure_af_geometry/cifar_attack_axis_projection/attack_axis_projection_timeseries.csv")
    p.add_argument("--layerwise-dir", default="analysis_outputs/pure_af_geometry/cifar_layerwise_success_flow_c10_s3_g120")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_attack_transport_decomposition")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
