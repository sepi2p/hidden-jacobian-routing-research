#!/usr/bin/env python3
"""Step 4 sign- and time-sensitive optimizer comparison.

Earlier cross-optimizer results used unsigned projection energy, which can make
different paths look similar if they activate the same PCs with different signs
or temporal order.  This script compares signed coefficient trajectories,
cumulative paths, curvature, DTW/Frechet-style distances, and mobility/random
controls on matched images.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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


def stable_offset(*parts: str) -> int:
    text = "::".join(parts)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 100000


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    kk = min(len(a), len(b))
    a = a[:kk]
    b = b[:kk]
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-12:
        return np.nan
    return float(np.dot(a, b) / den)


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


def residualize(x: np.ndarray, basis: np.ndarray | None, k: int) -> np.ndarray:
    if basis is None or k < 1:
        return x.astype(np.float32, copy=True)
    kk = min(k, basis.shape[0], x.shape[1])
    if kk < 1:
        return x.astype(np.float32, copy=True)
    b = basis[:kk]
    return (x - (x @ b.T) @ b).astype(np.float32)


def resample_sequence(seq: np.ndarray, length: int) -> np.ndarray:
    seq = np.asarray(seq, dtype=float)
    if len(seq) == 0:
        return np.zeros((length, 0), dtype=float)
    if len(seq) == length:
        return seq.copy()
    if len(seq) == 1:
        return np.repeat(seq, length, axis=0)
    old = np.linspace(0.0, 1.0, len(seq))
    new = np.linspace(0.0, 1.0, length)
    out = np.zeros((length, seq.shape[1]), dtype=float)
    for j in range(seq.shape[1]):
        out[:, j] = np.interp(new, old, seq[:, j])
    return out


def normalize_path(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if len(path) == 0:
        return path
    step_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.array([])
    scale = float(step_lengths.sum())
    if scale < 1e-12:
        scale = float(np.linalg.norm(path[-1] - path[0])) if len(path) > 1 else float(np.linalg.norm(path[0]))
    if scale < 1e-12:
        return path.copy()
    return path / scale


def cumulative_path(local_coeff: np.ndarray) -> np.ndarray:
    local_coeff = np.asarray(local_coeff, dtype=float)
    if len(local_coeff) == 0:
        return np.zeros((1, local_coeff.shape[1] if local_coeff.ndim == 2 else 0), dtype=float)
    return np.vstack([np.zeros((1, local_coeff.shape[1])), np.cumsum(local_coeff, axis=0)])


def direction_sequence(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq, dtype=float)
    if len(seq) == 0:
        return seq
    n = np.linalg.norm(seq, axis=1, keepdims=True)
    return seq / np.clip(n, 1e-12, None)


def temporal_direction_cosine(a: np.ndarray, b: np.ndarray, length: int) -> float:
    aa = direction_sequence(resample_sequence(a, length))
    bb = direction_sequence(resample_sequence(b, length))
    if aa.shape[1] == 0 or bb.shape[1] == 0:
        return np.nan
    vals = []
    for i in range(length):
        vals.append(cosine(aa[i], bb[i]))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    return float(vals.mean())


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    dp = np.full((len(a) + 1, len(b) + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[-1, -1] / (len(a) + len(b)))


def discrete_frechet(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    ca = np.full((len(a), len(b)), -1.0, dtype=float)

    def c(i: int, j: int) -> float:
        if ca[i, j] > -0.5:
            return ca[i, j]
        d = np.linalg.norm(a[i] - b[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), d)
        elif i > 0 and j > 0:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), d)
        else:
            ca[i, j] = np.inf
        return ca[i, j]

    return float(c(len(a) - 1, len(b) - 1))


def path_curvature(local_coeff: np.ndarray) -> tuple[float, float, float]:
    local_coeff = np.asarray(local_coeff, dtype=float)
    if len(local_coeff) < 2:
        return np.nan, np.nan, np.nan
    vals = []
    for i in range(len(local_coeff) - 1):
        vals.append(cosine(local_coeff[i], local_coeff[i + 1]))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    path = cumulative_path(local_coeff)
    path_len = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    straight = float(np.linalg.norm(path[-1] - path[0]))
    ratio = path_len / max(straight, 1e-12)
    if len(vals) == 0:
        return np.nan, ratio, path_len
    return float(vals.mean()), ratio, path_len


def energy_profile(coeff: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coeff, dtype=float)
    if len(coeff) == 0:
        return np.zeros(0, dtype=float)
    e = np.mean(coeff * coeff, axis=0)
    return e / np.clip(e.sum(), 1e-12, None)


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


@dataclass
class Trajectory:
    source: str
    image_ord: int
    final_success: int
    final_pred: int
    label: int
    coeff: np.ndarray


def fit_null_basis(store: Store, model: str, null_source: str, layer: str, k: int, seed: int) -> np.ndarray | None:
    if null_source == "none" or k < 1:
        return None
    rows, x = store.rows_for(model, null_source, layer)
    if rows.empty:
        return None
    mask = rows["split"].to_numpy() == "train"
    if mask.sum() < max(8, min(k, x.shape[1]) + 2):
        return None
    _mean, basis = pca_basis(x[mask], max(k, 20), seed)
    return basis


def fit_common_basis(
    store: Store,
    model: str,
    layer: str,
    basis_sources: list[str],
    null_basis: np.ndarray | None,
    residual_k: int,
    basis_k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    mats = []
    total = 0
    for source in basis_sources:
        rows, x = store.rows_for(model, source, layer)
        if rows.empty:
            continue
        xr = residualize(x, null_basis, residual_k)
        mask = (rows["split"].to_numpy() == "train") & (rows["final_success"].to_numpy(dtype=int) == 1)
        if mask.any():
            mats.append(xr[mask])
            total += int(mask.sum())
    if not mats:
        raise RuntimeError("No train-success vectors available for common basis.")
    mat = np.concatenate(mats, axis=0)
    mean, basis = pca_basis(mat, basis_k, seed)
    return mean, basis, total


def build_trajectories(
    store: Store,
    model: str,
    source: str,
    layer: str,
    null_basis: np.ndarray | None,
    residual_k: int,
    mean: np.ndarray,
    basis: np.ndarray,
    split: str,
) -> dict[int, Trajectory]:
    rows, x = store.rows_for(model, source, layer)
    if rows.empty:
        return {}
    xr = residualize(x, null_basis, residual_k)
    coeff = (xr - mean) @ basis.T
    out = {}
    for image_ord, g in rows[rows["split"] == split].groupby("image_ord", sort=True):
        idx = g.sort_values("step").index.to_numpy()
        if len(idx) == 0:
            continue
        gg = rows.loc[idx]
        out[int(image_ord)] = Trajectory(
            source=source,
            image_ord=int(image_ord),
            final_success=int(gg["final_success"].iloc[0]),
            final_pred=int(gg["final_pred"].iloc[0]),
            label=int(gg["label"].iloc[0]),
            coeff=coeff[idx].astype(np.float64),
        )
    return out


def pair_mode_images(a: dict[int, Trajectory], b: dict[int, Trajectory], mode: str) -> list[int]:
    common = sorted(set(a) & set(b))
    if mode == "all":
        return common
    if mode == "both_success":
        return [i for i in common if a[i].final_success == 1 and b[i].final_success == 1]
    if mode == "both_failed":
        return [i for i in common if a[i].final_success == 0 and b[i].final_success == 0]
    if mode == "source_a_success":
        return [i for i in common if a[i].final_success == 1]
    if mode == "source_a_failed":
        return [i for i in common if a[i].final_success == 0]
    raise ValueError(f"unknown pair mode {mode}")


def compare_pair(ta: Trajectory, tb: Trajectory, resample_len: int) -> dict:
    ca = ta.coeff
    cb = tb.coeff
    ca_r = resample_sequence(ca, resample_len)
    cb_r = resample_sequence(cb, resample_len)
    pa = cumulative_path(ca)
    pb = cumulative_path(cb)
    pa_n = normalize_path(pa)
    pb_n = normalize_path(pb)
    pa_r = resample_sequence(pa_n, resample_len)
    pb_r = resample_sequence(pb_n, resample_len)
    curv_a, ratio_a, plen_a = path_curvature(ca)
    curv_b, ratio_b, plen_b = path_curvature(cb)
    return {
        "local_signed_flat_cosine": cosine(ca_r, cb_r),
        "local_temporal_direction_cosine": temporal_direction_cosine(ca, cb, resample_len),
        "net_displacement_cosine": cosine(np.sum(ca, axis=0), np.sum(cb, axis=0)),
        "cumulative_path_flat_cosine": cosine(pa_r, pb_r),
        "energy_profile_cosine": cosine(energy_profile(ca), energy_profile(cb)),
        "dtw_distance": dtw_distance(pa_n, pb_n),
        "frechet_distance": discrete_frechet(pa_n, pb_n),
        "curvature_cosine_a": curv_a,
        "curvature_cosine_b": curv_b,
        "curvature_ratio_a": ratio_a,
        "curvature_ratio_b": ratio_b,
        "path_length_a": plen_a,
        "path_length_b": plen_b,
        "steps_a": int(len(ca)),
        "steps_b": int(len(cb)),
    }


def summarize_pairs(pairs: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    metrics = [
        "local_signed_flat_cosine",
        "local_temporal_direction_cosine",
        "net_displacement_cosine",
        "cumulative_path_flat_cosine",
        "energy_profile_cosine",
        "dtw_distance",
        "frechet_distance",
    ]
    rows = []
    group_cols = ["model", "layer", "residual_null_source", "residual_k", "pair_type", "mode"]
    for key, g in pairs.groupby(group_cols):
        base = dict(zip(group_cols, key))
        base["n_pairs"] = int(len(g))
        for metric in metrics:
            vals = g[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                base[f"{metric}_mean"] = np.nan
                base[f"{metric}_median"] = np.nan
                base[f"{metric}_ci_lo"] = np.nan
                base[f"{metric}_ci_hi"] = np.nan
                continue
            boots = []
            for _ in range(reps):
                sample = rng.choice(vals, len(vals), replace=True)
                boots.append(float(np.mean(sample)))
            boots = np.asarray(boots)
            base[f"{metric}_mean"] = float(np.mean(vals))
            base[f"{metric}_median"] = float(np.median(vals))
            base[f"{metric}_ci_lo"] = float(np.quantile(boots, 0.025))
            base[f"{metric}_ci_hi"] = float(np.quantile(boots, 0.975))
        rows.append(base)
    return pd.DataFrame(rows)


def temporal_curves(trajectories: dict[str, dict[int, Trajectory]], sources: list[str], length: int) -> pd.DataFrame:
    rows = []
    for source in sources:
        for mode in ["all", "success", "failed"]:
            seqs = []
            for tr in trajectories.get(source, {}).values():
                if mode == "success" and tr.final_success != 1:
                    continue
                if mode == "failed" and tr.final_success != 0:
                    continue
                seqs.append(resample_sequence(tr.coeff, length))
            if not seqs:
                continue
            arr = np.stack(seqs)
            mean = arr.mean(axis=0)
            sem = arr.std(axis=0, ddof=1) / math.sqrt(len(seqs)) if len(seqs) > 1 else np.zeros_like(mean)
            for t in range(length):
                row = {"source": source, "mode": mode, "time_bin": t, "n_trajectories": len(seqs)}
                for pc in range(mean.shape[1]):
                    row[f"pc{pc+1}_mean"] = float(mean[t, pc])
                    row[f"pc{pc+1}_sem"] = float(sem[t, pc])
                rows.append(row)
    return pd.DataFrame(rows)


def write_summary(out_dir: Path, summary: pd.DataFrame):
    lines = [
        "# Step 4 Sign-Time Optimizer Similarity Summary",
        "",
        "This analysis tests whether PGD and Square similarity survives signed, time-ordered comparisons rather than only unsigned projection-energy profiles.",
        "",
    ]
    if summary.empty:
        lines.append("No summary metrics were produced.")
        out_dir.joinpath("sign_time_optimizer_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    key_metrics = [
        "local_signed_flat_cosine",
        "local_temporal_direction_cosine",
        "net_displacement_cosine",
        "cumulative_path_flat_cosine",
        "energy_profile_cosine",
        "dtw_distance",
        "frechet_distance",
    ]
    for residual in summary["residual_null_source"].drop_duplicates():
        subr = summary[(summary.residual_null_source == residual) & (summary.layer == "layer4") & (summary["mode"] == "all")]
        if subr.empty:
            continue
        lines += [f"## Layer4, all runs, remove {residual}", ""]
        for row in subr.sort_values(["residual_k", "pair_type"]).itertuples():
            lines.append(f"- {row.pair_type}, k={row.residual_k}, n={row.n_pairs}")
            for metric in key_metrics:
                val = getattr(row, f"{metric}_mean")
                lo = getattr(row, f"{metric}_ci_lo")
                hi = getattr(row, f"{metric}_ci_hi")
                lines.append(f"  - {metric}: {val:.3f} [{lo:.3f}, {hi:.3f}]")
            lines.append("")
    lines += [
        "## Gate",
        "",
        "Keep a shared-flow claim only if PGD/Square signed and time-ordered similarity is clearly stronger than random/mobility controls and survives all-run analysis. If the main surviving agreement is unsigned energy-profile similarity, use the weaker phrase shared high-gain mode usage.",
    ]
    out_dir.joinpath("sign_time_optimizer_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/sign_time_optimizer_bbb_resnet50_c200_auto"))
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layers", default="layer4,logits")
    p.add_argument("--basis-sources", default="pgd,square")
    p.add_argument("--sources", default="pgd,square,mobility_top_walk_square_budget,random_sign_walk_square_budget,correlated_random_walk_square_budget")
    p.add_argument("--residual-null-sources", default="none,jacobian_probe_all,mobility_top_walk_square_budget")
    p.add_argument("--residual-ks", default="0,20")
    p.add_argument("--basis-k", type=int, default=5)
    p.add_argument("--resample-len", type=int, default=20)
    p.add_argument("--bootstrap-reps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = Store(args.input_dir)
    layers = parse_csv(args.layers)
    basis_sources = parse_csv(args.basis_sources)
    sources = parse_csv(args.sources)
    null_sources = parse_csv(args.residual_null_sources)
    residual_ks = parse_int_csv(args.residual_ks)

    pair_specs = [
        ("pgd", "square", "pgd_vs_square_same_image", False),
        ("pgd", "square", "pgd_vs_square_permuted", True),
        ("pgd", "mobility_top_walk_square_budget", "pgd_vs_mobility_same_image", False),
        ("square", "mobility_top_walk_square_budget", "square_vs_mobility_same_image", False),
        ("pgd", "random_sign_walk_square_budget", "pgd_vs_random_same_image", False),
        ("square", "random_sign_walk_square_budget", "square_vs_random_same_image", False),
        ("pgd", "correlated_random_walk_square_budget", "pgd_vs_correlated_random_same_image", False),
        ("square", "correlated_random_walk_square_budget", "square_vs_correlated_random_same_image", False),
    ]

    pair_rows = []
    curve_frames = []
    basis_rows = []
    rng = np.random.default_rng(args.seed)
    for layer in layers:
        for null_source in null_sources:
            for residual_k in residual_ks:
                if null_source == "none" and residual_k != 0:
                    continue
                if null_source != "none" and residual_k == 0:
                    continue
                null_basis = fit_null_basis(store, args.model, null_source, layer, residual_k, args.seed)
                mean, basis, basis_n = fit_common_basis(
                    store, args.model, layer, basis_sources, null_basis, residual_k, args.basis_k, args.seed
                )
                basis_rows.append(
                    {
                        "model": args.model,
                        "layer": layer,
                        "residual_null_source": null_source,
                        "residual_k": residual_k,
                        "basis_k": args.basis_k,
                        "basis_train_success_segments": basis_n,
                    }
                )
                traj = {
                    source: build_trajectories(
                        store, args.model, source, layer, null_basis, residual_k, mean, basis, split="test"
                    )
                    for source in sources
                }
                curve = temporal_curves(traj, sources, args.resample_len)
                if not curve.empty:
                    curve["model"] = args.model
                    curve["layer"] = layer
                    curve["residual_null_source"] = null_source
                    curve["residual_k"] = residual_k
                    curve_frames.append(curve)

                for source_a, source_b, pair_type, permute in pair_specs:
                    if source_a not in traj or source_b not in traj:
                        continue
                    modes = ["all"]
                    if source_a in {"pgd", "square"} and source_b in {"pgd", "square"}:
                        modes += ["both_success", "both_failed"]
                    else:
                        modes += ["source_a_success", "source_a_failed"]
                    for mode in modes:
                        imgs = pair_mode_images(traj[source_a], traj[source_b], mode)
                        if len(imgs) < 2:
                            continue
                        b_imgs = imgs.copy()
                        if permute:
                            b_imgs = list(rng.permutation(b_imgs))
                        for ia, ib in zip(imgs, b_imgs):
                            ta = traj[source_a][ia]
                            tb = traj[source_b][ib]
                            row = {
                                "model": args.model,
                                "layer": layer,
                                "residual_null_source": null_source,
                                "residual_k": residual_k,
                                "basis_k": args.basis_k,
                                "pair_type": pair_type,
                                "mode": mode,
                                "image_ord_a": int(ia),
                                "image_ord_b": int(ib),
                                "source_a": source_a,
                                "source_b": source_b,
                                "success_a": int(ta.final_success),
                                "success_b": int(tb.final_success),
                                "label_a": int(ta.label),
                                "label_b": int(tb.label),
                                "final_pred_a": int(ta.final_pred),
                                "final_pred_b": int(tb.final_pred),
                            }
                            row.update(compare_pair(ta, tb, args.resample_len))
                            pair_rows.append(row)

    pairs = pd.DataFrame(pair_rows)
    summary = summarize_pairs(pairs, args.bootstrap_reps, args.seed)
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    basis_df = pd.DataFrame(basis_rows)
    pairs.to_csv(args.output_dir / "signed_time_pair_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "signed_time_pair_summary.csv", index=False)
    curves.to_csv(args.output_dir / "signed_temporal_activation_curves.csv", index=False)
    basis_df.to_csv(args.output_dir / "signed_time_basis_metadata.csv", index=False)
    metadata = {
        "script": "experiments/pure_af_geometry/analyze_sign_time_optimizer_similarity.py",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "layers": layers,
        "basis_sources": basis_sources,
        "sources": sources,
        "residual_null_sources": null_sources,
        "residual_ks": residual_ks,
        "basis_k": args.basis_k,
        "resample_len": args.resample_len,
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(args.output_dir, summary)
    print(f"[DONE] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
