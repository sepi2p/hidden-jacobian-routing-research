#!/usr/bin/env python3
"""Step 3 pseudoreplication checks for balanced transport trajectories.

The reviewer null is that many correlated local steps from the same image make
the success-flow effect look more significant than it is.  This script
recomputes dimensionality and success-vs-failed projection metrics under
sampling rules that reduce each trajectory to one or a few image-level units.
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


def parse_int_csv(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def vector_key(model: str, source: str, layer: str) -> str:
    return f"{model}__{source}__{layer}"


def stable_offset(*parts: str) -> int:
    text = "::".join(parts)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 100000


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


def cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    sp = math.sqrt(((len(pos) - 1) * np.var(pos, ddof=1) + (len(neg) - 1) * np.var(neg, ddof=1)) / (len(pos) + len(neg) - 2))
    if sp < 1e-12:
        return np.nan
    return float((np.mean(pos) - np.mean(neg)) / sp)


def pca_basis(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x) < 2:
        raise ValueError("Need at least two rows for PCA.")
    kk = min(k, x.shape[0] - 1, x.shape[1])
    if kk < 1:
        raise ValueError("PCA rank is zero.")
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    xc = x - mean
    _u, s, vt = randomized_svd(xc, n_components=kk, random_state=seed)
    eig = (s * s) / max(len(x) - 1, 1)
    return mean, vt.astype(np.float32), eig.astype(np.float64)


def full_spectrum(x: np.ndarray, seed: int, max_rank: int | None = None) -> np.ndarray:
    if len(x) < 2:
        return np.zeros(0, dtype=np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    rank = min(xc.shape[0] - 1, xc.shape[1])
    if max_rank is not None:
        rank = min(rank, max_rank)
    if rank < 1:
        return np.zeros(0, dtype=np.float64)
    _u, s, _vt = randomized_svd(xc, n_components=rank, random_state=seed)
    eig = (s * s) / max(len(x) - 1, 1)
    return eig.astype(np.float64)


def dimensionality(eig: np.ndarray) -> dict:
    eig = np.asarray(eig, dtype=float)
    eig = eig[np.isfinite(eig) & (eig > 0)]
    if len(eig) == 0 or eig.sum() <= 0:
        return {
            "pc1_var": np.nan,
            "pc10_cumvar": np.nan,
            "dim80": np.nan,
            "dim90": np.nan,
            "effective_rank": np.nan,
            "rank_observed": 0,
        }
    p = eig / eig.sum()
    cum = np.cumsum(p)
    ent = -np.sum(p * np.log(np.clip(p, 1e-12, None)))
    return {
        "pc1_var": float(p[0]),
        "pc10_cumvar": float(cum[min(9, len(cum) - 1)]),
        "dim80": int(np.searchsorted(cum, 0.80) + 1),
        "dim90": int(np.searchsorted(cum, 0.90) + 1),
        "effective_rank": float(np.exp(ent)),
        "rank_observed": int(len(eig)),
    }


def residualize(x: np.ndarray, basis: np.ndarray | None, k: int) -> np.ndarray:
    if basis is None or k < 1:
        return x.astype(np.float32, copy=True)
    kk = min(k, basis.shape[0], x.shape[1])
    if kk < 1:
        return x.astype(np.float32, copy=True)
    b = basis[:kk]
    return (x - (x @ b.T) @ b).astype(np.float32)


def projection_energy(x: np.ndarray, mean: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, basis.shape[0], x.shape[1])
    if kk < 1 or len(x) == 0:
        return np.zeros(len(x), dtype=np.float64)
    xc = x - mean
    coeff = xc @ basis[:kk].T
    return np.sum(coeff * coeff, axis=1) / np.clip(np.sum(xc * xc, axis=1), 1e-12, None)


class ArtifactStore:
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
        x = self.arrays[key][sub["vector_idx"].to_numpy(dtype=int)]
        sub["split"] = sub["image_ord"].map(self.split_by_image).fillna("")
        return sub.reset_index(drop=True), x


def select_rows_for_rule(rows: pd.DataFrame, x: np.ndarray, rule: str, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if rows.empty:
        return rows.copy(), x
    rng = np.random.default_rng(seed)

    if rule == "all_steps" or rule == "all_steps_image_equal":
        return rows.copy(), x.astype(np.float32, copy=True)

    chosen = []
    vecs = []
    for image_ord, g in rows.groupby("image_ord", sort=True):
        idx = g.index.to_numpy()
        if rule == "random_step_per_image":
            pick = int(rng.choice(idx))
        elif rule == "first_success_or_final":
            if int(g["final_success"].iloc[0]) == 1 and (g["step_success_after"].astype(int) == 1).any():
                candidates = g[g["step_success_after"].astype(int) == 1].sort_values("step")
                pick = int(candidates.index[0])
            else:
                pick = int(g.sort_values("step").index[-1])
        elif rule == "trajectory_mean":
            base = g.sort_values("step").iloc[-1].copy()
            base["step"] = -1
            chosen.append(base)
            vecs.append(x[idx].mean(axis=0))
            continue
        else:
            raise ValueError(f"unknown sampling rule {rule}")
        chosen.append(rows.loc[pick].copy())
        vecs.append(x[pick])

    out_rows = pd.DataFrame(chosen).reset_index(drop=True)
    out_x = np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, x.shape[1]), dtype=np.float32)
    return out_rows, out_x


def fit_null_basis(store: ArtifactStore, model: str, null_source: str, layer: str, k: int, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    if null_source == "none" or k < 1:
        return None
    rows, x = store.rows_for(model, null_source, layer)
    if rows.empty:
        return None
    mask = rows["split"].to_numpy() == "train"
    if mask.sum() < max(8, min(k, x.shape[1]) + 2):
        return None
    mean, basis, _eig = pca_basis(x[mask], max(k, 20), seed)
    return mean, basis


def bootstrap_auc_by_image(scored: pd.DataFrame, seed: int, reps: int) -> tuple[float, float, float]:
    pos = scored[scored.final_success.astype(int) == 1]
    neg = scored[scored.final_success.astype(int) == 0]
    pos_images = pos["image_ord"].drop_duplicates().to_numpy()
    neg_images = neg["image_ord"].drop_duplicates().to_numpy()
    if len(pos_images) < 2 or len(neg_images) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pi = rng.choice(pos_images, len(pos_images), replace=True)
        ni = rng.choice(neg_images, len(neg_images), replace=True)
        ps = pos[pos.image_ord.isin(pi)].groupby("image_ord")["projection_energy"].mean().to_numpy()
        ns = neg[neg.image_ord.isin(ni)].groupby("image_ord")["projection_energy"].mean().to_numpy()
        vals.append(safe_auc(ps, ns))
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    return float(vals.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def analyze_source(
    store: ArtifactStore,
    model: str,
    source: str,
    layer: str,
    sampling_rule: str,
    null_source: str,
    residual_k: int,
    basis_k: int,
    spectrum_rank: int,
    bootstrap_reps: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], pd.DataFrame]:
    raw_rows, raw_x = store.rows_for(model, source, layer)
    if raw_rows.empty:
        return [], [], [], pd.DataFrame()

    null_fit = fit_null_basis(store, model, null_source, layer, residual_k, seed)
    null_basis = null_fit[1] if null_fit is not None else None
    xr = residualize(raw_x, null_basis, residual_k)
    rows, x = select_rows_for_rule(raw_rows, xr, sampling_rule, seed + stable_offset(source, layer, sampling_rule))

    train = rows["split"].to_numpy() == "train"
    test = rows["split"].to_numpy() == "test"
    train_success = train & (rows["final_success"].to_numpy(dtype=int) == 1)
    test_success = test & (rows["final_success"].to_numpy(dtype=int) == 1)
    test_failed = test & (rows["final_success"].to_numpy(dtype=int) == 0)
    if train_success.sum() < max(8, min(basis_k, x.shape[1]) + 2) or test_success.sum() < 2 or test_failed.sum() < 2:
        return [], [], [], pd.DataFrame()

    mean, basis, _basis_eig = pca_basis(x[train_success], basis_k, seed)
    eig = full_spectrum(x[train_success], seed, max_rank=spectrum_rank)
    dim_row = {
        "model": model,
        "source": source,
        "layer": layer,
        "sampling_rule": sampling_rule,
        "residual_null_source": null_source,
        "residual_k": residual_k,
        "basis_k": basis_k,
        "train_success_vectors": int(train_success.sum()),
        "train_success_images": int(rows.loc[train_success, "image_ord"].nunique()),
        **dimensionality(eig),
    }

    scores = projection_energy(x[test], mean, basis, basis_k)
    scored = rows[test].copy()
    scored["projection_energy"] = scores
    scored["sampling_rule"] = sampling_rule
    scored["residual_null_source"] = null_source
    scored["residual_k"] = residual_k

    if sampling_rule == "all_steps_image_equal":
        image_scores = (
            scored.groupby(["image_ord", "final_success"], as_index=False)["projection_energy"]
            .mean()
            .assign(model=model, source=source, layer=layer)
        )
        pos_scores = image_scores[image_scores.final_success.astype(int) == 1]["projection_energy"].to_numpy()
        neg_scores = image_scores[image_scores.final_success.astype(int) == 0]["projection_energy"].to_numpy()
        unit = "image_mean_score"
    else:
        pos_scores = scored[scored.final_success.astype(int) == 1]["projection_energy"].to_numpy()
        neg_scores = scored[scored.final_success.astype(int) == 0]["projection_energy"].to_numpy()
        unit = "selected_vectors"

    metric_row = {
        "model": model,
        "source": source,
        "layer": layer,
        "sampling_rule": sampling_rule,
        "evaluation_unit": unit,
        "residual_null_source": null_source,
        "residual_k": residual_k,
        "basis_k": basis_k,
        "comparison": "success_vs_failed_same_optimizer",
        "auroc": safe_auc(pos_scores, neg_scores),
        "cohens_d": cohens_d(pos_scores, neg_scores),
        "pos_mean_energy": float(np.mean(pos_scores)) if len(pos_scores) else np.nan,
        "neg_mean_energy": float(np.mean(neg_scores)) if len(neg_scores) else np.nan,
        "pos_vectors": int(len(pos_scores)),
        "neg_vectors": int(len(neg_scores)),
        "pos_images": int(scored[scored.final_success.astype(int) == 1]["image_ord"].nunique()),
        "neg_images": int(scored[scored.final_success.astype(int) == 0]["image_ord"].nunique()),
    }
    bmean, blo, bhi = bootstrap_auc_by_image(scored, seed, bootstrap_reps)
    ci_row = {
        "model": model,
        "source": source,
        "layer": layer,
        "sampling_rule": sampling_rule,
        "residual_null_source": null_source,
        "residual_k": residual_k,
        "basis_k": basis_k,
        "comparison": "success_vs_failed_same_optimizer",
        "image_bootstrap_auroc_mean": bmean,
        "image_bootstrap_auroc_lo": blo,
        "image_bootstrap_auroc_hi": bhi,
        "bootstrap_reps": bootstrap_reps,
    }
    return [dim_row], [metric_row], [ci_row], scored


def write_summary(out_dir: Path, metrics: pd.DataFrame, dims: pd.DataFrame, ci: pd.DataFrame):
    lines = [
        "# Step 3 Pseudoreplication Summary",
        "",
        "This analysis tests whether success-vs-failed structure survives when correlated steps from the same image are not treated as independent evidence.",
        "",
    ]
    if metrics.empty:
        lines.append("No metrics were produced.")
        out_dir.joinpath("pseudoreplication_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines += ["## Success-vs-Failed AUROC", ""]
    key = metrics[(metrics.layer == "layer4") & (metrics.comparison == "success_vs_failed_same_optimizer")]
    for row in key.sort_values(["source", "residual_null_source", "residual_k", "sampling_rule"]).itertuples():
        lines.append(
            f"- {row.source}, {row.sampling_rule}, remove {row.residual_null_source} k={row.residual_k}: "
            f"AUROC={row.auroc:.3f}, d={row.cohens_d:.2f}, pos={row.pos_vectors}, neg={row.neg_vectors}"
        )

    lines += ["", "## Image-Level Bootstrap Snapshot", ""]
    for row in ci[(ci.layer == "layer4")].sort_values(["source", "residual_null_source", "residual_k", "sampling_rule"]).itertuples():
        lines.append(
            f"- {row.source}, {row.sampling_rule}, remove {row.residual_null_source} k={row.residual_k}: "
            f"image AUROC={row.image_bootstrap_auroc_mean:.3f} "
            f"[{row.image_bootstrap_auroc_lo:.3f}, {row.image_bootstrap_auroc_hi:.3f}]"
        )

    if not dims.empty:
        lines += ["", "## Dimensionality Snapshot", ""]
        for row in dims[(dims.layer == "layer4")].sort_values(["source", "residual_null_source", "residual_k", "sampling_rule"]).itertuples():
            lines.append(
                f"- {row.source}, {row.sampling_rule}, remove {row.residual_null_source} k={row.residual_k}: "
                f"dim80={row.dim80}, dim90={row.dim90}, eff_rank={row.effective_rank:.2f}, n={row.train_success_vectors}"
            )

    lines += [
        "",
        "## Gate",
        "",
        "If the effect survives one-vector-per-image, first-success/final-step, and trajectory-mean analyses with image-level CIs, it is not only a pooled-step artifact. If it appears only for all-step pooling, move the result to the appendix or weaken the claim substantially.",
    ]
    out_dir.joinpath("pseudoreplication_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/pure_af_geometry/jacobian_null_response/pseudoreplication_bbb_resnet50_c200_auto"))
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--sources", default="pgd,square")
    p.add_argument("--layers", default="layer4,logits")
    p.add_argument("--sampling-rules", default="all_steps,all_steps_image_equal,random_step_per_image,first_success_or_final,trajectory_mean")
    p.add_argument("--residual-null-sources", default="none,jacobian_probe_all,mobility_top_walk_square_budget")
    p.add_argument("--residual-ks", default="0,20")
    p.add_argument("--basis-k", type=int, default=20)
    p.add_argument("--spectrum-rank", type=int, default=120)
    p.add_argument("--bootstrap-reps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(args.input_dir)
    sources = parse_csv(args.sources)
    layers = parse_csv(args.layers)
    rules = parse_csv(args.sampling_rules)
    null_sources = parse_csv(args.residual_null_sources)
    residual_ks = parse_int_csv(args.residual_ks)

    dim_rows = []
    metric_rows = []
    ci_rows = []
    score_frames = []
    for layer in layers:
        for source in sources:
            for null_source in null_sources:
                for residual_k in residual_ks:
                    if null_source == "none" and residual_k != 0:
                        continue
                    if null_source != "none" and residual_k == 0:
                        continue
                    for rule in rules:
                        d, m, c, scored = analyze_source(
                            store,
                            args.model,
                            source,
                            layer,
                            rule,
                            null_source,
                            residual_k,
                            args.basis_k,
                            args.spectrum_rank,
                            args.bootstrap_reps,
                            args.seed,
                        )
                        dim_rows.extend(d)
                        metric_rows.extend(m)
                        ci_rows.extend(c)
                        if not scored.empty:
                            score_frames.append(scored)

    dims = pd.DataFrame(dim_rows)
    metrics = pd.DataFrame(metric_rows)
    ci = pd.DataFrame(ci_rows)
    scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    dims.to_csv(args.output_dir / "pseudorep_dimensionality.csv", index=False)
    metrics.to_csv(args.output_dir / "pseudorep_projection_metrics.csv", index=False)
    ci.to_csv(args.output_dir / "pseudorep_image_level_ci.csv", index=False)
    scores.to_csv(args.output_dir / "pseudorep_projection_scores.csv", index=False)
    metadata = {
        "script": "experiments/pure_af_geometry/analyze_pseudoreplication_statistics.py",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "sources": sources,
        "layers": layers,
        "sampling_rules": rules,
        "residual_null_sources": null_sources,
        "residual_ks": residual_ks,
        "basis_k": args.basis_k,
        "spectrum_rank": args.spectrum_rank,
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(args.output_dir, metrics, dims, ci)
    print(f"[DONE] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
