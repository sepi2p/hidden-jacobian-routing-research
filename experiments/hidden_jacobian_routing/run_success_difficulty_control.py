#!/usr/bin/env python3
"""Control success/failure transport separation for initial attack difficulty.

The script replays the registered first PGD step on exact final-test images,
merges frozen selected-layer transport energy, and evaluates whether transport
energy adds held-out information after clean margin, clean loss, gradient norm,
class, and first-step progress. Per-image collection is append-only/resumable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import load_model, margin, project_linf  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_metric(fn, y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or not np.isfinite(score).all():
        return float("nan")
    return float(fn(y, score))


def grouped_oof(df: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    x = pd.get_dummies(df[features], columns=["label"] if "label" in features else [], dtype=float).to_numpy(float)
    y = df.success.to_numpy(int)
    scores = np.full(len(df), np.nan)
    folds = min(5, int(np.bincount(y).min()))
    if folds < 2:
        return scores
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(x, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit(x[train], y[train])
        scores[test] = model.predict_proba(x[test])[:, 1]
    return scores


def bootstrap_delta(df: pd.DataFrame, base: np.ndarray, full: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y = df.success.to_numpy(int)
    values = []
    for _ in range(reps):
        idx = rng.integers(0, len(df), size=len(df))
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(average_precision_score(y[idx], full[idx]) - average_precision_score(y[idx], base[idx]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def greedy_match(df: pd.DataFrame) -> pd.DataFrame:
    """One-to-one success/failure matching on class, clean margin, grad norm."""
    work = df.copy()
    work["log_grad_l2"] = np.log1p(work.grad_l2)
    for col in ["clean_margin", "log_grad_l2"]:
        work[col + "_z"] = (work[col] - work[col].mean()) / max(work[col].std(), 1e-8)
    selected = []
    for label, group in work.groupby("label"):
        pos = group[group.success == 1].copy()
        neg = group[group.success == 0].copy()
        if pos.empty or neg.empty:
            continue
        if len(pos) < len(neg):
            anchors, pool = pos, neg
        else:
            anchors, pool = neg, pos
        available = set(pool.index.tolist())
        for idx, row in anchors.iterrows():
            if not available:
                break
            candidates = pool.loc[list(available)]
            dist = np.sqrt(
                (candidates.clean_margin_z - row.clean_margin_z) ** 2
                + (candidates.log_grad_l2_z - row.log_grad_l2_z) ** 2
            )
            partner = int(dist.idxmin())
            # A broad caliper avoids claiming matches across extreme difficulty.
            if float(dist.loc[partner]) <= 0.75:
                selected.extend([idx, partner])
                available.remove(partner)
    return work.loc[sorted(set(selected))].copy()


def analyze(df: pd.DataFrame, reps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_features = ["clean_margin", "clean_loss", "grad_l2", "grad_linf", "first_step_loss_change", "first_step_margin_drop", "label"]
    base = grouped_oof(df, base_features, seed)
    full = grouped_oof(df, base_features + ["transport_energy"], seed)
    y = df.success.to_numpy(int)
    delta = safe_metric(average_precision_score, y, full) - safe_metric(average_precision_score, y, base)
    lo, hi = bootstrap_delta(df, base, full, reps, seed + 17)
    matched = greedy_match(df)
    summary = pd.DataFrame(
        [
            {
                "model": df.model.iloc[0],
                "split_seed": int(df.split_seed.iloc[0]),
                "layer": df.layer.iloc[0],
                "n_images": len(df),
                "n_success": int(y.sum()),
                "n_failed": int((1 - y).sum()),
                "success_prevalence": float(y.mean()),
                "baseline_oof_auroc": safe_metric(roc_auc_score, y, base),
                "baseline_oof_auprc": safe_metric(average_precision_score, y, base),
                "plus_transport_oof_auroc": safe_metric(roc_auc_score, y, full),
                "plus_transport_oof_auprc": safe_metric(average_precision_score, y, full),
                "delta_transport_auprc": delta,
                "delta_transport_bootstrap_ci_low": lo,
                "delta_transport_bootstrap_ci_high": hi,
                "matched_n": len(matched),
                "matched_success": int(matched.success.sum()) if len(matched) else 0,
                "matched_failed": int((1 - matched.success).sum()) if len(matched) else 0,
                "matched_transport_auroc": safe_metric(
                    roc_auc_score, matched.success.to_numpy(int), matched.transport_energy.to_numpy(float)
                )
                if len(matched)
                else float("nan"),
            }
        ]
    )
    scored = df.copy()
    scored["baseline_oof_score"] = base
    scored["plus_transport_oof_score"] = full
    return summary, scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits-csv", default="artifacts/splits/cifar10_exact_splits.csv")
    parser.add_argument("--nested-root", default="analysis_outputs/hidden_jacobian_routing/exact_protocol/phase1_nested_layer_selection")
    parser.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    parser.add_argument("--eps", type=float, default=2.0)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--attack-seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-images", type=int, default=-1, help="Smoke/debug limit; negative uses the complete final-test split.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "DONE.json"
    if done.exists():
        print(f"[SKIP] {done}")
        return
    set_seed(args.split_seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wrapper = load_model(args.model, device)
    wrapper.eval()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    split_df = pd.read_csv(args.splits_csv)
    split_df = split_df[(split_df.model == args.model) & (split_df.split_seed == args.split_seed) & (split_df.split == "final_test")]
    split_df = split_df.sort_values(["label", "class_ord"])
    if args.max_images > 0:
        split_df = split_df.head(args.max_images)
    nested = Path(args.nested_root) / args.model / f"split_seed_{args.split_seed}"
    selection = pd.read_csv(nested / "nested_layer_selection_summary.csv")
    layer = str(selection[selection.layer_rule == "nested_selected_nonlogit"].reported_layer.iloc[0])
    energy = pd.read_csv(nested / "nested_layer_projection_scores.csv")
    energy = energy[(energy.split == "final_test") & (energy.layer == layer)].groupby("dataset_idx", as_index=False).energy.mean()
    outcomes = pd.read_csv(nested / "nested_layer_image_outcomes.csv")
    outcomes = outcomes[outcomes.split == "final_test"][["dataset_idx", "success", "clean_margin", "final_margin"]]

    rows_path = out / "difficulty_rows.csv"
    existing = pd.read_csv(rows_path) if rows_path.exists() else pd.DataFrame()
    completed = set(existing.dataset_idx.astype(int)) if len(existing) else set()
    rows = []
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    ordered = split_df
    for n, row in enumerate(ordered.itertuples(index=False), start=1):
        idx = int(row.dataset_idx)
        if idx in completed:
            continue
        x_cpu, label0 = dataset[idx]
        label = int(row.label)
        if int(label0) != label:
            raise RuntimeError(f"label mismatch for {idx}")
        x0 = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        x_req = x0.detach().requires_grad_(True)
        logits = wrapper(x_req)
        clean_loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(clean_loss, x_req)[0].detach()
        gen = torch.Generator(device=device).manual_seed(args.attack_seed + idx * 1009 + args.split_seed)
        x_start = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps, generator=gen), x0, eps).detach()
        xs = x_start.requires_grad_(True)
        start_logits = wrapper(xs)
        start_loss = F.cross_entropy(start_logits, y)
        start_grad = torch.autograd.grad(start_loss, xs)[0].detach()
        x1 = project_linf(x_start + step_size * start_grad.sign(), x0, eps).detach()
        with torch.no_grad():
            logits1 = wrapper(x1)
            loss1 = F.cross_entropy(logits1, y)
        rows.append(
            {
                "model": args.model,
                "split_seed": args.split_seed,
                "layer": layer,
                "dataset_idx": idx,
                "label": label,
                "clean_loss": float(clean_loss.item()),
                "grad_l2": float(grad.flatten(1).norm(dim=1).item()),
                "grad_linf": float(grad.abs().max().item()),
                "first_step_loss_change": float(loss1.item() - start_loss.item()),
                "first_step_margin_drop": float(margin(start_logits.detach(), y).item() - margin(logits1, y).item()),
            }
        )
        if len(rows) >= 20 or n == len(ordered):
            chunk = pd.DataFrame(rows)
            existing = pd.concat([existing, chunk], ignore_index=True)
            existing.to_csv(rows_path, index=False)
            completed.update(chunk.dataset_idx.astype(int))
            rows = []
            print(f"[{args.model} seed={args.split_seed}] {len(completed)}/{len(ordered)}", flush=True)

    merged = existing.merge(outcomes, on="dataset_idx", how="inner").merge(energy.rename(columns={"energy": "transport_energy"}), on="dataset_idx", how="inner")
    # Use the frozen clean margin from the exact attack run.
    merged["clean_margin"] = merged["clean_margin"]
    summary, scored = analyze(merged, args.bootstrap, args.split_seed)
    scored.to_csv(out / "difficulty_scored_rows.csv", index=False)
    summary.to_csv(out / "difficulty_summary.csv", index=False)
    done.write_text(json.dumps({"status": "complete", "model": args.model, "split_seed": args.split_seed, "rows": len(merged)}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
