#!/usr/bin/env python3
"""Cross-attack success-flow entry diagnostics.

Build a class success-flow basis from one attack family (PGD or Square), select
off/on-flow clean images under a possibly different test attack, then track
entry into the learned basis during attack optimization.
"""

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
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.offflow_entry_diagnostic import (  # noqa: E402
    LAYER_MAP,
    feature_vector,
    pgd_step,
    projection_energy,
    select_clean_correct_class,
)
from experiments.pure_af_geometry.plot_pgd_square_class_colored_flow import square_size  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import (  # noqa: E402
    load_model,
    margin,
    project_linf,
)


def load_basis(path: Path, k: int, basis_attack: str):
    z = np.load(path, allow_pickle=False)
    disp = z["feature_displacements"].astype(np.float32)
    meta = pd.DataFrame(json.loads(str(z["meta_json"])))
    local = []
    for _rid, g in meta.sort_values(["attack", "run_id", "step"]).groupby("run_id", sort=False):
        if basis_attack != "all" and str(g.attack.iloc[0]) != basis_attack:
            continue
        if int(g.final_success.max()) != 1:
            continue
        idx = g.index.to_numpy()
        for a, b in zip(idx[:-1], idx[1:]):
            v = disp[b] - disp[a]
            if np.linalg.norm(v) > 1e-12:
                local.append(v)
    if not local:
        raise RuntimeError(f"No local vectors found in {path} for basis_attack={basis_attack}")
    X = np.stack(local).astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(X, full_matrices=False)
    explained = (s * s) / max(float(np.sum(s * s)), 1e-12)
    return vt[:k].astype(np.float32), explained[:k]


def random_basis(dim: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((dim, k)).astype(np.float32)
    Q, _r = np.linalg.qr(X)
    return Q[:, :k].T.astype(np.float32)


def square_step(wrapper, x_adv, x0, y, eps, step, max_steps, min_square, gen, best_margin):
    _b, c, h, w = x0.shape
    side = square_size(step, max_steps, h, min_square)
    top = int(torch.randint(0, h - side + 1, (1,), generator=gen, device=x0.device).item())
    left = int(torch.randint(0, w - side + 1, (1,), generator=gen, device=x0.device).item())
    candidate = x_adv.clone()
    patch = (torch.rand((1, c, side, side), generator=gen, device=x0.device) * 2.0 - 1.0) * eps
    candidate[:, :, top : top + side, left : left + side] = x0[:, :, top : top + side, left : left + side] + patch
    candidate = project_linf(candidate, x0, eps)
    with torch.no_grad():
        cand_logits = wrapper(candidate)
        cand_margin = margin(cand_logits, y)
    if float(cand_margin.item()) < float(best_margin.item()):
        return candidate.detach(), cand_margin.detach(), 1
    return x_adv.detach(), best_margin.detach(), 0


def first_attack_vector(dataset, wrapper, layer, idx, label, basis, attack, eps, step_size, steps, square_min_size, seed, device):
    x_cpu, _ = dataset[idx]
    x0 = x_cpu.unsqueeze(0).to(device)
    y = torch.tensor([label], device=device)
    _logits0, h_prev = feature_vector(wrapper, x0, layer)
    if attack == "pgd":
        x1 = pgd_step(wrapper, x0, x0, y, eps, step_size)
        _logits1, h1 = feature_vector(wrapper, x1, layer)
        v = h1 - h_prev
        return projection_energy(v, basis)
    if attack == "square":
        gen = torch.Generator(device=device).manual_seed(seed)
        x_adv = x0.clone()
        with torch.no_grad():
            best_margin = margin(wrapper(x_adv), y)
        last_h = h_prev
        energies = []
        for step in range(1, min(steps, 10) + 1):
            x_adv, best_margin, accepted = square_step(
                wrapper, x_adv, x0, y, eps, step, steps, square_min_size, gen, best_margin
            )
            _logits, h = feature_vector(wrapper, x_adv, layer)
            v = h - last_h
            if np.linalg.norm(v) > 1e-12:
                energies.append(projection_energy(v, basis))
            last_h = h
        return float(np.mean(energies)) if energies else 0.0
    raise ValueError(attack)


def score_candidates(dataset, wrapper, layer, selected, basis, attack, eps, step_size, steps, square_min_size, seed, device):
    rows = []
    for image_ord, (idx, label) in enumerate(selected):
        pe = first_attack_vector(
            dataset,
            wrapper,
            layer,
            idx,
            label,
            basis,
            attack,
            eps,
            step_size,
            steps,
            square_min_size,
            seed + image_ord * 997,
            device,
        )
        x_cpu, _ = dataset[idx]
        x = x_cpu.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        with torch.no_grad():
            logits = wrapper(x)
        rows.append(
            {
                "dataset_idx": int(idx),
                "label": int(label),
                "initial_projection_energy": float(pe),
                "initial_margin": float(margin(logits, y).item()),
            }
        )
    return pd.DataFrame(rows)


def select_extreme_or_margin_matched(scores: pd.DataFrame, n_each: int, selection: str, margin_quantile: float):
    if selection == "extreme":
        off = scores.sort_values("initial_projection_energy").head(n_each).copy()
        on = scores.sort_values("initial_projection_energy").tail(n_each).copy()
    elif selection == "margin_matched":
        lo_thr = scores.initial_projection_energy.quantile(margin_quantile)
        hi_thr = scores.initial_projection_energy.quantile(1.0 - margin_quantile)
        low = scores[scores.initial_projection_energy <= lo_thr].copy().sort_values("initial_margin")
        high = scores[scores.initial_projection_energy >= hi_thr].copy().sort_values("initial_margin")
        pairs = []
        used_high: set[int] = set()
        for low_idx, low_row in low.iterrows():
            available = high.loc[[idx for idx in high.index if idx not in used_high]]
            if available.empty:
                break
            distances = (available.initial_margin - low_row.initial_margin).abs()
            high_idx = int(distances.idxmin())
            used_high.add(high_idx)
            pairs.append((low_idx, high_idx))
            if len(pairs) >= n_each:
                break
        if len(pairs) < n_each:
            raise RuntimeError(
                f"Only found {len(pairs)} margin-matched pairs; "
                f"try increasing --candidate-pool or --margin-quantile."
            )
        off = scores.loc[[p[0] for p in pairs]].copy()
        on = scores.loc[[p[1] for p in pairs]].copy()
        off["matched_pair_id"] = np.arange(len(pairs), dtype=int)
        on["matched_pair_id"] = np.arange(len(pairs), dtype=int)
        off["matched_margin_gap"] = [
            float(abs(scores.loc[hi, "initial_margin"] - scores.loc[lo, "initial_margin"])) for lo, hi in pairs
        ]
        on["matched_margin_gap"] = off["matched_margin_gap"].to_numpy()
    else:
        raise ValueError(selection)
    off["group"] = "offflow"
    on["group"] = "onflow"
    return pd.concat([off, on], ignore_index=True)


def attack_and_track(dataset, wrapper, layer, candidates, basis, random_basis_, attack, eps, step_size, steps, square_min_size, seed, device):
    rows = []
    for image_ord, row in candidates.reset_index(drop=True).iterrows():
        idx = int(row.dataset_idx)
        label = int(row.label)
        x_cpu, _ = dataset[idx]
        x0 = x_cpu.unsqueeze(0).to(device)
        x_adv = x0.clone()
        y = torch.tensor([label], device=device)
        gen = torch.Generator(device=device).manual_seed(seed + image_ord * 997)
        with torch.no_grad():
            best_margin = margin(wrapper(x_adv), y)
        prev_h = None
        crossed = False
        for step in range(steps + 1):
            logits, h = feature_vector(wrapper, x_adv, layer)
            pred = int(logits.argmax(1).item())
            if prev_h is None:
                learned_pe = np.nan
                random_pe = np.nan
            else:
                v = h - prev_h
                learned_pe = projection_energy(v, basis)
                random_pe = projection_energy(v, random_basis_)
            now_success = pred != label
            rows.append(
                {
                    "group": row.group,
                    "image_ord": int(image_ord),
                    "dataset_idx": idx,
                    "label": label,
                    "attack": attack,
                    "step": int(step),
                    "pred": pred,
                    "margin": float(margin(logits, y).item()),
                    "learned_projection_energy": learned_pe,
                    "random_projection_energy": random_pe,
                    "success": int(now_success),
                    "first_success": int(now_success and not crossed),
                    "initial_projection_energy": float(row.initial_projection_energy),
                }
            )
            if now_success:
                crossed = True
                break
            prev_h = h
            if step >= steps:
                continue
            if attack == "pgd":
                x_adv = pgd_step(wrapper, x_adv, x0, y, eps, step_size)
            elif attack == "square":
                x_adv, best_margin, _accepted = square_step(
                    wrapper, x_adv, x0, y, eps, step + 1, steps, square_min_size, gen, best_margin
                )
            else:
                raise ValueError(attack)
    return pd.DataFrame(rows)


def summarize(tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, image_ord), g in tracks.groupby(["group", "image_ord"]):
        g = g.sort_values("step")
        success_step = g[g.success == 1].step.min() if (g.success == 1).any() else np.nan
        pre = g[g.step > 0]
        before = pre[pre.step < success_step] if not np.isnan(success_step) else pre
        cross = pre[pre.step == success_step] if not np.isnan(success_step) else pre.iloc[0:0]
        rows.append(
            {
                "group": group,
                "image_ord": int(image_ord),
                "dataset_idx": int(g.dataset_idx.iloc[0]),
                "initial_projection_energy": float(g.initial_projection_energy.iloc[0]),
                "success": int((g.success == 1).any()),
                "success_step": float(success_step) if not np.isnan(success_step) else np.nan,
                "mean_pre_success_learned_pe": float(before.learned_projection_energy.mean()) if len(before) else np.nan,
                "max_pre_success_learned_pe": float(before.learned_projection_energy.max()) if len(before) else np.nan,
                "crossing_learned_pe": float(cross.learned_projection_energy.iloc[0]) if len(cross) else np.nan,
                "crossing_random_pe": float(cross.random_projection_energy.iloc[0]) if len(cross) else np.nan,
                "start_margin": float(g.margin.iloc[0]),
                "final_margin": float(g.margin.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("group", as_index=False)
        .agg(
            n=("success", "size"),
            success_rate=("success", "mean"),
            median_initial_pe=("initial_projection_energy", "median"),
            median_crossing_learned_pe=("crossing_learned_pe", "median"),
            median_crossing_random_pe=("crossing_random_pe", "median"),
            median_success_step=("success_step", "median"),
            mean_success_step=("success_step", "mean"),
            median_margin_drop=("final_margin", lambda x: float(np.nanmedian(summary.loc[x.index, "start_margin"] - x))),
        )
    )


def plot_tracks(tracks: pd.DataFrame, out_path: Path, title: str):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4), sharex="col", constrained_layout=True)
    colors = {"offflow": "#2563eb", "onflow": "#dc2626"}
    for group, g0 in tracks.groupby("group"):
        for _image_ord, g in g0.groupby("image_ord"):
            g = g.sort_values("step")
            axes[0, 0].plot(g.step, g.learned_projection_energy, color=colors[group], alpha=0.17, lw=0.9)
            axes[0, 1].plot(g.step, g.random_projection_energy, color=colors[group], alpha=0.17, lw=0.9)
            axes[1, 0].plot(g.step, g.margin, color=colors[group], alpha=0.17, lw=0.9)
        med = g0.groupby("step", as_index=False).agg(
            learned_pe=("learned_projection_energy", "median"),
            random_pe=("random_projection_energy", "median"),
            margin=("margin", "median"),
        )
        axes[0, 0].plot(med.step, med.learned_pe, color=colors[group], lw=2.4, label=group)
        axes[0, 1].plot(med.step, med.random_pe, color=colors[group], lw=2.4, label=group)
        axes[1, 0].plot(med.step, med.margin, color=colors[group], lw=2.4, label=group)
    axes[0, 0].set_title("Projection into learned basis")
    axes[0, 1].set_title("Projection into random basis")
    axes[1, 0].set_title("Margin")
    axes[1, 1].axis("off")
    for ax in [axes[0, 0], axes[0, 1]]:
        ax.set_ylim(0, 1)
        ax.set_ylabel("projection energy")
    axes[1, 0].axhline(0, color="black", lw=0.8, alpha=0.3)
    axes[1, 0].set_ylabel("true-vs-best-other margin")
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xlabel("attack step")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    wrapper = load_model(args.model, device)
    layer = LAYER_MAP[args.model][args.layer_group]
    basis, explained = load_basis(Path(args.basis_feature_npz), args.k, args.basis_attack)
    rand_basis = random_basis(basis.shape[1], args.k, args.seed + 1009)
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0 if args.step_size > 0 else eps / max(args.steps, 1)
    selected = select_clean_correct_class(dataset, wrapper, args.class_id, args.candidate_pool, device)
    scores = score_candidates(
        dataset, wrapper, layer, selected, basis, args.test_attack, eps, step_size, args.steps, args.square_min_size, args.seed, device
    )
    candidates = select_extreme_or_margin_matched(scores, args.n_each, args.selection, args.margin_quantile)
    tracks = attack_and_track(
        dataset, wrapper, layer, candidates, basis, rand_basis, args.test_attack, eps, step_size, args.steps, args.square_min_size, args.seed, device
    )
    summary = summarize(tracks)
    agg = aggregate(summary)
    stem = (
        f"cross_entry_{args.model}_{args.layer_group}_class{args.class_id}"
        f"_basis-{args.basis_attack}_test-{args.test_attack}_k{args.k}_n{args.n_each}"
    )
    if args.selection != "extreme":
        stem += f"_{args.selection}"
    scores.to_csv(out_dir / f"{stem}_candidate_scores.csv", index=False)
    candidates.to_csv(out_dir / f"{stem}_selected.csv", index=False)
    tracks.to_csv(out_dir / f"{stem}_tracks.csv", index=False)
    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)
    agg.to_csv(out_dir / f"{stem}_aggregate.csv", index=False)
    plot_tracks(
        tracks,
        out_dir / stem,
        f"class {args.class_id}: {args.test_attack.upper()} entry into {args.basis_attack.upper()} success-flow basis",
    )
    with open(out_dir / f"{stem}_metadata.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "layer": layer,
                "basis_explained": [float(x) for x in explained],
                "selection": args.selection,
                "margin_quantile": args.margin_quantile,
                "median_matched_margin_gap": float(candidates.get("matched_margin_gap", pd.Series(dtype=float)).median())
                if "matched_margin_gap" in candidates
                else None,
                "n_selected_clean_correct": len(selected),
                "aggregate": agg.to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    print(f"[SAVED] {out_dir / (stem + '.png')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--basis-feature-npz", required=True)
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/offflow_entry_diagnostic/cross_attack")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer-group", default="penultimate", choices=["hidden", "penultimate", "logits"])
    p.add_argument("--class-id", type=int, default=0)
    p.add_argument("--basis-attack", choices=["pgd", "square", "all"], default="pgd")
    p.add_argument("--test-attack", choices=["pgd", "square"], default="square")
    p.add_argument("--candidate-pool", type=int, default=300)
    p.add_argument("--n-each", type=int, default=40)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--step-size", type=float, default=0.0)
    p.add_argument("--square-min-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--selection", choices=["extreme", "margin_matched"], default="extreme")
    p.add_argument("--margin-quantile", type=float, default=0.4)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
