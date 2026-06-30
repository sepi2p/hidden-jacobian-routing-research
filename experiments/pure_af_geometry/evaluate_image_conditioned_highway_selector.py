#!/usr/bin/env python3
"""Evaluate image-conditioned selection of boundary-leading highway routes.

This experiment tests whether the useful highway route is image/state-dependent.
It fits a high-mobility basis from non-adversarial mobility controls, ranks
signed routes globally by train-split margin drop, and then evaluates held-out
clean images by applying one pullback step along signed highway routes.

The key comparison is:

* fixed global top route;
* random signed highway route;
* image-conditioned route selected by observed one-step margin drop;
* image-conditioned route predicted by the local pixel-space margin gradient.
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
from sklearn.utils.extmath import randomized_svd
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import margin, project_linf  # noqa: E402


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def vector_key(model: str, source: str, layer: str) -> str:
    return f"{model}__{source}__{layer}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pca_basis(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    kk = min(k, x.shape[0] - 1, x.shape[1])
    if kk < 1:
        raise ValueError("PCA rank is zero.")
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    xc = x - mean
    _u, _s, vt = randomized_svd(xc, n_components=kk, random_state=seed)
    return mean, vt.astype(np.float32)


class ArtifactStore:
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.rows = pd.read_csv(input_dir / "segment_metadata.csv")
        self.splits = pd.read_csv(input_dir / "image_splits.csv")
        self.outcomes = pd.read_csv(input_dir / "image_outcomes.csv")
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

    def eval_images(self, model: str, split: str, max_images: int) -> pd.DataFrame:
        base = self.outcomes[(self.outcomes.model == model) & (self.outcomes.source == "pgd")][
            ["image_ord", "dataset_idx", "label"]
        ].drop_duplicates()
        sub = base.merge(self.splits, on="image_ord", how="left")
        sub = sub[sub.split == split].sort_values("image_ord")
        if max_images > 0:
            sub = sub.head(max_images)
        return sub.reset_index(drop=True)


def fit_highway_basis(store: ArtifactStore, model: str, source: str, layer: str, k: int, seed: int):
    rows, x = store.rows_for(model, source, layer)
    if rows.empty:
        raise RuntimeError(f"No highway source rows for {source}/{layer}.")
    train = rows["split"].to_numpy() == "train"
    if train.sum() < max(8, k + 2):
        raise RuntimeError(f"Too few train highway vectors: {train.sum()}.")
    mean, basis = pca_basis(x[train], k, seed)
    return mean, basis, int(train.sum())


def project_attack_rows(store: ArtifactStore, model: str, sources: list[str], layer: str, basis: np.ndarray) -> pd.DataFrame:
    frames = []
    for source in sources:
        rows, x = store.rows_for(model, source, layer)
        if rows.empty:
            continue
        coeff = x @ basis.T
        out = rows.copy()
        out["margin_drop"] = out["margin_before"].astype(float) - out["margin_after"].astype(float)
        for j in range(coeff.shape[1]):
            out[f"pc{j+1}_coeff"] = coeff[:, j]
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rank_signed_routes(projected: pd.DataFrame, rank_sources: list[str], k: int) -> pd.DataFrame:
    train = projected[(projected["split"] == "train") & (projected["source"].isin(rank_sources))].copy()
    rows = []
    for pc in range(1, k + 1):
        coeff = train[f"pc{pc}_coeff"].to_numpy(dtype=float)
        drops = train["margin_drop"].to_numpy(dtype=float)
        for sign, sign_label in [(1, "+"), (-1, "-")]:
            w = np.maximum(sign * coeff, 0.0)
            if w.sum() <= 1e-12:
                score = np.nan
                n_active = 0
            else:
                score = float(np.sum(w * drops) / np.sum(w))
                n_active = int((w > 0).sum())
            rows.append(
                {
                    "pc": pc,
                    "sign": sign,
                    "route": f"pc{pc}{sign_label}",
                    "train_weighted_margin_drop_score": score,
                    "n_active_train_steps": n_active,
                }
            )
    out = pd.DataFrame(rows).sort_values("train_weighted_margin_drop_score", ascending=False, na_position="last")
    out = out.reset_index(drop=True)
    out["global_rank"] = np.arange(1, len(out) + 1)
    return out


def eval_clean(wrapper, x: torch.Tensor, y: torch.Tensor) -> dict:
    with torch.no_grad():
        logits = wrapper(x)
        return {
            "pred": int(logits.argmax(1).item()),
            "margin": float(margin(logits, y).item()),
        }


def feature_tensor(wrapper, x: torch.Tensor, layer: str) -> torch.Tensor:
    _logits, feats, _raw = wrapper.forward_with_features(x)
    if layer not in feats:
        raise RuntimeError(f"Layer {layer} not captured.")
    return feats[layer]


def route_pullback_step(
    wrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    layer: str,
    route_direction: torch.Tensor,
    eps: float,
    step_size: float,
    margin_grad_x: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    x_probe = x.detach().requires_grad_(True)
    h = feature_tensor(wrapper, x_probe, layer)
    scalar = (h * route_direction.view_as(h)).sum()
    grad = torch.autograd.grad(scalar, x_probe)[0]
    pixel_margin_grad_score = np.nan
    if margin_grad_x is not None:
        # Larger score means a sign step along this pullback direction should
        # decrease the true-class margin more strongly.
        pixel_margin_grad_score = float((-(margin_grad_x.detach()) * grad.detach().sign()).sum().item())
    x_next = project_linf(x + step_size * grad.sign(), x, eps)
    with torch.no_grad():
        logits = wrapper(x_next)
        h_next = feature_tensor(wrapper, x_next, layer)
        h0 = feature_tensor(wrapper, x, layer)
        dh = h_next - h0
        coeff = dh @ route_direction.view(1, -1).T
        route_energy = float((coeff.squeeze() ** 2 / (dh.pow(2).sum(dim=1).clamp_min(1e-12))).item())
    return x_next.detach(), {
        "route_energy": route_energy,
        "feature_speed": float(torch.norm(dh, dim=1).item()),
        "pixel_margin_grad_score": pixel_margin_grad_score,
    }


def ce_step(wrapper, x: torch.Tensor, y: torch.Tensor, eps: float, step_size: float) -> torch.Tensor:
    x_probe = x.detach().requires_grad_(True)
    logits = wrapper(x_probe)
    loss = torch.nn.functional.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_probe)[0]
    return project_linf(x + step_size * grad.sign(), x, eps).detach()


def margin_pixel_grad(wrapper, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_probe = x.detach().requires_grad_(True)
    logits = wrapper(x_probe)
    m = margin(logits, y)
    return torch.autograd.grad(m, x_probe)[0].detach()


def evaluate_one_image(
    wrapper,
    dataset,
    row,
    layer: str,
    basis_t: torch.Tensor,
    routes: pd.DataFrame,
    eps: float,
    step_size: float,
    top_sets: list[int],
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[list[dict], pd.DataFrame]:
    x_cpu, _ = dataset[int(row.dataset_idx)]
    x = x_cpu.unsqueeze(0).to(device)
    y = torch.tensor([int(row.label)], device=device)
    clean = eval_clean(wrapper, x, y)
    signed_routes = routes.to_dict("records")
    route_rows = []

    margin_grad_x = margin_pixel_grad(wrapper, x, y)
    for route in signed_routes:
        direction = torch.as_tensor(route["sign"], dtype=basis_t.dtype, device=device) * basis_t[int(route["pc"]) - 1]
        x_next, extra = route_pullback_step(wrapper, x, y, layer, direction, eps, step_size, margin_grad_x)
        after = eval_clean(wrapper, x_next, y)
        route_rows.append(
            {
                "image_ord": int(row.image_ord),
                "dataset_idx": int(row.dataset_idx),
                "label": int(row.label),
                "pc": int(route["pc"]),
                "sign": int(route["sign"]),
                "route": str(route["route"]),
                "global_rank": int(route["global_rank"]),
                "global_score": float(route["train_weighted_margin_drop_score"]),
                "clean_margin": float(clean["margin"]),
                "after_margin": float(after["margin"]),
                "margin_drop": float(clean["margin"] - after["margin"]),
                "success": int(after["pred"] != int(row.label)),
                "after_pred": int(after["pred"]),
                **extra,
            }
        )
    route_df = pd.DataFrame(route_rows)
    image_rows = []

    def choose_variant(name: str, sub: pd.DataFrame, criterion: str) -> None:
        if sub.empty:
            return
        chosen = sub.loc[sub[criterion].astype(float).idxmax()]
        image_rows.append(
            {
                "image_ord": int(row.image_ord),
                "dataset_idx": int(row.dataset_idx),
                "label": int(row.label),
                "variant": name,
                "chosen_route": str(chosen.route),
                "chosen_pc": int(chosen.pc),
                "chosen_sign": int(chosen.sign),
                "chosen_global_rank": int(chosen.global_rank),
                "clean_margin": float(chosen.clean_margin),
                "after_margin": float(chosen.after_margin),
                "margin_drop": float(chosen.margin_drop),
                "success": int(chosen.success),
                "route_energy": float(chosen.route_energy),
                "feature_speed": float(chosen.feature_speed),
                "selector_score": float(chosen[criterion]),
            }
        )

    choose_variant("image_best_observed_all", route_df, "margin_drop")
    choose_variant("image_best_pixel_margin_grad_all", route_df, "pixel_margin_grad_score")
    choose_variant("image_best_highway_energy_all", route_df, "route_energy")
    for n in top_sets:
        choose_variant(f"image_best_observed_top{n}", route_df[route_df.global_rank <= n], "margin_drop")

    global_top = route_df.loc[route_df.global_rank.astype(int).idxmin()]
    image_rows.append(
        {
            "image_ord": int(row.image_ord),
            "dataset_idx": int(row.dataset_idx),
            "label": int(row.label),
            "variant": "global_rank1",
            "chosen_route": str(global_top.route),
            "chosen_pc": int(global_top.pc),
            "chosen_sign": int(global_top.sign),
            "chosen_global_rank": int(global_top.global_rank),
            "clean_margin": float(global_top.clean_margin),
            "after_margin": float(global_top.after_margin),
            "margin_drop": float(global_top.margin_drop),
            "success": int(global_top.success),
            "route_energy": float(global_top.route_energy),
            "feature_speed": float(global_top.feature_speed),
            "selector_score": float(global_top.global_score),
        }
    )

    for n in [1, 5, 10]:
        choices = route_df.sample(n=min(n, len(route_df)), random_state=int(rng.integers(0, 2**31 - 1)))
        choose_variant(f"random_best_observed_{n}", choices, "margin_drop")

    x_ce = ce_step(wrapper, x, y, eps, step_size)
    ce_after = eval_clean(wrapper, x_ce, y)
    image_rows.append(
        {
            "image_ord": int(row.image_ord),
            "dataset_idx": int(row.dataset_idx),
            "label": int(row.label),
            "variant": "ce_one_step",
            "chosen_route": "",
            "chosen_pc": np.nan,
            "chosen_sign": np.nan,
            "chosen_global_rank": np.nan,
            "clean_margin": float(clean["margin"]),
            "after_margin": float(ce_after["margin"]),
            "margin_drop": float(clean["margin"] - ce_after["margin"]),
            "success": int(ce_after["pred"] != int(row.label)),
            "route_energy": np.nan,
            "feature_speed": np.nan,
            "selector_score": np.nan,
        }
    )

    return image_rows, route_df


def summarize(per_image: pd.DataFrame) -> pd.DataFrame:
    order = [
        "global_rank1",
        "random_best_observed_1",
        "random_best_observed_5",
        "random_best_observed_10",
        "image_best_observed_top3",
        "image_best_observed_top5",
        "image_best_observed_top10",
        "image_best_highway_energy_all",
        "image_best_pixel_margin_grad_all",
        "image_best_observed_all",
        "ce_one_step",
    ]
    out = (
        per_image.groupby("variant", dropna=False)
        .agg(
            n=("margin_drop", "size"),
            asr=("success", "mean"),
            mean_margin_drop=("margin_drop", "mean"),
            median_margin_drop=("margin_drop", "median"),
            mean_after_margin=("after_margin", "mean"),
            mean_chosen_rank=("chosen_global_rank", "mean"),
            median_chosen_rank=("chosen_global_rank", "median"),
            mean_route_energy=("route_energy", "mean"),
            mean_feature_speed=("feature_speed", "mean"),
        )
        .reset_index()
    )
    out["variant"] = pd.Categorical(out["variant"], categories=order, ordered=True)
    return out.sort_values("variant").reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    keep = [
        "global_rank1",
        "random_best_observed_1",
        "random_best_observed_5",
        "random_best_observed_10",
        "image_best_highway_energy_all",
        "image_best_pixel_margin_grad_all",
        "image_best_observed_all",
        "ce_one_step",
    ]
    sub = summary[summary.variant.astype(str).isin(keep)].copy()
    labels = {
        "global_rank1": "Global rank-1",
        "random_best_observed_1": "Random route",
        "random_best_observed_5": "Best of 5 random",
        "random_best_observed_10": "Best of 10 random",
        "image_best_highway_energy_all": "Max highway energy",
        "image_best_pixel_margin_grad_all": "Pixel-grad selected",
        "image_best_observed_all": "Observed best route",
        "ce_one_step": "CE gradient",
    }
    sub["label"] = [labels[str(v)] for v in sub.variant]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), dpi=180)
    axes[0].bar(sub["label"], sub["mean_margin_drop"], color="#4c78a8")
    axes[0].set_ylabel("Mean one-step margin drop")
    axes[0].set_title("Image-conditioned route selection")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(sub["label"], sub["asr"], color="#59a14f")
    axes[1].set_ylabel("One-step ASR")
    axes[1].set_ylim(0, max(0.05, float(sub["asr"].max()) * 1.25))
    axes[1].set_title("One-step success")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "image_conditioned_highway_selector.png", bbox_inches="tight")
    fig.savefig(out_dir / "image_conditioned_highway_selector.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(out_dir: Path, route_ranking: pd.DataFrame, summary: pd.DataFrame, args, n_highway_train: int) -> None:
    lines = [
        "# Image-Conditioned Highway Selector",
        "",
        "This experiment asks whether the useful highway route can be selected per image/state.",
        "",
        "## Setup",
        "",
        f"- Model: `{args.model}`",
        f"- Layer: `{args.layer}`",
        f"- Highway source: `{args.highway_source}`",
        f"- Highway train vectors: `{n_highway_train}`",
        f"- Signed highway routes: `{2 * args.highway_k}`",
        f"- Evaluation split: `{args.split}`",
        f"- Test images: `{args.images}`",
        f"- One-step eps: `{args.eps}/255`",
        f"- One-step step size: `{args.step_size}/255`",
        "",
        "## Top Global Routes",
        "",
    ]
    for r in route_ranking.head(10).itertuples():
        lines.append(
            f"- rank {int(r.global_rank)}: `{r.route}`, train score={float(r.train_weighted_margin_drop_score):.4f}, "
            f"active train steps={int(r.n_active_train_steps)}"
        )
    lines.extend(["", "## Held-Out Selector Summary", ""])
    for r in summary.itertuples():
        lines.append(
            f"- `{r.variant}`: mean_margin_drop={float(r.mean_margin_drop):.4f}, "
            f"median_margin_drop={float(r.median_margin_drop):.4f}, ASR={float(r.asr):.3f}, "
            f"mean_rank={float(r.mean_chosen_rank):.2f}" if np.isfinite(float(r.mean_chosen_rank)) else
            f"- `{r.variant}`: mean_margin_drop={float(r.mean_margin_drop):.4f}, "
            f"median_margin_drop={float(r.median_margin_drop):.4f}, ASR={float(r.asr):.3f}"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "If image-conditioned selectors substantially outperform global rank-1 and random highway routes, then highway usefulness is state-dependent. If the pixel-gradient selector approaches observed-best performance, the selector can be characterized locally rather than treated as an unexplained oracle.",
            "",
        ]
    )
    (out_dir / "image_conditioned_highway_selector_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = ArtifactStore(Path(args.input_dir))
    _mean, basis, n_highway_train = fit_highway_basis(store, args.model, args.highway_source, args.layer, args.highway_k, args.seed)
    projected = project_attack_rows(store, args.model, [x.strip() for x in args.rank_sources.split(",") if x.strip()], args.layer, basis)
    route_ranking = rank_signed_routes(projected, [x.strip() for x in args.rank_sources.split(",") if x.strip()], args.highway_k)
    eval_images = store.eval_images(args.model, args.split, args.images)
    if eval_images.empty:
        raise RuntimeError("No evaluation images found.")

    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    wrapper = load_model(args.model, device).eval()
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=device)
    eps = args.eps / 255.0
    step_size = args.step_size / 255.0
    rng = np.random.default_rng(args.seed)
    top_sets = parse_int_csv(args.top_sets)
    all_image_rows = []
    all_route_rows = []
    for i, row in enumerate(eval_images.itertuples(index=False), start=1):
        image_rows, route_rows = evaluate_one_image(
            wrapper,
            dataset,
            row,
            args.layer,
            basis_t,
            route_ranking,
            eps,
            step_size,
            top_sets,
            rng,
            device,
        )
        all_image_rows.extend(image_rows)
        all_route_rows.append(route_rows)
        if i % args.checkpoint_every == 0:
            pd.DataFrame(all_image_rows).to_csv(out_dir / "partial_image_conditioned_selector_per_image.csv", index=False)
            pd.concat(all_route_rows, ignore_index=True).to_csv(out_dir / "partial_image_conditioned_selector_route_candidates.csv", index=False)
            print(f"[{i}/{len(eval_images)}] rows={len(all_image_rows)}", flush=True)

    per_image = pd.DataFrame(all_image_rows)
    route_candidates = pd.concat(all_route_rows, ignore_index=True)
    summary = summarize(per_image)
    route_ranking.to_csv(out_dir / "global_route_ranking.csv", index=False)
    route_candidates.to_csv(out_dir / "image_conditioned_selector_route_candidates.csv", index=False)
    per_image.to_csv(out_dir / "image_conditioned_selector_per_image.csv", index=False)
    summary.to_csv(out_dir / "image_conditioned_selector_summary.csv", index=False)
    plot_summary(summary, out_dir)
    write_summary(out_dir, route_ranking, summary, args, n_highway_train)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(summary.to_string(index=False), flush=True)
    print(f"[SAVED] {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/balanced_full_bbb_resnet50_c200_auto")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/image_conditioned_highway_selector_bbb_resnet50_c50")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--layer", default="layer4")
    p.add_argument("--highway-source", default="mobility_top_walk_square_budget")
    p.add_argument("--rank-sources", default="pgd,square")
    p.add_argument("--highway-k", type=int, default=20)
    p.add_argument("--split", default="test")
    p.add_argument("--images", type=int, default=50)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--top-sets", default="3,5,10")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
