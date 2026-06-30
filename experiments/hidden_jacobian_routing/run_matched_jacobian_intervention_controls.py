#!/usr/bin/env python3
"""Matched intervention controls for hidden-Jacobian transport directions.

The old intervention experiment compared learned attack-transport PCs against
random feature directions.  After the hidden-Jacobian controls, the fair test is
transport directions versus JVP/Jacobian and matched-random baselines under the
same pullback attack protocol.
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
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import load_model  # noqa: E402
from experiments.hidden_jacobian_routing.common import margin  # noqa: E402
from experiments.hidden_jacobian_routing.test_jacobian_basis_and_residual_transport import (  # noqa: E402
    SegmentStore,
    collect_vectors,
    normalize_rows,
    pca_basis,
)


BASE = Path("analysis_outputs/hidden_jacobian_routing/jacobian_null_response")

MODEL_CONFIG = {
    "bbb_resnet50": {
        "layer": "layer4",
        "input_dir": BASE / "balanced_full_bbb_resnet50_c200_auto",
        "basis_dir": BASE / "jacobian_basis_residual_bbb_resnet50_d64",
        "whitened_dir": BASE / "clean_whitened_mobility_jvp_bbb_resnet50_layer4_c100_final_step2_whitened",
    },
    "bbb_vgg19_bn": {
        "layer": "block5",
        "input_dir": BASE / "balanced_full_bbb_vgg19_bn_c200_final_step1",
        "basis_dir": BASE / "jacobian_basis_residual_bbb_vgg19_bn_block5_c200_final_step1",
        "whitened_dir": BASE / "clean_whitened_mobility_jvp_bbb_vgg19_bn_block5_c100_final_step2_whitened",
    },
    "bbb_densenet": {
        "layer": "denseblock3",
        "input_dir": BASE / "balanced_full_bbb_densenet_c200_final_step1",
        "basis_dir": BASE / "jacobian_basis_residual_bbb_densenet_denseblock3_c200_final_step1",
        "whitened_dir": BASE / "clean_whitened_mobility_jvp_bbb_densenet_denseblock3_c100_final_step2_whitened",
    },
    "bbb_inception_v3": {
        "layer": "mixed6",
        "input_dir": BASE / "balanced_full_bbb_inception_v3_c200_final_step1",
        "basis_dir": BASE / "jacobian_basis_residual_bbb_inception_v3_mixed6_c200_final_step1",
        "whitened_dir": BASE / "clean_whitened_mobility_jvp_bbb_inception_v3_mixed6_c100_final_step2_whitened",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_linf(x_adv: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)


def feature_tensor(wrapper, x: torch.Tensor, layer: str) -> torch.Tensor:
    _logits, feats, _raw = wrapper.forward_with_features(x)
    if layer not in feats:
        raise RuntimeError(f"Layer {layer} not captured.")
    return feats[layer]


def eval_state(wrapper, x: torch.Tensor, y: torch.Tensor) -> dict:
    with torch.no_grad():
        logits = wrapper(x)
        probs = F.softmax(logits, dim=1)
    return {
        "pred": int(logits.argmax(1).item()),
        "success": int(logits.argmax(1).item() != int(y.item())),
        "margin": float(margin(logits, y).item()),
        "true_prob": float(probs[0, int(y.item())].item()),
    }


def ce_grad(wrapper, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    probe = x.detach().requires_grad_(True)
    logits = wrapper(probe)
    loss = F.cross_entropy(logits, y)
    return torch.autograd.grad(loss, probe)[0].detach()


def pullback_grad(wrapper, x: torch.Tensor, layer: str, u_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    probe = x.detach().requires_grad_(True)
    _logits, feats, _raw = wrapper.forward_with_features(probe)
    h = feats[layer]
    u = torch.as_tensor(u_np, dtype=h.dtype, device=h.device).view_as(h)
    scalar = (h * u).sum()
    grad = torch.autograd.grad(scalar, probe)[0].detach()
    return grad, h.detach()


def flat_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.flatten(1).float()
    bb = b.flatten(1).float()
    return float(((aa * bb).sum(1) / (aa.norm(dim=1).clamp_min(1e-12) * bb.norm(dim=1).clamp_min(1e-12))).item())


def np_cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def load_eval_images(input_dir: Path, model: str, max_images: int, split: str = "test") -> pd.DataFrame:
    outcomes = pd.read_csv(input_dir / "image_outcomes.csv")
    splits = pd.read_csv(input_dir / "image_splits.csv")
    base = outcomes[(outcomes.model == model) & (outcomes.source == "pgd")][
        ["image_ord", "dataset_idx", "label", "clean_pred", "clean_margin"]
    ].drop_duplicates()
    base = base.merge(splits, on="image_ord", how="left")
    sub = base[base.split == split].sort_values("image_ord").reset_index(drop=True)
    if max_images > 0:
        sub = sub.head(max_images)
    return sub


def fit_failed_basis(input_dir: Path, model: str, layer: str, k: int, seed: int) -> np.ndarray | None:
    store = SegmentStore(input_dir)
    _rows, x = collect_vectors(store, model, ["pgd", "square"], layer, split="train", final_success=0)
    if len(x) < max(k + 1, 4):
        return None
    x = normalize_rows(x)
    _mean, basis, _kk = pca_basis(x, k, seed + 33, False)
    return basis.astype(np.float32)


def normalized_basis(x: np.ndarray, k: int) -> np.ndarray:
    b = x[:k].astype(np.float32)
    return b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)


def load_basis_family(model: str, k: int, include_failed: bool, seed: int) -> tuple[str, dict[str, np.ndarray]]:
    cfg = MODEL_CONFIG[model]
    layer = cfg["layer"]
    raw = np.load(cfg["basis_dir"] / "basis_vectors.npz")
    out = {
        "transport_pc": normalized_basis(raw["transport_basis"], k),
        "jvp_sketch_pc": normalized_basis(raw["jvp_sketch_basis"], k),
        "residual_transport_pc": normalized_basis(raw["residual_transport_basis"], k),
    }
    wpath = cfg["whitened_dir"] / "whitened_basis_vectors.npz"
    whitener_path = cfg["whitened_dir"] / "clean_whitener.npz"
    if wpath.exists() and whitener_path.exists():
        w = np.load(wpath)
        wh = np.load(whitener_path)
        # z = W(h - mu), so <z, b> = <h, W^T b> plus a constant.
        raw_dirs = w["jvp_basis"][:k] @ wh["whiten"].astype(np.float32)
        out["whitened_jvp_pc_raw_pullback"] = normalized_basis(raw_dirs, k)
    if include_failed:
        failed = fit_failed_basis(cfg["input_dir"], model, layer, k, seed)
        if failed is not None:
            out["failed_attack_pc"] = normalized_basis(failed, k)
    return layer, out


def choose_signs(wrapper, dataset, train_images: pd.DataFrame, layer: str, basis_families: dict[str, np.ndarray], args, device: torch.device) -> pd.DataFrame:
    rows = []
    eps = args.eps / 255.0
    sign_images = train_images.head(args.sign_images)
    for basis_name, basis in basis_families.items():
        for pc_idx, u in enumerate(basis[: args.k], start=1):
            for sign in [-1, 1]:
                drops = []
                for row in sign_images.itertuples(index=False):
                    x_cpu, _ = dataset[int(row.dataset_idx)]
                    x = x_cpu.unsqueeze(0).to(device)
                    y = torch.tensor([int(row.label)], device=device)
                    clean = eval_state(wrapper, x, y)
                    grad, _h0 = pullback_grad(wrapper, x, layer, sign * u)
                    adv = project_linf(x + eps * grad.sign(), x, eps)
                    after = eval_state(wrapper, adv, y)
                    drops.append(clean["margin"] - after["margin"])
                rows.append(
                    {
                        "basis": basis_name,
                        "pc": pc_idx,
                        "sign": sign,
                        "mean_train_margin_drop": float(np.mean(drops)) if drops else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def sign_lookup(sign_df: pd.DataFrame) -> dict[tuple[str, int], int]:
    out = {}
    for (basis, pc), g in sign_df.groupby(["basis", "pc"]):
        best = g.sort_values("mean_train_margin_drop", ascending=False).iloc[0]
        out[(str(basis), int(pc))] = int(best.sign)
    return out


def random_unit_feature_dirs(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def initial_metrics(wrapper, x: torch.Tensor, y: torch.Tensor, layer: str, u: np.ndarray) -> tuple[torch.Tensor, float, float]:
    pb, _h = pullback_grad(wrapper, x, layer, u)
    cg = ce_grad(wrapper, x, y)
    return pb, float(pb.flatten(1).norm(dim=1).item()), flat_cos(pb, cg)


def choose_matched_random(wrapper, x: torch.Tensor, y: torch.Tensor, layer: str, target_norm: float, target_ce_cos: float, d: int, rng: np.random.Generator, n: int) -> tuple[np.ndarray, float, float]:
    best = None
    for u in random_unit_feature_dirs(rng, n, d):
        _g, norm, ce_cos = initial_metrics(wrapper, x, y, layer, u)
        score = abs(np.log(max(norm, 1e-12)) - np.log(max(target_norm, 1e-12))) + abs(ce_cos - target_ce_cos)
        if best is None or score < best[0]:
            best = (score, u.astype(np.float32), norm, ce_cos)
    assert best is not None
    return best[1], best[2], best[3]


def attack_feature_direction(wrapper, x: torch.Tensor, y: torch.Tensor, layer: str, u: np.ndarray, eps: float, steps: int, step_size: float) -> tuple[torch.Tensor, dict]:
    x0 = x.detach()
    x_adv = x0.clone()
    first_pb = None
    first_ce_cos = np.nan
    first_norm = np.nan
    for step in range(steps):
        pb, _h = pullback_grad(wrapper, x_adv, layer, u)
        if step == 0:
            first_pb = pb.detach()
            first_norm = float(pb.flatten(1).norm(dim=1).item())
            first_ce_cos = flat_cos(pb, ce_grad(wrapper, x_adv, y))
        x_adv = project_linf(x_adv + step_size * pb.sign(), x0, eps)
    assert first_pb is not None
    return x_adv.detach(), {"pullback_norm": first_norm, "pullback_ce_cos": first_ce_cos}


def ce_pgd(wrapper, x: torch.Tensor, y: torch.Tensor, eps: float, steps: int, step_size: float) -> torch.Tensor:
    x0 = x.detach()
    x_adv = x0.clone()
    for _ in range(steps):
        grad = ce_grad(wrapper, x_adv, y)
        x_adv = project_linf(x_adv + step_size * grad.sign(), x0, eps)
    return x_adv.detach()


def run(args) -> None:
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    rng = np.random.default_rng(args.seed)

    rows = []
    sign_rows = []
    completed: set[tuple] = set()
    partial_path = out_dir / "matched_intervention_per_image.partial.csv"
    final_path = out_dir / "matched_intervention_per_image.csv"
    if args.resume and (final_path.exists() or partial_path.exists()):
        old_path = final_path if final_path.exists() else partial_path
        old = pd.read_csv(old_path)
        rows = old.to_dict("records")
        for r in old[["model", "dataset_idx", "variant", "pc"]].drop_duplicates().itertuples(index=False):
            completed.add((str(r.model), int(r.dataset_idx), str(r.variant), int(r.pc)))
        sign_path = out_dir / "matched_intervention_sign_selection.csv"
        if sign_path.exists():
            sign_rows = pd.read_csv(sign_path).to_dict("records")
        print(f"[resume] loaded rows={len(rows)} completed={len(completed)} from {old_path}", flush=True)
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        if model not in MODEL_CONFIG:
            raise RuntimeError(f"Unsupported model: {model}")
        cfg = MODEL_CONFIG[model]
        layer, basis_families = load_basis_family(model, args.k, args.include_failed, args.seed)
        wrapper = load_model(model, device).eval()
        train_images = load_eval_images(cfg["input_dir"], model, args.sign_images, split="train")
        test_images = load_eval_images(cfg["input_dir"], model, args.images, split="test")
        signs_df = choose_signs(wrapper, dataset, train_images, layer, basis_families, args, device)
        signs = sign_lookup(signs_df)
        sign_rows.extend([{**r, "model": model, "layer": layer} for r in signs_df.to_dict("records")])
        eps = args.eps / 255.0
        step_size = args.step_size / 255.0 if args.step_size > 0 else eps / max(args.steps, 1)
        print(f"[model] {model} layer={layer} test={len(test_images)} bases={list(basis_families)}", flush=True)

        for image_i, row in enumerate(test_images.itertuples(index=False), start=1):
            x_cpu, _ = dataset[int(row.dataset_idx)]
            x = x_cpu.unsqueeze(0).to(device)
            y = torch.tensor([int(row.label)], device=device)
            clean = eval_state(wrapper, x, y)
            h0 = feature_tensor(wrapper, x, layer).detach().cpu().numpy()[0].astype(np.float32)

            adv_ce = ce_pgd(wrapper, x, y, eps, args.steps, step_size)
            ce_after = eval_state(wrapper, adv_ce, y)
            if (model, int(row.dataset_idx), "ce_pgd", 0) not in completed:
                rows.append(
                    {
                        "model": model,
                        "layer": layer,
                        "dataset_idx": int(row.dataset_idx),
                        "image_ord": int(row.image_ord),
                        "label": int(row.label),
                        "basis": "ce_pgd",
                        "pc": 0,
                        "variant": "ce_pgd",
                        "eps_255": args.eps,
                        "steps": args.steps,
                        "success": ce_after["success"],
                        "margin_drop": clean["margin"] - ce_after["margin"],
                        "true_prob_drop": clean["true_prob"] - ce_after["true_prob"],
                        "pullback_norm": np.nan,
                        "pullback_ce_cos": np.nan,
                        "realized_feature_cos": np.nan,
                    }
                )
                completed.add((model, int(row.dataset_idx), "ce_pgd", 0))

            for basis_name, basis in basis_families.items():
                for pc_idx, u0 in enumerate(basis[: args.k], start=1):
                    sign = signs[(basis_name, pc_idx)]
                    u = (sign * u0).astype(np.float32)
                    _target_pb, target_norm, target_ce_cos = initial_metrics(wrapper, x, y, layer, u)
                    variants = [(basis_name, u, target_norm, target_ce_cos)]
                    if args.random_matched > 0:
                        matched_u, matched_norm, matched_ce = choose_matched_random(
                            wrapper, x, y, layer, target_norm, target_ce_cos, u.shape[0], rng, args.random_matched
                        )
                        variants.append((basis_name + "_random_matched", matched_u, matched_norm, matched_ce))
                    if args.random_unmatched:
                        ru = random_unit_feature_dirs(rng, 1, u.shape[0])[0]
                        _g, rn, rc = initial_metrics(wrapper, x, y, layer, ru)
                        variants.append((basis_name + "_random_unmatched", ru, rn, rc))

                    for variant_name, direction, init_norm, init_ce in variants:
                        key = (model, int(row.dataset_idx), variant_name, int(pc_idx))
                        if key in completed:
                            continue
                        adv, metrics = attack_feature_direction(wrapper, x, y, layer, direction, eps, args.steps, step_size)
                        after = eval_state(wrapper, adv, y)
                        h1 = feature_tensor(wrapper, adv, layer).detach().cpu().numpy()[0].astype(np.float32)
                        rows.append(
                            {
                                "model": model,
                                "layer": layer,
                                "dataset_idx": int(row.dataset_idx),
                                "image_ord": int(row.image_ord),
                                "label": int(row.label),
                                "basis": basis_name,
                                "pc": int(pc_idx),
                                "variant": variant_name,
                                "eps_255": args.eps,
                                "steps": args.steps,
                                "success": after["success"],
                                "margin_drop": clean["margin"] - after["margin"],
                                "true_prob_drop": clean["true_prob"] - after["true_prob"],
                                "pullback_norm": float(metrics["pullback_norm"]),
                                "pullback_ce_cos": float(metrics["pullback_ce_cos"]),
                                "matched_initial_norm": float(init_norm),
                                "matched_initial_ce_cos": float(init_ce),
                                "realized_feature_cos": np_cos(h1 - h0, direction),
                            }
                        )
                        completed.add(key)
            if image_i % max(1, args.progress_every) == 0:
                pd.DataFrame(rows).to_csv(out_dir / "matched_intervention_per_image.partial.csv", index=False)
                print(f"[progress] {model} {image_i}/{len(test_images)} rows={len(rows)}", flush=True)
        del wrapper

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "matched_intervention_per_image.csv", index=False)
    pd.DataFrame(sign_rows).to_csv(out_dir / "matched_intervention_sign_selection.csv", index=False)
    summary = df.groupby(["model", "layer", "variant", "basis", "eps_255", "steps"], dropna=False).agg(
        asr=("success", "mean"),
        n=("success", "size"),
        mean_margin_drop=("margin_drop", "mean"),
        median_margin_drop=("margin_drop", "median"),
        mean_true_prob_drop=("true_prob_drop", "mean"),
        mean_pullback_norm=("pullback_norm", "mean"),
        mean_pullback_ce_cos=("pullback_ce_cos", "mean"),
        mean_realized_feature_cos=("realized_feature_cos", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "matched_intervention_summary.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"[done] wrote {out_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/jacobian_null_response/matched_jacobian_intervention_controls")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--models", default="bbb_resnet50")
    p.add_argument("--images", type=int, default=50)
    p.add_argument("--sign-images", type=int, default=20)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--random-matched", type=int, default=8)
    p.add_argument("--random-unmatched", action="store_true")
    p.add_argument("--include-failed", action="store_true")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
