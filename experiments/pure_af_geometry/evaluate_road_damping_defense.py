#!/usr/bin/env python3
"""Forward-pass road damping defense diagnostic.

This exploratory experiment tests whether hidden subspaces that are high
mobility and margin reducing can be damped in the forward pass while preserving
clean behavior.  It is intentionally framed as a diagnostic, not a certified or
state-of-the-art defense.

For CIFAR-10 BlackboxBench ResNet50 we insert damping after pooled layer4:

    h_tilde = h - (1 - gamma) U U^T (h - c)

and evaluate clean accuracy, adaptive PGD, a simple Square-style random search,
and whether the chosen road component is actually scaled by gamma.
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.load_models import load_cifar_model  # noqa: E402


BASE = Path("analysis_outputs/pure_af_geometry/jacobian_null_response")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, y[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y[:, None], -1e9)
    other = masked.max(1).values
    return true - other


def project_linf(x: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x, x0 + eps), x0 - eps).clamp(0, 1)


def orthonormal_rows(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x[:k], dtype=np.float32)
    q, _ = np.linalg.qr(x.T)
    return q[:, : min(k, q.shape[1])].T.astype(np.float32)


def pca_rows(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) < 2:
        raise RuntimeError("Need at least two vectors for PCA.")
    pca = PCA(n_components=min(k, x.shape[0], x.shape[1]), svd_solver="randomized", random_state=0)
    pca.fit(x.astype(np.float32))
    return orthonormal_rows(pca.components_.astype(np.float32), k)


class RoadDamping(nn.Module):
    def __init__(self, basis_rows: np.ndarray, center: np.ndarray, gamma: float):
        super().__init__()
        u = torch.as_tensor(orthonormal_rows(basis_rows, basis_rows.shape[0]).T, dtype=torch.float32)
        c = torch.as_tensor(center.reshape(-1), dtype=torch.float32)
        self.register_buffer("U", u)  # [d, k]
        self.register_buffer("center", c)
        self.gamma = float(gamma)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = h.flatten(1) - self.center[None, :]
        coeff = z @ self.U
        proj = coeff @ self.U.T
        return z.sub((1.0 - self.gamma) * proj).add(self.center[None, :]).view_as(h)


class DampedResNet50(nn.Module):
    """BlackboxBench CIFAR ResNet50 with optional damping after avg-pooled layer4."""

    def __init__(self, seq_model: nn.Sequential, damping: RoadDamping | None):
        super().__init__()
        self.normalize = seq_model[0]
        self.net = seq_model[1]
        self.damping = damping

    def pooled_layer4(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        out = F.relu(self.net.bn1(self.net.conv1(x)))
        out = self.net.layer1(out)
        out = self.net.layer2(out)
        out = self.net.layer3(out)
        out = self.net.layer4(out)
        out = F.avg_pool2d(out, 4)
        return out.view(out.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pooled_layer4(x)
        if self.damping is not None:
            h = self.damping(h)
        return self.net.linear(h)


def load_base_model(device: torch.device) -> nn.Module:
    model = load_cifar_model("bbb_resnet50")
    return model.to(device).eval()


def clean_correct_rows(model: nn.Module, dataset, n_total: int, device: torch.device) -> pd.DataFrame:
    rows = []
    for idx in range(len(dataset)):
        x, y0 = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([int(y0)], device=device)
        with torch.no_grad():
            logits = model(x)
        pred = int(logits.argmax(1).item())
        if pred == int(y0):
            rows.append({"dataset_idx": idx, "label": int(y0), "clean_margin": float(margin(logits, y).item())})
        if len(rows) >= n_total:
            break
    return pd.DataFrame(rows)


def batch_features(model: DampedResNet50, x: torch.Tensor) -> torch.Tensor:
    return model.pooled_layer4(x)


def compute_clean_center(model: DampedResNet50, dataset, rows: pd.DataFrame, device: torch.device, batch_size: int) -> np.ndarray:
    feats = []
    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start : start + batch_size]
        xs = torch.stack([dataset[int(r.dataset_idx)][0] for r in batch.itertuples(index=False)]).to(device)
        with torch.no_grad():
            feats.append(batch_features(model, xs).detach().cpu().numpy())
    return np.concatenate(feats, axis=0).mean(axis=0).astype(np.float32)


def fit_candidate_bases(
    model: DampedResNet50,
    dataset,
    calib_rows: pd.DataFrame,
    center: np.ndarray,
    args,
    device: torch.device,
) -> dict[str, np.ndarray]:
    eps = args.calib_eps / 255.0
    all_vecs, all_mob, all_drop, clean_vecs = [], [], [], []
    gen = torch.Generator(device=device).manual_seed(args.seed + 17)
    for row in calib_rows.itertuples(index=False):
        x0, _ = dataset[int(row.dataset_idx)]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        with torch.no_grad():
            h0 = batch_features(model, x0)
            m0 = margin(model(x0), y)
        remaining = args.calib_dirs
        while remaining > 0:
            b = min(args.batch_size, remaining)
            signs = torch.where(
                torch.rand((b,) + tuple(x0.shape[1:]), generator=gen, device=device) < 0.5,
                -torch.ones((b,) + tuple(x0.shape[1:]), device=device),
                torch.ones((b,) + tuple(x0.shape[1:]), device=device),
            )
            xp = (x0 + eps * signs).clamp(0, 1)
            with torch.no_grad():
                hp = batch_features(model, xp)
                logits = model(xp)
                pred = logits.argmax(1)
                vec = (hp - h0).detach().cpu().numpy()
                drop = (m0 - margin(logits, y)).detach().cpu().numpy()
            all_vecs.append(vec)
            all_mob.append(np.linalg.norm(vec, axis=1))
            all_drop.append(drop)
            keep_clean = (pred.detach().cpu().numpy() == int(row.label))
            if keep_clean.any():
                clean_vecs.append(vec[keep_clean])
            remaining -= b
    vecs = np.concatenate(all_vecs, axis=0).astype(np.float32)
    mob = np.concatenate(all_mob, axis=0)
    drop = np.concatenate(all_drop, axis=0)
    score = (mob - mob.mean()) / (mob.std() + 1e-8) + (np.maximum(drop, 0) - np.maximum(drop, 0).mean()) / (
        np.maximum(drop, 0).std() + 1e-8
    )
    n_select = min(args.basis_select, len(vecs))
    adv_idx = np.argsort(score)[-n_select:]
    mob_idx = np.argsort(mob)[-n_select:]
    bases = {
        "adv_road": pca_rows(vecs[adv_idx] - center[None, :] * 0.0, args.k),
        "mobility_only": pca_rows(vecs[mob_idx], args.k),
    }
    if clean_vecs:
        clean = np.concatenate(clean_vecs, axis=0)
        bases["clean_motion"] = pca_rows(clean[: max(n_select, args.k + 1)], args.k)
    return bases


def load_saved_bases(args) -> dict[str, np.ndarray]:
    out = {}
    bpath = Path(args.basis_dir) / "basis_vectors.npz"
    if bpath.exists():
        z = np.load(bpath)
        if "jvp_sketch_basis" in z.files:
            out["jvp_sketch"] = orthonormal_rows(z["jvp_sketch_basis"], args.k)
        if "transport_basis" in z.files:
            out["transport_pc"] = orthonormal_rows(z["transport_basis"], args.k)
    spath = Path(args.segment_dir)
    if (spath / "segment_metadata.csv").exists() and (spath / "segment_vectors.npz").exists():
        meta = pd.read_csv(spath / "segment_metadata.csv")
        arr = np.load(spath / "segment_vectors.npz")
        key = "bbb_resnet50__pgd__layer4"
        chunks = []
        for source in ["pgd", "square"]:
            key = f"bbb_resnet50__{source}__layer4"
            if key not in arr.files:
                continue
            sub = meta[(meta.model == "bbb_resnet50") & (meta.source == source) & (meta.layer == "layer4")]
            sub = sub[sub.final_success.astype(int) == 0]
            if not sub.empty:
                chunks.append(arr[key][sub.vector_idx.to_numpy(dtype=int)])
        if chunks:
            out["failed_attack"] = pca_rows(np.concatenate(chunks, axis=0), args.k)
    rng = np.random.default_rng(args.seed)
    d = 2048
    out["random"] = orthonormal_rows(rng.normal(size=(args.k, d)).astype(np.float32), args.k)
    return out


def eval_clean(model: nn.Module, dataset, rows: pd.DataFrame, device: torch.device, batch_size: int) -> dict:
    total = ok = 0
    margins = []
    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start : start + batch_size]
        xs = torch.stack([dataset[int(r.dataset_idx)][0] for r in batch.itertuples(index=False)]).to(device)
        ys = torch.tensor([int(r.label) for r in batch.itertuples(index=False)], device=device)
        with torch.no_grad():
            logits = model(xs)
        ok += int((logits.argmax(1) == ys).sum().item())
        total += len(batch)
        margins.extend(margin(logits, ys).detach().cpu().numpy().tolist())
    return {"clean_acc": ok / max(total, 1), "clean_margin_mean": float(np.mean(margins))}


def pgd_asr(model: nn.Module, dataset, rows: pd.DataFrame, args, device: torch.device, steps: int) -> dict:
    eps = args.eps / 255.0
    alpha = args.pgd_step / 255.0
    succ = total = 0
    margins = []
    for row in rows.itertuples(index=False):
        torch.manual_seed(args.seed + int(row.dataset_idx) * 1009 + steps)
        x0, _ = dataset[int(row.dataset_idx)]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
        for _ in range(steps):
            x.requires_grad_(True)
            loss = F.cross_entropy(model(x), y)
            grad = torch.autograd.grad(loss, x)[0]
            x = project_linf(x.detach() + alpha * grad.detach().sign(), x0, eps)
        with torch.no_grad():
            logits = model(x)
        succ += int(logits.argmax(1).item() != int(row.label))
        total += 1
        margins.append(float(margin(logits, y).item()))
    return {f"pgd{steps}_asr": succ / max(total, 1), f"pgd{steps}_robust_acc": 1 - succ / max(total, 1), f"pgd{steps}_margin_mean": float(np.mean(margins))}


def square_asr(model: nn.Module, dataset, rows: pd.DataFrame, args, device: torch.device) -> dict:
    if args.square_queries <= 0:
        return {"square_asr": np.nan, "square_robust_acc": np.nan}
    eps = args.eps / 255.0
    rng = np.random.default_rng(args.seed + 91)
    succ = total = 0
    for row in rows.itertuples(index=False):
        x0, _ = dataset[int(row.dataset_idx)]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
        with torch.no_grad():
            best_loss = float(F.cross_entropy(model(x), y).item())
        success = False
        for q in range(args.square_queries):
            size = max(1, int(round(32 * (1 - q / max(args.square_queries, 1)) ** 0.5)))
            i = int(rng.integers(0, 33 - size))
            j = int(rng.integers(0, 33 - size))
            cand = x.clone()
            sign = -1.0 if rng.random() < 0.5 else 1.0
            cand[:, :, i : i + size, j : j + size] = x0[:, :, i : i + size, j : j + size] + sign * eps
            cand = project_linf(cand, x0, eps)
            with torch.no_grad():
                logits = model(cand)
                loss = float(F.cross_entropy(logits, y).item())
                pred = int(logits.argmax(1).item())
            if pred != int(row.label):
                success = True
                break
            if loss > best_loss:
                x = cand
                best_loss = loss
        succ += int(success)
        total += 1
    return {"square_asr": succ / max(total, 1), "square_robust_acc": 1 - succ / max(total, 1)}


def mobility_scaling(original: DampedResNet50, damped: DampedResNet50, dataset, rows: pd.DataFrame, basis: np.ndarray, args, device: torch.device) -> dict:
    eps = args.scale_probe_eps / 255.0
    u = torch.as_tensor(orthonormal_rows(basis, args.k).T, dtype=torch.float32, device=device)
    road_ratios, orth_ratios = [], []
    gen = torch.Generator(device=device).manual_seed(args.seed + 701)
    for row in rows.head(args.scale_images).itertuples(index=False):
        x0, _ = dataset[int(row.dataset_idx)]
        x0 = x0.unsqueeze(0).to(device)
        signs = torch.where(
            torch.rand((args.scale_dirs,) + tuple(x0.shape[1:]), generator=gen, device=device) < 0.5,
            -torch.ones((args.scale_dirs,) + tuple(x0.shape[1:]), device=device),
            torch.ones((args.scale_dirs,) + tuple(x0.shape[1:]), device=device),
        )
        xp = (x0 + eps * signs).clamp(0, 1)
        with torch.no_grad():
            h0 = original.pooled_layer4(x0).repeat(args.scale_dirs, 1)
            h1 = original.pooled_layer4(xp)
            d0 = h1 - h0
            hd0 = damped.pooled_layer4(x0).repeat(args.scale_dirs, 1)
            # Need post-damping feature, not raw pooled feature.
            hd1_raw = damped.pooled_layer4(xp)
            hd1 = damped.damping(hd1_raw) if damped.damping is not None else hd1_raw
            hd0 = damped.damping(hd0) if damped.damping is not None else hd0
            dd = hd1 - hd0
            r0 = d0 @ u
            r1 = dd @ u
            p0 = r0 @ u.T
            p1 = r1 @ u.T
            road_ratios.extend((r1.norm(dim=1) / r0.norm(dim=1).clamp_min(1e-12)).cpu().numpy().tolist())
            orth_ratios.extend(((dd - p1).norm(dim=1) / (d0 - p0).norm(dim=1).clamp_min(1e-12)).cpu().numpy().tolist())
    return {"road_ratio_mean": float(np.mean(road_ratios)), "orth_ratio_mean": float(np.mean(orth_ratios))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/road_damping_defense_resnet50_pilot")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--basis-dir", default=str(BASE / "jacobian_basis_residual_bbb_resnet50_c800_with_jvp_meta"))
    p.add_argument("--segment-dir", default=str(BASE / "balanced_full_bbb_resnet50_c800_selector_material"))
    p.add_argument("--images", type=int, default=80)
    p.add_argument("--calib-images", type=int, default=80)
    p.add_argument("--calib-dirs", type=int, default=128)
    p.add_argument("--basis-select", type=int, default=2000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--basis-names", default="", help="Optional comma-separated subset of basis names to evaluate.")
    p.add_argument("--gammas", default="1.0,0.75,0.5,0.25,0.1,0.0")
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--calib-eps", type=float, default=8.0)
    p.add_argument("--pgd-steps", default="20,100")
    p.add_argument("--pgd-step", type=float, default=2.0)
    p.add_argument("--square-queries", type=int, default=300)
    p.add_argument("--scale-probe-eps", type=float, default=1.0)
    p.add_argument("--scale-images", type=int, default=20)
    p.add_argument("--scale-dirs", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    seq = load_base_model(device)
    original = DampedResNet50(seq, None).to(device).eval()
    rows = clean_correct_rows(original, dataset, args.images + args.calib_images + 20, device)
    calib = rows.head(args.calib_images).reset_index(drop=True)
    eval_rows = rows.iloc[args.calib_images : args.calib_images + args.images].reset_index(drop=True)
    center = compute_clean_center(original, dataset, calib, device, args.batch_size)

    bases = fit_candidate_bases(original, dataset, calib, center, args, device)
    bases.update(load_saved_bases(args))
    if args.basis_names.strip():
        keep = {x.strip() for x in args.basis_names.split(",") if x.strip()}
        missing = sorted(keep.difference(bases))
        if missing:
            raise RuntimeError(f"Requested basis names not available: {missing}; available={sorted(bases)}")
        bases = {k: v for k, v in bases.items() if k in keep}
    np.savez(out / "road_damping_bases.npz", center=center, **{k: v for k, v in bases.items()})

    summary_rows = []
    scale_rows = []
    gammas = [float(x) for x in args.gammas.split(",") if x.strip()]
    pgd_steps = [int(x) for x in args.pgd_steps.split(",") if x.strip()]
    for basis_name, basis in bases.items():
        for gamma in gammas:
            damping = None if gamma == 1.0 else RoadDamping(basis[: args.k], center, gamma).to(device)
            model = DampedResNet50(seq, damping).to(device).eval()
            row = {"basis": basis_name, "gamma": gamma, "k": args.k, "n_eval": len(eval_rows)}
            row.update(eval_clean(model, dataset, eval_rows, device, args.batch_size))
            for steps in pgd_steps:
                row.update(pgd_asr(model, dataset, eval_rows, args, device, steps))
            row.update(square_asr(model, dataset, eval_rows, args, device))
            summary_rows.append(row)
            if gamma != 1.0:
                s = {"basis": basis_name, "gamma": gamma}
                s.update(mobility_scaling(original, model, dataset, eval_rows, basis[: args.k], args, device))
                scale_rows.append(s)
            pd.DataFrame(summary_rows).to_csv(out / "road_damping_eval_summary.partial.csv", index=False)
            pd.DataFrame(scale_rows).to_csv(out / "road_damping_mobility_scaling.partial.csv", index=False)
            print(pd.DataFrame([row]).to_string(index=False), flush=True)

    pd.DataFrame(summary_rows).to_csv(out / "road_damping_eval_summary.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(out / "road_damping_mobility_scaling.csv", index=False)
    meta = vars(args) | {"device": str(device), "n_calib": int(len(calib)), "n_eval": int(len(eval_rows)), "bases": sorted(bases)}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
