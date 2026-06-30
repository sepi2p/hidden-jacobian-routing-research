#!/usr/bin/env python3
"""Trajectory-level road detector pilot.

Static road anomaly failed under margin matching.  This script tests the
transition-level hypothesis: adversarial roads are directions of motion, so
attack progress should be detected from hidden steps

    dh_t = h_l(x_t) - h_l(x_{t-1})

using road energy, road magnitude, and true-class margin degradation.
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
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.evaluate_road_damping_defense import (  # noqa: E402
    DampedResNet50,
    fit_candidate_bases,
    margin,
    project_linf,
)
from utils.load_models import load_cifar_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(device: torch.device) -> DampedResNet50:
    seq = load_cifar_model("bbb_resnet50").to(device).eval()
    return DampedResNet50(seq, None).to(device).eval()


def feature(model: DampedResNet50, x: torch.Tensor) -> torch.Tensor:
    return model.pooled_layer4(x).flatten(1)


def orthonormal_rows(x: np.ndarray, k: int) -> np.ndarray:
    q, _ = np.linalg.qr(x[:k].T)
    return q[:, : min(k, q.shape[1])].T.astype(np.float32)


def select_clean_correct(dataset, model, n: int, device: torch.device) -> pd.DataFrame:
    rows = []
    for idx in range(len(dataset)):
        x, y0 = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([int(y0)], device=device)
        with torch.no_grad():
            logits = model(x)
        if int(logits.argmax(1).item()) == int(y0):
            rows.append({"dataset_idx": idx, "label": int(y0), "clean_margin": float(margin(logits, y).item())})
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)


def get_margin_success(model, x: torch.Tensor, y: torch.Tensor) -> tuple[float, int]:
    with torch.no_grad():
        logits = model(x)
    return float(margin(logits, y).item()), int(logits.argmax(1).item() != int(y.item()))


def decompose(dh: torch.Tensor, u: torch.Tensor) -> tuple[float, float, float]:
    z = dh.flatten(1)
    coeff = z @ u
    proj = coeff @ u.T
    total = float(z.norm(dim=1).item())
    road = float(proj.norm(dim=1).item())
    orth = float((z - proj).norm(dim=1).item())
    energy = road * road / max(total * total, 1e-12)
    return energy, road, orth


def add_transition(rows: list[dict], model, x_prev, x_next, y, u, kind, image_id, label, step, accepted=1):
    with torch.no_grad():
        h_prev = feature(model, x_prev)
        h_next = feature(model, x_next)
    e, r, o = decompose(h_next - h_prev, u)
    m_prev, s_prev = get_margin_success(model, x_prev, y)
    m_next, s_next = get_margin_success(model, x_next, y)
    drop = max(m_prev - m_next, 0.0)
    rows.append(
        {
            "kind": kind,
            "image_id": int(image_id),
            "label": int(label),
            "step": int(step),
            "accepted": int(accepted),
            "current_success": int(s_next),
            "margin_prev": m_prev,
            "margin_next": m_next,
            "margin_drop_pos": drop,
            "road_energy": e,
            "road_mag": r,
            "orth_mag": o,
            "step_mag": float((h_next - h_prev).flatten(1).norm(dim=1).item()),
            "score_E": e,
            "score_R": r,
            "score_D": drop,
            "score_ED": e * drop,
            "score_ERD": e * r * drop,
            "score_RD": r * drop,
        }
    )


def run_pgd(model, x0, y, u, args, image_id, label):
    rows = []
    eps = args.eps / 255.0
    alpha = args.pgd_step / 255.0
    torch.manual_seed(args.seed + image_id * 101)
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
    for t in range(1, args.steps + 1):
        x.requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        grad = torch.autograd.grad(loss, x)[0]
        x_next = project_linf(x.detach() + alpha * grad.detach().sign(), x0, eps)
        add_transition(rows, model, x, x_next, y, u, "attack_pgd", image_id, label, t)
        x = x_next
    return rows


def run_square(model, x0, y, u, args, image_id, label):
    rows = []
    eps = args.eps / 255.0
    rng = np.random.default_rng(args.seed + image_id * 103)
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
    with torch.no_grad():
        best_loss = float(F.cross_entropy(model(x), y).item())
    for q in range(1, args.square_queries + 1):
        size = max(1, int(round(32 * (1 - (q - 1) / max(args.square_queries, 1)) ** 0.5)))
        i = int(rng.integers(0, 33 - size))
        j = int(rng.integers(0, 33 - size))
        cand = x.clone()
        sign = -1.0 if rng.random() < 0.5 else 1.0
        cand[:, :, i : i + size, j : j + size] = x0[:, :, i : i + size, j : j + size] + sign * eps
        cand = project_linf(cand, x0, eps)
        with torch.no_grad():
            logits = model(cand)
            loss = float(F.cross_entropy(logits, y).item())
            succ = int(logits.argmax(1).item() != int(y.item()))
        if loss > best_loss or succ:
            add_transition(rows, model, x, cand, y, u, "attack_square", image_id, label, q)
            x = cand
            best_loss = max(best_loss, loss)
        if succ:
            break
    return rows


def run_random_walk(model, x0, y, u, args, image_id, label, kind: str, mobility_basis: np.ndarray | None = None):
    rows = []
    eps = args.eps / 255.0
    rng = torch.Generator(device=x0.device).manual_seed(args.seed + image_id * 107 + (1 if kind == "benign_random" else 2))
    x = x0.clone()
    if mobility_basis is not None:
        # Pull back a random mobility/road feature direction, objective-neutral.
        basis_t = torch.as_tensor(mobility_basis.T, dtype=torch.float32, device=x0.device)
    for t in range(1, args.steps + 1):
        if mobility_basis is None:
            direction = torch.where(torch.rand(x.shape, generator=rng, device=x.device) < 0.5, -torch.ones_like(x), torch.ones_like(x))
        else:
            coeff = torch.randn((basis_t.shape[1],), generator=rng, device=x.device)
            target = basis_t @ coeff
            x_req = x.detach().requires_grad_(True)
            h = feature(model, x_req).flatten(1)
            objective = (h * target[None, :]).sum()
            grad = torch.autograd.grad(objective, x_req)[0]
            direction = grad.sign()
        x_next = project_linf(x.detach() + (args.pgd_step / 255.0) * direction, x0, eps)
        add_transition(rows, model, x, x_next, y, u, kind, image_id, label, t)
        x = x_next
    return rows


def run_benign_aug_sequence(model, x0, y, u, args, image_id, label):
    rows = []
    rng = torch.Generator(device=x0.device).manual_seed(args.seed + image_id * 109)
    x = x0.clone()
    for t in range(1, args.steps + 1):
        noise = torch.randn(x.shape, generator=rng, device=x.device) * (1.0 / 255.0)
        scale = 0.98 + 0.04 * torch.rand((1, 1, 1, 1), generator=rng, device=x.device)
        x_next = (x * scale + noise).clamp(0, 1)
        # Keep benign sequence class-preserving; otherwise skip transition.
        with torch.no_grad():
            if int(model(x_next).argmax(1).item()) != int(label):
                continue
        add_transition(rows, model, x, x_next, y, u, "benign_aug", image_id, label, t)
        x = x_next
    return rows


def metric_summary(df: pd.DataFrame, scores: list[str]) -> pd.DataFrame:
    controls = df[df.kind.str.startswith("benign")]
    rows = []
    for attack in ["attack_pgd", "attack_square"]:
        atk = df[df.kind == attack]
        if atk.empty:
            continue
        sub = pd.concat([controls, atk], ignore_index=True)
        y = sub.kind.eq(attack).astype(int).to_numpy()
        for s in scores:
            vals = sub[s].to_numpy(float)
            if len(np.unique(y)) < 2:
                auc = ap = np.nan
                tpr1 = tpr5 = np.nan
            else:
                auc = float(roc_auc_score(y, vals))
                ap = float(average_precision_score(y, vals))
                fpr, tpr, _ = roc_curve(y, vals)
                tpr1 = float(tpr[fpr <= 0.01].max()) if (fpr <= 0.01).any() else 0.0
                tpr5 = float(tpr[fpr <= 0.05].max()) if (fpr <= 0.05).any() else 0.0
            rows.append({"score": s, "attack": attack, "AUROC": auc, "AUPRC": ap, "TPR@1FPR": tpr1, "TPR@5FPR": tpr5})
    return pd.DataFrame(rows)


def window_summary(df: pd.DataFrame, scores: list[str], window: int) -> pd.DataFrame:
    frames = []
    for (kind, image_id), g in df.sort_values("step").groupby(["kind", "image_id"]):
        gg = g.copy()
        for s in scores:
            gg[f"win_{s}"] = gg[s].rolling(window=window, min_periods=1).sum()
        frames.append(gg)
    wdf = pd.concat(frames, ignore_index=True)
    return metric_summary(wdf, [f"win_{s}" for s in scores])


def flag_before_success(df: pd.DataFrame, score: str, fpr: float) -> pd.DataFrame:
    controls = df[df.kind.str.startswith("benign")]
    thresh = float(np.quantile(controls[score].dropna(), 1 - fpr))
    rows = []
    for attack in ["attack_pgd", "attack_square"]:
        for image_id, g in df[df.kind == attack].sort_values("step").groupby("image_id"):
            hit = g[g[score] > thresh]
            succ = g[g.current_success.astype(int) == 1]
            first_hit = int(hit.step.min()) if not hit.empty else -1
            first_succ = int(succ.step.min()) if not succ.empty else -1
            rows.append(
                {
                    "attack": attack,
                    "image_id": int(image_id),
                    "score": score,
                    "fpr": fpr,
                    "first_flag_step": first_hit,
                    "first_success_step": first_succ,
                    "flag_before_success": int(first_hit > 0 and (first_succ < 0 or first_hit <= first_succ)),
                }
            )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
    for ax, s in zip(axes, ["score_E", "score_D", "score_ERD"]):
        for kind in ["attack_pgd", "attack_square", "benign_random", "benign_mobility", "benign_aug"]:
            sub = df[df.kind == kind]
            if len(sub):
                ax.hist(sub[s], bins=50, alpha=0.4, density=True, label=kind)
        ax.set_title(s)
        ax.legend(fontsize=7)
    fig.savefig(out / "trajectory_score_histograms.png", dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/trajectory_road_detector_resnet50_quick")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--n-calib", type=int, default=300)
    p.add_argument("--n-eval", type=int, default=80)
    p.add_argument("--basis-k", type=int, default=20)
    p.add_argument("--calib-dirs", type=int, default=64)
    p.add_argument("--basis-select", type=int, default=2000)
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--pgd-step", type=float, default=2.0)
    p.add_argument("--square-queries", type=int, default=100)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    train = datasets.CIFAR10(args.dataset_root, train=True, download=False, transform=transforms.ToTensor())
    test = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    calib = select_clean_correct(train, model, args.n_calib, device)
    eval_rows = select_clean_correct(test, model, args.n_eval, device)
    calib.to_csv(out / "calib_clean_correct.csv", index=False)
    eval_rows.to_csv(out / "eval_clean_correct.csv", index=False)

    road_args = argparse.Namespace(
        calib_eps=args.eps,
        calib_dirs=args.calib_dirs,
        batch_size=128,
        seed=args.seed,
        basis_select=args.basis_select,
        k=args.basis_k,
    )
    bases = fit_candidate_bases(model, train, calib, np.zeros(2048, dtype=np.float32), road_args, device)
    adv_basis = orthonormal_rows(bases["adv_road"], args.basis_k)
    mobility_basis = orthonormal_rows(bases["mobility_only"], args.basis_k)
    np.savez(out / "trajectory_detector_bases.npz", adv_road=adv_basis, mobility_only=mobility_basis)
    u = torch.as_tensor(adv_basis.T, dtype=torch.float32, device=device)

    rows = []
    for row in eval_rows.itertuples(index=False):
        x0, _ = test[int(row.dataset_idx)]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([int(row.label)], device=device)
        rows += run_pgd(model, x0, y, u, args, int(row.dataset_idx), int(row.label))
        rows += run_square(model, x0, y, u, args, int(row.dataset_idx), int(row.label))
        rows += run_random_walk(model, x0, y, u, args, int(row.dataset_idx), int(row.label), "benign_random")
        rows += run_random_walk(model, x0, y, u, args, int(row.dataset_idx), int(row.label), "benign_mobility", mobility_basis)
        rows += run_benign_aug_sequence(model, x0, y, u, args, int(row.dataset_idx), int(row.label))

    df = pd.DataFrame(rows)
    df.to_csv(out / "trajectory_transition_scores.csv", index=False)
    scores = ["score_E", "score_R", "score_D", "score_ED", "score_RD", "score_ERD"]
    summ = metric_summary(df, scores)
    summ.to_csv(out / "summary_transition_detection.csv", index=False)
    win = window_summary(df, scores, args.window)
    win.to_csv(out / "summary_window_detection.csv", index=False)
    flags = pd.concat([flag_before_success(df, "score_ERD", 0.01), flag_before_success(df, "score_ERD", 0.05)], ignore_index=True)
    flags.to_csv(out / "summary_flag_before_success.csv", index=False)
    plot(df, out)
    (out / "metadata.json").write_text(json.dumps(vars(args) | {"device": str(device)}, indent=2))

    print("Transition detection:", flush=True)
    print(summ.sort_values(["attack", "AUROC"], ascending=[True, False]).to_string(index=False), flush=True)
    print("\nWindow detection:", flush=True)
    print(win.sort_values(["attack", "AUROC"], ascending=[True, False]).to_string(index=False), flush=True)
    print("\nFlag before success:", flush=True)
    print(flags.groupby(["attack", "fpr"]).flag_before_success.mean().reset_index().to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
