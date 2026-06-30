#!/usr/bin/env python3
"""Square attack with representation-trajectory correction proposals.

This pilot asks whether successful PGD transport can act as a correction prior
for a search-based attack.  It learns a PGD hidden-transport basis on one split,
then evaluates Square-like attacks on a held-out split.  Correction proposals
are accepted only if they improve the same untargeted margin objective and are
counted against the query budget.

Important: correction directions use source-model hidden-feature gradients, so
this is a proof-of-concept / surrogate-prior experiment, not a target-only
black-box attack.
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
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.square import p_selection  # noqa: E402
from experiments.pure_af_geometry.evaluate_road_damping_defense import margin, project_linf  # noqa: E402
from experiments.pure_af_geometry.trace_jacobian_singular_roads import feat, load_model, set_seed  # noqa: E402


def hidden(model, x: torch.Tensor) -> torch.Tensor:
    return feat(model, x)


def select_clean_correct(dataset, model, n: int, device):
    rows = []
    for idx in range(len(dataset)):
        x, y0 = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([int(y0)], device=device)
        with torch.no_grad():
            pred = int(model(x).argmax(1).item())
        if pred == int(y0):
            rows.append((idx, int(y0)))
        if len(rows) >= n:
            break
    return rows


def eval_attack_state(model, x, y):
    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(1).item())
        m = float(margin(logits, y).item())
        loss = float(F.cross_entropy(logits, y).item())
    return pred, m, loss


def pgd_states(model, x0, y, eps: float, step_size: float, steps: int):
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
    states = [x0.detach(), x.detach()]
    for _ in range(steps):
        probe = x.detach().requires_grad_(True)
        loss = F.cross_entropy(model(probe), y)
        grad = torch.autograd.grad(loss, probe)[0]
        x = project_linf(x.detach() + step_size * grad.sign(), x0, eps).detach()
        states.append(x)
    return states


def fit_pgd_transport_basis(model, dataset, rows, eps: float, step_size: float, steps: int, k: int, device):
    segs = []
    meta = []
    for image_id, label in rows:
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        states = pgd_states(model, x0, y, eps, step_size, steps)
        pred, _m, _loss = eval_attack_state(model, states[-1], y)
        success = pred != label
        meta.append({"image_id": image_id, "label": label, "success": int(success)})
        if not success:
            continue
        with torch.no_grad():
            hs = [hidden(model, x).detach().cpu().numpy()[0].astype(np.float32) for x in states]
        for a, b in zip(hs[:-1], hs[1:]):
            v = b - a
            n = np.linalg.norm(v)
            if n > 1e-8:
                segs.append(v / n)
    if len(segs) < max(4, k):
        raise RuntimeError(f"Too few successful PGD segments: {len(segs)}")
    X = np.stack(segs).astype(np.float64)
    mu = X.mean(axis=0)
    mu = mu / max(np.linalg.norm(mu), 1e-12)
    Xc = X - X.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    basis = vt[: min(k, len(vt))].astype(np.float32)
    var = (s**2) / max(float((s**2).sum()), 1e-12)
    return {
        "mu": torch.tensor(mu.astype(np.float32), device=device).view(1, -1),
        "basis": torch.tensor(basis, device=device),
        "n_segments": len(segs),
        "pc_var": var[: min(k, len(var))].astype(float).tolist(),
        "meta": pd.DataFrame(meta),
    }


def correction_candidate(model, x, x0, y, prior, mode: str, eps: float, step_size: float, rng: torch.Generator):
    probe = x.detach().requires_grad_(True)
    h = hidden(model, probe)
    if mode == "pgd_mean":
        score = (h * prior["mu"]).sum()
    elif mode == "pgd_energy":
        dh = h - hidden(model, x0).detach()
        coeff = dh @ prior["basis"].T
        score = (coeff**2).sum()
    elif mode == "random_feature":
        r = torch.randn(h.shape, generator=rng, device=h.device)
        r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)
        score = (h * r).sum()
    elif mode == "ce_gradient":
        # Diagnostic upper/practical ceiling: this is just a CE-gradient proposal.
        score = F.cross_entropy(model(probe), y)
    else:
        raise ValueError(mode)
    grad = torch.autograd.grad(score, probe)[0]
    return project_linf(x.detach() + step_size * grad.sign(), x0, eps).detach()


def square_candidate(model, x_best, x0, eps: float, query_idx: int, query_budget: int, p_init: float, rng_np):
    c, h, w = x0.shape[1:]
    perturbation = (x_best - x0).clone()
    p = p_selection(p_init, query_idx, query_budget)
    s = int(round(np.sqrt(p * c * h * w / c)))
    s = min(max(s, 1), h - 1)
    top = int(rng_np.integers(0, h - s + 1))
    left = int(rng_np.integers(0, w - s + 1))
    sign = -1.0 if rng_np.random() < 0.5 else 1.0
    perturbation[:, :, top : top + s, left : left + s] = sign * eps
    return project_linf(x0 + perturbation, x0, eps).detach()


def run_square_corrected(model, x0, y, prior, args, mode: str, seed: int):
    rng_np = np.random.default_rng(seed)
    rng_t = torch.Generator(device=x0.device).manual_seed(seed + 100003)
    eps = args.eps / 255.0
    square_queries = 0
    correction_queries = 0
    x_best = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps).detach()
    pred, best_margin, best_loss = eval_attack_state(model, x_best, y)
    queries = 1
    success_step = 0 if pred != int(y.item()) else -1
    curve = []
    h0 = hidden(model, x0).detach()
    for q in range(queries, args.query_budget + 1):
        use_corr = mode != "vanilla_square" and (q % args.correction_every == 0)
        if use_corr:
            cand = correction_candidate(model, x_best, x0, y, prior, mode, eps, args.correction_step / 255.0, rng_t)
            correction_queries += 1
            proposal = f"corr_{mode}"
        else:
            cand = square_candidate(model, x_best, x0, eps, q + args.square_init, args.query_budget, args.p_init, rng_np)
            square_queries += 1
            proposal = "square"
        pred_c, m_c, loss_c = eval_attack_state(model, cand, y)
        accepted = int(m_c < best_margin)
        if accepted:
            x_best = cand
            best_margin = m_c
            best_loss = loss_c
            pred = pred_c
        with torch.no_grad():
            dh = hidden(model, x_best) - h0
            coeff = dh @ prior["basis"].T
            pgd_energy = float((coeff**2).sum().item() / dh.norm(dim=1).pow(2).clamp_min(1e-12).item())
            mu_align = float((dh * prior["mu"]).sum().item() / dh.norm(dim=1).clamp_min(1e-12).item())
        succ = pred != int(y.item())
        if succ and success_step < 0:
            success_step = q
        curve.append(
            {
                "query": q,
                "proposal": proposal,
                "accepted": accepted,
                "margin": best_margin,
                "loss": best_loss,
                "pred": pred,
                "success": int(succ),
                "pgd_basis_energy": pgd_energy,
                "pgd_mean_alignment": mu_align,
                "square_queries": square_queries,
                "correction_queries": correction_queries,
            }
        )
        if succ and args.early_stop:
            break
    return {
        "success": int(success_step >= 0),
        "success_query": success_step,
        "final_margin": best_margin,
        "final_pred": pred,
        "square_queries": square_queries,
        "correction_queries": correction_queries,
        "curve": curve,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("method")
        .agg(
            n=("image_id", "count"),
            asr=("success", "mean"),
            mean_success_query=("success_query", lambda x: np.mean([v for v in x if v >= 0]) if any(v >= 0 for v in x) else np.nan),
            median_success_query=("success_query", lambda x: np.median([v for v in x if v >= 0]) if any(v >= 0 for v in x) else np.nan),
            mean_final_margin=("final_margin", "mean"),
            mean_square_queries=("square_queries", "mean"),
            mean_correction_queries=("correction_queries", "mean"),
            mean_final_pgd_energy=("final_pgd_basis_energy", "mean"),
            mean_final_mu_align=("final_pgd_mean_alignment", "mean"),
        )
        .reset_index()
    )


def make_plot(summary: pd.DataFrame, out: Path):
    order = ["vanilla_square", "random_feature", "pgd_energy", "pgd_mean", "ce_gradient"]
    d = summary.set_index("method").reindex([m for m in order if m in set(summary.method)]).reset_index()
    labels = {
        "vanilla_square": "Square",
        "random_feature": "Square + random feature",
        "pgd_energy": "Square + PGD-basis energy",
        "pgd_mean": "Square + PGD-mean direction",
        "ce_gradient": "Square + CE grad ceiling",
    }
    x = np.arange(len(d))
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.4), constrained_layout=True)
    axes[0].bar(x, d.asr, color="#4C78A8")
    axes[0].set_ylabel("ASR")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(x, d.mean_success_query, color="#54A24B")
    axes[1].set_ylabel("mean query to success")
    axes[2].bar(x, d.mean_final_margin, color="#F58518")
    axes[2].axhline(0, color="black", lw=1, ls="--")
    axes[2].set_ylabel("mean final margin")
    axes[3].bar(x, d.mean_final_pgd_energy, color="#72B7B2")
    axes[3].set_ylabel("PGD-basis energy")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(m, m) for m in d.method], rotation=25, ha="right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "square_trajectory_correction_summary.png", dpi=220)
    fig.savefig(out / "square_trajectory_correction_summary.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/square_trajectory_correction_pilot")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--train-images", type=int, default=80)
    p.add_argument("--test-images", type=int, default=60)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--pgd-step-size", type=float, default=1.0)
    p.add_argument("--basis-k", type=int, default=10)
    p.add_argument("--query-budget", type=int, default=100)
    p.add_argument("--correction-every", type=int, default=5)
    p.add_argument("--correction-step", type=float, default=1.0)
    p.add_argument("--p-init", type=float, default=0.3)
    p.add_argument("--square-init", type=int, default=0)
    p.add_argument("--methods", default="vanilla_square,random_feature,pgd_energy,pgd_mean,ce_gradient")
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    selected = select_clean_correct(dataset, model, args.train_images + args.test_images, device)
    train_rows = selected[: args.train_images]
    test_rows = selected[args.train_images :]
    prior = fit_pgd_transport_basis(
        model,
        dataset,
        train_rows,
        args.eps / 255.0,
        args.pgd_step_size / 255.0,
        args.pgd_steps,
        args.basis_k,
        device,
    )
    prior["meta"].to_csv(out / "pgd_basis_train_meta.csv", index=False)
    pd.DataFrame(
        [{"n_segments": prior["n_segments"], "basis_k": int(prior["basis"].shape[0]), "pc_var": json.dumps(prior["pc_var"])}]
    ).to_csv(out / "pgd_basis_summary.csv", index=False)

    methods = [m for m in args.methods.split(",") if m]
    rows = []
    curves = []
    for image_id, label in test_rows:
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        for method in methods:
            res = run_square_corrected(model, x0, y, prior, args, method, args.seed + image_id * 917 + len(method))
            with torch.no_grad():
                h0 = hidden(model, x0).detach()
                # Reconstruct final energy from last curve if present.
                if res["curve"]:
                    final_energy = res["curve"][-1]["pgd_basis_energy"]
                    final_align = res["curve"][-1]["pgd_mean_alignment"]
                else:
                    final_energy = np.nan
                    final_align = np.nan
            rows.append(
                {
                    "image_id": image_id,
                    "label": label,
                    "method": method,
                    "success": res["success"],
                    "success_query": res["success_query"],
                    "final_margin": res["final_margin"],
                    "final_pred": res["final_pred"],
                    "square_queries": res["square_queries"],
                    "correction_queries": res["correction_queries"],
                    "final_pgd_basis_energy": final_energy,
                    "final_pgd_mean_alignment": final_align,
                }
            )
            for c in res["curve"]:
                c.update({"image_id": image_id, "label": label, "method": method})
                curves.append(c)
    df = pd.DataFrame(rows)
    curve_df = pd.DataFrame(curves)
    df.to_csv(out / "square_trajectory_correction_results.csv", index=False)
    curve_df.to_csv(out / "square_trajectory_correction_curves.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "square_trajectory_correction_summary.csv", index=False)
    make_plot(summary, out)
    meta = vars(args)
    meta.update({"n_train": len(train_rows), "n_test": len(test_rows), "n_pgd_segments": prior["n_segments"]})
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
