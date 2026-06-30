#!/usr/bin/env python3
"""Learned trajectory-correction policy for Square-like attacks.

The policy observes the current Square state in a PGD-transport coordinate
system and predicts a feature-space correction direction.  The training target
is the local untargeted CE/PGD feature step from the same state.

This is a proof-of-concept: correction proposals use source-model feature
gradients to pull the predicted hidden direction back to input space.
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
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.square import p_selection  # noqa: E402
from experiments.hidden_jacobian_routing.common import margin, project_linf  # noqa: E402
from experiments.hidden_jacobian_routing.trace_jacobian_singular_roads import feat, load_model, set_seed  # noqa: E402


def hidden(model, x):
    return feat(model, x)


def select_clean_correct(dataset, model, n: int, device):
    rows = []
    for idx in range(len(dataset)):
        x, y0 = dataset[idx]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(model(x).argmax(1).item())
        if pred == int(y0):
            rows.append((idx, int(y0)))
        if len(rows) >= n:
            break
    return rows


def eval_state(model, x, y):
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = int(logits.argmax(1).item())
        m = float(margin(logits, y).item())
        ce = float(F.cross_entropy(logits, y).item())
        py = float(probs[0, int(y.item())].item())
    return pred, m, ce, py, logits.detach()


def ce_feature_step(model, x, x0, y, eps: float, step_size: float):
    probe = x.detach().requires_grad_(True)
    loss = F.cross_entropy(model(probe), y)
    grad = torch.autograd.grad(loss, probe)[0]
    return project_linf(x.detach() + step_size * grad.sign(), x0, eps).detach()


def pgd_states(model, x0, y, eps: float, step_size: float, steps: int):
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
    states = [x0.detach(), x.detach()]
    for _ in range(steps):
        x = ce_feature_step(model, x, x0, y, eps, step_size)
        states.append(x)
    return states


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def fit_transport_basis(model, dataset, rows, eps: float, step_size: float, steps: int, k: int, device):
    segs = []
    meta = []
    for image_id, label in rows:
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        states = pgd_states(model, x0, y, eps, step_size, steps)
        pred, m, _ce, _py, _logits = eval_state(model, states[-1], y)
        success = int(pred != label)
        meta.append({"image_id": image_id, "label": label, "success": success, "final_margin": m})
        if not success:
            continue
        with torch.no_grad():
            hs = [hidden(model, x).detach().cpu().numpy()[0].astype(np.float32) for x in states]
        for a, b in zip(hs[:-1], hs[1:]):
            v = b - a
            if np.linalg.norm(v) > 1e-8:
                segs.append(v)
    X = normalize_rows(np.stack(segs).astype(np.float64))
    Xc = X - X.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    basis = vt[:k].astype(np.float32)
    return {
        "basis": torch.tensor(basis, device=device),
        "n_segments": len(segs),
        "pc_var": ((s**2) / max(float((s**2).sum()), 1e-12))[:k].astype(float).tolist(),
        "meta": pd.DataFrame(meta),
    }


def square_candidate(x_best, x0, eps: float, query_idx: int, query_budget: int, p_init: float, rng_np):
    c, h, w = x0.shape[1:]
    perturbation = (x_best - x0).clone()
    p = p_selection(p_init, query_idx, query_budget)
    side = int(round(np.sqrt(p * c * h * w / c)))
    side = min(max(side, 1), h - 1)
    top = int(rng_np.integers(0, h - side + 1))
    left = int(rng_np.integers(0, w - side + 1))
    sign = -1.0 if rng_np.random() < 0.5 else 1.0
    perturbation[:, :, top : top + side, left : left + side] = sign * eps
    return project_linf(x0 + perturbation, x0, eps).detach()


def coeffs(h_delta: torch.Tensor, basis: torch.Tensor):
    return h_delta @ basis.T


def make_policy_input(model, x, x0, h_prev, y, basis, query_frac: float):
    with torch.no_grad():
        h0 = hidden(model, x0)
        h = hidden(model, x)
        disp = h - h0
        step = h - h_prev
        disp_c = coeffs(disp, basis)
        step_c = coeffs(step, basis)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        py = probs[:, int(y.item()) : int(y.item()) + 1]
        m = margin(logits, y).view(1, 1)
        ce = F.cross_entropy(logits, y).view(1, 1)
        q = torch.tensor([[query_frac]], device=x.device, dtype=torch.float32)
        # Full logits/probs are useful for target state but keep input compact.
        return torch.cat([disp_c, step_c, py, m, ce, q, logits.detach(), probs.detach()], dim=1)


def collect_policy_dataset(model, dataset, rows, basis, args, device):
    xs = []
    ys = []
    meta = []
    eps = args.eps / 255.0
    step_size = args.pgd_step_size / 255.0
    for image_id, label in rows:
        x0, _ = dataset[image_id]
        x0 = x0.unsqueeze(0).to(device)
        y = torch.tensor([label], device=device)
        rng_np = np.random.default_rng(args.seed + image_id * 101)
        x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps).detach()
        h_prev = hidden(model, x0).detach()
        pred, best_margin, _loss, _py, _logits = eval_state(model, x, y)
        for q in range(1, args.train_square_steps + 1):
            if q % args.collect_every == 0:
                inp = make_policy_input(model, x, x0, h_prev, y, basis, q / max(args.train_square_steps, 1))
                x_teacher = ce_feature_step(model, x, x0, y, eps, step_size)
                with torch.no_grad():
                    dh_teacher = hidden(model, x_teacher) - hidden(model, x)
                    target = coeffs(dh_teacher, basis)
                    target = target / target.norm(dim=1, keepdim=True).clamp_min(1e-12)
                xs.append(inp.cpu().numpy()[0].astype(np.float32))
                ys.append(target.cpu().numpy()[0].astype(np.float32))
                meta.append({"image_id": image_id, "label": label, "query": q, "margin": best_margin})
            cand = square_candidate(x, x0, eps, q, args.train_square_steps, args.p_init, rng_np)
            pred_c, m_c, _ce_c, _py_c, _logits_c = eval_state(model, cand, y)
            if m_c < best_margin:
                h_prev = hidden(model, x).detach()
                x = cand
                best_margin = m_c
                pred = pred_c
            if pred != label and args.stop_train_on_success:
                break
    return np.stack(xs), np.stack(ys), pd.DataFrame(meta)


class CorrectionPolicy(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_out),
        )

    def forward(self, x):
        y = self.net(x)
        return y / y.norm(dim=1, keepdim=True).clamp_min(1e-12)


def train_policy(X: np.ndarray, Y: np.ndarray, args, device):
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(X))
    n_val = max(32, int(0.15 * len(X)))
    val_idx = order[:n_val]
    tr_idx = order[n_val:]
    mean = X[tr_idx].mean(axis=0, keepdims=True).astype(np.float32)
    std = (X[tr_idx].std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    Xn = (X - mean) / std
    model = CorrectionPolicy(X.shape[1], Y.shape[1], args.policy_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xt = torch.tensor(Xn[tr_idx], device=device)
    yt = torch.tensor(Y[tr_idx], device=device)
    xv = torch.tensor(Xn[val_idx], device=device)
    yv = torch.tensor(Y[val_idx], device=device)
    bs = min(args.batch_size, len(xt))
    history = []
    for epoch in range(args.epochs):
        perm = torch.randperm(len(xt), device=device)
        losses = []
        for start in range(0, len(xt), bs):
            idx = perm[start : start + bs]
            pred = model(xt[idx])
            cos = (pred * yt[idx]).sum(dim=1)
            loss = (1.0 - cos).mean() + 0.05 * F.mse_loss(pred, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        with torch.no_grad():
            pv = model(xv)
            val_cos = float((pv * yv).sum(dim=1).mean().item())
            tr_cos = float((model(xt[: min(512, len(xt))]) * yt[: min(512, len(xt))]).sum(dim=1).mean().item())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "train_cos": tr_cos, "val_cos": val_cos})
    return model, torch.tensor(mean, device=device), torch.tensor(std, device=device), pd.DataFrame(history)


def policy_direction_step(model, policy, mean, std, x, x0, h_prev, y, basis, query_frac: float, eps: float, step_size: float):
    inp = make_policy_input(model, x, x0, h_prev, y, basis, query_frac)
    inp = (inp - mean) / std
    coeff = policy(inp)
    direction = coeff @ basis
    probe = x.detach().requires_grad_(True)
    score = (hidden(model, probe) * direction.detach()).sum()
    grad = torch.autograd.grad(score, probe)[0]
    return project_linf(x.detach() + step_size * grad.sign(), x0, eps).detach(), coeff.detach()


def pgd_energy_step(model, x, x0, y, basis, eps: float, step_size: float):
    probe = x.detach().requires_grad_(True)
    dh = hidden(model, probe) - hidden(model, x0).detach()
    c = coeffs(dh, basis)
    score = (c**2).sum()
    grad = torch.autograd.grad(score, probe)[0]
    return project_linf(x.detach() + step_size * grad.sign(), x0, eps).detach()


def random_feature_step(model, x, x0, eps: float, step_size: float, rng):
    probe = x.detach().requires_grad_(True)
    h = hidden(model, probe)
    r = torch.randn(h.shape, generator=rng, device=h.device)
    r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)
    score = (h * r).sum()
    grad = torch.autograd.grad(score, probe)[0]
    return project_linf(x.detach() + step_size * grad.sign(), x0, eps).detach()


def run_attack(model, policy, mean, std, basis, dataset, image_id, label, args, method, device):
    x0, _ = dataset[image_id]
    x0 = x0.unsqueeze(0).to(device)
    y = torch.tensor([label], device=device)
    eps = args.eps / 255.0
    rng_np = np.random.default_rng(args.seed + image_id * 997 + len(method))
    rng_t = torch.Generator(device=device).manual_seed(args.seed + image_id * 1871 + len(method))
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps).detach()
    h_prev = hidden(model, x0).detach()
    pred, best_margin, best_loss, _py, _logits = eval_state(model, x, y)
    queries = 1
    corr_q = 0
    sq_q = 0
    first_success = 0 if pred != label else -1
    curves = []
    for q in range(1, args.query_budget + 1):
        use_corr = method != "vanilla_square" and (q % args.correction_every == 0)
        if use_corr:
            if method == "learned_policy":
                cand, out_coeff = policy_direction_step(
                    model, policy, mean, std, x, x0, h_prev, y, basis, q / args.query_budget, eps, args.correction_step / 255.0
                )
            elif method == "pgd_energy":
                cand = pgd_energy_step(model, x, x0, y, basis, eps, args.correction_step / 255.0)
                out_coeff = torch.zeros((1, basis.shape[0]), device=device)
            elif method == "random_feature":
                cand = random_feature_step(model, x, x0, eps, args.correction_step / 255.0, rng_t)
                out_coeff = torch.zeros((1, basis.shape[0]), device=device)
            elif method == "ce_gradient":
                cand = ce_feature_step(model, x, x0, y, eps, args.correction_step / 255.0)
                out_coeff = torch.zeros((1, basis.shape[0]), device=device)
            else:
                raise ValueError(method)
            corr_q += 1
            proposal = method
        else:
            cand = square_candidate(x, x0, eps, q + args.square_init, args.query_budget, args.p_init, rng_np)
            proposal = "square"
            sq_q += 1
        pred_c, m_c, loss_c, _py_c, _logits_c = eval_state(model, cand, y)
        accepted = int(m_c < best_margin)
        if accepted:
            h_prev = hidden(model, x).detach()
            x = cand
            pred = pred_c
            best_margin = m_c
            best_loss = loss_c
        with torch.no_grad():
            dh = hidden(model, x) - hidden(model, x0)
            c = coeffs(dh, basis)
            energy = float((c**2).sum().item() / dh.norm(dim=1).pow(2).clamp_min(1e-12).item())
        success = int(pred != label)
        if success and first_success < 0:
            first_success = q
        curves.append(
            {
                "query": q,
                "proposal": proposal,
                "accepted": accepted,
                "margin": best_margin,
                "loss": best_loss,
                "pred": pred,
                "success": success,
                "pgd_basis_energy": energy,
                "square_queries": sq_q,
                "correction_queries": corr_q,
            }
        )
        if success and args.early_stop:
            break
    return {
        "success": int(first_success >= 0),
        "success_query": first_success,
        "final_margin": best_margin,
        "final_pred": pred,
        "square_queries": sq_q,
        "correction_queries": corr_q,
        "final_pgd_basis_energy": curves[-1]["pgd_basis_energy"] if curves else np.nan,
        "curves": curves,
    }


def summarize(df):
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
        )
        .reset_index()
    )


def make_plot(summary, history, out):
    order = ["vanilla_square", "random_feature", "pgd_energy", "learned_policy", "ce_gradient"]
    d = summary.set_index("method").reindex([m for m in order if m in set(summary.method)]).reset_index()
    labels = {
        "vanilla_square": "Square",
        "random_feature": "random feature",
        "pgd_energy": "PGD-basis energy",
        "learned_policy": "learned policy",
        "ce_gradient": "CE ceiling",
    }
    x = np.arange(len(d))
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.4), constrained_layout=True)
    axes[0].bar(x, d.asr, color="#4C78A8")
    axes[0].set_ylabel("ASR")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(x, d.mean_success_query, color="#54A24B")
    axes[1].set_ylabel("mean query to success")
    axes[2].bar(x, d.mean_final_margin, color="#F58518")
    axes[2].axhline(0, color="black", lw=1, ls="--")
    axes[2].set_ylabel("mean final margin")
    axes[3].plot(history.epoch, history.train_cos, label="train")
    axes[3].plot(history.epoch, history.val_cos, label="val")
    axes[3].set_ylabel("teacher cosine")
    axes[3].set_xlabel("epoch")
    axes[3].legend(frameon=False, fontsize=8)
    for ax in axes[:3]:
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(m, m) for m in d.method], rotation=25, ha="right", fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "square_learned_correction_policy_summary.png", dpi=220)
    fig.savefig(out / "square_learned_correction_policy_summary.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/square_learned_correction_policy_pilot")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--basis-train-images", type=int, default=80)
    p.add_argument("--policy-train-images", type=int, default=120)
    p.add_argument("--test-images", type=int, default=50)
    p.add_argument("--eps", type=float, default=2.0)
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--pgd-step-size", type=float, default=1.0)
    p.add_argument("--basis-k", type=int, default=10)
    p.add_argument("--train-square-steps", type=int, default=40)
    p.add_argument("--collect-every", type=int, default=2)
    p.add_argument("--stop-train-on-success", action="store_true")
    p.add_argument("--query-budget", type=int, default=100)
    p.add_argument("--correction-every", type=int, default=5)
    p.add_argument("--correction-step", type=float, default=1.0)
    p.add_argument("--p-init", type=float, default=0.3)
    p.add_argument("--square-init", type=int, default=0)
    p.add_argument("--methods", default="vanilla_square,random_feature,pgd_energy,learned_policy,ce_gradient")
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--policy-hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    selected = select_clean_correct(dataset, model, args.basis_train_images + args.policy_train_images + args.test_images, device)
    basis_rows = selected[: args.basis_train_images]
    policy_rows = selected[args.basis_train_images : args.basis_train_images + args.policy_train_images]
    test_rows = selected[args.basis_train_images + args.policy_train_images :]
    prior = fit_transport_basis(model, dataset, basis_rows, args.eps / 255.0, args.pgd_step_size / 255.0, args.pgd_steps, args.basis_k, device)
    prior["meta"].to_csv(out / "pgd_basis_train_meta.csv", index=False)
    X, Y, policy_meta = collect_policy_dataset(model, dataset, policy_rows, prior["basis"], args, device)
    np.savez_compressed(out / "policy_train_dataset.npz", X=X, Y=Y)
    policy_meta.to_csv(out / "policy_train_metadata.csv", index=False)
    policy, mean, std, history = train_policy(X, Y, args, device)
    history.to_csv(out / "policy_train_history.csv", index=False)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "basis": prior["basis"].detach().cpu(),
            "input_mean": mean.detach().cpu(),
            "input_std": std.detach().cpu(),
            "args": vars(args),
        },
        out / "learned_correction_policy.pt",
    )

    methods = [m for m in args.methods.split(",") if m]
    rows = []
    curves = []
    for image_id, label in test_rows:
        for method in methods:
            res = run_attack(model, policy, mean, std, prior["basis"], dataset, image_id, label, args, method, device)
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
                    "final_pgd_basis_energy": res["final_pgd_basis_energy"],
                }
            )
            for c in res["curves"]:
                c.update({"image_id": image_id, "label": label, "method": method})
                curves.append(c)
    df = pd.DataFrame(rows)
    curve_df = pd.DataFrame(curves)
    summary = summarize(df)
    df.to_csv(out / "square_learned_correction_policy_results.csv", index=False)
    curve_df.to_csv(out / "square_learned_correction_policy_curves.csv", index=False)
    summary.to_csv(out / "square_learned_correction_policy_summary.csv", index=False)
    make_plot(summary, history, out)
    meta = vars(args)
    meta.update({"n_basis_rows": len(basis_rows), "n_policy_rows": len(policy_rows), "n_test_rows": len(test_rows), "n_policy_examples": len(X), "n_pgd_segments": prior["n_segments"]})
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("Policy final history:")
    print(history.tail(5).to_string(index=False))
    print("\nAttack summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
