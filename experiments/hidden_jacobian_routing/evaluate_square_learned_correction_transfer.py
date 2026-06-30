#!/usr/bin/env python3
"""Surrogate-trained Square trajectory correction in a black-box target setting.

This pilot tests whether a learned correction policy can be trained from
surrogate Square states and then used to propose target-model black-box
queries.  The policy observes the surrogate state of the current Square attack
and predicts a feature-space correction direction.  Candidate images are
generated through the surrogate, but accepted or rejected only by target-model
queries.

The target threat model here is surrogate-assisted black-box: the attacker has
white-box access to a surrogate model and query access to the target model.
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

from experiments.hidden_jacobian_routing.common import margin, project_linf  # noqa: E402
from experiments.hidden_jacobian_routing.evaluate_square_learned_correction_policy import (  # noqa: E402
    ce_feature_step,
    collect_policy_dataset,
    eval_state,
    fit_transport_basis,
    make_policy_input,
    pgd_energy_step,
    policy_direction_step,
    random_feature_step,
    square_candidate,
    summarize,
    train_policy,
)
from experiments.hidden_jacobian_routing.trace_jacobian_singular_roads import feat, load_model, set_seed  # noqa: E402
from utils.load_models import load_cifar_model  # noqa: E402


def target_eval(target_model, x: torch.Tensor, y: torch.Tensor):
    with torch.no_grad():
        logits = target_model(x)
        probs = F.softmax(logits, dim=1)
        pred = int(logits.argmax(1).item())
        m = float(margin(logits, y).item())
        ce = float(F.cross_entropy(logits, y).item())
        py = float(probs[0, int(y.item())].item())
    return pred, m, ce, py, logits.detach()


def hidden(model, x: torch.Tensor) -> torch.Tensor:
    return feat(model, x)


def select_source_clean_correct(dataset, source_model, n: int, device):
    rows = []
    for idx in range(len(dataset)):
        x, y0 = dataset[idx]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(source_model(x).argmax(1).item())
        if pred == int(y0):
            rows.append((idx, int(y0)))
        if len(rows) >= n:
            break
    return rows


def select_common_clean_correct(dataset, source_model, target_model, n: int, device):
    rows = []
    for idx in range(len(dataset)):
        x, y0 = dataset[idx]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            ps = int(source_model(x).argmax(1).item())
            pt = int(target_model(x).argmax(1).item())
        if ps == int(y0) and pt == int(y0):
            rows.append((idx, int(y0)))
        if len(rows) >= n:
            break
    return rows


def run_transfer_attack(
    source_model,
    target_model,
    policy,
    mean,
    std,
    basis,
    dataset,
    image_id: int,
    label: int,
    args,
    method: str,
    device,
):
    x0, _ = dataset[image_id]
    x0 = x0.unsqueeze(0).to(device)
    y = torch.tensor([label], device=device)
    eps = args.eps / 255.0
    rng_np = np.random.default_rng(args.seed + image_id * 997 + len(method))
    rng_t = torch.Generator(device=device).manual_seed(args.seed + image_id * 1871 + len(method))
    init_gen = torch.Generator(device=device).manual_seed(args.seed + image_id * 431)
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps, generator=init_gen), x0, eps).detach()
    h_prev = hidden(source_model, x0).detach()
    pred, best_margin, best_loss, _py, _logits = target_eval(target_model, x, y)
    queries = 1
    corr_q = 0
    sq_q = 0
    first_success = 0 if pred != label else -1
    curves = []
    for q in range(1, args.query_budget + 1):
        use_corr = method != "vanilla_square" and (q % args.correction_every == 0)
        if use_corr:
            if method == "learned_policy":
                cand, _out_coeff = policy_direction_step(
                    source_model,
                    policy,
                    mean,
                    std,
                    x,
                    x0,
                    h_prev,
                    y,
                    basis,
                    q / args.query_budget,
                    eps,
                    args.correction_step / 255.0,
                )
            elif method == "pgd_energy":
                cand = pgd_energy_step(source_model, x, x0, y, basis, eps, args.correction_step / 255.0)
            elif method == "random_feature":
                cand = random_feature_step(source_model, x, x0, eps, args.correction_step / 255.0, rng_t)
            elif method == "surrogate_ce":
                cand = ce_feature_step(source_model, x, x0, y, eps, args.correction_step / 255.0)
            else:
                raise ValueError(method)
            corr_q += 1
            proposal = method
        else:
            cand = square_candidate(x, x0, eps, q + args.square_init, args.query_budget, args.p_init, rng_np)
            proposal = "square"
            sq_q += 1
        queries += 1
        pred_c, m_c, loss_c, _py_c, _logits_c = target_eval(target_model, cand, y)
        accepted = int(m_c < best_margin)
        if accepted:
            h_prev = hidden(source_model, x).detach()
            x = cand
            pred = pred_c
            best_margin = m_c
            best_loss = loss_c
        with torch.no_grad():
            dh = hidden(source_model, x) - hidden(source_model, x0)
            c = dh @ basis.T
            energy = float((c**2).sum().item() / dh.norm(dim=1).pow(2).clamp_min(1e-12).item())
        success = int(pred != label)
        if success and first_success < 0:
            first_success = q
        curves.append(
            {
                "query": q,
                "proposal": proposal,
                "accepted": accepted,
                "target_margin": best_margin,
                "target_loss": best_loss,
                "target_pred": pred,
                "success": success,
                "source_pgd_basis_energy": energy,
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
        "queries": queries,
        "square_queries": sq_q,
        "correction_queries": corr_q,
        "final_pgd_basis_energy": curves[-1]["source_pgd_basis_energy"] if curves else np.nan,
        "curves": curves,
    }


def run_nes_attack(target_model, dataset, image_id: int, label: int, args, device):
    """Target-only score-based NES baseline with antithetic samples."""
    x0, _ = dataset[image_id]
    x0 = x0.unsqueeze(0).to(device)
    y = torch.tensor([label], device=device)
    eps = args.eps / 255.0
    sigma = args.nes_sigma / 255.0
    step = args.nes_step / 255.0
    rng = torch.Generator(device=device).manual_seed(args.seed + image_id * 4337)
    x = project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps, generator=rng), x0, eps).detach()
    pred, best_margin, best_loss, _py, _logits = target_eval(target_model, x, y)
    queries = 1
    first_success = 0 if pred != label else -1
    curves = []
    q_index = 0
    while queries + 2 * args.nes_samples + 1 <= args.query_budget + 1:
        dirs = torch.randn((args.nes_samples,) + tuple(x.shape[1:]), generator=rng, device=device)
        dirs = dirs / dirs.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        grad_est = torch.zeros_like(x)
        for u in dirs:
            u = u.unsqueeze(0)
            xp = project_linf(x + sigma * u, x0, eps)
            xm = project_linf(x - sigma * u, x0, eps)
            _pred_p, _m_p, ce_p, _py_p, _log_p = target_eval(target_model, xp, y)
            _pred_m, _m_m, ce_m, _py_m, _log_m = target_eval(target_model, xm, y)
            queries += 2
            grad_est += ((ce_p - ce_m) / max(2.0 * sigma, 1e-12)) * u
        grad_est = grad_est / max(args.nes_samples, 1)
        cand = project_linf(x + step * grad_est.sign(), x0, eps)
        queries += 1
        pred_c, m_c, loss_c, _py_c, _logits_c = target_eval(target_model, cand, y)
        q_index += 1
        accepted = int(m_c < best_margin)
        if accepted:
            x = cand.detach()
            pred = pred_c
            best_margin = m_c
            best_loss = loss_c
        success = int(pred != label)
        if success and first_success < 0:
            first_success = queries - 1
        curves.append(
            {
                "query": queries - 1,
                "proposal": "nes",
                "accepted": accepted,
                "target_margin": best_margin,
                "target_loss": best_loss,
                "target_pred": pred,
                "success": success,
                "source_pgd_basis_energy": np.nan,
                "square_queries": 0,
                "correction_queries": 0,
                "nes_iterations": q_index,
            }
        )
        if success and args.early_stop:
            break
    return {
        "success": int(first_success >= 0),
        "success_query": first_success,
        "final_margin": best_margin,
        "final_pred": pred,
        "queries": queries,
        "square_queries": 0,
        "correction_queries": 0,
        "final_pgd_basis_energy": np.nan,
        "curves": curves,
    }


def pullback_pixel_basis(source_model, x_ref: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Pull hidden transport directions back to input-space directions."""
    dirs = []
    for u in basis:
        probe = x_ref.detach().requires_grad_(True)
        score = (hidden(source_model, probe) * u.view(1, -1).detach()).sum()
        grad = torch.autograd.grad(score, probe)[0].detach()
        grad = grad / grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        dirs.append(grad[0])
    return torch.stack(dirs, dim=0)


def random_pixel_basis_like(pixel_basis: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=pixel_basis.device).manual_seed(seed)
    r = torch.randn(pixel_basis.shape, generator=gen, device=pixel_basis.device)
    return r / r.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)


def jvp_sketch_pixel_basis(source_model, x_ref: torch.Tensor, k: int, n_probe: int, seed: int) -> torch.Tensor:
    """Pick high-hidden-mobility input directions by randomized JVP probes."""
    gen = torch.Generator(device=x_ref.device).manual_seed(seed)

    def f(inp):
        return hidden(source_model, inp)

    scored = []
    for _ in range(max(n_probe, k)):
        v = torch.randn(x_ref.shape, generator=gen, device=x_ref.device)
        v = v / v.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        with torch.no_grad():
            _h, jv = torch.autograd.functional.jvp(f, x_ref.detach(), v, create_graph=False, strict=False)
            score = float(jv.flatten(1).norm(dim=1).item())
        scored.append((score, v.detach()[0]))
    scored.sort(key=lambda z: z[0], reverse=True)
    basis = torch.stack([v for _s, v in scored[:k]], dim=0)
    return basis / basis.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)


def coeff_direction(pixel_basis: torch.Tensor, rng_np: np.random.Generator) -> torch.Tensor:
    alpha = torch.tensor(rng_np.normal(size=(pixel_basis.shape[0],)), device=pixel_basis.device, dtype=pixel_basis.dtype)
    direction = (alpha.view(-1, 1, 1, 1) * pixel_basis).sum(dim=0, keepdim=True)
    return direction / direction.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)


def biased_coeff_direction(pixel_basis: torch.Tensor, g_alpha: torch.Tensor, rng_np: np.random.Generator, noise_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.tensor(rng_np.normal(size=(pixel_basis.shape[0],)), device=pixel_basis.device, dtype=pixel_basis.dtype)
    if float(g_alpha.norm().item()) > 1e-8:
        alpha = g_alpha / g_alpha.norm().clamp_min(1e-12) + noise_scale * noise / noise.norm().clamp_min(1e-12)
    else:
        alpha = noise
    direction = (alpha.view(-1, 1, 1, 1) * pixel_basis).sum(dim=0, keepdim=True)
    direction = direction / direction.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
    return direction, alpha.detach()


def parse_alt_probability(method: str) -> float:
    if method.endswith("_p50"):
        return 0.50
    if method.endswith("_p25"):
        return 0.25
    if method.endswith("_p75"):
        return 0.75
    return 0.25


def parse_coeff_init_queries(method: str, default: int) -> int:
    prefix = "square_init_coeff"
    if not method.startswith(prefix):
        return default
    suffix = method[len(prefix) :]
    if suffix.isdigit():
        return int(suffix)
    return default


def canonical_method(method: str) -> str:
    aliases = {
        "square": "vanilla_square",
        "coeff_only_transport": "coeff_only",
        "ce_transport_alternating": "ce_transport_alt",
    }
    return aliases.get(method, method)


def coeff_proposal(x0: torch.Tensor, eps: float, pixel_basis: torch.Tensor, rng_np: np.random.Generator) -> torch.Tensor:
    direction = coeff_direction(pixel_basis, rng_np)
    return project_linf(x0 + eps * direction.sign(), x0, eps).detach()


def coeff_update_proposal(
    x: torch.Tensor,
    x0: torch.Tensor,
    eps: float,
    step_size: float,
    pixel_basis: torch.Tensor,
    rng_np: np.random.Generator,
) -> torch.Tensor:
    direction = coeff_direction(pixel_basis, rng_np)
    return project_linf(x + step_size * direction.sign(), x0, eps).detach()


def run_hybrid_coeff_attack(
    source_model,
    target_model,
    basis,
    dataset,
    image_id: int,
    label: int,
    args,
    method: str,
    device,
):
    x0, _ = dataset[image_id]
    x0 = x0.unsqueeze(0).to(device)
    y = torch.tensor([label], device=device)
    eps = args.eps / 255.0
    rng_np = np.random.default_rng(args.seed + image_id * 7681 + len(method))
    x = x0.detach().clone()
    pred, best_margin, best_loss, _py, _logits = target_eval(target_model, x, y)
    first_success = 0 if pred != label else -1
    square_q = 0
    coeff_q = 0
    curves = []

    transport_dirs = pullback_pixel_basis(source_model, x0, basis)
    if "random_coeff" in method:
        pixel_dirs = random_pixel_basis_like(transport_dirs, args.seed + image_id * 991)
        coeff_label = "random_coeff"
    elif "jvp_coeff" in method:
        pixel_dirs = jvp_sketch_pixel_basis(source_model, x0, transport_dirs.shape[0], args.jvp_probe_dirs, args.seed + image_id * 1237)
        coeff_label = "jvp_coeff"
    else:
        pixel_dirs = transport_dirs
        coeff_label = "transport_coeff"

    for q in range(1, args.query_budget + 1):
        if method == "coeff_only":
            cand = coeff_proposal(x0, eps, pixel_dirs, rng_np)
            proposal = coeff_label
            coeff_q += 1
        elif method.startswith("square_init_coeff") or method in {"random_coeff_square", "jvp_coeff_square"}:
            init_queries = parse_coeff_init_queries(method, args.coeff_init_queries)
            if q <= init_queries:
                cand = coeff_proposal(x0, eps, pixel_dirs, rng_np)
                proposal = coeff_label
                coeff_q += 1
            else:
                cand = square_candidate(x, x0, eps, q + args.square_init, args.query_budget, args.p_init, rng_np)
                proposal = "square"
                square_q += 1
        elif method.startswith("square_alt_transport") or method.startswith("square_alt_random_coeff"):
            p_coeff = parse_alt_probability(method)
            if rng_np.random() < p_coeff:
                cand = coeff_update_proposal(x, x0, eps, args.correction_step / 255.0, pixel_dirs, rng_np)
                proposal = coeff_label
                coeff_q += 1
            else:
                cand = square_candidate(x, x0, eps, q + args.square_init, args.query_budget, args.p_init, rng_np)
                proposal = "square"
                square_q += 1
        else:
            raise ValueError(method)

        pred_c, m_c, loss_c, _py_c, _logits_c = target_eval(target_model, cand, y)
        accepted = int(m_c < best_margin)
        if accepted:
            x = cand.detach()
            pred = pred_c
            best_margin = m_c
            best_loss = loss_c
        success = int(pred != label)
        if success and first_success < 0:
            first_success = q
        curves.append(
            {
                "query": q,
                "proposal": proposal,
                "accepted": accepted,
                "target_margin": best_margin,
                "target_loss": best_loss,
                "target_pred": pred,
                "success": success,
                "source_pgd_basis_energy": np.nan,
                "square_queries": square_q,
                "correction_queries": coeff_q,
            }
        )
        if success and args.early_stop:
            break
    return {
        "success": int(first_success >= 0),
        "success_query": first_success,
        "final_margin": best_margin,
        "final_pred": pred,
        "queries": len(curves) + 1,
        "square_queries": square_q,
        "correction_queries": coeff_q,
        "final_pgd_basis_energy": np.nan,
        "curves": curves,
    }


def input_projection_coeff(pixel_basis: torch.Tensor, update: torch.Tensor, lam: float) -> tuple[torch.Tensor, float]:
    d = pixel_basis.flatten(1)
    u = update.flatten()
    gram = d @ d.T
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    rhs = d @ u
    p = torch.linalg.solve(gram + lam * eye, rhs)
    recon = p @ d
    frac = float(recon.norm().item() / max(float(u.norm().item()), 1e-12))
    return p.detach(), frac


def feature_projection_coeff(source_model, basis: torch.Tensor, x_old: torch.Tensor, x_new: torch.Tensor) -> tuple[torch.Tensor, float]:
    with torch.no_grad():
        dh = hidden(source_model, x_new) - hidden(source_model, x_old)
        p = (dh @ basis.T)[0]
        frac = float(p.norm().item() / max(float(dh.norm(dim=1).item()), 1e-12))
    return p.detach(), frac


def update_feedback(g_alpha: torch.Tensor, p: torch.Tensor, improvement: float, rho: float, lam: float) -> torch.Tensor:
    if improvement <= 0 or float(p.norm().item()) <= 1e-10:
        return g_alpha
    scaled = float(improvement) * p / (p.norm().pow(2) + lam)
    return (1.0 - rho) * g_alpha + rho * scaled


def run_square_feedback_attack(
    source_model,
    target_model,
    basis,
    dataset,
    image_id: int,
    label: int,
    args,
    method: str,
    device,
):
    x0, _ = dataset[image_id]
    x0 = x0.unsqueeze(0).to(device)
    y = torch.tensor([label], device=device)
    eps = args.eps / 255.0
    step_size = args.correction_step / 255.0
    rng_np = np.random.default_rng(args.seed + image_id * 9137 + len(method))
    x = x0.detach().clone()
    pred, best_margin, best_loss, _py, _logits = target_eval(target_model, x, y)
    first_success = 0 if pred != label else -1
    transport_dirs = pullback_pixel_basis(source_model, x0, basis)
    if "random" in method:
        pixel_dirs = random_pixel_basis_like(transport_dirs, args.seed + image_id * 2999)
        feedback_kind = "input"
    else:
        pixel_dirs = transport_dirs
        feedback_kind = "feature" if "feature" in method else "input"
    g_alpha = torch.zeros((pixel_dirs.shape[0],), device=device, dtype=torch.float32)
    square_q = 0
    coeff_q = 0
    curves = []
    for q in range(1, args.query_budget + 1):
        use_coeff = rng_np.random() < args.feedback_coeff_prob
        x_before = x.detach()
        margin_before = best_margin
        if use_coeff:
            direction, dalpha = biased_coeff_direction(pixel_dirs, g_alpha, rng_np, args.feedback_noise)
            cand = project_linf(x + step_size * direction.sign(), x0, eps).detach()
            proposal = "feedback_coeff"
            coeff_q += 1
        else:
            cand = square_candidate(x, x0, eps, q + args.square_init, args.query_budget, args.p_init, rng_np)
            proposal = "square"
            dalpha = None
            square_q += 1
        pred_c, m_c, loss_c, _py_c, _logits_c = target_eval(target_model, cand, y)
        improvement = float(best_margin - m_c)
        accepted = int(m_c < best_margin)
        proj_norm = np.nan
        proj_frac = np.nan
        if accepted:
            x = cand.detach()
            pred = pred_c
            best_margin = m_c
            best_loss = loss_c
            if proposal == "square":
                if feedback_kind == "feature":
                    p, proj_frac = feature_projection_coeff(source_model, basis, x_before, cand)
                    proj_norm = float(p.norm().item())
                else:
                    p, proj_frac = input_projection_coeff(pixel_dirs, cand - x_before, args.feedback_lam)
                    proj_norm = float(p.norm().item())
                if proj_frac >= args.feedback_min_fraction:
                    g_alpha = update_feedback(g_alpha, p, improvement, args.feedback_rho, args.feedback_lam)
            elif proposal == "feedback_coeff" and dalpha is not None:
                g_alpha = update_feedback(g_alpha, dalpha.detach(), improvement, args.feedback_rho, args.feedback_lam)
        success = int(pred != label)
        if success and first_success < 0:
            first_success = q
        curves.append(
            {
                "query": q,
                "proposal": proposal,
                "accepted": accepted,
                "target_margin": best_margin,
                "target_loss": best_loss,
                "target_pred": pred,
                "success": success,
                "source_pgd_basis_energy": np.nan,
                "square_queries": square_q,
                "correction_queries": coeff_q,
                "margin_before": margin_before,
                "improvement": improvement,
                "road_projection_norm": proj_norm,
                "road_projection_fraction": proj_frac,
                "g_alpha_norm": float(g_alpha.norm().item()),
                "delta_linf": float((x - x0).abs().max().item()),
            }
        )
        if success and args.early_stop:
            break
    return {
        "success": int(first_success >= 0),
        "success_query": first_success,
        "final_margin": best_margin,
        "final_pred": pred,
        "queries": len(curves) + 1,
        "square_queries": square_q,
        "correction_queries": coeff_q,
        "final_pgd_basis_energy": np.nan,
        "curves": curves,
    }


def surrogate_pgd_init(source_model, x0: torch.Tensor, y: torch.Tensor, eps: float, step_size: float, steps: int) -> torch.Tensor:
    x = x0.detach().clone()
    for _ in range(steps):
        x = ce_feature_step(source_model, x, x0, y, eps, step_size)
    return x.detach()


def run_query_refined_transfer_attack(
    source_model,
    target_model,
    basis,
    dataset,
    image_id: int,
    label: int,
    args,
    method: str,
    device,
):
    x0, _ = dataset[image_id]
    x0 = x0.unsqueeze(0).to(device)
    y = torch.tensor([label], device=device)
    eps = args.eps / 255.0
    step_size = args.correction_step / 255.0
    rng_np = np.random.default_rng(args.seed + image_id * 8713 + len(method))
    x = x0.detach().clone()
    square_q = 0
    correction_q = 0
    curves = []

    if method in {"one_shot_surrogate_pgd", "pgd_init_square"}:
        x = surrogate_pgd_init(source_model, x0, y, eps, step_size, args.pgd_init_steps)

    pred, best_margin, best_loss, _py, _logits = target_eval(target_model, x, y)
    first_success = 0 if pred != label else -1

    if method == "one_shot_surrogate_pgd":
        return {
            "success": int(first_success >= 0),
            "success_query": first_success,
            "final_margin": best_margin,
            "final_pred": pred,
            "queries": 1,
            "square_queries": 0,
            "correction_queries": 0,
            "final_pgd_basis_energy": np.nan,
            "curves": [
                {
                    "query": 0,
                    "proposal": "surrogate_pgd",
                    "accepted": 1,
                    "target_margin": best_margin,
                    "target_loss": best_loss,
                    "target_pred": pred,
                    "success": int(first_success >= 0),
                    "source_pgd_basis_energy": np.nan,
                    "square_queries": 0,
                    "correction_queries": 0,
                }
            ],
        }

    pixel_dirs = pullback_pixel_basis(source_model, x0, basis)
    for q in range(1, args.query_budget + 1):
        if method == "target_accepted_ce":
            cand = ce_feature_step(source_model, x, x0, y, eps, step_size)
            proposal = "surrogate_ce"
            correction_q += 1
        elif method == "target_accepted_transport":
            cand = coeff_update_proposal(x, x0, eps, step_size, pixel_dirs, rng_np)
            proposal = "transport_coeff"
            correction_q += 1
        elif method == "ce_transport_alt":
            if q % 2 == 1:
                cand = ce_feature_step(source_model, x, x0, y, eps, step_size)
                proposal = "surrogate_ce"
            else:
                cand = coeff_update_proposal(x, x0, eps, step_size, pixel_dirs, rng_np)
                proposal = "transport_coeff"
            correction_q += 1
        elif method == "pgd_init_square":
            cand = square_candidate(x, x0, eps, q + args.square_init, args.query_budget, args.p_init, rng_np)
            proposal = "square"
            square_q += 1
        else:
            raise ValueError(method)

        pred_c, m_c, loss_c, _py_c, _logits_c = target_eval(target_model, cand, y)
        accepted = int(m_c < best_margin)
        if accepted:
            x = cand.detach()
            pred = pred_c
            best_margin = m_c
            best_loss = loss_c
        success = int(pred != label)
        if success and first_success < 0:
            first_success = q
        curves.append(
            {
                "query": q,
                "proposal": proposal,
                "accepted": accepted,
                "target_margin": best_margin,
                "target_loss": best_loss,
                "target_pred": pred,
                "success": success,
                "source_pgd_basis_energy": np.nan,
                "square_queries": square_q,
                "correction_queries": correction_q,
            }
        )
        if success and args.early_stop:
            break
    return {
        "success": int(first_success >= 0),
        "success_query": first_success,
        "final_margin": best_margin,
        "final_pred": pred,
        "queries": len(curves) + 1,
        "square_queries": square_q,
        "correction_queries": correction_q,
        "final_pgd_basis_energy": np.nan,
        "curves": curves,
    }


def make_plot(summary: pd.DataFrame, history: pd.DataFrame, out: Path):
    order = [
        "vanilla_square",
        "nes",
        "coeff_only",
        "coeff_only_transport",
        "square_init_coeff20",
        "square_init_coeff10",
        "square_init_coeff30",
        "random_coeff_square",
        "jvp_coeff_square",
        "square_alt_transport_p25",
        "square_alt_transport_p50",
        "square_alt_random_coeff_p25",
        "one_shot_surrogate_pgd",
        "target_accepted_ce",
        "target_accepted_transport",
        "ce_transport_alt",
        "pgd_init_square",
        "square_feedback_feature",
        "square_feedback_input",
        "square_feedback_random_input",
        "random_feature",
        "pgd_energy",
        "learned_policy",
        "surrogate_ce",
    ]
    labels = {
        "vanilla_square": "Square",
        "nes": "NES",
        "coeff_only": "coeff only",
        "coeff_only_transport": "coeff only",
        "square_init_coeff20": "coeff init + Square",
        "square_init_coeff10": "coeff10 + Square",
        "square_init_coeff30": "coeff30 + Square",
        "random_coeff_square": "random coeff + Square",
        "jvp_coeff_square": "JVP coeff + Square",
        "square_alt_transport_p25": "alt transport 25%",
        "square_alt_transport_p50": "alt transport 50%",
        "square_alt_random_coeff_p25": "alt random 25%",
        "one_shot_surrogate_pgd": "one-shot PGD",
        "target_accepted_ce": "accepted CE",
        "target_accepted_transport": "accepted transport",
        "ce_transport_alt": "CE/transport alt",
        "pgd_init_square": "PGD init + Square",
        "square_feedback_feature": "feedback feature",
        "square_feedback_input": "feedback input",
        "square_feedback_random_input": "feedback random",
        "random_feature": "random feature",
        "pgd_energy": "PGD-basis energy",
        "learned_policy": "learned policy",
        "surrogate_ce": "surrogate CE",
    }
    d = summary.set_index("method").reindex([m for m in order if m in set(summary.method)]).reset_index()
    x = np.arange(len(d))
    fig, axes = plt.subplots(1, 4, figsize=(15.4, 3.4), constrained_layout=True)
    axes[0].bar(x, d.asr, color="#4C78A8")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("target ASR")
    axes[1].bar(x, d.mean_success_query, color="#54A24B")
    axes[1].set_ylabel("target mean query")
    axes[2].bar(x, d.mean_final_margin, color="#F58518")
    axes[2].axhline(0, color="black", lw=1, ls="--")
    axes[2].set_ylabel("target final margin")
    if len(history) and {"epoch", "train_cos", "val_cos"}.issubset(history.columns):
        axes[3].plot(history.epoch, history.train_cos, label="train")
        axes[3].plot(history.epoch, history.val_cos, label="val")
        axes[3].set_ylabel("surrogate teacher cosine")
        axes[3].set_xlabel("epoch")
        axes[3].legend(frameon=False, fontsize=8)
    else:
        axes[3].axis("off")
    for ax in axes[:3]:
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(m, m) for m in d.method], rotation=25, ha="right", fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "square_learned_correction_transfer_summary.png", dpi=220)
    fig.savefig(out / "square_learned_correction_transfer_summary.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/square_learned_correction_transfer_pilot")
    p.add_argument("--dataset-root", default="data/cifar10")
    p.add_argument("--source", default="bbb_resnet50")
    p.add_argument("--target", default="bbb_vgg19_bn")
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
    p.add_argument("--methods", default="vanilla_square,random_feature,pgd_energy,learned_policy,surrogate_ce")
    p.add_argument("--nes-samples", type=int, default=5)
    p.add_argument("--nes-sigma", type=float, default=0.5)
    p.add_argument("--nes-step", type=float, default=1.0)
    p.add_argument("--coeff-init-queries", type=int, default=20)
    p.add_argument("--pgd-init-steps", type=int, default=10)
    p.add_argument("--feedback-coeff-prob", type=float, default=0.5)
    p.add_argument("--feedback-noise", type=float, default=0.35)
    p.add_argument("--feedback-rho", type=float, default=0.35)
    p.add_argument("--feedback-lam", type=float, default=1e-3)
    p.add_argument("--feedback-min-fraction", type=float, default=0.01)
    p.add_argument("--jvp-probe-dirs", type=int, default=24)
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--policy-hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.source != "bbb_resnet50":
        raise ValueError("This pilot currently supports bbb_resnet50 as the surrogate because it uses the existing pooled-layer4 wrapper.")

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_model = load_model(device)
    target_model = load_cifar_model(args.target).to(device).eval()
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    methods = [canonical_method(m) for m in args.methods.split(",") if m]

    train_needed = args.basis_train_images + args.policy_train_images
    source_rows = select_source_clean_correct(dataset, source_model, train_needed, device)
    basis_rows = source_rows[: args.basis_train_images]
    policy_rows = source_rows[args.basis_train_images : train_needed]
    test_rows = select_common_clean_correct(dataset, source_model, target_model, args.test_images, device)

    prior = fit_transport_basis(source_model, dataset, basis_rows, args.eps / 255.0, args.pgd_step_size / 255.0, args.pgd_steps, args.basis_k, device)
    prior["meta"].to_csv(out / "source_pgd_basis_train_meta.csv", index=False)
    policy = None
    mean = None
    std = None
    history = pd.DataFrame()
    n_policy_examples = 0
    if "learned_policy" in methods:
        X, Y, policy_meta = collect_policy_dataset(source_model, dataset, policy_rows, prior["basis"], args, device)
        n_policy_examples = int(len(X))
        np.savez_compressed(out / "surrogate_policy_train_dataset.npz", X=X, Y=Y)
        policy_meta.to_csv(out / "surrogate_policy_train_metadata.csv", index=False)
        policy, mean, std, history = train_policy(X, Y, args, device)
        history.to_csv(out / "surrogate_policy_train_history.csv", index=False)
        torch.save(
            {
                "policy_state_dict": policy.state_dict(),
                "basis": prior["basis"].detach().cpu(),
                "input_mean": mean.detach().cpu(),
                "input_std": std.detach().cpu(),
                "args": vars(args),
            },
            out / "surrogate_learned_correction_policy.pt",
        )
    rows = []
    curves = []
    for image_id, label in test_rows:
        for method in methods:
            if method == "nes":
                res = run_nes_attack(target_model, dataset, image_id, label, args, device)
            elif method == "coeff_only" or method.startswith("square_init_coeff") or method.startswith("square_alt_transport") or method.startswith("square_alt_random_coeff") or method in {"random_coeff_square", "jvp_coeff_square"}:
                res = run_hybrid_coeff_attack(
                    source_model,
                    target_model,
                    prior["basis"],
                    dataset,
                    image_id,
                    label,
                    args,
                    method,
                    device,
                )
            elif method in {"one_shot_surrogate_pgd", "target_accepted_ce", "target_accepted_transport", "ce_transport_alt", "pgd_init_square"}:
                res = run_query_refined_transfer_attack(
                    source_model,
                    target_model,
                    prior["basis"],
                    dataset,
                    image_id,
                    label,
                    args,
                    method,
                    device,
                )
            elif method in {"square_feedback_feature", "square_feedback_input", "square_feedback_random_input"}:
                res = run_square_feedback_attack(
                    source_model,
                    target_model,
                    prior["basis"],
                    dataset,
                    image_id,
                    label,
                    args,
                    method,
                    device,
                )
            else:
                res = run_transfer_attack(
                    source_model,
                    target_model,
                    policy,
                    mean,
                    std,
                    prior["basis"],
                    dataset,
                    image_id,
                    label,
                    args,
                    method,
                    device,
                )
            rows.append(
                {
                    "image_id": image_id,
                    "label": label,
                    "source": args.source,
                    "target": args.target,
                    "method": method,
                    "success": res["success"],
                    "success_query": res["success_query"],
                    "final_margin": res["final_margin"],
                    "final_pred": res["final_pred"],
                    "queries": res["queries"],
                    "square_queries": res["square_queries"],
                    "correction_queries": res["correction_queries"],
                    "final_source_pgd_basis_energy": res["final_pgd_basis_energy"],
                }
            )
            for c in res["curves"]:
                c.update({"image_id": image_id, "label": label, "source": args.source, "target": args.target, "method": method})
                curves.append(c)

    df = pd.DataFrame(rows)
    curve_df = pd.DataFrame(curves)
    summary = summarize(df.rename(columns={"final_source_pgd_basis_energy": "final_pgd_basis_energy"}))
    df.to_csv(out / "square_learned_correction_transfer_results.csv", index=False)
    curve_df.to_csv(out / "square_learned_correction_transfer_curves.csv", index=False)
    summary.to_csv(out / "square_learned_correction_transfer_summary.csv", index=False)
    make_plot(summary, history, out)
    meta = vars(args)
    meta.update(
        {
            "n_basis_rows": len(basis_rows),
            "n_policy_rows": len(policy_rows),
            "n_test_rows": len(test_rows),
            "n_policy_examples": n_policy_examples,
            "n_pgd_segments": int(prior["n_segments"]),
            "threat_model": "surrogate-assisted black-box: source gradients/features generate proposals; target queries accept/reject",
        }
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    if len(history):
        print("Policy final history:")
        print(history.tail(5).to_string(index=False))
    else:
        print("Policy training skipped because learned_policy is not in --methods.")
    print("\nTransfer attack summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
