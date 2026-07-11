#!/usr/bin/env python3
"""RobustBench pilot for hidden-Jacobian road/mobility diagnostics.

This script checks whether the paper's hidden-Jacobian mobility picture is
visible on official RobustBench CIFAR-10 Linf models.  It is deliberately
resumable: each model writes clean-image selection, APGD summaries, candidate
direction diagnostics, and a short findings note independently.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from robustbench.utils import load_model as load_robustbench_model
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hidden_jacobian_routing.common import margin, project_linf  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace(",", "_")


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_csv(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def top_other_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return margin(logits, y)


def logits_stats(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1)
        m = top_other_margin(logits, y)
        ce = F.cross_entropy(logits, y, reduction="none")
    return pred.detach().cpu().numpy(), m.detach().cpu().numpy(), ce.detach().cpu().numpy()


class HookedModel(nn.Module):
    def __init__(self, model: nn.Module, layer_name: str):
        super().__init__()
        self.model = model
        modules = dict(model.named_modules())
        if layer_name not in modules:
            available = [n for n in modules if n.endswith("layer4") or n in {"layer3", "layer4", "linear"}]
            raise ValueError(f"Layer {layer_name!r} not found. Candidate modules: {available[:40]}")
        self.layer_name = layer_name
        self._feature: torch.Tensor | None = None
        self._handle = modules[layer_name].register_forward_hook(self._hook)

    def _hook(self, _module, _inp, out):
        self._feature = out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def close(self) -> None:
        self._handle.remove()

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        self._feature = None
        _ = self.model(x)
        if self._feature is None:
            raise RuntimeError(f"Hook did not capture layer {self.layer_name}")
        h = self._feature
        if h.dim() > 2:
            h = h.mean(dim=tuple(range(2, h.dim())))
        return h.flatten(1)


def select_clean_correct(
    model: nn.Module,
    dataset,
    n_images: int,
    device: torch.device,
    batch_size: int,
    max_scan: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    idx = 0
    while idx < len(dataset) and (max_scan <= 0 or idx < max_scan) and len(rows) < n_images:
        end = min(idx + batch_size, len(dataset))
        batch = [dataset[j] for j in range(idx, end)]
        x = torch.stack([b[0] for b in batch]).to(device)
        y = torch.tensor([int(b[1]) for b in batch], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(1)
            m = top_other_margin(logits, y)
            ce = F.cross_entropy(logits, y, reduction="none")
        for off, yy in enumerate(y.detach().cpu().numpy()):
            if int(pred[off].item()) == int(yy):
                rows.append(
                    {
                        "image_ord": len(rows),
                        "dataset_idx": idx + off,
                        "label": int(yy),
                        "clean_pred": int(pred[off].item()),
                        "clean_margin": float(m[off].item()),
                        "clean_ce": float(ce[off].item()),
                    }
                )
                if len(rows) >= n_images:
                    break
        idx = end
    if len(rows) < n_images:
        raise RuntimeError(f"Only found {len(rows)} clean-correct images, requested {n_images}.")
    return pd.DataFrame(rows)


def load_images_from_rows(dataset, rows: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for r in rows.itertuples(index=False):
        x, y = dataset[int(r.dataset_idx)]
        if int(y) != int(r.label):
            raise RuntimeError(f"label mismatch at dataset_idx={r.dataset_idx}")
        xs.append(x)
        ys.append(int(y))
    return torch.stack(xs).to(device), torch.tensor(ys, dtype=torch.long, device=device)


def run_apgd(
    model: nn.Module,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    *,
    eps: float,
    loss: str,
    n_iter: int,
    restarts: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    from autoattack.autopgd_pt import APGDAttack

    attack = APGDAttack(
        model,
        n_iter=n_iter,
        norm="Linf",
        n_restarts=restarts,
        eps=eps,
        seed=seed,
        loss=loss,
        verbose=False,
        device=str(device),
    )
    adv_batches = []
    for start in range(0, len(x_all), batch_size):
        xb = x_all[start : start + batch_size]
        yb = y_all[start : start + batch_size]
        adv = attack.perturb(xb, yb)
        if isinstance(adv, tuple):
            adv = adv[1]
        adv_batches.append(adv.detach())
    return torch.cat(adv_batches, dim=0)


def summarize_attack(
    model_name: str,
    method: str,
    loss: str,
    n_iter: int,
    restarts: int,
    rows: pd.DataFrame,
    model: nn.Module,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    adv_all: torch.Tensor,
    runtime_s: float,
) -> pd.DataFrame:
    clean_pred, clean_margin, clean_ce = logits_stats(model, x_all, y_all)
    pred, final_margin, final_ce = logits_stats(model, adv_all, y_all)
    linf = (adv_all - x_all).abs().flatten(1).max(1).values.detach().cpu().numpy()
    out = rows[["image_ord", "dataset_idx", "label"]].copy()
    out.insert(0, "model", model_name)
    out.insert(1, "method", method)
    out["loss"] = loss
    out["n_iter"] = int(n_iter)
    out["restarts"] = int(restarts)
    out["clean_pred"] = clean_pred.astype(int)
    out["clean_margin"] = clean_margin.astype(float)
    out["clean_ce"] = clean_ce.astype(float)
    out["final_pred"] = pred.astype(int)
    out["success"] = (pred != y_all.detach().cpu().numpy()).astype(int)
    out["final_margin"] = final_margin.astype(float)
    out["margin_drop"] = clean_margin.astype(float) - final_margin.astype(float)
    out["final_ce"] = final_ce.astype(float)
    out["linf"] = linf.astype(float)
    out["runtime_s_total"] = float(runtime_s)
    return out


def random_signs(shape: tuple[int, ...], gen: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.where(
        torch.rand(shape, generator=gen, device=device) < 0.5,
        -torch.ones(shape, device=device),
        torch.ones(shape, device=device),
    )


def feature_jvp(wrapper: HookedModel, x: torch.Tensor, tangent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    def f(inp: torch.Tensor) -> torch.Tensor:
        return wrapper.feature(inp)

    return torch.autograd.functional.jvp(f, x.detach(), tangent, create_graph=False, strict=False)


def candidate_direction_diagnostics(
    model_name: str,
    wrapper: HookedModel,
    rows: pd.DataFrame,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    *,
    eps: float,
    probe_eps: float,
    directions: int,
    direction_batch_size: int,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    records: list[dict] = []
    gen = torch.Generator(device=device).manual_seed(seed)
    for image_i, r in enumerate(rows.itertuples(index=False)):
        x0 = x_all[image_i : image_i + 1]
        y = y_all[image_i : image_i + 1]
        with torch.no_grad():
            h0 = wrapper.feature(x0)
            logits0 = wrapper(x0)
            m0 = top_other_margin(logits0, y)
            ce0 = F.cross_entropy(logits0, y, reduction="none")
        done = 0
        while done < directions:
            b = min(direction_batch_size, directions - done)
            x_rep = x0.repeat(b, 1, 1, 1)
            y_rep = y.repeat(b)
            d = random_signs(tuple(x_rep.shape), gen, device)
            probe_tangent = probe_eps * d
            _, jvp = feature_jvp(wrapper, x_rep, probe_tangent)
            with torch.no_grad():
                x_probe = project_linf(x_rep + probe_tangent, x_rep, probe_eps)
                h_probe = wrapper.feature(x_probe)
                fd = h_probe - h0.repeat(b, 1)
                x_full = project_linf(x_rep + eps * d, x_rep, eps)
                logits_full = wrapper(x_full)
                pred_full = logits_full.argmax(1)
                m_full = top_other_margin(logits_full, y_rep)
                ce_full = F.cross_entropy(logits_full, y_rep, reduction="none")
            fd_norm = fd.flatten(1).norm(dim=1)
            jvp_norm = jvp.flatten(1).norm(dim=1)
            nonlinear = (fd - jvp).flatten(1).norm(dim=1) / fd_norm.clamp_min(1e-12)
            cos = (fd.flatten(1) * jvp.flatten(1)).sum(dim=1) / (fd_norm * jvp_norm).clamp_min(1e-12)
            for j in range(b):
                records.append(
                    {
                        "model": model_name,
                        "image_ord": int(r.image_ord),
                        "dataset_idx": int(r.dataset_idx),
                        "label": int(r.label),
                        "direction_idx": int(done + j),
                        "probe_eps": float(probe_eps),
                        "full_eps": float(eps),
                        "fd_mobility": float(fd_norm[j].item()),
                        "jvp_gain": float(jvp_norm[j].item()),
                        "fd_jvp_cos": float(cos[j].item()),
                        "nonlinear_ratio": float(nonlinear[j].item()),
                        "clean_margin": float(m0.item()),
                        "full_margin": float(m_full[j].item()),
                        "margin_drop": float((m0 - m_full[j]).item()),
                        "clean_ce": float(ce0.item()),
                        "full_ce": float(ce_full[j].item()),
                        "success": int(pred_full[j].item() != int(r.label)),
                        "score_mobility_x_margin": float(fd_norm[j].item() * max(float((m0 - m_full[j]).item()), 0.0)),
                        "score_jvp_x_margin": float(jvp_norm[j].item() * max(float((m0 - m_full[j]).item()), 0.0)),
                    }
                )
            done += b
        if (image_i + 1) % 25 == 0:
            print(f"[{model_name}] candidate diagnostics {image_i + 1}/{len(rows)} images", flush=True)
    return pd.DataFrame(records)


def safe_auc(y: Iterable[int], score: Iterable[float]) -> float:
    yy = np.asarray(list(y), dtype=int)
    ss = np.asarray(list(score), dtype=float)
    if len(np.unique(yy)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(yy, ss))
    except ValueError:
        return float("nan")


def summarize_candidates(df: pd.DataFrame, topk_values: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    selector_rows = []
    for model_name, sub in df.groupby("model"):
        metric_rows.append(
            {
                "model": model_name,
                "n_images": int(sub.image_ord.nunique()),
                "n_candidates": int(len(sub)),
                "candidate_success_rate": float(sub.success.mean()),
                "median_fd_mobility": float(sub.fd_mobility.median()),
                "median_jvp_gain": float(sub.jvp_gain.median()),
                "median_fd_jvp_cos": float(sub.fd_jvp_cos.median()),
                "median_nonlinear_ratio": float(sub.nonlinear_ratio.median()),
                "spearman_mobility_jvp": float(sub[["fd_mobility", "jvp_gain"]].corr(method="spearman").iloc[0, 1]),
                "spearman_mobility_margin_drop": float(sub[["fd_mobility", "margin_drop"]].corr(method="spearman").iloc[0, 1]),
                "spearman_jvp_margin_drop": float(sub[["jvp_gain", "margin_drop"]].corr(method="spearman").iloc[0, 1]),
                "auc_mobility_for_success": safe_auc(sub.success, sub.fd_mobility),
                "auc_jvp_for_success": safe_auc(sub.success, sub.jvp_gain),
                "auc_margin_drop_for_success": safe_auc(sub.success, sub.margin_drop),
                "auc_mobility_x_margin_for_success": safe_auc(sub.success, sub.score_mobility_x_margin),
                "auc_jvp_x_margin_for_success": safe_auc(sub.success, sub.score_jvp_x_margin),
            }
        )
        selectors = {
            "random": None,
            "fd_mobility": "fd_mobility",
            "jvp_gain": "jvp_gain",
            "margin_drop": "margin_drop",
            "mobility_x_margin": "score_mobility_x_margin",
            "jvp_x_margin": "score_jvp_x_margin",
        }
        for k in topk_values:
            for selector, col in selectors.items():
                successes = []
                margins = []
                for _image_ord, g in sub.groupby("image_ord"):
                    if selector == "random":
                        take = g.sort_values("direction_idx").head(k)
                    else:
                        take = g.sort_values(col, ascending=False).head(k)
                    successes.append(int(take.success.max()))
                    margins.append(float(take.margin_drop.max()))
                selector_rows.append(
                    {
                        "model": model_name,
                        "selector": selector,
                        "top_k": int(k),
                        "n_images": int(len(successes)),
                        "topk_asr": float(np.mean(successes)),
                        "mean_best_margin_drop": float(np.mean(margins)),
                        "median_best_margin_drop": float(np.median(margins)),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(selector_rows)


def summarize_attack_table(per_image: pd.DataFrame) -> pd.DataFrame:
    if per_image.empty:
        return pd.DataFrame()
    return (
        per_image.groupby(["model", "method", "loss", "n_iter", "restarts"], dropna=False)
        .agg(
            n=("success", "size"),
            asr=("success", "mean"),
            mean_margin_drop=("margin_drop", "mean"),
            median_margin_drop=("margin_drop", "median"),
            max_linf=("linf", "max"),
            runtime_s_total=("runtime_s_total", "first"),
        )
        .reset_index()
        .sort_values(["model", "method", "loss", "n_iter"])
    )


def write_findings(out: Path, attack_summary: pd.DataFrame, metrics: pd.DataFrame, topk: pd.DataFrame) -> None:
    lines = [
        "# RobustBench Hidden-Jacobian Pilot Findings",
        "",
        "This pilot uses official RobustBench CIFAR-10 Linf models and evaluates the layer4 hidden-Jacobian mobility diagnostic on clean-correct images.",
        "",
        "## Official APGD Baselines",
        "",
    ]
    if attack_summary.empty:
        lines.append("APGD baselines were not run.")
    else:
        for r in attack_summary.itertuples(index=False):
            lines.append(
                f"- {r.model} {r.method} ({r.loss}, {int(r.n_iter)} iters): ASR {100*float(r.asr):.1f}%, "
                f"median margin drop {float(r.median_margin_drop):.3f}."
            )
    lines += ["", "## Mobility/JVP Diagnostics", ""]
    for r in metrics.itertuples(index=False):
        lines.append(
            f"- {r.model}: median finite-difference mobility {float(r.median_fd_mobility):.4f}, "
            f"median JVP gain {float(r.median_jvp_gain):.4f}, median FD/JVP cosine {float(r.median_fd_jvp_cos):.3f}, "
            f"Spearman(mobility,JVP) {float(r.spearman_mobility_jvp):.3f}, "
            f"AUC(JVP -> candidate success) {float(r.auc_jvp_for_success):.3f}."
        )
    lines += ["", "## Candidate Selection", ""]
    show = topk[topk.top_k.isin([1, 5, 10])].copy()
    for r in show.itertuples(index=False):
        lines.append(
            f"- {r.model} selector={r.selector} top-{int(r.top_k)}: one-step candidate ASR {100*float(r.topk_asr):.1f}%, "
            f"median best margin drop {float(r.median_best_margin_drop):.3f}."
        )
    lines += [
        "",
        "Interpretation guardrail: this pilot tests whether robust models still expose high-mobility hidden directions and whether simple state-conditioned selectors identify adversarially useful candidate moves. It is not a full robust-model road-routing benchmark by itself.",
    ]
    (out / "robustbench_findings.md").write_text("\n".join(lines) + "\n")


def run_model(args, model_name: str, dataset, device: torch.device) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_out = out / safe_name(model_name)
    model_out.mkdir(parents=True, exist_ok=True)
    print(f"[load] {model_name}", flush=True)
    model = load_robustbench_model(
        model_name=model_name,
        dataset="cifar10",
        threat_model="Linf",
        model_dir=args.model_dir,
    ).to(device).eval()
    wrapper = HookedModel(model, args.layer)
    try:
        selection_csv = model_out / "robustbench_image_selection.csv"
        if selection_csv.exists() and not args.overwrite:
            rows = pd.read_csv(selection_csv)
            print(f"[{model_name}] loaded existing image selection: {len(rows)}", flush=True)
        else:
            rows = select_clean_correct(model, dataset, args.images, device, args.batch_size, args.max_scan)
            rows.to_csv(selection_csv, index=False)
            print(f"[{model_name}] selected {len(rows)} clean-correct images", flush=True)
        x_all, y_all = load_images_from_rows(dataset, rows, device)

        per_attack_rows = []
        if not args.skip_apgd:
            for loss in parse_csv(args.apgd_losses):
                for n_iter in parse_int_csv(args.apgd_iters):
                    method = f"official_apgd_{loss}_s{n_iter}_r{args.apgd_restarts}"
                    fpath = model_out / f"{safe_name(method)}_per_image.csv"
                    if fpath.exists() and not args.overwrite:
                        per_attack_rows.extend(pd.read_csv(fpath).to_dict("records"))
                        print(f"[{model_name}] skip {method}", flush=True)
                        continue
                    t0 = time.perf_counter()
                    adv = run_apgd(
                        model,
                        x_all,
                        y_all,
                        eps=args.eps / 255.0,
                        loss=loss,
                        n_iter=n_iter,
                        restarts=args.apgd_restarts,
                        batch_size=args.attack_batch_size,
                        seed=args.seed,
                        device=device,
                    )
                    runtime = time.perf_counter() - t0
                    df = summarize_attack(
                        model_name,
                        method,
                        loss,
                        n_iter,
                        args.apgd_restarts,
                        rows,
                        model,
                        x_all,
                        y_all,
                        adv,
                        runtime,
                    )
                    df.to_csv(fpath, index=False)
                    per_attack_rows.extend(df.to_dict("records"))
                    print(f"[{model_name}] done {method} runtime={runtime:.1f}s", flush=True)

        candidate_csv = model_out / "candidate_level_mobility_jvp.csv"
        if candidate_csv.exists() and not args.overwrite:
            cand = pd.read_csv(candidate_csv)
            print(f"[{model_name}] loaded existing candidate diagnostics", flush=True)
        else:
            cand = candidate_direction_diagnostics(
                model_name,
                wrapper,
                rows,
                x_all,
                y_all,
                eps=args.eps / 255.0,
                probe_eps=args.probe_eps / 255.0,
                directions=args.directions,
                direction_batch_size=args.direction_batch_size,
                seed=args.seed + 100_003,
                device=device,
            )
            cand.to_csv(candidate_csv, index=False)

        attack_per_image = pd.DataFrame(per_attack_rows)
        attack_per_image.to_csv(model_out / "robustbench_apgd_per_image.csv", index=False)
        attack_summary = summarize_attack_table(attack_per_image)
        attack_summary.to_csv(model_out / "robustbench_baseline_summary.csv", index=False)
        metrics, topk = summarize_candidates(cand, parse_int_csv(args.topk))
        metrics.to_csv(model_out / "robustbench_mobility_jvp_summary.csv", index=False)
        topk.to_csv(model_out / "robustbench_selector_summary.csv", index=False)
        write_findings(model_out, attack_summary, metrics, topk)
        (model_out / "metadata.json").write_text(
            json.dumps(
                {
                    "model": model_name,
                    "dataset": "cifar10",
                    "threat_model": "Linf",
                    "model_dir": args.model_dir,
                    "data_dir": args.data_dir,
                    "layer": args.layer,
                    "images": int(args.images),
                    "eps_255": float(args.eps),
                    "probe_eps_255": float(args.probe_eps),
                    "directions": int(args.directions),
                    "direction_batch_size": int(args.direction_batch_size),
                    "apgd_losses": parse_csv(args.apgd_losses),
                    "apgd_iters": parse_int_csv(args.apgd_iters),
                    "apgd_restarts": int(args.apgd_restarts),
                    "seed": int(args.seed),
                    "device": str(device),
                },
                indent=2,
            )
        )
    finally:
        wrapper.close()


def combine_outputs(args) -> None:
    out = Path(args.output_dir)
    attack, metrics, topk, selections = [], [], [], []
    for model_name in parse_csv(args.models):
        model_out = out / safe_name(model_name)
        for fname, bucket in [
            ("robustbench_baseline_summary.csv", attack),
            ("robustbench_mobility_jvp_summary.csv", metrics),
            ("robustbench_selector_summary.csv", topk),
            ("robustbench_image_selection.csv", selections),
        ]:
            path = model_out / fname
            if path.exists():
                df = pd.read_csv(path)
                if "model" not in df.columns:
                    df.insert(0, "model", model_name)
                bucket.append(df)
    if attack:
        pd.concat(attack, ignore_index=True).to_csv(out / "robustbench_baseline_summary.csv", index=False)
    if metrics:
        pd.concat(metrics, ignore_index=True).to_csv(out / "robustbench_mobility_jvp_summary.csv", index=False)
    if topk:
        pd.concat(topk, ignore_index=True).to_csv(out / "robustbench_selector_summary.csv", index=False)
    if selections:
        pd.concat(selections, ignore_index=True).to_csv(out / "robustbench_image_selection.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/hidden_jacobian_routing/robustbench_hidden_jacobian_pilot_c200")
    p.add_argument("--models", default="Wong2020Fast,Engstrom2019Robustness,Addepalli2022Efficient_RN18")
    p.add_argument("--data-dir", default="data/cifar10")
    p.add_argument("--model-dir", default="models/robustbench")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--max-scan", type=int, default=5000)
    p.add_argument("--layer", default="layer4")
    p.add_argument("--eps", type=float, default=8.0)
    p.add_argument("--probe-eps", type=float, default=0.25)
    p.add_argument("--directions", type=int, default=64)
    p.add_argument("--direction-batch-size", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--attack-batch-size", type=int, default=50)
    p.add_argument("--skip-apgd", action="store_true")
    p.add_argument("--apgd-losses", default="ce,dlr")
    p.add_argument("--apgd-iters", default="20")
    p.add_argument("--apgd-restarts", type=int, default=1)
    p.add_argument("--topk", default="1,5,10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.CIFAR10(args.data_dir, train=False, download=False, transform=transforms.ToTensor())
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "run_metadata.json").write_text(
        json.dumps(
            {
                "start_time": start_time,
                "command": " ".join(sys.argv),
                "device": str(device),
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "args": vars(args),
            },
            indent=2,
        )
    )

    for model_name in parse_csv(args.models):
        run_model(args, model_name, dataset, device)
        combine_outputs(args)
    combine_outputs(args)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
