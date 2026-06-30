#!/usr/bin/env python3
"""Evaluate image-conditioned Jacobian-pullback initializations for Square attack."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import datasets, models, transforms


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class ResNet18WithFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.base = models.resnet18(pretrained=True)

    def forward_with_feature(self, x, layer: str):
        z = self.normalize(x)
        z = self.base.conv1(z)
        z = self.base.bn1(z)
        z = self.base.relu(z)
        z = self.base.maxpool(z)
        z = self.base.layer1(z)
        z = self.base.layer2(z)
        z = self.base.layer3(z)
        z = self.base.layer4(z)
        pooled = self.base.avgpool(z).flatten(1)
        logits = self.base.fc(pooled)
        if layer in {"clf_avgpool", "clf_layer4"}:
            return logits, pooled
        if layer == "clf_logits":
            return logits, logits
        raise ValueError(layer)


def load_target(name: str, device):
    if name == "densenet121":
        base = models.densenet121(pretrained=True)
    elif name == "vgg16":
        base = models.vgg16_bn(pretrained=True)
    else:
        raise ValueError(name)
    return nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), base).to(device).eval()


def normalize_rows(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def softmax_np(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def fit_pca_predictors(manifest_path: str, layers: list[str], ks: list[int], seed: int, ridge_alpha: float):
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[(manifest["trajectory_features_npz"].notna()) & (manifest["success"].astype(int) == 1)].copy()
    max_k = max(ks)
    rng = np.random.default_rng(seed)
    bases = {}
    models_by_layer = {}
    diagnostics = []

    for layer in layers:
        all_segments = []
        runs = []
        for _idx, row in manifest.iterrows():
            z = np.load(row["trajectory_features_npz"])
            feats = z[layer].astype(np.float64)
            logits = z["clf_logits"].astype(np.float64)
            segments = []
            for t in range(len(feats) - 1):
                v = feats[t + 1] - feats[t]
                if np.linalg.norm(v) > 1e-12:
                    segments.append(v)
            if not segments:
                continue
            seg = normalize_rows(np.stack(segments))
            all_segments.append(seg)
            probs = softmax_np(logits[1:])[:, int(row["target_class"])]
            runs.append({
                "run_name": row["run_name"],
                "target_class": int(row["target_class"]),
                "input_feature": feats[-1].copy(),
                "segments": seg,
                "weights": probs[: len(seg)],
            })

        x = np.concatenate(all_segments, axis=0)
        x = x[rng.permutation(len(x))]
        mean = x.mean(axis=0, keepdims=True)
        xc = x - mean
        _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
        basis = vt[:max_k].astype(np.float32)
        bases[layer] = {
            "mean": mean.astype(np.float32),
            "basis": torch.from_numpy(basis),
        }

        x_train = np.stack([r["input_feature"] for r in runs]).astype(np.float64)
        models_by_layer[layer] = {}
        for k in ks:
            kk = min(k, basis.shape[0])
            targets = {"pred_mean": [], "pred_final": [], "pred_conf_weighted": []}
            for r in runs:
                coeff = (r["segments"] - mean) @ basis[:kk].T
                weights = np.clip(r["weights"].astype(np.float64), 1e-12, None)
                weights = weights / weights.sum()
                targets["pred_mean"].append(coeff.mean(axis=0))
                targets["pred_final"].append(coeff[-1])
                targets["pred_conf_weighted"].append((coeff * weights[:, None]).sum(axis=0))
            models_by_layer[layer][k] = {}
            for name, y_list in targets.items():
                y_train = np.stack(y_list).astype(np.float64)
                model = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=ridge_alpha, random_state=seed),
                )
                model.fit(x_train, y_train)
                score = float(model.score(x_train, y_train))
                models_by_layer[layer][k][name] = model
                diagnostics.append({
                    "layer": layer,
                    "k": int(k),
                    "predictor": name,
                    "n_train_runs": len(runs),
                    "n_train_segments": int(len(x)),
                    "train_r2": score,
                })
    return bases, models_by_layer, pd.DataFrame(diagnostics)


def make_pullback_dirs(source, clean, layer: str, basis: torch.Tensor):
    clean_req = clean.detach().clone().requires_grad_(True)
    _logits, h = source.forward_with_feature(clean_req, layer)
    dirs = []
    for j in range(basis.shape[0]):
        grad = torch.autograd.grad((h * basis[j:j + 1].to(clean.device)).sum(), clean_req, retain_graph=True)[0].detach()
        grad = grad / grad.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        dirs.append(grad)
    return torch.cat(dirs, dim=0)


def project_delta(delta, eps):
    return delta.clamp(-eps, eps)


def to_eps_boundary(delta, eps):
    maxabs = delta.abs().max().clamp_min(1e-12)
    return (delta / maxabs * eps).clamp(-eps, eps)


def loss_value(model, x, y):
    with torch.no_grad():
        logits = model(x)
        return float(F.cross_entropy(logits, y).item()), int(logits.argmax(1).item())


def square_attack_from_delta(model, clean, y, init_delta, args, gen, milestones):
    delta = project_delta(init_delta.clone(), args.eps)
    clean_pred = int(model(clean).argmax(1).item())
    if clean_pred != int(y.item()):
        return {"clean_correct": 0, "success": 0, "queries": 0, "final_pred": clean_pred, **{f"success_at_q{q}": 0 for q in milestones}}
    best_loss, pred = loss_value(model, (clean + delta).clamp(0, 1), y)
    queries = 1
    first_success_q = queries if pred != int(y.item()) else -1
    c, h, w = delta.shape[1:]
    for q in range(2, args.max_queries + 1):
        side = max(args.min_square, int(round(args.square_frac * h * (1 - (q / max(args.max_queries, 1))) + args.min_square)))
        side = min(side, h, w)
        top = int(torch.randint(0, h - side + 1, (1,), generator=gen, device=clean.device).item())
        left = int(torch.randint(0, w - side + 1, (1,), generator=gen, device=clean.device).item())
        candidate = delta.clone()
        patch = torch.rand((1, c, side, side), generator=gen, device=clean.device) * (2.0 * args.eps) - args.eps
        candidate[:, :, top:top + side, left:left + side] = patch
        candidate = project_delta(candidate, args.eps)
        cand_loss, cand_pred = loss_value(model, (clean + candidate).clamp(0, 1), y)
        queries = q
        if cand_loss >= best_loss:
            delta = candidate
            best_loss = cand_loss
            pred = cand_pred
        if pred != int(y.item()) and first_success_q < 0:
            first_success_q = queries
            break
    return {
        "clean_correct": 1,
        "success": int(first_success_q > 0),
        "queries": int(first_success_q if first_success_q > 0 else args.max_queries),
        "final_pred": int(pred),
        **{f"success_at_q{q}": int(first_success_q > 0 and first_success_q <= q) for q in milestones},
    }


def bootstrap_ci(values, seed=0, reps=5000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(reps)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_target(args, target_name, source, target, bases, models_by_layer, dataset, indices, layers, ks, milestones, device):
    rows = []
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 4242 + abs(hash(target_name)) % 10000)
    started = time.time()
    predictor_names = ["pred_mean", "pred_final", "pred_conf_weighted"]

    for image_i, idx in enumerate(indices, start=1):
        clean_cpu, label_int = dataset[int(idx)]
        clean = clean_cpu.unsqueeze(0).to(device)
        y = torch.tensor([int(label_int)], device=device)

        rand_delta = to_eps_boundary(torch.randn(clean.shape, generator=gen, device=device), args.eps)
        res = square_attack_from_delta(target, clean, y, rand_delta, args, gen, milestones)
        rows.append({"target_model": target_name, "dataset_idx": int(idx), "source_class": int(label_int), "method": "random_init", "layer": "", "k": 0, **res})

        for layer in layers:
            with torch.no_grad():
                _src_logits, clean_h = source.forward_with_feature(clean, layer)
            clean_feat = clean_h.detach().cpu().numpy().astype(np.float64)
            basis = bases[layer]["basis"].to(device)
            pull = make_pullback_dirs(source, clean, layer, basis)
            for k in ks:
                alpha = torch.randn(k, generator=gen, device=device)
                delta = (alpha.view(-1, 1, 1, 1) * pull[:k]).sum(dim=0, keepdim=True)
                delta = to_eps_boundary(delta, args.eps)
                res = square_attack_from_delta(target, clean, y, delta, args, gen, milestones)
                rows.append({"target_model": target_name, "dataset_idx": int(idx), "source_class": int(label_int), "method": "global_pullback", "layer": layer, "k": k, **res})

                for predictor in predictor_names:
                    pred_alpha = models_by_layer[layer][k][predictor].predict(clean_feat)[0]
                    alpha_t = torch.from_numpy(pred_alpha.astype(np.float32)).to(device)
                    delta = (alpha_t.view(-1, 1, 1, 1) * pull[:k]).sum(dim=0, keepdim=True)
                    delta = to_eps_boundary(delta, args.eps)
                    res = square_attack_from_delta(target, clean, y, delta, args, gen, milestones)
                    rows.append({"target_model": target_name, "dataset_idx": int(idx), "source_class": int(label_int), "method": predictor, "layer": layer, "k": k, **res})
        if image_i % 10 == 0:
            print(f"[{target_name}] images={image_i}/{len(indices)} elapsed={time.time() - started:.1f}s", flush=True)
    return rows


def summarize_and_test(df, milestones, seed):
    df_eval = df[df["clean_correct"].astype(int) == 1].copy()
    summary = df_eval.groupby(["target_model", "method", "layer", "k"]).agg(
        n=("success", "size"),
        asr=("success", "mean"),
        mean_queries=("queries", "mean"),
        median_queries=("queries", "median"),
        **{f"asr_at_q{m}": (f"success_at_q{m}", "mean") for m in milestones},
    ).reset_index()

    tests = []
    success_metrics = ["success"] + [f"success_at_q{m}" for m in milestones if f"success_at_q{m}" in df_eval.columns]
    for target_model, target_df in df_eval.groupby("target_model"):
        base = target_df[target_df["method"] == "random_init"].set_index("dataset_idx")
        for (_method, layer, k), group in target_df[target_df["method"] != "random_init"].groupby(["method", "layer", "k"]):
            method = str(group["method"].iloc[0])
            common = base.index.intersection(group["dataset_idx"])
            g = group.set_index("dataset_idx").loc[common]
            b = base.loc[g.index]
            for metric in success_metrics:
                diff = g[metric].to_numpy(dtype=float) - b[metric].to_numpy(dtype=float)
                lo, hi = bootstrap_ci(diff, seed)
                pval = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                tests.append({
                    "target_model": target_model,
                    "comparison": f"{method}_vs_random",
                    "method": method,
                    "layer": layer,
                    "k": int(k),
                    "metric": metric,
                    "mean_diff": float(diff.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "wilcoxon_p": pval,
                })
            succ_mask = (g["success"].astype(int) == 1) & (b["success"].astype(int) == 1)
            if succ_mask.any():
                diff = g.loc[succ_mask, "queries"].to_numpy(dtype=float) - b.loc[succ_mask, "queries"].to_numpy(dtype=float)
                lo, hi = bootstrap_ci(diff, seed)
                pval = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                tests.append({
                    "target_model": target_model,
                    "comparison": f"{method}_vs_random",
                    "method": method,
                    "layer": layer,
                    "k": int(k),
                    "metric": "queries_success_only",
                    "mean_diff": float(diff.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "wilcoxon_p": pval,
                })
    return summary, pd.DataFrame(tests)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    p.add_argument("--trajectory-manifest", default="analysis_outputs/pure_af_geometry/random_ga_classifier_trajectories_resnet18_c10_s5/manifest.csv")
    p.add_argument("--indices-metadata", default="analysis_outputs/pure_af_geometry/away_from_true_attractor_resnet18_all_available_c10.csv.metadata.json")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_pullback_nes/init_square_predictor")
    p.add_argument("--target-models", default="densenet121,vgg16")
    p.add_argument("--max-images", type=int, default=50)
    p.add_argument("--layers", default="clf_logits,clf_avgpool")
    p.add_argument("--ks", default="10,50")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--max-queries", type=int, default=100)
    p.add_argument("--milestones", default="1,10,50,100")
    p.add_argument("--square-frac", type=float, default=0.35)
    p.add_argument("--min-square", type=int, default=8)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    milestones = [int(x) for x in args.milestones.split(",") if x.strip()]
    target_names = [x.strip() for x in args.target_models.split(",") if x.strip()]

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    indices = json.loads(Path(args.indices_metadata).read_text())["indices"][: args.max_images]

    source = ResNet18WithFeatures().to(device).eval()
    bases, models_by_layer, diagnostics = fit_pca_predictors(args.trajectory_manifest, layers, ks, args.seed, args.ridge_alpha)
    diagnostics.to_csv(out_dir / "predictor_diagnostics.csv", index=False)

    rows = []
    started = time.time()
    for target_name in target_names:
        target = load_target(target_name, device)
        rows.extend(evaluate_target(args, target_name, source, target, bases, models_by_layer, dataset, indices, layers, ks, milestones, device))
        del target
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = out_dir / "pullback_init_square_predictor_results.csv"
    fieldnames = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    summary, tests = summarize_and_test(df, milestones, args.seed)
    summary_path = out_dir / "pullback_init_square_predictor_summary.csv"
    tests_path = out_dir / "pullback_init_square_predictor_paired_tests.csv"
    summary.to_csv(summary_path, index=False)
    tests.to_csv(tests_path, index=False)
    (out_dir / "metadata.json").write_text(json.dumps({
        "args": vars(args),
        "indices": indices,
        "elapsed_sec": time.time() - started,
        "train_input_note": "Predictors are trained from final GA-pure source features to aggregated PCA coefficients from the same successful GA trajectory.",
    }, indent=2))

    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nPAIRED TESTS")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
