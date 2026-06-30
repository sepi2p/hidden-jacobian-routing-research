#!/usr/bin/env python3
"""Compare ordinary CIFAR model similarity with success-flow geometry similarity."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import (  # noqa: E402
    KS,
    CIFARFeatureWrapper,
    fit_basis,
    get_npz,
    load_model,
    normalize_rows,
    projection_energy,
)


MODELS = ["bbb_resnet50", "bbb_vgg19_bn", "bbb_densenet", "bbb_inception_v3"]

STAGE_MAP = {
    "stage1": {
        "bbb_resnet50": "layer1",
        "bbb_vgg19_bn": "block1",
        "bbb_densenet": "denseblock1",
        "bbb_inception_v3": "mixed5",
    },
    "stage2": {
        "bbb_resnet50": "layer2",
        "bbb_vgg19_bn": "block2",
        "bbb_densenet": "denseblock2",
        "bbb_inception_v3": "mixed6",
    },
    "stage3": {
        "bbb_resnet50": "layer3",
        "bbb_vgg19_bn": "block3",
        "bbb_densenet": "denseblock3",
        "bbb_inception_v3": "mixed7",
    },
    "stage4": {
        "bbb_resnet50": "layer4",
        "bbb_vgg19_bn": "block4",
        "bbb_densenet": "penultimate",
        "bbb_inception_v3": "penultimate",
    },
    "penultimate": {
        "bbb_resnet50": "avgpool",
        "bbb_vgg19_bn": "penultimate",
        "bbb_densenet": "penultimate",
        "bbb_inception_v3": "penultimate",
    },
    "logits": {
        "bbb_resnet50": "logits",
        "bbb_vgg19_bn": "logits",
        "bbb_densenet": "logits",
        "bbb_inception_v3": "logits",
    },
}


def center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = center(x.astype(np.float64))
    y = center(y.astype(np.float64))
    xy = np.linalg.norm(x.T @ y, ord="fro") ** 2
    xx = np.linalg.norm(x.T @ x, ord="fro")
    yy = np.linalg.norm(y.T @ y, ord="fro")
    return float(xy / np.clip(xx * yy, 1e-12, None))


def corr_flat(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        return np.nan
    a = x.reshape(-1)
    b = y.reshape(-1)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def pca_reduce(x: np.ndarray, max_components: int = 50, var_keep: float = 0.99) -> np.ndarray:
    x = center(x.astype(np.float64))
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    var = s**2
    csum = np.cumsum(var / np.clip(var.sum(), 1e-12, None))
    k = int(np.searchsorted(csum, var_keep) + 1)
    k = max(1, min(k, max_components, vt.shape[0], x.shape[0] - 1))
    return x @ vt[:k].T


def svcca_similarity(x: np.ndarray, y: np.ndarray) -> float:
    n = min(len(x), len(y))
    if n < 4:
        return np.nan
    xr = pca_reduce(x[:n], max_components=min(50, n - 2))
    yr = pca_reduce(y[:n], max_components=min(50, n - 2))
    k = min(xr.shape[1], yr.shape[1], n - 2)
    if k < 1:
        return np.nan
    cca = CCA(n_components=k, max_iter=1000)
    try:
        x_c, y_c = cca.fit_transform(xr, yr)
    except Exception:
        return np.nan
    vals = []
    for i in range(k):
        if np.std(x_c[:, i]) <= 1e-12 or np.std(y_c[:, i]) <= 1e-12:
            continue
        vals.append(np.corrcoef(x_c[:, i], y_c[:, i])[0, 1])
    return float(np.mean(vals)) if vals else np.nan


def sample_space_basis(x: np.ndarray, k: int) -> np.ndarray:
    x = normalize_rows(x)
    x = center(x)
    u, _s, _vt = np.linalg.svd(x, full_matrices=False)
    kk = min(k, u.shape[1])
    return u[:, :kk]


def subspace_metrics_from_sample_basis(a: np.ndarray, b: np.ndarray, k: int) -> dict:
    kk = min(k, a.shape[1], b.shape[1])
    if kk < 1:
        return {}
    s = np.linalg.svd(a[:, :kk].T @ b[:, :kk], compute_uv=False)
    s = np.clip(s, 0, 1)
    angles = np.arccos(s)
    return {
        "k": int(k),
        "mean_principal_angle_deg": float(np.degrees(angles).mean()),
        "max_principal_angle_deg": float(np.degrees(angles).max()),
        "projection_overlap": float(np.sum(s**2) / kk),
        "grassmann_distance": float(np.linalg.norm(angles)),
        "subspace_affinity": float(np.sqrt(np.sum(s**2) / kk)),
    }


def segment_key(run_id: str, target_class: int, start_generation: int, end_generation: int) -> tuple:
    seed = run_id.rsplit("_seed", 1)[-1]
    return int(target_class), int(seed), int(start_generation), int(end_generation)


def collect_common_clean(args, device):
    dataset = datasets.CIFAR10(args.dataset_root, train=False, download=False, transform=transforms.ToTensor())
    wrappers: dict[str, CIFARFeatureWrapper] = {m: load_model(m, device) for m in MODELS}
    selected, labels = [], []
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        xb = x.unsqueeze(0).to(device)
        ok = True
        with torch.no_grad():
            for wrapper in wrappers.values():
                logits = wrapper(xb)
                if int(logits.argmax(1).item()) != int(y):
                    ok = False
                    break
        if ok:
            selected.append(idx)
            labels.append(int(y))
        if len(selected) >= args.clean_images:
            break
    features = {m: {layer: [] for layer in wrappers[m].labels} for m in MODELS}
    logits_by_model, probs_by_model, preds_by_model, grads_by_model = {}, {}, {}, {}
    labels_arr = np.asarray(labels, dtype=np.int64)

    for model, wrapper in wrappers.items():
        logits_list, probs_list, preds_list, grad_list = [], [], [], []
        for idx, y in zip(selected, labels):
            x, _ = dataset[idx]
            xb = x.unsqueeze(0).to(device)
            with torch.no_grad():
                logits, feats, _raw = wrapper.forward_with_features(xb)
            for layer, h in feats.items():
                features[model][layer].append(h.detach().cpu().numpy()[0].astype(np.float32))
            logits_list.append(logits.detach().cpu().numpy()[0].astype(np.float32))
            probs_list.append(F.softmax(logits, dim=1).detach().cpu().numpy()[0].astype(np.float32))
            preds_list.append(int(logits.argmax(1).item()))

            probe = xb.detach().requires_grad_(True)
            logits_g = wrapper(probe)
            logp = F.log_softmax(logits_g, dim=1)[0, y]
            gx = torch.autograd.grad(logp, probe)[0]
            grad_list.append(gx.detach().flatten().cpu().numpy().astype(np.float32))

        logits_by_model[model] = np.stack(logits_list)
        probs_by_model[model] = np.stack(probs_list)
        preds_by_model[model] = np.asarray(preds_list)
        grads_by_model[model] = np.stack(grad_list)
        for layer, vals in list(features[model].items()):
            if vals:
                features[model][layer] = np.stack(vals)
            else:
                del features[model][layer]
        del wrapper
        torch.cuda.empty_cache()
    return selected, labels_arr, features, logits_by_model, probs_by_model, preds_by_model, grads_by_model


def compute_standard_metrics(args, out_dir, device):
    cache = out_dir / "clean_feature_cache.npz"
    meta_path = out_dir / "clean_feature_cache_metadata.json"
    if cache.exists() and meta_path.exists() and args.reuse_clean_cache:
        npz = np.load(cache)
        with open(meta_path) as f:
            meta = json.load(f)
        selected = meta["indices"]
        features = {}
        for key in npz.files:
            if key.startswith("feat__"):
                _prefix, model, layer = key.split("__", 2)
                features.setdefault(model, {})[layer] = npz[key]
        logits_by_model = {m: npz[f"logits__{m}"] for m in MODELS}
        probs_by_model = {m: npz[f"probs__{m}"] for m in MODELS}
        preds_by_model = {m: npz[f"preds__{m}"] for m in MODELS}
        grads_by_model = {m: npz[f"grads__{m}"] for m in MODELS}
    else:
        selected, labels, features, logits_by_model, probs_by_model, preds_by_model, grads_by_model = collect_common_clean(args, device)
        packed = {}
        for model, layer_map in features.items():
            for layer, arr in layer_map.items():
                packed[f"feat__{model}__{layer}"] = arr
        for model in MODELS:
            packed[f"logits__{model}"] = logits_by_model[model]
            packed[f"probs__{model}"] = probs_by_model[model]
            packed[f"preds__{model}"] = preds_by_model[model]
            packed[f"grads__{model}"] = grads_by_model[model]
        np.savez_compressed(cache, **packed)
        with open(meta_path, "w") as f:
            json.dump({"indices": selected, "n": len(selected), "clean_images_requested": args.clean_images}, f, indent=2)

    rows = []
    for ma, mb in combinations(MODELS, 2):
        prob_corr = corr_flat(probs_by_model[ma], probs_by_model[mb])
        logit_corr = corr_flat(logits_by_model[ma], logits_by_model[mb])
        pred_agree = float(np.mean(preds_by_model[ma] == preds_by_model[mb]))
        grad_cos = float(np.mean(np.sum(normalize_rows(grads_by_model[ma]) * normalize_rows(grads_by_model[mb]), axis=1)))
        for stage, mapping in STAGE_MAP.items():
            la, lb = mapping.get(ma), mapping.get(mb)
            if la not in features.get(ma, {}) or lb not in features.get(mb, {}):
                continue
            xa, xb = features[ma][la], features[mb][lb]
            rows.append({
                "model_a": ma,
                "model_b": mb,
                "stage": stage,
                "layer_a": la,
                "layer_b": lb,
                "n_clean": int(len(xa)),
                "cka": linear_cka(xa, xb),
                "svcca": svcca_similarity(xa, xb),
                "pwcca": np.nan,
                "output_probability_correlation": prob_corr,
                "prediction_agreement": pred_agree,
                "input_gradient_cosine": grad_cos,
                "logit_correlation": logit_corr,
            })
    return pd.DataFrame(rows)


def basis_transfer_auc(pos_train: np.ndarray, pos_test: np.ndarray, neg: np.ndarray, k: int = 20) -> float:
    if len(pos_train) < 5 or len(pos_test) < 3 or len(neg) < 3:
        return np.nan
    if pos_train.shape[1] != pos_test.shape[1] or pos_train.shape[1] != neg.shape[1]:
        return np.nan
    mean, basis = fit_basis(normalize_rows(pos_train), k)
    pe_pos = projection_energy(normalize_rows(pos_test), mean, basis)[f"energy_k{k}"].to_numpy()
    pe_neg = projection_energy(normalize_rows(neg), mean, basis)[f"energy_k{k}"].to_numpy()
    y = np.r_[np.ones(len(pe_pos)), np.zeros(len(pe_neg))]
    score = np.r_[pe_pos, pe_neg]
    return float(roc_auc_score(y, score))


def matched_vectors(seg: pd.DataFrame, npz, model: str, layer: str) -> dict[tuple, np.ndarray]:
    group = seg[(seg["model"] == model) & (seg["layer"] == layer) & (seg["success"] == 1)]
    arr = get_npz(npz, "vectors", model, layer)
    out = {}
    for r in group.itertuples():
        out[segment_key(r.run_id, r.target_class, r.start_generation, r.end_generation)] = arr[int(r.vector_idx)]
    return out


def random_subspace_overlap(n: int, k: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    a, _ = np.linalg.qr(rng.normal(size=(n, k)))
    b, _ = np.linalg.qr(rng.normal(size=(n, k)))
    return subspace_metrics_from_sample_basis(a, b, k)


def compute_flow_similarity(args, out_dir):
    source_dir = Path(args.layerwise_dir)
    seg = pd.read_csv(source_dir / "segments.csv")
    vec_npz = np.load(source_dir / "segment_vectors.npz")
    clean_npz = np.load(source_dir / "clean_vectors.npz")
    grad_npz = np.load(source_dir / "segment_grads.npz")
    pred = pd.read_csv(source_dir / "layerwise_predictiveness_metrics.csv")
    rows = []
    for ma, mb in combinations(MODELS, 2):
        for stage, mapping in STAGE_MAP.items():
            la, lb = mapping.get(ma), mapping.get(mb)
            va = matched_vectors(seg, vec_npz, ma, la)
            vb = matched_vectors(seg, vec_npz, mb, lb)
            keys = sorted(set(va) & set(vb))
            if len(keys) < 5:
                continue
            xa = np.stack([va[k] for k in keys])
            xb = np.stack([vb[k] for k in keys])
            for k in [5, 10, 20, 50]:
                kk = min(k, len(keys) - 1)
                if kk < 2:
                    continue
                ba = sample_space_basis(xa, kk)
                bb = sample_space_basis(xb, kk)
                row = subspace_metrics_from_sample_basis(ba, bb, kk)
                row.update({
                    "model_a": ma,
                    "model_b": mb,
                    "stage": stage,
                    "layer_a": la,
                    "layer_b": lb,
                    "geometry_type": "success_flow",
                    "n_paired_segments": len(keys),
                    "paired_segment_cka": linear_cka(xa, xb),
                })
                rows.append(row)

                ca = normalize_rows(get_npz(clean_npz, "clean", ma, la))
                cb = normalize_rows(get_npz(clean_npz, "clean", mb, lb))
                n_clean = min(len(ca), len(cb))
                if n_clean >= kk + 1:
                    cba = sample_space_basis(ca[:n_clean], kk)
                    cbb = sample_space_basis(cb[:n_clean], kk)
                    crow = subspace_metrics_from_sample_basis(cba, cbb, kk)
                    crow.update({
                        "model_a": ma,
                        "model_b": mb,
                        "stage": stage,
                        "layer_a": la,
                        "layer_b": lb,
                        "geometry_type": "clean_motion",
                        "n_paired_segments": n_clean,
                        "paired_segment_cka": linear_cka(ca[:n_clean], cb[:n_clean]),
                    })
                    rows.append(crow)

                ga = normalize_rows(get_npz(grad_npz, "grads", ma, la))
                gb = normalize_rows(get_npz(grad_npz, "grads", mb, lb))
                sg_a = seg[(seg.model == ma) & (seg.layer == la)].reset_index(drop=True)
                sg_b = seg[(seg.model == mb) & (seg.layer == lb)].reset_index(drop=True)
                # Pair gradient rows by the same trajectory keys where possible.
                gda, gdb = {}, {}
                for r in sg_a.itertuples():
                    gda[segment_key(r.run_id, r.target_class, r.start_generation, r.end_generation)] = ga[int(r.vector_idx)]
                for r in sg_b.itertuples():
                    gdb[segment_key(r.run_id, r.target_class, r.start_generation, r.end_generation)] = gb[int(r.vector_idx)]
                gkeys = sorted(set(gda) & set(gdb))
                if len(gkeys) >= kk + 1:
                    gxa = np.stack([gda[x] for x in gkeys])
                    gxb = np.stack([gdb[x] for x in gkeys])
                    grow = subspace_metrics_from_sample_basis(sample_space_basis(gxa, kk), sample_space_basis(gxb, kk), kk)
                    grow.update({
                        "model_a": ma,
                        "model_b": mb,
                        "stage": stage,
                        "layer_a": la,
                        "layer_b": lb,
                        "geometry_type": "gradient_only",
                        "n_paired_segments": len(gkeys),
                        "paired_segment_cka": linear_cka(gxa, gxb),
                    })
                    rows.append(grow)

                rrow = random_subspace_overlap(len(keys), kk, args.seed + hash((ma, mb, stage, k)) % 100000)
                rrow.update({
                    "model_a": ma,
                    "model_b": mb,
                    "stage": stage,
                    "layer_a": la,
                    "layer_b": lb,
                    "geometry_type": "random",
                    "n_paired_segments": len(keys),
                    "paired_segment_cka": np.nan,
                })
                rows.append(rrow)

            for train_model, test_model, train_layer, test_layer in [(ma, mb, la, lb), (mb, ma, lb, la)]:
                train_seg = seg[(seg.model == train_model) & (seg.layer == train_layer)].reset_index(drop=True)
                test_seg = seg[(seg.model == test_model) & (seg.layer == test_layer)].reset_index(drop=True)
                x_train = get_npz(vec_npz, "vectors", train_model, train_layer)
                x_test = get_npz(vec_npz, "vectors", test_model, test_layer)
                succ_train = train_seg["success"].to_numpy() == 1
                succ_test = test_seg["success"].to_numpy() == 1
                neg_test = ~succ_test
                if neg_test.sum() < 5:
                    clean = get_npz(clean_npz, "clean", test_model, test_layer)
                    neg = clean[: max(5, succ_test.sum())]
                    comp = "basis_transfer_success_vs_clean"
                else:
                    neg = x_test[neg_test]
                    comp = "basis_transfer_success_vs_failed"
                auc = basis_transfer_auc(x_train[succ_train], x_test[succ_test], neg, k=20)
                rows.append({
                    "model_a": train_model,
                    "model_b": test_model,
                    "stage": stage,
                    "layer_a": train_layer,
                    "layer_b": test_layer,
                    "geometry_type": comp,
                    "k": 20,
                    "cross_model_basis_transfer_auroc": auc,
                    "n_paired_segments": int(succ_test.sum()),
                })

    flow = pd.DataFrame(rows)
    own = pred[(pred["comparison"] == "success_vs_clean") & (pred["k"].astype(str) == "20")]
    own = own[["model", "layer", "variant", "auroc"]].rename(columns={"auroc": "own_success_vs_clean_auroc"})
    flow = flow.merge(own[own.variant == "raw"].drop(columns=["variant"]), left_on=["model_a", "layer_a"], right_on=["model", "layer"], how="left").drop(columns=["model", "layer"], errors="ignore")
    return flow


def correlate(standard: pd.DataFrame, flow: pd.DataFrame):
    primary = flow[flow["geometry_type"].isin(["success_flow", "clean_motion", "gradient_only", "random"])].copy()
    merged = standard.merge(primary, on=["model_a", "model_b", "stage", "layer_a", "layer_b"], how="inner", suffixes=("_standard", "_flow"))
    xcols = ["cka", "svcca", "output_probability_correlation", "prediction_agreement", "input_gradient_cosine", "logit_correlation"]
    ycols = ["projection_overlap", "subspace_affinity", "grassmann_distance", "paired_segment_cka", "own_success_vs_clean_auroc"]
    rows = []
    for geometry_type, gmerged in merged.groupby("geometry_type"):
        for x in xcols:
            for y in ycols:
                data = gmerged[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(data) < 3 or data[x].nunique() < 2 or data[y].nunique() < 2:
                    continue
                pr, pp = pearsonr(data[x], data[y])
                sr, sp = spearmanr(data[x], data[y])
                rows.append({
                    "geometry_type": geometry_type,
                    "standard_metric": x,
                    "flow_metric": y,
                    "n": len(data),
                    "pearson_r": float(pr),
                    "pearson_p": float(pp),
                    "spearman_r": float(sr),
                    "spearman_p": float(sp),
                })
    return pd.DataFrame(rows), merged


def plot_scatter(merged: pd.DataFrame, corr: pd.DataFrame, out_dir: Path):
    pairs = [("cka", "projection_overlap"), ("cka", "paired_segment_cka"), ("svcca", "projection_overlap"), ("input_gradient_cosine", "projection_overlap")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    merged = merged[merged["geometry_type"] == "success_flow"].copy()
    for ax, (x, y) in zip(axes.ravel(), pairs):
        data = merged[[x, y, "stage"]].dropna()
        for stage, g in data.groupby("stage"):
            ax.scatter(g[x], g[y], label=stage, s=35, alpha=0.75)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.25)
        c = corr[(corr.geometry_type == "success_flow") & (corr.standard_metric == x) & (corr.flow_metric == y)]
        if len(c):
            ax.set_title(f"Pearson r={c.iloc[0].pearson_r:.2f}, Spearman r={c.iloc[0].spearman_r:.2f}")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.savefig(out_dir / "similarity_vs_flow_scatterplots.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layerwise-dir", default="analysis_outputs/pure_af_geometry/cifar_layerwise_success_flow_c10_s3_g120")
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/cifar_similarity_vs_success_flow")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--clean-images", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reuse-clean-cache", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    standard = compute_standard_metrics(args, out_dir, device)
    flow = compute_flow_similarity(args, out_dir)
    corr, merged = correlate(standard, flow)
    standard.to_csv(out_dir / "model_similarity_metrics.csv", index=False)
    flow.to_csv(out_dir / "success_flow_similarity_metrics.csv", index=False)
    corr.to_csv(out_dir / "similarity_vs_flow_correlation.csv", index=False)
    merged.to_csv(out_dir / "similarity_vs_flow_merged_for_scatter.csv", index=False)
    plot_scatter(merged, corr, out_dir)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "args": vars(args),
            "notes": [
                "Cross-architecture success-flow principal-angle metrics are computed in paired trajectory-index space, not direct feature-coordinate space.",
                "PWCCA is left as NaN because no project implementation was available; SVCCA is computed directly.",
                "Basis-transfer AUROC uses success-vs-failed when target failures exist, otherwise success-vs-clean controls.",
            ],
        }, f, indent=2)
    print(f"[SAVED] {out_dir}")
    if len(corr):
        print(corr.sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
