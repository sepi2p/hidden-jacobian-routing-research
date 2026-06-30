#!/usr/bin/env python3
"""Source-family readout with GA and attack families in DifAttack++ AF/VF space."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.flow_latent import generate_interface, latent_initialize, latent_operate
from utils.load_models import load_generator, load_imagenet_model

LAYER_NAMES = ["sem0", "sem1", "sem2", "sem3", "sem4"]
VIS_NAMES = ["vf0", "vf1", "vf2", "vf3", "vf4"]
DEFAULT_LAYERS = ["sem0", "sem1", "sem4", "vf0", "vf1", "vf4"]
FAMILY_ORDER = ["clean", "ga_real_pure", "ga_random_pure", "fgsm", "pgd", "mcg", "difattackpp"]


def parse_ints(text: str) -> list[int]:
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = [int(x) for x in chunk.split("-", 1)]
            out.extend(range(a, b + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def parse_layers(text: str) -> list[str]:
    layers = [x.strip() for x in text.split(",") if x.strip()]
    known = set(LAYER_NAMES + VIS_NAMES)
    bad = [x for x in layers if x not in known]
    if bad:
        raise ValueError(f"Unknown layers {bad}; known={sorted(known)}")
    return layers


def mcg_args(generator_path: str, linf: float):
    return SimpleNamespace(
        generator_path=generator_path,
        x_size=(3, 224, 224),
        y_size=(3, 224, 224),
        x_hidden_channels=64,
        x_hidden_size=128,
        y_hidden_channels=256,
        flow_depth=8,
        num_levels=3,
        learn_top=False,
        label_scale=1,
        label_bias=0.0,
        x_bins=256.0,
        y_bins=2.0,
        optimizer="adam",
        lr=0.0002,
        betas=(0.9, 0.9999),
        eps=1e-8,
        regularizer=0.0,
        num_steps=0,
        margin=1.0,
        Lambda=1e-2,
        num_epochs=10,
        batch_size=1,
        down_sample_x=8,
        down_sample_y=8,
        max_grad_clip=5,
        max_grad_norm=0,
        checkpoints_gap=1000,
        nll_gap=1,
        inference_gap=1000,
        save_gap=1000,
        adv_loss=False,
        targeted=False,
        tanh=False,
        only=False,
        partial=False,
        rand=False,
        clamp=False,
        class_size=-1,
        linf=linf,
        class_num=1000,
        target_label=1,
    )


def load_autoencoder(checkpoint_path: Path, device: torch.device, key: str):
    module_path = REPO_ROOT / "external_repos" / "DifAttack" / "autoencoder.py"
    spec = importlib.util.spec_from_file_location("difattack_autoencoder", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if key not in ckpt:
        raise KeyError(f"Checkpoint {checkpoint_path} lacks key {key!r}; keys={sorted(ckpt.keys())}")
    model = module.Autoencoder().to(device).eval()
    model.load_state_dict(ckpt[key])
    return model


def encode(ae, image: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        parts = ae(image.mul(2.0).sub(1.0))
    return {**dict(zip(VIS_NAMES, parts[1:6])), **dict(zip(LAYER_NAMES, parts[6:11]))}


def pooled(x: torch.Tensor, pool_size: int) -> np.ndarray:
    y = F.adaptive_avg_pool2d(x.detach().float(), (pool_size, pool_size))
    return y.flatten(start_dim=1).cpu().numpy()[0].astype(np.float32)


def tensor_to_unit(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    if x.max() > 2.0:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


def logits_info(model, image: torch.Tensor, label: int) -> dict[str, float | int]:
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
    masked = logits.clone()
    masked[:, label] = -torch.inf
    next_logit, next_idx = masked.max(dim=1)
    return {
        "pred": int(logits.argmax(1).item()),
        "success_untargeted": int(logits.argmax(1).item() != label),
        "prob": float(probs[0, label].item()),
        "margin": float((logits[0, label] - next_logit[0]).item()),
        "next_best_class": int(next_idx.item()),
    }


def clamp_linf(x: torch.Tensor, clean: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.max(torch.min(x, clean + eps), clean - eps).clamp(0.0, 1.0)


def fgsm(model, clean: torch.Tensor, label: int, eps: float) -> torch.Tensor:
    x = clean.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), torch.tensor([label], device=x.device))
    grad = torch.autograd.grad(loss, x)[0]
    return clamp_linf(clean + eps * grad.sign(), clean, eps).detach()


def pgd(model, clean: torch.Tensor, label: int, eps: float, step_size: float, steps: int) -> torch.Tensor:
    x = clean.detach().clone()
    target = torch.tensor([label], device=x.device)
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x), target)
        grad = torch.autograd.grad(loss, x)[0]
        x = clamp_linf(x + step_size * grad.sign(), clean, eps)
    return x.detach()


def find_strong_clean(dataset, classifier, label: int, device: torch.device) -> tuple[int, str, torch.Tensor]:
    best = None
    for idx, (path, y) in enumerate(dataset.samples):
        if int(y) != label:
            continue
        image, _ = dataset[idx]
        image = image.unsqueeze(0).to(device)
        info = logits_info(classifier, image, label)
        if info["pred"] != label:
            continue
        if best is None or info["margin"] > best[3]["margin"]:
            best = (idx, path, image, info)
    if best is None:
        raise RuntimeError(f"No clean-correct image found for class {label}")
    return best[0], best[1], best[2]


def choose_ga_row(df: pd.DataFrame, label: int, init: str) -> pd.Series:
    sub = df[(df["target_class"].astype(int) == label) & (df["init_mode"] == init)].copy()
    if sub.empty:
        raise RuntimeError(f"Missing GA pure row label={label} init={init}")
    return sub.iloc[0]


def choose_difattack_row(df: pd.DataFrame, label: int) -> pd.Series:
    sub = df[(df["label"].astype(int) == label) & (df["success"].astype(int) == 1)].copy()
    if sub.empty:
        raise RuntimeError(f"Missing DifAttack++ success row for label={label}")
    if "query_cnt" in sub.columns:
        sub = sub.sort_values(["query_cnt", "dataset_idx"])
    return sub.iloc[0]


def evaluate(rows: list[dict[str, object]], vectors: dict[str, np.ndarray], out_dir: Path) -> list[dict[str, object]]:
    y = np.array([r["family"] for r in rows])
    labels = np.array([int(r["label"]) for r in rows])
    sample_ids = np.array([r["sample_id"] for r in rows])
    families = [f for f in FAMILY_ORDER if f in set(y)]
    pred_rows = []
    summary = []
    for layer, X in vectors.items():
        pred_all = np.empty_like(y, dtype=object)
        fold_rows = []
        for heldout in sorted(set(labels)):
            train = labels != heldout
            test = labels == heldout
            clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0, class_weight="balanced"))
            clf.fit(X[train], y[train])
            pred = clf.predict(X[test])
            pred_all[test] = pred
            fold_rows.append({"layer": layer, "heldout_label": int(heldout), "n_test": int(test.sum()), "accuracy": float(accuracy_score(y[test], pred))})
            for sid, true, pr in zip(sample_ids[test], y[test], pred):
                pred_rows.append({"layer": layer, "heldout_label": int(heldout), "sample_id": sid, "true_family": true, "pred_family": pr, "correct": int(true == pr)})
        report = classification_report(y, pred_all, labels=families, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, pred_all, labels=families)
        pd.DataFrame(cm, index=families, columns=families).to_csv(out_dir / f"confusion_{layer}.csv")
        pd.DataFrame(fold_rows).to_csv(out_dir / f"fold_accuracy_{layer}.csv", index=False)
        (out_dir / f"classification_report_{layer}.json").write_text(json.dumps(report, indent=2))
        summary.append({
            "layer": layer,
            "n_samples": int(len(y)),
            "n_classes_heldout": int(len(set(labels))),
            "n_families": int(len(families)),
            "leave_one_class_out_accuracy": float(accuracy_score(y, pred_all)),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
        })
    pd.DataFrame(pred_rows).to_csv(out_dir / "leave_one_class_predictions.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-manifest", required=True)
    parser.add_argument("--difattack-successes-csv", required=True)
    parser.add_argument("--difattackpp-checkpoint", default="external_repos/DifAttack_assets/difattack_plus/ResNet18.pth.tar")
    parser.add_argument("--checkpoint-key", default="state_dict_adv")
    parser.add_argument("--mcg-generator-path", default="checkpoints/imagenet_mcg.pth.tar")
    parser.add_argument("--target-model", default="resnet18")
    parser.add_argument("--imagenet-root", default="/home/sepi/Study/coding/data/imagenet/val")
    parser.add_argument("--classes", default="0-9")
    parser.add_argument("--layers", default=",".join(DEFAULT_LAYERS))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--linf", type=float, default=0.05)
    parser.add_argument("--pgd-steps", type=int, default=40)
    parser.add_argument("--pgd-step-size", type=float, default=0.005)
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/attack_family_classifier_multiclass10_resnet18_adv")
    args = parser.parse_args()

    labels = parse_ints(args.classes)
    layers = parse_layers(args.layers)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.imagenet_root, transform=transform)
    classifier = load_imagenet_model(args.target_model).to(device).eval()
    ae = load_autoencoder(Path(args.difattackpp_checkpoint), device, args.checkpoint_key)
    generator = load_generator(mcg_args(args.mcg_generator_path, args.linf)).to(device).eval()
    generate = generate_interface(generator, latent_operate, args.linf)

    pure = pd.read_csv(args.pure_manifest)
    pure = pure[(pure["final_pred"].astype(int) == pure["target_class"].astype(int)) & (pure["final_prob"].astype(float) >= 0.9999)]
    dif = pd.read_csv(args.difattack_successes_csv)

    rows: list[dict[str, object]] = []
    vectors: dict[str, list[np.ndarray]] = {layer: [] for layer in layers}

    def record(sample_id: str, family: str, label: int, image: torch.Tensor, path: str, extra: dict[str, object]) -> None:
        image = image.detach().clamp(0.0, 1.0).to(device)
        parts = encode(ae, image)
        info = logits_info(classifier, image, label)
        delta = (image - extra.get("clean_image", image)).detach() if torch.is_tensor(extra.get("clean_image", None)) else torch.zeros_like(image)
        clean_free_extra = {k: v for k, v in extra.items() if k != "clean_image"}
        rows.append({
            "sample_id": sample_id,
            "family": family,
            "label": int(label),
            "path": path,
            "pred": info["pred"],
            "success_untargeted": info["success_untargeted"],
            "prob_true_label": info["prob"],
            "margin_true_vs_next": info["margin"],
            "next_best_class": info["next_best_class"],
            "linf_from_clean": float(delta.abs().max().item()),
            "l2_from_clean": float(delta.flatten(1).norm(p=2, dim=1).item()),
            **clean_free_extra,
        })
        for layer in layers:
            vectors[layer].append(pooled(parts[layer], args.pool_size))

    for label in labels:
        idx, clean_path, clean = find_strong_clean(dataset, classifier, label, device)
        record(f"clean_c{label:04d}", "clean", label, clean, clean_path, {"dataset_idx": idx, "clean_image": clean})

        for init, family in [("real", "ga_real_pure"), ("random", "ga_random_pure")]:
            row = choose_ga_row(pure, label, init)
            image = transform(Image.open(row["final_image"]).convert("RGB")).unsqueeze(0).to(device)
            record(f"{family}_c{label:04d}", family, label, image, str(row["final_image"]), {"dataset_idx": "", "init_mode": init, "source_run": row.get("run_name", "")})

        x_fgsm = fgsm(classifier, clean, label, args.linf)
        record(f"fgsm_c{label:04d}", "fgsm", label, x_fgsm, clean_path, {"dataset_idx": idx, "clean_image": clean, "eps": args.linf})

        x_pgd = pgd(classifier, clean, label, args.linf, args.pgd_step_size, args.pgd_steps)
        record(f"pgd_c{label:04d}", "pgd", label, x_pgd, clean_path, {"dataset_idx": idx, "clean_image": clean, "eps": args.linf, "pgd_steps": args.pgd_steps, "pgd_step_size": args.pgd_step_size})

        with torch.no_grad():
            latent, _ = latent_initialize(clean, generator, latent_operate)
            perturbation = generate(clean, latent)
            x_mcg = (clean + perturbation.view_as(clean)).clamp(0.0, 1.0)
        record(f"mcg_c{label:04d}", "mcg", label, x_mcg, clean_path, {"dataset_idx": idx, "clean_image": clean, "eps": args.linf})

        drow = choose_difattack_row(dif, label)
        data = torch.load(drow["tensor_path"], map_location=device)
        clean_dif = tensor_to_unit(data["clean_uint8"]).unsqueeze(0).to(device)
        adv_dif = tensor_to_unit(data["adv_uint8"]).unsqueeze(0).to(device)
        record(
            f"difattackpp_c{label:04d}",
            "difattackpp",
            label,
            adv_dif,
            str(drow["tensor_path"]),
            {"dataset_idx": int(drow["dataset_idx"]), "clean_image": clean_dif, "query_cnt": int(drow.get("query_cnt", -1)), "attacked_surrogate_model": drow.get("attacked_surrogate_model", "")},
        )
        print(f"[class {label}] recorded 7 families", flush=True)

    sample_path = out_dir / "sample_manifest.csv"
    with sample_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out_dir / "pooled_attack_family_features.npz", sample_id=np.array([r["sample_id"] for r in rows]), **{k: np.stack(v, axis=0) for k, v in vectors.items()})

    summary = evaluate(rows, {k: np.stack(v, axis=0) for k, v in vectors.items()}, out_dir)
    summary_path = out_dir / "layer_attack_family_classifier_summary.csv"
    pd.DataFrame(summary).sort_values("leave_one_class_out_accuracy", ascending=False).to_csv(summary_path, index=False)

    metadata = {
        "pure_manifest": args.pure_manifest,
        "difattack_successes_csv": args.difattack_successes_csv,
        "labels": labels,
        "families": [f for f in FAMILY_ORDER if f in {r["family"] for r in rows}],
        "layers": layers,
        "n_samples": len(rows),
        "protocol": "RidgeClassifier on standardized pooled DifAttack++ AF/VF features; leave-one-target-class-out.",
        "note": "FGSM/PGD/MCG are generated from strong clean representatives. DifAttack++ samples are reused from saved quota-5 successes for the same labels, not necessarily the same clean representatives.",
        "outputs": {"sample_manifest": str(sample_path), "summary": str(summary_path), "predictions": str(out_dir / "leave_one_class_predictions.csv")},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
