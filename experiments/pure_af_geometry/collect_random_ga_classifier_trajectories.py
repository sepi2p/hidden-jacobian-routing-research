#!/usr/bin/env python3
"""Collect random-noise-to-class-pure GA trajectories in classifier feature space.

Primary representation: target classifier internal features. Secondary records:
logits, pixel/frequency statistics, and optional DifAttack++ AF/VF pooled features.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, utils as tv_utils

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pure_af_geometry.generate_pure_images_ga import (
    evaluate_population,
    make_children,
    margin_and_prob,
    mutate,
    parse_ints,
    set_seed,
    topk_dict,
)
from utils.load_models import load_imagenet_model

CLASSIFIER_LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool", "logits"]
AFVF_LAYERS = ["sem0", "sem1", "sem4", "vf0", "vf1", "vf4"]
LAYER_NAMES = ["sem0", "sem1", "sem2", "sem3", "sem4"]
VIS_NAMES = ["vf0", "vf1", "vf2", "vf3", "vf4"]


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


class ImageNetFeatureRecorder:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.base = model[1] if isinstance(model, torch.nn.Sequential) else model
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles = []
        self.enabled = False
        self.layer_names = CLASSIFIER_LAYERS
        for name, module in self._modules_to_record():
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _modules_to_record(self):
        if all(hasattr(self.base, name) for name in ["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"]):
            return [(name, getattr(self.base, name)) for name in ["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"]]

        if hasattr(self.base, "features") and hasattr(self.base.features, "denseblock1"):
            features = self.base.features
            return [
                ("conv1", features.conv0),
                ("layer1", features.denseblock1),
                ("layer2", features.denseblock2),
                ("layer3", features.denseblock3),
                ("layer4", features.denseblock4),
                ("avgpool", features.norm5),
            ]

        if hasattr(self.base, "features") and hasattr(self.base, "avgpool"):
            features = self.base.features
            if len(features) >= 44:
                return [
                    ("conv1", features[0]),
                    ("layer1", features[6]),
                    ("layer2", features[13]),
                    ("layer3", features[23]),
                    ("layer4", features[43]),
                    ("avgpool", self.base.avgpool),
                ]

        raise NotImplementedError(f"Unsupported ImageNet feature recorder base: {type(self.base).__name__}")

    def _hook(self, name: str):
        def fn(_module, _inp, out):
            if self.enabled:
                self.outputs[name] = out.detach().float().cpu()
        return fn

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> tuple[dict[str, np.ndarray], np.ndarray]:
        self.outputs = {}
        self.enabled = True
        try:
            logits = self.model(image).detach().float().cpu()
        finally:
            self.enabled = False
        feats: dict[str, np.ndarray] = {}
        for name in ["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"]:
            # Global average pooling keeps dimensions compact while preserving layer identity.
            out = self.outputs[name]
            if out.ndim == 4:
                out = F.adaptive_avg_pool2d(out, (1, 1))
            feats[name] = out.flatten(1).numpy()[0].astype(np.float32)
        feats["logits"] = logits.numpy()[0].astype(np.float32)
        return feats, logits.numpy()[0].astype(np.float32)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def encode_afvf(ae, image: torch.Tensor, pool_size: int) -> dict[str, np.ndarray]:
    parts = ae(image.mul(2.0).sub(1.0))
    layer_map = {**dict(zip(VIS_NAMES, parts[1:6])), **dict(zip(LAYER_NAMES, parts[6:11]))}
    out = {}
    for layer in AFVF_LAYERS:
        pooled = F.adaptive_avg_pool2d(layer_map[layer].detach().float(), (pool_size, pool_size)).flatten(1)
        out[layer] = pooled.cpu().numpy()[0].astype(np.float32)
    return out


def frequency_stats(image: torch.Tensor) -> dict[str, float]:
    x = image.detach().float().cpu()[0]
    gray = (0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]).numpy()
    gray = gray - float(gray.mean())
    spec = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(spec) ** 2
    h, w = power.shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    rmax = float(rr.max())
    total = float(power.sum() + 1e-12)
    low = float(power[rr <= 0.15 * rmax].sum() / total)
    mid = float(power[(rr > 0.15 * rmax) & (rr <= 0.45 * rmax)].sum() / total)
    high = float(power[rr > 0.45 * rmax].sum() / total)
    centroid = float((power * rr).sum() / total / rmax)
    return {"freq_low": low, "freq_mid": mid, "freq_high": high, "freq_centroid": centroid}


def pixel_stats(image: torch.Tensor, start: torch.Tensor) -> dict[str, float]:
    delta = (image - start).detach().float()
    return {
        "pixel_linf_from_start": float(delta.abs().max().item()),
        "pixel_l2_from_start": float(delta.flatten(1).norm(p=2, dim=1).item()),
        "pixel_mean": float(image.mean().item()),
        "pixel_std": float(image.std(unbiased=False).item()),
    }


def save_checkpoint(
    *,
    run_dir: Path,
    run_name: str,
    generation: int,
    image: torch.Tensor,
    start: torch.Tensor,
    target: int,
    fitness: float,
    margin: float,
    prob: float,
    pred: int,
    mean_fitness: float,
    recorder: ImageNetFeatureRecorder,
    ae,
    afvf_pool_size: int,
    rows: list[dict[str, object]],
    features: dict[str, list[np.ndarray]],
    logits_list: list[np.ndarray],
    save_images: bool,
) -> None:
    feats, logits = recorder(image)
    for layer, vec in feats.items():
        features[f"clf_{layer}"].append(vec)
    if ae is not None:
        afvf = encode_afvf(ae, image, afvf_pool_size)
        for layer, vec in afvf.items():
            features[f"afvf_{layer}"].append(vec)
    logits_list.append(logits)
    image_path = ""
    tensor_path = ""
    if save_images:
        image_path = str(run_dir / f"best_gen{generation:06d}.png")
        tensor_path = str(run_dir / f"best_gen{generation:06d}.pt")
        tv_utils.save_image(image.cpu(), image_path)
        torch.save(image.detach().cpu(), tensor_path)
    row = {
        "run_name": run_name,
        "target_class": target,
        "generation": generation,
        "sample_index": len(rows),
        "fitness": fitness,
        "margin": margin,
        "prob": prob,
        "pred": pred,
        "mean_fitness": mean_fitness,
        "image_path": image_path,
        "tensor_path": tensor_path,
        **pixel_stats(image, start),
        **frequency_stats(image),
    }
    top_vals = np.argsort(logits)[-5:][::-1]
    for rank, cls in enumerate(top_vals, start=1):
        row[f"top{rank}_class"] = int(cls)
        row[f"top{rank}_logit"] = float(logits[cls])
    rows.append(row)


def run_one(args, model, recorder, ae, target: int, seed: int, device: torch.device) -> dict[str, object]:
    set_seed(seed)
    run_name = f"class{target:04d}_random_seed{seed:04d}"
    run_dir = Path(args.output_dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pop = torch.rand((args.population, 3, args.image_size, args.image_size), device=device)
    start = pop[0:1].detach().clone()
    rows: list[dict[str, object]] = []
    features: dict[str, list[np.ndarray]] = {f"clf_{layer}": [] for layer in recorder.layer_names}
    if ae is not None:
        features.update({f"afvf_{layer}": [] for layer in AFVF_LAYERS})
    logits_list: list[np.ndarray] = []

    best = None
    generations_to_success = None
    save_set = set(range(0, args.generations + 1, args.save_every))
    save_set.update(parse_ints(args.extra_save_generations) if args.extra_save_generations else [])

    for gen in range(args.generations + 1):
        stats = evaluate_population(model, pop, target, None, 0.0, 0.0, args.eval_batch_size)
        order = torch.argsort(stats["fitness"], descending=True)
        pop = pop[order]
        for key in stats:
            stats[key] = stats[key][order]

        cur = {
            "image": pop[0:1].detach().clone(),
            "fitness": float(stats["fitness"][0].item()),
            "margin": float(stats["margin"][0].item()),
            "prob": float(stats["prob"][0].item()),
            "pred": int(stats["pred"][0].item()),
            "mean_fitness": float(stats["fitness"].mean().item()),
            "generation": gen,
        }
        if best is None or cur["fitness"] > best["fitness"]:
            best = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in cur.items()}

        success_now = cur["prob"] >= args.prob_threshold and cur["pred"] == target
        if generations_to_success is None and success_now:
            generations_to_success = gen
            save_set.add(gen)

        should_save = gen in save_set or gen == args.generations or (args.save_on_improvement and best is not None and best["generation"] == gen)
        if should_save:
            save_checkpoint(
                run_dir=run_dir,
                run_name=run_name,
                generation=gen,
                image=cur["image"],
                start=start,
                target=target,
                fitness=cur["fitness"],
                margin=cur["margin"],
                prob=cur["prob"],
                pred=cur["pred"],
                mean_fitness=cur["mean_fitness"],
                recorder=recorder,
                ae=ae,
                afvf_pool_size=args.afvf_pool_size,
                rows=rows,
                features=features,
                logits_list=logits_list,
                save_images=args.save_images,
            )

        if args.stop_on_success and success_now:
            break
        if gen == args.generations:
            break

        parents = pop[: args.parents]
        elite = pop[: args.elite]
        children = make_children(parents, args.population - args.elite, args.crossover)
        children = mutate(
            children,
            pixel_sigma=args.pixel_sigma,
            pixel_rate=args.pixel_rate,
            block_rate=args.block_rate,
            block_size=args.block_size,
        )
        pop = torch.cat([elite, children], dim=0)

    assert best is not None
    traj_path = run_dir / "trajectory.csv"
    with traj_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    npz_path = run_dir / "trajectory_features.npz"
    np.savez_compressed(
        npz_path,
        **{k: np.stack(v, axis=0) for k, v in features.items()},
        logits=np.stack(logits_list, axis=0),
        generation=np.array([int(r["generation"]) for r in rows], dtype=np.int64),
    )
    final_path = run_dir / "final_best.png"
    final_tensor_path = run_dir / "final_best.pt"
    tv_utils.save_image(best["image"].cpu(), final_path)
    torch.save(best["image"].cpu(), final_tensor_path)
    with torch.no_grad():
        logits_t = model(best["image"])
    meta = {
        "run_name": run_name,
        "target_class": target,
        "seed": seed,
        "init_mode": "random",
        "trajectory_csv": str(traj_path),
        "trajectory_features_npz": str(npz_path),
        "final_image": str(final_path),
        "final_tensor": str(final_tensor_path),
        "final_generation": int(best["generation"]),
        "final_margin": float(best["margin"]),
        "final_prob": float(best["prob"]),
        "final_pred": int(best["pred"]),
        "generations_to_success": generations_to_success,
        "success": int(generations_to_success is not None),
        "saved_checkpoints": len(rows),
        "top5": topk_dict(logits_t),
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="resnet18")
    parser.add_argument("--classes", default="0-9")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/random_ga_classifier_trajectories_resnet18")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--parents", type=int, default=16)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--generations", type=int, default=30000)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--prob-threshold", type=float, default=0.9999)
    parser.add_argument("--pixel-sigma", type=float, default=0.08)
    parser.add_argument("--pixel-rate", type=float, default=0.03)
    parser.add_argument("--block-rate", type=float, default=0.4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--crossover", choices=["uniform", "average"], default="uniform")
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--extra-save-generations", default="1,2,5,10,20,50,100")
    parser.add_argument("--save-on-improvement", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--include-afvf", action="store_true")
    parser.add_argument("--difattackpp-checkpoint", default="external_repos/DifAttack_assets/difattack_plus/ResNet18.pth.tar")
    parser.add_argument("--checkpoint-key", default="state_dict_adv")
    parser.add_argument("--afvf-pool-size", type=int, default=4)
    args = parser.parse_args()

    if args.parents < args.elite:
        raise ValueError("--parents must be >= --elite")
    if args.population <= args.elite:
        raise ValueError("--population must be > --elite")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_imagenet_model(args.target_model).to(device).eval()
    recorder = ImageNetFeatureRecorder(model)
    ae = load_autoencoder(Path(args.difattackpp_checkpoint), device, args.checkpoint_key) if args.include_afvf else None

    rows = []
    started = time.time()
    try:
        for target in parse_ints(args.classes):
            for seed in parse_ints(args.seeds):
                meta = run_one(args, model, recorder, ae, target, seed, device)
                rows.append(meta)
                print(f"[DONE] {meta['run_name']} success={meta['success']} gen={meta['generations_to_success']} prob={meta['final_prob']:.6f}", flush=True)
    finally:
        recorder.close()

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "experiment": "random_noise_to_pure_classifier_trajectory",
        "target_model": args.target_model,
        "classes": parse_ints(args.classes),
        "seeds": parse_ints(args.seeds),
        "classifier_layers": recorder.layer_names,
        "include_afvf": bool(args.include_afvf),
        "afvf_layers": AFVF_LAYERS if args.include_afvf else [],
        "manifest": str(manifest_path),
        "elapsed_sec": time.time() - started,
        "args": vars(args),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
