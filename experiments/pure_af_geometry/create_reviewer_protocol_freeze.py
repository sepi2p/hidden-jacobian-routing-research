#!/usr/bin/env python3
"""Create protocol-freeze artifacts for reviewer-critical reruns.

The goal is not to run an experiment.  It records enough model, layer, attack,
split, and schema information that later trajectory outputs can be audited
against a fixed protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd
import torch
import torchvision
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pure_af_geometry.analyze_cifar10_layerwise_success_flow import load_model  # noqa: E402
from experiments.pure_af_geometry.run_cifar_away_from_pure_flow_attack import select_clean_correct  # noqa: E402


MODEL_CHECKPOINTS = {
    "bbb_resnet50": "checkpoints/blackboxbench_cifar10/kaggle/trained_models_cifar10/resnet50_cifar10_lr01.pth",
    "bbb_vgg19_bn": "checkpoints/blackboxbench_cifar10/ckpt/vgg19_bn/model_best.pth.tar",
    "bbb_densenet": "checkpoints/blackboxbench_cifar10/ckpt/densenet-bc-L190-k40/model_best.pth.tar",
    "bbb_inception_v3": "checkpoints/blackboxbench_cifar10/kaggle/trained_models_cifar10/inceptionv3_cifar10_lr01.pth",
}


LAYER_ROWS = [
    {
        "model": "bbb_resnet50",
        "layer": "layer4",
        "module": "1.layer4",
        "status": "true_hidden",
        "pooling": "adaptive_avg_pool2d_to_1x1_then_flatten",
        "note": "Use this as the hidden layer for the balanced rerun. Do not count avgpool as independent.",
    },
    {
        "model": "bbb_resnet50",
        "layer": "avgpool",
        "module": "1.layer4",
        "status": "alias_of_layer4",
        "pooling": "adaptive_avg_pool2d_to_1x1_then_flatten",
        "note": "The local CIFAR ResNet wrapper exposes avgpool as pooled layer4; exclude from independent evidence.",
    },
    {
        "model": "bbb_resnet50",
        "layer": "logits",
        "module": "model_output",
        "status": "output",
        "pooling": "none",
        "note": "Use k capped below dimension; never use k >= d for subspace claims.",
    },
]


ATTACK_ROWS = [
    {
        "attack": "pgd_tuning",
        "threat_model": "white_box_untargeted_linf",
        "eps_grid_255": "0.5,1,1.5,2",
        "steps_or_queries": "1,2,3,5,10",
        "step_size_rule": "min(2/255, eps/2)",
        "loss": "cross_entropy",
        "random_start": "false",
        "early_stopping": "record_first_success_but_continue_to_budget",
        "accepted_update_logging": "all_pgd_steps_are_accepted_projected_updates",
    },
    {
        "attack": "square_tuning",
        "threat_model": "score_or_logit_query_untargeted_linf",
        "eps_grid_255": "2,4,6,8",
        "steps_or_queries": "100,250,500,1000,2000",
        "step_size_rule": "square_schedule",
        "loss": "margin/probability as implemented by imported square_trajectory",
        "random_start": "square_init_epochs=1,p_init=0.8",
        "early_stopping": "final_success_measured_at_budget_for_tuning",
        "accepted_update_logging": "full rerun must distinguish accepted states from query proposals if available",
    },
]


TRAJECTORY_SCHEMA = """# Reviewer-Critical Trajectory Schema

Each promoted trajectory artifact must distinguish the following concepts.

## State-level columns

- `model`
- `dataset`
- `image_ord`
- `dataset_idx`
- `label`
- `clean_pred`
- `attack`
- `step_index`
- `query_count`
- `accepted_update`
- `pred`
- `margin`
- `loss`
- `p_y`
- `linf_from_clean`
- `l2_from_clean`
- `first_success_step`
- `final_success`

## Feature-level columns

- `layer`
- `feature_key`
- `feature_idx`
- `feature_shape`
- `pooling`
- `h_t`
- `local_step = h_{t+1} - h_t`
- `cumulative_displacement = h_t - h_0`

## Split and weighting rules

- Splits are by `image_ord`, never by segment.
- PCA bases are fit only on train images.
- Layer and `k` are chosen only on validation images.
- Final AUROC/statistics are reported only on test images.
- Bootstrap resampling unit is image ID, not segment row.
"""


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="analysis_outputs/pure_af_geometry/jacobian_null_response/protocol_freeze")
    p.add_argument("--dataset-root", default="/home/sepi/data/cifar10")
    p.add_argument("--model", default="bbb_resnet50")
    p.add_argument("--images", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_rows = []
    for name, ckpt in MODEL_CHECKPOINTS.items():
        ckpt_path = Path(ckpt)
        model_rows.append(
            {
                "model": name,
                "dataset": "CIFAR-10",
                "checkpoint_path": ckpt,
                "checkpoint_sha256": sha256_file(ckpt_path),
                "preprocessing": "ToTensor plus NormalizeByChannelMeanStd(mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010]) inside model wrapper",
                "eval_mode": "true",
                "clean_accuracy_scope": "not_computed_for_all_models_in_step0",
            }
        )
    pd.DataFrame(model_rows).to_csv(out_dir / "model_registry.csv", index=False)
    pd.DataFrame(LAYER_ROWS).to_csv(out_dir / "layer_registry.csv", index=False)
    pd.DataFrame(ATTACK_ROWS).to_csv(out_dir / "attack_registry.csv", index=False)

    tfm = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=args.dataset_root, train=False, download=False, transform=tfm)
    wrapper = load_model(args.model, device)
    selected = select_clean_correct(dataset, {args.model: wrapper}, argparse.Namespace(models=[args.model], images=args.images), device)
    split_rows = []
    for image_ord, (dataset_idx, label) in enumerate(selected):
        if image_ord < int(round(0.5 * len(selected))):
            split = "train"
        elif image_ord < int(round(0.75 * len(selected))):
            split = "validation"
        else:
            split = "test"
        split_rows.append(
            {
                "model": args.model,
                "dataset": "CIFAR-10 test",
                "image_ord": image_ord,
                "dataset_idx": int(dataset_idx),
                "label": int(label),
                "split": split,
                "clean_correct_by_selection": 1,
            }
        )
    pd.DataFrame(split_rows).to_csv(out_dir / "image_splits.csv", index=False)

    (out_dir / "trajectory_schema.md").write_text(TRAJECTORY_SCHEMA, encoding="utf-8")

    manifest = {
        "script": "experiments/pure_af_geometry/create_reviewer_protocol_freeze.py",
        "output_dir": str(out_dir),
        "primary_model": args.model,
        "dataset_root": args.dataset_root,
        "selected_images": len(selected),
        "seed": args.seed,
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "artifacts": [
            "model_registry.csv",
            "layer_registry.csv",
            "attack_registry.csv",
            "image_splits.csv",
            "trajectory_schema.md",
        ],
    }
    (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    checklist = """# Reproducibility Checklist

- [x] Model checkpoint paths recorded.
- [x] Checkpoint hashes recorded where files exist.
- [x] Preprocessing recorded.
- [x] Primary layer aliases recorded.
- [x] Attack tuning grids recorded.
- [x] Clean-correct image IDs and image-level splits recorded.
- [x] Local-step and cumulative-displacement schema separated.
- [ ] Full balanced trajectory artifacts generated.
- [ ] Balanced rerun config linked to this manifest.
- [ ] Figure/table reproduction commands listed.
"""
    (out_dir / "reproducibility_checklist.md").write_text(checklist, encoding="utf-8")
    print(f"[DONE] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
