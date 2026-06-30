# BlackboxBench CIFAR-10 Controlled Setup

This is a new controlled experimental setup. It is not the original MCG paper's CIFAR-10 target-model setup.

## Motivation

The original MCG paper reports CIFAR-10 experiments on author-trained target models, but the public MCG-Blackbox repository only documents downloadable MCG generator checkpoints. It does not provide the exact CIFAR-10 target classifier checkpoints used in the paper.

To avoid using randomly initialized classifiers or unverifiable ad hoc checkpoints, this project will use trained CIFAR-10 classifiers from BlackboxBench as a controlled benchmark. BlackboxBench is a cited black-box adversarial attack benchmark, and using its models gives us a clean source to report.

## Source

BlackboxBench repository: `https://github.com/SCLBD/BlackboxBench`

Relevant BlackboxBench facts from its README and transfer configuration:

- It is a PyTorch black-box adversarial attack benchmark.
- It provides model checkpoints for user convenience.
- Its paper supplement, section D-A, provides embedded checkpoint/source links for the CIFAR-10 models used in its evaluations.
- Its CIFAR-10 supported model names include `densenet`, `pyramidnet272`, `resnext`, `vgg19_bn`, `wrn`, `gdas`, `adv_wrn_28_10`, `resnet50`, and `inception_v3`.
- Its CIFAR-10 transfer configs commonly use `CIFAR10/pretrained/vgg19_bn` as the source/surrogate model.
- Its CIFAR-10 target lists include `vgg19_bn`, `wrn`, `resnet50`, `resnext`, `densenet`, `inception_v3`, `pyramidnet272`, `gdas`, and `adv_wrn_28_10`.
- Its model loader wraps CIFAR-10 models with normalization mean `[0.4914, 0.4822, 0.4465]` and std `[0.2023, 0.1994, 0.2010]`.

## Download Status

The BlackboxBench paper supplement link for transfer-based CIFAR-10 `PyramidNet` and `GDAS` points to a Google Drive archive named `cifar10_ckpt.tar`. This archive was downloaded locally on 2026-05-31 and extracted under `checkpoints/blackboxbench_cifar10/`.

Local checkpoint files currently available:

- `checkpoints/blackboxbench_cifar10/ckpt/vgg19_bn/model_best.pth.tar`
- `checkpoints/blackboxbench_cifar10/ckpt/densenet-bc-L190-k40/model_best.pth.tar`
- `checkpoints/blackboxbench_cifar10/ckpt/WRN-28-10-drop/model_best.pth.tar`
- `checkpoints/blackboxbench_cifar10/ckpt/resnext-8x64d/model_best.pth.tar`
- `checkpoints/blackboxbench_cifar10/ckpt/pyramidnet272-checkpoint.pth`
- `checkpoints/blackboxbench_cifar10/ckpt/gdas-cifar10-best.pth`

The archive itself is kept at `checkpoints/blackboxbench_cifar10/cifar10_ckpt.tar`. These files are ignored by git through the existing `checkpoints/` ignore rule.

Additional BlackboxBench CIFAR-10 links from the supplement that were not downloaded automatically:

- Transfer ResNet-50 and Inception-V3 are linked to Kaggle: `https://www.kaggle.com/datasets/firuzjuraev/trained-models-for-cifar10-dataset/download?datasetVersionNumber=1`. A direct `curl` attempt returned HTTP 403, so this likely needs browser login or Kaggle API credentials.
- Query-model Google Drive links exist for CIFAR-10 ResNet-50, VGG-19, Inception-V3, and DenseNet-121, but those are not the transfer setup selected for this controlled MCG experiment.

The Kaggle archive was downloaded manually as `/home/sepi/projects/MCG-Blackbox/archive (1).zip`. The ResNet-50 and Inception-V3 checkpoints were extracted under:

- `checkpoints/blackboxbench_cifar10/kaggle/trained_models_cifar10/resnet50_cifar10_lr01.pth`
- `checkpoints/blackboxbench_cifar10/kaggle/trained_models_cifar10/inceptionv3_cifar10_lr01.pth`

The archive also contains additional CIFAR-10 checkpoints such as VGG-19, DenseNet-169, MobileNetV2, GoogLeNet, and Xception, but only ResNet-50 and Inception-V3 are currently mapped into the controlled BlackboxBench loader.

## Proposed First Experiment

Use one BlackboxBench source/surrogate and one BlackboxBench target first:

- Surrogate model: `bbb_vgg19_bn`, loaded from `checkpoints/blackboxbench_cifar10/ckpt/vgg19_bn/model_best.pth.tar`, because BlackboxBench CIFAR-10 transfer configs use `vgg19_bn` as `source_model_path`.
- Target model: `bbb_densenet`, loaded from `checkpoints/blackboxbench_cifar10/ckpt/densenet-bc-L190-k40/model_best.pth.tar`, because it is one of the standard BlackboxBench CIFAR-10 targets and is close in spirit to the DenseNet target in the MCG paper, while still being explicitly reported as a BlackboxBench model.
- Dataset: CIFAR-10 test/validation path already available locally at `/home/sepi/data/cifar10`.
- Generator: existing MCG CIFAR-10 generator checkpoint, `checkpoints/cifar10_mcg.pth.tar`.
- Attack method: `square`.
- Variants: `base`, `surpgd_pre`, `surpgd_only` via `--compare_surpgd`.
- Helper script: `scripts/cifar10_blackboxbench_attack_untargeted.sh`.

This first pair should be treated as a validation run for the controlled setup. After it works, expand to additional BlackboxBench targets.

## Loader Verification

The dedicated loader path is exposed through explicit `bbb_` model names and does not change the legacy CIFAR-10 model names.

Lightweight checks run on 2026-05-31:

- Forward pass check on random tensors succeeded for `bbb_vgg19_bn` and `bbb_densenet`; both returned finite `(batch, 10)` logits.
- CPU state-dict and forward-pass checks succeeded for `bbb_resnet50` and `bbb_inception_v3`; both had no missing or unexpected checkpoint keys and returned finite `(batch, 10)` logits.
- Clean accuracy on this repository's current CIFAR-10 validation subset, which contains 1,000 images selected by `data.datasets.cifar10(..., mode="validation")`:
  - `bbb_vgg19_bn`: 934 / 1000 = 93.4%.
  - `bbb_densenet`: 970 / 1000 = 97.0%.
  - `bbb_resnet50`: 952 / 1000 = 95.2%.
  - `bbb_inception_v3`: 956 / 1000 = 95.6%.

Reproducible clean-accuracy command:

```bash
/home/sepi/jupyterenv/bin/python scripts/check_cifar10_blackboxbench_clean_accuracy.py --device cpu
```

CUDA verification and attack pilots could not be run at this point because the local NVIDIA stack reported `Driver/library version mismatch` through `nvidia-smi` and PyTorch reported CUDA initialization error 804.

## Reporting Requirements

Every result table or run artifact must state:

- This is `MCG-Blackbox + BlackboxBench CIFAR-10 classifiers`, not the original MCG paper target checkpoints.
- Target model name, checkpoint path, and checkpoint source.
- Surrogate model name, checkpoint path, and checkpoint source.
- Dataset root and subset policy.
- Epsilon / `linf`.
- Query budget.
- MCG generator checkpoint path.
- Whether surrogate fine-tuning and generator fine-tuning flags were enabled.
- The three variants: `base`, `surpgd_pre`, and `surpgd_only`.
- Conditional metrics for images where initial MCG was not adversarial.

## Implementation Plan

Keep this integration isolated from existing ImageNet and legacy CIFAR-10 behavior.

1. Download the BlackboxBench CIFAR-10 model archive from the public link documented by BlackboxBench.
2. Store the downloaded files under a clearly named directory such as `checkpoints/blackboxbench_cifar10/`.
3. Inspect checkpoint structure and verify model architecture names before adding any loader code.
4. Add a dedicated BlackboxBench CIFAR-10 loader path, for example `load_blackboxbench_cifar10_model`, rather than overloading existing `load_cifar_model` silently.
5. Add CLI/model names with an explicit prefix, for example `bbb_vgg19_bn` and `bbb_densenet`.
6. Validate clean accuracy on the CIFAR-10 test set or the exact selected subset before attack runs.
7. Run the three-variant comparison and archive command, logs, CSVs, valid indices, git commit hash, and clean-accuracy report together.

## Notes On Compatibility

The existing local CIFAR model definitions are not guaranteed to match BlackboxBench definitions:

- This repo's `DenseNet121` is not the same as BlackboxBench `densenet`, which is DenseNet-BC L190 k40 in BlackboxBench's transfer model code.
- This repo has `VGG('VGG19')` with batch normalization, but BlackboxBench names the model `vgg19_bn` and provides its own architecture definition/checkpoint path.
- BlackboxBench also includes `pyramidnet272`, while the original MCG paper reports `PyramidNet-110`.

Because of these differences, use BlackboxBench model code/checkpoint mapping explicitly instead of assuming checkpoint compatibility with this repo's current `surro_models/cifar10_models`.
