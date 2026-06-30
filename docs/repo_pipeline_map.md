# Repository Pipeline Map

This audit uses the uploaded paper only as the intended MCG reference. The paper describes MCG as a conditional c-Glow meta generator trained on surrogate models, adapted during meta-test by surrogate fine-tuning and generator fine-tuning, then used to initialize or parameterize off-the-shelf black-box attacks. The current repository implements that broad shape, with additional local changes such as random valid-image sampling and SurPGD comparison paths.

## Current Working Entry Point

Command:

```bash
bash scripts/imagenet_attack_untargeted.sh \
  --dataset_root /home/sepi/Study/coding/data/imagenet \
  --attack_method square \
  --compare_surpgd
```

Pipeline:

1. `scripts/imagenet_attack_untargeted.sh` calls `python attack.py` with ImageNet defaults: target model `resnet18`, generator checkpoint `checkpoints/imagenet_mcg.pth.tar`, surrogate `vgg19`, `max_query=1000`, `class_num=1000`, `linf=0.05`, `max_images=1000`, `attack_method=square`, `--finetune_glow`, `--finetune_reload`, and `--finetune_perturbation`. The script appends `"$@"`, so the command-line overrides are forwarded.
2. `attack.py` parses all evaluation flags in its `if __name__ == "__main__"` block and calls `attack(args)`.
3. `attack.attack` seeds RNGs, calls `utils.attack_init.data_init`, `model_init`, `attacker_init`, `buffer_init`, `trainer_init`, and `log_init`.
4. `attack.attack` intentionally disables `args.max_images` while building the dataloader, then collects `max_images` correctly classified validation samples with `build_valid_buffer_random`.
5. For each valid image, `attack.attack` optionally fine-tunes surrogate models with `utils.surrogate_trainer.TrainModelSurrogate`, initializes a latent code with `models.flow_latent.latent_initialize`, optionally fine-tunes latent or generator with `utils.finetune.finetune_latent` / `meta_finetune`, then dispatches one or more variants through `attack.run_single_variant`.
6. `attack.run_single_variant` generates the MCG perturbation with `models.flow_latent.generate_interface`, queries the target through `attacks.base_attack.margin_loss_interface`, optionally runs SurPGD stages, then calls the selected attacker.

## Main Scripts

- `attack.py`: main attack/evaluation driver. It handles CLI parsing, dataset/model/attacker setup, valid-image filtering, MCG initialization, optional fine-tuning, SurPGD comparison mode, and result logging.
- `train.py`: c-Glow pretraining/meta-training driver. With `--adv_loss=False`, it uses `trainners.Learner.Trainer`; with `--adv_loss=True`, it uses `trainners.MetaLearner.Trainer`.
- `viz_attack_outputs.py`: visualization helper. It loads datasets/models/attackers and writes images/probability plots, but does not load the c-Glow generator.
- `attack-Copy1.py`: stale/copy evaluation script with older SurPGD flags. It is not called by current shell scripts.
- `scripts/imagenet_metatrain.sh` references `train_imagenet.py`, which is not present in this repository.

## Dataset Loading

- Evaluation data starts in `utils.attack_init.data_init`.
- ImageNet calls `data.datasets.imagenet(args.dataset_root, mode="validation")`, then sets `args.x_size` and `args.y_size` to `(3, 224, 224)`.
- CIFAR-10 calls `data.datasets.cifar10(args.dataset_root, mode="validation")`, then sets `args.x_size` and `args.y_size` to `(3, 32, 32)`.
- `data.datasets.imagenet` currently ignores the supplied `root` and hard-codes `/home/sepi/Study/coding/data/imagenet/val`, using `torchvision.datasets.ImageFolder` with `Resize(256)`, `CenterCrop(224)`, and `ToTensor()`.
- `data.datasets.cifar10` downloads/loads torchvision CIFAR-10 and returns a class-balanced validation subset.

## Model Loading

- `utils.attack_init.model_init` chooses the loader by `args.dataset_name`.
- `utils.load_models.load_imagenet_model` constructs torchvision models (`vgg16`, `resnet18`, `squeezenet`, `resnet50`, `inceptionv3`, `wrn50`, `resnext50`, `densenet121`, `vgg19`) and wraps them with ImageNet normalization via `NormalizeByChannelMeanStd`.
- `utils.load_models.load_cifar_model` constructs local CIFAR-10 architectures from `surro_models/cifar10_models` and loads checkpoints from `checkpoints/cifar10_target_models/` if present.
- `utils.load_models.load_generator` constructs `models.cglow.CondGlowModel`, loads `args.generator_path`, moves it to CUDA, and sets eval mode.

## Attack Selection

- `utils.attack_init.attacker_init` maps `args.attack_method` to:
  - `attacks.SquareAttack`
  - `attacks.SignHunter`
  - `attacks.CGAttack`
  - `attacks.MyAttack` with `use_mcg=True`
  - `attacks.MyAttack` with `use_mcg=False`
  - `attacks.HybridGA`
- `attacker_init` overwrites `args.class_num` and `args.linf` based on dataset: ImageNet gets `1000` and `0.05`; all non-ImageNet currently get `10` and `8/255`.
- `attack.run_single_variant` passes the selected attack an initial perturbation from MCG except for `my_attack_plain`; `cgattack` receives a generator-loss wrapper over latent space.

## Results And Logs

- `utils.attack_init.log_init` returns `args.log_root` if set, otherwise `./logs/{dataset}_{T|UT}_{target_model}_{attack_method}`.
- `attack.attack` writes:
  - text logs to `log_path`;
  - `runs_summary_randomvalid.csv` in the log directory;
  - `{run_id}_perimage_randomvalid.csv` in the log directory;
  - `{run_id}_valid_indices_seed{seed}_k{max_images}.txt` in the log directory.
- `trainners.Learner.Trainer` and `trainners.MetaLearner.Trainer` write training args, loss files, and checkpoints under `os.path.join(args.log_root, args.name or timestamp)`.
- `viz_attack_outputs.py` writes image artifacts under `--out_dir`.

## Paper Comparison Notes

- Matches the paper at a high level: c-Glow is the conditional generator; surrogate models guide generator adaptation; generated perturbations initialize or support black-box attacks.
- Differs from the paper in current evaluation behavior: `attack.py` collects randomly permuted correctly classified validation images via `build_valid_buffer_random`, while the paper describes fixed evaluation subsets: CIFAR-10 1,000 test images evenly across classes, ImageNet 10 selected classes with 500 validation images each.
- Differs from the paper in SurPGD support: the paper mentions PGD mainly as a pretraining/transfer baseline source for perturbations, not as the runtime SurPGD comparison/refinement pipeline implemented here.
- Differs from paper defaults in the current shell script: the paper states ImageNet surrogate ResNet-50, while `scripts/imagenet_attack_untargeted.sh` uses `vgg19`.

## Original Paper Evaluation Setup

From the paper's experimental settings:

- CIFAR-10: randomly select 1,000 images from the test set, covering all classes evenly. Images are resized to `32x32`.
- ImageNet: randomly select 10 classes from the 1,000 ImageNet classes, then use 500 validation images from each selected class. Images are resized to `224x224`.
- Meta-generator training data: full CIFAR-10 training set for CIFAR-10; training set of the 10 selected ImageNet classes for ImageNet.
- Perturbation limits with pixels rescaled to `[0, 1]`: `linf=0.031` for CIFAR-10 and `linf=0.05` for ImageNet.
- Query budget: 10,000 target queries in all experiments.
- Main reported metrics: ASR, mean query count, and median query count. The paper also uses FASR in ablations to measure direct success from the generated initial perturbation.
- Target models: CIFAR-10 uses ResNet-Preact-110, DenseNet-121, VGG-19, and PyramidNet-110. ImageNet uses ResNet-18, VGG-16-BN, WRN-50, and InceptionV3.
- Surrogate models: ResNet-18 for CIFAR-10 and ResNet-50 for ImageNet.

## Lightweight Verification

- `bash -n scripts/*.sh`: passed.
- `python3` AST parse passed for `attack.py`, `train.py`, `utils/attack_init.py`, `data/datasets.py`, `utils/load_models.py`, and `attacks/hybrid_ga.py`.
- `python3 attack.py --help` and `python3 train.py --help` could not run in the current shell because required Python packages are missing (`torch` for `attack.py`, `numpy` for `train.py`).
- `python --help` could not run because `python` is not installed on PATH; scripts invoke `python`, so the runtime environment needs either a `python` executable or script updates after environment policy is decided.
