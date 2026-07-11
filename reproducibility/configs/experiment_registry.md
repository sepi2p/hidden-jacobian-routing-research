# Experiment Registry

## Primary Protocol

- Dataset: CIFAR-10 clean-correct images.
- Natural models: `bbb_resnet50`, `bbb_vgg19_bn`, `bbb_densenet`, `bbb_inception_v3`.
- Exact split seeds: `1001`, `1002`, `1003`; split fractions: 40% basis fit, 20% layer validation, 40% final test.
- K&O candidate seeds: `0,1,2,3,4`; `k=20`; 12 power iterations; tolerance `1e-4`; signs `+/-`; step grid `1,2,4,6,8 / 255`.
- Small-probe JVP comparison: `epsilon=2/255`, probe `0.125/255`.
- Finite-budget diagnostic: PGD-CE20 and controlled APGD-style CE/DLR50 at `epsilon in {1,2}/255`.
- Difficulty control: image-grouped out-of-fold models using clean margin/loss, gradient norms, class, and first-step progress before adding transport energy.
- Norm-native comparator: approximate induced `(infinity,2)` maximization with five restarts, paired against signed Euclidean singular directions.
- Pullback avoidance: margin-PGD20, step `2/255`, three uniform random starts, `n=200`, primary `epsilon=1/255`; ResNet50 `8/255` rerouting check.
- Bootstrap unit: image ID unless a caption explicitly labels a fitted-basis point estimate.

The exact image rows, model registry, layer registry, and attack registry are tracked in `artifacts/splits/`.
`artifacts/splits/model_registry.csv` hashes the loaded model state after wrapping; `checkpoint_registry.csv` hashes the checkpoint files on disk. The two hashes intentionally measure different objects.

## Supporting Pilots

- ImageNet: 200 clean-correct validation images/model; ResNet50, ConvNeXt-Tiny, ViT-B/16; benchmark Square5000 at `4/255`. This tests attack-versus-random separation, not the full mechanism.
- RobustBench: 200 clean-correct CIFAR-10 images/model; Wong2020Fast, Engstrom2019Robustness, Addepalli2022Efficient_RN18; official APGD sanity checks plus local FD/JVP probes. This tests persistence of local mobility, not the full iterative mechanism.

## Access Scope

Mechanism and pullback experiments are white-box. The paper does not claim a practical or state-of-the-art attack.
