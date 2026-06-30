"""Model loading utilities for the paper's CIFAR-10 experiments."""

from __future__ import annotations

import os

import torch

from surro_models.blackboxbench_cifar10 import densenet as bbb_densenet
from surro_models.blackboxbench_cifar10 import inceptionv3 as bbb_inceptionv3
from surro_models.blackboxbench_cifar10 import resnet50 as bbb_resnet50
from surro_models.blackboxbench_cifar10 import vgg19_bn as bbb_vgg19_bn


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean.to(dtype=x.dtype, device=x.device)) / self.std.to(dtype=x.dtype, device=x.device)


def _load_state_dict_strict(model, checkpoint_path, state_dict_key="state_dict"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Required checkpoint not found: {checkpoint_path}. "
            "Download the external checkpoint bundle before running model-dependent scripts."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    state_dict = checkpoint[state_dict_key] if state_dict_key in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


def load_blackboxbench_cifar_model(model_name, home_path="checkpoints/blackboxbench_cifar10/ckpt"):
    if model_name == "bbb_vgg19_bn":
        pretrained_model = bbb_vgg19_bn(num_classes=10)
        model_checkpoint_path = os.path.join(home_path, "vgg19_bn", "model_best.pth.tar")
        pretrained_model.features = torch.nn.DataParallel(pretrained_model.features)
        _load_state_dict_strict(pretrained_model, model_checkpoint_path)
        pretrained_model.features = pretrained_model.features.module
    elif model_name == "bbb_densenet":
        pretrained_model = bbb_densenet(num_classes=10)
        model_checkpoint_path = os.path.join(home_path, "densenet-bc-L190-k40", "model_best.pth.tar")
        wrapped_model = torch.nn.DataParallel(pretrained_model)
        _load_state_dict_strict(wrapped_model, model_checkpoint_path)
        pretrained_model = wrapped_model.module
    elif model_name == "bbb_resnet50":
        pretrained_model = bbb_resnet50()
        model_checkpoint_path = os.path.join(
            os.path.dirname(home_path),
            "kaggle",
            "trained_models_cifar10",
            "resnet50_cifar10_lr01.pth",
        )
        _load_state_dict_strict(pretrained_model, model_checkpoint_path, state_dict_key="net")
    elif model_name == "bbb_inception_v3":
        pretrained_model = bbb_inceptionv3()
        model_checkpoint_path = os.path.join(
            os.path.dirname(home_path),
            "kaggle",
            "trained_models_cifar10",
            "inceptionv3_cifar10_lr01.pth",
        )
        _load_state_dict_strict(pretrained_model, model_checkpoint_path, state_dict_key="net")
    else:
        raise NotImplementedError(f"Model is not part of the paper release: {model_name}")

    normalize = NormalizeByChannelMeanStd(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    model = torch.nn.Sequential(normalize, pretrained_model)
    if torch.cuda.is_available():
        model = model.cuda()
    return model.eval()


def load_cifar_model(model_name, require_optim=False, defence_method=None):
    if defence_method is not None:
        raise NotImplementedError("Defense wrappers are not part of the paper release.")
    model = load_blackboxbench_cifar_model(model_name)
    if require_optim:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        return model, optimizer
    return model
