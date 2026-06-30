import numpy as np
import os
import torch
import torchvision
import torch.backends.cudnn as cudnn
from advertorch.utils import NormalizeByChannelMeanStd
from models.cglow import CondGlowModel
from io import BytesIO
from PIL import Image

try:
    from surro_models.cifar10_models.resnet import ResNet18, ResNet34
    from surro_models.cifar10_models.vgg import VGG
    from surro_models.cifar10_models.pyramidnet import pyramid_net110
    from surro_models.cifar10_models.densenet import DenseNet121
    from surro_models.cifar10_models.preact_resnet import PreActResNet18
    from surro_models.blackboxbench_cifar10 import densenet as bbb_densenet
    from surro_models.blackboxbench_cifar10 import inceptionv3 as bbb_inceptionv3
    from surro_models.blackboxbench_cifar10 import resnet50 as bbb_resnet50
    from surro_models.blackboxbench_cifar10 import vgg19_bn as bbb_vgg19_bn
except Exception:
    pass


def load_generator(args):
    print(f'C-Glow path: {args.generator_path}')
    G = CondGlowModel(args)
    ckpt = torch.load(args.generator_path, map_location='cpu')['model']
    G.load_state_dict(ckpt)
    G = G.cuda()
    G.eval()
    return G


def freeze_part_parameters(model_name, model):
    print('Freeze model: ', model_name)
    if model_name == 'VGG19':
        for p in model.named_parameters():
            if p[0].startswith('classifier.') or p[0].startswith('features.5'):
                p[1].requires_grad = True
            else:
                p[1].requires_grad = False
    elif model_name == 'Resnet50':
        for p in model.named_parameters():
            if p[0].startswith('fc') or p[0].startswith('layer4') or p[0].startswith('layer3'):
                p[1].requires_grad = True
            else:
                p[1].requires_grad = False
    elif model_name == 'Densenet121':
        for p in model.named_parameters():
            if p[0].startswith('classifier.'):
                p[1].requires_grad = True
            else:
                p[1].requires_grad = False


def unfreeze_parameters(model):
    print('UnFreeze model: ', )
    for p in model.parameters():
        p.requires_grad = True


def _load_state_dict_strict(model, checkpoint_path, state_dict_key='state_dict'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Required checkpoint not found: {checkpoint_path}. "
            "BlackboxBench model names must not fall back to random weights."
        )

    print(f"[INFO] Loading BlackboxBench CIFAR-10 checkpoint from {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    )
    state_dict = checkpoint[state_dict_key] if state_dict_key in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


def load_blackboxbench_cifar_model(
    model_name,
    home_path='checkpoints/blackboxbench_cifar10/ckpt',
    require_optim=False,
):
    print('Load BlackboxBench cifar model: ', model_name)

    if model_name == 'bbb_vgg19_bn':
        pretrained_model = bbb_vgg19_bn(num_classes=10)
        model_checkpoint_path = os.path.join(home_path, 'vgg19_bn', 'model_best.pth.tar')
        # BlackboxBench's checkpoint was saved with only the feature extractor wrapped.
        pretrained_model.features = torch.nn.DataParallel(pretrained_model.features)
        _load_state_dict_strict(pretrained_model, model_checkpoint_path)
        pretrained_model.features = pretrained_model.features.module
    elif model_name == 'bbb_densenet':
        pretrained_model = bbb_densenet(num_classes=10)
        model_checkpoint_path = os.path.join(home_path, 'densenet-bc-L190-k40', 'model_best.pth.tar')
        # BlackboxBench's checkpoint was saved from a DataParallel model.
        wrapped_model = torch.nn.DataParallel(pretrained_model)
        _load_state_dict_strict(wrapped_model, model_checkpoint_path)
        pretrained_model = wrapped_model.module
    elif model_name == 'bbb_resnet50':
        pretrained_model = bbb_resnet50()
        model_checkpoint_path = os.path.join(
            os.path.dirname(home_path),
            'kaggle',
            'trained_models_cifar10',
            'resnet50_cifar10_lr01.pth',
        )
        _load_state_dict_strict(pretrained_model, model_checkpoint_path, state_dict_key='net')
    elif model_name == 'bbb_inception_v3':
        pretrained_model = bbb_inceptionv3()
        model_checkpoint_path = os.path.join(
            os.path.dirname(home_path),
            'kaggle',
            'trained_models_cifar10',
            'inceptionv3_cifar10_lr01.pth',
        )
        _load_state_dict_strict(pretrained_model, model_checkpoint_path, state_dict_key='net')
    else:
        raise NotImplementedError

    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
    model = torch.nn.Sequential(
        normalize,
        pretrained_model
    )

    model = model.cuda()
    model.eval()
    if require_optim:
        optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=1e-4)
        return model, optimizer
    return model


def load_imagenet_model(model_name, require_optim=False, defence_method=None):
    print('model_name: ', model_name)
    if model_name == "vgg16":
        pretrained_model = torchvision.models.vgg16_bn(pretrained=True)
    elif model_name == 'resnet18':
        pretrained_model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'squeezenet':
        pretrained_model = torchvision.models.squeezenet1_1(pretrained=True)
    elif model_name == 'resnet50':
        pretrained_model = torchvision.models.resnet50(pretrained=True)
    elif model_name == 'inceptionv3':
        print('model_name: ', model_name)
        pretrained_model = torchvision.models.inception_v3(pretrained=True)
    
        # Disable auxiliary logits head so training mode doesn't run AuxLogits
        pretrained_model.aux_logits = False
        pretrained_model.AuxLogits = None

    elif model_name == 'wrn50':
        pretrained_model = torchvision.models.wide_resnet50_2(pretrained=True)
    elif model_name == 'resnext50':
        pretrained_model = torchvision.models.resnext50_32x4d(pretrained=True)
    elif model_name == 'densenet121':
        pretrained_model = torchvision.models.densenet121(pretrained=True)
    elif model_name == 'vgg19':
        pretrained_model = torchvision.models.vgg19_bn(pretrained=True)
    else:
        raise NotImplementedError

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
    model = torch.nn.Sequential(
        normalize,
        pretrained_model
    )
    if defence_method == 'jpeg_compression':
        print('Defence method: jpeg compression', model_name)
        model = torch.nn.Sequential(
            JpegCompression(),
            model
        )
    elif defence_method == 'SND':
        print('Defence method: small noise defense', model_name)
        model = torch.nn.Sequential(
            SND(),
            model
        )
    elif defence_method is not None:
        raise NotImplementedError

    model = model.cuda()
    model.eval()
    if require_optim:
        # Only train the pre-trained part, not other part
        optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=3e-4)
        return model, optimizer
    return model


def load_cifar_model(model_name, home_path='checkpoints/cifar10_target_models/', require_optim=False, defence_method=None):
    if model_name.startswith('bbb_'):
        if defence_method is not None:
            raise NotImplementedError("Defence wrappers are not implemented for BlackboxBench CIFAR-10 models.")
        return load_blackboxbench_cifar_model(model_name, require_optim=require_optim)

    print('Load cifar model: ', model_name)
    if model_name == 'resnet18':
        pretrained_model = ResNet18()
        model_checkpoint_path = os.path.join(home_path, 'ResNet18_ckpt.t7')
    elif model_name == 'vgg13':
        pretrained_model = VGG('VGG13')
        model_checkpoint_path = os.path.join(home_path, 'VGG13_ckpt.t7')
    elif model_name == 'vgg19':
        pretrained_model = VGG('VGG19')
        model_checkpoint_path = os.path.join(home_path, 'VGG19_ckpt.t7')
    elif model_name == 'pyramidnet':
        pretrained_model = pyramid_net110()
        model_checkpoint_path = os.path.join(home_path, 'PyramidNet_ckpt.t7')
    elif model_name == 'densenet':
        pretrained_model = DenseNet121()
        model_checkpoint_path = os.path.join(home_path, 'DenseNet_ckpt.t7')
    elif model_name == 'preactresnet':
        pretrained_model = PreActResNet18()
        model_checkpoint_path = os.path.join(home_path, 'PreActResNet_ckpt.t7')
    elif model_name == 'norm_preactResnet':
        pretrained_model = PreActResNet18()
        model_checkpoint_path = os.path.join(home_path, 'cifar10_PreactResnet18_normed_ckpt.t7')
    elif model_name == 'norm_densenet':
        pretrained_model = DenseNet121()
        model_checkpoint_path = os.path.join(home_path, 'cifar10_Densenet121_normed_ckpt.t7')
    elif model_name == 'norm_vgg19':
        pretrained_model = VGG('VGG19')
        model_checkpoint_path = os.path.join(home_path, 'cifar10_VGG19_normed_ckpt.t7')
    elif model_name == 'norm_pyramidnet':
        pretrained_model = pyramid_net110()
        model_checkpoint_path = os.path.join(home_path, 'cifar10_Pyramidnet110_normed_ckpt.t7')
    elif model_name == 'norm_resnet18':
        pretrained_model = ResNet18()
        model_checkpoint_path = os.path.join(home_path, 'cifar10_Resnet18_normed_ckpt.t7')
    else:
        raise NotImplementedError

    # NEW: don’t crash if the checkpoint is missing
    if os.path.exists(model_checkpoint_path):
        print(f"[INFO] Loading CIFAR-10 target model from {model_checkpoint_path}")
        checkpoint = torch.load(
            model_checkpoint_path,
            map_location='cuda' if torch.cuda.is_available() else 'cpu'
        )
        try:
            pretrained_model.load_state_dict(checkpoint['net'])
        except Exception:
            pretrained_model.load_state_dict(checkpoint)
    else:
        print(
            f"[WARNING] CIFAR-10 checkpoint not found at {model_checkpoint_path}. "
            f"Using randomly initialized {model_name}. "
            "Results will NOT match the paper; this is just for pipeline testing."
        )

    if model_name.startswith('norm'):
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
        model = torch.nn.Sequential(
            normalize,
            pretrained_model
        )
    else:
        model = pretrained_model

    model = model.cuda()
    model.eval()
    if require_optim:
        optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=1e-4)
        return model, optimizer
    return model


class JpegCompression(torch.nn.Module):
    def __init__(self):
        # print('jpeg defence')
        super(JpegCompression, self).__init__()

    def forward(self, x):
        x = x.detach().cpu().numpy()
        x = np.transpose(x, (0, 2, 3, 1))
        x = x * 255
        x = x.astype("uint8")
        x_jpeg = np.zeros_like(x)

        for i in range(x.shape[0]):
            tmp_jpeg = BytesIO()
            x_image = Image.fromarray(x[i], mode='RGB')
            x_image.save(tmp_jpeg, format="jpeg", quality=100)  # 50 will be more easy to attack
            x_jpeg[i] = np.array(Image.open(tmp_jpeg))
            tmp_jpeg.close()

        x_jpeg = x_jpeg / 255.0
        x_jpeg = x_jpeg.astype(np.float32)
        x_jpeg = np.transpose(x_jpeg, (0, 3, 1, 2))
        x_jpeg = torch.from_numpy(x_jpeg).cuda()
        return x_jpeg


class SND(torch.nn.Module):
    # Small input Noise is enough to Defend against Query-based black-box attacks
    # https://arxiv.org/pdf/2101.04829.pdf
    def __init__(self):
        super(SND, self).__init__()
        self.sigma = 0.01

    def forward(self, x):
        x = x + self.sigma * torch.randn_like(x)
        return x
