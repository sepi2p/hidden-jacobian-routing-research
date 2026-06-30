import h5py
import os
import numpy as np
import torch
import PIL.Image as Image
from torch.utils import data
import torchvision
import torch.nn as nn
import torch.utils.data
import json
from torch.utils.data import Dataset
from torchvision import datasets, transforms


PRE_100_CUR = [960, 1100, 1000, 960, 1140, 1200, 900, 960, 950, 930]


class RawIndexSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.raw_indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.raw_indices)

    def __getitem__(self, idx):
        return self.dataset[self.raw_indices[idx]]


class AdvTrainDataset(Dataset):
    def __init__(self, root_dir):
        print('Load imagenet dataset from:', root_dir)
        self.root_dir = root_dir
        self.transform = transforms.Compose([])
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        try:
            dataset = np.load(root_dir, allow_pickle=True).item()
        except:
            dataset = np.load(root_dir, allow_pickle=True)

        self.cln_imgs = dataset['cln_img']
        self.cln_labs = dataset['cln_lab']
        self.adv_imgs = dataset['adv_img']
        self.adv_labs = dataset['adv_lab']
        self.true_labs = dataset['true_lab']

        self.num = len(self.cln_imgs)
        print('data load done.')

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        cln_img, adv_img = self.cln_imgs[idx].copy(), self.adv_imgs[idx].copy()
        cln_lab, adv_lab, true_lab = self.cln_labs[idx], self.adv_labs[idx], self.true_labs[idx]
        # transform
        if self.transform is not None:
            # print(adv_img.shape)
            adv_img = self.transform(adv_img)
            cln_img = self.transform(cln_img)
            # adv_img = (adv_img - cln_img) not efficient here using minor
            # adv_img = ((adv_img - adv_img.min()) / (adv_img.max() - adv_img.min()) - 0.5) * 2
        if not torch.is_tensor(cln_img):
            cln_img = torch.from_numpy(cln_img)
        if not torch.is_tensor(adv_img):
            adv_img = torch.from_numpy(adv_img)
        cln_img = cln_img.float()
        adv_img = adv_img.float()
        if cln_img.max() > 2.0:
            cln_img = cln_img / 255.0
        if adv_img.max() > 2.0:
            adv_img = adv_img / 255.0
        return {
            "adv_img": adv_img,
            "cln_img": cln_img,
            "cln_lab": cln_lab,
            'true_lab': true_lab,
            "adv_lab": adv_lab
        }


def imagenet(root, mode='validation'):
    """
    ImageNet loader for local setup.

    We ignore `root` and always use the actual ImageNet val directory:
        /home/sepi/Study/coding/data/imagenet/val
    """
    from torchvision import datasets as tv_datasets, transforms

    if mode != 'validation':
        raise NotImplementedError("Only validation mode is supported in this setup.")

    val_root = "/home/sepi/Study/coding/data/imagenet/val"
    print(f"[INFO] Using ImageNet validation root: {val_root}")

    dataset = tv_datasets.ImageFolder(
        val_root,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    )
    return dataset

def cifar10(root, mode='train'):
    if mode == 'train':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    elif mode == 'validation':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        valid_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
        # Select 100 images from each class in CIFAR10
        # Note, this is the valid dataset pre_number == 100 cursor
        idx = torch.zeros(10000).bool()
        for i, pre_cur in enumerate(PRE_100_CUR):
            idx_i = torch.tensor(valid_set.targets) == i
            idx_i[pre_cur:] = False
            idx += idx_i
        dataset = torch.utils.data.dataset.Subset(valid_set, np.where(idx == 1)[0])
    elif mode == 'full_test':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    elif mode == 'unseen_test':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        full_test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
        seen = torch.zeros(10000).bool()
        targets = torch.tensor(full_test.targets)
        for i, pre_cur in enumerate(PRE_100_CUR):
            idx_i = targets == i
            idx_i[pre_cur:] = False
            seen += idx_i
        dataset = RawIndexSubset(full_test, np.where(seen.numpy() == 0)[0])
    else:
        raise NotImplementedError
    return dataset
