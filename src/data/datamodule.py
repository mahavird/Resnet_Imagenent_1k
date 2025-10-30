import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytorch_lightning as pl


def build_transforms(image_size: int, policy: str = "randaugment", rand_num_ops: int = 2, rand_magnitude: int = 9):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_aug = []
    train_aug.append(transforms.Resize(int(image_size * 1.14), antialias=True))
    train_aug.append(transforms.CenterCrop(image_size))
    if policy == "randaugment":
        try:
            from torchvision.transforms import RandAugment
            train_aug = [transforms.Resize(int(image_size * 1.14), antialias=True), transforms.CenterCrop(image_size), RandAugment(num_ops=rand_num_ops, magnitude=rand_magnitude)]
        except Exception:
            pass
    elif policy == "autoaugment":
        try:
            from torchvision.transforms import AutoAugment, AutoAugmentPolicy
            train_aug = [transforms.Resize(int(image_size * 1.14), antialias=True), transforms.CenterCrop(image_size), AutoAugment(AutoAugmentPolicy.IMAGENET)]
        except Exception:
            pass
    train_aug.extend([transforms.ToTensor(), normalize])

    train_tf = transforms.Compose(train_aug)
    eval_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14), antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


class ImageFolderDataModule(pl.LightningDataModule):
    def __init__(self, train_dir: str, val_dir: str, batch_size: int = 128, num_workers: int = 8, image_size: int = 224, policy: str = "randaugment", rand_num_ops: int = 2, rand_magnitude: int = 9):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.policy = policy
        self.rand_num_ops = rand_num_ops
        self.rand_magnitude = rand_magnitude
        self.train_tf = None
        self.eval_tf = None
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: Optional[str] = None):
        self.train_tf, self.eval_tf = build_transforms(self.image_size, self.policy, self.rand_num_ops, self.rand_magnitude)
        if stage in (None, "fit"):
            self.train_ds = datasets.ImageFolder(self.train_dir, transform=self.train_tf)
            self.val_ds = datasets.ImageFolder(self.val_dir, transform=self.eval_tf)
        if stage == "validate" and self.val_ds is None:
            self.val_ds = datasets.ImageFolder(self.val_dir, transform=self.eval_tf)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)


