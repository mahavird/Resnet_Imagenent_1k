from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import Accuracy

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


def build_backbone(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    name = name.lower()
    if name in ("resnet50", "resnet_50"):
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name in ("mobilenet_v3_large", "mobilenetv3_large", "mobilenetv3_large_100"):
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    raise ValueError(f"Unknown model name: {name}")


class ImageClassifier(pl.LightningModule):
    def __init__(self, model_name: str = "resnet50", num_classes: int = 1000, pretrained: bool = True, optim_name: str = "adamw", lr: float = 1e-3, weight_decay: float = 0.05, scheduler_name: str = "cosine", max_epochs: int = 90):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_backbone(model_name, num_classes, pretrained)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.val_acc5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc.update(logits, y)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc1", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc.update(logits, y)
        self.val_acc5.update(logits, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc1", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc5", self.val_acc5, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if self.hparams.optim_name.lower() == "sgd":
            optimizer = SGD(params, lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.weight_decay, nesterov=True)
        else:
            optimizer = AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        return optimizer


