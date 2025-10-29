from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import timm
from torchmetrics.classification import Accuracy


class ImageNetLightningModule(pl.LightningModule):
    def __init__(self, model_name: str = "resnet50", num_classes: int = 1000, pretrained: bool = True, optim_name: str = "adamw", lr: float = 1e-3, weight_decay: float = 0.05, scheduler_name: str = "cosine", max_epochs: int = 90, warmup_epochs: int = 5):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
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
        else:
            return optimizer


