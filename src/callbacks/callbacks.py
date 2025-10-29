from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor


def build_callbacks(trainer_cfg: Dict) -> List[pl.callbacks.Callback]:
    callbacks: List[pl.callbacks.Callback] = []

    monitor_metric = trainer_cfg.get("monitor", "val/acc1")
    mode = trainer_cfg.get("monitor_mode", "max")
    ckpt_dir = Path(trainer_cfg.get("checkpoint_dir", "outputs/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks.append(ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="epoch{epoch:03d}-valacc{val/acc1:.4f}",
        monitor=monitor_metric,
        mode=mode,
        save_top_k=int(trainer_cfg.get("save_top_k", 1)),
        save_last=True,
        auto_insert_metric_name=False,
    ))

    if trainer_cfg.get("early_stopping", False):
        callbacks.append(EarlyStopping(
            monitor=monitor_metric,
            mode=mode,
            patience=int(trainer_cfg.get("patience", 5)),
            min_delta=float(trainer_cfg.get("min_delta", 0.0)),
        ))

    if trainer_cfg.get("log_lr", True):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    return callbacks


