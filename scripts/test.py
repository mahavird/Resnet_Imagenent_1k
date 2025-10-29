import os
from pathlib import Path

import pytorch_lightning as pl

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

from src.data.datamodule import ImageFolderDataModule
from src.models.lightning_module import ImageNetLightningModule


def load_config(path: str):
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    cfg_path = os.environ.get("TEST_CONFIG", "configs/test.toml")
    cfg = load_config(cfg_path)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    trainer_cfg = cfg.get("trainer", {})

    dm = ImageFolderDataModule(
        train_dir=data_cfg.get("val_dir", ""),  # unused
        val_dir=data_cfg["val_dir"],
        batch_size=int(data_cfg.get("batch_size", 128)),
        num_workers=int(data_cfg.get("num_workers", 8)),
        image_size=int(data_cfg.get("image_size", 224)),
        policy="none",
    )

    precision = trainer_cfg.get("precision", "bf16")
    if isinstance(precision, int):
        precision = str(precision)

    ckpt_path = model_cfg.get("checkpoint_path")
    assert ckpt_path, "checkpoint_path must be set in [model]"

    model = ImageNetLightningModule.load_from_checkpoint(ckpt_path)

    trainer = pl.Trainer(
        devices=int(trainer_cfg.get("devices", 1)),
        precision=precision,
    )

    trainer.validate(model, datamodule=dm)


if __name__ == "__main__":
    main()


