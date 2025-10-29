import os
from pathlib import Path

import pytorch_lightning as pl

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

from src.data.datamodule import ImageFolderDataModule
from src.models.lightning_module import ImageNetLightningModule


def load_config(path: str):
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    cfg_path = os.environ.get("TRAIN_CONFIG", "configs/train.toml")
    cfg = load_config(cfg_path)

    data_cfg = cfg.get("data", {})
    aug_cfg = cfg.get("augment", {})
    model_cfg = cfg.get("model", {})
    optim_cfg = cfg.get("optim", {})
    sched_cfg = cfg.get("scheduler", {})
    trainer_cfg = cfg.get("trainer", {})

    seed = int(trainer_cfg.get("seed", 0))
    pl.seed_everything(seed, workers=True)

    dm = ImageFolderDataModule(
        train_dir=data_cfg["train_dir"],
        val_dir=data_cfg["val_dir"],
        batch_size=int(data_cfg.get("batch_size", 128)),
        num_workers=int(data_cfg.get("num_workers", 8)),
        image_size=int(data_cfg.get("image_size", 224)),
        policy=str(aug_cfg.get("policy", "randaugment")),
        rand_num_ops=int(aug_cfg.get("randaugment_num_ops", 2)),
        rand_magnitude=int(aug_cfg.get("randaugment_magnitude", 9)),
    )

    model = ImageNetLightningModule(
        model_name=str(model_cfg.get("name", "resnet50")),
        num_classes=int(model_cfg.get("num_classes", 1000)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        optim_name=str(optim_cfg.get("name", "adamw")),
        lr=float(optim_cfg.get("lr", 1e-3)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.05)),
        scheduler_name=str(sched_cfg.get("name", "cosine")),
        max_epochs=int(sched_cfg.get("max_epochs", trainer_cfg.get("max_epochs", 90))),
        warmup_epochs=int(sched_cfg.get("warmup_epochs", 5)),
    )

    precision = trainer_cfg.get("precision", "bf16")
    if isinstance(precision, int):
        precision = str(precision)

    ckpt_dir = Path(trainer_cfg.get("checkpoint_dir", "outputs/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = False

    trainer = pl.Trainer(
        max_epochs=int(trainer_cfg.get("max_epochs", 90)),
        devices=int(trainer_cfg.get("devices", 1)),
        precision=precision,
        accumulate_grad_batches=int(trainer_cfg.get("accumulate_grad_batches", 1)),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 50)),
        default_root_dir=str(ckpt_dir),
    )

    trainer.fit(model, datamodule=dm)

    trainer.validate(model, datamodule=dm)


if __name__ == "__main__":
    main()


