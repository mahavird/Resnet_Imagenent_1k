import os
from datetime import datetime

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Python < 3.11

import torch
from torch import optim

from src.data.datasets import DataConfig, make_dataloaders
from src.engine.engine import train_one_epoch, evaluate
from src.utils.utils import set_seed, get_device, save_checkpoint
from src.models import resnet50

def main(cfg_path: str = "configs/experiment.toml"):
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)

    set_seed(cfg.get("seed", 42))
    device = get_device()

    # Data
    data_cfg = DataConfig(**cfg["data"])
    train_ld, val_ld, num_classes = make_dataloaders(data_cfg)

    # Model
    model = resnet50(num_classes=num_classes, **cfg.get("model", {})).to(device)

    # Optimizer & Scheduler
    opt_cfg = cfg["optim"]
    optimizer = optim.SGD(model.parameters(), lr=opt_cfg.get("lr", 0.1), momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("amp", True))

    best_acc = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        train_metrics = train_one_epoch(model, train_ld, optimizer, device, scaler=scaler)
        val_metrics = evaluate(model, val_ld, device)

        scheduler.step()

        print(f"Epoch {epoch+1}/{cfg['train']['epochs']}",
              f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f}",
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}")

        # Checkpoint
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            ckpt_name = f"checkpoints/best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            save_checkpoint({
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_acc,
                "config": cfg,
            }, ckpt_name)

if __name__ == "__main__":
    main()
