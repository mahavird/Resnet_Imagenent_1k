from typing import Dict
from tqdm import tqdm
import torch
from torch import nn

def train_one_epoch(model: nn.Module, loader, optimizer, device, scaler=None, loss_fn=None) -> Dict[str, float]:
    model.train()
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        pbar.set_postfix(loss=total_loss/total, acc=total_correct/total if total else 0.0)
    return {"loss": total_loss/total, "acc": total_correct/total}

@torch.no_grad()
def evaluate(model: nn.Module, loader, device, loss_fn=None) -> Dict[str, float]:
    model.eval()
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="val", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        pbar.set_postfix(loss=total_loss/total, acc=total_correct/total if total else 0.0)
    return {"loss": total_loss/total, "acc": total_correct/total}
