import os
from pathlib import Path
from typing import List
import csv

import torch
from PIL import Image
from torchvision import transforms

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

from src.models.imageclassifier import ImageClassifier


def load_config(path: str):
    with open(path, "rb") as f:
        return tomllib.load(f)


def list_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    acc: List[Path] = []
    for dp, _, fns in os.walk(path):
        for fn in fns:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                acc.append(Path(dp) / fn)
    return acc


def main():
    cfg_path = os.environ.get("INFER_CONFIG", "configs/infer.toml")
    cfg = load_config(cfg_path)
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("infer", {})
    data_cfg = cfg.get("data", {})

    ckpt_path = model_cfg.get("checkpoint_path")
    assert ckpt_path, "checkpoint_path must be set in [model]"

    model = ImageClassifier.load_from_checkpoint(ckpt_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image_size = int(data_cfg.get("image_size", 224))
    tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14), antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_path = Path(infer_cfg.get("input_path", "samples/"))
    out_csv = Path(infer_cfg.get("output_csv", "outputs/infer/predictions.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    images = list_images(input_path)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "pred_class", "confidence"])
        for img_path in images:
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    x = tf(im).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)
                        conf, pred = probs.max(dim=1)
                writer.writerow([str(img_path), int(pred.item()), float(conf.item())])
            except Exception:
                writer.writerow([str(img_path), "ERROR", 0.0])

    print(f"Wrote predictions to {out_csv}")


if __name__ == "__main__":
    main()


