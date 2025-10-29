import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

root = "/mnt/imagenet"
splits = {"train": "train", "validation": "val"}

for split, outdir in splits.items():
    print(f"=== Downloading {split} split ===")
    ds = load_dataset("imagenet-1k", split=split)
    class_names = ds.features["label"].names
    out_root = os.path.join(root, outdir)
    os.makedirs(out_root, exist_ok=True)
    for ex in tqdm(ds, desc=f"Saving {split} images"):
        cls = class_names[ex["label"]]
        cls_dir = os.path.join(out_root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        fp = os.path.join(cls_dir, f"{cls}_{ex['id']}.jpg")
        ex["image"].convert("RGB").save(fp, "JPEG", quality=95)
