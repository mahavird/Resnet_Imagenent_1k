#!/usr/bin/env python3
"""
Safe + deterministic ImageNet (Kaggle) downloader/stager/merger.

Key properties:
- NO shared "newest zip" guessing (each dataset downloads into its own zip dir)
- DOES NOT delete zips (you will delete manually)
- Extracts each dataset into its own staging folder under /mnt/imagenet/parts/<dataset>/
- Merges all staged parts into /mnt/imagenet/train with optional SHA1 dedup
- Extracts validation into /mnt/imagenet/val

Layout:
  /mnt/imagenet/zips/<dataset>/*.zip
  /mnt/imagenet/parts/<dataset>/(extracted files)
  /mnt/imagenet/train/  (merged)
  /mnt/imagenet/val/    (validation)
"""

import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Set

from tqdm import tqdm
import zipfile

# ------------------- CONFIG ------------------- #

BASE = Path("/mnt/imagenet")
ZIPS = BASE / "zips"
PARTS = BASE / "parts"
TRAIN = BASE / "train"
VAL = BASE / "val"

DATASETS_TRAIN = [
    "sautkin/imagenet1k0",
    "sautkin/imagenet1k1",
    "sautkin/imagenet1k2",
    "sautkin/imagenet1k3",
]
DATASETS_VAL = [
    "sautkin/imagenet1kvalid",
]

FORCE_CLEAN = False          # True wipes PARTS/TRAIN/VAL before running
DEDUP = True                 # True = SHA1 content dedup across ALL parts
HASH_CHUNK = 1024 * 1024     # 1MB read chunks for hashing
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ------------------- LOGGING ------------------- #

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run(cmd, check=True):
    p = subprocess.run(cmd, text=True, capture_output=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")
    return p

def ensure_tools():
    log("Checking tools (kaggle, unzip)")
    run(["kaggle", "--version"])
    run(["unzip", "-v"])

def ensure_empty_or_clean(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        if not FORCE_CLEAN:
            raise RuntimeError(f"{path} is not empty. Set FORCE_CLEAN=True or delete it manually.")
        log(f"Cleaning {path}")
        shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

# ------------------- DOWNLOAD + EXTRACT ------------------- #

def dataset_name(dataset: str) -> str:
    return dataset.split("/")[-1]  # imagenet1k0, imagenet1k1, ...

def kaggle_download_into_dataset_dir(dataset: str) -> Path:
    """
    Download a dataset into a dedicated zip directory:
      /mnt/imagenet/zips/<dataset_name>/
    Returns that directory path.
    """
    ds = dataset_name(dataset)
    ds_zip_dir = ZIPS / ds
    ds_zip_dir.mkdir(parents=True, exist_ok=True)

    log(f"Downloading {dataset} -> {ds_zip_dir}")
    run(["kaggle", "datasets", "download", "-d", dataset, "-p", str(ds_zip_dir)])

    zips = sorted(ds_zip_dir.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"No zip file found in {ds_zip_dir} after downloading {dataset}")

    # We return the directory, not a single zip, because Kaggle datasets sometimes ship multiple zips.
    return ds_zip_dir

def unzip_with_progress(zip_path: Path, dest: Path):
    log(f"Extracting {zip_path.name} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        files = z.namelist()
        for f in tqdm(files, desc=f"Unzipping {zip_path.name}", unit="files"):
            z.extract(f, dest)

def unzip_all_zips_in_dir(zip_dir: Path, dest: Path):
    zips = sorted(zip_dir.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"No zip files found in {zip_dir}")
    for z in zips:
        unzip_with_progress(z, dest)

# ------------------- MERGE + DEDUP ------------------- #

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_SUFFIXES

def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(HASH_CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def detect_class_dirs(root: Path):
    """
    Detect top-level class folders:
    - numeric: 00000..00999
    - synset:  n########
    Returns list of directories.
    """
    subs = [d for d in root.iterdir() if d.is_dir()]
    numeric = [d for d in subs if d.name.isdigit() and len(d.name) == 5]
    if len(numeric) >= 50:
        return sorted(numeric, key=lambda p: p.name)

    synset = [d for d in subs if d.name.startswith("n") and len(d.name) == 9 and d.name[1:].isdigit()]
    if len(synset) >= 50:
        return sorted(synset, key=lambda p: p.name)

    return []

def merge_parts_into_train(parts_root: Path, train_root: Path):
    log("Starting merge into final TRAIN directory")
    train_root.mkdir(parents=True, exist_ok=True)

    seen_hashes: Set[str] = set()
    moved = 0
    skipped = 0
    name_collisions = 0

    part_dirs = [p for p in sorted(parts_root.iterdir()) if p.is_dir()]
    if not part_dirs:
        raise RuntimeError(f"No staging parts found in {parts_root}")

    for part in part_dirs:
        log(f"Merging part: {part.name}")
        class_dirs = detect_class_dirs(part)
        if not class_dirs:
            raise RuntimeError(
                f"No recognizable class folders in {part}. "
                f"Expected numeric 00000.. or synset n########."
            )

        for cdir in tqdm(class_dirs, desc=f"Merging classes from {part.name}", unit="class"):
            out_cdir = train_root / cdir.name
            out_cdir.mkdir(parents=True, exist_ok=True)

            for img in cdir.rglob("*"):
                if not is_image(img):
                    continue

                h = None
                if DEDUP:
                    h = sha1_file(img)
                    if h in seen_hashes:
                        skipped += 1
                        continue
                    seen_hashes.add(h)

                target = out_cdir / img.name
                if target.exists():
                    # Handle filename collision (keep both)
                    name_collisions += 1
                    suffix = (h[:8] if (DEDUP and h is not None) else sha1_file(img)[:8])
                    target = out_cdir / f"{img.stem}_{suffix}{img.suffix}"

                shutil.move(str(img), str(target))
                moved += 1

        log(f"Completed {part.name}: moved={moved:,}, skipped_dupes={skipped:,}, name_collisions={name_collisions:,}")

    log(f"Merge complete. moved={moved:,}, skipped_dupes={skipped:,}, name_collisions={name_collisions:,}")

# ------------------- COUNT ------------------- #

def count_images(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if is_image(p))

# ------------------- MAIN ------------------- #

def main():
    ensure_tools()

    log("Preparing directories")
    ZIPS.mkdir(parents=True, exist_ok=True)  # we never auto-delete zips
    ensure_empty_or_clean(PARTS)
    ensure_empty_or_clean(TRAIN)
    ensure_empty_or_clean(VAL)

    # ---- TRAIN: download + stage extract each part ----
    for ds in DATASETS_TRAIN:
        ds_short = dataset_name(ds)
        log(f"=== TRAIN dataset: {ds_short} ===")
        zip_dir = kaggle_download_into_dataset_dir(ds)
        part_dest = PARTS / ds_short
        unzip_all_zips_in_dir(zip_dir, part_dest)

    # ---- TRAIN: merge staged parts into TRAIN ----
    merge_parts_into_train(PARTS, TRAIN)

    # ---- VAL: download + extract ----
    for ds in DATASETS_VAL:
        ds_short = dataset_name(ds)
        log(f"=== VAL dataset: {ds_short} ===")
        zip_dir = kaggle_download_into_dataset_dir(ds)
        unzip_all_zips_in_dir(zip_dir, VAL)

    # ---- Summary ----
    log("Counting final images (this can take a bit)")
    train_count = count_images(TRAIN)
    val_count = count_images(VAL)

    log("DONE ✔")
    log(f"TRAIN: {TRAIN}  images={train_count:,}")
    log(f"VAL:   {VAL}    images={val_count:,}")
    log(f"PARTS: {PARTS}")
    log(f"ZIPS:  {ZIPS} (not deleted; delete manually when you want)")
    log(f"DEDUP: {DEDUP}")

if __name__ == "__main__":
    main()
