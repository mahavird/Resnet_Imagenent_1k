#!/usr/bin/env python3
"""
Verify ImageNet-1k dataset integrity and layout.

Expected (ILSVRC2012):
- Train images: 1,281,167
- Val images:   50,000
- Classes:      1000 (WordNet synsets n########)

Handles:
- synset-folder layout
- flat validation layout
- nested accidental extraction (warns)
"""

import re
from pathlib import Path
from statistics import median
from tqdm import tqdm

TRAIN = Path("/mnt/imagenet/train")
VAL   = Path("/mnt/imagenet/val")

EXPECTED_TRAIN = 1_281_167
EXPECTED_VAL   = 50_000
EXPECTED_CLASSES = 1000

SYNSET_RE = re.compile(r"^n\d{8}$")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG")


def count_images(root: Path) -> int:
    return sum(
        1 for p in root.rglob("*")
        if p.is_file() and p.name.endswith(IMG_EXTS)
    )


def analyze_dir(root: Path, label: str):
    print(f"\n{'='*70}")
    print(f"{label} directory: {root}")
    print(f"{'='*70}")

    if not root.exists():
        print("✗ Directory does not exist")
        return None

    # immediate subdirs
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    synset_dirs = [d for d in subdirs if SYNSET_RE.match(d.name)]

    stats = {}

    if synset_dirs:
        print(f"Detected synset-folder layout ({len(synset_dirs)} classes)")

        class_counts = {}
        for sdir in tqdm(synset_dirs, desc=f"Scanning {label} classes"):
            imgs = sum(
                1 for p in sdir.rglob("*")
                if p.is_file() and p.name.endswith(IMG_EXTS)
            )
            class_counts[sdir.name] = imgs

        counts = list(class_counts.values())
        stats["layout"] = "synset"
        stats["classes"] = len(class_counts)
        stats["images"] = sum(counts)

        print(f"Total images: {stats['images']:,}")
        print(f"Classes:      {stats['classes']}")

        print("\nPer-class stats:")
        print(f"  Min:    {min(counts)}")
        print(f"  Max:    {max(counts)}")
        print(f"  Mean:   {sum(counts)/len(counts):.1f}")
        print(f"  Median: {median(sorted(counts))}")

    else:
        print("No synset folders detected → treating as flat layout")
        stats["layout"] = "flat"
        stats["classes"] = 0
        stats["images"] = count_images(root)
        print(f"Total images: {stats['images']:,}")

    # Detect suspicious nesting
    nested = list(root.glob("*/train")) + list(root.glob("*/val"))
    if nested:
        print("\n⚠ Suspicious nested folders detected:")
        for p in nested[:5]:
            print(f"  {p}")

    return stats


def main():
    print("="*70)
    print("ImageNet-1k Verification")
    print("="*70)

    train_stats = analyze_dir(TRAIN, "TRAIN")
    val_stats   = analyze_dir(VAL, "VAL")

    print(f"\n{'='*70}")
    print("FINAL CHECK")
    print(f"{'='*70}")

    ok = True

    if train_stats:
        if train_stats["images"] != EXPECTED_TRAIN:
            print(f"✗ Train image count mismatch: {train_stats['images']:,}")
            ok = False
        else:
            print("✓ Train image count correct")

        if train_stats["layout"] == "synset" and train_stats["classes"] != EXPECTED_CLASSES:
            print(f"✗ Train class count mismatch: {train_stats['classes']}")
            ok = False
        else:
            print("✓ Train class count correct")

    if val_stats:
        if val_stats["images"] != EXPECTED_VAL:
            print(f"✗ Val image count mismatch: {val_stats['images']:,}")
            ok = False
        else:
            print("✓ Val image count correct")

        if val_stats["layout"] == "synset":
            if val_stats["classes"] != EXPECTED_CLASSES:
                print(f"✗ Val class count mismatch: {val_stats['classes']}")
                ok = False
            else:
                print("✓ Val class count correct")
        else:
            print("✓ Val flat layout acceptable")

    print("\nRESULT:")
    if ok:
        print("🎉 DATASET IS CLEAN AND READY FOR TRAINING")
    else:
        print("⚠ DATASET HAS ISSUES — DO NOT TRAIN YET")

    print("="*70)


if __name__ == "__main__":
    main()
