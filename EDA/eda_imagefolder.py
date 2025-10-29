import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from torchvision import datasets, transforms

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        tomllib = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_class_distribution(root_dir: Path) -> Tuple[Dict[str, int], List[str]]:
    """
    Walk an ImageFolder-style directory and count images per class.

    Returns:
      class_to_count: mapping class_name -> num_images
      classes: sorted list of class names (by name)
    """
    class_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    classes = sorted([p.name for p in class_dirs])
    class_to_count: Dict[str, int] = {}
    for cls in classes:
        cls_dir = root_dir / cls
        count = 0
        for dirpath, _, filenames in os.walk(cls_dir):
            for fname in filenames:
                # basic image file check by extension
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    count += 1
        class_to_count[cls] = count
    return class_to_count, classes


def plot_class_distribution(dist: Dict[str, int], title: str, out_path: Path, top_k: int = 30) -> None:
    items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items[:top_k]]
    counts = [v for _, v in items[:top_k]]
    plt.figure(figsize=(14, 6))
    sns.barplot(x=list(range(len(labels))), y=counts, color="#4e79a7")
    plt.title(title)
    plt.xlabel("Class (top by frequency)")
    plt.ylabel("Image count")
    plt.xticks(ticks=list(range(len(labels))), labels=[s[:12] for s in labels], rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def try_open_image(fp: Path) -> Tuple[bool, Tuple[int, int]]:
    try:
        with Image.open(fp) as im:
            im = im.convert("RGB")
            return True, im.size  # (width, height)
    except Exception:
        return False, (0, 0)


def sample_filepaths(root_dir: Path, max_images: int, seed: int = 0) -> List[Path]:
    rng = random.Random(seed)
    all_paths: List[Path] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                all_paths.append(Path(dirpath) / fname)
    if len(all_paths) <= max_images:
        return all_paths
    return rng.sample(all_paths, max_images)


def collect_size_and_aspect_stats(paths: List[Path]) -> Tuple[List[int], List[int], List[float], int]:
    widths: List[int] = []
    heights: List[int] = []
    aspects: List[float] = []
    corrupt = 0
    for p in tqdm(paths, desc="Scanning image sizes"):
        ok, size = try_open_image(p)
        if not ok:
            corrupt += 1
            continue
        w, h = size
        if w > 0 and h > 0:
            widths.append(w)
            heights.append(h)
            aspects.append(w / h)
    return widths, heights, aspects, corrupt


def plot_hist(data: List[float], title: str, xlabel: str, out_path: Path, bins: int = 50) -> None:
    plt.figure(figsize=(10, 5))
    sns.histplot(data, bins=bins, kde=False, color="#59a14f")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_channel_stats(paths: List[Path], max_images: int, resize_to: int = 256, crop_to: int = 224) -> Tuple[np.ndarray, np.ndarray, int]:
    transform = transforms.Compose([
        transforms.Resize(resize_to, antialias=True),
        transforms.CenterCrop(crop_to),
        transforms.ToTensor(),
    ])
    selected = paths if len(paths) <= max_images else paths[:max_images]
    pixels_sum = torch.zeros(3)
    pixels_sum_sq = torch.zeros(3)
    total_pixels = 0
    processed = 0
    for p in tqdm(selected, desc="Computing channel stats"):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                t = transform(im)  # CxHxW in [0,1]
        except Exception:
            continue
        c, h, w = t.shape
        num = h * w
        pixels_sum += t.reshape(c, -1).sum(dim=1)
        pixels_sum_sq += (t.reshape(c, -1) ** 2).sum(dim=1)
        total_pixels += num
        processed += 1
    if processed == 0 or total_pixels == 0:
        return np.zeros(3), np.ones(3), processed
    mean = (pixels_sum / total_pixels).numpy()
    var = (pixels_sum_sq / total_pixels - torch.tensor(mean) ** 2).numpy()
    std = np.sqrt(np.clip(var, 1e-12, None))
    return mean, std, processed


def show_sample_grid(root_dir: Path, out_path: Path, num_images: int = 16, seed: int = 0) -> None:
    paths = sample_filepaths(root_dir, max_images=num_images, seed=seed)
    cols = 4
    rows = max(1, (len(paths) + cols - 1) // cols)
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, p in enumerate(paths):
        plt.subplot(rows, cols, i + 1)
        ok, _ = try_open_image(p)
        if not ok:
            plt.text(0.5, 0.5, "Corrupt", ha="center", va="center")
            plt.axis("off")
            continue
        with Image.open(p) as im:
            im = im.convert("RGB")
            plt.imshow(im)
            plt.axis("off")
            plt.title(p.parent.name[:18])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def summarize_counts(label: str, class_counts: Dict[str, int]) -> str:
    total = sum(class_counts.values())
    nonzero = sum(1 for v in class_counts.values() if v > 0)
    return f"{label}: total_images={total}, nonempty_classes={nonzero}, classes={len(class_counts)}"


def run_eda(train_dir: Path, val_dir: Path, out_dir: Path, max_images_stats: int, seed: int) -> None:
    ensure_dir(out_dir)

    # Class distributions
    train_counts, train_classes = discover_class_distribution(train_dir)
    val_counts, val_classes = discover_class_distribution(val_dir)

    # Save class distribution plots
    plot_class_distribution(train_counts, "Train class distribution (top)", out_dir / "train_class_distribution.png")
    plot_class_distribution(val_counts, "Val class distribution (top)", out_dir / "val_class_distribution.png")

    # Sample grids
    show_sample_grid(train_dir, out_dir / "train_samples.png", num_images=16, seed=seed)
    show_sample_grid(val_dir, out_dir / "val_samples.png", num_images=16, seed=seed)

    # Size/aspect and corrupt stats (sampled)
    train_sample_paths = sample_filepaths(train_dir, max_images=max_images_stats, seed=seed)
    val_sample_paths = sample_filepaths(val_dir, max_images=max_images_stats, seed=seed + 1)

    tr_w, tr_h, tr_ar, tr_corrupt = collect_size_and_aspect_stats(train_sample_paths)
    va_w, va_h, va_ar, va_corrupt = collect_size_and_aspect_stats(val_sample_paths)

    # Histograms
    plot_hist(tr_w, "Train image widths", "width (px)", out_dir / "train_width_hist.png")
    plot_hist(tr_h, "Train image heights", "height (px)", out_dir / "train_height_hist.png")
    plot_hist(tr_ar, "Train aspect ratios (w/h)", "w/h", out_dir / "train_aspect_ratio_hist.png")
    plot_hist(va_w, "Val image widths", "width (px)", out_dir / "val_width_hist.png")
    plot_hist(va_h, "Val image heights", "height (px)", out_dir / "val_height_hist.png")
    plot_hist(va_ar, "Val aspect ratios (w/h)", "w/h", out_dir / "val_aspect_ratio_hist.png")

    # Channel stats
    tr_mean, tr_std, tr_n = compute_channel_stats(train_sample_paths, max_images=max_images_stats)
    va_mean, va_std, va_n = compute_channel_stats(val_sample_paths, max_images=max_images_stats)

    # Write summary
    summary_lines: List[str] = []
    summary_lines.append("ImageNet EDA Summary\n" + "=" * 20)
    summary_lines.append("")
    summary_lines.append(f"Train dir: {train_dir}")
    summary_lines.append(f"Val dir:   {val_dir}")
    summary_lines.append("")
    summary_lines.append(summarize_counts("Train", train_counts))
    summary_lines.append(summarize_counts("Val", val_counts))
    summary_lines.append("")
    summary_lines.append(f"Train corrupt (in sample {len(train_sample_paths)}): {tr_corrupt}")
    summary_lines.append(f"Val corrupt (in sample {len(val_sample_paths)}): {va_corrupt}")
    summary_lines.append("")
    summary_lines.append(f"Train channel mean: {np.round(tr_mean, 4).tolist()}  std: {np.round(tr_std, 4).tolist()}  (n={tr_n})")
    summary_lines.append(f"Val channel mean:   {np.round(va_mean, 4).tolist()}  std: {np.round(va_std, 4).tolist()}  (n={va_n})")
    summary_lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(summary_lines))


def load_config(config_path: Path) -> Dict[str, object]:
    if tomllib is None:
        raise RuntimeError(
            "TOML parser not available. Use Python 3.11+ (tomllib) or install 'tomli'."
        )
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as f:
        return tomllib.load(f)


def main() -> None:
    # Default config path; can be overridden by EDA_CONFIG env var
    config_path = Path(os.environ.get("EDA_CONFIG", "configs/eda.toml"))
    cfg = load_config(config_path)

    train_dir = Path(str(cfg.get("train_dir", "")))
    val_dir = Path(str(cfg.get("val_dir", "")))
    out_dir = Path(str(cfg.get("output_dir", "EDA/reports")))
    max_images_stats = int(cfg.get("max_images_stats", 2000))
    seed = int(cfg.get("seed", 0))

    ensure_dir(out_dir)
    run_eda(
        train_dir=train_dir,
        val_dir=val_dir,
        out_dir=out_dir,
        max_images_stats=max_images_stats,
        seed=seed,
    )
    print(f"EDA complete. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()


