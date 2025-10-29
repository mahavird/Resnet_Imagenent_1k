import os, io, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from itertools import islice

import datasets
from datasets import load_dataset
from PIL import Image, ImageFile
from tqdm import tqdm

ROOT = "/mnt/imagenet"

# If unset -> FULL split; if set -> subset size
TRAIN_COUNT_ENV = os.environ.get("IMAGENET_TRAIN_COUNT")   # e.g., "12812"
VAL_COUNT_ENV   = os.environ.get("IMAGENET_VAL_COUNT")     # e.g., "2500"
TRAIN_COUNT = int(TRAIN_COUNT_ENV) if TRAIN_COUNT_ENV else None
VAL_COUNT   = int(VAL_COUNT_ENV)   if VAL_COUNT_ENV   else None

MAX_WORKERS = int(os.environ.get("IMAGENET_SAVE_WORKERS", "16"))
SHUFFLE     = os.environ.get("IMAGENET_STREAM_SHUFFLE", "1") == "1"
SHUFFLE_BUF = int(os.environ.get("IMAGENET_SHUFFLE_BUFFER", "8192"))

# New: resumable + bounded futures + progress
SKIP_IF_EXISTS = os.environ.get("IMAGENET_SKIP_IF_EXISTS", "1") == "1"
FLUSH_FACTOR   = int(os.environ.get("IMAGENET_FLUSH_FACTOR", "64"))   # chunk size = workers * factor
REPORT_EVERY   = int(os.environ.get("IMAGENET_REPORT_EVERY", "5000")) # live log interval

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate rare truncs

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def to_pil(img):
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, dict):
        if img.get("path"):  return Image.open(img["path"])
        if img.get("bytes"): return Image.open(io.BytesIO(img["bytes"]))
    raise TypeError(f"Unexpected image type: {type(img)}")

def save_example(ex, idx, out_root, class_names):
    cls = class_names[ex["label"]]
    cls_dir = os.path.join(out_root, cls)
    ensure_dir(cls_dir)
    fp = os.path.join(cls_dir, f"{cls}_{idx:09d}.jpg")  # no 'id' dependency
    if SKIP_IF_EXISTS and os.path.exists(fp):
        return fp
    tmp = fp + ".tmp"
    to_pil(ex["image"]).convert("RGB").save(tmp, "JPEG", quality=95)
    os.replace(tmp, fp)
    return fp

def take_n(it, n): return islice(it, n)

def precreate_class_dirs(class_names, roots):
    for root in roots:
        for cls in class_names:
            ensure_dir(os.path.join(root, cls))

def process_split(name, outdir, n_take, precreate=False, full_total_hint=None):
    mode = "FULL" if n_take is None else f"SUBSET({n_take})"
    print(f"=== Downloading streaming {name}: {mode} ===")

    ds = load_dataset("ILSVRC/imagenet-1k", split=name, streaming=True)
    ds = ds.cast_column("image", datasets.Image(decode=True))
    class_names = ds.features["label"].names

    out_root = os.path.join(ROOT, outdir); ensure_dir(out_root)
    if precreate: precreate_class_dirs(class_names, [out_root])
    if SHUFFLE:   ds = ds.shuffle(seed=42, buffer_size=SHUFFLE_BUF)

    save_fn = partial(save_example, out_root=out_root, class_names=class_names)

    successes = errors = 0
    chunk_limit = max(1, MAX_WORKERS * FLUSH_FACTOR)
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = []
        enumerator = enumerate(ds) if n_take is None else enumerate(take_n(ds, n_take))
        for idx, ex in enumerator:
            futures.append(pool.submit(save_fn, ex, idx))
            if len(futures) >= chunk_limit:
                for fut in as_completed(futures):
                    try:
                        fut.result()
                        successes += 1
                    except Exception as e:
                        errors += 1
                        if errors <= 5:
                            print(f"[{name}] save error: {e!r}")
                    if REPORT_EVERY and successes % REPORT_EVERY == 0:
                        elapsed = max(time.time() - start, 1e-6)
                        ips = successes / elapsed
                        target = (full_total_hint if n_take is None else n_take)
                        remaining = max(target - successes, 0)
                        eta_h = (remaining / ips / 3600) if ips > 0 and remaining > 0 else 0.0
                        print(f"[{name}] {successes}/{target} | {ips:.1f} img/s | ETA≈{eta_h:.2f}h")
                futures.clear()

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Saving {name} tail"):
            try:
                fut.result()
                successes += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"[{name}] save error: {e!r}")

    elapsed = max(time.time() - start, 1e-6)
    ips = successes / elapsed if successes else 0.0
    target = (full_total_hint if n_take is None else n_take)
    print(f"{name}: saved={successes}/{target}, errors={errors}, elapsed={elapsed/3600:.2f}h, throughput={ips:.1f} img/s")

def main():
    # Precreate both splits for consistent class_to_idx
    tmp = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    tmp = tmp.cast_column("image", datasets.Image(decode=True))
    class_names = tmp.features["label"].names
    precreate_class_dirs(class_names, [os.path.join(ROOT, "train"), os.path.join(ROOT, "val")])

    FULL_TRAIN, FULL_VAL = 1_281_167, 50_000
    process_split("train", "train", TRAIN_COUNT, precreate=False, full_total_hint=FULL_TRAIN)
    process_split("validation", "val", VAL_COUNT, precreate=False, full_total_hint=FULL_VAL)
    print("Done.")

if __name__ == "__main__":
    main()
