# =============================================================================
# step1_data_pipeline.py — Download → Clean → Preprocess → Split
#
# Run this FIRST before training.
# What this script does, step by step:
#   1. Download PlantVillage via tensorflow_datasets
#   2. Filter only Udupi-relevant disease classes
#   3. Clean: remove corrupt / blurry / too-small images
#   4. Analyze class distribution and handle imbalance
#   5. Split into train / val / test sets (70 / 15 / 15)
#   6. Build tf.data pipelines with augmentation for training
#   7. Save a data summary report
# =============================================================================

import os
import sys
import shutil
import hashlib
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — DOWNLOAD PlantVillage via TensorFlow Datasets
# tensorflow_datasets downloads directly; no Kaggle key needed.
# Dataset: ~800 MB, ~54,000 images across 38 plant/disease combos.
# ──────────────────────────────────────────────────────────────────────────────

def download_plantvillage():
    """
    Download PlantVillage using tensorflow_datasets.
    Saves raw images to DATA_RAW_DIR organised by label.
    Returns a dict: {class_folder_name: [list_of_image_paths]}
    """
    print("\n" + "="*60)
    print("STEP 1 — Downloading PlantVillage Dataset")
    print("="*60)
    print("Source : tensorflow_datasets (plant_village)")
    print("This may take 5-15 minutes on first run...")

    # Load the full dataset (tfds caches it for future runs)
    ds_all, info = tfds.load(
        "plant_village",
        split="train",           # PlantVillage only has a 'train' split in tfds
        with_info=True,
        as_supervised=False,     # we need the label dict, not just (image, label)
        shuffle_files=False,
    )

    label_names = info.features["label"].names
    print(f"\nTotal images in PlantVillage : {info.splits['train'].num_examples:,}")
    print(f"Total classes                : {len(label_names)}")
    print("\nAll available classes:")
    for i, name in enumerate(label_names):
        print(f"  [{i:02d}] {name}")

    # Identify which label IDs we want (only Udupi-relevant classes)
    wanted_ids = {}
    for label_id, label_name in enumerate(label_names):
        if label_name in DISEASE_CLASSES:
            wanted_ids[label_id] = label_name

    print(f"\nFiltering to {len(wanted_ids)} relevant classes for Udupi...")

    # Save images to disk organised by class folder
    saved_paths = defaultdict(list)
    skipped = 0

    for sample in tqdm(ds_all, desc="Saving images"):
        label_id  = sample["label"].numpy()
        image_arr = sample["image"].numpy()   # shape (H, W, 3), uint8

        if label_id not in wanted_ids:
            skipped += 1
            continue

        class_folder = wanted_ids[label_id]
        save_dir = Path(DATA_RAW_DIR) / class_folder
        save_dir.mkdir(parents=True, exist_ok=True)

        # Unique filename based on content hash (avoids duplicates)
        img_hash = hashlib.md5(image_arr.tobytes()).hexdigest()[:12]
        save_path = save_dir / f"{img_hash}.jpg"

        if not save_path.exists():
            img_bgr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        saved_paths[class_folder].append(str(save_path))

    print(f"\nSaved images  : {sum(len(v) for v in saved_paths.values()):,}")
    print(f"Skipped (not needed): {skipped:,}")
    for cls, paths in saved_paths.items():
        print(f"  {cls:<35} {len(paths):>5} images")

    return dict(saved_paths)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — CLEAN THE RAW IMAGES
# Removes:
#   • Images smaller than MIN_IMAGE_SIZE_BYTES (corrupt / truncated)
#   • Images below MIN_PIXEL_DIMENSION (too small to be useful)
#   • Blurry images (Laplacian variance < MAX_BLUR_VARIANCE)
#   • Near-duplicate images (same MD5 hash)
# ──────────────────────────────────────────────────────────────────────────────

def is_blurry(image_bgr: np.ndarray, threshold: float = MAX_BLUR_VARIANCE) -> bool:
    """
    Laplacian variance method for blur detection.
    A focused image has sharp edges → high variance.
    Blurry image has smooth gradients → low variance.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance < threshold


def clean_dataset(raw_paths: dict) -> dict:
    """
    Clean raw images and copy survivors to DATA_CLEAN_DIR.
    Returns dict: {class_name: [clean_image_paths]}
    """
    print("\n" + "="*60)
    print("STEP 2 — Cleaning Dataset")
    print("="*60)

    clean_paths = defaultdict(list)
    removal_log = []

    for class_name, paths in raw_paths.items():
        print(f"\n  Cleaning: {class_name} ({len(paths)} images)")
        clean_dir = Path(DATA_CLEAN_DIR) / class_name
        clean_dir.mkdir(parents=True, exist_ok=True)

        seen_hashes = set()
        class_removed = Counter()

        for path in tqdm(paths, desc=f"    {class_name[:30]}", leave=False):
            reason = None

            # ── Check 1: File size ───────────────────────────────────────────
            file_size = os.path.getsize(path)
            if file_size < MIN_IMAGE_SIZE_BYTES:
                reason = f"too_small_file ({file_size}B)"

            else:
                # ── Load image ───────────────────────────────────────────────
                img = cv2.imread(path)
                if img is None:
                    reason = "unreadable"
                else:
                    h, w = img.shape[:2]

                    # ── Check 2: Pixel dimensions ────────────────────────────
                    if h < MIN_PIXEL_DIMENSION or w < MIN_PIXEL_DIMENSION:
                        reason = f"too_small_pixels ({w}x{h})"

                    # ── Check 3: Blur ────────────────────────────────────────
                    elif is_blurry(img):
                        reason = "blurry"

                    # ── Check 4: Duplicate (hash-based) ─────────────────────
                    else:
                        img_hash = hashlib.md5(img.tobytes()).hexdigest()
                        if img_hash in seen_hashes:
                            reason = "duplicate"
                        else:
                            seen_hashes.add(img_hash)

            if reason:
                class_removed[reason] += 1
                removal_log.append({"file": path, "class": class_name, "reason": reason})
            else:
                # Copy to clean directory
                dest = clean_dir / Path(path).name
                shutil.copy2(path, dest)
                clean_paths[class_name].append(str(dest))

        total_removed = sum(class_removed.values())
        print(f"    Kept: {len(clean_paths[class_name])} | Removed: {total_removed}")
        for reason, count in class_removed.items():
            print(f"      ↳ {reason}: {count}")

    # Save removal log
    log_df = pd.DataFrame(removal_log)
    log_path = os.path.join(LOGS_DIR, "cleaning_removal_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"\nRemoval log saved → {log_path}")
    print(f"Total removed: {len(removal_log)} images")

    return dict(clean_paths)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — ANALYZE CLASS DISTRIBUTION AND PLAN BALANCING
# ──────────────────────────────────────────────────────────────────────────────

def analyze_and_balance(clean_paths: dict) -> dict:
    """
    Visualize class distribution.
    If any class has < 200 images (rare for PlantVillage, but possible
    after filtering), flag it for oversampling during training.
    Returns the same dict (no files moved; balancing is done via class_weight).
    """
    print("\n" + "="*60)
    print("STEP 3 — Class Distribution Analysis")
    print("="*60)

    counts = {cls: len(paths) for cls, paths in clean_paths.items()}
    total  = sum(counts.values())

    print(f"\n{'Class':<35} {'Count':>6}  {'%':>6}")
    print("-" * 52)
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{cls:<35} {count:>6}  {count/total*100:>5.1f}%")
    print(f"{'TOTAL':<35} {total:>6}  100.0%")

    # ── Plot distribution ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    classes = list(counts.keys())
    values  = list(counts.values())
    colors  = plt.cm.Greens(np.linspace(0.4, 0.9, len(classes)))
    bars = ax.bar(range(len(classes)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=8, rotation=0)
    ax.set_ylabel("Number of Images")
    ax.set_title("KrishiAI — Class Distribution After Cleaning")
    ax.set_facecolor("#0f1f0f")
    fig.patch.set_facecolor("#0a150a")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(bar.get_height()), ha="center", va="bottom", fontsize=8, color="white")
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nDistribution plot saved → {plot_path}")

    # ── Compute class weights for imbalanced training ─────────────────────────
    max_count = max(counts.values())
    class_weights = {}
    for idx, cls in enumerate(sorted(counts.keys())):
        # Inversely proportional to frequency
        class_weights[idx] = max_count / counts[cls]

    print("\nClass weights (for imbalanced training):")
    for idx, cls in enumerate(sorted(counts.keys())):
        print(f"  [{idx}] {cls:<35} weight = {class_weights[idx]:.2f}")

    # Save weights
    with open(os.path.join(LOGS_DIR, "class_weights.json"), "w") as f:
        json.dump(class_weights, f, indent=2)

    return clean_paths, class_weights


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN / VAL / TEST SPLIT
# Stratified split: same proportion of each class in each subset.
# ──────────────────────────────────────────────────────────────────────────────

def stratified_split(clean_paths: dict) -> tuple[list, list, list]:
    """
    Stratified split into (train_pairs, val_pairs, test_pairs).
    Each pair is a tuple (image_path, class_index).
    """
    print("\n" + "="*60)
    print("STEP 4 — Stratified Train / Val / Test Split")
    print("="*60)

    all_classes = sorted(clean_paths.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    train_pairs, val_pairs, test_pairs = [], [], []

    rng = np.random.default_rng(RANDOM_SEED)

    for cls, paths in clean_paths.items():
        idx = class_to_idx[cls]
        paths_arr = np.array(paths)
        rng.shuffle(paths_arr)

        n      = len(paths_arr)
        n_test = max(1, int(n * TEST_SPLIT))
        n_val  = max(1, int(n * VAL_SPLIT))

        test_paths  = paths_arr[:n_test]
        val_paths   = paths_arr[n_test : n_test + n_val]
        train_paths = paths_arr[n_test + n_val:]

        train_pairs.extend([(p, idx) for p in train_paths])
        val_pairs.extend(  [(p, idx) for p in val_paths])
        test_pairs.extend( [(p, idx) for p in test_paths])

        print(f"  {cls:<35} train={len(train_paths):>4} val={len(val_paths):>4} test={len(test_paths):>4}")

    rng.shuffle(train_pairs)

    print(f"\n  TOTAL  train={len(train_pairs):>5} val={len(val_pairs):>5} test={len(test_pairs):>5}")

    # Save split manifest
    def save_manifest(pairs, name):
        df = pd.DataFrame(pairs, columns=["path", "label"])
        df.to_csv(os.path.join(LOGS_DIR, f"{name}_manifest.csv"), index=False)

    save_manifest(train_pairs, "train")
    save_manifest(val_pairs,   "val")
    save_manifest(test_pairs,  "test")

    # Save class index mapping
    with open(os.path.join(LOGS_DIR, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)
    with open(os.path.join(LOGS_DIR, "idx_to_class.json"), "w") as f:
        json.dump({v: k for k, v in class_to_idx.items()}, f, indent=2)

    print(f"\nManifests and class mappings saved → {LOGS_DIR}")
    return train_pairs, val_pairs, test_pairs, class_to_idx


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — BUILD tf.data PIPELINES
# ──────────────────────────────────────────────────────────────────────────────

def load_and_preprocess_image(path: str, label: int, augment: bool = False):
    """
    Load one JPEG, resize, normalize to [0,1].
    EfficientNetB0 expects (224, 224, 3) float32 in [0, 255] when using
    tf.keras.applications.efficientnet.preprocess_input,  OR [0,1] if we
    rescale ourselves. We use the built-in preprocessor inside the model.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)        # (224, 224, 3) float32
    img = tf.cast(img, tf.float32)               # keep in [0, 255] for EfficientNet

    # ── Augmentation (training only) ─────────────────────────────────────────
    if augment:
        # Random horizontal flip
        img = tf.image.random_flip_left_right(img)
        # Random brightness (simulates monsoon overcast ↔ harsh afternoon sun)
        img = tf.image.random_brightness(img, max_delta=0.3 * 255)
        # Random contrast
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        # Random saturation (simulate different soil / lighting conditions)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        # Random hue (minor)
        img = tf.image.random_hue(img, max_delta=0.05)
        # Clip back to valid range
        img = tf.clip_by_value(img, 0.0, 255.0)

    label_onehot = tf.one_hot(label, NUM_CLASSES)
    return img, label_onehot


def build_tf_datasets(train_pairs, val_pairs, test_pairs):
    """
    Converts split lists → tf.data.Dataset objects, ready for model.fit().
    Returns (train_ds, val_ds, test_ds)
    """
    print("\n" + "="*60)
    print("STEP 5 — Building tf.data Pipelines")
    print("="*60)

    AUTOTUNE = tf.data.AUTOTUNE

    def make_dataset(pairs, augment=False, shuffle=False):
        paths  = [p for p, _ in pairs]
        labels = [l for _, l in pairs]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(pairs), seed=RANDOM_SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda p, l: load_and_preprocess_image(p, l, augment),
            num_parallel_calls=AUTOTUNE
        )
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = make_dataset(train_pairs, augment=True,  shuffle=True)
    val_ds   = make_dataset(val_pairs,   augment=False, shuffle=False)
    test_ds  = make_dataset(test_pairs,  augment=False, shuffle=False)

    print(f"  train_ds : {len(train_pairs)} samples → {len(train_ds)} batches")
    print(f"  val_ds   : {len(val_pairs)} samples → {len(val_ds)} batches")
    print(f"  test_ds  : {len(test_pairs)} samples → {len(test_ds)} batches")

    # Quick sanity check: visualize one batch
    _visualize_batch(train_ds, "train_sample_batch")

    return train_ds, val_ds, test_ds


def _visualize_batch(ds, name):
    """Save a grid of 16 sample images from the dataset."""
    batch_imgs, batch_labels = next(iter(ds))
    batch_imgs  = batch_imgs.numpy()
    batch_labels = batch_labels.numpy()

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i >= len(batch_imgs): break
        img = batch_imgs[i].astype(np.uint8)
        label_idx = np.argmax(batch_labels[i])
        label_name = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else str(label_idx)
        ax.imshow(img)
        ax.set_title(label_name, fontsize=8, color="white")
        ax.axis("off")
        ax.set_facecolor("#0a150a")
    fig.patch.set_facecolor("#0a150a")
    plt.suptitle("Sample Training Batch (after augmentation)", color="white", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name}.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Sample batch plot saved → {PLOTS_DIR}/{name}.png")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       KrishiAI — Data Pipeline (Step 1 of 4)            ║")
    print("║   Download → Clean → Analyze → Split → Build Datasets   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. Download
    raw_paths = download_plantvillage()

    # 2. Clean
    clean_paths = clean_dataset(raw_paths)

    # 3. Analyze + class weights
    clean_paths, class_weights = analyze_and_balance(clean_paths)

    # 4. Split
    train_pairs, val_pairs, test_pairs, class_to_idx = stratified_split(clean_paths)

    # 5. Build tf.data datasets
    train_ds, val_ds, test_ds = build_tf_datasets(train_pairs, val_pairs, test_pairs)

    print("\n✅ Data pipeline complete!")
    print(f"   Class mapping  → {LOGS_DIR}/class_to_idx.json")
    print(f"   Class weights  → {LOGS_DIR}/class_weights.json")
    print(f"   Train manifest → {LOGS_DIR}/train_manifest.csv")
    print(f"\nNext step: python step2_train.py")
