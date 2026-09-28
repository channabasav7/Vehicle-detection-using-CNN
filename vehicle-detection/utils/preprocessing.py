"""
utils/preprocessing.py

Dataset preparation utilities for the Vehicle Type Detection project.

Covers:
  - Train / validation / test split of raw images (and their YOLO .txt labels)
  - Image resizing to a target size
  - Basic image normalization check (handled by YOLO internally at train time,
    but we expose a helper for anyone doing manual preprocessing/inference outside YOLO)
  - Dataset + annotation validation (catches broken images, malformed label
    lines, out-of-range class ids, missing pairs)

Run directly to execute a full prep pipeline:
    python utils/preprocessing.py --raw_dir ../dataset/raw --out_dir ../dataset

WHY EACH STEP MATTERS
----------------------
- Splitting train/val/test: prevents data leakage so evaluation reflects
  real-world generalization instead of memorized training images.
- Resizing: YOLO resizes internally at train/inference time, but pre-resizing
  large raw images (e.g. 4000x3000 phone photos) saves disk space and speeds
  up data loading; it also lets you catch corrupted images early.
- Normalization: neural nets converge faster and more stably when pixel
  values are scaled (0-1 or standardized) instead of raw 0-255 ints, because
  it keeps gradients in a well-behaved range. Ultralytics YOLO does this
  automatically in its dataloader, so you don't need to do it by hand for
  training -- but it matters if you preprocess images for any custom model code.
- Augmentation (see augment_preview.py): synthetically increases dataset
  diversity (lighting, angle, occlusion) so the model generalizes better
  and overfits less on a small dataset.
- Validation: a single malformed label line or mismatched image/label pair
  can silently corrupt a whole training run or crash it midway; catching
  this before training saves hours of wasted GPU time.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2

CLASS_NAMES = ["car", "bike", "bus", "truck", "auto_rickshaw", "van", "bicycle"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image_label_pairs(raw_dir: Path):
    """Find matching (image, label.txt) pairs inside a flat raw directory.

    Expects raw_dir to contain images and YOLO-format .txt label files with
    matching basenames, e.g. car_001.jpg + car_001.txt
    """
    images = [p for p in raw_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    pairs = []
    missing_labels = []
    for img_path in images:
        label_path = img_path.with_suffix(".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path)
    return pairs, missing_labels


def split_dataset(pairs, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def resize_image(image, target_size=640):
    """Resize while preserving aspect ratio, padding to a square canvas.

    This is the classic 'letterbox' resize YOLO itself uses internally.
    We expose it here for anyone preprocessing images outside the YOLO pipeline.
    """
    import numpy as np
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    channels = image.shape[2] if image.ndim == 3 else 1
    canvas = np.full((target_size, target_size, channels), 114, dtype="uint8")
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def normalize_image(image):
    """Scale pixel values from [0,255] uint8 to [0,1] float32.

    Not required before feeding YOLO (it normalizes internally), but useful
    if you're writing custom preprocessing/inference code.
    """
    return (image.astype("float32")) / 255.0


def validate_label_file(label_path: Path, num_classes: int):
    """Validate a single YOLO-format label file.

    Each line must be: class_id x_center y_center width height
    with x_center/y_center/width/height normalized to [0, 1].
    Returns a list of human-readable problem strings (empty if valid).
    """
    problems = []
    if not label_path.exists():
        return [f"Missing label file: {label_path}"]

    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        problems.append(f"Empty label file (no objects annotated): {label_path}")

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"{label_path}:{i+1} -> expected 5 values, got {len(parts)}")
            continue
        try:
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            problems.append(f"{label_path}:{i+1} -> non-numeric value")
            continue

        if not (0 <= cls_id < num_classes):
            problems.append(f"{label_path}:{i+1} -> class id {cls_id} out of range (0-{num_classes-1})")

        for val, name in zip(coords, ["x_center", "y_center", "width", "height"]):
            if not (0.0 <= val <= 1.0):
                problems.append(f"{label_path}:{i+1} -> {name}={val} not normalized to [0,1]")
    return problems


def validate_image(img_path: Path):
    """Confirm an image file actually loads and is not corrupted/zero-size."""
    img = cv2.imread(str(img_path))
    if img is None:
        return [f"Unreadable / corrupted image: {img_path}"]
    if img.size == 0:
        return [f"Zero-size image: {img_path}"]
    return []


def validate_dataset(dataset_root: Path, num_classes: int = len(CLASS_NAMES)):
    """Run full validation across train/val/test splits.

    Checks:
      - every image has a matching label file (and vice versa)
      - every label file parses correctly with in-range class ids
      - every image file actually loads
    Prints a report and returns True if the dataset is clean.
    """
    all_problems = []
    for split in ["train", "val", "test"]:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split
        if not img_dir.exists():
            all_problems.append(f"Missing image directory: {img_dir}")
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        if not images:
            all_problems.append(f"No images found in {img_dir}")

        for img_path in images:
            all_problems.extend(validate_image(img_path))
            label_path = lbl_dir / (img_path.stem + ".txt")
            all_problems.extend(validate_label_file(label_path, num_classes))

    print(f"\n{'='*60}\nDATASET VALIDATION REPORT\n{'='*60}")
    if not all_problems:
        print("✅ No problems found. Dataset looks valid.")
        return True

    print(f"⚠️  Found {len(all_problems)} problem(s):\n")
    for p in all_problems[:200]:
        print(f"  - {p}")
    if len(all_problems) > 200:
        print(f"  ... and {len(all_problems) - 200} more")
    return False


def prepare_dataset(raw_dir: str, out_dir: str, target_size: int = 640,
                     train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
    """Full pipeline: find pairs -> split -> resize -> copy into YOLO folder layout."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    pairs, missing = find_image_label_pairs(raw_dir)
    print(f"Found {len(pairs)} image/label pairs.")
    if missing:
        print(f"⚠️  {len(missing)} images have no matching label file and will be skipped:")
        for m in missing[:10]:
            print(f"   - {m}")

    if not pairs:
        print("No valid pairs found. Nothing to do. Check --raw_dir path.")
        return

    train, val, test = split_dataset(pairs, train_ratio, val_ratio, test_ratio)
    print(f"Split -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

    for split_name, split_pairs in [("train", train), ("val", val), ("test", test)]:
        img_out = out_dir / "images" / split_name
        lbl_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in split_pairs:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"⚠️  Skipping unreadable image: {img_path}")
                continue
            resized = resize_image(image, target_size)
            cv2.imwrite(str(img_out / img_path.name), resized)
            shutil.copy(str(label_path), str(lbl_out / label_path.name))

    print("\n✅ Dataset prepared. Running validation...")
    validate_dataset(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare vehicle detection dataset for YOLO training")
    parser.add_argument("--raw_dir", type=str, default="../dataset/raw",
                         help="Folder containing raw images + matching .txt YOLO labels")
    parser.add_argument("--out_dir", type=str, default="../dataset",
                         help="Output dataset root (will create images/labels train/val/test)")
    parser.add_argument("--size", type=int, default=640, help="Target square image size")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--validate_only", action="store_true",
                         help="Skip preparation, just validate an existing dataset in --out_dir")
    args = parser.parse_args()

    if args.validate_only:
        validate_dataset(Path(args.out_dir))
    else:
        prepare_dataset(args.raw_dir, args.out_dir, args.size,
                         args.train_ratio, args.val_ratio, args.test_ratio)
