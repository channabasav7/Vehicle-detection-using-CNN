"""
utils/augment_preview.py

Lets you SEE what YOLO's training-time augmentation does to your images
before committing to a full training run. YOLO applies augmentation
on-the-fly during training (configured in config/train_config.yaml under
`augmentation:`), so you don't need to pre-generate augmented copies of your
dataset -- this script is purely a visual sanity check.

Why augmentation matters: with a limited dataset (a few hundred to a few
thousand images per class), the model can memorize specific lighting,
angles, or backgrounds instead of learning the general shape of "truck" vs
"bus". Random flips, color jitter, scaling, mosaic (stitching 4 images
together) and mixup force the model to learn more robust, generalizable
features and reduce overfitting.

Usage:
    python utils/augment_preview.py --image ../dataset/images/train/car_001.jpg
"""

import argparse
import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt


def build_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=0.5),
        A.Affine(rotate=(-5, 5), translate_percent=0.1, scale=(0.85, 1.15), p=0.6),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.RandomShadow(p=0.2),
    ])


def preview(image_path: str, n_samples: int = 6, save_path: str = None):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = build_transform()
    fig, axes = plt.subplots(2, (n_samples + 1) // 2, figsize=(4 * ((n_samples + 1) // 2), 8))
    axes = axes.flatten()

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i in range(1, n_samples):
        random.seed(i)
        augmented = transform(image=image_rgb)["image"]
        axes[i].imshow(augmented)
        axes[i].set_title(f"Augmented #{i}")
        axes[i].axis("off")

    plt.tight_layout()
    out_path = save_path or "../results/augmentation_preview.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved augmentation preview to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview YOLO-style augmentations on a sample image")
    parser.add_argument("--image", type=str, required=True, help="Path to a sample image")
    parser.add_argument("--n", type=int, default=6, help="Number of panels (1 original + n-1 augmented)")
    parser.add_argument("--out", type=str, default=None, help="Where to save the preview PNG")
    args = parser.parse_args()
    preview(args.image, args.n, args.out)
