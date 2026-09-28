"""
training/train.py

Trains a YOLO model (YOLO11 by default) to detect and classify vehicles
into: car, bike, bus, truck, auto_rickshaw, van, bicycle.

Reads hyperparameters from config/train_config.yaml and the dataset
definition from config/dataset.yaml, so you rarely need to touch this file
directly -- just edit the YAML configs.

Automatically uses GPU (CUDA) if available, otherwise falls back to CPU
with a warning (CPU training will be much slower -- expect hours instead
of minutes per epoch on a real dataset).

Usage:
    cd training
    python train.py
    python train.py --epochs 50 --batch 8          # override config values
    python train.py --config ../config/train_config.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path_value: str, fallback_base: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()

    candidates = [
        (Path.cwd() / path).resolve(),
        (fallback_base / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_config(config_path: str) -> dict:
    path = resolve_path(config_path)
    if not path.exists():
        print(f"❌ Config file not found: {path.resolve()}")
        sys.exit(1)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if "data" in cfg and "data_yaml" in cfg["data"]:
        cfg["data"]["data_yaml"] = str((PROJECT_ROOT / "config" / "dataset.yaml").resolve())

    if "output" in cfg:
        project_dir = cfg["output"].get("project_dir")
        if project_dir and not Path(project_dir).is_absolute():
            cfg["output"]["project_dir"] = str(resolve_path(project_dir, PROJECT_ROOT))
    return cfg


def get_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA available -- training on GPU: {name}")
        return "0"
    print("⚠️  CUDA not available -- falling back to CPU. Training will be slow; "
          "consider reducing epochs/image size or using a smaller model (yolo11n.pt).")
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train YOLO vehicle detection/classification model")
    parser.add_argument("--config", type=str, default="config/train_config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size from config")
    parser.add_argument("--weights", type=str, default=None, help="Override base weights from config")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint in the run dir")
    args = parser.parse_args()

    cfg = load_config(args.config)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics is not installed. Run: pip install ultralytics --break-system-packages")
        sys.exit(1)

    data_yaml = cfg["data"]["data_yaml"]
    if not Path(data_yaml).exists():
        print(f"❌ dataset.yaml not found at {Path(data_yaml).resolve()}\n"
              f"   Run utils/preprocessing.py first to prepare the dataset, "
              f"and check the 'path' field in config/dataset.yaml points at it.")
        sys.exit(1)

    base_weights = args.weights or cfg["model"]["base_weights"]
    hp = cfg["hyperparameters"]
    aug = cfg.get("augmentation", {})
    out = cfg["output"]

    device = get_device()

    print(f"\nLoading base weights: {base_weights}")
    model = YOLO(base_weights)

    epochs = args.epochs or hp["epochs"]
    batch = args.batch or hp["batch_size"]
    imgsz = args.imgsz or hp["image_size"]

    print(f"\n{'='*60}\nSTARTING TRAINING\n{'='*60}")
    print(f"  epochs:       {epochs}")
    print(f"  batch_size:   {batch}")
    print(f"  image_size:   {imgsz}")
    print(f"  lr0:          {hp['learning_rate']}")
    print(f"  device:       {device}")
    print(f"{'='*60}\n")

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=hp["learning_rate"],
            lrf=hp.get("lrf", 0.01),
            optimizer=hp.get("optimizer", "auto"),
            patience=hp.get("patience", 20),
            workers=hp.get("workers", 8),
            seed=hp.get("seed", 42),
            device=device,
            project=out["project_dir"],
            name=out["run_name"],
            resume=args.resume,
            hsv_h=aug.get("hsv_h", 0.015),
            hsv_s=aug.get("hsv_s", 0.7),
            hsv_v=aug.get("hsv_v", 0.4),
            degrees=aug.get("degrees", 0.0),
            translate=aug.get("translate", 0.1),
            scale=aug.get("scale", 0.5),
            shear=aug.get("shear", 0.0),
            flipud=aug.get("flipud", 0.0),
            fliplr=aug.get("fliplr", 0.5),
            mosaic=aug.get("mosaic", 1.0),
            mixup=aug.get("mixup", 0.0),
            exist_ok=True,
        )
    except FileNotFoundError as e:
        print(f"❌ File not found during training: {e}")
        sys.exit(1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ CUDA out of memory. Try a smaller --batch size or smaller --imgsz.")
        else:
            print(f"❌ Runtime error during training: {e}")
        sys.exit(1)

    run_dir = Path(out["project_dir"]) / out["run_name"]
    best_weights = run_dir / "weights" / "best.pt"
    print(f"\n✅ Training complete. Results saved to: {run_dir}")

    if best_weights.exists():
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        dest = models_dir / "best.pt"
        dest.write_bytes(best_weights.read_bytes())
        print(f"✅ Copied best weights to: {dest.resolve()}")
        print(f"\nNext steps:\n"
              f"  Evaluate:   python ../utils/metrics.py --weights ../models/best.pt\n"
              f"  Image test: python ../inference/detect_image.py --weights ../models/best.pt --source <image>\n"
              f"  Video test: python ../inference/detect_video.py --weights ../models/best.pt --source <video>\n"
              f"  App:        streamlit run ../frontend/app.py")
    else:
        print("⚠️  Could not find best.pt in expected location -- check the run directory above.")


if __name__ == "__main__":
    main()
