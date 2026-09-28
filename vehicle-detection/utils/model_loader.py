"""
utils/model_loader.py

Shared, cached model-loading helper for the Streamlit app, with clear error
messages for the common failure cases (missing weights file, CUDA
unavailable, corrupted checkpoint).
"""

from pathlib import Path

import torch


class ModelLoadError(Exception):
    pass


def get_device_info():
    if torch.cuda.is_available():
        return {"device": "0", "name": torch.cuda.get_device_name(0), "type": "GPU"}
    return {"device": "cpu", "name": "CPU", "type": "CPU"}


def load_yolo_model(weights_path: str):
    """Load a YOLO model, raising ModelLoadError with a clear message on failure."""
    path = Path(weights_path)
    if not path.exists():
        raise ModelLoadError(
            f"Model weights not found at '{weights_path}'. "
            f"Train a model first with training/train.py, or place a .pt file at this path."
        )

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ModelLoadError(
            "The 'ultralytics' package is not installed. Run: "
            "pip install ultralytics --break-system-packages"
        )

    try:
        model = YOLO(str(path))
    except Exception as e:
        raise ModelLoadError(f"Failed to load model weights (file may be corrupted): {e}")

    return model
