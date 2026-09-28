"""
inference/detect_image.py

Run vehicle detection + classification on a single image.

Draws bounding boxes, class labels, and confidence scores; prints and saves
per-class vehicle counts; saves the annotated output image.

Usage:
    python detect_image.py --weights ../models/best.pt --source ../test/sample_car.jpg
    python detect_image.py --weights ../models/best.pt --source ../test/sample.jpg --conf 0.4
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.counting import VehicleCounter, keep_top_detections  # noqa: E402

BOX_COLOR = (0, 200, 255)
TEXT_COLOR = (255, 255, 255)


def load_model(weights_path: str):
    if not Path(weights_path).exists():
        print(f"❌ Model weights not found at: {weights_path}\n"
              f"   Train a model first (training/train.py) or point --weights at a valid .pt file.")
        sys.exit(1)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics --break-system-packages")
        sys.exit(1)
    return YOLO(weights_path)


def draw_detections(image, boxes, class_names):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{class_names[cls_id]}: {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), BOX_COLOR, -1)
        cv2.putText(image, label, (x1 + 3, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    return image


def detect_image(weights_path: str, source_path: str, conf_threshold: float = 0.35,
                  out_path: str = None):
    source = Path(source_path)
    if not source.exists():
        print(f"❌ Invalid image path: {source}")
        sys.exit(1)

    image = cv2.imread(str(source))
    if image is None:
        print(f"❌ Could not read image (unsupported format or corrupt file): {source}")
        sys.exit(1)

    model = load_model(weights_path)
    class_names = model.names

    results = model.predict(source=str(source), conf=conf_threshold, verbose=False)
    result = results[0]
    boxes = keep_top_detections(result.boxes, max_detections=1)

    counter = VehicleCounter(list(class_names.values()) if isinstance(class_names, dict) else class_names)
    for box in boxes:
        counter.add_detection(int(box.cls[0]))

    annotated = draw_detections(image.copy(), boxes, class_names)

    out_path = out_path or f"../results/detected_{source.stem}.jpg"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, annotated)

    print(f"\n{'='*50}\nDETECTION RESULTS: {source.name}\n{'='*50}")
    if counter.total() == 0:
        print("No vehicles detected. Try lowering --conf or check the image.")
    else:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = class_names[cls_id] if isinstance(class_names, dict) else class_names[cls_id]
            print(f"  {name}: {conf:.2f}")
        print(f"\nVehicle counts:\n{counter.summary_text()}")
    print(f"\n✅ Saved annotated image to: {out_path}")
    return counter, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and classify vehicles in a single image")
    parser.add_argument("--weights", type=str, default="../models/best.pt")
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--out", type=str, default=None, help="Output image path")
    args = parser.parse_args()
    detect_image(args.weights, args.source, args.conf, args.out)
