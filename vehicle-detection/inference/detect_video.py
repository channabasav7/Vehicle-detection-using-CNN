"""
inference/detect_video.py

Runs frame-by-frame vehicle detection + classification on a video file
using OpenCV, drawing boxes/labels/confidence and tracking running counts.
Uses Ultralytics' built-in ByteTrack integration (model.track) so vehicles
get a persistent ID across frames -- this lets us report a "unique vehicles
seen" count in addition to a per-frame count, and is the same mechanism
the advanced tracking/counting features (Stage 14) build on.

Usage:
    python detect_video.py --weights ../models/best.pt --source ../test/traffic.mp4
    python detect_video.py --weights ../models/best.pt --source 0   # webcam via this script too
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.counting import VehicleCounter  # noqa: E402

BOX_COLOR = (0, 200, 255)
TEXT_COLOR = (255, 255, 255)


def load_model(weights_path: str):
    if not Path(weights_path).exists():
        print(f"❌ Model weights not found at: {weights_path}")
        sys.exit(1)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics --break-system-packages")
        sys.exit(1)
    return YOLO(weights_path)


def draw_frame(frame, boxes, class_names, counter: VehicleCounter, fps: float):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else None
        name = class_names[cls_id] if isinstance(class_names, dict) else class_names[cls_id]
        label = f"{name} {conf:.2f}" + (f" #{track_id}" if track_id is not None else "")

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 3, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

    y_off = 25
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_off += 25
    cv2.putText(frame, f"Unique vehicles: {counter.total()}", (10, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for name, count in counter.summary().items():
        if count > 0:
            y_off += 22
            cv2.putText(frame, f"  {name}: {count}", (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return frame


def detect_video(weights_path: str, source, conf_threshold: float = 0.35,
                  out_path: str = None, show: bool = False):
    model = load_model(weights_path)
    class_names = model.names
    class_list = list(class_names.values()) if isinstance(class_names, dict) else class_names

    source_str = str(source)
    is_webcam = source_str.isdigit()
    cap = cv2.VideoCapture(int(source_str) if is_webcam else source_str)
    if not cap.isOpened():
        if is_webcam:
            print("❌ Could not open webcam. Check that a camera is connected and not in use by another app.")
        else:
            print(f"❌ Could not open video file: {source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out_path = out_path or "../results/detected_video.mp4"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height))

    counter = VehicleCounter(class_list)
    frame_count = 0
    t_start = time.time()

    print("Processing video... press 'q' in the preview window to stop early (if --show is set).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        try:
            results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)
        except Exception as e:
            print(f"⚠️  Tracking failed on frame {frame_count} ({e}); falling back to plain detection.")
            results = model.predict(frame, conf=conf_threshold, verbose=False)

        result = results[0]
        boxes = result.boxes
        for box in boxes:
            track_id = int(box.id[0]) if box.id is not None else None
            counter.add_detection(int(box.cls[0]), track_id=track_id)

        elapsed = time.time() - t_start
        live_fps = frame_count / elapsed if elapsed > 0 else 0.0
        annotated = draw_frame(frame, boxes, class_names, counter, live_fps)

        writer.write(annotated)
        if show:
            cv2.imshow("Vehicle Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\n{'='*50}\nVIDEO DETECTION SUMMARY\n{'='*50}")
    print(f"Frames processed: {frame_count}")
    print(f"Unique vehicle counts:\n{counter.summary_text()}")
    print(f"\n✅ Saved annotated video to: {out_path}")
    return counter, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and classify vehicles in a video")
    parser.add_argument("--weights", type=str, default="../models/best.pt")
    parser.add_argument("--source", type=str, required=True, help="Path to video file, or 0 for webcam")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--show", action="store_true", help="Show a live preview window")
    args = parser.parse_args()
    detect_video(args.weights, args.source, args.conf, args.out, args.show)
