"""
inference/detect_webcam.py

Real-time vehicle detection from the computer's webcam. This is a thin,
dedicated wrapper around the same logic as detect_video.py (which already
supports source=0 for webcam) -- kept as a separate entry point since the
project brief calls for a distinct webcam stage, and because it defaults to
showing a live preview window (video mode defaults to headless/batch).

Usage:
    python detect_webcam.py --weights ../models/best.pt
    python detect_webcam.py --weights ../models/best.pt --conf 0.4 --camera 1
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))
from inference.detect_video import detect_video  # noqa: E402


def check_camera_available(camera_index: int) -> bool:
    cap = cv2.VideoCapture(camera_index)
    ok = cap.isOpened()
    cap.release()
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time vehicle detection via webcam")
    parser.add_argument("--weights", type=str, default="../models/best.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0 = default webcam)")
    parser.add_argument("--out", type=str, default="../results/webcam_recording.mp4",
                         help="Where to save the recorded, annotated session")
    parser.add_argument("--no_display", action="store_true",
                         help="Run headless (no preview window), e.g. on a server without a display")
    args = parser.parse_args()

    if not check_camera_available(args.camera):
        print(f"❌ Webcam unavailable at index {args.camera}. "
              f"Check it's connected, not in use by another application, "
              f"and that this process has camera permission. "
              f"On a headless server, use --source with a video file via detect_video.py instead.")
        sys.exit(1)

    detect_video(
        weights_path=args.weights,
        source=str(args.camera),
        conf_threshold=args.conf,
        out_path=args.out,
        show=not args.no_display,
    )
