"""
test/test_utils.py

Basic unit tests for the utility modules (no GPU / trained model required).
Run with:
    cd test
    python -m pytest test_utils.py -v
  or simply:
    python test_utils.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from utils.counting import VehicleCounter, keep_top_detections  # noqa: E402
from utils.preprocessing import (  # noqa: E402
    normalize_image,
    resize_image,
    validate_label_file,
)

CLASS_NAMES = ["car", "bike", "bus", "truck", "auto_rickshaw", "van", "bicycle"]


def test_vehicle_counter_basic():
    counter = VehicleCounter(CLASS_NAMES)
    counter.add_detection(0)  # car
    counter.add_detection(0)  # car
    counter.add_detection(2)  # bus
    assert counter.total() == 3
    assert counter.summary()["car"] == 2
    assert counter.summary()["bus"] == 1
    assert counter.summary()["truck"] == 0
    print("test_vehicle_counter_basic passed")


def test_vehicle_counter_dedup_by_track_id():
    counter = VehicleCounter(CLASS_NAMES)
    counter.add_detection(0, track_id=101)
    counter.add_detection(0, track_id=101)  # same vehicle, later frame -> should not double count
    counter.add_detection(0, track_id=102)  # different vehicle
    assert counter.total() == 2
    print("test_vehicle_counter_dedup_by_track_id passed")


def test_keep_top_detections_single_vehicle():
    class FakeBox:
        def __init__(self, conf):
            self.conf = [conf]

    boxes = [FakeBox(0.25), FakeBox(0.9), FakeBox(0.7)]
    filtered = keep_top_detections(boxes, max_detections=1)
    assert len(filtered) == 1
    assert filtered[0].conf[0] == 0.9
    print("test_keep_top_detections_single_vehicle passed")


def test_resize_image_shape():
    dummy = np.random.randint(0, 255, (300, 500, 3), dtype="uint8")
    resized = resize_image(dummy, target_size=640)
    assert resized.shape == (640, 640, 3)
    print("test_resize_image_shape passed")


def test_normalize_image_range():
    dummy = np.random.randint(0, 255, (100, 100, 3), dtype="uint8")
    normalized = normalize_image(dummy)
    assert normalized.max() <= 1.0
    assert normalized.min() >= 0.0
    print("test_normalize_image_range passed")


def test_validate_label_file_missing(tmp_path=Path("/tmp")):
    fake_path = tmp_path / "does_not_exist.txt"
    problems = validate_label_file(fake_path, num_classes=7)
    assert len(problems) == 1
    assert "Missing label file" in problems[0]
    print("test_validate_label_file_missing passed")


def test_validate_label_file_bad_class_id(tmp_path=Path("/tmp")):
    label_path = tmp_path / "test_label.txt"
    label_path.write_text("9 0.5 0.5 0.2 0.2\n")  # class id 9 is out of range for 7 classes
    problems = validate_label_file(label_path, num_classes=7)
    assert any("out of range" in p for p in problems)
    label_path.unlink()
    print("test_validate_label_file_bad_class_id passed")


def test_validate_label_file_valid(tmp_path=Path("/tmp")):
    label_path = tmp_path / "test_label_valid.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.4 0.1 0.15\n")
    problems = validate_label_file(label_path, num_classes=7)
    assert problems == []
    label_path.unlink()
    print("test_validate_label_file_valid passed")


if __name__ == "__main__":
    test_vehicle_counter_basic()
    test_vehicle_counter_dedup_by_track_id()
    test_resize_image_shape()
    test_normalize_image_range()
    test_validate_label_file_missing()
    test_validate_label_file_bad_class_id()
    test_validate_label_file_valid()
    print("\n✅ All tests passed.")
