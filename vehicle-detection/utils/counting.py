"""
utils/counting.py

Shared vehicle-counting helpers used by image, video, and webcam inference,
and by the Streamlit app.
"""

from collections import defaultdict


def keep_top_detections(boxes, max_detections=1):
    """Return the strongest detections, sorted by confidence."""
    if not boxes:
        return []
    ranked = sorted(boxes, key=lambda box: float(box.conf[0]), reverse=True)
    return ranked[:max_detections]


class VehicleCounter:
    """Accumulates per-class detection counts across an image, or across
    frames of a video/webcam stream (with optional simple track-based
    de-duplication when a tracker is used).
    """

    def __init__(self, class_names):
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.counts = defaultdict(int)
        self.seen_track_ids = set()  # for de-duplicated counting across frames

    def add_detection(self, class_id, track_id=None):
        """Register one detected object.

        If track_id is provided (from ByteTrack/DeepSORT), the same physical
        vehicle is only counted once across frames, no matter how many times
        it appears. Without a track_id, every detection call increments the
        count (correct for single-image counting; for video this
        over-counts a vehicle across frames -- use track_id for video/webcam
        "unique vehicle" counts).
        """
        name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
        if track_id is not None:
            key = (name, track_id)
            if key in self.seen_track_ids:
                return
            self.seen_track_ids.add(key)
        self.counts[name] += 1

    def total(self):
        return sum(self.counts.values())

    def summary(self):
        """Returns {class_name: count} including zero-count classes, sorted by name."""
        return {name: self.counts.get(name, 0) for name in self.class_names}

    def summary_text(self):
        lines = [f"{name}: {count}" for name, count in self.summary().items() if count > 0]
        lines.append(f"Total: {self.total()}")
        return "\n".join(lines)
