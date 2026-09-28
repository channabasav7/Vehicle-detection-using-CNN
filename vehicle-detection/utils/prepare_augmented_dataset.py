"""Merge the project dataset with the compatible classes from a YOLO export."""

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = Path(r"C:\Users\chann\Downloads\vehicle type det.v1i.yolov8")
OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "combined"

# Only map labels whose meaning is clear enough to preserve class quality.
SOURCE_TO_PROJECT = {
    "3-Wheeler": 0,
    "Bike.": 1,
    "Car": 3,
    "Minitruck": 6,
    "Motorbike": 1,
    "auto": 0,
    "bike": 1,
    "bus": 2,
    "car": 3,
    "heavy": 6,
    "motorcycle": 1,
    "rickshaw": 0,
    "scooty": 1,
    "semi-trailer": 6,
    "truck": 6,
}


def copy_existing_split(split: str) -> None:
    source_images = PROJECT_ROOT / "dataset" / "images" / split
    source_labels = PROJECT_ROOT / "dataset" / "labels" / split
    output_images = OUTPUT_ROOT / "images" / split
    output_labels = OUTPUT_ROOT / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    for image in source_images.iterdir():
        if image.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            name = f"original_{image.name}"
            shutil.copy2(image, output_images / name)
            label = source_labels / f"{image.stem}.txt"
            if label.exists():
                shutil.copy2(label, output_labels / f"{Path(name).stem}.txt")


def copy_source_split(source_split: str, output_split: str, class_names: list[str]) -> None:
    source_images = SOURCE_ROOT / source_split / "images"
    source_labels = SOURCE_ROOT / source_split / "labels"
    output_images = OUTPUT_ROOT / "images" / output_split
    output_labels = OUTPUT_ROOT / "labels" / output_split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    for image in source_images.iterdir():
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label = source_labels / f"{image.stem}.txt"
        if not label.exists():
            continue
        converted = []
        for line in label.read_text().splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            source_name = class_names[int(fields[0])]
            project_id = SOURCE_TO_PROJECT.get(source_name)
            if project_id is not None:
                converted.append(" ".join([str(project_id), *fields[1:]]))
        if converted:
            name = f"roboflow_{image.name}"
            shutil.copy2(image, output_images / name)
            (output_labels / f"{Path(name).stem}.txt").write_text("\n".join(converted) + "\n")


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE_ROOT}")
    class_names = [
        "0", "1", "2", "3", "3-Wheeler", "4", "Bike.", "Car", "Minitruck",
        "Motorbike", "ambulance", "auto", "bike", "bus", "car", "heavy",
        "license_plate", "medium", "motorcycle", "police vehicle", "rickshaw",
        "scooty", "semi-trailer", "small", "truck",
    ]
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    for split in ("train", "val", "test"):
        copy_existing_split(split)
    for source_split, output_split in (("train", "train"), ("valid", "val"), ("test", "test")):
        copy_source_split(source_split, output_split, class_names)
    for split in ("train", "val", "test"):
        count = len(list((OUTPUT_ROOT / "images" / split).glob("*")))
        print(f"{split}: {count} images")


if __name__ == "__main__":
    main()