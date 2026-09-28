# Vehicle Type Detection and Classification Using Deep Learning

An AI-based computer vision system that detects vehicles in images, videos,
and live webcam feeds, and classifies them into seven types: **Car, Bike,
Bus, Truck, Auto Rickshaw, Van, and Bicycle** — built on YOLO11, OpenCV,
PyTorch, and Streamlit.

---

## Abstract

Manual vehicle monitoring for traffic management, tolling, and parking is
slow and error-prone. This project builds an automated, deep-learning-based
system that detects vehicles from images, video, or a live webcam and
classifies each one into a specific vehicle type in real time, reporting
per-class counts and confidence scores through an interactive dashboard.

## Introduction

Vehicle detection and classification is a core computer vision task with
applications in traffic monitoring, smart tolling, parking management, and
urban planning. Deep learning object detectors — particularly the YOLO
("You Only Look Once") family — treat detection and classification as a
single regression problem, predicting bounding boxes and class labels in
one forward pass, which makes them fast enough for real-time video and
webcam use.

## Problem Statement

Design and implement a system that can:
1. Accept an image, video, or live webcam stream as input.
2. Detect all vehicles present.
3. Classify each detected vehicle into one of seven categories.
4. Report bounding boxes, class labels, confidence scores, and per-class
   counts.
5. Present all of this through an accessible web interface.

## Objectives

- Build a clean, modular deep learning pipeline: data prep → training →
  evaluation → inference → deployment.
- Achieve real-time (or near-real-time) inference speed suitable for video
  and webcam use.
- Provide a professional, interactive Streamlit interface with analytics.
- Support GPU acceleration with automatic CPU fallback.

## Existing System

Traditional vehicle counting relies on manual observation, inductive loop
sensors embedded in roads, or classical computer vision (background
subtraction, Haar cascades). These approaches are either labor-intensive,
expensive to install, or fail under varying lighting/occlusion and cannot
easily distinguish fine-grained vehicle types.

## Proposed System

A single-stage deep learning detector (YOLO11) trained on a labeled vehicle
dataset, wrapped in an inference pipeline that works uniformly across
images, video files, and live webcam streams, with a Streamlit dashboard
for interaction and analytics.

## Literature Survey

- **Redmon et al., "You Only Look Once" (2016)** introduced single-shot
  detection, framing detection as one regression from image pixels to
  bounding box coordinates and class probabilities.
- Subsequent YOLO versions (v5, v8, v11 from Ultralytics) improved backbone
  architectures, anchor-free detection heads, and training recipes,
  improving both speed and accuracy.
- **ByteTrack** and **DeepSORT** extend detectors with object tracking
  across frames, enabling unique-object counting in video rather than
  re-counting the same object every frame.

## System Architecture

```
Input Image/Video/Webcam
        |
        v
   Preprocessing
        |
        v
YOLO Deep Learning Model
        |
        v
  Vehicle Detection
        |
        v
 Vehicle Classification
        |
        v
Bounding Box + Confidence
        |
        v
  Vehicle Counting
        |
        v
Analytics / Dashboard
```

## Methodology

1. **Data collection & annotation** — gather vehicle images per class, label
   bounding boxes in YOLO format.
2. **Preprocessing** — split into train/val/test, resize, validate.
3. **Training** — fine-tune a pretrained YOLO11 checkpoint on the vehicle
   dataset.
4. **Evaluation** — measure precision, recall, F1, IoU, mAP@50, mAP@50:95;
   inspect the confusion matrix.
5. **Inference** — run the trained model on images, videos, and webcam
   streams with bounding-box drawing and counting.
6. **Deployment** — expose everything through a Streamlit web app.

## Dataset Description

The system expects a YOLO-format object detection dataset organized as:

```
dataset/
├── images/{train,val,test}/*.jpg
└── labels/{train,val,test}/*.txt   # one .txt per image, same basename
```

Each label line: `class_id x_center y_center width height` (all normalized
to `[0, 1]`), with `class_id` in `0..6` matching `config/dataset.yaml`.

**Where to get data:**
- Public sources: [Kaggle vehicle detection datasets](https://www.kaggle.com/datasets)
  (search "vehicle detection dataset", "traffic vehicle classification"),
  [Roboflow Universe](https://universe.roboflow.com) (many pre-annotated
  vehicle datasets exportable directly in YOLO format), or the COCO dataset
  (has car/bus/truck/motorcycle/bicycle classes you can filter and remap).
- Custom collection: dashcam footage, traffic-camera footage, or your own
  street photography, annotated with a tool like
  [LabelImg](https://github.com/heartexlabs/labelImg),
  [CVAT](https://www.cvat.ai/), or Roboflow's annotation UI (all can export
  YOLO-format labels directly).
- `auto_rickshaw` is uncommon in most public Western datasets — if targeting
  South Asian traffic scenes, you'll likely need to collect and annotate
  this class yourself, or use a Roboflow project that includes it.

## Data Preprocessing

Implemented in `utils/preprocessing.py`:
| Step | Why it's needed |
|---|---|
| Train/val/test split | Prevents data leakage; ensures evaluation reflects real generalization, not memorization. |
| Resizing (letterbox) | Normalizes input dimensions, catches corrupted images early, reduces storage/load time. |
| Normalization | Keeps gradients well-scaled during training (handled internally by YOLO's dataloader; exposed here for custom pipelines). |
| Augmentation | Increases effective dataset diversity (flips, color jitter, mosaic, mixup) to reduce overfitting — configured in `config/train_config.yaml`, previewed with `utils/augment_preview.py`. |
| Dataset/annotation validation | Catches malformed labels, out-of-range class IDs, and unreadable images before they silently corrupt a training run. |

## Model Architecture

YOLO11 (or YOLOv8, interchangeable via `config/train_config.yaml ->
model.base_weights`): a single-stage, anchor-free convolutional detector
with a CSP-based backbone, a feature-pyramid neck for multi-scale
detection, and a decoupled detection head predicting box coordinates,
objectness, and class probabilities simultaneously.

## Training Process

See `training/train.py`. Reads `config/dataset.yaml` (data) and
`config/train_config.yaml` (hyperparameters: epochs, batch size, image
size, learning rate, augmentation). Automatically detects and uses GPU
(CUDA) if available, otherwise falls back to CPU with a warning. Saves the
best checkpoint to `models/best.pt`.

## Evaluation Metrics

See `utils/metrics.py` for computation and full definitions of Precision,
Recall, F1-score, IoU, mAP@50, mAP@50:95, and the confusion matrix, plus
training/validation loss curve plotting.

## Results

*(Populate this section after training on your own dataset — run
`utils/metrics.py` to generate `results/evaluation_report.csv` and
`results/training_curves.png`, and Ultralytics' own confusion matrix plot
in the training run directory.)*

## Advantages

- Single unified pipeline for image, video, and webcam inference.
- Real-time capable on GPU; usable (slower) on CPU-only machines.
- Modular config-driven design — swap datasets, models, or hyperparameters
  without touching code.
- Interactive analytics dashboard for non-technical stakeholders.

## Limitations

- Model accuracy is entirely dependent on the quality and diversity of the
  training dataset provided by the user (not bundled with this codebase).
- Webcam mode in the Streamlit app requires a physically connected camera
  and a local (non-cloud) run; browser-based webcam capture isn't wired up
  in this reference build.
- Occlusion (vehicles overlapping heavily) and extreme weather/lighting can
  reduce detection accuracy, as with any vision model.
- Video-mode "unique vehicle" counting relies on ByteTrack's track IDs,
  which can occasionally fragment or merge tracks in dense traffic.

## Applications

- Smart traffic signal control based on live vehicle-type mix.
- Automated toll classification (car vs. truck vs. bus pricing tiers).
- Parking lot occupancy and vehicle-type analytics.
- Urban planning and traffic-flow studies.

## Future Scope

- Vehicle speed estimation and direction detection (see Stage 14 hooks).
- Traffic density heatmaps over time.
- License plate recognition integration.
- Edge deployment (Jetson Nano/Xavier) for standalone roadside units.

## Conclusion

This project delivers a complete, end-to-end deep learning pipeline for
vehicle detection and classification — from raw data preparation through
training, evaluation, and a production-style interactive interface —
built entirely on well-established, actively maintained open-source tools.

---

## Project Structure

```
vehicle-detection/
├── requirements.txt
├── README.md
├── config/
│   ├── dataset.yaml          # YOLO dataset definition (classes, paths)
│   └── train_config.yaml     # training hyperparameters
├── dataset/
│   ├── raw/                  # drop your raw collected images + labels here
│   └── images|labels/{train,val,test}/   # populated by preprocessing.py
├── training/
│   └── train.py               # main training script
├── models/
│   └── best.pt                # trained weights land here after training
├── inference/
│   ├── detect_image.py
│   ├── detect_video.py
│   └── detect_webcam.py
├── frontend/
│   └── app.py                 # Streamlit web application
├── utils/
│   ├── preprocessing.py       # split / resize / validate dataset
│   ├── augment_preview.py     # visualize augmentation effects
│   ├── counting.py            # shared vehicle counting logic
│   ├── metrics.py             # evaluation + plotting
│   └── model_loader.py        # cached model loading for the Streamlit app
├── results/                   # outputs: annotated images/video, plots, CSVs
└── test/
    └── test_utils.py          # unit tests (no GPU/model required)
```

## Installation

```bash
cd vehicle-detection
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> GPU users: install the CUDA build of PyTorch matching your driver from
> https://pytorch.org/get-started/locally/ *before* `pip install -r requirements.txt`,
> so `ultralytics` picks up the GPU-enabled torch instead of a CPU-only wheel.

## Dataset Preparation

1. Collect images per vehicle class and annotate them in YOLO format
   (see "Dataset Description" above for sources/tools).
2. Place all raw images + matching `.txt` label files together in
   `dataset/raw/` (flat folder, matching basenames: `car_001.jpg` +
   `car_001.txt`).
3. Run:
   ```bash
   cd utils
   python preprocessing.py --raw_dir ../dataset/raw --out_dir ../dataset
   ```
   This splits (70/20/10 by default), resizes, copies into the
   `images|labels/{train,val,test}` layout YOLO expects, and validates the
   result.
4. (Optional) Preview augmentation on a sample image:
   ```bash
   python augment_preview.py --image ../dataset/images/train/<some_image>.jpg
   ```

## Training

```bash
cd training
python train.py
# or override config values on the command line:
python train.py --epochs 50 --batch 8 --imgsz 640
```

Edit `config/train_config.yaml` to change the base model, epochs, batch
size, learning rate, or augmentation settings. Trained weights are copied
to `models/best.pt` automatically when training finishes.

## Evaluation

```bash
cd utils
python metrics.py --weights ../models/best.pt --data ../config/dataset.yaml
# after locating results.csv in the training run directory:
python metrics.py --results_csv ../results/train_runs/vehicle_yolo/results.csv
```

## Testing / Inference

**Image:**
```bash
cd inference
python detect_image.py --weights ../models/best.pt --source ../test/sample.jpg
```

**Video:**
```bash
python detect_video.py --weights ../models/best.pt --source ../test/traffic.mp4
```

**Webcam:**
```bash
python detect_webcam.py --weights ../models/best.pt
```

## Running the Web Application

```bash
cd frontend
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).
Set the model weights path and confidence threshold in the sidebar, then
use the Dashboard, Image Detection, Video Detection, Live Detection, and
Analytics pages.

## Unit Tests

```bash
cd test
python test_utils.py
```

## Example Output

```
DETECTION RESULTS: sample.jpg
==================================================
  car: 0.94
  bus: 0.91
  truck: 0.87

Vehicle counts:
car: 1
bus: 1
truck: 1
Total: 3

✅ Saved annotated image to: ../results/detected_sample.jpg
```

## Common Errors and Fixes

| Error | Fix |
|---|---|
| `Model weights not found` | Train a model first (`training/train.py`), or point `--weights` / the sidebar field at a valid `.pt` file. |
| `ultralytics not installed` | `pip install ultralytics --break-system-packages` (or plain `pip install ultralytics` in a venv). |
| `CUDA out of memory` | Lower `--batch` size or `--imgsz` in training, or use a smaller base model (`yolo11n.pt`). |
| `Could not open webcam` | Check the camera is connected, not in use by another app, and the process has camera permission. On headless servers, use `detect_video.py` with a file instead. |
| `dataset.yaml not found` / bad `path:` | Run `utils/preprocessing.py` first, and confirm `config/dataset.yaml`'s `path:` points at your prepared `dataset/` folder. |
| Corrupted / unreadable image or video | `utils/preprocessing.py --validate_only` will flag corrupted files; re-collect or remove them. |
| Low mAP / poor detections | Usually a small or imbalanced dataset — collect more images per class, verify annotation quality, and check `results/training_curves.png` for under/overfitting. |

## Future Improvements

- Integrate license plate recognition (ANPR) alongside vehicle classification.
- Add vehicle speed estimation using frame-to-frame displacement and camera calibration.
- Deploy on edge hardware (Jetson Nano) for standalone roadside traffic units.
- Add multi-camera support with cross-camera vehicle re-identification.
