"""
utils/metrics.py

Post-training evaluation: pulls metrics out of a trained Ultralytics YOLO
model, prints a readable report, and generates the required plots
(training/validation loss curves, precision, recall, mAP, confusion matrix).

METRIC DEFINITIONS (for the project report / viva):
- Precision: of everything the model labeled as class X, what fraction was
  actually class X. High precision = few false positives.
- Recall: of everything that actually was class X, what fraction did the
  model find. High recall = few false negatives (missed vehicles).
- F1-score: harmonic mean of precision and recall -- a single number that
  penalizes models that only do well on one of the two.
- IoU (Intersection over Union): overlap between a predicted box and the
  ground-truth box, divided by their union area. A detection is usually
  only counted "correct" if IoU exceeds a threshold (commonly 0.5).
- mAP@50: mean Average Precision at an IoU threshold of 0.5 -- averages
  per-class AP (area under the precision-recall curve) at a lenient overlap
  requirement. Good headline number for "does it find the right class
  roughly in the right place".
- mAP@50:95: mAP averaged over IoU thresholds from 0.5 to 0.95 in steps of
  0.05 -- a stricter, more comprehensive measure that also rewards tightly
  fitting boxes, not just roughly-correct ones. This is the standard COCO
  metric and the best single number for comparing detector quality.
- Confusion matrix: shows which classes get mistaken for which (e.g. "van"
  predicted as "car") -- useful for spotting systematic class confusion the
  aggregate numbers hide.

Usage:
    python utils/metrics.py --weights ../models/best.pt --data ../config/dataset.yaml
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def evaluate_model(weights_path: str, data_yaml: str, out_dir: str = "../results"):
    from ultralytics import YOLO  # imported here so this module can be read/linted without ultralytics installed

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_yaml).resolve()
    dataset_config = yaml.safe_load(data_path.read_text())
    dataset_config["path"] = str(data_path.parent.parent)
    resolved_data_path = data_path.with_name(f".{data_path.stem}.resolved.yaml")
    resolved_data_path.write_text(yaml.safe_dump(dataset_config, sort_keys=False))

    model = YOLO(weights_path)
    try:
      metrics = model.val(data=str(resolved_data_path), split="test", plots=True, save_json=True)
    finally:
      resolved_data_path.unlink(missing_ok=True)

    print(f"\n{'='*60}\nEVALUATION REPORT\n{'='*60}")
    print(f"mAP@50:      {metrics.box.map50:.4f}")
    print(f"mAP@50:95:   {metrics.box.map:.4f}")
    print(f"Precision:   {metrics.box.mp:.4f}")
    print(f"Recall:      {metrics.box.mr:.4f}")

    precision, recall = metrics.box.mp, metrics.box.mr
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"F1-score:    {f1:.4f}")

    print("\nPer-class AP@50:")
    class_names = model.names
    for i, ap in enumerate(metrics.box.ap50):
        name = class_names.get(i, str(i)) if isinstance(class_names, dict) else class_names[i]
        print(f"  {name:15s}: {ap:.4f}")

    print(f"\nUltralytics also saved a confusion matrix and PR/F1 curve plots to: {metrics.save_dir}")

    report_path = out_dir / "evaluation_report.csv"
    pd.DataFrame([{
        "mAP50": metrics.box.map50,
        "mAP50_95": metrics.box.map,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }]).to_csv(report_path, index=False)
    print(f"Saved summary CSV to {report_path}")

    return metrics


def plot_training_curves(results_csv: str, out_dir: str = "../results"):
    """Plot training/validation loss, precision, recall, and mAP curves from
    the results.csv that Ultralytics automatically writes during training
    (found inside the run's save_dir, e.g. results/train_runs/vehicle_yolo/results.csv).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    sns.set_style("darkgrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    loss_cols_train = [c for c in df.columns if "train" in c and "loss" in c]
    loss_cols_val = [c for c in df.columns if "val" in c and "loss" in c]
    for c in loss_cols_train:
        axes[0, 0].plot(df["epoch"], df[c], label=c)
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    for c in loss_cols_val:
        axes[0, 1].plot(df["epoch"], df[c], label=c, color="orange")
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    prec_col = next((c for c in df.columns if "precision" in c.lower()), None)
    rec_col = next((c for c in df.columns if "recall" in c.lower()), None)
    if prec_col:
        axes[1, 0].plot(df["epoch"], df[prec_col], label="Precision", color="green")
    if rec_col:
        axes[1, 0].plot(df["epoch"], df[rec_col], label="Recall", color="red")
    axes[1, 0].set_title("Precision / Recall")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()

    map50_col = next((c for c in df.columns if "map50" in c.lower() and "95" not in c), None)
    map5095_col = next((c for c in df.columns if "map50-95" in c.lower() or "map50_95" in c.lower()), None)
    if map50_col:
        axes[1, 1].plot(df["epoch"], df[map50_col], label="mAP@50", color="purple")
    if map5095_col:
        axes[1, 1].plot(df["epoch"], df[map5095_col], label="mAP@50:95", color="brown")
    axes[1, 1].set_title("mAP")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()

    plt.tight_layout()
    save_path = out_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved training curves to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO vehicle detection model")
    parser.add_argument("--weights", type=str, default="../models/best.pt")
    parser.add_argument("--data", type=str, default="../config/dataset.yaml")
    parser.add_argument("--results_csv", type=str, default=None,
                         help="Path to results.csv from a training run, to plot loss/mAP curves")
    parser.add_argument("--out_dir", type=str, default="../results")
    args = parser.parse_args()

    if args.results_csv:
        plot_training_curves(args.results_csv, args.out_dir)
    else:
        evaluate_model(args.weights, args.data, args.out_dir)
