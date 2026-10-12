"""Tables and figures for the paired WHU external-replication pilot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch

from .model import build_model
from .training import restore_checkpoint, select_device
from .training_data import WHUTensorStore
from .whu_data import load_whu_binary_target, read_whu_image, read_whu_mask
from .whu_pilot import load_whu_pilot_runs


INITIALIZATION_LABELS = {"random": "Random", "imagenet": "ImageNet"}


def collect_whu_pilot_tables(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect compact result, history and per-image tables from saved runs."""
    result_rows, history_frames, metric_frames = [], [], []
    for run in load_whu_pilot_runs(project_root):
        summary = run["summary"]
        validation = summary["best_validation"]
        training = summary["matched_training_evaluation"]
        groups = validation["reference_groups"]
        history = pd.DataFrame(run["history"])
        for column in history.columns:
            if column != "initialization":
                history[column] = pd.to_numeric(history[column], errors="coerce")
        last = history.iloc[-1]
        result_rows.append({
            "initialization": run["initialization"],
            "best_step": int(summary["best_step"]),
            "best_validation_iou": validation["mean_iou"],
            "last_validation_iou": float(last.validation_mean_iou),
            "last_validation_loss": float(last.validation_loss),
            "selected_validation_dice": validation["mean_dice"],
            "selected_validation_ap_positive": validation["mean_average_precision"],
            "ap_image_count": validation["ap_image_count"],
            "positive_image_count": groups["positive_reference"]["image_count"],
            "positive_mean_iou": groups["positive_reference"]["mean_iou"],
            "positive_mean_dice": groups["positive_reference"]["mean_dice"],
            "empty_image_count": groups["empty_reference"]["image_count"],
            "empty_mean_predicted_building_fraction": groups["empty_reference"]["mean_predicted_building_fraction"],
            "empty_fraction_with_any_prediction": groups["empty_reference"]["fraction_with_any_predicted_building"],
            "selected_training_iou": training["mean_iou"],
            "training_validation_iou_gap": training["mean_iou"] - validation["mean_iou"],
            "training_seconds": summary["training_seconds"],
            "evaluation_seconds_during_training": summary["evaluation_seconds_during_training"],
            "total_seconds": summary["total_seconds"],
        })
        history.insert(0, "initialization", run["initialization"])
        history_frames.append(history)
        metrics = pd.DataFrame(run["metrics"])
        for column in ("tp", "fp", "fn", "iou", "dice", "average_precision"):
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
        metrics.insert(0, "initialization", run["initialization"])
        metric_frames.append(metrics)
    return (
        pd.DataFrame(result_rows).sort_values("initialization"),
        pd.concat(history_frames, ignore_index=True),
        pd.concat(metric_frames, ignore_index=True),
    )


def choose_whu_validation_examples(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select common success/failure cases symmetrically from both models.

    Positive-reference success and failure maximize and minimize mean paired
    IoU. Empty-reference success and failure minimize and maximize the mean
    paired predicted building fraction. Ties use lexicographic sample ID.
    """
    validation = metrics[metrics.split == "validation"].copy()
    validation["reference_positive"] = validation.tp + validation.fn > 0
    validation["predicted_fraction"] = (validation.tp + validation.fp) / (512 * 512)
    paired = validation.groupby("id", as_index=False).agg(
        reference_positive=("reference_positive", "first"),
        mean_iou=("iou", "mean"),
        mean_predicted_fraction=("predicted_fraction", "mean"),
    )
    positive = paired[paired.reference_positive].sort_values(["mean_iou", "id"])
    empty = paired[~paired.reference_positive].sort_values(["mean_predicted_fraction", "id"])
    choices = [
        ("positive failure", positive.iloc[0]),
        ("positive success", positive.iloc[-1]),
        ("empty success", empty.iloc[0]),
        ("empty failure", empty.iloc[-1]),
    ]
    return pd.DataFrame({
        "selection": [label for label, _ in choices],
        "id": [row.id for _, row in choices],
        "reference_positive": [bool(row.reference_positive) for _, row in choices],
        "paired_mean_iou": [float(row.mean_iou) for _, row in choices],
        "paired_mean_predicted_fraction": [float(row.mean_predicted_fraction) for _, row in choices],
    })


def plot_whu_learning_curves(history: pd.DataFrame):
    """Plot validation IoU and loss for the two paired WHU runs."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    colors = {"random": "#E76F51", "imagenet": "#2878B5"}
    for initialization in ("random", "imagenet"):
        subset = history[history.initialization == initialization]
        label = INITIALIZATION_LABELS[initialization]
        axes[0].plot(subset.step, subset.validation_mean_iou, label=label, color=colors[initialization])
        axes[1].plot(subset.step, subset.validation_loss, label=label, color=colors[initialization])
    for axis in axes:
        axis.axvline(2000, color="#777777", linestyle="--", linewidth=1, label="LR drop" if axis is axes[0] else None)
        axis.set_xlabel("Optimizer update")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean per-image building IoU")
    axes[1].set_ylabel("Validation BCE–soft-Dice loss")
    axes[0].legend(frameon=False)
    figure.suptitle("WHU validation trajectories under the shared 4,000-update recipe")
    figure.tight_layout()
    return figure


def _predictions(project_root: Path, runs: list[dict], sample_ids: list[str]) -> dict[str, np.ndarray]:
    manifest = json.loads(
        (project_root / "data/metadata/whu/data_manifest.json").read_text()
    )
    store = WHUTensorStore(project_root, manifest)
    device = select_device()
    batch = store.batch(sample_ids)
    predictions = {}
    for run in runs:
        model, _ = build_model(run["initialization"], run["config"]["seeds"]["model"])
        restore_checkpoint(
            model, run["directory"] / "checkpoint.pt",
            run["metadata"]["compatibility_identity"]["config_sha256"],
            run["metadata"]["compatibility_identity"]["input_sha256"],
        )
        model.to(device).eval()
        with torch.inference_mode():
            predictions[run["initialization"]] = model(batch["image"].to(device)).sigmoid().cpu().numpy()[:, 0]
    return predictions


def plot_whu_validation_predictions(project_root: Path, examples: pd.DataFrame):
    """Plot identical WHU validation cases for both selected checkpoints."""
    runs = load_whu_pilot_runs(project_root)
    manifest = json.loads(
        (project_root / "data/metadata/whu/data_manifest.json").read_text()
    )
    records = {sample["id"]: sample for sample in manifest["samples"]}
    sample_ids = examples.id.tolist()
    predictions = _predictions(project_root, runs, sample_ids)
    cmap = ListedColormap(["#111111", "#2DC653", "#FF8C1A", "#4361EE"])
    figure, axes = plt.subplots(len(sample_ids), 4, figsize=(10, 2.65 * len(sample_ids)), squeeze=False)
    for row, choice in enumerate(examples.itertuples(index=False)):
        record = records[choice.id]
        image = read_whu_image(project_root / record["image"])
        target = load_whu_binary_target(read_whu_mask(project_root / record["mask"]))
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"{choice.selection}\n{choice.id}", fontsize=8)
        for column, initialization in enumerate(("random", "imagenet"), 2):
            prediction = predictions[initialization][row] >= 0.5
            error = np.zeros_like(target, dtype=np.uint8)
            error[prediction & target] = 1
            error[prediction & ~target] = 2
            error[~prediction & target] = 3
            union = np.logical_or(prediction, target).sum()
            iou = np.logical_and(prediction, target).sum() / union if union else 1.0
            axes[row, column].imshow(error, cmap=cmap, vmin=0, vmax=3)
            axes[row, column].set_title(f"{INITIALIZATION_LABELS[initialization]} · IoU {iou:.3f}", fontsize=9)
    axes[0, 0].set_title("RGB image")
    axes[0, 1].set_title("Building-footprint reference")
    for axis in axes.ravel():
        axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle("Common WHU validation cases: green TP, orange FP, blue FN", y=1.005)
    figure.tight_layout()
    return figure
