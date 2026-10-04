"""Compact tables and figures for the completed Stage 2 training pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch

from .data import load_binary_target, read_image, read_mask
from .integrity import load_json
from .model import build_model
from .pilot import PILOT_SIZES, load_pilot_results, pilot_run_root, run_name
from .training import restore_checkpoint, select_device
from .training_data import RIDTensorStore


INITIALIZATION_LABELS = {"random": "Random", "imagenet": "ImageNet"}
INITIALIZATION_COLOURS = {"random": "#0072B2", "imagenet": "#D55E00"}


def collect_pilot_tables(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect summaries, histories and per-image metrics for completed pilot runs.

    Parameters
    ----------
    project_root:
        Repository root containing the active split and completed run directories.

    Returns
    -------
    results, history, metrics:
        One row per run, one row per evaluation step, and one row per evaluated
        image respectively. Metric columns use values in ``[0, 1]``.
    """
    summaries, histories = load_pilot_results(project_root)
    result_rows, history_rows, metric_frames = [], [], []
    splits = load_json(project_root / "data/metadata/splits.json")
    run_root = pilot_run_root(project_root, splits)
    for run in summaries:
        summary = run["summary"]
        validation = summary["best_validation"]
        training = summary["matched_training_evaluation"]
        result_rows.append({
            "training_size": run["training_size"],
            "initialization": run["initialization"],
            "best_step": summary["best_step"],
            "validation_iou": validation["mean_iou"],
            "validation_dice": validation["mean_dice"],
            "validation_ap": validation["mean_average_precision"],
            "validation_ap_images": validation["ap_image_count"],
            "training_iou": training["mean_iou"],
            "training_dice": training["mean_dice"],
            "training_ap": training["mean_average_precision"],
            "training_ap_images": training["ap_image_count"],
            "steps": summary["steps_completed"],
            "examples_processed": summary["examples_processed"],
            "data_passages": summary["equivalent_data_passages"],
            "training_minutes": summary["training_seconds"] / 60,
            "total_minutes": summary["total_seconds"] / 60,
        })
        for row in histories[run["name"]]:
            history_rows.append({"training_size": run["training_size"],
                                 "initialization": run["initialization"], **row})
        with (run_root / run["name"] / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            frame = pd.DataFrame(csv.DictReader(handle))
        frame.insert(0, "initialization", run["initialization"])
        frame.insert(0, "training_size", run["training_size"])
        metric_frames.append(frame)

    results = pd.DataFrame(result_rows).sort_values(["training_size", "initialization"])
    history = pd.DataFrame(history_rows)
    numeric_history = [column for column in history if column not in {"initialization"}]
    history[numeric_history] = history[numeric_history].apply(pd.to_numeric, errors="coerce")
    metrics = pd.concat(metric_frames, ignore_index=True)
    for column in ("training_size", "tp", "fp", "fn", "iou", "dice", "average_precision"):
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    return results.reset_index(drop=True), history, metrics


def choose_validation_examples(metrics: pd.DataFrame) -> list[str]:
    """Choose common validation cases without consulting the locked test role.

    Parameters
    ----------
    metrics:
        Per-image metric table returned by :func:`collect_pilot_tables`.

    Returns
    -------
    list of str
        Unique validation IDs representing low paired IoU, initialization
        disagreement, high paired IoU and the known empty reference.
    """
    selected = metrics[(metrics["training_size"] == 500) & (metrics["split"] == "validation")]
    pivot = selected.pivot(index="id", columns="initialization", values="iou")
    pivot["pair_mean"] = pivot[["random", "imagenet"]].mean(axis=1)
    pivot["absolute_gap"] = (pivot["imagenet"] - pivot["random"]).abs()
    choices = [pivot["pair_mean"].idxmin(), pivot["absolute_gap"].idxmax(),
               pivot["pair_mean"].idxmax(), "39"]
    return list(dict.fromkeys(choices))


def plot_learning_curves(history: pd.DataFrame) -> Figure:
    """Plot validation IoU and the augmented training objective over updates.

    Parameters
    ----------
    history:
        Evaluation history with optimizer step, validation mean per-image IoU
        in ``[0, 1]`` and the preceding interval's training objective.

    Returns
    -------
    matplotlib.figure.Figure
        Two-row figure with one column per training-set size.
    """
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for column, size in enumerate(PILOT_SIZES):
        for initialization in ("random", "imagenet"):
            rows = history[(history.training_size == size)
                           & (history.initialization == initialization)].sort_values("step")
            axes[0, column].plot(rows.step, rows.validation_mean_iou,
                                 color=INITIALIZATION_COLOURS[initialization],
                                 label=INITIALIZATION_LABELS[initialization])
            trained = rows[rows.step > 0]
            axes[1, column].plot(trained.step, trained.train_loss,
                                 color=INITIALIZATION_COLOURS[initialization])
        axes[0, column].set_title(f"{size} training images")
        axes[0, column].set_ylim(0, 1)
        axes[1, column].set_xlabel("Optimizer step")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Validation mean per-image IoU")
    axes[1, 0].set_ylabel("Augmented training objective")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Learning trajectories under the common 2,000-update budget")
    figure.tight_layout()
    return figure


def plot_paired_differences(results: pd.DataFrame) -> Figure:
    """Plot ImageNet-minus-random validation differences.

    Parameters
    ----------
    results:
        One-row-per-run result table containing selected-checkpoint metrics.

    Returns
    -------
    matplotlib.figure.Figure
        Bar charts for IoU, Dice and average precision differences.
    """
    metrics = [("validation_iou", "IoU"), ("validation_dice", "Dice"),
               ("validation_ap", "Average precision")]
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    for axis, (column, label) in zip(axes, metrics):
        pivot = results.pivot(index="training_size", columns="initialization", values=column)
        differences = pivot["imagenet"] - pivot["random"]
        colours = ["#009E73" if value >= 0 else "#CC79A7" for value in differences]
        axis.bar(differences.index.astype(str), differences, color=colours)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(label)
        axis.set_xlabel("Training images")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("ImageNet − random")
    figure.suptitle("Paired initialization differences on validation")
    figure.tight_layout()
    return figure


def plot_train_validation_iou(results: pd.DataFrame) -> Figure:
    """Compare training and validation IoU in matching evaluation mode.

    Parameters
    ----------
    results:
        One-row-per-run table with mean per-image IoU values in ``[0, 1]``.

    Returns
    -------
    matplotlib.figure.Figure
        Grouped bars for training-subset and validation IoU.
    """
    labels = [f"{int(row.training_size)}\n{INITIALIZATION_LABELS[row.initialization]}"
              for row in results.itertuples()]
    positions = np.arange(len(results))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(positions - width / 2, results.training_iou, width, label="Training", color="#56B4E9")
    axis.bar(positions + width / 2, results.validation_iou, width, label="Validation", color="#E69F00")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean per-image IoU")
    axis.set_title("Selected checkpoints in the same evaluation mode")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return figure


def _prediction_for_run(project_root: Path, size: int, initialization: str, sample_ids: list[str]):
    splits = load_json(project_root / "data/metadata/splits.json")
    manifest = load_json(project_root / "data/metadata/data_manifest.json")
    directory = pilot_run_root(project_root, splits) / run_name(size, initialization)
    config = json.loads((directory / "config.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    model, _ = build_model(initialization, config["seeds"]["model"])
    restore_checkpoint(
        model,
        directory / "checkpoint.pt",
        metadata["compatibility_identity"]["config_sha256"],
        metadata["compatibility_identity"]["input_sha256"],
    )
    device = select_device()
    model.to(device).eval()
    batch = RIDTensorStore(project_root, manifest).batch(sample_ids)
    with torch.inference_mode():
        probabilities = model(batch["image"].to(device)).sigmoid().cpu().numpy()[:, 0]
    return probabilities


def plot_validation_predictions(
    project_root: Path,
    size: int,
    sample_ids: list[str],
) -> Figure:
    """Plot RGB, reference and paired thresholded validation error maps.

    Parameters
    ----------
    project_root:
        Repository root containing data records and completed checkpoints.
    size:
        Number of distinct images in the paired training subset.
    sample_ids:
        Validation image IDs. Each input and target has spatial shape
        ``(512, 512)``; predictions are thresholded at 0.5.

    Returns
    -------
    matplotlib.figure.Figure
        Four-column figure with RGB, binary target and error maps for both
        initializations. Green, orange and blue denote true positives, false
        positives and false negatives.
    """
    manifest = load_json(project_root / "data/metadata/data_manifest.json")
    records = {sample["id"]: sample for sample in manifest["samples"]}
    predictions = {
        initialization: _prediction_for_run(project_root, size, initialization, sample_ids)
        for initialization in ("random", "imagenet")
    }
    error_cmap = ListedColormap(["#111111", "#2DC653", "#FF8C1A", "#4361EE"])
    figure, axes = plt.subplots(len(sample_ids), 4, figsize=(10, 2.65 * len(sample_ids)), squeeze=False)
    for row, sample_id in enumerate(sample_ids):
        record = records[sample_id]
        image = read_image(project_root / record["image"])
        target = load_binary_target(read_mask(project_root / record["mask"]))
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"ID {sample_id}")
        for column, initialization in enumerate(("random", "imagenet"), 2):
            prediction = predictions[initialization][row] >= 0.5
            error = np.zeros_like(target, dtype=np.uint8)
            error[prediction & target] = 1
            error[prediction & ~target] = 2
            error[~prediction & target] = 3
            union = np.logical_or(prediction, target).sum()
            iou = np.logical_and(prediction, target).sum() / union if union else 1.0
            axes[row, column].imshow(error, cmap=error_cmap, vmin=0, vmax=3)
            axes[row, column].set_title(f"{INITIALIZATION_LABELS[initialization]} · IoU {iou:.3f}")
    axes[0, 0].set_title("RGB image")
    axes[0, 1].set_title("Reference")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"Common validation cases at n={size}\nerror maps: green true positive, orange false positive, blue false negative",
        y=1.01,
    )
    figure.tight_layout()
    return figure
