"""Tables and figures for the Stage 3 learning-rate and threshold analyses."""

from __future__ import annotations

import csv
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
from .objectives import mean_metrics_at_thresholds
from .optimization import (
    INITIALIZATIONS,
    LEARNING_RATES,
    REFERENCE_LEARNING_RATE,
    learning_rate_label,
    load_learning_rate_results,
)
from .training import restore_checkpoint, select_device
from .training_data import RIDTensorStore


THRESHOLDS = tuple(value / 10 for value in range(1, 10))
INITIALIZATION_LABELS = {"random": "Random", "imagenet": "ImageNet"}
LEARNING_RATE_COLOURS = {1e-4: "#0072B2", 3e-4: "#009E73", 1e-3: "#D55E00"}


def collect_optimization_tables(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect selected results, histories and per-image metrics for six runs.

    Parameters
    ----------
    project_root:
        Repository root containing Stage 2 references and Stage 3 runs.

    Returns
    -------
    results, history, metrics:
        One row per configuration, one row per evaluation step and one row per
        evaluated image respectively.
    """
    runs, histories = load_learning_rate_results(project_root)
    result_rows, history_rows, metric_frames = [], [], []
    for run in runs:
        summary = run["summary"]
        last = histories[run["key"]][-1]
        result_rows.append({
            "initialization": run["initialization"],
            "learning_rate": run["learning_rate"],
            "is_stage2_reference": run["is_reference"],
            "best_step": summary["best_step"],
            "best_validation_iou": summary["best_validation"]["mean_iou"],
            "last_validation_iou": float(last["validation_mean_iou"]),
            "validation_dice": summary["best_validation"]["mean_dice"],
            "validation_ap": summary["best_validation"]["mean_average_precision"],
            "validation_ap_images": summary["best_validation"]["ap_image_count"],
            "training_iou": summary["matched_training_evaluation"]["mean_iou"],
            "steps": summary["steps_completed"],
            "examples_processed": summary["examples_processed"],
            "data_passages": summary["equivalent_data_passages"],
            "training_minutes": summary["training_seconds"] / 60,
            "total_minutes": summary["total_seconds"] / 60,
        })
        for row in histories[run["key"]]:
            history_rows.append({
                "initialization": run["initialization"],
                "learning_rate": run["learning_rate"],
                **row,
            })
        with (run["directory"] / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            frame = pd.DataFrame(csv.DictReader(handle))
        frame.insert(0, "learning_rate", run["learning_rate"])
        frame.insert(0, "initialization", run["initialization"])
        metric_frames.append(frame)

    results = pd.DataFrame(result_rows).sort_values(["initialization", "learning_rate"])
    history = pd.DataFrame(history_rows)
    numeric_history = [column for column in history if column != "initialization"]
    history[numeric_history] = history[numeric_history].apply(pd.to_numeric, errors="coerce")
    metrics = pd.concat(metric_frames, ignore_index=True)
    for column in ("learning_rate", "tp", "fp", "fn", "iou", "dice", "average_precision"):
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    return results.reset_index(drop=True), history, metrics


def plot_learning_rate_curves(history: pd.DataFrame) -> Figure:
    """Plot validation IoU and augmented training objective by initialization.

    Parameters
    ----------
    history:
        Evaluation history for the six learning-rate configurations.

    Returns
    -------
    matplotlib.figure.Figure
        Two rows for validation IoU and training objective, with one column per
        initialization.
    """
    figure, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for column, initialization in enumerate(INITIALIZATIONS):
        for learning_rate in LEARNING_RATES:
            rows = history[
                (history.initialization == initialization)
                & np.isclose(history.learning_rate, learning_rate)
            ].sort_values("step")
            label = learning_rate_label(learning_rate)
            colour = LEARNING_RATE_COLOURS[learning_rate]
            axes[0, column].plot(rows.step, rows.validation_mean_iou,
                                 color=colour, label=label)
            trained = rows[rows.step > 0]
            axes[1, column].plot(trained.step, trained.train_loss, color=colour)
        axes[0, column].set_title(INITIALIZATION_LABELS[initialization])
        axes[0, column].set_ylim(0, 1)
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
        axes[1, column].set_xlabel("Optimizer step")
        axes[0, column].legend(title="Learning rate", frameon=False)
    axes[0, 0].set_ylabel("Validation mean per-image IoU")
    axes[1, 0].set_ylabel("Augmented training objective")
    figure.suptitle("Learning-rate trajectories at 100 training images")
    figure.tight_layout()
    return figure


def plot_paired_learning_rate_differences(results: pd.DataFrame) -> Figure:
    """Plot ImageNet-minus-random validation differences at each learning rate.

    Parameters
    ----------
    results:
        One-row-per-configuration selected-checkpoint table.

    Returns
    -------
    matplotlib.figure.Figure
        Bar charts for paired IoU, Dice and AP differences.
    """
    metrics = [
        ("best_validation_iou", "IoU"),
        ("validation_dice", "Dice"),
        ("validation_ap", "Average precision"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for axis, (column, label) in zip(axes, metrics):
        pivot = results.pivot(index="learning_rate", columns="initialization", values=column)
        difference = pivot["imagenet"] - pivot["random"]
        axis.bar(
            [learning_rate_label(value) for value in difference.index],
            difference,
            color=["#009E73" if value >= 0 else "#CC79A7" for value in difference],
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(label)
        axis.set_xlabel("Learning rate")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("ImageNet − random")
    figure.suptitle("Paired initialization differences at fixed learning rates")
    figure.tight_layout()
    return figure


def restore_run_model(run: dict, device: torch.device) -> torch.nn.Module:
    """Restore one verified run checkpoint to evaluation mode on ``device``."""
    model, _ = build_model(run["initialization"], run["config"]["seeds"]["model"])
    restore_checkpoint(
        model,
        run["directory"] / "checkpoint.pt",
        run["metadata"]["compatibility_identity"]["config_sha256"],
        run["metadata"]["compatibility_identity"]["input_sha256"],
    )
    return model.to(device).eval()


def evaluate_reference_thresholds(project_root: Path) -> pd.DataFrame:
    """Evaluate fixed thresholds for the two selected Stage 2 n=100 checkpoints.

    Parameters
    ----------
    project_root:
        Repository root containing reference checkpoints and validation data.

    Returns
    -------
    pandas.DataFrame
        Mean per-image IoU and Dice for 289 validation images at each fixed
        threshold. No probabilities are retained.
    """
    runs, _ = load_learning_rate_results(project_root)
    manifest = load_json(project_root / "data/metadata/data_manifest.json")
    store = RIDTensorStore(project_root, manifest)
    device = select_device()
    rows = []
    for run in runs:
        if run["learning_rate"] != REFERENCE_LEARNING_RATE:
            continue
        model = restore_run_model(run, device)
        sums = {threshold: {"iou": 0.0, "dice": 0.0, "count": 0} for threshold in THRESHOLDS}
        validation_ids = run["config"]["validation_ids"]
        with torch.inference_mode():
            for start in range(0, len(validation_ids), run["config"]["batch_size"]):
                batch = store.batch(validation_ids[start:start + run["config"]["batch_size"]])
                probability = model(batch["image"].to(device)).sigmoid().cpu().numpy()[:, 0]
                target = batch["target"].numpy()[:, 0]
                for result in mean_metrics_at_thresholds(probability, target, THRESHOLDS):
                    aggregate = sums[result["threshold"]]
                    aggregate["iou"] += result["mean_iou"] * result["image_count"]
                    aggregate["dice"] += result["mean_dice"] * result["image_count"]
                    aggregate["count"] += result["image_count"]
        for threshold, aggregate in sums.items():
            rows.append({
                "initialization": run["initialization"],
                "threshold": threshold,
                "mean_iou": aggregate["iou"] / aggregate["count"],
                "mean_dice": aggregate["dice"] / aggregate["count"],
                "image_count": aggregate["count"],
            })
        del model
    return pd.DataFrame(rows)


def plot_threshold_curves(thresholds: pd.DataFrame) -> Figure:
    """Plot reference validation IoU across the predeclared threshold list.

    Parameters
    ----------
    thresholds:
        Threshold-result table returned by :func:`evaluate_reference_thresholds`.

    Returns
    -------
    matplotlib.figure.Figure
        Joint threshold curves with the primary 0.5 threshold marked.
    """
    figure, axis = plt.subplots(figsize=(7, 4))
    for initialization in INITIALIZATIONS:
        rows = thresholds[thresholds.initialization == initialization]
        axis.plot(rows.threshold, rows.mean_iou, marker="o",
                  label=INITIALIZATION_LABELS[initialization])
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1,
                 label="Primary threshold 0.5")
    axis.set_xlabel("Probability threshold")
    axis.set_ylabel("Validation mean per-image IoU")
    axis.set_xticks(THRESHOLDS)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    axis.set_title("Diagnostic threshold sensitivity of the 3e-4 references")
    figure.tight_layout()
    return figure


def choose_optimization_examples(results: pd.DataFrame, metrics: pd.DataFrame) -> list[str]:
    """Choose validation failure, disagreement, success and empty-target cases.

    Parameters
    ----------
    results:
        Selected-checkpoint summary table.
    metrics:
        Per-image selected-checkpoint metrics for all six configurations.

    Returns
    -------
    list of str
        Unique validation IDs; selection never consults the test role.
    """
    best_rates = results.loc[
        results.groupby("initialization")["best_validation_iou"].idxmax(),
        ["initialization", "learning_rate"],
    ]
    selected_frames = []
    for row in best_rates.itertuples(index=False):
        selected_frames.append(metrics[
            (metrics.initialization == row.initialization)
            & np.isclose(metrics.learning_rate, row.learning_rate)
            & (metrics.split == "validation")
        ][["id", "initialization", "iou"]])
    pivot = pd.concat(selected_frames).pivot(index="id", columns="initialization", values="iou")
    pivot["pair_mean"] = pivot[list(INITIALIZATIONS)].mean(axis=1)
    pivot["absolute_gap"] = (pivot["imagenet"] - pivot["random"]).abs()
    choices = [pivot.pair_mean.idxmin(), pivot.absolute_gap.idxmax(),
               pivot.pair_mean.idxmax(), "39"]
    return list(dict.fromkeys(choices))


def _probabilities_for_run(run: dict, project_root: Path, sample_ids: list[str]) -> np.ndarray:
    manifest = load_json(project_root / "data/metadata/data_manifest.json")
    device = select_device()
    model = restore_run_model(run, device)
    batch = RIDTensorStore(project_root, manifest).batch(sample_ids)
    with torch.inference_mode():
        return model(batch["image"].to(device)).sigmoid().cpu().numpy()[:, 0]


def plot_optimization_examples(
    project_root: Path,
    results: pd.DataFrame,
    sample_ids: list[str],
) -> Figure:
    """Plot common validation cases for each initialization's best observed rate.

    Parameters
    ----------
    project_root:
        Repository root containing data and checkpoints.
    results:
        Selected-checkpoint summary table.
    sample_ids:
        Validation IDs selected for shared qualitative comparison.

    Returns
    -------
    matplotlib.figure.Figure
        RGB, binary reference and paired threshold-0.5 error maps.
    """
    runs, _ = load_learning_rate_results(project_root)
    best_rates = results.loc[
        results.groupby("initialization")["best_validation_iou"].idxmax(),
        ["initialization", "learning_rate"],
    ].set_index("initialization")["learning_rate"].to_dict()
    selected_runs = {
        initialization: next(
            run for run in runs
            if run["initialization"] == initialization
            and np.isclose(run["learning_rate"], best_rates[initialization])
        )
        for initialization in INITIALIZATIONS
    }
    predictions = {
        initialization: _probabilities_for_run(run, project_root, sample_ids)
        for initialization, run in selected_runs.items()
    }
    manifest = load_json(project_root / "data/metadata/data_manifest.json")
    records = {sample["id"]: sample for sample in manifest["samples"]}
    error_cmap = ListedColormap(["#111111", "#2DC653", "#FF8C1A", "#4361EE"])
    figure, axes = plt.subplots(
        len(sample_ids), 4, figsize=(10, 2.65 * len(sample_ids)), squeeze=False
    )
    for row, sample_id in enumerate(sample_ids):
        record = records[sample_id]
        image = read_image(project_root / record["image"])
        target = load_binary_target(read_mask(project_root / record["mask"]))
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"ID {sample_id}")
        for column, initialization in enumerate(INITIALIZATIONS, 2):
            prediction = predictions[initialization][row] >= 0.5
            error = np.zeros_like(target, dtype=np.uint8)
            error[prediction & target] = 1
            error[prediction & ~target] = 2
            error[~prediction & target] = 3
            union = np.logical_or(prediction, target).sum()
            iou = np.logical_and(prediction, target).sum() / union if union else 1.0
            rate = learning_rate_label(best_rates[initialization])
            axes[row, column].imshow(error, cmap=error_cmap, vmin=0, vmax=3)
            axes[row, column].set_title(
                f"{INITIALIZATION_LABELS[initialization]} {rate} · IoU {iou:.3f}"
            )
    axes[0, 0].set_title("RGB image")
    axes[0, 1].set_title("Reference")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        "Common validation cases for the best observed rates\n"
        "green true positive, orange false positive, blue false negative",
        y=1.01,
    )
    figure.tight_layout()
    return figure
