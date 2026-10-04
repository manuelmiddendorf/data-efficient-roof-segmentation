"""Tables and figures for the fixed-recipe data-efficiency comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

from .late_training_analysis import _history_frame, summarize_late_history
from .optimization import INITIALIZATIONS, load_fixed_drop_data_efficiency_results
from .optimization_analysis import INITIALIZATION_LABELS

SIZES = (25, 100, 500)
COLOURS = {"random": "#D55E00", "imagenet": "#0072B2"}


def collect_data_efficiency_results(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect comparable summaries, paired differences and histories."""
    runs, histories = load_fixed_drop_data_efficiency_results(project_root)
    result_rows, history_frames = [], []
    for run in runs:
        history = _history_frame(histories[run["key"]])
        late, _ = summarize_late_history(history)
        trained = history[history.step > 0]
        best = trained.loc[trained.validation_mean_iou.idxmax()]
        endpoint = trained.loc[trained.step == 4000].iloc[0]
        summary = run["summary"]
        validation = summary["best_validation"]
        training = summary["matched_training_evaluation"]
        result_rows.append({
            "training_size": run["training_size"],
            "initialization": run["initialization"],
            "best_step": int(best.step),
            "best_validation_iou": float(best.validation_mean_iou),
            "iou_step_4000": float(endpoint.validation_mean_iou),
            "validation_loss_step_4000": float(endpoint.validation_loss),
            "selected_validation_loss": validation["loss"],
            "selected_validation_dice": validation["mean_dice"],
            "selected_validation_ap": validation["mean_average_precision"],
            "selected_training_loss": training["loss"],
            "selected_training_iou": training["mean_iou"],
            "selected_training_dice": training["mean_dice"],
            "selected_training_ap": training["mean_average_precision"],
            "training_minus_validation_iou": (
                training["mean_iou"] - validation["mean_iou"]
            ),
            "updates": summary["steps_completed"],
            "examples_processed": summary["examples_processed"],
            "data_passages": summary["equivalent_data_passages"],
            "incomplete_batches_retained": summary["incomplete_batches_retained"],
            "training_minutes": summary["training_seconds"] / 60,
            "total_minutes": summary["total_seconds"] / 60,
            **late,
        })
        history_frames.append(history.assign(
            training_size=run["training_size"], initialization=run["initialization"]
        ))
    results = pd.DataFrame(result_rows).sort_values(["training_size", "initialization"])
    paired_rows = []
    for training_size, group in results.groupby("training_size"):
        by_init = group.set_index("initialization")
        paired_rows.append({
            "training_size": int(training_size),
            "imagenet_minus_random_best_iou": float(
                by_init.loc["imagenet", "best_validation_iou"]
                - by_init.loc["random", "best_validation_iou"]
            ),
            "imagenet_minus_random_iou_step_4000": float(
                by_init.loc["imagenet", "iou_step_4000"]
                - by_init.loc["random", "iou_step_4000"]
            ),
            "imagenet_minus_random_dice": float(
                by_init.loc["imagenet", "selected_validation_dice"]
                - by_init.loc["random", "selected_validation_dice"]
            ),
            "imagenet_minus_random_ap": float(
                by_init.loc["imagenet", "selected_validation_ap"]
                - by_init.loc["random", "selected_validation_ap"]
            ),
        })
    return results, pd.DataFrame(paired_rows), pd.concat(history_frames, ignore_index=True)


def plot_data_efficiency_summary(results: pd.DataFrame) -> Figure:
    """Plot selected maxima and fixed endpoints against training-set size."""
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for initialization in INITIALIZATIONS:
        rows = results[results.initialization == initialization].sort_values("training_size")
        colour = COLOURS[initialization]
        label = INITIALIZATION_LABELS[initialization]
        axis.plot(rows.training_size, rows.best_validation_iou, color=colour,
                  marker="o", linewidth=2, label=f"{label}: selected maximum")
        axis.plot(rows.training_size, rows.iou_step_4000, color=colour,
                  marker="x", linestyle="--", label=f"{label}: step 4,000")
    axis.set_xscale("log")
    axis.set_xticks(SIZES, labels=[str(size) for size in SIZES])
    axis.set_xlabel("Labelled training images")
    axis.set_ylabel("Validation mean per-image IoU")
    axis.set_title("Preliminary data-efficiency curve under one common recipe")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncols=2)
    figure.tight_layout()
    return figure


def plot_data_efficiency_histories(history: pd.DataFrame) -> Figure:
    """Plot validation trajectories and augmented training objectives by size."""
    figure, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    metrics = (
        ("validation_mean_iou", "Validation mean per-image IoU"),
        ("validation_loss", "Validation loss"),
        ("train_loss", "Augmented training objective"),
    )
    for column, training_size in enumerate(SIZES):
        for initialization in INITIALIZATIONS:
            rows = history[
                (history.training_size == training_size)
                & (history.initialization == initialization)
            ].sort_values("step")
            for row_index, (metric, _) in enumerate(metrics):
                axes[row_index, column].plot(
                    rows.step, rows[metric], color=COLOURS[initialization],
                    label=INITIALIZATION_LABELS[initialization],
                )
        for axis in axes[:, column]:
            axis.axvline(2000, color="#555555", linestyle="--", linewidth=1)
            axis.grid(alpha=0.2)
        axes[0, column].set_title(f"{training_size} training images")
        axes[0, column].legend(frameon=False)
        axes[2, column].set_xlabel("Optimizer step")
    for row_index, (_, label) in enumerate(metrics):
        axes[row_index, 0].set_ylabel(label)
    figure.suptitle("Training trajectories under the common fixed-drop recipe")
    figure.tight_layout()
    return figure
