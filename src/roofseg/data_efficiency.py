"""Tables and figures for the fixed-recipe data-efficiency comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter
import pandas as pd

from .late_training_analysis import _history_frame, summarize_late_history
from .optimization import INITIALIZATIONS, load_fixed_drop_data_efficiency_results
from .optimization_analysis import INITIALIZATION_LABELS

SIZES = (25, 100, 500)
SUBSET_REPETITIONS = (1, 2)
COLOURS = {"random": "#D55E00", "imagenet": "#0072B2"}


def collect_data_efficiency_results(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect summaries, paired differences, histories, gains and ID overlap."""
    runs, histories = load_fixed_drop_data_efficiency_results(
        project_root, SUBSET_REPETITIONS
    )
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
        identity = {
            "subset_repetition": run["subset_repetition"],
            "subset_seed": run["subset_seed"],
            "training_size": run["training_size"],
            "initialization": run["initialization"],
        }
        result_rows.append({
            **identity,
            "best_step": int(best.step),
            "best_validation_iou": float(best.validation_mean_iou),
            "iou_step_4000": float(endpoint.validation_mean_iou),
            "validation_loss_step_4000": float(endpoint.validation_loss),
            "selected_validation_loss": validation["loss"],
            "selected_validation_iou": validation["mean_iou"],
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
        history_frames.append(history.assign(**identity))

    results = pd.DataFrame(result_rows).sort_values(
        ["subset_seed", "training_size", "initialization"]
    )
    paired_rows = []
    for (repetition, seed, size), group in results.groupby(
        ["subset_repetition", "subset_seed", "training_size"]
    ):
        by_init = group.set_index("initialization")
        paired_rows.append({
            "subset_repetition": int(repetition),
            "subset_seed": int(seed),
            "training_size": int(size),
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

    gain_rows = []
    for (repetition, seed, initialization), group in results.groupby(
        ["subset_repetition", "subset_seed", "initialization"]
    ):
        by_size = group.set_index("training_size")
        for lower_size, upper_size in zip(SIZES[:-1], SIZES[1:]):
            gain_rows.append({
                "subset_repetition": int(repetition),
                "subset_seed": int(seed),
                "initialization": initialization,
                "size_change": f"{lower_size}_to_{upper_size}",
                "best_iou_gain": float(
                    by_size.loc[upper_size, "best_validation_iou"]
                    - by_size.loc[lower_size, "best_validation_iou"]
                ),
                "endpoint_iou_gain": float(
                    by_size.loc[upper_size, "iou_step_4000"]
                    - by_size.loc[lower_size, "iou_step_4000"]
                ),
            })

    splits = json.loads((project_root / "data/metadata/splits.json").read_text())
    repetitions = splits["training_subsets"]["repetitions"]
    overlap_rows = []
    for size in SIZES:
        ids17 = set(repetitions["1"]["subsets"][str(size)])
        ids29 = set(repetitions["2"]["subsets"][str(size)])
        overlap_rows.append({
            "training_size": size,
            "shared_training_ids": len(ids17 & ids29),
            "share_of_each_subset": len(ids17 & ids29) / size,
        })
    return (
        results,
        pd.DataFrame(paired_rows),
        pd.concat(history_frames, ignore_index=True),
        pd.DataFrame(gain_rows),
        pd.DataFrame(overlap_rows),
    )


def plot_data_efficiency_summary(results: pd.DataFrame) -> Figure:
    """Plot selected maxima and fixed endpoints for both subset repetitions."""
    seeds = sorted(results.subset_seed.unique())
    figure, axes = plt.subplots(1, len(seeds), figsize=(12, 4.8), sharey=True)
    for axis, subset_seed in zip(axes, seeds):
        subset = results[results.subset_seed == subset_seed]
        for initialization in INITIALIZATIONS:
            rows = subset[subset.initialization == initialization].sort_values(
                "training_size"
            )
            colour = COLOURS[initialization]
            label = INITIALIZATION_LABELS[initialization]
            axis.plot(
                rows.training_size, rows.best_validation_iou, color=colour,
                marker="o", linewidth=2, label=f"{label}: selected maximum",
            )
            axis.plot(
                rows.training_size, rows.iou_step_4000, color=colour,
                marker="x", linestyle="--", label=f"{label}: step 4,000",
            )
        axis.set_xscale("log")
        axis.set_xticks(SIZES, labels=[str(size) for size in SIZES])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("Labelled training images")
        axis.set_title(f"Subset seed {subset_seed}")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Validation mean per-image IoU")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Data-efficiency curves under the common fixed-drop recipe")
    figure.tight_layout()
    return figure


def plot_data_efficiency_histories(history: pd.DataFrame, subset_seed: int) -> Figure:
    """Plot validation trajectories and training objectives for one subset seed."""
    subset = history[history.subset_seed == subset_seed]
    if subset.empty:
        raise ValueError(f"No history is available for subset seed {subset_seed}.")
    figure, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    metrics = (
        ("validation_mean_iou", "Validation mean per-image IoU"),
        ("validation_loss", "Validation loss"),
        ("train_loss", "Augmented training objective"),
    )
    for column, training_size in enumerate(SIZES):
        for initialization in INITIALIZATIONS:
            rows = subset[
                (subset.training_size == training_size)
                & (subset.initialization == initialization)
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
    figure.suptitle(f"Training trajectories for subset seed {subset_seed}")
    figure.tight_layout()
    return figure
