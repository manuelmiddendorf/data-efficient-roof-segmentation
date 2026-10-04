"""Tables and figures for the Stage 3 training-horizon comparison."""

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
from .optimization import (
    HORIZON_LEARNING_RATES,
    INITIALIZATIONS,
    learning_rate_label,
    load_training_horizon_results,
)
from .optimization_analysis import (
    INITIALIZATION_LABELS,
    LEARNING_RATE_COLOURS,
    restore_run_model,
)
from .training import select_device
from .training_data import RIDTensorStore


def summarize_horizon_history(history: pd.DataFrame) -> dict:
    """Summarize predeclared 2,000- and 4,000-step views of one new curve.

    Parameters
    ----------
    history:
        Evaluation rows containing ``step`` and ``validation_mean_iou``.

    Returns
    -------
    dict
        Best trained steps and IoUs for each horizon plus the two endpoint IoUs.
        Earlier steps win exact ties.
    """
    ordered = history.sort_values("step")
    trained = ordered[ordered.step > 0]
    first_half = trained[trained.step <= 2000]
    if set([2000, 4000]) - set(trained.step):
        raise ValueError("A long-run history lacks step 2,000 or 4,000.")
    first_best = first_half.loc[first_half.validation_mean_iou.idxmax()]
    full_best = trained.loc[trained.validation_mean_iou.idxmax()]
    return {
        "best_step_first_2000": int(first_best.step),
        "best_iou_first_2000": float(first_best.validation_mean_iou),
        "best_step_all_4000": int(full_best.step),
        "best_iou_all_4000": float(full_best.validation_mean_iou),
        "iou_step_2000": float(trained.loc[trained.step == 2000, "validation_mean_iou"].iloc[0]),
        "iou_step_4000": float(trained.loc[trained.step == 4000, "validation_mean_iou"].iloc[0]),
    }


def collect_horizon_tables(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect summaries, histories and selected-checkpoint metrics for four runs."""
    runs, histories = load_training_horizon_results(project_root)
    result_rows, history_rows, metric_frames = [], [], []
    for run in runs:
        history = pd.DataFrame(histories[run["key"]])
        numeric = [column for column in history if column != "initialization"]
        history[numeric] = history[numeric].apply(pd.to_numeric, errors="coerce")
        budget = summarize_horizon_history(history)
        summary = run["summary"]
        if summary["best_step"] != budget["best_step_all_4000"]:
            raise ValueError(f"Saved checkpoint selection disagrees for {run['key']}.")
        second_half = history[(history.step >= 2100) & (history.step <= 4000)]
        result_rows.append({
            "initialization": run["initialization"],
            "learning_rate": run["learning_rate"],
            **budget,
            "selected_validation_dice": summary["best_validation"]["mean_dice"],
            "selected_validation_ap": summary["best_validation"]["mean_average_precision"],
            "selected_validation_ap_images": summary["best_validation"]["ap_image_count"],
            "selected_validation_loss": summary["best_validation"]["loss"],
            "selected_training_iou": summary["matched_training_evaluation"]["mean_iou"],
            "selected_training_dice": summary["matched_training_evaluation"]["mean_dice"],
            "selected_training_ap": summary["matched_training_evaluation"]["mean_average_precision"],
            "selected_training_loss": summary["matched_training_evaluation"]["loss"],
            "second_half_iou_mean": second_half.validation_mean_iou.mean(),
            "second_half_iou_sd": second_half.validation_mean_iou.std(ddof=0),
            "second_half_iou_min": second_half.validation_mean_iou.min(),
            "second_half_iou_max": second_half.validation_mean_iou.max(),
            "updates": summary["steps_completed"],
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


def plot_horizon_curves(history: pd.DataFrame) -> Figure:
    """Plot 4,000-step validation and training trajectories with the old limit marked."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for column, initialization in enumerate(INITIALIZATIONS):
        for learning_rate in HORIZON_LEARNING_RATES:
            rows = history[
                (history.initialization == initialization)
                & np.isclose(history.learning_rate, learning_rate)
            ].sort_values("step")
            colour = LEARNING_RATE_COLOURS[learning_rate]
            axes[0, column].plot(
                rows.step, rows.validation_mean_iou, color=colour,
                label=learning_rate_label(learning_rate),
            )
            trained = rows[rows.step > 0]
            axes[1, column].plot(trained.step, trained.train_loss, color=colour)
        for axis in axes[:, column]:
            axis.axvline(2000, color="#555555", linestyle="--", linewidth=1)
            axis.grid(alpha=0.2)
        axes[0, column].set_title(INITIALIZATION_LABELS[initialization])
        axes[0, column].set_ylim(0, 1)
        axes[0, column].legend(title="Learning rate", frameon=False)
        axes[1, column].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("Validation mean per-image IoU")
    axes[1, 0].set_ylabel("Augmented training objective")
    figure.suptitle("Training-horizon comparison at 100 images (dashed: 2,000 steps)")
    figure.tight_layout()
    return figure


def choose_horizon_examples(results: pd.DataFrame, metrics: pd.DataFrame) -> list[str]:
    """Choose shared validation failure, disagreement, success and empty-target cases."""
    best_rates = results.loc[
        results.groupby("initialization")["best_iou_all_4000"].idxmax(),
        ["initialization", "learning_rate"],
    ]
    selected = []
    for row in best_rates.itertuples(index=False):
        selected.append(metrics[
            (metrics.initialization == row.initialization)
            & np.isclose(metrics.learning_rate, row.learning_rate)
            & (metrics.split == "validation")
        ][["id", "initialization", "iou"]])
    pivot = pd.concat(selected).pivot(index="id", columns="initialization", values="iou")
    pivot["pair_mean"] = pivot[list(INITIALIZATIONS)].mean(axis=1)
    pivot["absolute_gap"] = (pivot["imagenet"] - pivot["random"]).abs()
    choices = [pivot.pair_mean.idxmin(), pivot.absolute_gap.idxmax(),
               pivot.pair_mean.idxmax(), "39"]
    return list(dict.fromkeys(choices))


def _probabilities(run: dict, project_root: Path, sample_ids: list[str]) -> np.ndarray:
    manifest = load_json(project_root / "data/metadata/data_manifest.json")
    device = select_device()
    model = restore_run_model(run, device)
    batch = RIDTensorStore(project_root, manifest).batch(sample_ids)
    with torch.inference_mode():
        return model(batch["image"].to(device)).sigmoid().cpu().numpy()[:, 0]


def plot_horizon_examples(
    project_root: Path, results: pd.DataFrame, sample_ids: list[str]
) -> Figure:
    """Plot common validation cases for each initialization's selected long run."""
    runs, _ = load_training_horizon_results(project_root)
    best_rates = results.loc[
        results.groupby("initialization")["best_iou_all_4000"].idxmax(),
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
        initialization: _probabilities(run, project_root, sample_ids)
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
        "Selected checkpoints from 4,000-step runs on common validation cases\n"
        "green true positive, orange false positive, blue false negative",
        y=1.01,
    )
    figure.tight_layout()
    return figure
