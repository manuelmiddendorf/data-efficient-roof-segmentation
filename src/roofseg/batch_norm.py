"""Configurations, result summaries and figures for encoder BatchNorm strategy."""

from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

from .late_training_analysis import _history_frame, summarize_late_history
from .optimization import build_late_lr_drop_config, late_lr_drop_run_directory
from .training import FROZEN_ENCODER_BATCH_NORM

TRAINING_SIZE = 25
SUBSET_REPETITIONS = (1, 2)
COLOURS = {False: "#777777", True: "#CC79A7"}


def frozen_batch_norm_run_name(subset_seed: int) -> str:
    """Return the unique name for one fixed encoder-statistics run."""
    return (
        f"n25_imagenet_subsetseed{subset_seed}_frozen_encoder_bn_"
        "lr1e-3_to_1e-4_step2001_steps4000"
    )


def frozen_batch_norm_run_root(project_root: Path, splits: dict) -> Path:
    """Return the split-versioned directory for this BatchNorm experiment."""
    return project_root / "runs/targeted" / splits["split_name"] / "encoder_batch_norm_n25"


def build_frozen_batch_norm_config(splits: dict, subset_repetition: int) -> dict:
    """Change only run identity and encoder BatchNorm behavior from its reference."""
    config = deepcopy(build_late_lr_drop_config(
        splits, "imagenet", subset_repetition, TRAINING_SIZE
    ))
    try:
        subset_seed = int(
            splits["training_subsets"]["repetitions"][str(subset_repetition)]["seed"]
        )
    except KeyError as error:
        raise ValueError(f"Unknown saved subset repetition: {subset_repetition}") from error
    config.update({
        "run_name": frozen_batch_norm_run_name(subset_seed),
        "experiment": "stage4_frozen_encoder_batch_norm_n25",
        "batch_normalization_strategy": FROZEN_ENCODER_BATCH_NORM,
        "batch_normalization": (
            "encoder BatchNorm2d uses original ImageNet running_mean and running_var "
            "during training; encoder affine parameters and all other parameters remain "
            "trainable; decoder BatchNorm2d updates running statistics normally"
        ),
    })
    return config


def frozen_batch_norm_run_directory(
    project_root: Path, splits: dict, subset_repetition: int
) -> Path:
    """Return the distinct directory for one fixed encoder-statistics run."""
    subset_seed = int(
        splits["training_subsets"]["repetitions"][str(subset_repetition)]["seed"]
    )
    return frozen_batch_norm_run_root(project_root, splits) / frozen_batch_norm_run_name(
        subset_seed
    )


def _load_comparison_runs(project_root: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load two references and two fixed-statistics runs without checkpoints."""
    splits = json.loads((project_root / "data/metadata/splits.json").read_text())
    runs, histories = [], {}
    for repetition in SUBSET_REPETITIONS:
        subset_seed = int(
            splits["training_subsets"]["repetitions"][str(repetition)]["seed"]
        )
        for frozen in (False, True):
            directory = (
                frozen_batch_norm_run_directory(project_root, splits, repetition)
                if frozen else late_lr_drop_run_directory(
                    project_root, splits, "imagenet", repetition, TRAINING_SIZE
                )
            )
            config = json.loads((directory / "config.json").read_text())
            metadata = json.loads((directory / "metadata.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            if summary.get("status") != "completed":
                raise ValueError(f"BatchNorm comparison run is incomplete: {directory.name}")
            key = f"seed{subset_seed}_{'frozen' if frozen else 'updated'}"
            runs.append({
                "key": key, "subset_repetition": repetition, "subset_seed": subset_seed,
                "frozen_encoder_statistics": frozen, "directory": directory,
                "config": config, "metadata": metadata, "summary": summary,
            })
            with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
                histories[key] = list(csv.DictReader(handle))
    return runs, histories


def collect_batch_norm_results(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect four-run summaries, paired frozen-minus-updated effects and histories."""
    runs, histories = _load_comparison_runs(project_root)
    rows, history_frames = [], []
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
            "frozen_encoder_statistics": run["frozen_encoder_statistics"],
        }
        rows.append({
            **identity,
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
            "training_minus_validation_iou": training["mean_iou"] - validation["mean_iou"],
            "examples_processed": summary["examples_processed"],
            "equivalent_data_passages": summary["equivalent_data_passages"],
            "training_minutes": summary["training_seconds"] / 60,
            "total_minutes": summary["total_seconds"] / 60,
            **late,
        })
        history_frames.append(history.assign(**identity))
    results = pd.DataFrame(rows).sort_values(
        ["subset_seed", "frozen_encoder_statistics"]
    )
    effects = []
    for (repetition, seed), group in results.groupby(["subset_repetition", "subset_seed"]):
        by_strategy = group.set_index("frozen_encoder_statistics")
        frozen, updated = by_strategy.loc[True], by_strategy.loc[False]
        effects.append({
            "subset_repetition": int(repetition), "subset_seed": int(seed),
            "best_iou_change": frozen.best_validation_iou - updated.best_validation_iou,
            "endpoint_iou_change": frozen.iou_step_4000 - updated.iou_step_4000,
            "endpoint_validation_loss_change": (
                frozen.validation_loss_step_4000 - updated.validation_loss_step_4000
            ),
            "selected_iou_gap_change": (
                frozen.training_minus_validation_iou - updated.training_minus_validation_iou
            ),
            "selected_dice_change": (
                frozen.selected_validation_dice - updated.selected_validation_dice
            ),
            "selected_ap_change": frozen.selected_validation_ap - updated.selected_validation_ap,
        })
    return results, pd.DataFrame(effects), pd.concat(history_frames, ignore_index=True)


def plot_batch_norm_histories(history: pd.DataFrame, metric: str) -> Figure:
    """Plot one validation metric for paired BatchNorm strategies by subset."""
    labels = {
        "validation_mean_iou": "Validation mean per-image IoU",
        "validation_loss": "Validation loss",
    }
    if metric not in labels:
        raise ValueError("Metric must be validation_mean_iou or validation_loss.")
    figure, axes = plt.subplots(
        1, 2, figsize=(11, 4.6), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, subset_seed in zip(axes, (17, 29)):
        for frozen in (False, True):
            values = history[
                (history.subset_seed == subset_seed)
                & (history.frozen_encoder_statistics == frozen)
            ].sort_values("step")
            axis.plot(
                values.step, values[metric], color=COLOURS[frozen],
                label="fixed ImageNet statistics" if frozen else "updated statistics",
            )
        axis.axvline(2000, color="#555555", linestyle="--", linewidth=1)
        axis.grid(alpha=0.2)
        axis.set_title(f"ImageNet, subset seed {subset_seed}")
        axis.set_xlabel("Optimizer step")
    axes[0].set_ylabel(labels[metric])
    axes[0].legend(frameon=False)
    figure.suptitle(f"n=25 encoder BatchNorm strategy: {labels[metric].lower()}")
    return figure
