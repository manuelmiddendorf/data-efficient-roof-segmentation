"""Configurations, result summaries and figures for n=25 decoder channel dropout."""

from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

from .late_training_analysis import _history_frame, summarize_late_history
from .optimization import INITIALIZATIONS, build_late_lr_drop_config, late_lr_drop_run_directory
from .optimization_analysis import INITIALIZATION_LABELS

TRAINING_SIZE = 25
SUBSET_REPETITIONS = (1, 2)
DROPOUT_PROBABILITY = 0.1
DROPOUT_SEED = 1704
COLOURS = {False: "#777777", True: "#D55E00"}


def decoder_dropout_run_name(initialization: str, subset_seed: int) -> str:
    """Return the unique name for one decoder channel-dropout run."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("Unknown initialization.")
    return (
        f"n25_{initialization}_subsetseed{subset_seed}_decoder_dropout_p0.1_"
        "lr1e-3_to_1e-4_step2001_steps4000"
    )


def decoder_dropout_run_root(project_root: Path, splits: dict) -> Path:
    """Return the split-versioned directory for this dropout experiment."""
    return project_root / "runs/targeted" / splits["split_name"] / "decoder_dropout_n25"


def build_decoder_dropout_config(
    splits: dict, initialization: str, subset_repetition: int
) -> dict:
    """Add only fixed decoder channel dropout and required identity to a reference."""
    config = deepcopy(build_late_lr_drop_config(
        splits, initialization, subset_repetition, TRAINING_SIZE
    ))
    try:
        subset_seed = int(
            splits["training_subsets"]["repetitions"][str(subset_repetition)]["seed"]
        )
    except KeyError as error:
        raise ValueError(f"Unknown saved subset repetition: {subset_repetition}") from error
    config.update({
        "run_name": decoder_dropout_run_name(initialization, subset_seed),
        "experiment": "stage4_decoder_channel_dropout_n25",
        "decoder_channel_dropout": {
            "type": "channel dropout with PyTorch Dropout2d semantics",
            "probability": DROPOUT_PROBABILITY,
            "keep_scaling": 1.0 / (1.0 - DROPOUT_PROBABILITY),
            "mask_shape": "N×C×1×1, independent per example and channel",
            "position": "after up5 and before the final 1×1 convolution",
            "evaluation": "identity",
            "trainable_parameters": 0,
            "random_source": "dedicated CPU torch.Generator",
        },
        "seeds": {**config["seeds"], "decoder_dropout": DROPOUT_SEED},
    })
    return config


def decoder_dropout_run_directory(
    project_root: Path, splits: dict, initialization: str, subset_repetition: int
) -> Path:
    """Return the distinct directory for one decoder channel-dropout run."""
    subset_seed = int(
        splits["training_subsets"]["repetitions"][str(subset_repetition)]["seed"]
    )
    return decoder_dropout_run_root(project_root, splits) / decoder_dropout_run_name(
        initialization, subset_seed
    )


def _load_comparison_runs(project_root: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load four references and four decoder-dropout runs without checkpoints."""
    splits = json.loads((project_root / "data/metadata/splits.json").read_text())
    runs, histories = [], {}
    for repetition in SUBSET_REPETITIONS:
        subset_seed = int(
            splits["training_subsets"]["repetitions"][str(repetition)]["seed"]
        )
        for initialization in INITIALIZATIONS:
            for dropout in (False, True):
                directory = (
                    decoder_dropout_run_directory(
                        project_root, splits, initialization, repetition
                    ) if dropout else late_lr_drop_run_directory(
                        project_root, splits, initialization, repetition, TRAINING_SIZE
                    )
                )
                config = json.loads((directory / "config.json").read_text())
                metadata = json.loads((directory / "metadata.json").read_text())
                summary = json.loads((directory / "summary.json").read_text())
                if summary.get("status") != "completed":
                    raise ValueError(f"Dropout comparison run is incomplete: {directory.name}")
                key = f"seed{subset_seed}_{initialization}_{'dropout' if dropout else 'reference'}"
                runs.append({
                    "key": key, "subset_repetition": repetition,
                    "subset_seed": subset_seed, "initialization": initialization,
                    "decoder_channel_dropout": dropout, "directory": directory,
                    "config": config, "metadata": metadata, "summary": summary,
                })
                with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
                    histories[key] = list(csv.DictReader(handle))
    return runs, histories


def collect_decoder_dropout_results(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect eight-run summaries, paired dropout effects and histories."""
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
            "initialization": run["initialization"],
            "decoder_channel_dropout": run["decoder_channel_dropout"],
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
        ["subset_seed", "initialization", "decoder_channel_dropout"]
    )
    effects = []
    for (repetition, seed, initialization), group in results.groupby(
        ["subset_repetition", "subset_seed", "initialization"]
    ):
        by_strategy = group.set_index("decoder_channel_dropout")
        dropout, reference = by_strategy.loc[True], by_strategy.loc[False]
        effects.append({
            "subset_repetition": int(repetition), "subset_seed": int(seed),
            "initialization": initialization,
            "best_iou_change": dropout.best_validation_iou - reference.best_validation_iou,
            "endpoint_iou_change": dropout.iou_step_4000 - reference.iou_step_4000,
            "endpoint_validation_loss_change": (
                dropout.validation_loss_step_4000 - reference.validation_loss_step_4000
            ),
            "selected_iou_gap_change": (
                dropout.training_minus_validation_iou
                - reference.training_minus_validation_iou
            ),
            "selected_dice_change": (
                dropout.selected_validation_dice - reference.selected_validation_dice
            ),
            "selected_ap_change": dropout.selected_validation_ap - reference.selected_validation_ap,
        })
    return results, pd.DataFrame(effects), pd.concat(history_frames, ignore_index=True)


def plot_decoder_dropout_histories(history: pd.DataFrame, metric: str) -> Figure:
    """Plot one validation metric for paired dropout strategies by subset and initialization."""
    labels = {
        "validation_mean_iou": "Validation mean per-image IoU",
        "validation_loss": "Validation loss",
    }
    if metric not in labels:
        raise ValueError("Metric must be validation_mean_iou or validation_loss.")
    figure, axes = plt.subplots(
        2, 2, figsize=(11, 8.5), sharex=True, sharey="row", constrained_layout=True
    )
    for row, initialization in enumerate(INITIALIZATIONS):
        for column, subset_seed in enumerate((17, 29)):
            axis = axes[row, column]
            for dropout in (False, True):
                values = history[
                    (history.initialization == initialization)
                    & (history.subset_seed == subset_seed)
                    & (history.decoder_channel_dropout == dropout)
                ].sort_values("step")
                axis.plot(
                    values.step, values[metric], color=COLOURS[dropout],
                    label="channel dropout p=0.1" if dropout else "no dropout",
                )
            axis.axvline(2000, color="#555555", linestyle="--", linewidth=1)
            axis.grid(alpha=0.2)
            axis.set_title(f"{INITIALIZATION_LABELS[initialization]}, subset seed {subset_seed}")
            if row == 0 and column == 0:
                axis.legend(frameon=False)
            if row == 1:
                axis.set_xlabel("Optimizer step")
            if column == 0:
                axis.set_ylabel(labels[metric])
    figure.suptitle(f"n=25 decoder channel dropout: {labels[metric].lower()}")
    return figure
