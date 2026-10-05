"""Configurations, result tables and figures for the n=25 weight-decay block."""

from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

from .late_training_analysis import _history_frame, summarize_late_history
from .optimization import (
    INITIALIZATIONS,
    build_late_lr_drop_config,
    late_lr_drop_run_directory,
)
from .optimization_analysis import INITIALIZATION_LABELS

TRAINING_SIZE = 25
REFERENCE_WEIGHT_DECAY = 1e-4
STRONG_WEIGHT_DECAY = 1e-2
SUBSET_REPETITIONS = (1, 2)
COLOURS = {REFERENCE_WEIGHT_DECAY: "#777777", STRONG_WEIGHT_DECAY: "#009E73"}


def weight_decay_run_name(initialization: str, subset_seed: int) -> str:
    """Return the unique name for one strong-decay n=25 run."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("Unknown initialization.")
    return (
        f"n25_{initialization}_subsetseed{subset_seed}_wd1e-2_"
        "lr1e-3_to_1e-4_step2001_steps4000"
    )


def weight_decay_run_root(project_root: Path, splits: dict) -> Path:
    """Return the split-versioned directory for this targeted experiment."""
    return (
        project_root / "runs/targeted" / splits["split_name"] / "weight_decay_n25"
    )


def build_weight_decay_config(
    splits: dict, initialization: str, subset_repetition: int
) -> dict:
    """Change only run identity and weight decay from the matched n=25 reference."""
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
        "run_name": weight_decay_run_name(initialization, subset_seed),
        "experiment": "stage4_weight_decay_n25",
        "weight_decay": STRONG_WEIGHT_DECAY,
    })
    return config


def weight_decay_run_directory(
    project_root: Path, splits: dict, initialization: str, subset_repetition: int
) -> Path:
    """Return the distinct directory for one strong-decay run."""
    subset_seed = int(
        splits["training_subsets"]["repetitions"][str(subset_repetition)]["seed"]
    )
    return weight_decay_run_root(project_root, splits) / weight_decay_run_name(
        initialization, subset_seed
    )


def _load_comparison_runs(
    project_root: Path,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load four references and four strong-decay runs without checkpoints."""
    splits = json.loads((project_root / "data/metadata/splits.json").read_text())
    runs, histories = [], {}
    for repetition in SUBSET_REPETITIONS:
        subset_seed = int(
            splits["training_subsets"]["repetitions"][str(repetition)]["seed"]
        )
        for initialization in INITIALIZATIONS:
            for weight_decay in (REFERENCE_WEIGHT_DECAY, STRONG_WEIGHT_DECAY):
                if weight_decay == REFERENCE_WEIGHT_DECAY:
                    directory = late_lr_drop_run_directory(
                        project_root, splits, initialization, repetition, TRAINING_SIZE
                    )
                else:
                    directory = weight_decay_run_directory(
                        project_root, splits, initialization, repetition
                    )
                config = json.loads((directory / "config.json").read_text())
                metadata = json.loads((directory / "metadata.json").read_text())
                summary = json.loads((directory / "summary.json").read_text())
                if summary.get("status") != "completed":
                    raise ValueError(f"Weight-decay run is incomplete: {directory.name}")
                key = f"seed{subset_seed}_{initialization}_wd{weight_decay:.0e}"
                runs.append({
                    "key": key,
                    "subset_repetition": repetition,
                    "subset_seed": subset_seed,
                    "initialization": initialization,
                    "weight_decay": weight_decay,
                    "directory": directory,
                    "config": config,
                    "metadata": metadata,
                    "summary": summary,
                })
                with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
                    histories[key] = list(csv.DictReader(handle))
    return runs, histories


def shrinkage_factor(config: dict) -> float:
    """Return the no-gradient AdamW multiplicative factor over the saved schedule."""
    factor = 1.0
    for phase in config["scheduler"]["phases"]:
        steps = phase["end_step"] - phase["start_step"] + 1
        factor *= (1.0 - phase["learning_rate"] * config["weight_decay"]) ** steps
    return factor


def collect_weight_decay_results(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect eight-run summaries, paired strong-minus-reference effects and histories."""
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
            "weight_decay": run["weight_decay"],
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
            "training_minus_validation_iou": (
                training["mean_iou"] - validation["mean_iou"]
            ),
            "shrinkage_factor_without_gradients": shrinkage_factor(run["config"]),
            "training_minutes": summary["training_seconds"] / 60,
            "total_minutes": summary["total_seconds"] / 60,
            **late,
        })
        history_frames.append(history.assign(**identity))
    results = pd.DataFrame(rows).sort_values(
        ["subset_seed", "initialization", "weight_decay"]
    )
    effect_rows = []
    for (repetition, seed, initialization), group in results.groupby(
        ["subset_repetition", "subset_seed", "initialization"]
    ):
        by_decay = group.set_index("weight_decay")
        strong = by_decay.loc[STRONG_WEIGHT_DECAY]
        reference = by_decay.loc[REFERENCE_WEIGHT_DECAY]
        effect_rows.append({
            "subset_repetition": int(repetition),
            "subset_seed": int(seed),
            "initialization": initialization,
            "best_iou_change": strong.best_validation_iou - reference.best_validation_iou,
            "endpoint_iou_change": strong.iou_step_4000 - reference.iou_step_4000,
            "endpoint_validation_loss_change": (
                strong.validation_loss_step_4000 - reference.validation_loss_step_4000
            ),
            "selected_iou_gap_change": (
                strong.training_minus_validation_iou
                - reference.training_minus_validation_iou
            ),
            "selected_training_iou_change": (
                strong.selected_training_iou - reference.selected_training_iou
            ),
            "selected_dice_change": (
                strong.selected_validation_dice - reference.selected_validation_dice
            ),
            "selected_ap_change": (
                strong.selected_validation_ap - reference.selected_validation_ap
            ),
        })
    return results, pd.DataFrame(effect_rows), pd.concat(history_frames, ignore_index=True)


def plot_weight_decay_histories(history: pd.DataFrame, metric: str) -> Figure:
    """Plot one validation metric for paired decays by subset and initialization."""
    labels = {
        "validation_mean_iou": "Validation mean per-image IoU",
        "validation_loss": "Validation loss",
    }
    if metric not in labels:
        raise ValueError("Metric must be validation_mean_iou or validation_loss.")
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey="row")
    for row, initialization in enumerate(INITIALIZATIONS):
        for column, subset_seed in enumerate((17, 29)):
            axis = axes[row, column]
            for weight_decay in (REFERENCE_WEIGHT_DECAY, STRONG_WEIGHT_DECAY):
                values = history[
                    (history.initialization == initialization)
                    & (history.subset_seed == subset_seed)
                    & (history.weight_decay == weight_decay)
                ].sort_values("step")
                axis.plot(
                    values.step, values[metric], color=COLOURS[weight_decay],
                    label=f"weight decay {weight_decay:.0e}",
                )
            axis.axvline(2000, color="#555555", linestyle="--", linewidth=1)
            axis.grid(alpha=0.2)
            axis.set_title(
                f"{INITIALIZATION_LABELS[initialization]}, subset seed {subset_seed}"
            )
            if row == 0 and column == 0:
                axis.legend(frameon=False)
            if row == 1:
                axis.set_xlabel("Optimizer step")
            if column == 0:
                axis.set_ylabel(labels[metric])
    figure.suptitle(f"n=25 paired weight-decay comparison: {labels[metric].lower()}")
    figure.tight_layout()
    return figure
