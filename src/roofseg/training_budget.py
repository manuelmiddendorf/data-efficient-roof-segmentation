"""Configuration, loading and analysis for the n=500 training-budget block."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
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


TRAINING_SIZE = 500
SUBSET_REPETITIONS = (1, 2)
MAX_STEPS = 8000
PREFIX_STEPS = 4000
CHECKPOINT_NAME = "best_through_step_4000"
CHECKPOINT_FILENAME = "checkpoint_best_through_step4000.pt"
COLOURS = {"random": "#D55E00", "imagenet": "#0072B2"}


def training_budget_run_name(initialization: str, subset_seed: int) -> str:
    """Return a run name that identifies subset, initialization and budget."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("Unknown initialization.")
    return (
        f"n{TRAINING_SIZE}_{initialization}_subsetseed{subset_seed}_"
        "lr1e-3_to_1e-4_step2001_steps8000"
    )


def build_training_budget_config(
    splits: dict, initialization: str, subset_repetition: int
) -> dict:
    """Extend one saved n=500 reference recipe from 4,000 to 8,000 updates.

    The training IDs and every random seed remain those of the selected saved
    subset repetition. The learning rate stays at ``1e-4`` after update 2,000.
    """
    reference = build_late_lr_drop_config(
        splits, initialization, subset_repetition, TRAINING_SIZE
    )
    repetition = splits["training_subsets"]["repetitions"][str(subset_repetition)]
    subset_seed = int(repetition["seed"])
    config = deepcopy(reference)
    config.update({
        "run_name": training_budget_run_name(initialization, subset_seed),
        "experiment": "stage5_n500_training_budget_sensitivity",
        "max_steps": MAX_STEPS,
        "evaluation_steps": [0] + list(range(100, MAX_STEPS + 1, 100)),
        "additional_checkpoints": [{
            "name": CHECKPOINT_NAME,
            "filename": CHECKPOINT_FILENAME,
            "through_step": PREFIX_STEPS,
        }],
    })
    config["scheduler"]["phases"][-1]["end_step"] = MAX_STEPS
    return config


def training_budget_run_directory(
    project_root: Path, splits: dict, initialization: str, subset_repetition: int
) -> Path:
    """Return the split-versioned directory for one 8,000-update run."""
    repetition = splits["training_subsets"]["repetitions"][str(subset_repetition)]
    subset_seed = int(repetition["seed"])
    return (
        project_root
        / "runs/optimization"
        / splits["split_name"]
        / "training_budget_n500"
        / training_budget_run_name(initialization, subset_seed)
    )


def load_training_budget_runs(project_root: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load the four compatible completed 8,000-update runs and histories."""
    splits = json.loads((project_root / "data/metadata/splits.json").read_text())
    runs, histories = [], {}
    for repetition in SUBSET_REPETITIONS:
        subset_seed = int(
            splits["training_subsets"]["repetitions"][str(repetition)]["seed"]
        )
        for initialization in INITIALIZATIONS:
            directory = training_budget_run_directory(
                project_root, splits, initialization, repetition
            )
            config = json.loads((directory / "config.json").read_text())
            metadata = json.loads((directory / "metadata.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            if summary.get("status") != "completed":
                raise ValueError(f"Training-budget run is incomplete: {directory.name}")
            key = f"seed{subset_seed}_{initialization}_steps8000"
            runs.append({
                "key": key,
                "subset_repetition": repetition,
                "subset_seed": subset_seed,
                "initialization": initialization,
                "directory": directory,
                "config": config,
                "metadata": metadata,
                "summary": summary,
            })
            with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
                histories[key] = list(csv.DictReader(handle))
    return runs, histories


def _checkpoint_metrics(checkpoint: dict) -> dict:
    validation = checkpoint["best_validation"]
    training = checkpoint["matched_training_evaluation"]
    return {
        "best_step": int(checkpoint["best_step"]),
        "best_validation_iou": float(validation["mean_iou"]),
        "selected_validation_dice": float(validation["mean_dice"]),
        "selected_validation_ap": float(validation["mean_average_precision"]),
        "selected_validation_loss": float(validation["loss"]),
        "selected_training_iou": float(training["mean_iou"]),
        "selected_training_loss": float(training["loss"]),
        "training_minus_validation_iou": float(
            training["mean_iou"] - validation["mean_iou"]
        ),
    }


def selected_history_row(history: pd.DataFrame, through_step: int) -> pd.Series:
    """Select the highest post-initialization IoU, using the earlier-step tie break."""
    eligible = history[(history.step > 0) & (history.step <= through_step)]
    if eligible.empty:
        raise ValueError("No trained evaluation is available in the selection range.")
    return eligible.sort_values(
        ["validation_mean_iou", "step"], ascending=[False, True]
    ).iloc[0]


def collect_training_budget_results(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize endpoints, stable end windows, trends and paired gaps."""
    runs, histories = load_training_budget_runs(project_root)
    splits = json.loads((project_root / "data/metadata/splits.json").read_text())
    result_rows, history_frames = [], []
    for run in runs:
        history = _history_frame(histories[run["key"]])
        trained = history[history.step > 0]
        step_4000 = trained.loc[trained.step == 4000].iloc[0]
        step_8000 = trained.loc[trained.step == 8000].iloc[0]
        early_trend, _ = summarize_late_history(history, 2100, 4000)
        late_trend, _ = summarize_late_history(history, 6100, 8000)
        prefix = run["summary"]["additional_checkpoints"][CHECKPOINT_NAME]
        full = {
            "best_step": run["summary"]["best_step"],
            "best_validation": run["summary"]["best_validation"],
            "matched_training_evaluation": run["summary"]["matched_training_evaluation"],
        }
        expected_prefix = selected_history_row(history, PREFIX_STEPS)
        expected_full = selected_history_row(history, MAX_STEPS)
        if int(prefix["best_step"]) != int(expected_prefix.step):
            raise ValueError(f"Prefix checkpoint selection disagrees for {run['key']}.")
        if int(full["best_step"]) != int(expected_full.step):
            raise ValueError(f"Full checkpoint selection disagrees for {run['key']}.")
        historical_directory = late_lr_drop_run_directory(
            project_root, splits, run["initialization"],
            run["subset_repetition"], TRAINING_SIZE,
        )
        historical_summary = json.loads(
            (historical_directory / "summary.json").read_text()
        )
        historical_history = pd.read_csv(historical_directory / "history.csv")
        historical_endpoint = historical_history.loc[
            historical_history.step == PREFIX_STEPS
        ].iloc[0]
        row = {
            "subset_seed": run["subset_seed"],
            "initialization": run["initialization"],
            "iou_step_4000": float(step_4000.validation_mean_iou),
            "iou_step_8000": float(step_8000.validation_mean_iou),
            "validation_loss_step_4000": float(step_4000.validation_loss),
            "validation_loss_step_8000": float(step_8000.validation_loss),
            "mean_iou_last5_to_4000": float(
                trained[trained.step.between(3600, 4000)].validation_mean_iou.mean()
            ),
            "mean_iou_last5_to_8000": float(
                trained[trained.step.between(7600, 8000)].validation_mean_iou.mean()
            ),
            "historical_best_step_4000": int(historical_summary["best_step"]),
            "historical_best_iou_4000": float(
                historical_summary["best_validation"]["mean_iou"]
            ),
            "historical_iou_step_4000": float(
                historical_endpoint.validation_mean_iou
            ),
            "new_minus_historical_prefix_best_iou": float(
                prefix["best_validation"]["mean_iou"]
                - historical_summary["best_validation"]["mean_iou"]
            ),
            "new_minus_historical_iou_step_4000": float(
                step_4000.validation_mean_iou
                - historical_endpoint.validation_mean_iou
            ),
            **{f"prefix_{key}": value for key, value in _checkpoint_metrics(prefix).items()},
            **{f"full_{key}": value for key, value in _checkpoint_metrics(full).items()},
            **{f"early_{key}": value for key, value in early_trend.items()},
            **{f"late_{key}": value for key, value in late_trend.items()},
            "training_minutes": run["summary"]["training_seconds"] / 60,
            "total_minutes": run["summary"]["total_seconds"] / 60,
        }
        result_rows.append(row)
        history_frames.append(history.assign(
            subset_seed=run["subset_seed"], initialization=run["initialization"]
        ))

    results = pd.DataFrame(result_rows).sort_values(["subset_seed", "initialization"])
    paired_rows = []
    for subset_seed, group in results.groupby("subset_seed"):
        pair = group.set_index("initialization")
        paired_rows.append({
            "subset_seed": int(subset_seed),
            "imagenet_minus_random_iou_step_4000": float(
                pair.loc["imagenet", "iou_step_4000"]
                - pair.loc["random", "iou_step_4000"]
            ),
            "imagenet_minus_random_iou_step_8000": float(
                pair.loc["imagenet", "iou_step_8000"]
                - pair.loc["random", "iou_step_8000"]
            ),
            "imagenet_minus_random_mean_last5_to_4000": float(
                pair.loc["imagenet", "mean_iou_last5_to_4000"]
                - pair.loc["random", "mean_iou_last5_to_4000"]
            ),
            "imagenet_minus_random_mean_last5_to_8000": float(
                pair.loc["imagenet", "mean_iou_last5_to_8000"]
                - pair.loc["random", "mean_iou_last5_to_8000"]
            ),
        })
    return results, pd.DataFrame(paired_rows), pd.concat(history_frames, ignore_index=True)


def plot_training_budget_histories(history: pd.DataFrame) -> Figure:
    """Plot complete validation IoU and loss histories for both subsets."""
    seeds = sorted(history.subset_seed.unique())
    figure, axes = plt.subplots(2, len(seeds), figsize=(12, 7), sharex=True)
    for column, subset_seed in enumerate(seeds):
        for initialization in INITIALIZATIONS:
            rows = history[
                (history.subset_seed == subset_seed)
                & (history.initialization == initialization)
            ].sort_values("step")
            label = INITIALIZATION_LABELS[initialization]
            axes[0, column].plot(
                rows.step, rows.validation_mean_iou,
                color=COLOURS[initialization], label=label,
            )
            axes[1, column].plot(
                rows.step, rows.validation_loss, color=COLOURS[initialization]
            )
        for axis in axes[:, column]:
            axis.axvline(2000, color="#555555", linestyle="--", linewidth=1)
            axis.axvline(4000, color="#777777", linestyle=":", linewidth=1)
            axis.grid(alpha=0.2)
        axes[0, column].set_title(f"Subset seed {subset_seed}")
        axes[0, column].legend(frameon=False)
        axes[1, column].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("Validation mean per-image IoU")
    axes[1, 0].set_ylabel("Validation loss")
    figure.suptitle("n=500 training-budget sensitivity")
    figure.tight_layout()
    return figure
