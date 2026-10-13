"""Configuration and analysis for the WHU n=500 training-budget block."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

from .late_training_analysis import _history_frame, summarize_late_history
from .pilot import INITIALIZATIONS
from .training_budget import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_NAME,
    MAX_STEPS,
    PREFIX_STEPS,
    selected_history_row,
)
from .whu_pilot import (
    LEARNING_CURVE_REPETITIONS,
    build_whu_pilot_config,
    whu_run_directory,
    whu_subset_record,
)


TRAINING_SIZE = 500
COLOURS = {"random": "#E76F51", "imagenet": "#2878B5"}
INITIALIZATION_LABELS = {"random": "Random", "imagenet": "ImageNet"}


def whu_training_budget_run_name(initialization: str, subset_seed: int) -> str:
    """Return a WHU run name identifying subset, initialization and budget."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("Initialization must be random or imagenet.")
    return (
        f"n{TRAINING_SIZE}_{initialization}_subsetseed{subset_seed}_"
        "fixed_drop_steps8000"
    )


def build_whu_training_budget_config(
    splits: dict, initialization: str, subset_repetition: int
) -> dict:
    """Extend one saved WHU n=500 recipe from 4,000 to 8,000 updates."""
    reference = build_whu_pilot_config(
        splits, initialization, subset_repetition, TRAINING_SIZE
    )
    subset_seed = int(
        whu_subset_record(splits, subset_repetition, TRAINING_SIZE)["seed"]
    )
    config = deepcopy(reference)
    config.update({
        "run_name": whu_training_budget_run_name(initialization, subset_seed),
        "experiment": "whu_n500_training_budget_sensitivity",
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


def whu_training_budget_run_directory(
    project_root: Path, splits: dict, initialization: str, subset_repetition: int
) -> Path:
    """Return the provider-split-versioned directory for one WHU budget run."""
    subset_seed = int(
        whu_subset_record(splits, subset_repetition, TRAINING_SIZE)["seed"]
    )
    return (
        project_root
        / "runs/external_replication_pilot"
        / splits["split_name"]
        / "training_budget_n500"
        / whu_training_budget_run_name(initialization, subset_seed)
    )


def load_whu_training_budget_runs(
    project_root: Path,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load the six compatible completed WHU 8,000-update runs."""
    splits = json.loads(
        (project_root / "data/metadata/whu/splits.json").read_text()
    )
    runs, histories = [], {}
    for repetition in LEARNING_CURVE_REPETITIONS:
        subset_seed = int(
            whu_subset_record(splits, repetition, TRAINING_SIZE)["seed"]
        )
        for initialization in INITIALIZATIONS:
            directory = whu_training_budget_run_directory(
                project_root, splits, initialization, repetition
            )
            config = json.loads((directory / "config.json").read_text())
            metadata = json.loads((directory / "metadata.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            if summary.get("status") != "completed":
                raise ValueError(f"WHU budget run is incomplete: {directory.name}")
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
            with (directory / "history.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                histories[key] = list(csv.DictReader(handle))
    return runs, histories


def _checkpoint_metrics(checkpoint: dict) -> dict:
    """Flatten selected-checkpoint metrics including WHU reference groups."""
    validation = checkpoint["best_validation"]
    training = checkpoint["matched_training_evaluation"]
    groups = validation["reference_groups"]
    return {
        "best_step": int(checkpoint["best_step"]),
        "best_validation_iou": float(validation["mean_iou"]),
        "selected_validation_dice": float(validation["mean_dice"]),
        "selected_validation_ap_positive": float(
            validation["mean_average_precision"]
        ),
        "selected_validation_loss": float(validation["loss"]),
        "selected_training_iou": float(training["mean_iou"]),
        "selected_training_loss": float(training["loss"]),
        "training_minus_validation_iou": float(
            training["mean_iou"] - validation["mean_iou"]
        ),
        "positive_image_count": int(
            groups["positive_reference"]["image_count"]
        ),
        "positive_mean_iou": float(
            groups["positive_reference"]["mean_iou"]
        ),
        "positive_mean_dice": float(
            groups["positive_reference"]["mean_dice"]
        ),
        "empty_image_count": int(groups["empty_reference"]["image_count"]),
        "empty_mean_predicted_building_fraction": float(
            groups["empty_reference"]["mean_predicted_building_fraction"]
        ),
        "empty_fraction_with_any_prediction": float(
            groups["empty_reference"]["fraction_with_any_predicted_building"]
        ),
    }


def collect_whu_training_budget_results(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize WHU endpoints, windows, selected checkpoints and paired gaps."""
    runs, histories = load_whu_training_budget_runs(project_root)
    splits = json.loads(
        (project_root / "data/metadata/whu/splits.json").read_text()
    )
    result_rows, history_frames = [], []
    for run in runs:
        history = _history_frame(histories[run["key"]])
        trained = history[history.step > 0]
        step_4000 = trained.loc[trained.step == PREFIX_STEPS].iloc[0]
        step_8000 = trained.loc[trained.step == MAX_STEPS].iloc[0]
        early_trend, _ = summarize_late_history(history, 2100, 4000)
        late_trend, _ = summarize_late_history(history, 6100, 8000)
        prefix = run["summary"]["additional_checkpoints"][CHECKPOINT_NAME]
        full = {
            "best_step": run["summary"]["best_step"],
            "best_validation": run["summary"]["best_validation"],
            "matched_training_evaluation": run["summary"][
                "matched_training_evaluation"
            ],
        }
        if int(prefix["best_step"]) != int(
            selected_history_row(history, PREFIX_STEPS).step
        ):
            raise ValueError(f"Prefix checkpoint selection disagrees for {run['key']}.")
        if int(full["best_step"]) != int(
            selected_history_row(history, MAX_STEPS).step
        ):
            raise ValueError(f"Full checkpoint selection disagrees for {run['key']}.")

        historical_directory = whu_run_directory(
            project_root,
            splits,
            run["initialization"],
            run["subset_repetition"],
            TRAINING_SIZE,
        )
        historical_summary = json.loads(
            (historical_directory / "summary.json").read_text()
        )
        historical_history = pd.read_csv(historical_directory / "history.csv")
        historical_endpoint = historical_history.loc[
            historical_history.step == PREFIX_STEPS
        ].iloc[0]
        result_rows.append({
            "subset_repetition": run["subset_repetition"],
            "subset_seed": run["subset_seed"],
            "initialization": run["initialization"],
            "iou_step_4000": float(step_4000.validation_mean_iou),
            "iou_step_8000": float(step_8000.validation_mean_iou),
            "iou_change_8000_minus_4000": float(
                step_8000.validation_mean_iou - step_4000.validation_mean_iou
            ),
            "validation_loss_step_4000": float(step_4000.validation_loss),
            "validation_loss_step_8000": float(step_8000.validation_loss),
            "mean_iou_last5_to_4000": float(
                trained[trained.step.between(3600, 4000)]
                .validation_mean_iou.mean()
            ),
            "mean_iou_last5_to_8000": float(
                trained[trained.step.between(7600, 8000)]
                .validation_mean_iou.mean()
            ),
            "mean_last5_change_8000_minus_4000": float(
                trained[trained.step.between(7600, 8000)]
                .validation_mean_iou.mean()
                - trained[trained.step.between(3600, 4000)]
                .validation_mean_iou.mean()
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
            **{
                f"prefix_{key}": value
                for key, value in _checkpoint_metrics(prefix).items()
            },
            **{
                f"full_{key}": value
                for key, value in _checkpoint_metrics(full).items()
            },
            **{f"early_{key}": value for key, value in early_trend.items()},
            **{f"late_{key}": value for key, value in late_trend.items()},
            "examples_processed": int(run["summary"]["examples_processed"]),
            "equivalent_data_passages": float(
                run["summary"]["equivalent_data_passages"]
            ),
            "training_minutes": run["summary"]["training_seconds"] / 60,
            "total_minutes": run["summary"]["total_seconds"] / 60,
        })
        history_frames.append(history.assign(
            subset_seed=run["subset_seed"],
            initialization=run["initialization"],
        ))

    results = pd.DataFrame(result_rows).sort_values(
        ["subset_seed", "initialization"]
    )
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
            "gap_change_8000_minus_4000": float(
                pair.loc["imagenet", "iou_change_8000_minus_4000"]
                - pair.loc["random", "iou_change_8000_minus_4000"]
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
    paired = pd.DataFrame(paired_rows)
    summary_rows = []
    for initialization, group in results.groupby("initialization"):
        summary_rows.append({
            "initialization": initialization,
            "repetitions": len(group),
            "mean_iou_change_8000_minus_4000": group[
                "iou_change_8000_minus_4000"
            ].mean(),
            "sample_sd_iou_change_8000_minus_4000": group[
                "iou_change_8000_minus_4000"
            ].std(ddof=1),
            "mean_last5_change_8000_minus_4000": group[
                "mean_last5_change_8000_minus_4000"
            ].mean(),
            "sample_sd_last5_change_8000_minus_4000": group[
                "mean_last5_change_8000_minus_4000"
            ].std(ddof=1),
        })
    summary = pd.DataFrame(summary_rows)
    return results, paired, summary, pd.concat(history_frames, ignore_index=True)


def rid_whu_budget_pattern_comparison(
    project_root: Path, whu_results: pd.DataFrame, whu_paired: pd.DataFrame
) -> pd.DataFrame:
    """Compare within-dataset budget changes without ranking absolute scores."""
    from .training_budget import collect_training_budget_results

    rid_results, rid_paired, _ = collect_training_budget_results(project_root)
    rows = []
    for dataset, results, paired in (
        ("RID", rid_results, rid_paired),
        ("WHU", whu_results, whu_paired),
    ):
        indexed_paired = paired.set_index("subset_seed")
        for result in results.itertuples(index=False):
            pair = indexed_paired.loc[result.subset_seed]
            rows.append({
                "dataset": dataset,
                "subset_seed": int(result.subset_seed),
                "initialization": result.initialization,
                "endpoint_iou_change_8000_minus_4000": (
                    result.iou_step_8000 - result.iou_step_4000
                ),
                "last5_iou_change_8000_minus_4000": (
                    result.mean_iou_last5_to_8000
                    - result.mean_iou_last5_to_4000
                ),
                "imagenet_minus_random_iou_step_4000": pair[
                    "imagenet_minus_random_iou_step_4000"
                ],
                "imagenet_minus_random_iou_step_8000": pair[
                    "imagenet_minus_random_iou_step_8000"
                ],
            })
    return pd.DataFrame(rows)


def plot_whu_training_budget_histories(history: pd.DataFrame) -> Figure:
    """Plot complete validation IoU and loss histories for three WHU subsets."""
    seeds = sorted(history.subset_seed.unique())
    figure, axes = plt.subplots(2, len(seeds), figsize=(12, 7), sharex=True)
    for column, subset_seed in enumerate(seeds):
        for initialization in INITIALIZATIONS:
            rows = history[
                (history.subset_seed == subset_seed)
                & (history.initialization == initialization)
            ].sort_values("step")
            axes[0, column].plot(
                rows.step,
                rows.validation_mean_iou,
                color=COLOURS[initialization],
                label=INITIALIZATION_LABELS[initialization],
            )
            axes[1, column].plot(
                rows.step,
                rows.validation_loss,
                color=COLOURS[initialization],
            )
        for axis in axes[:, column]:
            axis.axvline(2000, color="#555555", linestyle="--", linewidth=1)
            axis.axvline(4000, color="#777777", linestyle=":", linewidth=1)
            axis.grid(alpha=0.2)
        axes[0, column].set_title(f"Subset seed {subset_seed}")
        axes[0, column].legend(frameon=False)
        axes[1, column].set_xlabel("Optimizer update")
    axes[0, 0].set_ylabel("Validation mean per-image IoU")
    axes[1, 0].set_ylabel("Validation loss")
    figure.suptitle("WHU n=500 training-budget sensitivity")
    figure.tight_layout()
    return figure
