"""Late-trajectory diagnostics and fixed learning-rate-drop comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .optimization import (
    INITIALIZATIONS,
    learning_rate_label,
    load_late_lr_drop_repetitions,
    load_late_lr_drop_results,
    load_training_horizon_results,
)
from .optimization_analysis import INITIALIZATION_LABELS, LEARNING_RATE_COLOURS


LATE_STEPS = list(range(2100, 4001, 100))
ROLLING_MEASUREMENTS = 10
METRICS = {
    "validation_mean_iou": "iou",
    "validation_loss": "loss",
}


def summarize_late_history(history: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Summarize fixed late-window trends for validation IoU and loss.

    Parameters
    ----------
    history:
        One run's evaluation history with steps and validation metrics. The
        required late window is the 20 measurements at steps 2,100--4,000.

    Returns
    -------
    summary, rolling:
        Descriptive slopes per 1,000 updates, residual standard deviations and
        half-window means, plus slopes for every ten-measurement moving window.
    """
    late = history[history.step.isin(LATE_STEPS)].sort_values("step").copy()
    if late.step.astype(int).tolist() != LATE_STEPS:
        raise ValueError("Late history must contain every 100-step measurement from 2,100 to 4,000.")

    summary: dict[str, float] = {}
    rolling_rows: list[dict] = []
    steps = late.step.to_numpy(dtype=float)
    for column, label in METRICS.items():
        values = late[column].to_numpy(dtype=float)
        slope, intercept = np.polyfit(steps, values, 1)
        residuals = values - (slope * steps + intercept)
        summary.update({
            f"{label}_trend_per_1000": float(slope * 1000),
            f"{label}_residual_sd": float(np.std(residuals, ddof=0)),
            f"{label}_mean_2100_3000": float(values[:10].mean()),
            f"{label}_mean_3100_4000": float(values[10:].mean()),
        })
        for start in range(len(late) - ROLLING_MEASUREMENTS + 1):
            window = late.iloc[start:start + ROLLING_MEASUREMENTS]
            window_slope = np.polyfit(
                window.step.to_numpy(dtype=float), window[column].to_numpy(dtype=float), 1
            )[0]
            if len(rolling_rows) <= start:
                rolling_rows.append({
                    "window_start_step": int(window.step.iloc[0]),
                    "window_end_step": int(window.step.iloc[-1]),
                })
            rolling_rows[start][f"{label}_trend_per_1000"] = float(window_slope * 1000)
    return summary, pd.DataFrame(rolling_rows)


def _history_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    numeric = [column for column in frame if column != "initialization"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame


def collect_constant_late_trends(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect late-window diagnostics for the four completed constant-rate runs."""
    runs, histories = load_training_horizon_results(project_root)
    summaries, rolling_frames, history_frames = [], [], []
    for run in runs:
        history = _history_frame(histories[run["key"]])
        summary, rolling = summarize_late_history(history)
        identity = {
            "initialization": run["initialization"],
            "learning_rate": run["learning_rate"],
        }
        summaries.append({**identity, **summary})
        rolling_frames.append(rolling.assign(**identity))
        history_frames.append(history.assign(**identity))
    return (
        pd.DataFrame(summaries).sort_values(["initialization", "learning_rate"]),
        pd.concat(rolling_frames, ignore_index=True),
        pd.concat(history_frames, ignore_index=True),
    )


def plot_late_trends(history: pd.DataFrame, rolling: pd.DataFrame) -> Figure:
    """Plot late raw validation curves, fitted lines and moving-window trends."""
    figure, axes = plt.subplots(4, 2, figsize=(11, 11), sharex="row")
    for column_index, initialization in enumerate(INITIALIZATIONS):
        for learning_rate in (1e-4, 1e-3):
            rows = history[
                (history.initialization == initialization)
                & np.isclose(history.learning_rate, learning_rate)
                & history.step.isin(LATE_STEPS)
            ].sort_values("step")
            moving = rolling[
                (rolling.initialization == initialization)
                & np.isclose(rolling.learning_rate, learning_rate)
            ].sort_values("window_end_step")
            colour = LEARNING_RATE_COLOURS[learning_rate]
            label = learning_rate_label(learning_rate)
            for raw_row, trend_row, metric in ((0, 1, "iou"), (2, 3, "loss")):
                value_column = "validation_mean_iou" if metric == "iou" else "validation_loss"
                values = rows[value_column].to_numpy(dtype=float)
                steps = rows.step.to_numpy(dtype=float)
                slope, intercept = np.polyfit(steps, values, 1)
                axes[raw_row, column_index].plot(
                    steps, values, color=colour, marker="o", markersize=3, label=label
                )
                axes[raw_row, column_index].plot(
                    steps, slope * steps + intercept, color=colour, linestyle="--", linewidth=1.5
                )
                axes[trend_row, column_index].plot(
                    moving.window_end_step,
                    moving[f"{metric}_trend_per_1000"],
                    color=colour,
                    marker="o",
                    markersize=3,
                )
        axes[0, column_index].set_title(INITIALIZATION_LABELS[initialization])
        axes[0, column_index].legend(title="Constant rate", frameon=False)
        axes[1, column_index].axhline(0, color="#555555", linewidth=1)
        axes[3, column_index].axhline(0, color="#555555", linewidth=1)
        for axis in axes[:, column_index]:
            axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Validation mean per-image IoU")
    axes[1, 0].set_ylabel("IoU trend per 1,000 updates")
    axes[2, 0].set_ylabel("Validation loss")
    axes[3, 0].set_ylabel("Loss trend per 1,000 updates")
    axes[3, 0].set_xlabel("End step of 1,000-update window")
    axes[3, 1].set_xlabel("End step of 1,000-update window")
    figure.suptitle("Late validation trajectories: raw values, linear fits and moving trends")
    figure.tight_layout()
    return figure


def collect_lr_drop_comparison(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare the two fixed-drop runs with their constant-1e-3 references."""
    reference_runs, reference_histories = load_training_horizon_results(project_root)
    drop_runs, drop_histories = load_late_lr_drop_results(project_root)
    rows, history_frames, rolling_frames = [], [], []
    candidates = []
    for run in reference_runs:
        if np.isclose(run["learning_rate"], 1e-3):
            candidates.append(("constant 1e-3", run, reference_histories[run["key"]]))
    for run in drop_runs:
        candidates.append(("1e-3 → 1e-4 at step 2,001", run, drop_histories[run["key"]]))

    for protocol, run, raw_history in candidates:
        history = _history_frame(raw_history)
        trend, rolling = summarize_late_history(history)
        trained = history[history.step > 0]
        best = trained.loc[trained.validation_mean_iou.idxmax()]
        endpoint = trained.loc[trained.step == 4000].iloc[0]
        summary = run["summary"]
        identity = {"initialization": run["initialization"], "protocol": protocol}
        rows.append({
            **identity,
            "best_step": int(best.step),
            "best_validation_iou": float(best.validation_mean_iou),
            "iou_step_4000": float(endpoint.validation_mean_iou),
            "validation_loss_step_4000": float(endpoint.validation_loss),
            "selected_validation_dice": summary["best_validation"]["mean_dice"],
            "selected_validation_ap": summary["best_validation"]["mean_average_precision"],
            "selected_training_iou": summary["matched_training_evaluation"]["mean_iou"],
            "total_minutes": summary["total_seconds"] / 60,
            **trend,
        })
        if protocol.startswith("constant"):
            history["learning_rate_used"] = 1e-3
        else:
            history["learning_rate_used"] = np.where(history.step <= 2000, 1e-3, 1e-4)
        history_frames.append(history.assign(**identity))
        rolling_frames.append(rolling.assign(**identity))
    return (
        pd.DataFrame(rows).sort_values(["initialization", "protocol"]),
        pd.concat(history_frames, ignore_index=True),
        pd.concat(rolling_frames, ignore_index=True),
    )


def plot_lr_drop_comparison(history: pd.DataFrame) -> Figure:
    """Plot validation IoU and loss for fixed-drop runs and constant references."""
    colours = {"constant 1e-3": "#D55E00", "1e-3 → 1e-4 at step 2,001": "#0072B2"}
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for column, initialization in enumerate(INITIALIZATIONS):
        for protocol, colour in colours.items():
            rows = history[
                (history.initialization == initialization) & (history.protocol == protocol)
            ].sort_values("step")
            axes[0, column].plot(rows.step, rows.validation_mean_iou, color=colour, label=protocol)
            axes[1, column].plot(rows.step, rows.validation_loss, color=colour)
        for axis in axes[:, column]:
            axis.axvline(2000, color="#555555", linestyle="--", linewidth=1)
            axis.grid(alpha=0.2)
        axes[0, column].set_title(INITIALIZATION_LABELS[initialization])
        axes[0, column].legend(frameon=False)
        axes[1, column].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("Validation mean per-image IoU")
    axes[1, 0].set_ylabel("Validation loss")
    figure.suptitle("Predeclared learning-rate drop versus constant 1e-3")
    figure.tight_layout()
    return figure


def collect_subset_replication_comparison(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Collect the matched fixed-drop results for subset seeds 17 and 29.

    Returns
    -------
    results, paired_differences, history, overlap_count:
        Four run summaries, ImageNet-minus-random differences within each
        subset, all evaluation histories and the number of shared training IDs.
    """
    runs, histories = load_late_lr_drop_repetitions(project_root, (1, 2))
    result_rows, history_frames = [], []
    training_sets: dict[int, set[str]] = {}
    for run in runs:
        history = _history_frame(histories[run["key"]])
        trend, _ = summarize_late_history(history)
        trained = history[history.step > 0]
        best = trained.loc[trained.validation_mean_iou.idxmax()]
        endpoint = trained.loc[trained.step == 4000].iloc[0]
        summary = run["summary"]
        result_rows.append({
            "subset_seed": run["subset_seed"],
            "initialization": run["initialization"],
            "best_step": int(best.step),
            "best_validation_iou": float(best.validation_mean_iou),
            "iou_step_4000": float(endpoint.validation_mean_iou),
            "validation_loss_step_4000": float(endpoint.validation_loss),
            "selected_validation_dice": summary["best_validation"]["mean_dice"],
            "selected_validation_ap": summary["best_validation"]["mean_average_precision"],
            "total_minutes": summary["total_seconds"] / 60,
            **trend,
        })
        history_frames.append(history.assign(
            subset_seed=run["subset_seed"], initialization=run["initialization"]
        ))
        training_sets[run["subset_seed"]] = set(run["config"]["training_ids"])

    results = pd.DataFrame(result_rows).sort_values(["subset_seed", "initialization"])
    paired_rows = []
    for subset_seed, group in results.groupby("subset_seed"):
        by_initialization = group.set_index("initialization")
        paired_rows.append({
            "subset_seed": int(subset_seed),
            "imagenet_minus_random_best_iou": float(
                by_initialization.loc["imagenet", "best_validation_iou"]
                - by_initialization.loc["random", "best_validation_iou"]
            ),
            "imagenet_minus_random_iou_step_4000": float(
                by_initialization.loc["imagenet", "iou_step_4000"]
                - by_initialization.loc["random", "iou_step_4000"]
            ),
            "imagenet_minus_random_dice": float(
                by_initialization.loc["imagenet", "selected_validation_dice"]
                - by_initialization.loc["random", "selected_validation_dice"]
            ),
            "imagenet_minus_random_ap": float(
                by_initialization.loc["imagenet", "selected_validation_ap"]
                - by_initialization.loc["random", "selected_validation_ap"]
            ),
        })
    overlap_count = len(training_sets[17] & training_sets[29])
    return (
        results,
        pd.DataFrame(paired_rows),
        pd.concat(history_frames, ignore_index=True),
        overlap_count,
    )


def plot_subset_replication_curves(history: pd.DataFrame) -> Figure:
    """Plot fixed-drop validation curves for both subset repetitions."""
    colours = {"random": "#D55E00", "imagenet": "#0072B2"}
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for column, subset_seed in enumerate((17, 29)):
        for initialization in INITIALIZATIONS:
            rows = history[
                (history.subset_seed == subset_seed)
                & (history.initialization == initialization)
            ].sort_values("step")
            label = INITIALIZATION_LABELS[initialization]
            axes[0, column].plot(
                rows.step, rows.validation_mean_iou,
                color=colours[initialization], label=label,
            )
            axes[1, column].plot(
                rows.step, rows.validation_loss, color=colours[initialization]
            )
        for axis in axes[:, column]:
            axis.axvline(2000, color="#555555", linestyle="--", linewidth=1)
            axis.grid(alpha=0.2)
        axes[0, column].set_title(f"Training-subset seed {subset_seed}")
        axes[0, column].legend(frameon=False)
        axes[1, column].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("Validation mean per-image IoU")
    axes[1, 0].set_ylabel("Validation loss")
    figure.suptitle("Fixed learning-rate drop across two training subsets")
    figure.tight_layout()
    return figure
