"""Tables and figures for the paired WHU external-replication pilot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
import torch

from .late_training_analysis import summarize_late_history
from .model import build_model
from .training import restore_checkpoint, select_device
from .training_data import WHUTensorStore
from .whu_data import load_whu_binary_target, read_whu_image, read_whu_mask
from .whu_pilot import load_whu_learning_curve_runs, load_whu_pilot_runs


INITIALIZATION_LABELS = {"random": "Random", "imagenet": "ImageNet"}
LEARNING_CURVE_SIZES = (25, 100, 500)


def _collect_whu_tables(runs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect result, history and per-image tables from resolved WHU runs."""
    result_rows, history_frames, metric_frames = [], [], []
    for run in runs:
        summary = run["summary"]
        validation = summary["best_validation"]
        training = summary["matched_training_evaluation"]
        groups = validation["reference_groups"]
        history = pd.DataFrame(run["history"])
        for column in history.columns:
            if column != "initialization":
                history[column] = pd.to_numeric(history[column], errors="coerce")
        last = history.iloc[-1]
        late, _ = summarize_late_history(history)
        result_rows.append({
            "subset_repetition": run["subset_repetition"],
            "subset_seed": run["subset_seed"],
            "training_size": run["training_size"],
            "initialization": run["initialization"],
            "best_step": int(summary["best_step"]),
            "best_validation_iou": validation["mean_iou"],
            "last_validation_iou": float(last.validation_mean_iou),
            "last_validation_loss": float(last.validation_loss),
            "last_minus_best_validation_iou": float(last.validation_mean_iou) - validation["mean_iou"],
            "selected_validation_dice": validation["mean_dice"],
            "selected_validation_ap_positive": validation["mean_average_precision"],
            "ap_image_count": validation["ap_image_count"],
            "positive_image_count": groups["positive_reference"]["image_count"],
            "positive_mean_iou": groups["positive_reference"]["mean_iou"],
            "positive_mean_dice": groups["positive_reference"]["mean_dice"],
            "empty_image_count": groups["empty_reference"]["image_count"],
            "empty_mean_predicted_building_fraction": groups["empty_reference"]["mean_predicted_building_fraction"],
            "empty_fraction_with_any_prediction": groups["empty_reference"]["fraction_with_any_predicted_building"],
            "selected_training_iou": training["mean_iou"],
            "selected_training_loss": training["loss"],
            "selected_training_dice": training["mean_dice"],
            "selected_training_ap_positive": training["mean_average_precision"],
            "training_validation_iou_gap": training["mean_iou"] - validation["mean_iou"],
            "examples_processed": summary["examples_processed"],
            "equivalent_data_passages": summary["equivalent_data_passages"],
            "incomplete_batches_retained": summary["incomplete_batches_retained"],
            "training_seconds": summary["training_seconds"],
            "evaluation_seconds_during_training": summary["evaluation_seconds_during_training"],
            "total_seconds": summary["total_seconds"],
            **late,
        })
        history.insert(0, "training_size", run["training_size"])
        history.insert(0, "subset_seed", run["subset_seed"])
        history.insert(0, "subset_repetition", run["subset_repetition"])
        history.insert(0, "initialization", run["initialization"])
        history_frames.append(history)
        metrics = pd.DataFrame(run["metrics"])
        for column in ("tp", "fp", "fn", "iou", "dice", "average_precision"):
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
        metrics.insert(0, "subset_seed", run["subset_seed"])
        metrics.insert(0, "subset_repetition", run["subset_repetition"])
        metrics.insert(0, "initialization", run["initialization"])
        metric_frames.append(metrics)
    return (
        pd.DataFrame(result_rows).sort_values(
            ["subset_seed", "training_size", "initialization"]
        ),
        pd.concat(history_frames, ignore_index=True),
        pd.concat(metric_frames, ignore_index=True),
    )


def collect_whu_pilot_tables(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect the six n=100 subset-replication runs."""
    return _collect_whu_tables(load_whu_pilot_runs(project_root))


def collect_whu_learning_curve_tables(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect the eighteen comparable n=25/100/500 WHU runs."""
    return _collect_whu_tables(load_whu_learning_curve_runs(project_root))


def choose_whu_validation_examples(metrics: pd.DataFrame, subset_seed: int = 17) -> pd.DataFrame:
    """Select common success/failure cases symmetrically from both models.

    Positive-reference success and failure maximize and minimize mean paired
    IoU. Empty-reference success and failure minimize and maximize the mean
    paired predicted building fraction. Ties use lexicographic sample ID.
    """
    validation = metrics[
        (metrics.split == "validation") & (metrics.subset_seed == subset_seed)
    ].copy()
    validation["reference_positive"] = validation.tp + validation.fn > 0
    validation["predicted_fraction"] = (validation.tp + validation.fp) / (512 * 512)
    paired = validation.groupby("id", as_index=False).agg(
        reference_positive=("reference_positive", "first"),
        mean_iou=("iou", "mean"),
        mean_predicted_fraction=("predicted_fraction", "mean"),
    )
    positive = paired[paired.reference_positive].sort_values(["mean_iou", "id"])
    empty = paired[~paired.reference_positive].sort_values(["mean_predicted_fraction", "id"])
    choices = [
        ("positive failure", positive.iloc[0]),
        ("positive success", positive.iloc[-1]),
        ("empty success", empty.iloc[0]),
        ("empty failure", empty.iloc[-1]),
    ]
    return pd.DataFrame({
        "selection": [label for label, _ in choices],
        "id": [row.id for _, row in choices],
        "reference_positive": [bool(row.reference_positive) for _, row in choices],
        "paired_mean_iou": [float(row.mean_iou) for _, row in choices],
        "paired_mean_predicted_fraction": [float(row.mean_predicted_fraction) for _, row in choices],
    })


def whu_initialization_effects(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paired ImageNet-minus-Random effects and descriptive summaries."""
    rows = []
    for seed, group in results.groupby("subset_seed"):
        pair = group.set_index("initialization")
        rows.append({
            "dataset": "WHU", "subset_seed": int(seed),
            "random_best_iou": pair.loc["random", "best_validation_iou"],
            "imagenet_best_iou": pair.loc["imagenet", "best_validation_iou"],
            "imagenet_minus_random_best_iou": (
                pair.loc["imagenet", "best_validation_iou"]
                - pair.loc["random", "best_validation_iou"]
            ),
            "imagenet_minus_random_endpoint_iou": (
                pair.loc["imagenet", "last_validation_iou"]
                - pair.loc["random", "last_validation_iou"]
            ),
        })
    effects = pd.DataFrame(rows).sort_values("subset_seed")
    summary = pd.DataFrame([{
        "dataset": "WHU", "repetitions": len(effects),
        "mean_imagenet_minus_random_best_iou": effects.imagenet_minus_random_best_iou.mean(),
        "sample_sd_imagenet_minus_random_best_iou": effects.imagenet_minus_random_best_iou.std(ddof=1),
        "mean_imagenet_minus_random_endpoint_iou": effects.imagenet_minus_random_endpoint_iou.mean(),
        "sample_sd_imagenet_minus_random_endpoint_iou": effects.imagenet_minus_random_endpoint_iou.std(ddof=1),
    }])
    return effects, summary


def rid_whu_initialization_effects(project_root: Path, whu_effects: pd.DataFrame) -> pd.DataFrame:
    """Combine WHU effects with matched RID uniform-random n=100 effects."""
    rid = pd.read_csv(project_root / "reports/selection_strategy_n100_results.csv")
    rid = rid[rid.strategy == "uniform_random"]
    rows = []
    for seed, group in rid.groupby("subset_seed"):
        pair = group.set_index("initialization")
        rows.append({
            "dataset": "RID", "subset_seed": int(seed),
            "random_best_iou": pair.loc["random", "best_validation_iou"],
            "imagenet_best_iou": pair.loc["imagenet", "best_validation_iou"],
            "imagenet_minus_random_best_iou": (
                pair.loc["imagenet", "best_validation_iou"]
                - pair.loc["random", "best_validation_iou"]
            ),
        })
    comparison = pd.concat([pd.DataFrame(rows), whu_effects.drop(
        columns="imagenet_minus_random_endpoint_iou"
    )], ignore_index=True)
    if len(comparison) != 6:
        raise ValueError("Expected three paired n=100 effects for each dataset.")
    return comparison.sort_values(["dataset", "subset_seed"])


def whu_learning_curve_comparisons(
    results: pd.DataFrame, splits: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return paired effects, size gains, descriptive summaries and ID overlaps."""
    paired_rows = []
    for (repetition, seed, size), group in results.groupby(
        ["subset_repetition", "subset_seed", "training_size"]
    ):
        pair = group.set_index("initialization")
        paired_rows.append({
            "subset_repetition": int(repetition), "subset_seed": int(seed),
            "training_size": int(size),
            "imagenet_minus_random_best_iou": (
                pair.loc["imagenet", "best_validation_iou"]
                - pair.loc["random", "best_validation_iou"]
            ),
            "imagenet_minus_random_last_iou": (
                pair.loc["imagenet", "last_validation_iou"]
                - pair.loc["random", "last_validation_iou"]
            ),
        })

    gain_rows = []
    for (repetition, seed, initialization), group in results.groupby(
        ["subset_repetition", "subset_seed", "initialization"]
    ):
        by_size = group.set_index("training_size")
        for lower, upper in zip(LEARNING_CURVE_SIZES[:-1], LEARNING_CURVE_SIZES[1:]):
            gain_rows.append({
                "subset_repetition": int(repetition), "subset_seed": int(seed),
                "initialization": initialization, "size_change": f"{lower}_to_{upper}",
                "best_iou_gain": (
                    by_size.loc[upper, "best_validation_iou"]
                    - by_size.loc[lower, "best_validation_iou"]
                ),
                "last_iou_gain": (
                    by_size.loc[upper, "last_validation_iou"]
                    - by_size.loc[lower, "last_validation_iou"]
                ),
            })
    summary = (
        results.groupby(["training_size", "initialization"])
        .agg(
            best_iou_mean=("best_validation_iou", "mean"),
            best_iou_sample_sd=("best_validation_iou", "std"),
            last_iou_mean=("last_validation_iou", "mean"),
            last_iou_sample_sd=("last_validation_iou", "std"),
        )
        .reset_index()
    )
    overlaps = pd.DataFrame(
        splits["learning_curve_subsets"]["pairwise_id_overlaps"]
    )
    return pd.DataFrame(paired_rows), pd.DataFrame(gain_rows), summary, overlaps


def plot_whu_data_efficiency(results: pd.DataFrame):
    """Plot best and final WHU IoU for all saved image selections."""
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    colours = {"random": "#E76F51", "imagenet": "#2878B5"}
    seed_markers = {17: "o", 29: "s", 43: "^"}
    for axis, (metric, title) in zip(
        axes,
        (("best_validation_iou", "Selected maximum"),
         ("last_validation_iou", "Step 4,000")),
    ):
        for initialization in ("random", "imagenet"):
            rows = results[results.initialization == initialization]
            for seed, selection in rows.groupby("subset_seed"):
                selection = selection.sort_values("training_size")
                axis.plot(
                    selection.training_size, selection[metric], marker=seed_markers[int(seed)],
                    color=colours[initialization], alpha=0.45, linewidth=1,
                )
            means = rows.groupby("training_size")[metric].mean()
            axis.plot(
                means.index, means.values, marker="D", linewidth=2.4,
                color=colours[initialization], label=INITIALIZATION_LABELS[initialization],
            )
        axis.set_xscale("log")
        axis.set_xticks(LEARNING_CURVE_SIZES, labels=[str(size) for size in LEARNING_CURVE_SIZES])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("Labelled training images")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Validation mean per-image IoU")
    axes[0].legend(frameon=False)
    axes[1].legend(
        handles=[
            Line2D([0], [0], marker=marker, color="#777777", linewidth=1, label=str(seed))
            for seed, marker in seed_markers.items()
        ],
        title="Subset seed", frameon=False, loc="lower right",
    )
    figure.suptitle("WHU data-efficiency across three nested image selections")
    figure.tight_layout()
    return figure


def plot_whu_learning_curve_histories(history: pd.DataFrame, subset_seed: int):
    """Plot complete validation IoU and loss trajectories for one WHU selection."""
    subset = history[history.subset_seed == subset_seed]
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    colours = {"random": "#E76F51", "imagenet": "#2878B5"}
    for column, size in enumerate(LEARNING_CURVE_SIZES):
        for initialization in ("random", "imagenet"):
            rows = subset[
                (subset.training_size == size)
                & (subset.initialization == initialization)
            ].sort_values("step")
            axes[0, column].plot(
                rows.step, rows.validation_mean_iou, color=colours[initialization],
                label=INITIALIZATION_LABELS[initialization],
            )
            axes[1, column].plot(
                rows.step, rows.validation_loss, color=colours[initialization]
            )
        for axis in axes[:, column]:
            axis.axvline(2000, color="#777777", linestyle="--", linewidth=1)
            axis.grid(alpha=0.2)
        axes[0, column].set_title(f"{size} training images")
        axes[1, column].set_xlabel("Optimizer update")
    axes[0, 0].set_ylabel("Validation mean IoU")
    axes[1, 0].set_ylabel("Validation loss")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(f"WHU validation trajectories: subset seed {subset_seed}")
    figure.tight_layout()
    return figure


def plot_whu_learning_curves(history: pd.DataFrame):
    """Plot validation IoU and loss for all three paired WHU repetitions."""
    seeds = sorted(history.subset_seed.unique())
    figure, axes = plt.subplots(len(seeds), 2, figsize=(10, 9), sharex=True)
    colors = {"random": "#E76F51", "imagenet": "#2878B5"}
    for row, seed in enumerate(seeds):
        for initialization in ("random", "imagenet"):
            subset = history[
                (history.initialization == initialization) & (history.subset_seed == seed)
            ]
            axes[row, 0].plot(subset.step, subset.validation_mean_iou,
                              label=INITIALIZATION_LABELS[initialization], color=colors[initialization])
            axes[row, 1].plot(subset.step, subset.validation_loss,
                              label=INITIALIZATION_LABELS[initialization], color=colors[initialization])
        for axis in axes[row]:
            axis.axvline(2000, color="#777777", linestyle="--", linewidth=1)
            axis.grid(alpha=0.2)
        axes[row, 0].set_ylabel(f"Seed {seed}\nmean IoU")
        axes[row, 1].set_ylabel("Validation loss")
    axes[0, 0].legend(frameon=False)
    axes[-1, 0].set_xlabel("Optimizer update")
    axes[-1, 1].set_xlabel("Optimizer update")
    figure.suptitle("WHU validation trajectories under the shared 4,000-update recipe")
    figure.tight_layout()
    return figure


def plot_cross_dataset_effects(comparison: pd.DataFrame):
    """Plot six paired initialization effects without comparing absolute IoU."""
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    colors = {"RID": "#6A4C93", "WHU": "#1982C4"}
    offsets = {"RID": -0.08, "WHU": 0.08}
    for dataset in ("RID", "WHU"):
        subset = comparison[comparison.dataset == dataset]
        x = np.arange(len(subset)) + offsets[dataset]
        axis.scatter(x, subset.imagenet_minus_random_best_iou, s=55,
                     color=colors[dataset], label=dataset, zorder=3)
        axis.plot(x, subset.imagenet_minus_random_best_iou, color=colors[dataset], alpha=0.5)
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_xticks(np.arange(3), ["Seed 17", "Seed 29", "Seed 43"])
    axis.set_ylabel("ImageNet − Random best validation IoU")
    axis.set_title("Paired n=100 initialization effects within each dataset")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return figure


def _predictions(project_root: Path, runs: list[dict], sample_ids: list[str]) -> dict[str, np.ndarray]:
    manifest = json.loads(
        (project_root / "data/metadata/whu/data_manifest.json").read_text()
    )
    store = WHUTensorStore(project_root, manifest)
    device = select_device()
    batch = store.batch(sample_ids)
    predictions = {}
    for run in runs:
        model, _ = build_model(run["initialization"], run["config"]["seeds"]["model"])
        restore_checkpoint(
            model, run["directory"] / "checkpoint.pt",
            run["metadata"]["compatibility_identity"]["config_sha256"],
            run["metadata"]["compatibility_identity"]["input_sha256"],
        )
        model.to(device).eval()
        with torch.inference_mode():
            predictions[run["initialization"]] = model(batch["image"].to(device)).sigmoid().cpu().numpy()[:, 0]
    return predictions


def plot_whu_validation_predictions(project_root: Path, examples: pd.DataFrame):
    """Plot identical WHU validation cases for both selected checkpoints."""
    runs = [run for run in load_whu_pilot_runs(project_root) if run["subset_seed"] == 17]
    manifest = json.loads(
        (project_root / "data/metadata/whu/data_manifest.json").read_text()
    )
    records = {sample["id"]: sample for sample in manifest["samples"]}
    sample_ids = examples.id.tolist()
    predictions = _predictions(project_root, runs, sample_ids)
    cmap = ListedColormap(["#111111", "#2DC653", "#FF8C1A", "#4361EE"])
    figure, axes = plt.subplots(len(sample_ids), 4, figsize=(10, 2.65 * len(sample_ids)), squeeze=False)
    for row, choice in enumerate(examples.itertuples(index=False)):
        record = records[choice.id]
        image = read_whu_image(project_root / record["image"])
        target = load_whu_binary_target(read_whu_mask(project_root / record["mask"]))
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"{choice.selection}\n{choice.id}", fontsize=8)
        for column, initialization in enumerate(("random", "imagenet"), 2):
            prediction = predictions[initialization][row] >= 0.5
            error = np.zeros_like(target, dtype=np.uint8)
            error[prediction & target] = 1
            error[prediction & ~target] = 2
            error[~prediction & target] = 3
            union = np.logical_or(prediction, target).sum()
            iou = np.logical_and(prediction, target).sum() / union if union else 1.0
            axes[row, column].imshow(error, cmap=cmap, vmin=0, vmax=3)
            axes[row, column].set_title(f"{INITIALIZATION_LABELS[initialization]} · IoU {iou:.3f}", fontsize=9)
    axes[0, 0].set_title("RGB image")
    axes[0, 1].set_title("Building-footprint reference")
    for axis in axes.ravel():
        axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle("Common WHU validation cases: green TP, orange FP, blue FN", y=1.005)
    figure.tight_layout()
    return figure
