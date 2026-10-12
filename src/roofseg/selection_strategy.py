"""Random-versus-spatial n=100 subset configurations and analysis."""

from __future__ import annotations

import csv
from copy import deepcopy
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from shapely.ops import unary_union
import torch

from .data import read_image
from .integrity import load_json
from .late_training_analysis import _history_frame
from .model import build_model
from .objectives import binary_metrics
from .optimization import (INITIALIZATIONS, build_late_lr_drop_config,
                           late_lr_drop_run_directory)
from .optimization_analysis import INITIALIZATION_LABELS
from .spatial import metric_footprints
from .training import restore_checkpoint, select_device
from .training_data import RIDTensorStore

TRAINING_SIZE = 100
SUBSET_REPETITIONS = (1, 2, 3)
SPATIAL = "spatial_maximin"
RANDOM = "uniform_random"
STRATEGIES = (SPATIAL, RANDOM)
STRATEGY_LABELS = {SPATIAL: "Spatial maximin", RANDOM: "Uniform random"}
STRATEGY_COLOURS = {SPATIAL: "#0072B2", RANDOM: "#D55E00"}


def subset_record(splits: dict, strategy: str, repetition: int) -> dict:
    """Return one saved selection record without changing historical paths."""
    if strategy == SPATIAL:
        source = splits["training_subsets"]["repetitions"]
    elif strategy == RANDOM:
        source = splits["training_subsets"]["additional_strategies"][RANDOM]["repetitions"]
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
    try:
        return source[str(repetition)]
    except KeyError as error:
        raise ValueError(f"Unknown saved subset repetition: {repetition}") from error


def random_selection_run_name(initialization: str, subset_seed: int) -> str:
    """Return a run name that identifies strategy, size, seed and recipe."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("Unknown initialization.")
    return (f"n100_{initialization}_uniform_random_subsetseed{subset_seed}_"
            "lr1e-3_to_1e-4_step2001_steps4000")


def build_random_selection_config(
    splits: dict, initialization: str, subset_repetition: int
) -> dict:
    """Resolve one random-subset run while preserving the n=100 recipe."""
    config = deepcopy(build_late_lr_drop_config(
        splits, initialization, subset_repetition, TRAINING_SIZE
    ))
    record = subset_record(splits, RANDOM, subset_repetition)
    subset_seed = int(record["seed"])
    training_ids = record["subsets"][str(TRAINING_SIZE)]
    config.update({
        "run_name": random_selection_run_name(initialization, subset_seed),
        "experiment": "stage5_random_vs_spatial_selection_n100",
        "selection_strategy": RANDOM,
        "training_ids": training_ids,
    })
    config["data"]["training_subset"] = (
        "training_subsets.additional_strategies.uniform_random.repetitions."
        f"{subset_repetition}.subsets.{TRAINING_SIZE}"
    )
    return config


def random_selection_run_directory(
    project_root: Path, splits: dict, initialization: str, subset_repetition: int
) -> Path:
    """Return the unique directory for one random-selection run."""
    seed = int(subset_record(splits, RANDOM, subset_repetition)["seed"])
    return (project_root / "runs/learning_curves" / splits["split_name"]
            / "random_vs_spatial_n100" / random_selection_run_name(initialization, seed))


def _ids(splits: dict, strategy: str, repetition: int) -> list[str]:
    return subset_record(splits, strategy, repetition)["subsets"][str(TRAINING_SIZE)]


def spatial_subset_diagnostics(manifest: dict, splits: dict) -> pd.DataFrame:
    """Measure footprint coverage, overlap and centre spacing for six subsets.

    Areas are expressed in square kilometres. The overlap fraction is one minus
    union area divided by summed image-footprint area.
    """
    footprints = metric_footprints(manifest["samples"])
    training_union_area = unary_union(
        [footprints[sample_id] for sample_id in splits["roles"]["training"]]
    ).area
    rows = []
    for strategy in STRATEGIES:
        for repetition in SUBSET_REPETITIONS:
            record = subset_record(splits, strategy, repetition)
            ids = _ids(splits, strategy, repetition)
            geometries = [footprints[sample_id] for sample_id in ids]
            union_area = unary_union(geometries).area
            summed_area = sum(geometry.area for geometry in geometries)
            centres = np.asarray([[geometry.centroid.x, geometry.centroid.y]
                                  for geometry in geometries])
            distances = np.sqrt(((centres[:, None] - centres[None, :]) ** 2).sum(axis=2))
            np.fill_diagonal(distances, np.inf)
            rows.append({
                "selection_strategy": strategy,
                "subset_repetition": repetition,
                "subset_seed": int(record["seed"]),
                "training_images": len(ids),
                "union_area_km2": union_area / 1e6,
                "available_training_area_fraction": union_area / training_union_area,
                "summed_area_km2": summed_area / 1e6,
                "overlap_fraction": max(0.0, 1.0 - union_area / summed_area),
                "median_nearest_centre_distance_m": float(np.median(distances.min(axis=1))),
            })
    return pd.DataFrame(rows)


def subset_id_overlaps(splits: dict) -> pd.DataFrame:
    """Count paired-strategy and between-repetition shared training IDs."""
    rows = []
    for repetition in SUBSET_REPETITIONS:
        spatial = set(_ids(splits, SPATIAL, repetition))
        random = set(_ids(splits, RANDOM, repetition))
        seed = int(subset_record(splits, SPATIAL, repetition)["seed"])
        rows.append({"comparison": "matched_strategies", "strategy": "spatial_vs_random",
                     "first_seed": seed, "second_seed": seed,
                     "shared_ids": len(spatial & random)})
    for strategy in STRATEGIES:
        for first, second in combinations(SUBSET_REPETITIONS, 2):
            first_record = subset_record(splits, strategy, first)
            second_record = subset_record(splits, strategy, second)
            rows.append({"comparison": "between_repetitions", "strategy": strategy,
                         "first_seed": int(first_record["seed"]),
                         "second_seed": int(second_record["seed"]),
                         "shared_ids": len(set(_ids(splits, strategy, first))
                                           & set(_ids(splits, strategy, second)))})
    return pd.DataFrame(rows)


def plot_selection_map(manifest: dict, splits: dict) -> Figure:
    """Plot all six saved selections over the retained training footprint."""
    footprints = metric_footprints(manifest["samples"])
    background = [Polygon(np.asarray(footprints[sample_id].exterior.coords), closed=True)
                  for sample_id in splits["roles"]["training"]]
    figure, axes = plt.subplots(3, 2, figsize=(9, 11), sharex=True, sharey=True)
    for row, repetition in enumerate(SUBSET_REPETITIONS):
        for column, strategy in enumerate(STRATEGIES):
            axis = axes[row, column]
            axis.add_collection(PatchCollection(background, facecolor="#E5E5E5",
                                                 edgecolor="none", alpha=0.35))
            chosen = [Polygon(np.asarray(footprints[sample_id].exterior.coords), closed=True)
                      for sample_id in _ids(splits, strategy, repetition)]
            axis.add_collection(PatchCollection(chosen, facecolor=STRATEGY_COLOURS[strategy],
                                                 edgecolor="white", linewidth=0.15, alpha=0.75))
            seed = subset_record(splits, strategy, repetition)["seed"]
            axis.set_title(f"{STRATEGY_LABELS[strategy]} · seed {seed}")
            axis.set_aspect("equal")
            axis.autoscale_view()
            axis.ticklabel_format(style="plain", useOffset=False)
            if column == 0:
                axis.set_ylabel("Northing (m)")
            if row == 2:
                axis.set_xlabel("Easting (m)")
    figure.suptitle("n=100 training selections within the retained training region")
    figure.tight_layout()
    return figure


def _runs(project_root: Path) -> list[dict]:
    splits = load_json(project_root / "data/metadata/splits.json")
    runs = []
    for strategy in STRATEGIES:
        for repetition in SUBSET_REPETITIONS:
            seed = int(subset_record(splits, strategy, repetition)["seed"])
            for initialization in INITIALIZATIONS:
                directory = (late_lr_drop_run_directory(project_root, splits, initialization,
                                                         repetition, TRAINING_SIZE)
                             if strategy == SPATIAL else
                             random_selection_run_directory(project_root, splits, initialization,
                                                            repetition))
                config = load_json(directory / "config.json")
                metadata = load_json(directory / "metadata.json")
                summary = load_json(directory / "summary.json")
                if summary.get("status") != "completed":
                    raise ValueError(f"Incomplete comparison run: {directory.name}")
                with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
                    history = list(csv.DictReader(handle))
                runs.append({"strategy": strategy, "subset_repetition": repetition,
                             "subset_seed": seed, "initialization": initialization,
                             "directory": directory, "config": config, "metadata": metadata,
                             "summary": summary, "history": history})
    return runs


def collect_selection_results(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect run metrics and paired spatial-minus-random effects."""
    rows, histories = [], []
    for run in _runs(project_root):
        history = _history_frame(run["history"])
        trained = history[history.step > 0]
        endpoint = trained.loc[trained.step == 4000].iloc[0]
        summary = run["summary"]
        validation = summary["best_validation"]
        training = summary["matched_training_evaluation"]
        identity = {key: run[key] for key in
                    ("strategy", "subset_repetition", "subset_seed", "initialization")}
        rows.append({**identity, "best_step": summary["best_step"],
                     "best_validation_iou": validation["mean_iou"],
                     "iou_step_4000": float(endpoint.validation_mean_iou),
                     "selected_validation_dice": validation["mean_dice"],
                     "selected_validation_ap": validation["mean_average_precision"],
                     "selected_training_iou": training["mean_iou"],
                     "training_minus_validation_iou": training["mean_iou"] - validation["mean_iou"],
                     "training_minutes": summary["training_seconds"] / 60,
                     "total_minutes": summary["total_seconds"] / 60})
        histories.append(history.assign(**identity))
    results = pd.DataFrame(rows).sort_values(["subset_seed", "initialization", "strategy"])
    effects = []
    for (seed, initialization), group in results.groupby(["subset_seed", "initialization"]):
        values = group.set_index("strategy")
        spatial, random = values.loc[SPATIAL], values.loc[RANDOM]
        effects.append({"subset_seed": int(seed), "initialization": initialization,
                        "spatial_minus_random_best_iou": spatial.best_validation_iou - random.best_validation_iou,
                        "spatial_minus_random_endpoint_iou": spatial.iou_step_4000 - random.iou_step_4000,
                        "spatial_minus_random_dice": spatial.selected_validation_dice - random.selected_validation_dice,
                        "spatial_minus_random_ap": spatial.selected_validation_ap - random.selected_validation_ap})
    effects = pd.DataFrame(effects)
    summary = effects.groupby("initialization").agg(
        mean_best_iou_effect=("spatial_minus_random_best_iou", "mean"),
        sample_sd_best_iou_effect=("spatial_minus_random_best_iou", "std"),
        mean_endpoint_iou_effect=("spatial_minus_random_endpoint_iou", "mean"),
        sample_sd_endpoint_iou_effect=("spatial_minus_random_endpoint_iou", "std"),
    ).reset_index()
    return results, effects, summary


def plot_selection_performance(results: pd.DataFrame, effects: pd.DataFrame) -> Figure:
    """Plot individual paired scores and spatial-minus-random effects."""
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for initialization in INITIALIZATIONS:
        label = INITIALIZATION_LABELS[initialization]
        subset = results[results.initialization == initialization]
        for seed, pair in subset.groupby("subset_seed"):
            pair = pair.set_index("strategy")
            axes[0].plot([0, 1], [pair.loc[RANDOM, "best_validation_iou"],
                                  pair.loc[SPATIAL, "best_validation_iou"]],
                         marker="o", alpha=0.7, label=label if seed == 17 else None)
        selected = effects[effects.initialization == initialization]
        axes[1].scatter(selected.subset_seed, selected.spatial_minus_random_best_iou,
                        s=55, label=label)
    axes[0].set_xticks([0, 1], ["Uniform random", "Spatial maximin"])
    axes[0].set_ylabel("Best validation mean per-image IoU")
    axes[0].set_title("Matched subset-seed comparisons")
    axes[1].axhline(0, color="#555555", linewidth=1)
    axes[1].set_xticks([17, 29, 43])
    axes[1].set_xlabel("Selection seed")
    axes[1].set_ylabel("Spatial minus random IoU")
    axes[1].set_title("Paired strategy effect")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def _restore(run: dict, device: torch.device) -> torch.nn.Module:
    model, _ = build_model(run["initialization"], run["config"]["seeds"]["model"])
    restore_checkpoint(model, run["directory"] / "checkpoint.pt",
                       run["metadata"]["compatibility_identity"]["config_sha256"],
                       run["metadata"]["compatibility_identity"]["input_sha256"])
    return model.to(device).eval()


def choose_qualitative_examples(project_root: Path) -> pd.DataFrame:
    """Select two spatial improvements and two deteriorations by validation IoU."""
    rows = []
    for run in _runs(project_root):
        with (run["directory"] / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            for metric in csv.DictReader(handle):
                if metric["split"] == "validation":
                    rows.append({"subset_seed": run["subset_seed"],
                                 "initialization": run["initialization"],
                                 "strategy": run["strategy"], "id": metric["id"],
                                 "iou": float(metric["iou"])})
    pivot = pd.DataFrame(rows).pivot(index=["subset_seed", "initialization", "id"],
                                     columns="strategy", values="iou").dropna()
    values = (pivot[SPATIAL] - pivot[RANDOM]).rename("spatial_minus_random_iou").reset_index()
    worsened = values.sort_values("spatial_minus_random_iou").drop_duplicates("id").head(2)
    improved = values.sort_values("spatial_minus_random_iou", ascending=False).drop_duplicates("id").head(2)
    return pd.concat([worsened, improved]).sort_values("spatial_minus_random_iou").reset_index(drop=True)


def plot_qualitative_examples(project_root: Path, selected: pd.DataFrame) -> Figure:
    """Plot identical validation cases for both selection strategies."""
    runs = _runs(project_root)
    manifest = load_json(project_root / "data/metadata/data_manifest.json")
    records = {sample["id"]: sample for sample in manifest["samples"]}
    store = RIDTensorStore(project_root, manifest)
    device = select_device()
    cmap = ListedColormap(["#111111", "#2DC653", "#FF8C1A", "#4361EE"])
    figure, axes = plt.subplots(len(selected), 4, figsize=(11, 2.8 * len(selected)), squeeze=False)
    for row, choice in enumerate(selected.itertuples(index=False)):
        matching = {run["strategy"]: run for run in runs
                    if run["subset_seed"] == choice.subset_seed
                    and run["initialization"] == choice.initialization}
        batch = store.batch([choice.id])
        target = batch["target"].numpy()[0, 0].astype(bool)
        axes[row, 0].imshow(read_image(project_root / records[choice.id]["image"]))
        axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(
            f"ID {choice.id} · seed {choice.subset_seed}\n{INITIALIZATION_LABELS[choice.initialization]}"
            f" · ΔIoU {choice.spatial_minus_random_iou:+.3f}", rotation=0,
            ha="right", va="center", labelpad=12)
        for column, strategy in enumerate((RANDOM, SPATIAL), 2):
            model = _restore(matching[strategy], device)
            with torch.inference_mode():
                prediction = model(batch["image"].to(device)).sigmoid().cpu().numpy()[0, 0] >= 0.5
            error = np.zeros_like(target, dtype=np.uint8)
            error[prediction & target] = 1
            error[prediction & ~target] = 2
            error[~prediction & target] = 3
            iou = binary_metrics(prediction.astype(float), target)["iou"]
            axes[row, column].imshow(error, cmap=cmap, vmin=0, vmax=3)
            axes[row, column].set_title(f"{STRATEGY_LABELS[strategy]}\nIoU {iou:.3f}")
            del model
    for axis, title in zip(axes[0, :2], ("RGB validation image", "Binary reference")):
        axis.set_title(title)
    for axis in axes.ravel():
        axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle("Metric-selected improvements and deteriorations\n"
                    "green true positive, orange false positive, blue false negative", y=1.01)
    figure.tight_layout()
    return figure
