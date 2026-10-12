"""Fixed boundary-weighted BCE configurations, diagnostics and report figures."""

from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch

from .data import load_binary_target, read_image, read_mask
from .integrity import load_json
from .late_training_analysis import _history_frame, summarize_late_history
from .model import build_model
from .objectives import binary_metrics, boundary_band, boundary_iou
from .optimization import INITIALIZATIONS, build_late_lr_drop_config, late_lr_drop_run_directory
from .optimization_analysis import INITIALIZATION_LABELS
from .training import restore_checkpoint, select_device
from .training_data import RIDTensorStore

TRAINING_SIZE = 25
SUBSET_REPETITIONS = (1, 2)
REPLICATION_STRATEGY_REPETITIONS = {
    "reference": (1, 2, 3),
    "area_proportional": (1, 2),
    "equal_per_image": (1, 2, 3),
}
BOUNDARY_RADIUS = 3
VARIANTS = ("area_proportional", "equal_per_image")
NORMALIZATIONS = {
    "area_proportional": "proportional_band_area",
    "equal_per_image": "equal_per_nonempty_band",
}
VARIANT_LABELS = {
    "reference": "reference L0",
    "area_proportional": "A · area proportional",
    "equal_per_image": "B · equal per image",
}
COLOURS = {
    "reference": "#777777",
    "area_proportional": "#0072B2",
    "equal_per_image": "#D55E00",
}


def boundary_run_name(initialization: str, subset_seed: int, variant: str) -> str:
    """Return the unique name for one boundary-weighted n=25 run."""
    if initialization not in INITIALIZATIONS or variant not in VARIANTS:
        raise ValueError("Unknown boundary experiment configuration.")
    label = "boundary_area" if variant == "area_proportional" else "boundary_equal"
    return (
        f"n25_{initialization}_subsetseed{subset_seed}_{label}_r3_"
        "lr1e-3_to_1e-4_step2001_steps4000"
    )


def boundary_run_root(project_root: Path, splits: dict) -> Path:
    """Return the split-versioned directory for the boundary-weighted block."""
    return project_root / "runs/targeted" / splits["split_name"] / "boundary_weighted_bce_n25"


def training_band_fractions(
    store: RIDTensorStore,
    sample_ids: list[str],
    radius: int = BOUNDARY_RADIUS,
) -> dict[str, float]:
    """Return two-sided boundary-band fractions for fixed training images.

    Bands are derived from original-orientation binary targets. Square D4
    symmetries preserve their pixel counts and map the excluded border zone onto
    itself, so these fixed fractions also apply to every scheduled training view.
    """
    fractions = {}
    for sample_id in sample_ids:
        target = store.load(sample_id)[1].unsqueeze(0)
        fractions[sample_id] = float(boundary_band(target, radius=radius).mean())
    return fractions


def build_boundary_config(
    splits: dict,
    initialization: str,
    subset_repetition: int,
    variant: str,
    mean_training_band_fraction: float,
) -> dict:
    """Add exactly one fixed boundary-weighted BCE variant to its reference."""
    config = deepcopy(build_late_lr_drop_config(
        splits, initialization, subset_repetition, TRAINING_SIZE
    ))
    if variant not in VARIANTS:
        raise ValueError(f"Unknown boundary variant: {variant}")
    subset_seed = int(
        splits["training_subsets"]["repetitions"][str(subset_repetition)]["seed"]
    )
    config.update({
        "run_name": boundary_run_name(initialization, subset_seed, variant),
        "experiment": "stage4_boundary_weighted_bce_n25",
        "loss": {
            **config["loss"],
            "optimized_formula": (
                "L0_i + f_i * E_i" if variant == "area_proportional"
                else "L0_i + mean_training_band_fraction * E_i"
            ),
            "batch_aggregation": "equal mean over images",
            "validation_loss": "L0 only",
        },
        "boundary_weighted_bce": {
            "variant": variant,
            "normalization": NORMALIZATIONS[variant],
            "radius_pixels": BOUNDARY_RADIUS,
            "structuring_element": "square 7x7",
            "band": "binary-target dilation minus erosion",
            "excluded_crop_border_pixels": BOUNDARY_RADIUS,
            "empty_band_addition": 0.0,
            "mean_training_band_fraction": float(mean_training_band_fraction),
            "coefficient_source": "the 25 training masks only; fixed before training",
            "transform_policy": "derive from the D4-transformed binary target",
        },
    })
    return config


def boundary_run_directory(
    project_root: Path,
    splits: dict,
    initialization: str,
    subset_repetition: int,
    variant: str,
) -> Path:
    """Return the distinct directory for one boundary-weighted run."""
    subset_seed = int(
        splits["training_subsets"]["repetitions"][str(subset_repetition)]["seed"]
    )
    return boundary_run_root(project_root, splits) / boundary_run_name(
        initialization, subset_seed, variant
    )


def choose_band_examples(store: RIDTensorStore, sample_ids: list[str]) -> list[str]:
    """Choose three training masks spanning band fraction, including a crop-edge roof."""
    fractions = training_band_fractions(store, sample_ids)
    clipped = []
    for sample_id in sample_ids:
        target = store.load(sample_id)[1][0].bool()
        if target[0].any() or target[-1].any() or target[:, 0].any() or target[:, -1].any():
            clipped.append(sample_id)
    if not clipped:
        raise ValueError("No crop-edge roof exists in the selected training examples.")
    clipped_id = max(clipped, key=fractions.get)
    ordered = sorted(sample_ids, key=fractions.get)
    choices = [ordered[len(ordered) // 5], ordered[len(ordered) // 2], clipped_id]
    return list(dict.fromkeys(choices))


def plot_band_examples(
    project_root: Path,
    manifest: dict,
    sample_ids: list[str],
) -> Figure:
    """Plot RGB, binary target and fixed two-sided band for training examples."""
    store = RIDTensorStore(project_root, manifest)
    records = {sample["id"]: sample for sample in manifest["samples"]}
    figure, axes = plt.subplots(len(sample_ids), 3, figsize=(9, 3 * len(sample_ids)), squeeze=False)
    for row, sample_id in enumerate(sample_ids):
        record = records[sample_id]
        image = read_image(project_root / record["image"])
        target = store.load(sample_id)[1].unsqueeze(0)
        band = boundary_band(target, radius=BOUNDARY_RADIUS)[0, 0].numpy()
        crop_edge = bool(
            target[0, 0, 0].any() or target[0, 0, -1].any()
            or target[0, 0, :, 0].any() or target[0, 0, :, -1].any()
        )
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(target[0, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(band, cmap="magma", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(
            f"ID {sample_id}\nband {band.mean():.4f}" + (" · crop-edge roof" if crop_edge else "")
        )
    for axis, title in zip(axes[0], ("RGB training image", "Binary roof target", "Two-sided r=3 band")):
        axis.set_title(title)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Boundary-band definition; outer three pixels excluded", y=1.01)
    figure.tight_layout()
    return figure


def _comparison_runs(
    project_root: Path,
    strategy_repetitions: dict[str, tuple[int, ...]] | None = None,
) -> list[dict]:
    splits = load_json(project_root / "data/metadata/splits.json")
    strategy_repetitions = strategy_repetitions or {
        strategy: SUBSET_REPETITIONS for strategy in ("reference", *VARIANTS)
    }
    runs = []
    for strategy in ("reference", *VARIANTS):
        for repetition in strategy_repetitions[strategy]:
            seed = int(splits["training_subsets"]["repetitions"][str(repetition)]["seed"])
            for initialization in INITIALIZATIONS:
                directory = (
                    late_lr_drop_run_directory(project_root, splits, initialization, repetition, 25)
                    if strategy == "reference"
                    else boundary_run_directory(
                        project_root, splits, initialization, repetition, strategy
                    )
                )
                config = json.loads((directory / "config.json").read_text())
                metadata = json.loads((directory / "metadata.json").read_text())
                summary = json.loads((directory / "summary.json").read_text())
                if summary.get("status") != "completed":
                    raise ValueError(f"Boundary comparison run is incomplete: {directory.name}")
                key = f"seed{seed}_{initialization}_{strategy}"
                with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
                    history = list(csv.DictReader(handle))
                runs.append({
                    "key": key, "subset_repetition": repetition, "subset_seed": seed,
                    "initialization": initialization, "strategy": strategy,
                    "directory": directory, "config": config, "metadata": metadata,
                    "summary": summary, "history": history,
                })
    return runs


def _restore_model(run: dict, device: torch.device) -> torch.nn.Module:
    model, _ = build_model(run["initialization"], run["config"]["seeds"]["model"])
    restore_checkpoint(
        model,
        run["directory"] / "checkpoint.pt",
        run["metadata"]["compatibility_identity"]["config_sha256"],
        run["metadata"]["compatibility_identity"]["input_sha256"],
    )
    return model.to(device).eval()


def evaluate_boundary_runs(project_root: Path, runs: list[dict]) -> pd.DataFrame:
    """Evaluate symmetric inner-band IoU on one reference-defined subset."""
    manifest = load_json(project_root / "data/metadata/data_manifest.json")
    store = RIDTensorStore(project_root, manifest)
    device = select_device()
    rows = []
    for run in runs:
        model = _restore_model(run, device)
        validation_ids = run["config"]["validation_ids"]
        with torch.inference_mode():
            for start in range(0, len(validation_ids), run["config"]["batch_size"]):
                batch = store.batch(validation_ids[start:start + run["config"]["batch_size"]])
                probability = model(batch["image"].to(device)).sigmoid().cpu().numpy()[:, 0]
                targets = batch["target"].numpy()[:, 0].astype(bool)
                for index, sample_id in enumerate(batch["id"]):
                    prediction = probability[index] >= 0.5
                    metric = binary_metrics(probability[index], targets[index])
                    rows.append({
                        "key": run["key"], "subset_seed": run["subset_seed"],
                        "initialization": run["initialization"], "strategy": run["strategy"],
                        "id": sample_id, "iou": metric["iou"],
                        "boundary_iou": boundary_iou(
                            prediction, targets[index], radius=BOUNDARY_RADIUS
                        ),
                    })
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    return pd.DataFrame(rows)


def collect_boundary_results(
    project_root: Path,
    strategy_repetitions: dict[str, tuple[int, ...]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect run summaries, paired effects, histories and Boundary IoU."""
    runs = _comparison_runs(project_root, strategy_repetitions)
    boundary_metrics = evaluate_boundary_runs(project_root, runs)
    rows, history_frames = [], []
    for run in runs:
        history = _history_frame(run["history"])
        late, _ = summarize_late_history(history)
        trained = history[history.step > 0]
        best = trained.loc[trained.validation_mean_iou.idxmax()]
        endpoint = trained.loc[trained.step == 4000].iloc[0]
        summary = run["summary"]
        validation = summary["best_validation"]
        training = summary["matched_training_evaluation"]
        per_image = boundary_metrics[boundary_metrics.key == run["key"]]
        included = per_image.boundary_iou.notna()
        rows.append({
            "subset_repetition": run["subset_repetition"],
            "subset_seed": run["subset_seed"],
            "initialization": run["initialization"], "strategy": run["strategy"],
            "best_step": int(best.step), "best_validation_iou": float(best.validation_mean_iou),
            "iou_step_4000": float(endpoint.validation_mean_iou),
            "validation_loss_step_4000": float(endpoint.validation_loss),
            "selected_validation_loss": validation["loss"],
            "selected_validation_dice": validation["mean_dice"],
            "selected_validation_ap": validation["mean_average_precision"],
            "selected_training_iou": training["mean_iou"],
            "training_minus_validation_iou": training["mean_iou"] - validation["mean_iou"],
            "mean_boundary_iou": float(per_image.loc[included, "boundary_iou"].mean()),
            "boundary_images_included": int(included.sum()),
            "boundary_images_excluded": int((~included).sum()),
            "examples_processed": summary["examples_processed"],
            "equivalent_data_passages": summary["equivalent_data_passages"],
            "training_minutes": summary["training_seconds"] / 60,
            "total_minutes": summary["total_seconds"] / 60,
            **late,
        })
        history_frames.append(history.assign(
            subset_seed=run["subset_seed"], initialization=run["initialization"],
            strategy=run["strategy"],
        ))
    results = pd.DataFrame(rows).sort_values(["subset_seed", "initialization", "strategy"])
    effects = []
    for (seed, initialization), group in results.groupby(["subset_seed", "initialization"]):
        by_strategy = group.set_index("strategy")
        reference = by_strategy.loc["reference"]
        for variant in VARIANTS:
            if variant not in by_strategy.index:
                continue
            candidate = by_strategy.loc[variant]
            effects.append({
                "subset_seed": int(seed), "initialization": initialization, "variant": variant,
                "best_iou_change": candidate.best_validation_iou - reference.best_validation_iou,
                "endpoint_iou_change": candidate.iou_step_4000 - reference.iou_step_4000,
                "boundary_iou_change": candidate.mean_boundary_iou - reference.mean_boundary_iou,
                "dice_change": candidate.selected_validation_dice - reference.selected_validation_dice,
                "ap_change": candidate.selected_validation_ap - reference.selected_validation_ap,
                "best_step_change": int(candidate.best_step - reference.best_step),
                "total_minutes": candidate.total_minutes,
            })
    return (
        results,
        pd.DataFrame(effects),
        pd.concat(history_frames, ignore_index=True),
        boundary_metrics,
    )


def plot_boundary_histories(history: pd.DataFrame, metric: str) -> Figure:
    """Plot reference and both boundary-weighted validation trajectories."""
    labels = {"validation_mean_iou": "Validation mean per-image IoU", "validation_loss": "Validation L0"}
    if metric not in labels:
        raise ValueError("Unknown history metric.")
    seeds = sorted(history.subset_seed.unique())
    figure, axes = plt.subplots(
        2, len(seeds), figsize=(5.3 * len(seeds), 8.5),
        sharex=True, sharey="row", constrained_layout=True, squeeze=False,
    )
    for row, initialization in enumerate(INITIALIZATIONS):
        for column, seed in enumerate(seeds):
            axis = axes[row, column]
            for strategy in ("reference", *VARIANTS):
                values = history[
                    (history.initialization == initialization)
                    & (history.subset_seed == seed)
                    & (history.strategy == strategy)
                ].sort_values("step")
                if not values.empty:
                    axis.plot(values.step, values[metric], color=COLOURS[strategy], label=VARIANT_LABELS[strategy])
            axis.axvline(2000, color="#555555", linestyle="--", linewidth=1)
            axis.grid(alpha=0.2)
            axis.set_title(f"{INITIALIZATION_LABELS[initialization]}, subset seed {seed}")
            if row == 0 and column == 0:
                axis.legend(frameon=False)
            if row == 1:
                axis.set_xlabel("Optimizer step")
            if column == 0:
                axis.set_ylabel(labels[metric])
    figure.suptitle(f"n=25 boundary-weighted BCE: {labels[metric].lower()}")
    return figure


def choose_qualitative_examples(boundary_metrics: pd.DataFrame) -> pd.DataFrame:
    """Choose two improved and two worsened validation cases by Boundary IoU."""
    pivot = boundary_metrics.pivot_table(
        index=["subset_seed", "initialization", "id"], columns="strategy", values="boundary_iou"
    ).dropna(subset=["reference"])
    candidates = []
    for variant in VARIANTS:
        values = pivot[variant] - pivot["reference"]
        for index, value in values.items():
            candidates.append({
                "subset_seed": index[0], "initialization": index[1], "id": index[2],
                "variant": variant, "boundary_iou_change": value,
            })
    ordered = pd.DataFrame(candidates).sort_values("boundary_iou_change")
    worsened = ordered[ordered.boundary_iou_change < 0].drop_duplicates("id").head(2)
    improved = (
        ordered[ordered.boundary_iou_change > 0]
        .sort_values("boundary_iou_change", ascending=False)
        .drop_duplicates("id")
        .head(2)
    )
    if len(worsened) != 2 or len(improved) != 2:
        raise ValueError("Qualitative comparison requires two unique improvements and deteriorations.")
    chosen = pd.concat([worsened, improved])
    return chosen.sort_values("boundary_iou_change").reset_index(drop=True)


def plot_qualitative_examples(project_root: Path, selected: pd.DataFrame) -> Figure:
    """Plot identical validation images with reference and three prediction maps."""
    runs = _comparison_runs(project_root)
    manifest = load_json(project_root / "data/metadata/data_manifest.json")
    records = {sample["id"]: sample for sample in manifest["samples"]}
    store = RIDTensorStore(project_root, manifest)
    device = select_device()
    cmap = ListedColormap(["#111111", "#2DC653", "#FF8C1A", "#4361EE"])
    figure, axes = plt.subplots(len(selected), 5, figsize=(13, 2.7 * len(selected)), squeeze=False)
    for row, choice in enumerate(selected.itertuples(index=False)):
        matching = {
            run["strategy"]: run for run in runs
            if run["subset_seed"] == choice.subset_seed
            and run["initialization"] == choice.initialization
        }
        batch = store.batch([choice.id])
        target = batch["target"].numpy()[0, 0].astype(bool)
        predictions = {}
        for strategy, run in matching.items():
            model = _restore_model(run, device)
            with torch.inference_mode():
                predictions[strategy] = (
                    model(batch["image"].to(device)).sigmoid().cpu().numpy()[0, 0] >= 0.5
                )
            del model
        axes[row, 0].imshow(read_image(project_root / records[choice.id]["image"]))
        axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        selected_label = "A" if choice.variant == "area_proportional" else "B"
        axes[row, 0].set_ylabel(
            f"ID {choice.id} · seed {choice.subset_seed}\n"
            f"{INITIALIZATION_LABELS[choice.initialization]} · selected {selected_label}\n"
            f"Δ BIoU {choice.boundary_iou_change:+.3f}",
            rotation=0, ha="right", va="center", labelpad=14,
        )
        for column, strategy in enumerate(("reference", *VARIANTS), 2):
            prediction = predictions[strategy]
            error = np.zeros_like(target, dtype=np.uint8)
            error[prediction & target] = 1
            error[prediction & ~target] = 2
            error[~prediction & target] = 3
            regular = binary_metrics(prediction.astype(float), target)["iou"]
            boundary = boundary_iou(prediction, target, radius=BOUNDARY_RADIUS)
            axes[row, column].imshow(error, cmap=cmap, vmin=0, vmax=3)
            axes[row, column].set_title(f"{VARIANT_LABELS[strategy]}\nIoU {regular:.3f} · BIoU {boundary:.3f}")
    for axis, title in zip(axes[0, :2], ("RGB validation image", "Binary reference")):
        axis.set_title(title)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        "Metric-selected boundary improvements and deteriorations\n"
        "green true positive, orange false positive, blue false negative",
        y=1.01,
    )
    figure.tight_layout()
    return figure
