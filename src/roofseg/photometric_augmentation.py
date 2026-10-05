"""Configurations, diagnostics and analysis for mild n=25 colour augmentation."""

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
from .training_data import RIDTensorStore, apply_brightness_contrast

TRAINING_SIZE = 25
SUBSET_REPETITIONS = (1, 2)
PHOTOMETRIC_SEED = 1703
FACTOR_MIN = 0.85
FACTOR_MAX = 1.15
COLOURS = {False: "#777777", True: "#0072B2"}


def photometric_run_name(initialization: str, subset_seed: int) -> str:
    """Return the unique name for one mild colour-augmentation run."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("Unknown initialization.")
    return (
        f"n25_{initialization}_subsetseed{subset_seed}_brightness_contrast_"
        "lr1e-3_to_1e-4_step2001_steps4000"
    )


def photometric_run_root(project_root: Path, splits: dict) -> Path:
    """Return the split-versioned directory for the colour experiment."""
    return project_root / "runs/targeted" / splits["split_name"] / "photometric_n25"


def build_photometric_config(
    splits: dict, initialization: str, subset_repetition: int
) -> dict:
    """Add only the fixed mild colour rule and required run identity to a reference."""
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
        "run_name": photometric_run_name(initialization, subset_seed),
        "experiment": "stage4_photometric_augmentation_n25",
        "augmentation": (
            "D4 shared by image/target; then whole-RGB brightness and contrast "
            "with independent Uniform(0.85, 1.15) factors on every training occurrence"
        ),
        "photometric_augmentation": {
            "library": "torchvision.transforms.functional",
            "order": ["adjust_brightness", "adjust_contrast"],
            "scope": "whole RGB image",
            "input": "float32 RGB in [0, 1] before ImageNet normalization",
            "application_probability": 1.0,
            "factor_distribution": "independent uniform",
            "factor_min": FACTOR_MIN,
            "factor_max": FACTOR_MAX,
            "clipping": "torchvision tensor transforms clip to the valid image range",
            "evaluation": "disabled",
        },
        "seeds": {**config["seeds"], "photometric": PHOTOMETRIC_SEED},
    })
    return config


def photometric_run_directory(
    project_root: Path, splits: dict, initialization: str, subset_repetition: int
) -> Path:
    """Return the distinct directory for one mild colour run."""
    subset_seed = int(
        splits["training_subsets"]["repetitions"][str(subset_repetition)]["seed"]
    )
    return photometric_run_root(project_root, splits) / photometric_run_name(
        initialization, subset_seed
    )


def plot_photometric_examples(
    project_root: Path, manifest: dict, sample_ids: list[str]
) -> Figure:
    """Show two fixed raw images and boundary-factor transformations."""
    if len(sample_ids) != 2:
        raise ValueError("Exactly two sample IDs are required.")
    store = RIDTensorStore(project_root, manifest)
    variants = [
        ("Original", None),
        ("Brightness 0.85, contrast 0.85", (FACTOR_MIN, FACTOR_MIN)),
        ("Brightness 1.15, contrast 1.15", (FACTOR_MAX, FACTOR_MAX)),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(10, 7))
    for row, sample_id in enumerate(sample_ids):
        image, _ = store.load_raw(sample_id)
        for column, (label, factors) in enumerate(variants):
            shown = image if factors is None else apply_brightness_contrast(image, *factors)
            axes[row, column].imshow(shown.permute(1, 2, 0).numpy())
            axes[row, column].set_title(f"RID {sample_id}\n{label}")
            axes[row, column].axis("off")
    figure.suptitle("Fixed examples of the proposed whole-image colour augmentation")
    figure.tight_layout()
    return figure


def _load_comparison_runs(project_root: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load four references and four colour-augmentation runs without checkpoints."""
    splits = json.loads((project_root / "data/metadata/splits.json").read_text())
    runs, histories = [], {}
    for repetition in SUBSET_REPETITIONS:
        subset_seed = int(
            splits["training_subsets"]["repetitions"][str(repetition)]["seed"]
        )
        for initialization in INITIALIZATIONS:
            for augmented in (False, True):
                directory = (
                    photometric_run_directory(project_root, splits, initialization, repetition)
                    if augmented else late_lr_drop_run_directory(
                        project_root, splits, initialization, repetition, TRAINING_SIZE
                    )
                )
                config = json.loads((directory / "config.json").read_text())
                metadata = json.loads((directory / "metadata.json").read_text())
                summary = json.loads((directory / "summary.json").read_text())
                if summary.get("status") != "completed":
                    raise ValueError(f"Photometric comparison run is incomplete: {directory.name}")
                key = f"seed{subset_seed}_{initialization}_{'colour' if augmented else 'reference'}"
                runs.append({
                    "key": key, "subset_repetition": repetition,
                    "subset_seed": subset_seed, "initialization": initialization,
                    "photometric_augmentation": augmented, "directory": directory,
                    "config": config, "metadata": metadata, "summary": summary,
                })
                with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
                    histories[key] = list(csv.DictReader(handle))
    return runs, histories


def collect_photometric_results(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect eight-run summaries, paired colour effects and histories."""
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
            "photometric_augmentation": run["photometric_augmentation"],
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
        ["subset_seed", "initialization", "photometric_augmentation"]
    )
    effects = []
    for (repetition, seed, initialization), group in results.groupby(
        ["subset_repetition", "subset_seed", "initialization"]
    ):
        by_rule = group.set_index("photometric_augmentation")
        augmented, reference = by_rule.loc[True], by_rule.loc[False]
        effects.append({
            "subset_repetition": int(repetition), "subset_seed": int(seed),
            "initialization": initialization,
            "best_iou_change": augmented.best_validation_iou - reference.best_validation_iou,
            "endpoint_iou_change": augmented.iou_step_4000 - reference.iou_step_4000,
            "endpoint_validation_loss_change": (
                augmented.validation_loss_step_4000 - reference.validation_loss_step_4000
            ),
            "selected_iou_gap_change": (
                augmented.training_minus_validation_iou
                - reference.training_minus_validation_iou
            ),
            "selected_dice_change": (
                augmented.selected_validation_dice - reference.selected_validation_dice
            ),
            "selected_ap_change": (
                augmented.selected_validation_ap - reference.selected_validation_ap
            ),
        })
    return results, pd.DataFrame(effects), pd.concat(history_frames, ignore_index=True)


def plot_photometric_histories(history: pd.DataFrame, metric: str) -> Figure:
    """Plot one validation metric for paired colour rules by subset and initialization."""
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
            for augmented in (False, True):
                values = history[
                    (history.initialization == initialization)
                    & (history.subset_seed == subset_seed)
                    & (history.photometric_augmentation == augmented)
                ].sort_values("step")
                axis.plot(
                    values.step, values[metric], color=COLOURS[augmented],
                    label="brightness + contrast" if augmented else "D4 reference",
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
    figure.suptitle(f"n=25 mild colour augmentation: {labels[metric].lower()}")
    figure.tight_layout()
    return figure
