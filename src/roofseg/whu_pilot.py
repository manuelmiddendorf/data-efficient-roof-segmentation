"""Configuration and saved-result access for the paired WHU training pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .pilot import INITIALIZATIONS, SEEDS
from .run_artifacts import json_digest, selected_input_digest, verify_completed_run
from .training_data import IMAGENET_MEAN, IMAGENET_STD, paired_schedule


TRAINING_SIZE = 100
LEARNING_CURVE_SIZES = (25, 100, 500)
LEARNING_CURVE_REPETITIONS = (1, 2, 3)
SUBSET_REPETITIONS = (1, 2, 3)
SEED17_IDS_SHA256 = "bb0d64cae5bf4043ad64c09dbd1343936289ee89773ae63b45496ca96e74ad2e"
SEED29_IDS_SHA256 = "da02469128c85264a8251e353c6cae42d0513a0b7040b4023269b267b5c569fc"
SEED43_IDS_SHA256 = "5288f71232d6607edcf169ead37639c342dd224d0bcf72ebb646c96d4f335bbf"
HISTORICAL_N100_IDS_SHA256 = {
    1: SEED17_IDS_SHA256,
    2: SEED29_IDS_SHA256,
    3: SEED43_IDS_SHA256,
}


def whu_subset_record(
    splits: dict, subset_repetition: int, training_size: int = TRAINING_SIZE
) -> dict:
    """Return a saved WHU subset while preserving historical n=100 paths."""
    if training_size == TRAINING_SIZE:
        if subset_repetition == 1:
            return splits["future_pilot_subset"]
        try:
            return splits["additional_pilot_subsets"]["repetitions"][str(subset_repetition)]
        except KeyError as error:
            raise ValueError(f"Unknown WHU subset repetition: {subset_repetition}") from error
    if training_size not in LEARNING_CURVE_SIZES or subset_repetition not in LEARNING_CURVE_REPETITIONS:
        raise ValueError(
            f"Unknown WHU learning-curve subset: n={training_size}, repetition={subset_repetition}"
        )
    repetition = splits["learning_curve_subsets"]["repetitions"][str(subset_repetition)]
    return {
        "size": training_size,
        "seed": repetition["subset_seed"],
        "ids": repetition["subsets"][str(training_size)],
    }


def whu_run_name(
    initialization: str, subset_seed: int = 17, training_size: int = TRAINING_SIZE
) -> str:
    """Return the explicit run name for one WHU initialization."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("Initialization must be random or imagenet.")
    return f"n{training_size}_{initialization}_subsetseed{subset_seed}_fixed_drop"


def whu_run_root(project_root: Path, splits: dict) -> Path:
    """Return the provider-split-versioned WHU pilot directory."""
    return project_root / "runs/external_replication_pilot" / splits["split_name"]


def build_whu_pilot_config(
    splits: dict, initialization: str, subset_repetition: int = 1,
    training_size: int = TRAINING_SIZE,
) -> dict:
    """Resolve the fixed 4,000-update recipe and one saved WHU subset."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("Initialization must be random or imagenet.")
    subset = whu_subset_record(splits, subset_repetition, training_size)
    if subset["size"] != training_size:
        raise ValueError(f"The saved WHU subset is not an n={training_size} set.")
    subset_seed = int(subset["seed"])
    return {
        "schema_version": 1,
        "run_name": whu_run_name(initialization, subset_seed, training_size),
        "experiment": "whu_external_replication_pilot",
        "architecture": {
            "name": "U-Net",
            "encoder": "torchvision EfficientNet-B0",
            "encoder_stages": "all feature stages; 1280-channel bottleneck",
            "decoder_channels": [256, 128, 64, 32, 16],
            "decoder_upsampling": "nearest",
            "output": "N×1×512×512 float32 logits",
        },
        "initialization": initialization,
        "pretrained_encoder": initialization == "imagenet",
        "decoder_initialization": "fresh and paired by model seed",
        "data": {
            "dataset": "whu_aerial",
            "manifest": "data/metadata/whu/data_manifest.json",
            "splits": "data/metadata/whu/splits.json",
            "split_name": splits["split_name"],
            "training_subset": (
                ("future_pilot_subset.ids" if subset_repetition == 1 else
                 f"additional_pilot_subsets.repetitions.{subset_repetition}.ids")
                if training_size == TRAINING_SIZE else
                f"learning_curve_subsets.repetitions.{subset_repetition}.subsets.{training_size}"
            ),
            "validation_role": "roles.val",
            "test_used": False,
            "image_shape": [3, 512, 512],
            "target_mapping": "building footprint = mask == 0; background = mask == 1",
            "invalid_code": None,
            "normalization_mean": list(IMAGENET_MEAN),
            "normalization_std": list(IMAGENET_STD),
            "spatial_basis": "provider-documented separate areas; tile footprints not independently reconstructable",
        },
        "training_ids": list(subset["ids"]),
        "validation_ids": list(splits["roles"]["val"]),
        "seeds": {**SEEDS, "subset": subset_seed},
        "augmentation": "D4: shared horizontal reflection and rotations by multiples of 90 degrees",
        "batch_size": 4,
        "precision": "float32",
        "loss": {
            "formula": "0.5 * mean-per-image BCEWithLogits + 0.5 * mean-per-image soft Dice loss",
            "dice_epsilon": 1e-6,
        },
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "scheduler": {
            "type": "piecewise_constant",
            "adaptive": False,
            "phases": [
                {"start_step": 1, "end_step": 2000, "learning_rate": 1e-3},
                {"start_step": 2001, "end_step": 4000, "learning_rate": 1e-4},
            ],
        },
        "weight_decay": 1e-4,
        "weight_decay_exclusion": "biases and one-dimensional normalization parameters",
        "max_steps": 4000,
        "evaluation_steps": [0] + list(range(100, 4001, 100)),
        "evaluation_interval": 100,
        "probability_threshold": 0.5,
        "early_stopping": False,
        "batch_normalization": "all affine parameters trainable; running statistics update in train mode and are used in eval mode",
        "incomplete_batch_policy": "retain the last incomplete batch of every shuffled data passage",
        "reference_group_diagnostics": True,
    }


def whu_run_directory(
    project_root: Path, splits: dict, initialization: str, subset_repetition: int = 1,
    training_size: int = TRAINING_SIZE,
) -> Path:
    """Return one WHU pilot run directory."""
    seed = int(whu_subset_record(splits, subset_repetition, training_size)["seed"])
    return whu_run_root(project_root, splits) / whu_run_name(
        initialization, seed, training_size
    )


def verify_historical_whu_run(
    project_root: Path, manifest: dict, splits: dict, initialization: str,
    subset_repetition: int, training_size: int = TRAINING_SIZE,
) -> dict:
    """Verify a saved run by its actual config, selected inputs and checkpoint.

    The central split JSON may be additively extended after a run. Its historical
    whole-file split hash is therefore retained rather than recomputed.
    """
    directory = whu_run_directory(
        project_root, splits, initialization, subset_repetition, training_size
    )
    expected = build_whu_pilot_config(
        splits, initialization, subset_repetition, training_size
    )
    saved = json.loads((directory / "config.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    if saved != expected:
        raise ValueError(f"Historical WHU configuration changed: {directory.name}")
    if json_digest(saved) != metadata["compatibility_identity"]["config_sha256"]:
        raise ValueError(f"Historical WHU config identity changed: {directory.name}")
    current_ids = whu_subset_record(splits, subset_repetition, training_size)["ids"]
    if saved["training_ids"] != current_ids:
        raise ValueError(f"Historical WHU training IDs changed: {directory.name}")
    if (
        training_size == TRAINING_SIZE
        and json_digest(saved["training_ids"])
        != HISTORICAL_N100_IDS_SHA256[subset_repetition]
    ):
        raise ValueError(
            f"Historical Seed-{saved['seeds']['subset']} training IDs changed."
        )
    if saved["validation_ids"] != splits["roles"]["val"]:
        raise ValueError("Historical WHU validation role changed.")
    identity = metadata["compatibility_identity"]
    if selected_input_digest(manifest, saved["training_ids"]) != identity["training_inputs_sha256"]:
        raise ValueError("Historical WHU training inputs changed.")
    if selected_input_digest(manifest, saved["validation_ids"]) != identity["validation_inputs_sha256"]:
        raise ValueError("Historical WHU validation inputs changed.")
    _, schedule = paired_schedule(
        saved["training_ids"], saved["max_steps"], saved["batch_size"],
        saved["seeds"]["data_order"], saved["seeds"]["augmentation"],
    )
    if schedule["sha256"] != identity["schedule_sha256"]:
        raise ValueError("Historical WHU schedule changed.")
    return verify_completed_run(directory)


def verify_historical_seed17_run(
    project_root: Path, manifest: dict, splits: dict, initialization: str
) -> dict:
    """Retain the original public verifier for the first WHU pilot pair."""
    return verify_historical_whu_run(
        project_root, manifest, splits, initialization, subset_repetition=1
    )


def _load_whu_runs(
    project_root: Path, combinations: list[tuple[int, int]]
) -> list[dict]:
    """Load completed WHU runs for ``(repetition, training size)`` combinations."""
    splits = json.loads((project_root / "data/metadata/whu/splits.json").read_text())
    runs = []
    for subset_repetition, training_size in combinations:
        subset = whu_subset_record(splits, subset_repetition, training_size)
        for initialization in INITIALIZATIONS:
            directory = whu_run_directory(
                project_root, splits, initialization, subset_repetition, training_size
            )
            config = json.loads((directory / "config.json").read_text())
            metadata = json.loads((directory / "metadata.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            if summary.get("status") != "completed":
                raise ValueError(f"WHU run is incomplete: {directory.name}")
            with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
            with (directory / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                metrics = list(csv.DictReader(handle))
            runs.append({
                "subset_repetition": subset_repetition,
                "subset_seed": int(subset["seed"]),
                "training_size": training_size,
                "initialization": initialization,
                "directory": directory,
                "config": config,
                "metadata": metadata,
                "summary": summary,
                "history": history,
                "metrics": metrics,
            })
    return runs


def load_whu_pilot_runs(project_root: Path) -> list[dict]:
    """Load all six completed WHU subset-replication runs."""
    return _load_whu_runs(
        project_root,
        [(repetition, TRAINING_SIZE) for repetition in SUBSET_REPETITIONS],
    )


def load_whu_learning_curve_runs(project_root: Path) -> list[dict]:
    """Load the eighteen completed n=25/100/500 WHU learning-curve runs."""
    return _load_whu_runs(
        project_root,
        [
            (repetition, training_size)
            for repetition in LEARNING_CURVE_REPETITIONS
            for training_size in LEARNING_CURVE_SIZES
        ],
    )
