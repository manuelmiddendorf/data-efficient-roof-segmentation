"""Configuration and saved-result access for the paired WHU training pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .pilot import INITIALIZATIONS, SEEDS
from .run_artifacts import json_digest, selected_input_digest, verify_completed_run
from .training_data import IMAGENET_MEAN, IMAGENET_STD, paired_schedule


TRAINING_SIZE = 100
SUBSET_REPETITIONS = (1, 2, 3)
SEED17_IDS_SHA256 = "bb0d64cae5bf4043ad64c09dbd1343936289ee89773ae63b45496ca96e74ad2e"


def whu_subset_record(splits: dict, subset_repetition: int) -> dict:
    """Return one saved WHU subset while preserving the historical Seed-17 path."""
    if subset_repetition == 1:
        return splits["future_pilot_subset"]
    try:
        return splits["additional_pilot_subsets"]["repetitions"][str(subset_repetition)]
    except KeyError as error:
        raise ValueError(f"Unknown WHU subset repetition: {subset_repetition}") from error


def whu_run_name(initialization: str, subset_seed: int = 17) -> str:
    """Return the explicit run name for one WHU initialization."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("Initialization must be random or imagenet.")
    return f"n100_{initialization}_subsetseed{subset_seed}_fixed_drop"


def whu_run_root(project_root: Path, splits: dict) -> Path:
    """Return the provider-split-versioned WHU pilot directory."""
    return project_root / "runs/external_replication_pilot" / splits["split_name"]


def build_whu_pilot_config(
    splits: dict, initialization: str, subset_repetition: int = 1
) -> dict:
    """Resolve the fixed 4,000-update recipe and saved WHU n=100 IDs."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("Initialization must be random or imagenet.")
    subset = whu_subset_record(splits, subset_repetition)
    if subset["size"] != TRAINING_SIZE:
        raise ValueError("The saved WHU pilot subset is not an n=100 set.")
    subset_seed = int(subset["seed"])
    return {
        "schema_version": 1,
        "run_name": whu_run_name(initialization, subset_seed),
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
                "future_pilot_subset.ids" if subset_repetition == 1 else
                f"additional_pilot_subsets.repetitions.{subset_repetition}.ids"
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
    project_root: Path, splits: dict, initialization: str, subset_repetition: int = 1
) -> Path:
    """Return one WHU pilot run directory."""
    seed = int(whu_subset_record(splits, subset_repetition)["seed"])
    return whu_run_root(project_root, splits) / whu_run_name(initialization, seed)


def verify_historical_seed17_run(
    project_root: Path, manifest: dict, splits: dict, initialization: str
) -> dict:
    """Verify the old pilot by its actual config, inputs and checkpoint.

    The central split JSON was additively extended after this run. Its historical
    whole-file split hash is therefore retained rather than recomputed.
    """
    directory = whu_run_directory(project_root, splits, initialization, 1)
    expected = build_whu_pilot_config(splits, initialization, 1)
    saved = json.loads((directory / "config.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    if saved != expected:
        raise ValueError(f"Historical Seed-17 configuration changed: {directory.name}")
    if json_digest(saved) != metadata["compatibility_identity"]["config_sha256"]:
        raise ValueError(f"Historical Seed-17 config identity changed: {directory.name}")
    if json_digest(saved["training_ids"]) != SEED17_IDS_SHA256:
        raise ValueError("Historical Seed-17 training IDs changed.")
    if saved["validation_ids"] != splits["roles"]["val"]:
        raise ValueError("Historical WHU validation role changed.")
    identity = metadata["compatibility_identity"]
    if selected_input_digest(manifest, saved["training_ids"]) != identity["training_inputs_sha256"]:
        raise ValueError("Historical Seed-17 training inputs changed.")
    if selected_input_digest(manifest, saved["validation_ids"]) != identity["validation_inputs_sha256"]:
        raise ValueError("Historical WHU validation inputs changed.")
    _, schedule = paired_schedule(
        saved["training_ids"], saved["max_steps"], saved["batch_size"],
        saved["seeds"]["data_order"], saved["seeds"]["augmentation"],
    )
    if schedule["sha256"] != identity["schedule_sha256"]:
        raise ValueError("Historical Seed-17 schedule changed.")
    return verify_completed_run(directory)


def load_whu_pilot_runs(project_root: Path) -> list[dict]:
    """Load all six completed WHU subset-replication runs."""
    splits = json.loads((project_root / "data/metadata/whu/splits.json").read_text())
    runs = []
    for subset_repetition in SUBSET_REPETITIONS:
        subset = whu_subset_record(splits, subset_repetition)
        for initialization in INITIALIZATIONS:
            directory = whu_run_directory(project_root, splits, initialization, subset_repetition)
            config = json.loads((directory / "config.json").read_text())
            metadata = json.loads((directory / "metadata.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            if summary.get("status") != "completed":
                raise ValueError(f"WHU pilot run is incomplete: {directory.name}")
            with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
            with (directory / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                metrics = list(csv.DictReader(handle))
            runs.append({
                "subset_repetition": subset_repetition, "subset_seed": int(subset["seed"]),
                "initialization": initialization, "directory": directory,
                "config": config, "metadata": metadata, "summary": summary,
                "history": history, "metrics": metrics,
            })
    return runs
