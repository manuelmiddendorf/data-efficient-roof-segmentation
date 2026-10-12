"""Configuration and saved-result access for the paired WHU training pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .pilot import INITIALIZATIONS, SEEDS
from .training_data import IMAGENET_MEAN, IMAGENET_STD


TRAINING_SIZE = 100


def whu_run_name(initialization: str) -> str:
    """Return the explicit run name for one WHU initialization."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("Initialization must be random or imagenet.")
    return f"n100_{initialization}_subsetseed17_fixed_drop"


def whu_run_root(project_root: Path, splits: dict) -> Path:
    """Return the provider-split-versioned WHU pilot directory."""
    return project_root / "runs/external_replication_pilot" / splits["split_name"]


def build_whu_pilot_config(splits: dict, initialization: str) -> dict:
    """Resolve the fixed 4,000-update recipe and saved WHU n=100 IDs."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("Initialization must be random or imagenet.")
    subset = splits["future_pilot_subset"]
    if subset["size"] != TRAINING_SIZE or subset["seed"] != 17:
        raise ValueError("The saved WHU pilot subset is not the required seed-17 n=100 set.")
    return {
        "schema_version": 1,
        "run_name": whu_run_name(initialization),
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
            "training_subset": "future_pilot_subset.ids",
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
        "seeds": dict(SEEDS),
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


def whu_run_directory(project_root: Path, splits: dict, initialization: str) -> Path:
    """Return one WHU pilot run directory."""
    return whu_run_root(project_root, splits) / whu_run_name(initialization)


def load_whu_pilot_runs(project_root: Path) -> list[dict]:
    """Load both completed WHU pilot runs without importing checkpoints."""
    splits = json.loads((project_root / "data/metadata/whu/splits.json").read_text())
    runs = []
    for initialization in INITIALIZATIONS:
        directory = whu_run_directory(project_root, splits, initialization)
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
            "initialization": initialization, "directory": directory,
            "config": config, "metadata": metadata, "summary": summary,
            "history": history, "metrics": metrics,
        })
    return runs
