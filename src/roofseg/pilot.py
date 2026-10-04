"""Resolved configurations and result loading for the fixed Stage 2 pilot."""

from __future__ import annotations

import csv
from importlib.metadata import version
import json
import platform
from pathlib import Path
import subprocess

import torch

from .integrity import sha256_file
from .run_artifacts import json_digest, selected_input_digest
from .training_data import IMAGENET_MEAN, IMAGENET_STD, paired_schedule


PILOT_SIZES = (25, 100, 500)
INITIALIZATIONS = ("random", "imagenet")
SEEDS = {"subset": 17, "model": 20261004, "data_order": 1701, "augmentation": 1702}


def run_name(training_size: int, initialization: str) -> str:
    """Return the stable readable directory name for one pilot run."""
    return f"n{training_size}_{initialization}"


def pilot_run_root(project_root: Path, splits: dict) -> Path:
    """Return the split-versioned directory for the active six-run pilot."""
    return project_root / "runs/training_pilot" / splits["split_name"]


def build_config(splits: dict, training_size: int, initialization: str) -> dict:
    """Resolve the fixed pilot recipe and exact repetition-1 training IDs."""
    if training_size not in PILOT_SIZES or initialization not in INITIALIZATIONS:
        raise ValueError("Requested run is outside the fixed Stage 2 pilot.")
    training_ids = splits["training_subsets"]["repetitions"]["1"]["subsets"][str(training_size)]
    return {
        "schema_version": 1,
        "run_name": run_name(training_size, initialization),
        "experiment": "stage2_initialization_pilot",
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
            "manifest": "data/metadata/data_manifest.json",
            "splits": "data/metadata/splits.json",
            "split_name": splits["split_name"],
            "training_subset": f"training_subsets.repetitions.1.subsets.{training_size}",
            "validation_role": "roles.validation",
            "test_used": False,
            "image_shape": [3, 512, 512],
            "target_mapping": "roof = mask != 17",
            "normalization_mean": list(IMAGENET_MEAN),
            "normalization_std": list(IMAGENET_STD),
        },
        "training_ids": training_ids,
        "validation_ids": splits["roles"]["validation"],
        "seeds": SEEDS,
        "augmentation": "D4: shared horizontal reflection and rotations by multiples of 90 degrees",
        "batch_size": 4,
        "precision": "float32",
        "loss": {
            "formula": "0.5 * mean-per-image BCEWithLogits + 0.5 * mean-per-image soft Dice loss",
            "dice_epsilon": 1e-6,
        },
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "weight_decay_exclusion": "biases and one-dimensional normalization parameters",
        "max_steps": 2000,
        "evaluation_steps": [0] + list(range(100, 2001, 100)),
        "evaluation_interval": 100,
        "probability_threshold": 0.5,
        "early_stopping": False,
        "scheduler": None,
        "batch_normalization": "all affine parameters trainable; running statistics update in train mode and are used in eval mode",
        "incomplete_batch_policy": "retain the last incomplete batch of every shuffled data passage",
    }


def build_metadata(project_root: Path, manifest: dict, splits: dict, config: dict, device: torch.device) -> dict:
    """Record source, environment and exact identities required for reuse."""
    _, schedule = paired_schedule(
        config["training_ids"], config["max_steps"], config["batch_size"],
        config["seeds"]["data_order"], config["seeds"]["augmentation"],
    )
    config_sha = json_digest(config)
    inputs = {
        "manifest_sha256": sha256_file(project_root / config["data"]["manifest"]),
        "splits_sha256": sha256_file(project_root / config["data"]["splits"]),
        "training_inputs_sha256": selected_input_digest(manifest, config["training_ids"]),
        "validation_inputs_sha256": selected_input_digest(manifest, config["validation_ids"]),
        "schedule_sha256": schedule["sha256"],
    }
    inputs["input_sha256"] = json_digest(inputs)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema_version": 1,
        "git_commit": git_commit,
        "device": str(device),
        "mps_fallback_enabled": False,
        "platform": {"python": platform.python_version(), "machine": platform.machine(),
                     "system": platform.system()},
        "packages": {name: version(name) for name in ("torch", "torchvision", "numpy", "pillow")},
        "compatibility_identity": {"config_sha256": config_sha, **inputs},
        "raw_reference": {
            "kind": "provider SHA-256 list",
            "path": manifest["dataset"]["provider_checksum_file"],
            "sha256": manifest["dataset"]["provider_checksum_file_sha256"],
            "verified_once_before_block": True,
        },
        "declared_weight_source": (
            {"kind": "torchvision trusted weight enum", "enum": "EfficientNet_B0_Weights.IMAGENET1K_V1",
             "url": "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
             "trusted_sha256_prefix": "7f5810bc"}
            if config["pretrained_encoder"] else {"kind": "fresh random initialization"}
        ),
        "schedule": schedule,
        "reproducibility_limit": "Seeds and schedules are paired; MPS arithmetic is not guaranteed bitwise reproducible.",
    }


def load_pilot_results(project_root: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load summaries and histories without importing checkpoints."""
    splits = json.loads((project_root / "data/metadata/splits.json").read_text())
    run_root = pilot_run_root(project_root, splits)
    summaries, histories = [], {}
    for size in PILOT_SIZES:
        for initialization in INITIALIZATIONS:
            name = run_name(size, initialization)
            directory = run_root / name
            config = json.loads((directory / "config.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            metadata = json.loads((directory / "metadata.json").read_text())
            if summary.get("status") != "completed":
                raise ValueError(f"Pilot run is incomplete: {name}")
            summaries.append({"name": name, "training_size": size,
                              "initialization": initialization, "config": config,
                              "summary": summary, "metadata": metadata})
            with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
                histories[name] = list(csv.DictReader(handle))
    return summaries, histories
