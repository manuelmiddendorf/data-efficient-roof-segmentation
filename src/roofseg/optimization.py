"""Resolved configurations and result loading for the Stage 3 learning-rate block."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

from .pilot import INITIALIZATIONS, build_config, pilot_run_root, run_name


TRAINING_SIZE = 100
LEARNING_RATES = (1e-4, 3e-4, 1e-3)
NEW_LEARNING_RATES = (1e-4, 1e-3)
REFERENCE_LEARNING_RATE = 3e-4


def learning_rate_label(learning_rate: float) -> str:
    """Return the stable concise label for an approved learning rate."""
    if learning_rate not in LEARNING_RATES:
        raise ValueError("Learning rate is outside the fixed Stage 3 block.")
    return f"{learning_rate:.0e}".replace("e-0", "e-")


def learning_rate_run_name(initialization: str, learning_rate: float) -> str:
    """Return the readable directory name for one new learning-rate run."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("Unknown initialization.")
    return f"n{TRAINING_SIZE}_{initialization}_lr{learning_rate_label(learning_rate)}"


def optimization_run_root(project_root: Path, splits: dict) -> Path:
    """Return the split-versioned directory for new Stage 3 runs."""
    return project_root / "runs/optimization" / splits["split_name"] / "learning_rate_n100"


def build_learning_rate_config(splits: dict, initialization: str, learning_rate: float) -> dict:
    """Resolve one approved configuration while changing only the learning rate.

    The 3e-4 configuration is returned byte-for-byte equivalent to its Stage 2
    reference configuration. New candidates receive a distinct experiment and
    run identity while preserving every other setting.
    """
    base = build_config(splits, TRAINING_SIZE, initialization)
    if learning_rate == REFERENCE_LEARNING_RATE:
        return base
    if learning_rate not in NEW_LEARNING_RATES:
        raise ValueError("Learning rate is outside the fixed Stage 3 block.")
    config = deepcopy(base)
    config.update({
        "run_name": learning_rate_run_name(initialization, learning_rate),
        "experiment": "stage3_learning_rate_n100",
        "learning_rate": learning_rate,
    })
    return config


def learning_rate_run_directory(
    project_root: Path,
    splits: dict,
    initialization: str,
    learning_rate: float,
) -> Path:
    """Return the original reference or new optimization run directory."""
    if learning_rate == REFERENCE_LEARNING_RATE:
        return pilot_run_root(project_root, splits) / run_name(TRAINING_SIZE, initialization)
    return optimization_run_root(project_root, splits) / learning_rate_run_name(
        initialization, learning_rate
    )


def load_learning_rate_results(project_root: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load the six summaries and histories without importing checkpoints.

    Parameters
    ----------
    project_root:
        Repository root containing the active split and completed run artifacts.

    Returns
    -------
    runs, histories:
        Run records and evaluation histories for both initializations at all
        three approved learning rates.
    """
    splits = json.loads((project_root / "data/metadata/splits.json").read_text())
    runs, histories = [], {}
    for initialization in INITIALIZATIONS:
        for learning_rate in LEARNING_RATES:
            directory = learning_rate_run_directory(
                project_root, splits, initialization, learning_rate
            )
            config = json.loads((directory / "config.json").read_text())
            metadata = json.loads((directory / "metadata.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            if summary.get("status") != "completed":
                raise ValueError(f"Learning-rate run is incomplete: {directory.name}")
            key = f"{initialization}_{learning_rate_label(learning_rate)}"
            runs.append({
                "key": key,
                "initialization": initialization,
                "learning_rate": learning_rate,
                "directory": directory,
                "config": config,
                "metadata": metadata,
                "summary": summary,
                "is_reference": learning_rate == REFERENCE_LEARNING_RATE,
            })
            with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
                histories[key] = list(csv.DictReader(handle))
    return runs, histories
