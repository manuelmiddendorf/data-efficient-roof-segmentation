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
HORIZON_LEARNING_RATES = (1e-4, 1e-3)
LONG_HORIZON_STEPS = 4000
LATE_DROP_INITIAL_RATE = 1e-3
LATE_DROP_FINAL_RATE = 1e-4
LATE_DROP_FIRST_FINAL_STEP = 2001


def learning_rate_label(learning_rate: float) -> str:
    """Return the stable concise label for an approved learning rate.

    Parameters
    ----------
    learning_rate:
        Candidate learning rate from the fixed Stage 3 set.

    Returns
    -------
    str
        Scientific-notation label such as ``1e-4``.
    """
    if learning_rate not in LEARNING_RATES:
        raise ValueError("Learning rate is outside the fixed Stage 3 block.")
    return f"{learning_rate:.0e}".replace("e-0", "e-")


def learning_rate_run_name(initialization: str, learning_rate: float) -> str:
    """Return the readable directory name for one new learning-rate run.

    Parameters
    ----------
    initialization:
        ``random`` or ``imagenet``.
    learning_rate:
        Candidate learning rate from the fixed Stage 3 set.

    Returns
    -------
    str
        Run name containing training size, initialization and learning rate.
    """
    if initialization not in INITIALIZATIONS:
        raise ValueError("Unknown initialization.")
    return f"n{TRAINING_SIZE}_{initialization}_lr{learning_rate_label(learning_rate)}"


def optimization_run_root(project_root: Path, splits: dict) -> Path:
    """Return the split-versioned directory for new Stage 3 runs.

    Parameters
    ----------
    project_root:
        Repository root.
    splits:
        Active split record containing ``split_name``.

    Returns
    -------
    pathlib.Path
        Directory reserved for the n=100 learning-rate block.
    """
    return project_root / "runs/optimization" / splits["split_name"] / "learning_rate_n100"


def build_learning_rate_config(splits: dict, initialization: str, learning_rate: float) -> dict:
    """Resolve one approved configuration while changing only the learning rate.

    Parameters
    ----------
    splits:
        Active split record with the saved repetition-1 n=100 subset.
    initialization:
        ``random`` or ``imagenet``.
    learning_rate:
        One of the three approved rates.

    Returns
    -------
    dict
        Fully resolved run configuration. The 3e-4 configuration is returned
        byte-for-byte equivalent to its Stage 2 reference configuration. New
        candidates receive a distinct experiment and run identity while
        preserving every other setting.
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
    """Return the original reference or new optimization run directory.

    Parameters
    ----------
    project_root:
        Repository root.
    splits:
        Active split record.
    initialization:
        ``random`` or ``imagenet``.
    learning_rate:
        One of the approved rates.

    Returns
    -------
    pathlib.Path
        Original Stage 2 directory for 3e-4 or a distinct Stage 3 directory.
    """
    if learning_rate == REFERENCE_LEARNING_RATE:
        return pilot_run_root(project_root, splits) / run_name(TRAINING_SIZE, initialization)
    return optimization_run_root(project_root, splits) / learning_rate_run_name(
        initialization, learning_rate
    )


def training_horizon_run_name(initialization: str, learning_rate: float) -> str:
    """Return the distinct name for one approved 4,000-step run."""
    if initialization not in INITIALIZATIONS or learning_rate not in HORIZON_LEARNING_RATES:
        raise ValueError("Run is outside the fixed training-horizon block.")
    return (
        f"n{TRAINING_SIZE}_{initialization}_lr{learning_rate_label(learning_rate)}"
        f"_steps{LONG_HORIZON_STEPS}"
    )


def training_horizon_run_root(project_root: Path, splits: dict) -> Path:
    """Return the split-versioned directory for the 4,000-step comparison."""
    return project_root / "runs/optimization" / splits["split_name"] / "training_horizon_n100"


def build_training_horizon_config(
    splits: dict, initialization: str, learning_rate: float
) -> dict:
    """Extend one matched 2,000-step recipe to the approved 4,000-step horizon.

    Parameters
    ----------
    splits:
        Active split record with the saved repetition-1 n=100 subset.
    initialization:
        ``random`` or ``imagenet``.
    learning_rate:
        ``1e-4`` or ``1e-3``.

    Returns
    -------
    dict
        A resolved fresh-run configuration. Apart from run identity, only the
        maximum step and corresponding evaluation-step list differ from the
        matched 2,000-step configuration.
    """
    if learning_rate not in HORIZON_LEARNING_RATES:
        raise ValueError("Learning rate is outside the fixed training-horizon block.")
    config = deepcopy(build_learning_rate_config(splits, initialization, learning_rate))
    config.update({
        "run_name": training_horizon_run_name(initialization, learning_rate),
        "experiment": "stage3_training_horizon_n100",
        "max_steps": LONG_HORIZON_STEPS,
        "evaluation_steps": [0] + list(range(100, LONG_HORIZON_STEPS + 1, 100)),
    })
    return config


def training_horizon_run_directory(
    project_root: Path, splits: dict, initialization: str, learning_rate: float
) -> Path:
    """Return the directory for one approved 4,000-step run."""
    return training_horizon_run_root(project_root, splits) / training_horizon_run_name(
        initialization, learning_rate
    )


def late_lr_drop_run_name(initialization: str) -> str:
    """Return the distinct name for one approved fixed-drop run."""
    if initialization not in INITIALIZATIONS:
        raise ValueError("Unknown initialization.")
    return (
        f"n{TRAINING_SIZE}_{initialization}_lr1e-3_to_1e-4_"
        f"step{LATE_DROP_FIRST_FINAL_STEP}_steps{LONG_HORIZON_STEPS}"
    )


def late_lr_drop_run_root(project_root: Path, splits: dict) -> Path:
    """Return the split-versioned directory for the fixed-drop comparison."""
    return project_root / "runs/optimization" / splits["split_name"] / "late_lr_drop_n100"


def build_late_lr_drop_config(splits: dict, initialization: str) -> dict:
    """Resolve one fresh 4,000-step run with a fixed drop after step 2,000.

    The first 2,000 updates use ``1e-3`` and updates 2,001--4,000 use
    ``1e-4``. Every non-schedule training setting comes from the matched
    constant-``1e-3`` training-horizon configuration.
    """
    config = deepcopy(build_training_horizon_config(
        splits, initialization, LATE_DROP_INITIAL_RATE
    ))
    config.update({
        "run_name": late_lr_drop_run_name(initialization),
        "experiment": "stage3_late_lr_drop_n100",
        "scheduler": {
            "type": "piecewise_constant",
            "adaptive": False,
            "phases": [
                {"start_step": 1, "end_step": 2000,
                 "learning_rate": LATE_DROP_INITIAL_RATE},
                {"start_step": LATE_DROP_FIRST_FINAL_STEP,
                 "end_step": LONG_HORIZON_STEPS,
                 "learning_rate": LATE_DROP_FINAL_RATE},
            ],
        },
    })
    return config


def late_lr_drop_run_directory(
    project_root: Path, splits: dict, initialization: str
) -> Path:
    """Return the directory for one approved fixed-drop run."""
    return late_lr_drop_run_root(project_root, splits) / late_lr_drop_run_name(initialization)


def load_late_lr_drop_results(
    project_root: Path,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load the two completed fixed-drop summaries and evaluation histories."""
    splits = json.loads((project_root / "data/metadata/splits.json").read_text())
    runs, histories = [], {}
    for initialization in INITIALIZATIONS:
        directory = late_lr_drop_run_directory(project_root, splits, initialization)
        config = json.loads((directory / "config.json").read_text())
        metadata = json.loads((directory / "metadata.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        if summary.get("status") != "completed":
            raise ValueError(f"Fixed-drop run is incomplete: {directory.name}")
        key = f"{initialization}_late_lr_drop"
        runs.append({
            "key": key,
            "initialization": initialization,
            "directory": directory,
            "config": config,
            "metadata": metadata,
            "summary": summary,
        })
        with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
            histories[key] = list(csv.DictReader(handle))
    return runs, histories


def load_training_horizon_results(
    project_root: Path,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load the four completed long-run summaries and evaluation histories."""
    splits = json.loads((project_root / "data/metadata/splits.json").read_text())
    runs, histories = [], {}
    for initialization in INITIALIZATIONS:
        for learning_rate in HORIZON_LEARNING_RATES:
            directory = training_horizon_run_directory(
                project_root, splits, initialization, learning_rate
            )
            config = json.loads((directory / "config.json").read_text())
            metadata = json.loads((directory / "metadata.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            if summary.get("status") != "completed":
                raise ValueError(f"Training-horizon run is incomplete: {directory.name}")
            key = f"{initialization}_{learning_rate_label(learning_rate)}_steps4000"
            runs.append({
                "key": key,
                "initialization": initialization,
                "learning_rate": learning_rate,
                "directory": directory,
                "config": config,
                "metadata": metadata,
                "summary": summary,
            })
            with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
                histories[key] = list(csv.DictReader(handle))
    return runs, histories


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
