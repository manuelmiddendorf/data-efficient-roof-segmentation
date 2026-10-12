import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from roofseg.integrity import sha256_file
from roofseg.optimization import build_late_lr_drop_config
from roofseg.training import learning_rate_for_step
from roofseg.training_budget import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_NAME,
    build_training_budget_config,
    selected_history_row,
    training_budget_run_directory,
)
from roofseg.run_artifacts import REQUIRED_COMPLETED_FILES, verify_completed_run
from roofseg.training import restore_checkpoint
from roofseg.training_data import paired_schedule


def _splits() -> dict:
    repetitions = {}
    for repetition, seed, offset in ((1, 17, 0), (2, 29, 600)):
        repetitions[str(repetition)] = {
            "seed": seed,
            "subsets": {"500": [str(index) for index in range(offset, offset + 500)]},
        }
    return {
        "split_name": "geographic_v2",
        "roles": {"validation": ["1500"], "test": ["1600"]},
        "training_subsets": {"repetitions": repetitions},
    }


def test_extended_budget_changes_only_identity_horizon_and_checkpoint_policy(tmp_path: Path):
    splits = _splits()
    for repetition in (1, 2):
        for initialization in ("random", "imagenet"):
            reference = build_late_lr_drop_config(splits, initialization, repetition, 500)
            extended = build_training_budget_config(splits, initialization, repetition)
            differences = {
                key for key in reference | extended
                if reference.get(key) != extended.get(key)
            }
            assert differences == {
                "run_name", "experiment", "max_steps", "evaluation_steps",
                "scheduler", "additional_checkpoints",
            }
            assert extended["training_ids"] == reference["training_ids"]
            assert extended["seeds"] == reference["seeds"]
            assert extended["max_steps"] == 8000
            assert extended["additional_checkpoints"] == [{
                "name": CHECKPOINT_NAME,
                "filename": CHECKPOINT_FILENAME,
                "through_step": 4000,
            }]
            assert learning_rate_for_step(extended, 2000) == 1e-3
            assert learning_rate_for_step(extended, 2001) == 1e-4
            assert learning_rate_for_step(extended, 8000) == 1e-4

    directories = {
        training_budget_run_directory(tmp_path, splits, initialization, repetition)
        for repetition in (1, 2) for initialization in ("random", "imagenet")
    }
    assert len(directories) == 4
    assert all("steps8000" in directory.name for directory in directories)


def test_extended_schedule_has_the_exact_historical_4000_step_prefix():
    splits = _splits()
    for repetition in (1, 2):
        prefixes = []
        for initialization in ("random", "imagenet"):
            reference = build_late_lr_drop_config(splits, initialization, repetition, 500)
            extended = build_training_budget_config(splits, initialization, repetition)
            short, _ = paired_schedule(
                reference["training_ids"], 4000, reference["batch_size"],
                reference["seeds"]["data_order"], reference["seeds"]["augmentation"],
            )
            long, _ = paired_schedule(
                extended["training_ids"], 8000, extended["batch_size"],
                extended["seeds"]["data_order"], extended["seeds"]["augmentation"],
            )
            assert long[:4000] == short
            prefixes.append(long[:4000])
        assert prefixes[0] == prefixes[1]


def test_checkpoint_selection_ranges_exclude_step_zero_and_break_ties_early():
    history = pd.DataFrame({
        "step": [0, 100, 4000, 4100, 8000],
        "validation_mean_iou": [0.99, 0.80, 0.80, 0.82, 0.82],
    })
    assert int(selected_history_row(history, 4000).step) == 100
    assert int(selected_history_row(history, 8000).step) == 4100


def test_additional_checkpoint_is_restorable_and_required_for_completed_reuse(tmp_path: Path):
    model = torch.nn.Conv2d(1, 1, 1)
    payload = {
        "model": model.state_dict(), "step": 4000,
        "config_sha256": "config", "input_sha256": "input",
    }
    torch.save(payload, tmp_path / "checkpoint.pt")
    torch.save(payload, tmp_path / CHECKPOINT_FILENAME)
    for filename in REQUIRED_COMPLETED_FILES - {"checkpoint.pt", "summary.json"}:
        (tmp_path / filename).write_text("{}" if filename.endswith(".json") else "x")
    summary = {
        "status": "completed",
        "checkpoint_sha256": sha256_file(tmp_path / "checkpoint.pt"),
        "additional_checkpoints": {CHECKPOINT_NAME: {
            "filename": CHECKPOINT_FILENAME,
            "checkpoint_sha256": sha256_file(tmp_path / CHECKPOINT_FILENAME),
        }},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    verify_completed_run(tmp_path)
    assert restore_checkpoint(
        model, tmp_path / CHECKPOINT_FILENAME, "config", "input"
    ) == 4000
    (tmp_path / CHECKPOINT_FILENAME).unlink()
    with pytest.raises(ValueError, match="Additional checkpoint is missing"):
        verify_completed_run(tmp_path)
