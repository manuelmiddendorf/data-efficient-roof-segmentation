"""Tests for the bounded random-versus-spatial selection block."""
from pathlib import Path

from roofseg.optimization import build_late_lr_drop_config
from roofseg.selection_strategy import (RANDOM, build_random_selection_config,
                                         random_selection_run_directory, subset_record)


def _splits() -> dict:
    repetitions = {}
    random = {}
    for repetition, seed in enumerate((17, 29, 43), 1):
        repetitions[str(repetition)] = {"seed": seed, "subsets": {"100": [str(i) for i in range(100)]}}
        random[str(repetition)] = {"seed": seed, "subsets": {"100": [str(i) for i in range(100, 200)]}}
    return {"split_name": "geographic", "roles": {"training": [str(i) for i in range(300)],
            "validation": ["300"], "test": ["301"]},
            "training_subsets": {"repetitions": repetitions,
              "additional_strategies": {RANDOM: {"repetitions": random}}}}


def test_random_config_changes_only_selection_identity_and_ids():
    splits = _splits()
    reference = build_late_lr_drop_config(splits, "random", 2, 100)
    candidate = build_random_selection_config(splits, "random", 2)
    assert candidate["training_ids"] == [str(i) for i in range(100, 200)]
    assert candidate["selection_strategy"] == RANDOM
    assert candidate["seeds"] == reference["seeds"]
    differences = {key for key in reference | candidate if reference.get(key) != candidate.get(key)}
    assert differences == {"run_name", "experiment", "selection_strategy", "data", "training_ids"}
    assert subset_record(splits, RANDOM, 2)["seed"] == 29


def test_random_runs_are_unique_by_seed_and_initialization(tmp_path: Path):
    splits = _splits()
    directories = {random_selection_run_directory(tmp_path, splits, initialization, repetition)
                   for repetition in (1, 2, 3) for initialization in ("random", "imagenet")}
    assert len(directories) == 6
