"""Verify the six random-selection runs without starting training."""
from pathlib import Path
import json

from roofseg.audit import build_splits
from roofseg.integrity import load_json
from roofseg.optimization import INITIALIZATIONS, build_late_lr_drop_config, late_lr_drop_run_directory
from roofseg.run_artifacts import verify_completed_run
from roofseg.selection_strategy import (RANDOM, SPATIAL, SUBSET_REPETITIONS,
    build_random_selection_config, random_selection_run_directory,
    spatial_subset_diagnostics, subset_id_overlaps, subset_record)
from roofseg.spatial import select_uniform_random_subset
from roofseg.training import select_device
from roofseg.training_data import paired_schedule

ROOT = Path(__file__).resolve().parents[1]
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active split and saved selection strategies do not reproduce exactly.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"Selection-strategy training requires MPS; selected {device}.")
training_role, validation_role, test_role = (set(splits["roles"][role]) for role in ("training", "validation", "test"))
run_directories, schedules = [], {}
for repetition in SUBSET_REPETITIONS:
    random_record = subset_record(splits, RANDOM, repetition)
    spatial_record = subset_record(splits, SPATIAL, repetition)
    seed = int(random_record["seed"])
    if seed != int(spatial_record["seed"]):
        raise AssertionError("Paired strategies do not use the same selection-seed label.")
    expected = select_uniform_random_subset(splits["roles"]["training"], 100, seed)
    ids = random_record["subsets"]["100"]
    if ids != expected or len(ids) != 100 or len(set(ids)) != 100 or not set(ids) <= training_role:
        raise AssertionError(f"Random subset seed {seed} is not the fixed deterministic selection.")
    schedules[seed] = set()
    for initialization in INITIALIZATIONS:
        config = build_random_selection_config(splits, initialization, repetition)
        reference = build_late_lr_drop_config(splits, initialization, repetition, 100)
        if set(config["training_ids"]) & (validation_role | test_role):
            raise AssertionError("Random training IDs intersect a protected role.")
        if set(config["validation_ids"]) != validation_role or config["data"]["test_used"]:
            raise AssertionError("Validation or test-use contract changed.")
        immutable = {"architecture", "initialization", "pretrained_encoder", "decoder_initialization",
                     "seeds", "augmentation", "batch_size", "precision", "loss", "optimizer",
                     "learning_rate", "weight_decay", "weight_decay_exclusion", "max_steps",
                     "evaluation_steps", "evaluation_interval", "probability_threshold",
                     "early_stopping", "scheduler", "batch_normalization", "incomplete_batch_policy",
                     "validation_ids"}
        if any(config[key] != reference[key] for key in immutable):
            raise AssertionError("A fixed n=100 training condition changed.")
        _, schedule = paired_schedule(config["training_ids"], config["max_steps"], config["batch_size"],
                                      config["seeds"]["data_order"], config["seeds"]["augmentation"])
        schedules[seed].add(schedule["sha256"])
        directory = random_selection_run_directory(ROOT, splits, initialization, repetition)
        run_directories.append(directory)
    if len(schedules[seed]) != 1:
        raise AssertionError("Initializations are not paired within a random subset.")
    for initialization in INITIALIZATIONS:
        directory = late_lr_drop_run_directory(ROOT, splits, initialization, repetition, 100)
        expected_config = build_late_lr_drop_config(splits, initialization, repetition, 100)
        if load_json(directory / "config.json") != expected_config:
            raise AssertionError(f"Historical spatial reference changed: {directory.name}")
        verify_completed_run(directory)
if len(set(run_directories)) != 6 or any(path.exists() for path in run_directories):
    raise AssertionError("New run directories are not six unique unused paths.")
diagnostics = spatial_subset_diagnostics(manifest, splits)
overlap = subset_id_overlaps(splits)
diagnostics.to_csv(ROOT / "reports/selection_strategy_n100_spatial_diagnostics.csv", index=False)
overlap.to_csv(ROOT / "reports/selection_strategy_n100_id_overlap.csv", index=False)
report = {"device": str(device), "split_name": splits["split_name"], "run_count": 6,
          "selection_seeds": [17, 29, 43], "training_size": 100,
          "random_selection_reproduced": True, "paired_training_conditions_verified": True,
          "validation_and_test_separation_verified": True,
          "historical_spatial_references_verified": 6,
          "schedule_sha256_by_seed": {str(seed): next(iter(values)) for seed, values in schedules.items()},
          "run_directories": [str(path.relative_to(ROOT)) for path in run_directories]}
(ROOT / "reports/selection_strategy_n100_preflight.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
print("\nSpatial diagnostics\n", diagnostics.to_string(index=False))
print("\nID overlap\n", overlap.to_string(index=False))
