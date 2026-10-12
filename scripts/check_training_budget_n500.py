"""Verify the four-run n=500 training-budget block without starting training."""

import json
from pathlib import Path

from roofseg.audit import build_splits
from roofseg.integrity import load_json
from roofseg.optimization import (
    INITIALIZATIONS,
    build_late_lr_drop_config,
    late_lr_drop_run_directory,
)
from roofseg.pilot import build_metadata
from roofseg.run_artifacts import json_digest, verify_completed_run
from roofseg.training import learning_rate_for_step, select_device
from roofseg.training_budget import (
    MAX_STEPS,
    PREFIX_STEPS,
    SUBSET_REPETITIONS,
    TRAINING_SIZE,
    build_training_budget_config,
    training_budget_run_directory,
)
from roofseg.training_data import paired_schedule


ROOT = Path(__file__).resolve().parents[1]
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active geographic split does not reproduce exactly.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"Training-budget block requires MPS; selected {device}.")

training_role = set(splits["roles"]["training"])
validation_role = set(splits["roles"]["validation"])
test_role = set(splits["roles"]["test"])
run_directories, schedule_prefixes = [], {}
for repetition in SUBSET_REPETITIONS:
    subset_seed = int(
        splits["training_subsets"]["repetitions"][str(repetition)]["seed"]
    )
    schedule_prefixes[subset_seed] = set()
    for initialization in INITIALIZATIONS:
        reference = build_late_lr_drop_config(
            splits, initialization, repetition, TRAINING_SIZE
        )
        extended = build_training_budget_config(splits, initialization, repetition)
        differences = {
            key for key in reference | extended
            if reference.get(key) != extended.get(key)
        }
        if differences != {
            "run_name", "experiment", "max_steps", "evaluation_steps",
            "scheduler", "additional_checkpoints",
        }:
            raise AssertionError(f"Unexpected extended-run differences: {differences}")
        if extended["training_ids"] != reference["training_ids"]:
            raise AssertionError("The saved n=500 training IDs changed.")
        if extended["seeds"] != reference["seeds"]:
            raise AssertionError("A non-subset random seed changed.")
        if not set(extended["training_ids"]) <= training_role:
            raise AssertionError("A training ID lies outside the training role.")
        if set(extended["validation_ids"]) != validation_role:
            raise AssertionError("The validation role changed.")
        if (set(extended["training_ids"]) | set(extended["validation_ids"])) & test_role:
            raise AssertionError("A planned input intersects the locked test role.")
        if extended["evaluation_steps"] != [0] + list(range(100, MAX_STEPS + 1, 100)):
            raise AssertionError("The evaluation schedule is not step 0 then every 100 steps.")
        expected_rates = {1: 1e-3, 2000: 1e-3, 2001: 1e-4, 4000: 1e-4, 8000: 1e-4}
        if any(learning_rate_for_step(extended, step) != rate for step, rate in expected_rates.items()):
            raise AssertionError("The fixed learning-rate sequence changed.")
        long_schedule, _ = paired_schedule(
            extended["training_ids"], extended["max_steps"], extended["batch_size"],
            extended["seeds"]["data_order"], extended["seeds"]["augmentation"],
        )
        short_schedule, _ = paired_schedule(
            reference["training_ids"], reference["max_steps"], reference["batch_size"],
            reference["seeds"]["data_order"], reference["seeds"]["augmentation"],
        )
        if long_schedule[:PREFIX_STEPS] != short_schedule:
            raise AssertionError("The first 4,000 data and D4 choices changed.")
        schedule_prefixes[subset_seed].add(json_digest(long_schedule[:PREFIX_STEPS]))
        verify_completed_run(late_lr_drop_run_directory(
            ROOT, splits, initialization, repetition, TRAINING_SIZE
        ))
        directory = training_budget_run_directory(
            ROOT, splits, initialization, repetition
        )
        run_directories.append(directory)
        if directory.exists() and any(directory.iterdir()):
            saved_config = load_json(directory / "config.json")
            saved_metadata = load_json(directory / "metadata.json")
            expected_metadata = build_metadata(ROOT, manifest, splits, extended, device)
            if saved_config != extended or saved_metadata.get("compatibility_identity") != (
                expected_metadata["compatibility_identity"]
            ):
                raise AssertionError(f"Existing run conflicts with the plan: {directory.name}")
            verify_completed_run(directory)

if len(set(run_directories)) != 4:
    raise AssertionError("The four extended run directories are not unique.")
if any(len(hashes) != 1 for hashes in schedule_prefixes.values()):
    raise AssertionError("Initialization pairs do not share schedules.")

report = {
    "device": str(device),
    "split_name": splits["split_name"],
    "run_count": 4,
    "training_size": TRAINING_SIZE,
    "subset_seeds": [17, 29],
    "max_steps": MAX_STEPS,
    "prefix_checkpoint_through_step": PREFIX_STEPS,
    "evaluation_interval": 100,
    "historical_reference_runs_verified": 4,
    "training_validation_test_separation_verified": True,
    "paired_conditions_verified": True,
    "schedule_prefix_matches_historical": True,
    "schedule_prefix_sha256_by_subset_seed": {
        str(seed): next(iter(hashes)) for seed, hashes in schedule_prefixes.items()
    },
    "run_directories": [str(path.relative_to(ROOT)) for path in run_directories],
}
(ROOT / "reports/training_budget_n500_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
