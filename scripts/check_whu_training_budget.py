"""Verify the six-run WHU n=500 training-budget block without training."""

import json
from pathlib import Path

from roofseg.integrity import load_json
from roofseg.pilot import INITIALIZATIONS, build_metadata
from roofseg.run_artifacts import json_digest, verify_completed_run
from roofseg.training import learning_rate_for_step, select_device
from roofseg.training_data import paired_schedule
from roofseg.whu_data import build_whu_splits
from roofseg.whu_pilot import (
    LEARNING_CURVE_REPETITIONS,
    build_whu_pilot_config,
    verify_historical_whu_run,
    whu_subset_record,
)
from roofseg.whu_training_budget import (
    MAX_STEPS,
    PREFIX_STEPS,
    TRAINING_SIZE,
    build_whu_training_budget_config,
    whu_training_budget_run_directory,
)


ROOT = Path(__file__).resolve().parents[1]
manifest = load_json(ROOT / "data/metadata/whu/data_manifest.json")
splits = load_json(ROOT / "data/metadata/whu/splits.json")
if build_whu_splits(manifest, subset_seed=17) != splits:
    raise SystemExit("Saved WHU roles and subsets do not reproduce exactly.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"WHU training-budget block requires MPS; selected {device}.")

training_role = set(splits["roles"]["train"])
validation_role = set(splits["roles"]["val"])
test_role = set(splits["roles"]["test"])
run_directories, schedule_prefixes = [], {}
for repetition in LEARNING_CURVE_REPETITIONS:
    subset_seed = int(whu_subset_record(splits, repetition, TRAINING_SIZE)["seed"])
    schedule_prefixes[subset_seed] = set()
    for initialization in INITIALIZATIONS:
        reference = build_whu_pilot_config(
            splits, initialization, repetition, TRAINING_SIZE
        )
        extended = build_whu_training_budget_config(
            splits, initialization, repetition
        )
        differences = {
            key for key in reference | extended
            if reference.get(key) != extended.get(key)
        }
        if differences != {
            "run_name", "experiment", "max_steps", "evaluation_steps",
            "scheduler", "additional_checkpoints",
        }:
            raise AssertionError(f"Unexpected budget-run differences: {differences}")
        if extended["training_ids"] != reference["training_ids"]:
            raise AssertionError("The saved WHU n=500 IDs changed.")
        if extended["seeds"] != reference["seeds"]:
            raise AssertionError("A non-horizon random seed changed.")
        if not set(extended["training_ids"]) <= training_role:
            raise AssertionError("A training ID lies outside the provider train role.")
        if set(extended["validation_ids"]) != validation_role:
            raise AssertionError("The provider validation role changed.")
        if (set(extended["training_ids"]) | validation_role) & test_role:
            raise AssertionError("A planned input intersects the provider test role.")
        if extended["evaluation_steps"] != [0] + list(
            range(100, MAX_STEPS + 1, 100)
        ):
            raise AssertionError("Evaluation is not scheduled at 0 and every 100 updates.")
        expected_rates = {
            1: 1e-3, 2000: 1e-3, 2001: 1e-4, 4000: 1e-4, 8000: 1e-4,
        }
        if any(
            learning_rate_for_step(extended, step) != rate
            for step, rate in expected_rates.items()
        ):
            raise AssertionError("The fixed learning-rate sequence changed.")
        long_schedule, _ = paired_schedule(
            extended["training_ids"], extended["max_steps"],
            extended["batch_size"], extended["seeds"]["data_order"],
            extended["seeds"]["augmentation"],
        )
        short_schedule, _ = paired_schedule(
            reference["training_ids"], reference["max_steps"],
            reference["batch_size"], reference["seeds"]["data_order"],
            reference["seeds"]["augmentation"],
        )
        if long_schedule[:PREFIX_STEPS] != short_schedule:
            raise AssertionError("The 4,000-update ID and D4 prefix changed.")
        schedule_prefixes[subset_seed].add(
            json_digest(long_schedule[:PREFIX_STEPS])
        )
        verify_historical_whu_run(
            ROOT, manifest, splits, initialization, repetition, TRAINING_SIZE
        )
        directory = whu_training_budget_run_directory(
            ROOT, splits, initialization, repetition
        )
        run_directories.append(directory)
        if directory.exists() and any(directory.iterdir()):
            saved_config = load_json(directory / "config.json")
            saved_metadata = load_json(directory / "metadata.json")
            expected_metadata = build_metadata(
                ROOT, manifest, splits, extended, device
            )
            if saved_config != extended or saved_metadata.get(
                "compatibility_identity"
            ) != expected_metadata["compatibility_identity"]:
                raise AssertionError(
                    f"Existing run conflicts with the plan: {directory.name}"
                )
            verify_completed_run(directory)

if len(set(run_directories)) != 6:
    raise AssertionError("The six WHU budget run directories are not unique.")
if any(len(hashes) != 1 for hashes in schedule_prefixes.values()):
    raise AssertionError("Initialization pairs do not share schedules.")

report = {
    "device": str(device),
    "split_name": splits["split_name"],
    "run_count": 6,
    "training_size": TRAINING_SIZE,
    "subset_seeds": [17, 29, 43],
    "max_steps": MAX_STEPS,
    "prefix_checkpoint_through_step": PREFIX_STEPS,
    "evaluation_interval": 100,
    "historical_reference_runs_verified": 6,
    "provider_test_excluded": True,
    "paired_conditions_verified": True,
    "schedule_prefix_matches_historical": True,
    "schedule_prefix_sha256_by_subset_seed": {
        str(seed): next(iter(hashes))
        for seed, hashes in schedule_prefixes.items()
    },
    "run_directories": [
        str(path.relative_to(ROOT)) for path in run_directories
    ],
}
(ROOT / "reports/whu_training_budget_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
