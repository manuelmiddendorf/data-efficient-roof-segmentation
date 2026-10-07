"""Verify the eight planned Seed 43 runs without starting training."""

from pathlib import Path
import json

from roofseg.audit import build_splits
from roofseg.boundary_weighted import (
    boundary_run_directory,
    build_boundary_config,
    training_band_fractions,
)
from roofseg.integrity import load_json
from roofseg.optimization import (
    INITIALIZATIONS,
    build_late_lr_drop_config,
    late_lr_drop_run_directory,
)
from roofseg.run_artifacts import verify_completed_run
from roofseg.training import select_device
from roofseg.training_data import RIDTensorStore, paired_schedule

ROOT = Path(__file__).resolve().parents[1]
SIZES = (25, 100, 500)
REPETITION = 3

manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active geographic split does not reproduce exactly.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"Seed 43 training requires MPS; selected {device}.")

repetitions = splits["training_subsets"]["repetitions"]
if int(repetitions[str(REPETITION)]["seed"]) != 43:
    raise AssertionError("Saved repetition 3 is not subset seed 43.")
subsets = {
    size: repetitions[str(REPETITION)]["subsets"][str(size)] for size in SIZES
}
if not (set(subsets[25]) < set(subsets[100]) < set(subsets[500])):
    raise AssertionError("Seed 43 subsets are not strictly nested.")

training_role = set(splits["roles"]["training"])
validation_role = set(splits["roles"]["validation"])
test_role = set(splits["roles"]["test"])
run_directories, schedule_hashes = [], {}
for size in SIZES:
    schedule_hashes[size] = set()
    for initialization in INITIALIZATIONS:
        config = build_late_lr_drop_config(splits, initialization, REPETITION, size)
        earlier = build_late_lr_drop_config(splits, initialization, 1, size)
        differences = {key for key in config | earlier if config.get(key) != earlier.get(key)}
        if differences != {"run_name", "data", "training_ids", "seeds"}:
            raise AssertionError(f"Unexpected reference differences: {differences}")
        if config["training_ids"] != subsets[size]:
            raise AssertionError(f"n={size} does not use the saved Seed 43 subset.")
        if not set(config["training_ids"]) <= training_role:
            raise AssertionError("A training ID lies outside the training role.")
        if (set(config["training_ids"]) | set(config["validation_ids"])) & test_role:
            raise AssertionError("A planned input intersects the locked test role.")
        if set(config["validation_ids"]) != validation_role:
            raise AssertionError("The validation role changed.")
        _, schedule = paired_schedule(
            config["training_ids"], config["max_steps"], config["batch_size"],
            config["seeds"]["data_order"], config["seeds"]["augmentation"],
        )
        schedule_hashes[size].add(schedule["sha256"])
        run_directories.append(late_lr_drop_run_directory(
            ROOT, splits, initialization, REPETITION, size
        ))
    if len(schedule_hashes[size]) != 1:
        raise AssertionError(f"n={size} initialization pair is not schedule-paired.")

# Historical runs are inputs to the analyses and must remain compatible.
for repetition in (1, 2):
    for size in SIZES:
        for initialization in INITIALIZATIONS:
            directory = late_lr_drop_run_directory(
                ROOT, splits, initialization, repetition, size
            )
            expected = build_late_lr_drop_config(splits, initialization, repetition, size)
            if load_json(directory / "config.json") != expected:
                raise AssertionError(f"Historical reference changed: {directory.name}")
            verify_completed_run(directory)

store = RIDTensorStore(ROOT, manifest)
fractions = training_band_fractions(store, subsets[25])
mean_fraction = sum(fractions.values()) / len(fractions)
for initialization in INITIALIZATIONS:
    reference = build_late_lr_drop_config(splits, initialization, REPETITION, 25)
    boundary = build_boundary_config(
        splits, initialization, REPETITION, "equal_per_image", mean_fraction
    )
    differences = {key for key in reference | boundary if reference.get(key) != boundary.get(key)}
    if differences != {"run_name", "experiment", "loss", "boundary_weighted_bce"}:
        raise AssertionError(f"Unexpected boundary differences: {differences}")
    if boundary["boundary_weighted_bce"]["mean_training_band_fraction"] != mean_fraction:
        raise AssertionError("Boundary coefficient differs within the initialization pair.")
    run_directories.append(boundary_run_directory(
        ROOT, splits, initialization, REPETITION, "equal_per_image"
    ))

if len(set(run_directories)) != 8 or any(path.exists() for path in run_directories):
    raise AssertionError("The eight Seed 43 run directories are not unique and unused.")

overlap = {}
for other_repetition in (1, 2):
    other_seed = repetitions[str(other_repetition)]["seed"]
    overlap[str(other_seed)] = {
        str(size): len(
            set(subsets[size])
            & set(repetitions[str(other_repetition)]["subsets"][str(size)])
        )
        for size in SIZES
    }
report = {
    "device": str(device),
    "split_name": splits["split_name"],
    "run_count": 8,
    "reference_run_count": 6,
    "boundary_variant_b_run_count": 2,
    "subset_repetition": REPETITION,
    "subset_seed": 43,
    "nested_sizes_verified": True,
    "paired_training_conditions_verified": True,
    "validation_and_test_separation_verified": True,
    "historical_reference_runs_verified": 12,
    "boundary_mean_training_band_fraction": mean_fraction,
    "boundary_coefficient_source": "Seed 43 n=25 training masks only",
    "training_id_overlap_with_prior_seeds": overlap,
    "schedule_sha256_by_size": {
        str(size): next(iter(hashes)) for size, hashes in schedule_hashes.items()
    },
    "run_directories": [str(path.relative_to(ROOT)) for path in run_directories],
}
(ROOT / "reports/seed43_replication_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
