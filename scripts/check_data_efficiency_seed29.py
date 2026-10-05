"""Verify the four fresh Seed 29 data-efficiency runs on MPS."""

from pathlib import Path
import json

import torch

from roofseg.audit import build_splits
from roofseg.integrity import load_json
from roofseg.model import build_model
from roofseg.optimization import (
    INITIALIZATIONS,
    build_late_lr_drop_config,
    late_lr_drop_run_directory,
)
from roofseg.run_artifacts import verify_completed_run
from roofseg.training import select_device
from roofseg.training_data import paired_schedule

ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active geographic split does not reproduce exactly from RID metadata.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"Data-efficiency training requires MPS; selected {device}.")

repetitions = splits["training_subsets"]["repetitions"]
if repetitions["1"]["seed"] != 17 or repetitions["2"]["seed"] != 29:
    raise AssertionError("Saved subset repetitions do not have the expected seeds.")
subsets = {
    repetition: {
        size: repetitions[str(repetition)]["subsets"][str(size)]
        for size in (25, 100, 500)
    }
    for repetition in (1, 2)
}
for repetition, family in subsets.items():
    if not (set(family[25]) < set(family[100]) < set(family[500])):
        raise AssertionError(f"Repetition {repetition} subsets are not strictly nested.")
    if any(len(ids) != size or len(set(ids)) != size for size, ids in family.items()):
        raise AssertionError(f"Repetition {repetition} has an invalid saved subset.")

role_training = set(splits["roles"]["training"])
validation_ids = set(splits["roles"]["validation"])
test_ids = set(splits["roles"]["test"])
for repetition, sizes in ((1, (25, 100, 500)), (2, (100,))):
    for size in sizes:
        for initialization in INITIALIZATIONS:
            config = build_late_lr_drop_config(
                splits, initialization, repetition, size
            )
            directory = late_lr_drop_run_directory(
                ROOT, splits, initialization, repetition, size
            )
            if json.loads((directory / "config.json").read_text()) != config:
                raise AssertionError(f"Reference configuration changed: {directory.name}")
            verify_completed_run(directory)

schedule_records, initial_states, run_directories = {}, {}, []
for size in (25, 500):
    schedule_hashes = set()
    for initialization in INITIALIZATIONS:
        config = build_late_lr_drop_config(splits, initialization, 2, size)
        seed17_config = build_late_lr_drop_config(splits, initialization, 1, size)
        differences = {
            key for key in seed17_config | config
            if seed17_config.get(key) != config.get(key)
        }
        if differences != {"run_name", "data", "training_ids", "seeds"}:
            raise AssertionError(f"Unexpected Seed 29 differences: {differences}")
        if config["training_ids"] != subsets[2][size]:
            raise AssertionError(f"n={size} does not use the saved Seed 29 subset.")
        if not set(config["training_ids"]) <= role_training:
            raise AssertionError(f"n={size} contains an ID outside the training role.")
        if set(config["training_ids"]) & (validation_ids | test_ids):
            raise AssertionError(f"n={size} intersects validation or test roles.")
        if set(config["validation_ids"]) != validation_ids:
            raise AssertionError("Validation role changed.")
        _, schedule = paired_schedule(
            config["training_ids"], config["max_steps"], config["batch_size"],
            config["seeds"]["data_order"], config["seeds"]["augmentation"],
        )
        schedule_hashes.add(schedule["sha256"])
        schedule_records[size] = schedule
        _, model_report = build_model(initialization, config["seeds"]["model"])
        initial_states[f"n{size}_{initialization}"] = {
            "encoder_sha256": model_report["encoder_sha256"],
            "decoder_sha256": model_report["decoder_sha256"],
        }
        run_directories.append(late_lr_drop_run_directory(
            ROOT, splits, initialization, 2, size
        ))
    if len(schedule_hashes) != 1:
        raise AssertionError(f"n={size} model pair is not schedule-paired.")
    if initial_states[f"n{size}_random"]["decoder_sha256"] != (
        initial_states[f"n{size}_imagenet"]["decoder_sha256"]
    ):
        raise AssertionError(f"n={size} pair does not share decoder initialization.")

if len(set(run_directories)) != 4 or any(path.exists() for path in run_directories):
    raise AssertionError("The four new run directories are not unique and unused.")
report = {
    "device": str(device),
    "split_name": splits["split_name"],
    "run_count": 4,
    "training_sizes": [25, 500],
    "subset_repetition": 2,
    "subset_seed": 29,
    "nested_sizes_verified": True,
    "initialization_pairs_verified": True,
    "validation_and_test_excluded": True,
    "reference_runs_verified": 8,
    "test_role_absent": True,
    "run_directories": [str(path.relative_to(ROOT)) for path in run_directories],
    "training_id_overlap": {
        str(size): len(set(subsets[1][size]) & set(subsets[2][size]))
        for size in (25, 100, 500)
    },
    "schedule_records": {
        str(size): {key: schedule_records[size][key] for key in (
            "sha256", "optimizer_steps", "examples_processed",
            "equivalent_data_passages", "completed_or_partial_passages",
            "incomplete_batches_retained",
        )} for size in (25, 500)
    },
    "initial_states": initial_states,
}
(ROOT / "reports/data_efficiency_seed29_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
