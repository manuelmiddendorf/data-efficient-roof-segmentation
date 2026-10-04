"""Verify the four fresh Seed 17 data-efficiency runs on MPS."""

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

repetition = splits["training_subsets"]["repetitions"]["1"]
if repetition["seed"] != 17:
    raise AssertionError("Repetition 1 is not the saved Seed 17 subset family.")
subsets = {size: repetition["subsets"][str(size)] for size in (25, 100, 500)}
if not (set(subsets[25]) < set(subsets[100]) < set(subsets[500])):
    raise AssertionError("Seed 17 subsets are not strictly nested.")
if any(len(ids) != size or len(set(ids)) != size for size, ids in subsets.items()):
    raise AssertionError("A saved Seed 17 subset has the wrong size or duplicate IDs.")
role_training = set(splits["roles"]["training"])
forbidden = set(splits["roles"]["validation"]) | set(splits["roles"]["test"])

reference_configs = {
    initialization: build_late_lr_drop_config(splits, initialization)
    for initialization in INITIALIZATIONS
}
for initialization, config in reference_configs.items():
    directory = late_lr_drop_run_directory(ROOT, splits, initialization)
    if json.loads((directory / "config.json").read_text()) != config:
        raise AssertionError(f"n=100 reference changed: {directory.name}")
    verify_completed_run(directory)

schedule_records, initial_states, run_directories = {}, {}, []
for size in (25, 500):
    schedule_hashes = set()
    for initialization in INITIALIZATIONS:
        config = build_late_lr_drop_config(
            splits, initialization, subset_repetition=1, training_size=size
        )
        if config["training_ids"] != subsets[size] or not set(subsets[size]) <= role_training:
            raise AssertionError(f"n={size} does not use the saved Seed 17 subset.")
        if set(config["training_ids"]) & forbidden:
            raise AssertionError(f"n={size} intersects validation or test roles.")
        if config["seeds"] != reference_configs[initialization]["seeds"]:
            raise AssertionError("A non-subset training seed changed.")
        if config["scheduler"] != reference_configs[initialization]["scheduler"]:
            raise AssertionError("The selected fixed-drop schedule changed.")
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
            ROOT, splits, initialization, 1, size
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
    "subset_repetition": 1,
    "subset_seed": 17,
    "nested_sizes_verified": True,
    "validation_and_test_excluded": True,
    "n100_references_verified": True,
    "run_directories": [str(path.relative_to(ROOT)) for path in run_directories],
    "schedule_records": {
        str(size): {key: schedule_records[size][key] for key in (
            "sha256", "optimizer_steps", "examples_processed", "equivalent_data_passages",
            "completed_or_partial_passages", "incomplete_batches_retained",
        )} for size in (25, 500)
    },
    "initial_states": initial_states,
}
(ROOT / "reports/data_efficiency_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
