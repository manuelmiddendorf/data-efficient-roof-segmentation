"""Verify the two-run Seed 29 fixed-drop replication on MPS."""

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
    raise SystemExit(f"Stage 3 training requires available MPS on this host; selected {device}.")

repetition = splits["training_subsets"]["repetitions"]["2"]
expected_ids = repetition["subsets"]["100"]
if repetition["seed"] != 29 or len(expected_ids) != 100 or len(set(expected_ids)) != 100:
    raise AssertionError("Saved repetition 2 is not the expected Seed 29 n=100 subset.")
role_training = set(splits["roles"]["training"])
validation_ids = set(splits["roles"]["validation"])
test_ids = set(splits["roles"]["test"])
configs = [build_late_lr_drop_config(splits, initialization, 2)
           for initialization in INITIALIZATIONS]

schedule_hashes, initial_states = set(), {}
for config in configs:
    seed17_config = build_late_lr_drop_config(splits, config["initialization"], 1)
    seed17_directory = late_lr_drop_run_directory(
        ROOT, splits, config["initialization"], 1
    )
    if json.loads((seed17_directory / "config.json").read_text()) != seed17_config:
        raise AssertionError(f"Seed 17 reference configuration changed: {seed17_directory.name}")
    verify_completed_run(seed17_directory)
    differences = {
        key for key in seed17_config | config
        if seed17_config.get(key) != config.get(key)
    }
    if differences != {"run_name", "data", "training_ids", "seeds"}:
        raise AssertionError(f"Unexpected Seed 29 configuration differences: {differences}")
    if config["training_ids"] != expected_ids or not set(expected_ids) <= role_training:
        raise AssertionError("Seed 29 configuration does not use the saved training subset.")
    if set(config["training_ids"]) & (validation_ids | test_ids):
        raise AssertionError("Seed 29 training inputs intersect validation or test roles.")
    if set(config["validation_ids"]) != validation_ids or set(config["validation_ids"]) & test_ids:
        raise AssertionError("Validation configuration is inconsistent with the locked roles.")
    if config["seeds"] != {
        "subset": 29, "model": 20261004, "data_order": 1701, "augmentation": 1702
    }:
        raise AssertionError("Only the subset seed may differ from the Seed 17 recipe.")
    _, schedule = paired_schedule(
        config["training_ids"], config["max_steps"], config["batch_size"],
        config["seeds"]["data_order"], config["seeds"]["augmentation"],
    )
    schedule_hashes.add(schedule["sha256"])
    _, model_report = build_model(config["initialization"], config["seeds"]["model"])
    initial_states[config["initialization"]] = {
        "encoder_sha256": model_report["encoder_sha256"],
        "decoder_sha256": model_report["decoder_sha256"],
    }

if configs[0]["training_ids"] != configs[1]["training_ids"] or len(schedule_hashes) != 1:
    raise AssertionError("The Seed 29 model pair is not data-schedule paired.")
if initial_states["random"]["decoder_sha256"] != initial_states["imagenet"]["decoder_sha256"]:
    raise AssertionError("The Seed 29 model pair does not share the decoder initialization.")
run_directories = [late_lr_drop_run_directory(ROOT, splits, init, 2)
                   for init in INITIALIZATIONS]
if len(set(run_directories)) != 2 or any(path.exists() for path in run_directories):
    raise AssertionError("Seed 29 run directories are not two new unique paths.")

seed17_ids = set(splits["training_subsets"]["repetitions"]["1"]["subsets"]["100"])
report = {
    "device": str(device),
    "split_name": splits["split_name"],
    "run_count": 2,
    "training_size": 100,
    "subset_repetition": 2,
    "subset_seed": 29,
    "seed17_seed29_training_id_overlap": len(seed17_ids & set(expected_ids)),
    "paired_schedule_sha256": next(iter(schedule_hashes)),
    "run_directories": [str(path.relative_to(ROOT)) for path in run_directories],
    "initial_states": initial_states,
    "seed17_references_verified": True,
    "validation_and_test_excluded_from_training": True,
    "test_role_absent": True,
}
(ROOT / "reports/seed29_replication_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
