"""Verify the two additional WHU n=100 subset repetitions before training."""

from pathlib import Path
import json

import torch

from roofseg.integrity import load_json
from roofseg.model import build_model
from roofseg.run_artifacts import json_digest, verify_completed_run
from roofseg.training import restore_checkpoint, select_device
from roofseg.training_data import paired_schedule
from roofseg.whu_data import build_whu_splits
from roofseg.whu_pilot import (
    SEED17_IDS_SHA256, SUBSET_REPETITIONS, build_whu_pilot_config,
    verify_historical_seed17_run, whu_run_directory, whu_subset_record,
)


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/whu/data_manifest.json")
splits = load_json(ROOT / "data/metadata/whu/splits.json")
if build_whu_splits(manifest, subset_seed=17) != splits:
    raise SystemExit("Saved WHU roles and all three subsets do not reproduce exactly.")

training_role = set(splits["roles"]["train"])
protected = set(splits["roles"]["val"]) | set(splits["roles"]["test"])
schedule_hashes = {}
for repetition in SUBSET_REPETITIONS:
    record = whu_subset_record(splits, repetition)
    ids = record["ids"]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise AssertionError(f"WHU repetition {repetition} is not 100 unique IDs.")
    if not set(ids) <= training_role or set(ids) & protected:
        raise AssertionError(f"WHU repetition {repetition} violates provider roles.")
    pair = [build_whu_pilot_config(splits, initialization, repetition)
            for initialization in ("random", "imagenet")]
    if pair[0]["training_ids"] != pair[1]["training_ids"] or pair[0]["seeds"] != pair[1]["seeds"]:
        raise AssertionError(f"WHU repetition {repetition} is not paired.")
    schedules = [paired_schedule(
        config["training_ids"], config["max_steps"], config["batch_size"],
        config["seeds"]["data_order"], config["seeds"]["augmentation"],
    )[1]["sha256"] for config in pair]
    if len(set(schedules)) != 1:
        raise AssertionError(f"WHU repetition {repetition} schedules differ by initialization.")
    schedule_hashes[str(record["seed"])] = schedules[0]

if json_digest(whu_subset_record(splits, 1)["ids"]) != SEED17_IDS_SHA256:
    raise AssertionError("Historical WHU Seed-17 IDs changed.")
for initialization in ("random", "imagenet"):
    verify_historical_seed17_run(ROOT, manifest, splits, initialization)

historical_config = build_whu_pilot_config(splits, "imagenet", 1)
historical_directory = whu_run_directory(ROOT, splits, "imagenet", 1)
metadata = load_json(historical_directory / "metadata.json")
model, _ = build_model("random", historical_config["seeds"]["model"])
step = restore_checkpoint(
    model, historical_directory / "checkpoint.pt",
    metadata["compatibility_identity"]["config_sha256"],
    metadata["compatibility_identity"]["input_sha256"],
)
if step != verify_completed_run(historical_directory)["best_step"]:
    raise AssertionError("Historical WHU checkpoint step differs from its summary.")

device = select_device()
if device.type != "mps":
    raise SystemExit(f"WHU subset replication requires MPS; selected {device}.")
report = {
    "device": str(device),
    "subset_seeds": [whu_subset_record(splits, repetition)["seed"] for repetition in SUBSET_REPETITIONS],
    "pairwise_id_overlaps": splits["additional_pilot_subsets"]["pairwise_id_overlaps"],
    "seed17_ids_sha256": SEED17_IDS_SHA256,
    "paired_schedule_sha256_by_subset_seed": schedule_hashes,
    "provider_roles_preserved": True,
    "validation_image_count": len(splits["roles"]["val"]),
    "provider_test_excluded": True,
    "historical_seed17_runs_verified_without_metadata_rewrite": True,
    "historical_checkpoint_restoration": True,
    "new_run_directories": [
        str(whu_run_directory(ROOT, splits, initialization, repetition).relative_to(ROOT))
        for repetition in (2, 3) for initialization in ("random", "imagenet")
    ],
}
(ROOT / "reports/whu_subset_replication_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
