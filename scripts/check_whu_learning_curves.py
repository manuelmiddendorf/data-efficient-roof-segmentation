"""Verify the fixed WHU n=25/100/500 learning-curve block before training."""

from pathlib import Path
import json

import torch

from roofseg.integrity import load_json
from roofseg.model import build_model
from roofseg.run_artifacts import json_digest
from roofseg.training import adamw, select_device, synchronize, train_step
from roofseg.training_data import WHUTensorStore, paired_schedule
from roofseg.whu_data import build_whu_splits
from roofseg.whu_pilot import (
    LEARNING_CURVE_REPETITIONS, LEARNING_CURVE_SIZES, SEED17_IDS_SHA256,
    SEED29_IDS_SHA256, build_whu_pilot_config, verify_historical_whu_run,
    whu_run_directory, whu_subset_record,
)


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/whu/data_manifest.json")
splits = load_json(ROOT / "data/metadata/whu/splits.json")
if build_whu_splits(manifest, subset_seed=17) != splits:
    raise SystemExit("Saved WHU roles and learning-curve subsets do not reproduce exactly.")

historical_hashes = {
    1: SEED17_IDS_SHA256,
    2: SEED29_IDS_SHA256,
}
training_role = set(splits["roles"]["train"])
protected = set(splits["roles"]["val"]) | set(splits["roles"]["test"])
schedule_hashes = {}
for repetition in LEARNING_CURVE_REPETITIONS:
    sets = {
        size: whu_subset_record(splits, repetition, size)["ids"]
        for size in LEARNING_CURVE_SIZES
    }
    if not set(sets[25]) < set(sets[100]) < set(sets[500]):
        raise AssertionError(f"WHU repetition {repetition} is not strictly nested.")
    if json_digest(sets[100]) != historical_hashes[repetition]:
        raise AssertionError(f"Historical WHU n=100 IDs changed for repetition {repetition}.")
    for size, ids in sets.items():
        if len(ids) != size or len(set(ids)) != size:
            raise AssertionError(f"WHU n={size} repetition {repetition} has invalid IDs.")
        if not set(ids) <= training_role or set(ids) & protected:
            raise AssertionError(f"WHU n={size} repetition {repetition} violates provider roles.")
        configs = [
            build_whu_pilot_config(splits, initialization, repetition, size)
            for initialization in ("random", "imagenet")
        ]
        if configs[0]["training_ids"] != configs[1]["training_ids"]:
            raise AssertionError("Paired WHU training IDs differ.")
        schedules = [paired_schedule(
            config["training_ids"], config["max_steps"], config["batch_size"],
            config["seeds"]["data_order"], config["seeds"]["augmentation"],
        ) for config in configs]
        if schedules[0] != schedules[1]:
            raise AssertionError("Paired WHU order or D4 schedules differ.")
        schedule_hashes[f"seed{configs[0]['seeds']['subset']}_n{size}"] = schedules[0][1]["sha256"]

for repetition in LEARNING_CURVE_REPETITIONS:
    for initialization in ("random", "imagenet"):
        verify_historical_whu_run(ROOT, manifest, splits, initialization, repetition)

config = build_whu_pilot_config(splits, "random", subset_repetition=1, training_size=25)
schedule, schedule_record = paired_schedule(
    config["training_ids"], config["max_steps"], config["batch_size"],
    config["seeds"]["data_order"], config["seeds"]["augmentation"],
)
incomplete_ids, incomplete_codes = next(
    (ids, codes) for ids, codes in schedule if len(ids) < config["batch_size"]
)
if len(incomplete_ids) != 1 or not schedule_record["incomplete_batches_retained"]:
    raise AssertionError("The real WHU n=25 schedule did not retain one-sample batches.")

device = select_device()
if device.type != "mps":
    raise SystemExit(f"WHU learning curves require MPS; selected {device}.")
store = WHUTensorStore(ROOT, manifest)
model, _ = build_model("random", config["seeds"]["model"])
model.to(device)
optimizer, _ = adamw(model, config["learning_rate"], config["weight_decay"])
values = train_step(
    model, optimizer, store.batch(incomplete_ids, incomplete_codes), device
)
synchronize(device)
if not all(torch.isfinite(parameter.grad).all().item()
           for parameter in model.parameters() if parameter.grad is not None):
    raise FloatingPointError("WHU n=25 MPS gradients are non-finite.")

report = {
    "device": str(device),
    "training_sizes": list(LEARNING_CURVE_SIZES),
    "subset_seeds": [17, 29],
    "historical_n100_ids_unchanged": True,
    "strict_nesting": True,
    "deterministic_reproduction": True,
    "provider_test_excluded": True,
    "pairwise_id_overlaps": splits["learning_curve_subsets"]["pairwise_id_overlaps"],
    "paired_schedule_sha256": schedule_hashes,
    "real_n25_incomplete_batch": {
        "size": len(incomplete_ids), "ids": incomplete_ids,
        "d4_codes": incomplete_codes, "loss": values["loss"],
    },
    "historical_n100_runs_verified_without_metadata_rewrite": True,
    "new_run_directories": [
        str(whu_run_directory(ROOT, splits, initialization, repetition, size).relative_to(ROOT))
        for repetition in LEARNING_CURVE_REPETITIONS
        for size in (25, 500)
        for initialization in ("random", "imagenet")
    ],
}
(ROOT / "reports/whu_learning_curves_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
