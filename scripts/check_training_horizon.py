"""Verify the paired 4,000-step Stage 3 training-horizon block on MPS."""

from pathlib import Path
import json

import torch

from roofseg.audit import build_splits
from roofseg.integrity import load_json
from roofseg.model import build_model
from roofseg.optimization import (
    HORIZON_LEARNING_RATES,
    INITIALIZATIONS,
    LONG_HORIZON_STEPS,
    build_learning_rate_config,
    build_training_horizon_config,
    learning_rate_run_directory,
    training_horizon_run_directory,
)
from roofseg.run_artifacts import json_digest, verify_completed_run
from roofseg.training import adamw, select_device, synchronize, train_step
from roofseg.training_data import RIDTensorStore, paired_schedule


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active geographic split does not reproduce exactly from RID metadata.")

device = select_device()
if device.type != "mps":
    raise SystemExit(f"Stage 3 training requires available MPS on this host; selected {device}.")

configs = [
    build_training_horizon_config(splits, initialization, learning_rate)
    for initialization in INITIALIZATIONS
    for learning_rate in HORIZON_LEARNING_RATES
]
test_ids = set(splits["roles"]["test"])
if any((set(config["training_ids"]) | set(config["validation_ids"])) & test_ids
       for config in configs):
    raise AssertionError("Training-horizon inputs intersect the locked test role.")

base = configs[0]
schedule_hashes, prefix_hashes = set(), set()
for config in configs:
    if config["max_steps"] != LONG_HORIZON_STEPS:
        raise AssertionError("A long-run configuration does not contain 4,000 updates.")
    if config["evaluation_steps"] != [0] + list(range(100, 4001, 100)):
        raise AssertionError("The long-run evaluation schedule is not step 0 then every 100 steps.")
    if any(config[key] != base[key] for key in (
        "training_ids", "validation_ids", "seeds", "batch_size", "weight_decay",
        "max_steps", "evaluation_steps", "evaluation_interval", "augmentation", "loss",
        "precision", "scheduler", "early_stopping", "batch_normalization",
    )):
        raise AssertionError("A paired long-run setting differs unexpectedly.")
    long_schedule, long_record = paired_schedule(
        config["training_ids"], config["max_steps"], config["batch_size"],
        config["seeds"]["data_order"], config["seeds"]["augmentation"],
    )
    short_config = build_learning_rate_config(
        splits, config["initialization"], config["learning_rate"]
    )
    short_schedule, short_record = paired_schedule(
        short_config["training_ids"], short_config["max_steps"], short_config["batch_size"],
        short_config["seeds"]["data_order"], short_config["seeds"]["augmentation"],
    )
    if long_schedule[:2000] != short_schedule:
        raise AssertionError("The first 2,000 IDs or D4 transformations changed.")
    schedule_hashes.add(long_record["sha256"])
    prefix_hashes.add(json_digest(long_schedule[:2000]))
    if json_digest(long_schedule[:2000]) != json_digest(short_schedule):
        raise AssertionError("The 2,000-step prefix digest differs from its historical schedule.")
    if short_record["optimizer_steps"] != 2000:
        raise AssertionError("Historical comparison configuration is not 2,000 steps.")
    historical_directory = learning_rate_run_directory(
        ROOT, splits, config["initialization"], config["learning_rate"]
    )
    long_directory = training_horizon_run_directory(
        ROOT, splits, config["initialization"], config["learning_rate"]
    )
    if historical_directory == long_directory:
        raise AssertionError("Historical and extended artifacts share a directory.")
    verify_completed_run(historical_directory)

if len(schedule_hashes) != 1 or len(prefix_hashes) != 1:
    raise AssertionError("Long schedules are not paired across the four runs.")

initial_states = {}
for initialization in INITIALIZATIONS:
    reports = []
    for learning_rate in HORIZON_LEARNING_RATES:
        config = build_training_horizon_config(splits, initialization, learning_rate)
        model, report = build_model(initialization, config["seeds"]["model"])
        _, optimizer_record = adamw(model, config["learning_rate"], config["weight_decay"])
        if optimizer_record["learning_rates"] != [learning_rate, learning_rate]:
            raise AssertionError("A constant learning rate did not reach both optimizer groups.")
        reports.append((report["encoder_sha256"], report["decoder_sha256"]))
    if len(set(reports)) != 1:
        raise AssertionError("Rates within an initialization do not share their initial state.")
    initial_states[initialization] = reports[0]
if initial_states["random"][1] != initial_states["imagenet"][1]:
    raise AssertionError("Random and ImageNet runs do not share the paired decoder state.")

smoke_config = build_training_horizon_config(splits, "random", 1e-3)
model, _ = build_model("random", smoke_config["seeds"]["model"])
model.to(device)
optimizer, _ = adamw(model, smoke_config["learning_rate"], smoke_config["weight_decay"])
batch_ids = smoke_config["training_ids"][:4]
values = train_step(
    model, optimizer,
    RIDTensorStore(ROOT, manifest).batch(batch_ids, [0, 1, 2, 3]), device,
)
synchronize(device)
if not all(torch.isfinite(torch.tensor(value)).item() for value in values.values()):
    raise FloatingPointError("Training-horizon smoke step produced a non-finite value.")

report = {
    "device": str(device),
    "split_name": splits["split_name"],
    "training_size": 100,
    "learning_rates": list(HORIZON_LEARNING_RATES),
    "max_steps": LONG_HORIZON_STEPS,
    "evaluation_steps": [0] + list(range(100, 4001, 100)),
    "run_directories": [
        str(training_horizon_run_directory(
            ROOT, splits, config["initialization"], config["learning_rate"]
        ).relative_to(ROOT))
        for config in configs
    ],
    "locked_test_role_absent": True,
    "historical_runs_verified": True,
    "paired_long_schedule_sha256": next(iter(schedule_hashes)),
    "paired_2000_step_prefix_sha256": next(iter(prefix_hashes)),
    "prefix_matches_historical_schedule": True,
    "initial_states": initial_states,
    "optimizer_learning_rates_verified": True,
    "estimated_total_minutes": [100, 110],
    "smoke": {"initialization": "random", "learning_rate": 1e-3,
              "sample_ids": batch_ids, **values},
}
(ROOT / "reports/training_horizon_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
