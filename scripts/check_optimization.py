"""Run the real-data MPS smoke checks for the Stage 3 learning-rate path."""

from pathlib import Path
import json

import torch

from roofseg.audit import build_splits
from roofseg.integrity import load_json
from roofseg.model import build_model
from roofseg.optimization import (
    INITIALIZATIONS,
    LEARNING_RATES,
    build_learning_rate_config,
    learning_rate_run_directory,
)
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
    build_learning_rate_config(splits, initialization, learning_rate)
    for initialization in INITIALIZATIONS
    for learning_rate in LEARNING_RATES
]
test_ids = set(splits["roles"]["test"])
if any((set(config["training_ids"]) | set(config["validation_ids"])) & test_ids
       for config in configs):
    raise AssertionError("Optimization inputs intersect the locked test role.")

base = configs[0]
schedule_hashes = set()
for config in configs:
    if any(config[key] != base[key] for key in (
        "training_ids", "validation_ids", "seeds", "batch_size", "weight_decay",
        "max_steps", "evaluation_steps", "augmentation", "loss",
    )):
        raise AssertionError("A comparison setting other than learning rate differs.")
    _, record = paired_schedule(
        config["training_ids"], config["max_steps"], config["batch_size"],
        config["seeds"]["data_order"], config["seeds"]["augmentation"],
    )
    schedule_hashes.add(record["sha256"])
if len(schedule_hashes) != 1:
    raise AssertionError("Learning-rate schedules are not paired.")

initial_states = {}
for initialization in INITIALIZATIONS:
    reports = []
    for learning_rate in LEARNING_RATES:
        config = build_learning_rate_config(splits, initialization, learning_rate)
        model, report = build_model(initialization, config["seeds"]["model"])
        optimizer, optimizer_record = adamw(
            model, config["learning_rate"], config["weight_decay"]
        )
        if optimizer_record["learning_rates"] != [learning_rate, learning_rate]:
            raise AssertionError("Configured learning rate did not reach every optimizer group.")
        reports.append((report["encoder_sha256"], report["decoder_sha256"]))
    if len(set(reports)) != 1:
        raise AssertionError("Learning rates within an initialization do not share initial state.")
    initial_states[initialization] = reports[0]
if initial_states["random"][1] != initial_states["imagenet"][1]:
    raise AssertionError("Random and ImageNet decoders are not paired.")

smoke_config = build_learning_rate_config(splits, "random", 1e-3)
model, _ = build_model("random", smoke_config["seeds"]["model"])
model.to(device)
optimizer, _ = adamw(model, smoke_config["learning_rate"], smoke_config["weight_decay"])
store = RIDTensorStore(ROOT, manifest)
batch_ids = smoke_config["training_ids"][:4]
values = train_step(model, optimizer, store.batch(batch_ids, [0, 1, 2, 3]), device)
synchronize(device)
if not all(torch.isfinite(torch.tensor(value)).item() for value in values.values()):
    raise FloatingPointError("Learning-rate smoke step produced a non-finite value.")

report = {
    "device": str(device),
    "split_name": splits["split_name"],
    "training_size": 100,
    "learning_rates": list(LEARNING_RATES),
    "run_directories": [
        str(learning_rate_run_directory(
            ROOT, splits, config["initialization"], config["learning_rate"]
        ).relative_to(ROOT))
        for config in configs
    ],
    "locked_test_role_absent": True,
    "paired_schedule_sha256": next(iter(schedule_hashes)),
    "initial_states": initial_states,
    "optimizer_learning_rates_verified": True,
    "smoke": {"initialization": "random", "learning_rate": 1e-3,
              "sample_ids": batch_ids, **values},
}
(ROOT / "reports/optimization_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
