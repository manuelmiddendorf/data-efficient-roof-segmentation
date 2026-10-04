"""Verify the two-run fixed late learning-rate-drop block on MPS."""

from pathlib import Path
import json

import torch

from roofseg.audit import build_splits
from roofseg.integrity import load_json
from roofseg.model import build_model
from roofseg.optimization import (
    INITIALIZATIONS,
    build_late_lr_drop_config,
    build_training_horizon_config,
    late_lr_drop_run_directory,
    training_horizon_run_directory,
)
from roofseg.pilot import build_metadata
from roofseg.run_artifacts import verify_completed_run
from roofseg.training import (
    adamw,
    apply_learning_rate,
    learning_rate_for_step,
    select_device,
    synchronize,
    train_step,
)
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

test_ids = set(splits["roles"]["test"])
configs = [build_late_lr_drop_config(splits, initialization) for initialization in INITIALIZATIONS]
reference_schedule_hashes, new_schedule_hashes, initial_states = set(), set(), {}
for config in configs:
    reference = build_training_horizon_config(splits, config["initialization"], 1e-3)
    differences = {key for key in reference | config if reference.get(key) != config.get(key)}
    if differences != {"run_name", "experiment", "scheduler"}:
        raise AssertionError(f"Unexpected fixed-drop configuration differences: {differences}")
    if (set(config["training_ids"]) | set(config["validation_ids"])) & test_ids:
        raise AssertionError("Fixed-drop inputs intersect the locked test role.")
    reference_directory = training_horizon_run_directory(
        ROOT, splits, config["initialization"], 1e-3
    )
    verify_completed_run(reference_directory)
    if reference_directory == late_lr_drop_run_directory(ROOT, splits, config["initialization"]):
        raise AssertionError("Fixed-drop and reference runs share a directory.")
    for candidate, hashes in ((reference, reference_schedule_hashes), (config, new_schedule_hashes)):
        _, record = paired_schedule(
            candidate["training_ids"], candidate["max_steps"], candidate["batch_size"],
            candidate["seeds"]["data_order"], candidate["seeds"]["augmentation"],
        )
        hashes.add(record["sha256"])
    model, report = build_model(config["initialization"], config["seeds"]["model"])
    initial_states[config["initialization"]] = (
        report["encoder_sha256"], report["decoder_sha256"]
    )
    optimizer, optimizer_record = adamw(model, config["learning_rate"], config["weight_decay"])
    weight_decays = list(optimizer_record["weight_decays"])
    apply_learning_rate(optimizer, learning_rate_for_step(config, 2000))
    rates_2000 = [group["lr"] for group in optimizer.param_groups]
    apply_learning_rate(optimizer, learning_rate_for_step(config, 2001))
    rates_2001 = [group["lr"] for group in optimizer.param_groups]
    if rates_2000 != [1e-3, 1e-3] or rates_2001 != [1e-4, 1e-4]:
        raise AssertionError("The fixed learning-rate boundary is incorrect.")
    if [group["weight_decay"] for group in optimizer.param_groups] != weight_decays:
        raise AssertionError("The learning-rate change altered weight decay.")

if reference_schedule_hashes != new_schedule_hashes or len(new_schedule_hashes) != 1:
    raise AssertionError("The paired 4,000-step data and augmentation schedule changed.")
if initial_states["random"][1] != initial_states["imagenet"][1]:
    raise AssertionError("Random and ImageNet runs do not share the paired decoder state.")

smoke_config = configs[0]
model, _ = build_model("random", smoke_config["seeds"]["model"])
model.to(device)
optimizer, optimizer_record = adamw(
    model, smoke_config["learning_rate"], smoke_config["weight_decay"]
)
batch_ids = smoke_config["training_ids"][:4]
store = RIDTensorStore(ROOT, manifest)
smoke = []
for step in (2000, 2001):
    rate = learning_rate_for_step(smoke_config, step)
    apply_learning_rate(optimizer, rate)
    values = train_step(model, optimizer, store.batch(batch_ids, [0, 1, 2, 3]), device)
    synchronize(device)
    if not all(torch.isfinite(torch.tensor(value)).item() for value in values.values()):
        raise FloatingPointError("Fixed-drop smoke step produced a non-finite value.")
    smoke.append({"nominal_step": step, "learning_rate": rate, **values})

report = {
    "device": str(device),
    "split_name": splits["split_name"],
    "run_count": 2,
    "training_size": 100,
    "max_steps": 4000,
    "evaluation_steps": [0] + list(range(100, 4001, 100)),
    "learning_rate_schedule": configs[0]["scheduler"],
    "run_directories": [
        str(late_lr_drop_run_directory(ROOT, splits, config["initialization"]).relative_to(ROOT))
        for config in configs
    ],
    "locked_test_role_absent": True,
    "constant_references_verified": True,
    "paired_schedule_sha256": next(iter(new_schedule_hashes)),
    "initial_states": initial_states,
    "optimizer_boundary_verified": True,
    "weight_decay_groups_unchanged": True,
    "estimated_total_minutes": [55, 70],
    "smoke": smoke,
}
(ROOT / "reports/late_lr_drop_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
