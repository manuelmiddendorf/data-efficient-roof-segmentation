"""Verify the four fresh n=25 strong-weight-decay runs on MPS."""

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
from roofseg.training import adamw, select_device, synchronize, train_step
from roofseg.training_data import RIDTensorStore, paired_schedule
from roofseg.weight_decay import (
    REFERENCE_WEIGHT_DECAY,
    STRONG_WEIGHT_DECAY,
    SUBSET_REPETITIONS,
    build_weight_decay_config,
    shrinkage_factor,
    weight_decay_run_directory,
)

ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active geographic split does not reproduce exactly from RID metadata.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"Weight-decay training requires MPS; selected {device}.")

validation_ids = set(splits["roles"]["validation"])
test_ids = set(splits["roles"]["test"])
run_directories, schedule_hashes, initial_states, group_records = [], {}, {}, {}
for repetition in SUBSET_REPETITIONS:
    schedule_hashes[repetition] = set()
    for initialization in INITIALIZATIONS:
        reference = build_late_lr_drop_config(splits, initialization, repetition, 25)
        reference_directory = late_lr_drop_run_directory(
            ROOT, splits, initialization, repetition, 25
        )
        if json.loads((reference_directory / "config.json").read_text()) != reference:
            raise AssertionError(f"Reference configuration changed: {reference_directory.name}")
        verify_completed_run(reference_directory)

        config = build_weight_decay_config(splits, initialization, repetition)
        differences = {
            key for key in reference | config
            if reference.get(key) != config.get(key)
        }
        if differences != {"run_name", "experiment", "weight_decay"}:
            raise AssertionError(f"Unexpected strong-decay differences: {differences}")
        if config["weight_decay"] != STRONG_WEIGHT_DECAY:
            raise AssertionError("Strong weight decay was not set to 1e-2.")
        if config["weight_decay_exclusion"] != reference["weight_decay_exclusion"]:
            raise AssertionError("Weight-decay exclusions changed.")
        if config["training_ids"] != reference["training_ids"]:
            raise AssertionError("Training IDs changed.")
        if config["seeds"] != reference["seeds"]:
            raise AssertionError("A paired random seed changed.")
        if (set(config["training_ids"]) | set(config["validation_ids"])) & test_ids:
            raise AssertionError("Weight-decay inputs intersect the locked test role.")
        if set(config["validation_ids"]) != validation_ids:
            raise AssertionError("Validation role changed.")
        _, schedule = paired_schedule(
            config["training_ids"], config["max_steps"], config["batch_size"],
            config["seeds"]["data_order"], config["seeds"]["augmentation"],
        )
        schedule_hashes[repetition].add(schedule["sha256"])
        model, model_record = build_model(initialization, config["seeds"]["model"])
        optimizer, optimizer_record = adamw(
            model, config["learning_rate"], config["weight_decay"]
        )
        if optimizer_record["weight_decays"] != [STRONG_WEIGHT_DECAY, 0.0]:
            raise AssertionError("Optimizer groups have incorrect decay values.")
        decayed_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
        no_decay_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
        for name, parameter in model.named_parameters():
            expected_no_decay = parameter.ndim == 1 or name.endswith(".bias")
            if expected_no_decay != (id(parameter) in no_decay_ids):
                raise AssertionError(f"Incorrect decay group for {name}.")
            if not expected_no_decay and id(parameter) not in decayed_ids:
                raise AssertionError(f"Missing decayed parameter: {name}.")
        key = f"seed{config['seeds']['subset']}_{initialization}"
        initial_states[key] = {
            "encoder_sha256": model_record["encoder_sha256"],
            "decoder_sha256": model_record["decoder_sha256"],
        }
        group_records[key] = optimizer_record
        run_directories.append(weight_decay_run_directory(
            ROOT, splits, initialization, repetition
        ))
    if len(schedule_hashes[repetition]) != 1:
        raise AssertionError(f"Repetition {repetition} is not schedule-paired.")

for repetition in SUBSET_REPETITIONS:
    seed = splits["training_subsets"]["repetitions"][str(repetition)]["seed"]
    if initial_states[f"seed{seed}_random"]["decoder_sha256"] != (
        initial_states[f"seed{seed}_imagenet"]["decoder_sha256"]
    ):
        raise AssertionError(f"Subset seed {seed} does not share decoder initialization.")
if len(set(run_directories)) != 4 or any(path.exists() for path in run_directories):
    raise AssertionError("The four strong-decay directories are not unique and unused.")

# With a zero gradient, AdamW applies only its decoupled multiplicative decay.
toy = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.LayerNorm(2))
for parameter in toy.parameters():
    parameter.data.fill_(1.0)
    parameter.grad = torch.zeros_like(parameter)
toy_optimizer, _ = adamw(toy, 1e-3, STRONG_WEIGHT_DECAY)
toy_optimizer.step()
torch.testing.assert_close(
    toy[0].weight, torch.full_like(toy[0].weight, 1 - 1e-3 * STRONG_WEIGHT_DECAY)
)
torch.testing.assert_close(toy[0].bias, torch.ones_like(toy[0].bias))
torch.testing.assert_close(toy[1].weight, torch.ones_like(toy[1].weight))

# One real MPS forward/backward step checks the unchanged training path.
smoke_config = build_weight_decay_config(splits, "random", 1)
smoke_model, _ = build_model("random", smoke_config["seeds"]["model"])
smoke_model.to(device)
smoke_optimizer, _ = adamw(
    smoke_model, smoke_config["learning_rate"], smoke_config["weight_decay"]
)
store = RIDTensorStore(ROOT, manifest)
smoke_values = train_step(
    smoke_model, smoke_optimizer,
    store.batch(smoke_config["training_ids"][:4], [0, 1, 2, 3]), device,
)
synchronize(device)
if not all(torch.isfinite(torch.tensor(value)).item() for value in smoke_values.values()):
    raise FloatingPointError("Strong-decay MPS smoke step produced a non-finite value.")

reference_example = build_late_lr_drop_config(splits, "random", 1, 25)
strong_example = build_weight_decay_config(splits, "random", 1)
report = {
    "device": str(device),
    "split_name": splits["split_name"],
    "run_count": 4,
    "training_size": 25,
    "subset_repetitions": list(SUBSET_REPETITIONS),
    "subset_seeds": [17, 29],
    "reference_weight_decay": REFERENCE_WEIGHT_DECAY,
    "strong_weight_decay": STRONG_WEIGHT_DECAY,
    "reference_runs_verified": 4,
    "locked_test_role_absent": True,
    "paired_ids_seeds_and_schedules_verified": True,
    "optimizer_group_exclusions_verified": True,
    "toy_shrinkage_step_verified": True,
    "mps_forward_backward_verified": True,
    "illustrative_no_gradient_shrinkage": {
        "reference": shrinkage_factor(reference_example),
        "strong": shrinkage_factor(strong_example),
    },
    "run_directories": [str(path.relative_to(ROOT)) for path in run_directories],
    "schedule_sha256_by_repetition": {
        str(repetition): next(iter(hashes))
        for repetition, hashes in schedule_hashes.items()
    },
    "initial_states": initial_states,
    "optimizer_groups": group_records,
    "smoke": smoke_values,
}
(ROOT / "reports/weight_decay_n25_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
