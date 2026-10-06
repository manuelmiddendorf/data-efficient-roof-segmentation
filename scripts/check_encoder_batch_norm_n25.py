"""Verify fixed encoder BatchNorm statistics and run a short real MPS check."""

from pathlib import Path
import json
import tempfile

import torch
from torch import nn
from torchvision.ops import StochasticDepth

from roofseg.audit import build_splits
from roofseg.batch_norm import (
    SUBSET_REPETITIONS,
    build_frozen_batch_norm_config,
    frozen_batch_norm_run_directory,
)
from roofseg.integrity import load_json
from roofseg.model import build_model
from roofseg.optimization import build_late_lr_drop_config, late_lr_drop_run_directory
from roofseg.pilot import build_metadata
from roofseg.run_artifacts import verify_completed_run
from roofseg.training import (
    FROZEN_ENCODER_BATCH_NORM,
    adamw,
    evaluate,
    restore_checkpoint,
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
    raise SystemExit(f"BatchNorm training requires MPS; selected {device}.")

test_ids = set(splits["roles"]["test"])
validation_ids = set(splits["roles"]["validation"])
run_directories, schedule_hashes, initial_states = [], {}, {}
for repetition in SUBSET_REPETITIONS:
    reference = build_late_lr_drop_config(splits, "imagenet", repetition, 25)
    reference_directory = late_lr_drop_run_directory(
        ROOT, splits, "imagenet", repetition, 25
    )
    if json.loads((reference_directory / "config.json").read_text()) != reference:
        raise AssertionError(f"Reference configuration changed: {reference_directory.name}")
    reference_summary = verify_completed_run(reference_directory)

    config = build_frozen_batch_norm_config(splits, repetition)
    differences = {key for key in reference | config if reference.get(key) != config.get(key)}
    if differences != {
        "run_name", "experiment", "batch_normalization", "batch_normalization_strategy"
    }:
        raise AssertionError(f"Unexpected BatchNorm differences: {differences}")
    if config["training_ids"] != reference["training_ids"] or config["seeds"] != reference["seeds"]:
        raise AssertionError("Training IDs or paired seeds changed.")
    if config.get("photometric_augmentation") is not None:
        raise AssertionError("The BatchNorm experiment must not use colour augmentation.")
    if (set(config["training_ids"]) | set(config["validation_ids"])) & test_ids:
        raise AssertionError("BatchNorm inputs intersect the locked test role.")
    if set(config["validation_ids"]) != validation_ids:
        raise AssertionError("Validation role changed.")
    _, schedule_record = paired_schedule(
        config["training_ids"], config["max_steps"], config["batch_size"],
        config["seeds"]["data_order"], config["seeds"]["augmentation"],
    )
    reference_metadata = json.loads((reference_directory / "metadata.json").read_text())
    if schedule_record["sha256"] != reference_metadata["compatibility_identity"]["schedule_sha256"]:
        raise AssertionError("Existing ID/D4 schedule changed.")
    schedule_hashes[str(repetition)] = schedule_record["sha256"]
    model, model_record = build_model("imagenet", config["seeds"]["model"])
    if model_record["encoder_sha256"] != reference_summary["initialization"]["encoder_sha256"]:
        raise AssertionError("Fresh pretrained encoder differs from its reference initialization.")
    if model_record["decoder_sha256"] != reference_summary["initialization"]["decoder_sha256"]:
        raise AssertionError("Fresh decoder differs from its paired reference initialization.")
    initial_states[str(repetition)] = {
        "encoder_sha256": model_record["encoder_sha256"],
        "decoder_sha256": model_record["decoder_sha256"],
    }
    run_directories.append(frozen_batch_norm_run_directory(ROOT, splits, repetition))

if len(set(run_directories)) != 2 or any(path.exists() for path in run_directories):
    raise AssertionError("The two BatchNorm run directories are not unique and unused.")

# Real train -> validation -> train MPS check on the production model and data path.
config = build_frozen_batch_norm_config(splits, 1)
metadata = build_metadata(ROOT, manifest, splits, config, device)
schedule, _ = paired_schedule(
    config["training_ids"], config["max_steps"], config["batch_size"],
    config["seeds"]["data_order"], config["seeds"]["augmentation"],
)
model, _ = build_model("imagenet", config["seeds"]["model"])
model.to(device)
optimizer, _ = adamw(model, config["learning_rate"], config["weight_decay"])
store = RIDTensorStore(ROOT, manifest)
encoder_batch_norms = [
    module for module in model.encoder.modules() if isinstance(module, nn.BatchNorm2d)
]
decoder_batch_norms = [
    module for name, module in model.named_modules()
    if not name.startswith("encoder") and isinstance(module, nn.BatchNorm2d)
]
stochastic_depths = [
    module for module in model.encoder.modules() if isinstance(module, StochasticDepth)
]
if not encoder_batch_norms or not decoder_batch_norms or not stochastic_depths:
    raise AssertionError("Expected encoder/decoder BatchNorm and encoder stochastic depth.")
encoder_buffers_before = [
    (module.running_mean.detach().cpu().clone(), module.running_var.detach().cpu().clone(),
     module.num_batches_tracked.detach().cpu().clone())
    for module in encoder_batch_norms
]
decoder_counts_before = [module.num_batches_tracked.detach().cpu().clone() for module in decoder_batch_norms]
encoder_parameter = next(model.encoder[0][0].parameters())
encoder_parameter_before = encoder_parameter.detach().cpu().clone()
affine_parameter = encoder_batch_norms[0].weight
affine_parameter_before = affine_parameter.detach().cpu().clone()

smoke_values = []
for schedule_index in (0, 1):
    batch_ids, d4_codes = schedule[schedule_index]
    smoke_values.append(train_step(
        model, optimizer, store.batch(batch_ids, d4_codes), device,
        FROZEN_ENCODER_BATCH_NORM,
    ))
    synchronize(device)
    if schedule_index == 0:
        evaluate(model, store, config["validation_ids"][:4], 4, device)

for before, module in zip(encoder_buffers_before, encoder_batch_norms):
    for expected, observed in zip(before, (
        module.running_mean.detach().cpu(), module.running_var.detach().cpu(),
        module.num_batches_tracked.detach().cpu(),
    )):
        torch.testing.assert_close(expected, observed, rtol=0, atol=0)
if not all(after.item() > before.item() for before, after in zip(
    decoder_counts_before,
    [module.num_batches_tracked.detach().cpu() for module in decoder_batch_norms],
)):
    raise AssertionError("Decoder BatchNorm statistics did not update on both training steps.")
if any(module.training for module in encoder_batch_norms):
    raise AssertionError("Encoder BatchNorm returned to training mode after validation.")
if not all(module.training for module in decoder_batch_norms + stochastic_depths):
    raise AssertionError("Decoder BatchNorm or stochastic depth left training mode.")
if affine_parameter.grad is None or encoder_parameter.grad is None:
    raise AssertionError("Encoder BatchNorm affine or encoder weight lacks a gradient.")
if torch.equal(affine_parameter_before, affine_parameter.detach().cpu()):
    raise AssertionError("Encoder BatchNorm affine parameter did not update.")
if torch.equal(encoder_parameter_before, encoder_parameter.detach().cpu()):
    raise AssertionError("Encoder convolution parameter did not update.")

with tempfile.TemporaryDirectory() as temporary_directory:
    checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
    torch.save({
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "step": 2,
        "config_sha256": metadata["compatibility_identity"]["config_sha256"],
        "input_sha256": metadata["compatibility_identity"]["input_sha256"],
    }, checkpoint_path)
    restored, _ = build_model("imagenet", config["seeds"]["model"])
    restored_step = restore_checkpoint(
        restored, checkpoint_path,
        metadata["compatibility_identity"]["config_sha256"],
        metadata["compatibility_identity"]["input_sha256"],
    )
    if restored_step != 2:
        raise AssertionError("Checkpoint restored the wrong step.")

report = {
    "device": str(device),
    "split_name": splits["split_name"],
    "run_count": 2,
    "training_size": 25,
    "subset_repetitions": list(SUBSET_REPETITIONS),
    "subset_seeds": [17, 29],
    "initialization": "imagenet",
    "strategy": FROZEN_ENCODER_BATCH_NORM,
    "reference_runs_verified": 2,
    "locked_test_role_absent": True,
    "training_ids_seeds_and_schedules_verified": True,
    "original_pretrained_encoder_and_fresh_decoder_verified": True,
    "encoder_batch_norm_count": len(encoder_batch_norms),
    "decoder_batch_norm_count": len(decoder_batch_norms),
    "encoder_buffers_unchanged_over_two_steps": True,
    "decoder_buffers_updated": True,
    "encoder_and_affine_gradients_and_updates_verified": True,
    "train_validation_train_modes_verified": True,
    "stochastic_depth_remained_in_training_mode": True,
    "checkpoint_restore_and_config_identity_verified": True,
    "mps_forward_backward_steps_verified": 2,
    "run_directories": [str(path.relative_to(ROOT)) for path in run_directories],
    "schedule_sha256_by_repetition": schedule_hashes,
    "initial_states": initial_states,
    "smoke": smoke_values,
}
(ROOT / "reports/encoder_batch_norm_n25_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
