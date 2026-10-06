"""Verify isolated decoder channel dropout and run one real MPS step."""

from pathlib import Path
import json
import tempfile

import torch

from roofseg.audit import build_splits
from roofseg.decoder_dropout import (
    DROPOUT_PROBABILITY,
    DROPOUT_SEED,
    SUBSET_REPETITIONS,
    build_decoder_dropout_config,
    decoder_dropout_run_directory,
)
from roofseg.integrity import load_json
from roofseg.model import SeededChannelDropout2d, build_model
from roofseg.optimization import INITIALIZATIONS, build_late_lr_drop_config, late_lr_drop_run_directory
from roofseg.pilot import build_metadata
from roofseg.run_artifacts import verify_completed_run
from roofseg.training import adamw, restore_checkpoint, select_device, synchronize, train_step
from roofseg.training_data import RIDTensorStore, paired_schedule

ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active geographic split does not reproduce exactly from RID metadata.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"Decoder-dropout training requires MPS; selected {device}.")

test_ids = set(splits["roles"]["test"])
validation_ids = set(splits["roles"]["validation"])
run_directories, schedule_hashes, initial_states = [], {}, {}
for repetition in SUBSET_REPETITIONS:
    schedule_hashes[repetition] = set()
    for initialization in INITIALIZATIONS:
        reference = build_late_lr_drop_config(splits, initialization, repetition, 25)
        reference_directory = late_lr_drop_run_directory(
            ROOT, splits, initialization, repetition, 25
        )
        if json.loads((reference_directory / "config.json").read_text()) != reference:
            raise AssertionError(f"Reference configuration changed: {reference_directory.name}")
        reference_summary = verify_completed_run(reference_directory)

        config = build_decoder_dropout_config(splits, initialization, repetition)
        differences = {key for key in reference | config if reference.get(key) != config.get(key)}
        if differences != {"run_name", "experiment", "decoder_channel_dropout", "seeds"}:
            raise AssertionError(f"Unexpected dropout differences: {differences}")
        if {key: value for key, value in config["seeds"].items() if key != "decoder_dropout"} != reference["seeds"]:
            raise AssertionError("A pre-existing random seed changed.")
        if config["training_ids"] != reference["training_ids"]:
            raise AssertionError("Training IDs changed.")
        if config.get("photometric_augmentation") is not None:
            raise AssertionError("The dropout experiment must not use colour augmentation.")
        if (set(config["training_ids"]) | set(config["validation_ids"])) & test_ids:
            raise AssertionError("Dropout inputs intersect the locked test role.")
        if set(config["validation_ids"]) != validation_ids:
            raise AssertionError("Validation role changed.")
        _, schedule_record = paired_schedule(
            config["training_ids"], config["max_steps"], config["batch_size"],
            config["seeds"]["data_order"], config["seeds"]["augmentation"],
        )
        reference_metadata = json.loads((reference_directory / "metadata.json").read_text())
        if schedule_record["sha256"] != reference_metadata["compatibility_identity"]["schedule_sha256"]:
            raise AssertionError("Existing ID/D4 schedule changed.")
        schedule_hashes[repetition].add(schedule_record["sha256"])
        model, model_record = build_model(
            initialization, config["seeds"]["model"],
            DROPOUT_PROBABILITY, DROPOUT_SEED,
        )
        if model_record["encoder_sha256"] != reference_summary["initialization"]["encoder_sha256"]:
            raise AssertionError("Fresh encoder differs from its reference initialization.")
        if model_record["decoder_sha256"] != reference_summary["initialization"]["decoder_sha256"]:
            raise AssertionError("Fresh decoder differs from its paired reference initialization.")
        initial_states[f"seed{config['seeds']['subset']}_{initialization}"] = {
            "encoder_sha256": model_record["encoder_sha256"],
            "decoder_sha256": model_record["decoder_sha256"],
        }
        run_directories.append(decoder_dropout_run_directory(
            ROOT, splits, initialization, repetition
        ))
    if len(schedule_hashes[repetition]) != 1:
        raise AssertionError(f"Repetition {repetition} is not schedule-paired.")

if len(set(run_directories)) != 4 or any(path.exists() for path in run_directories):
    raise AssertionError("The four dropout run directories are not unique and unused.")

# Direct numerical checks use the same isolated generator implementation as training.
dropout = SeededChannelDropout2d(DROPOUT_PROBABILITY, DROPOUT_SEED)
dropout.train()
features = torch.ones(8, 16, 4, 4)
global_state_before = torch.random.get_rng_state().clone()
dropped = dropout(features)
torch.testing.assert_close(torch.random.get_rng_state(), global_state_before, rtol=0, atol=0)
channel_values = dropped[:, :, 0, 0]
if not torch.all((channel_values == 0) | torch.isclose(
    channel_values, torch.full_like(channel_values, 1 / 0.9)
)):
    raise AssertionError("Dropout values are neither zero nor correctly scaled.")
if not torch.all(dropped == channel_values[:, :, None, None]):
    raise AssertionError("Dropout did not apply one mask value per feature channel.")
dropout.eval()
torch.testing.assert_close(dropout(features), features, rtol=0, atol=0)
disabled = SeededChannelDropout2d(0.0)
disabled.train()
torch.testing.assert_close(disabled(features), features, rtol=0, atol=0)

# One real MPS step verifies the production model, dropout position and backward path.
config = build_decoder_dropout_config(splits, "random", 1)
metadata = build_metadata(ROOT, manifest, splits, config, device)
schedule, _ = paired_schedule(
    config["training_ids"], config["max_steps"], config["batch_size"],
    config["seeds"]["data_order"], config["seeds"]["augmentation"],
)
model, _ = build_model(
    "random", config["seeds"]["model"], DROPOUT_PROBABILITY, DROPOUT_SEED
)
model.to(device)
optimizer, _ = adamw(model, config["learning_rate"], config["weight_decay"])
store = RIDTensorStore(ROOT, manifest)
batch_ids, d4_codes = schedule[0]
smoke_values = train_step(model, optimizer, store.batch(batch_ids, d4_codes), device)
synchronize(device)
if not all(torch.isfinite(torch.tensor(value)).item() for value in smoke_values.values()):
    raise FloatingPointError("Dropout MPS smoke step produced a non-finite value.")

# Evaluation is identical to a no-dropout model with the same state, and strict restore works.
model.eval()
reference_model, _ = build_model("random", config["seeds"]["model"])
reference_model.load_state_dict(model.state_dict(), strict=True)
reference_model.to(device).eval()
evaluation_batch = store.batch(batch_ids)["image"].to(device)
with torch.inference_mode():
    torch.testing.assert_close(model(evaluation_batch), reference_model(evaluation_batch), rtol=0, atol=0)
with tempfile.TemporaryDirectory() as temporary_directory:
    checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
    torch.save({
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "step": 1,
        "config_sha256": metadata["compatibility_identity"]["config_sha256"],
        "input_sha256": metadata["compatibility_identity"]["input_sha256"],
    }, checkpoint_path)
    restored, _ = build_model("random", config["seeds"]["model"], DROPOUT_PROBABILITY, DROPOUT_SEED)
    if restore_checkpoint(
        restored, checkpoint_path,
        metadata["compatibility_identity"]["config_sha256"],
        metadata["compatibility_identity"]["input_sha256"],
    ) != 1:
        raise AssertionError("Checkpoint restored the wrong step.")

report = {
    "device": str(device),
    "split_name": splits["split_name"],
    "run_count": 4,
    "training_size": 25,
    "subset_repetitions": list(SUBSET_REPETITIONS),
    "subset_seeds": [17, 29],
    "initializations": list(INITIALIZATIONS),
    "dropout_probability": DROPOUT_PROBABILITY,
    "dropout_seed": DROPOUT_SEED,
    "position": "after up5 and before the final 1x1 convolution",
    "mask_shape": "N×C×1×1",
    "reference_runs_verified": 4,
    "locked_test_role_absent": True,
    "training_ids_existing_seeds_and_schedules_verified": True,
    "paired_initial_states_verified": True,
    "channel_mask_and_scaling_verified": True,
    "evaluation_and_disabled_identity_verified": True,
    "global_rng_unchanged_by_dropout": True,
    "state_dict_schema_compatible": True,
    "checkpoint_restore_and_config_identity_verified": True,
    "mps_forward_backward_verified": True,
    "run_directories": [str(path.relative_to(ROOT)) for path in run_directories],
    "schedule_sha256_by_repetition": {
        str(key): next(iter(value)) for key, value in schedule_hashes.items()
    },
    "initial_states": initial_states,
    "smoke": smoke_values,
}
(ROOT / "reports/decoder_dropout_n25_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
