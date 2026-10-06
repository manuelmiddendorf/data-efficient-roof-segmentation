"""Verify the fixed mild brightness/contrast block and run one real MPS step."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import torch

from roofseg.audit import build_splits
from roofseg.integrity import load_json
from roofseg.model import build_model
from roofseg.optimization import INITIALIZATIONS, build_late_lr_drop_config, late_lr_drop_run_directory
from roofseg.photometric_augmentation import (
    FACTOR_MAX,
    FACTOR_MIN,
    PHOTOMETRIC_SEED,
    SUBSET_REPETITIONS,
    build_photometric_config,
    photometric_run_directory,
    plot_photometric_examples,
)
from roofseg.run_artifacts import verify_completed_run
from roofseg.training import adamw, select_device, synchronize, train_step
from roofseg.training_data import RIDTensorStore, paired_schedule, photometric_schedule

ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active geographic split does not reproduce exactly from RID metadata.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"Photometric training requires MPS; selected {device}.")

validation_ids = set(splits["roles"]["validation"])
test_ids = set(splits["roles"]["test"])
run_directories, schedule_hashes, factor_hashes = [], {}, {}
factor_examples = {}
for repetition in SUBSET_REPETITIONS:
    schedule_hashes[repetition], factor_hashes[repetition] = set(), set()
    for initialization in INITIALIZATIONS:
        reference = build_late_lr_drop_config(splits, initialization, repetition, 25)
        reference_directory = late_lr_drop_run_directory(ROOT, splits, initialization, repetition, 25)
        if json.loads((reference_directory / "config.json").read_text()) != reference:
            raise AssertionError(f"Reference configuration changed: {reference_directory.name}")
        reference_summary = verify_completed_run(reference_directory)
        if reference_summary["optimizer_groups"]["weight_decays"] != [1e-4, 0.0]:
            raise AssertionError("A D4 reference does not use baseline weight decay.")

        config = build_photometric_config(splits, initialization, repetition)
        differences = {key for key in reference | config if reference.get(key) != config.get(key)}
        if differences != {
            "run_name", "experiment", "augmentation", "photometric_augmentation", "seeds"
        }:
            raise AssertionError(f"Unexpected photometric differences: {differences}")
        if {key: value for key, value in config["seeds"].items() if key != "photometric"} != reference["seeds"]:
            raise AssertionError("A pre-existing random seed changed.")
        if config["training_ids"] != reference["training_ids"]:
            raise AssertionError("Training IDs changed.")
        if (set(config["training_ids"]) | set(config["validation_ids"])) & test_ids:
            raise AssertionError("Photometric inputs intersect the locked test role.")
        if set(config["validation_ids"]) != validation_ids:
            raise AssertionError("Validation role changed.")
        schedule, schedule_record = paired_schedule(
            config["training_ids"], config["max_steps"], config["batch_size"],
            config["seeds"]["data_order"], config["seeds"]["augmentation"],
        )
        reference_schedule_hash = json.loads(
            (reference_directory / "metadata.json").read_text()
        )["compatibility_identity"]["schedule_sha256"]
        if schedule_record["sha256"] != reference_schedule_hash:
            raise AssertionError("Existing ID/D4 schedule changed.")
        factors, factor_record = photometric_schedule(
            schedule, PHOTOMETRIC_SEED, FACTOR_MIN, FACTOR_MAX
        )
        schedule_hashes[repetition].add(schedule_record["sha256"])
        factor_hashes[repetition].add(factor_record["sha256"])
        factor_examples[str(repetition)] = factors[0]
        run_directories.append(photometric_run_directory(
            ROOT, splits, initialization, repetition
        ))
    if len(schedule_hashes[repetition]) != 1 or len(factor_hashes[repetition]) != 1:
        raise AssertionError(f"Repetition {repetition} is not fully paired.")

if len(set(run_directories)) != 4 or any(path.exists() for path in run_directories):
    raise AssertionError("The four photometric run directories are not unique and unused.")

store = RIDTensorStore(ROOT, manifest)
fixed_ids = splits["training_subsets"]["repetitions"]["1"]["subsets"]["25"][:2]
preview = plot_photometric_examples(ROOT, manifest, fixed_ids)
preview_path = ROOT / "reports/figures/photometric_n25_examples.png"
preview.savefig(preview_path, dpi=170, bbox_inches="tight")
plt.close(preview)

# Evaluation remains the exact original-orientation, augmentation-free tensor path.
eval_batch = store.batch(fixed_ids)
manual_images = torch.stack([store.load(sample_id)[0] for sample_id in fixed_ids])
manual_targets = torch.stack([store.load(sample_id)[1] for sample_id in fixed_ids])
torch.testing.assert_close(eval_batch["image"], manual_images, rtol=0, atol=0)
torch.testing.assert_close(eval_batch["target"], manual_targets, rtol=0, atol=0)

smoke_config = build_photometric_config(splits, "random", 1)
smoke_schedule, _ = paired_schedule(
    smoke_config["training_ids"], smoke_config["max_steps"], smoke_config["batch_size"],
    smoke_config["seeds"]["data_order"], smoke_config["seeds"]["augmentation"],
)
smoke_factors, _ = photometric_schedule(smoke_schedule, PHOTOMETRIC_SEED)
smoke_ids, smoke_d4 = smoke_schedule[0]
plain_targets = store.batch(smoke_ids, smoke_d4)["target"]
augmented_batch = store.batch(smoke_ids, smoke_d4, smoke_factors[0])
torch.testing.assert_close(augmented_batch["target"], plain_targets, rtol=0, atol=0)

smoke_model, _ = build_model("random", smoke_config["seeds"]["model"])
smoke_model.to(device)
smoke_optimizer, _ = adamw(
    smoke_model, smoke_config["learning_rate"], smoke_config["weight_decay"]
)
smoke_values = train_step(smoke_model, smoke_optimizer, augmented_batch, device)
synchronize(device)
if not all(torch.isfinite(torch.tensor(value)).item() for value in smoke_values.values()):
    raise FloatingPointError("Photometric MPS smoke step produced a non-finite value.")

report = {
    "device": str(device),
    "split_name": splits["split_name"],
    "run_count": 4,
    "training_size": 25,
    "subset_repetitions": list(SUBSET_REPETITIONS),
    "subset_seeds": [17, 29],
    "photometric_seed": PHOTOMETRIC_SEED,
    "factor_bounds": [FACTOR_MIN, FACTOR_MAX],
    "transform_order": ["brightness", "contrast", "D4", "ImageNet normalization"],
    "application": "every training-image occurrence; disabled for evaluation",
    "reference_runs_verified": 4,
    "locked_test_role_absent": True,
    "paired_ids_existing_seeds_d4_and_factors_verified": True,
    "targets_unchanged": True,
    "evaluation_path_unchanged": True,
    "mps_forward_backward_verified": True,
    "preview_ids": fixed_ids,
    "preview_path": str(preview_path.relative_to(ROOT)),
    "run_directories": [str(path.relative_to(ROOT)) for path in run_directories],
    "schedule_sha256_by_repetition": {
        str(key): next(iter(value)) for key, value in schedule_hashes.items()
    },
    "photometric_schedule_sha256_by_repetition": {
        str(key): next(iter(value)) for key, value in factor_hashes.items()
    },
    "first_batch_factors_by_repetition": factor_examples,
    "smoke": smoke_values,
}
(ROOT / "reports/photometric_n25_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
