"""Verify the fixed boundary-weighted BCE block and run real MPS steps."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import torch

from roofseg.audit import build_splits
from roofseg.boundary_weighted import (
    BOUNDARY_RADIUS,
    SUBSET_REPETITIONS,
    VARIANTS,
    boundary_run_directory,
    build_boundary_config,
    choose_band_examples,
    plot_band_examples,
    training_band_fractions,
)
from roofseg.integrity import load_json
from roofseg.model import build_model
from roofseg.optimization import INITIALIZATIONS, build_late_lr_drop_config, late_lr_drop_run_directory
from roofseg.pilot import build_metadata
from roofseg.run_artifacts import verify_completed_run
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
    raise SystemExit(f"Boundary-weighted training requires MPS; selected {device}.")
store = RIDTensorStore(ROOT, manifest)
test_ids = set(splits["roles"]["test"])
validation_ids = set(splits["roles"]["validation"])

band_records, run_directories, schedule_hashes, initial_states = {}, [], {}, {}
for repetition in SUBSET_REPETITIONS:
    repetition_record = splits["training_subsets"]["repetitions"][str(repetition)]
    training_ids = repetition_record["subsets"]["25"]
    subset_seed = int(repetition_record["seed"])
    fractions = training_band_fractions(store, training_ids)
    mean_fraction = sum(fractions.values()) / len(fractions)
    band_records[str(subset_seed)] = {
        "mean_training_band_fraction": mean_fraction,
        "minimum": min(fractions.values()),
        "median": sorted(fractions.values())[len(fractions) // 2],
        "maximum": max(fractions.values()),
        "empty_band_count": sum(value == 0 for value in fractions.values()),
        "per_image": fractions,
    }
    schedule_hashes[repetition] = set()
    for initialization in INITIALIZATIONS:
        reference = build_late_lr_drop_config(splits, initialization, repetition, 25)
        reference_directory = late_lr_drop_run_directory(ROOT, splits, initialization, repetition, 25)
        if json.loads((reference_directory / "config.json").read_text()) != reference:
            raise AssertionError(f"Reference configuration changed: {reference_directory.name}")
        reference_summary = verify_completed_run(reference_directory)
        for variant in VARIANTS:
            config = build_boundary_config(
                splits, initialization, repetition, variant, mean_fraction
            )
            differences = {key for key in reference | config if reference.get(key) != config.get(key)}
            expected = {"run_name", "experiment", "loss", "boundary_weighted_bce"}
            if differences != expected:
                raise AssertionError(f"Unexpected boundary differences: {differences}")
            if config["training_ids"] != reference["training_ids"] or config["seeds"] != reference["seeds"]:
                raise AssertionError("Training IDs or random seeds changed.")
            if config.get("decoder_channel_dropout") is not None or config.get("photometric_augmentation") is not None:
                raise AssertionError("Boundary runs must not use dropout or colour augmentation.")
            if (set(config["training_ids"]) | set(config["validation_ids"])) & test_ids:
                raise AssertionError("Boundary inputs intersect the locked test role.")
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
            model, model_record = build_model(initialization, config["seeds"]["model"])
            if model_record["encoder_sha256"] != reference_summary["initialization"]["encoder_sha256"]:
                raise AssertionError("Fresh encoder differs from the paired reference.")
            if model_record["decoder_sha256"] != reference_summary["initialization"]["decoder_sha256"]:
                raise AssertionError("Fresh decoder differs from the paired reference.")
            initial_states[f"seed{subset_seed}_{initialization}"] = {
                "encoder_sha256": model_record["encoder_sha256"],
                "decoder_sha256": model_record["decoder_sha256"],
            }
            run_directories.append(boundary_run_directory(
                ROOT, splits, initialization, repetition, variant
            ))
    if len(schedule_hashes[repetition]) != 1:
        raise AssertionError(f"Repetition {repetition} is not schedule-paired.")

if len(set(run_directories)) != 8 or any(path.exists() for path in run_directories):
    raise AssertionError("The eight boundary run directories are not unique and unused.")

# A report figure is created before training from training masks only.
all_training_ids = list(dict.fromkeys(
    sample_id
    for repetition in SUBSET_REPETITIONS
    for sample_id in splits["training_subsets"]["repetitions"][str(repetition)]["subsets"]["25"]
))
example_ids = choose_band_examples(store, all_training_ids)
figure = plot_band_examples(ROOT, manifest, example_ids)
figure_path = ROOT / "reports/figures/boundary_weighted_band_examples.png"
figure.savefig(figure_path, dpi=170, bbox_inches="tight")
plt.close(figure)

# One genuine MPS update per formula checks the production forward/backward path.
smoke = {}
for variant in VARIANTS:
    repetition = 1
    seed_record = band_records["17"]
    config = build_boundary_config(
        splits, "random", repetition, variant, seed_record["mean_training_band_fraction"]
    )
    metadata = build_metadata(ROOT, manifest, splits, config, device)
    schedule, _ = paired_schedule(
        config["training_ids"], config["max_steps"], config["batch_size"],
        config["seeds"]["data_order"], config["seeds"]["augmentation"],
    )
    model, _ = build_model("random", config["seeds"]["model"])
    model.to(device)
    optimizer, _ = adamw(model, config["learning_rate"], config["weight_decay"])
    batch_ids, d4_codes = schedule[0]
    values = train_step(
        model, optimizer, store.batch(batch_ids, d4_codes), device,
        boundary_objective=config["boundary_weighted_bce"],
    )
    synchronize(device)
    if not all(torch.isfinite(torch.tensor(value)).item() for value in values.values()):
        raise FloatingPointError("Boundary MPS smoke step produced a non-finite value.")
    if not (values["loss"] >= values["base_loss"] and values["boundary_addition"] >= 0):
        raise AssertionError("Boundary objective components have implausible magnitudes.")
    smoke[variant] = values

report = {
    "device": str(device), "split_name": splits["split_name"], "run_count": 8,
    "training_size": 25, "subset_repetitions": list(SUBSET_REPETITIONS),
    "subset_seeds": [17, 29], "initializations": list(INITIALIZATIONS),
    "variants": list(VARIANTS), "radius_pixels": BOUNDARY_RADIUS,
    "structuring_element": "square 7x7", "excluded_crop_border_pixels": 3,
    "reference_runs_verified": 4, "locked_test_role_absent": True,
    "training_ids_existing_seeds_and_schedules_verified": True,
    "paired_initial_states_verified": True, "mps_forward_backward_verified": True,
    "band_statistics": band_records, "band_example_ids": example_ids,
    "band_figure": str(figure_path.relative_to(ROOT)),
    "run_directories": [str(path.relative_to(ROOT)) for path in run_directories],
    "schedule_sha256_by_repetition": {
        str(key): next(iter(value)) for key, value in schedule_hashes.items()
    },
    "initial_states": initial_states, "smoke": smoke,
}
(ROOT / "reports/boundary_weighted_n25_preflight.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
