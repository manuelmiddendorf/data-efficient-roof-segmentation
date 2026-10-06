"""Run the two fresh n=25 fixed encoder BatchNorm-statistics configurations."""

from pathlib import Path
import json

import torch

from roofseg.audit import build_splits
from roofseg.batch_norm import (
    SUBSET_REPETITIONS,
    build_frozen_batch_norm_config,
    frozen_batch_norm_run_directory,
)
from roofseg.integrity import load_json
from roofseg.optimization import build_late_lr_drop_config, late_lr_drop_run_directory
from roofseg.pilot import build_metadata
from roofseg.run_artifacts import prepare_run, verify_completed_run
from roofseg.training import fit_run, select_device

ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active geographic split does not reproduce exactly from RID metadata.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"BatchNorm training requires MPS; selected {device}.")
print(f"Training device: {device}", flush=True)

for repetition in SUBSET_REPETITIONS:
    reference = build_late_lr_drop_config(splits, "imagenet", repetition, 25)
    reference_directory = late_lr_drop_run_directory(
        ROOT, splits, "imagenet", repetition, 25
    )
    if json.loads((reference_directory / "config.json").read_text()) != reference:
        raise SystemExit(f"Reference differs: {reference_directory.name}")
    verify_completed_run(reference_directory)

    config = build_frozen_batch_norm_config(splits, repetition)
    if set(config["training_ids"]) & (
        set(splits["roles"]["validation"]) | set(splits["roles"]["test"])
    ):
        raise SystemExit("Training inputs intersect validation or test roles.")
    metadata = build_metadata(ROOT, manifest, splits, config, device)
    directory = frozen_batch_norm_run_directory(ROOT, splits, repetition)
    if prepare_run(directory, config, metadata):
        verify_completed_run(directory)
        print(f"Reusing compatible completed run: {directory.name}", flush=True)
        continue
    fit_run(ROOT, config, metadata, manifest, device, directory)
    verify_completed_run(directory)
