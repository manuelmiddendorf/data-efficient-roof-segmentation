"""Run the six reference-loss and two Variant B Seed 43 configurations."""

from pathlib import Path
import json

import torch

from roofseg.audit import build_splits
from roofseg.boundary_weighted import (
    boundary_run_directory,
    build_boundary_config,
    training_band_fractions,
)
from roofseg.integrity import load_json
from roofseg.optimization import (
    INITIALIZATIONS,
    build_late_lr_drop_config,
    late_lr_drop_run_directory,
)
from roofseg.pilot import build_metadata
from roofseg.run_artifacts import prepare_run, verify_completed_run
from roofseg.training import fit_run, select_device
from roofseg.training_data import RIDTensorStore

ROOT = Path(__file__).resolve().parents[1]
REPETITION = 3
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active geographic split does not reproduce exactly.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"Seed 43 training requires MPS; selected {device}.")
print(f"Training device: {device}", flush=True)

def execute(config: dict, directory: Path) -> None:
    metadata = build_metadata(ROOT, manifest, splits, config, device)
    if prepare_run(directory, config, metadata):
        verify_completed_run(directory)
        print(f"Reusing compatible completed run: {directory.name}", flush=True)
        return
    fit_run(ROOT, config, metadata, manifest, device, directory)
    verify_completed_run(directory)


for training_size in (25, 100, 500):
    for initialization in INITIALIZATIONS:
        config = build_late_lr_drop_config(
            splits, initialization, REPETITION, training_size
        )
        execute(config, late_lr_drop_run_directory(
            ROOT, splits, initialization, REPETITION, training_size
        ))

training_ids = splits["training_subsets"]["repetitions"][str(REPETITION)]["subsets"]["25"]
fractions = training_band_fractions(RIDTensorStore(ROOT, manifest), training_ids)
mean_fraction = sum(fractions.values()) / len(fractions)
for initialization in INITIALIZATIONS:
    config = build_boundary_config(
        splits, initialization, REPETITION, "equal_per_image", mean_fraction
    )
    execute(config, boundary_run_directory(
        ROOT, splits, initialization, REPETITION, "equal_per_image"
    ))
