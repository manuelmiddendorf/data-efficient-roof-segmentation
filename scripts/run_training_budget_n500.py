"""Run the four fresh n=500, 8,000-update training-budget configurations."""

from pathlib import Path

import torch

from roofseg.audit import build_splits
from roofseg.integrity import load_json
from roofseg.optimization import INITIALIZATIONS
from roofseg.pilot import build_metadata
from roofseg.run_artifacts import prepare_run, verify_completed_run
from roofseg.training import fit_run, select_device
from roofseg.training_budget import (
    SUBSET_REPETITIONS,
    build_training_budget_config,
    training_budget_run_directory,
)


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/data_manifest.json")
splits = load_json(ROOT / "data/metadata/splits.json")
if build_splits(manifest, ROOT) != splits:
    raise SystemExit("Active geographic split does not reproduce exactly.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"Training-budget block requires MPS; selected {device}.")
print(f"Training device: {device}", flush=True)

for repetition in SUBSET_REPETITIONS:
    for initialization in INITIALIZATIONS:
        config = build_training_budget_config(splits, initialization, repetition)
        directory = training_budget_run_directory(
            ROOT, splits, initialization, repetition
        )
        metadata = build_metadata(ROOT, manifest, splits, config, device)
        if prepare_run(directory, config, metadata):
            verify_completed_run(directory)
            print(f"Reusing compatible completed run: {directory.name}", flush=True)
            continue
        fit_run(ROOT, config, metadata, manifest, device, directory)
        verify_completed_run(directory)
