"""Run or safely reuse six fresh WHU n=500, 8,000-update models."""

from pathlib import Path

import torch

from roofseg.integrity import load_json
from roofseg.pilot import INITIALIZATIONS, build_metadata
from roofseg.run_artifacts import prepare_run, verify_completed_run
from roofseg.training import fit_run, select_device
from roofseg.whu_data import build_whu_splits
from roofseg.whu_pilot import LEARNING_CURVE_REPETITIONS
from roofseg.whu_training_budget import (
    build_whu_training_budget_config,
    whu_training_budget_run_directory,
)


ROOT = Path(__file__).resolve().parents[1]
torch.hub.set_dir(str(ROOT / ".cache/torch/hub"))
manifest = load_json(ROOT / "data/metadata/whu/data_manifest.json")
splits = load_json(ROOT / "data/metadata/whu/splits.json")
if build_whu_splits(manifest, subset_seed=17) != splits:
    raise SystemExit("Saved WHU roles and subsets do not reproduce exactly.")
device = select_device()
if device.type != "mps":
    raise SystemExit(f"WHU training-budget block requires MPS; selected {device}.")
print(f"Training device: {device}", flush=True)

test_ids = set(splits["roles"]["test"])
for repetition in LEARNING_CURVE_REPETITIONS:
    for initialization in INITIALIZATIONS:
        config = build_whu_training_budget_config(
            splits, initialization, repetition
        )
        directory = whu_training_budget_run_directory(
            ROOT, splits, initialization, repetition
        )
        if (set(config["training_ids"]) | set(config["validation_ids"])) & test_ids:
            raise SystemExit("WHU training or validation intersects provider test data.")
        metadata = build_metadata(ROOT, manifest, splits, config, device)
        if prepare_run(directory, config, metadata):
            verify_completed_run(directory)
            print(f"Reusing compatible completed run: {directory.name}", flush=True)
            continue
        fit_run(ROOT, config, metadata, manifest, device, directory)
        verify_completed_run(directory)
