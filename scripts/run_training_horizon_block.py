"""Run the four fresh 4,000-step Stage 3 training-horizon candidates."""

from pathlib import Path
import json

import torch

from roofseg.audit import build_splits
from roofseg.integrity import load_json
from roofseg.optimization import (
    HORIZON_LEARNING_RATES,
    INITIALIZATIONS,
    build_learning_rate_config,
    build_training_horizon_config,
    learning_rate_run_directory,
    training_horizon_run_directory,
)
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
    raise SystemExit(f"Stage 3 training requires available MPS on this host; selected {device}.")
print(f"Training device: {device}", flush=True)

test_ids = set(splits["roles"]["test"])
for initialization in INITIALIZATIONS:
    for learning_rate in HORIZON_LEARNING_RATES:
        historical_config = build_learning_rate_config(splits, initialization, learning_rate)
        historical_directory = learning_rate_run_directory(
            ROOT, splits, initialization, learning_rate
        )
        saved_config = json.loads((historical_directory / "config.json").read_text())
        if saved_config != historical_config:
            raise SystemExit(f"Historical configuration differs: {historical_directory.name}")
        verify_completed_run(historical_directory)

        config = build_training_horizon_config(splits, initialization, learning_rate)
        if (set(config["training_ids"]) | set(config["validation_ids"])) & test_ids:
            raise SystemExit("Training-horizon inputs intersect the locked test role.")
        metadata = build_metadata(ROOT, manifest, splits, config, device)
        directory = training_horizon_run_directory(
            ROOT, splits, initialization, learning_rate
        )
        if prepare_run(directory, config, metadata):
            verify_completed_run(directory)
            print(f"Reusing compatible completed run: {directory.name}", flush=True)
            continue
        fit_run(ROOT, config, metadata, manifest, device, directory)
        verify_completed_run(directory)
